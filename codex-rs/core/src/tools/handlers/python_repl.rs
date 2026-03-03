use async_trait::async_trait;
use codex_protocol::models::FunctionCallOutputBody;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::runtime::Handle;
use tokio::sync::Mutex;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::function_tool::FunctionCallError;
use crate::rlm::llm_query_callback::run_llm_query;
use crate::rlm::llm_query_callback::run_llm_query_batched;
use crate::rlm::monty_repl::MontyExecutionResult;
use crate::rlm::monty_repl::MontyHostCallbacks;
use crate::rlm::monty_repl::MontyReplError;
use crate::rlm::monty_repl::MontyReplRuntime;
use crate::rlm::monty_repl::StubHostCallbacks;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;
use monty::MontyObject;

type SharedMontyRepl = Arc<StdMutex<MontyReplRuntime>>;

static PYTHON_REPLS: Lazy<Mutex<HashMap<String, SharedMontyRepl>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct PythonReplHandler;

#[derive(Clone)]
struct PythonReplHostCallbacks {
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    runtime_handle: Handle,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonReplArgs {
    code: String,
}

impl PythonReplHostCallbacks {
    fn new(session: Arc<Session>, turn: Arc<TurnContext>) -> Self {
        Self {
            session,
            turn,
            runtime_handle: Handle::current(),
        }
    }
}

impl MontyHostCallbacks for PythonReplHostCallbacks {
    fn llm_query(&self, prompt: &str, schema: &MontyObject) -> Result<MontyObject, MontyReplError> {
        self.runtime_handle.block_on(run_llm_query(
            Arc::clone(&self.session),
            Arc::clone(&self.turn),
            prompt.to_string(),
            schema.clone(),
        ))
    }

    fn llm_query_batched(
        &self,
        prompts: &[String],
        schema: &MontyObject,
    ) -> Result<MontyObject, MontyReplError> {
        self.runtime_handle.block_on(run_llm_query_batched(
            Arc::clone(&self.session),
            Arc::clone(&self.turn),
            prompts.to_vec(),
            schema.clone(),
        ))
    }

    fn read_file(&self, path: &str) -> Result<MontyObject, MontyReplError> {
        StubHostCallbacks.read_file(path)
    }

    fn write_file(&self, path: &str, content: &str) -> Result<MontyObject, MontyReplError> {
        StubHostCallbacks.write_file(path, content)
    }

    fn list_files(&self, pattern: &str) -> Result<MontyObject, MontyReplError> {
        StubHostCallbacks.list_files(pattern)
    }

    fn shell(&self, command: &str) -> Result<MontyObject, MontyReplError> {
        StubHostCallbacks.shell(command)
    }
}

fn map_monty_error(err: MontyReplError) -> FunctionCallError {
    FunctionCallError::RespondToModel(format!("python_repl failed: {err}"))
}

fn render_execution_output(execution: MontyExecutionResult) -> String {
    let value = execution.value.to_string();
    if execution.stdout.is_empty() {
        value
    } else if value == "None" {
        execution.stdout
    } else {
        format!("{}{value}", execution.stdout)
    }
}

async fn session_repl(session_id: &str) -> Result<SharedMontyRepl, FunctionCallError> {
    let mut repls = PYTHON_REPLS.lock().await;
    if let Some(repl) = repls.get(session_id) {
        return Ok(Arc::clone(repl));
    }

    let runtime = MontyReplRuntime::new().map_err(map_monty_error)?;
    let repl = Arc::new(StdMutex::new(runtime));
    repls.insert(session_id.to_string(), Arc::clone(&repl));
    Ok(repl)
}

async fn execute_for_session_with_callbacks<C>(
    session_id: &str,
    code: &str,
    callbacks: C,
) -> Result<String, FunctionCallError>
where
    C: MontyHostCallbacks + Send + 'static,
{
    let repl = session_repl(session_id).await?;
    let code = code.to_string();
    let execution = tokio::task::spawn_blocking(move || {
        let mut repl = repl
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        repl.execute_code_with_callbacks(&code, &callbacks)
    })
    .await
    .map_err(|err| FunctionCallError::Fatal(format!("python_repl execution task failed: {err}")))?
    .map_err(map_monty_error)?;

    Ok(render_execution_output(execution))
}

#[cfg(test)]
async fn execute_for_session(session_id: &str, code: &str) -> Result<String, FunctionCallError> {
    execute_for_session_with_callbacks(session_id, code, StubHostCallbacks).await
}

#[async_trait]
impl ToolHandler for PythonReplHandler {
    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<ToolOutput, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "python_repl expects function payload".to_string(),
                ));
            }
        };

        let args: PythonReplArgs = parse_arguments(&arguments)?;
        let code = args.code.trim();
        if code.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "python_repl code must not be empty".to_string(),
            ));
        }

        let session_id = session.conversation_id.to_string();
        let callbacks = PythonReplHostCallbacks::new(Arc::clone(&session), Arc::clone(&turn));
        let output = execute_for_session_with_callbacks(&session_id, code, callbacks).await?;
        Ok(ToolOutput::Function {
            body: FunctionCallOutputBody::Text(output),
            success: Some(true),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use pretty_assertions::assert_eq;
    use tokio::sync::Mutex;

    static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    async fn clear_repls() {
        PYTHON_REPLS.lock().await.clear();
    }

    #[tokio::test]
    async fn execute_for_session_preserves_state() {
        let _guard = TEST_LOCK.lock().await;
        clear_repls().await;

        let first = execute_for_session("thread-1", "counter = 41")
            .await
            .expect("first execution");
        assert_eq!(first, "None");

        let second = execute_for_session("thread-1", "counter + 1")
            .await
            .expect("second execution");
        assert_eq!(second, "42");
    }

    #[tokio::test]
    async fn execute_for_session_isolated_across_threads() {
        let _guard = TEST_LOCK.lock().await;
        clear_repls().await;

        execute_for_session("thread-a", "value = 7")
            .await
            .expect("set value in thread a");
        let err = execute_for_session("thread-b", "value + 1")
            .await
            .expect_err("thread b should not see thread a state");
        assert!(
            err.to_string()
                .contains("python_repl failed: monty repl execution failed")
        );
    }

    #[tokio::test]
    async fn execute_for_session_print_omits_none_suffix() {
        let _guard = TEST_LOCK.lock().await;
        clear_repls().await;

        let output = execute_for_session("thread-1", "print('hello')")
            .await
            .expect("print execution");
        assert_eq!(output, "hello\n");
    }
}
