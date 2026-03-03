use crate::Prompt;
use crate::agent::next_thread_spawn_depth;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::codex::TurnContext;
use crate::compact::content_items_to_text;
use crate::function_tool::FunctionCallError;
use crate::rlm::subagent::RlmIterationRecord;
use crate::rlm::subagent::RlmSubagent;
use crate::rlm::subagent::RlmSubagentConfig;
use crate::rlm::subagent::RlmSubagentError;
use crate::rlm::subagent::RlmWorkerModel;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;
use async_trait::async_trait;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use tracing::info;

const DEFAULT_MAX_ITERATIONS: usize = 16;
const MIN_MAX_DEPTH: i32 = 2;
const HISTORY_TAIL: usize = 4;
const RLM_WORKER_INSTRUCTIONS: &str = r#"You are an RLM (Recursive Language Model) worker operating a persistent Python REPL environment. You will be queried iteratively until you provide a final answer.

## REPL Environment
- Execute Python code by wrapping it in ```repl``` fenced blocks.
- The REPL is persistent: variables, functions, and state carry across iterations.
- Use `print()` statements to inspect intermediate results. You will only see truncated output, so use print strategically.
- The REPL runs Monty (a Python subset in Rust). Avoid unsupported features:
  - Do NOT use `int.bit_length()`, `isinstance()`, or dynamic attribute access.
  - Do NOT use nested subscript assignment like `mat[i][j] = 0`. Instead use flat indexing: `mat[i * n + j] = 0`.
  - Stick to basic types (int, float, str, bool, list, dict, tuple) and standard control flow.

## How To Work
Think step by step carefully. Plan your approach, then execute immediately in ```repl``` blocks — do not just describe what you will do.
- Offload heavy computation and large intermediate state into the REPL rather than reasoning about it in prose.
- Reuse existing REPL state across iterations; do not recompute from scratch unless a prior error requires it.
- Prefer efficient algorithms. Avoid unnecessary all-pairs or global recomputation.
- Keep each iteration narrowly scoped to the next required step.
- Run minimal validation needed to guarantee correctness.

## Error Recovery
- If transcript includes REPL_ERROR, analyze the error message, patch the failing logic, and continue. Do NOT repeat the same code that caused the error.
- If transcript includes FINAL_RESOLUTION_ERROR, the variable you referenced may not exist. Create it in a ```repl``` block first, then use FINAL_VAR in a separate step.

## Completing Your Task
When you have finished, you MUST provide a final answer using exactly one of these directives (outside of code fences):
1. `FINAL(your answer here)` — provide the answer directly as text
2. `FINAL_VAR(variable_name)` — return an existing REPL variable's value

WARNING — COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE iteration. Example:
- WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
- CORRECT: First run ```repl
my_answer = compute_result()
print(my_answer)
``` then in the NEXT response call FINAL_VAR(my_answer)

Do NOT use FINAL/FINAL_VAR until you have completed the task and verified correctness.
If not complete, return more ```repl``` blocks to make progress."#;

pub struct RlmSubagentHandler;

#[derive(Debug, Deserialize)]
struct RunRlmSubagentArgs {
    prompt: String,
    #[serde(default)]
    max_depth: Option<i32>,
    #[serde(default)]
    max_iterations: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RunRlmSubagentResult {
    final_output: String,
    iteration_count: usize,
    recursion_depth: i32,
    max_depth: i32,
    model: String,
}

#[async_trait]
impl ToolHandler for RlmSubagentHandler {
    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
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
                    "run_rlm_subagent handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: RunRlmSubagentArgs = parse_arguments(&arguments)?;
        let prompt = args.prompt.trim();
        if prompt.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "prompt must not be empty".to_string(),
            ));
        }

        let recursion_depth = next_thread_spawn_depth(&turn.session_source);
        let default_max_depth = turn.config.agent_max_depth.max(MIN_MAX_DEPTH);
        let max_depth = args.max_depth.unwrap_or(default_max_depth);
        let max_iterations = args.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);

        // Early validation before config creation
        if max_depth < MIN_MAX_DEPTH {
            return Err(FunctionCallError::RespondToModel(format!(
                "max_depth must be at least {MIN_MAX_DEPTH}, got {max_depth}"
            )));
        }
        if recursion_depth > max_depth {
            return Err(FunctionCallError::RespondToModel(format!(
                "recursion depth {recursion_depth} exceeds max_depth {max_depth}"
            )));
        }

        let config = RlmSubagentConfig::new(max_depth, max_iterations)
            .map_err(map_rlm_error_to_tool_error)?;

        let model_name = turn
            .config
            .subagent_model
            .clone()
            .unwrap_or_else(|| turn.model_info.slug.clone());
        let model_info = resolve_model_info(&session, &turn, &model_name).await;

        let mut worker_model = ResponsesApiRlmWorkerModel::new(
            session.services.model_client.new_session(),
            model_info.clone(),
            turn,
        );
        let mut subagent = RlmSubagent::with_new_repl(config, recursion_depth)
            .map_err(map_rlm_error_to_tool_error)?;
        let result = subagent
            .run_loop_with_stub_callbacks(prompt, &mut worker_model)
            .await
            .map_err(map_rlm_error_to_tool_error)?;

        let response = RunRlmSubagentResult {
            final_output: result.final_output,
            iteration_count: result.iteration_count,
            recursion_depth,
            max_depth,
            model: model_name,
        };
        let content = serde_json::to_string(&response).map_err(|err| {
            FunctionCallError::Fatal(format!(
                "failed to serialize run_rlm_subagent result: {err}"
            ))
        })?;

        Ok(ToolOutput::Function {
            body: FunctionCallOutputBody::Text(content),
            success: Some(true),
        })
    }
}

async fn resolve_model_info(
    session: &crate::codex::Session,
    turn: &TurnContext,
    model_name: &str,
) -> ModelInfo {
    if model_name == turn.model_info.slug {
        return turn.model_info.clone();
    }
    session
        .services
        .models_manager
        .get_model_info(model_name, &turn.config)
        .await
}

fn map_rlm_error_to_tool_error(err: RlmSubagentError) -> FunctionCallError {
    FunctionCallError::RespondToModel(format!("run_rlm_subagent failed: {err}"))
}

struct ResponsesApiRlmWorkerModel {
    model_session: ModelClientSession,
    model_info: ModelInfo,
    turn: Arc<TurnContext>,
}

impl ResponsesApiRlmWorkerModel {
    fn new(
        model_session: ModelClientSession,
        model_info: ModelInfo,
        turn: Arc<TurnContext>,
    ) -> Self {
        Self {
            model_session,
            model_info,
            turn,
        }
    }
}

#[async_trait]
impl RlmWorkerModel for ResponsesApiRlmWorkerModel {
    async fn next_step(
        &mut self,
        prompt: &str,
        history: &[RlmIterationRecord],
    ) -> Result<String, RlmSubagentError> {
        let iteration = history.len() + 1;
        let message = build_iteration_message(prompt, history);
        info!(
            iteration,
            model = %self.model_info.slug,
            prompt = %message,
            "rlm iteration prompt"
        );
        let request = Prompt {
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText { text: message }],
                end_turn: None,
                phase: None,
            }],
            tools: Vec::new(),
            parallel_tool_calls: false,
            base_instructions: BaseInstructions {
                text: RLM_WORKER_INSTRUCTIONS.to_string(),
            },
            personality: self.turn.personality,
            output_schema: None,
        };

        let mut stream = self
            .model_session
            .stream(
                &request,
                &self.model_info,
                &self.turn.otel_manager,
                self.turn.reasoning_effort,
                self.turn.reasoning_summary,
                None,
            )
            .await
            .map_err(|err| RlmSubagentError::Model(format!("worker model stream failed: {err}")))?;

        let mut response_text = String::new();
        while let Some(event) = stream.next().await.transpose().map_err(|err| {
            RlmSubagentError::Model(format!("worker model stream event failed: {err}"))
        })? {
            match event {
                ResponseEvent::OutputTextDelta(delta) => {
                    response_text.push_str(&delta);
                }
                ResponseEvent::OutputItemDone(item) => {
                    if response_text.is_empty()
                        && let Some(text) = message_text_from_response_item(&item)
                    {
                        response_text.push_str(&text);
                    }
                }
                ResponseEvent::Completed { .. } => break,
                _ => {}
            }
        }

        let trimmed = response_text.trim();
        if trimmed.is_empty() {
            return Err(RlmSubagentError::Model(
                "worker model returned empty response".to_string(),
            ));
        }
        info!(
            iteration,
            model = %self.model_info.slug,
            response = %trimmed,
            "rlm iteration response"
        );
        Ok(trimmed.to_string())
    }
}

fn message_text_from_response_item(item: &ResponseItem) -> Option<String> {
    match item {
        ResponseItem::Message { content, .. } => content_items_to_text(content),
        _ => None,
    }
}

fn build_iteration_message(prompt: &str, history: &[RlmIterationRecord]) -> String {
    let mut message = format!("Task:\n{prompt}\n\n");
    if history.is_empty() {
        message.push_str("You have not interacted with the REPL environment yet. Your next action should be to plan your approach and start executing in ```repl``` blocks. Do not provide a final answer yet.\n");
    } else {
        message.push_str("The history below shows your previous interactions with the REPL environment.\n\nRecent iteration transcript:\n");
        let start = history.len().saturating_sub(HISTORY_TAIL);
        for (index, record) in history.iter().enumerate().skip(start) {
            let iteration = index + 1;
            message.push_str(&format!("\nIteration {iteration} model response:\n"));
            message.push_str(&record.model_response);
            if !record.repl_outputs.is_empty() {
                message.push_str("\nREPL outputs:\n");
                for output in &record.repl_outputs {
                    message.push_str("- ");
                    message.push_str(output);
                    message.push('\n');
                }
            }
        }
    }
    message.push_str(
        "\nContinue using the REPL environment by writing ```repl``` blocks. Use FINAL(...) or FINAL_VAR(...) only when your task is complete and verified.",
    );
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_iteration_message_includes_history_tail() {
        let history = vec![
            RlmIterationRecord {
                model_response: "one".to_string(),
                repl_outputs: vec!["a".to_string()],
            },
            RlmIterationRecord {
                model_response: "two".to_string(),
                repl_outputs: vec!["b".to_string()],
            },
        ];
        let msg = build_iteration_message("solve", &history);
        assert!(msg.contains("Task:\nsolve"));
        assert!(msg.contains("Iteration 1 model response:\none"));
        assert!(msg.contains("Iteration 2 model response:\ntwo"));
        assert!(msg.contains("REPL outputs"));
    }

    #[test]
    fn build_iteration_message_iteration_zero_safeguard() {
        let msg = build_iteration_message("solve", &[]);
        assert!(msg.contains("You have not interacted with the REPL environment yet"));
        assert!(msg.contains("Do not provide a final answer yet"));
    }
}
