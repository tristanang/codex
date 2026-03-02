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

const DEFAULT_MAX_ITERATIONS: usize = 16;
const MIN_MAX_DEPTH: i32 = 2;
const HISTORY_TAIL: usize = 4;
const RLM_WORKER_INSTRUCTIONS: &str = r#"You are an RLM worker operating a persistent Python REPL.
Use only ```repl``` fenced blocks for executable code.
Do not call host callbacks: llm_query, read_file, write_file, list_files, shell.
When complete, end with exactly one directive:
- FINAL(<concise answer>)
- FINAL_VAR(<python variable or expression>)
If not complete, return more repl blocks without FINAL."#;

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
        let message = build_iteration_message(prompt, history);
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
        message.push_str("No prior RLM iterations have run yet.\n");
    } else {
        message.push_str("Recent iteration transcript:\n");
        let start = history.len().saturating_sub(HISTORY_TAIL);
        for (index, record) in history.iter().enumerate().skip(start) {
            let iteration = index + 1;
            message.push_str(&format!("\nIteration {iteration} model response:\n"));
            message.push_str(&record.model_response);
            if !record.repl_outputs.is_empty() {
                message.push_str("\nIteration repl outputs:\n");
                for output in &record.repl_outputs {
                    message.push_str("- ");
                    message.push_str(output);
                    message.push('\n');
                }
            }
        }
    }
    message.push_str(
        "\nReturn the next step now. Use ```repl``` for code, then FINAL(...) or FINAL_VAR(...) only when done.",
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
        assert!(msg.contains("Iteration repl outputs"));
    }
}
