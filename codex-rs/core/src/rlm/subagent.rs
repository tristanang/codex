#![cfg_attr(not(test), allow(dead_code))]

use crate::rlm::monty_repl::MontyHostCallbacks;
use crate::rlm::monty_repl::MontyReplError;
use crate::rlm::monty_repl::MontyReplRuntime;
use crate::rlm::monty_repl::StubHostCallbacks;
use async_trait::async_trait;
use thiserror::Error;

const MIN_MAX_DEPTH: i32 = 2;
const DEFAULT_MAX_ITERATIONS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RlmIterationRecord {
    pub model_response: String,
    pub repl_outputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RlmSubagentResult {
    pub final_output: String,
    pub iteration_count: usize,
    pub history: Vec<RlmIterationRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RlmSubagentConfig {
    pub max_depth: i32,
    pub max_iterations: usize,
}

impl Default for RlmSubagentConfig {
    fn default() -> Self {
        Self {
            max_depth: MIN_MAX_DEPTH,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl RlmSubagentConfig {
    pub(crate) fn new(max_depth: i32, max_iterations: usize) -> Result<Self, RlmSubagentError> {
        if max_depth < MIN_MAX_DEPTH {
            return Err(RlmSubagentError::InvalidMaxDepthFloor { max_depth });
        }
        if max_iterations == 0 {
            return Err(RlmSubagentError::InvalidMaxIterations);
        }

        Ok(Self {
            max_depth,
            max_iterations,
        })
    }
}

#[async_trait]
pub(crate) trait RlmWorkerModel {
    async fn next_step(
        &mut self,
        prompt: &str,
        history: &[RlmIterationRecord],
    ) -> Result<String, RlmSubagentError>;
}

#[derive(Debug, Error)]
pub(crate) enum RlmSubagentError {
    #[error("max_depth must be at least 2 for subagent RLM recursion, got {max_depth}")]
    InvalidMaxDepthFloor { max_depth: i32 },
    #[error("max_iterations must be greater than zero")]
    InvalidMaxIterations,
    #[error("recursion depth {recursion_depth} exceeds max depth {max_depth}")]
    DepthExceeded {
        recursion_depth: i32,
        max_depth: i32,
    },
    #[error("subagent loop exhausted max iterations ({max_iterations}) before FINAL/FINAL_VAR")]
    IterationLimitExceeded { max_iterations: usize },
    #[error("worker model error: {0}")]
    Model(String),
    #[error(transparent)]
    Repl(#[from] MontyReplError),
}

pub(crate) struct RlmSubagent {
    repl: MontyReplRuntime,
    config: RlmSubagentConfig,
    recursion_depth: i32,
    iteration_count: usize,
    history: Vec<RlmIterationRecord>,
}

impl RlmSubagent {
    pub(crate) fn with_new_repl(
        config: RlmSubagentConfig,
        recursion_depth: i32,
    ) -> Result<Self, RlmSubagentError> {
        let repl = MontyReplRuntime::new()?;
        Self::new(repl, config, recursion_depth)
    }

    pub(crate) fn new(
        repl: MontyReplRuntime,
        config: RlmSubagentConfig,
        recursion_depth: i32,
    ) -> Result<Self, RlmSubagentError> {
        if config.max_depth < MIN_MAX_DEPTH {
            return Err(RlmSubagentError::InvalidMaxDepthFloor {
                max_depth: config.max_depth,
            });
        }
        if recursion_depth > config.max_depth {
            return Err(RlmSubagentError::DepthExceeded {
                recursion_depth,
                max_depth: config.max_depth,
            });
        }

        Ok(Self {
            repl,
            config,
            recursion_depth,
            iteration_count: 0,
            history: Vec::new(),
        })
    }

    pub(crate) async fn run_loop(
        &mut self,
        prompt: &str,
        model: &mut impl RlmWorkerModel,
        callbacks: &impl MontyHostCallbacks,
    ) -> Result<RlmSubagentResult, RlmSubagentError> {
        if self.recursion_depth > self.config.max_depth {
            return Err(RlmSubagentError::DepthExceeded {
                recursion_depth: self.recursion_depth,
                max_depth: self.config.max_depth,
            });
        }

        while self.iteration_count < self.config.max_iterations {
            self.iteration_count += 1;
            let model_response = model.next_step(prompt, &self.history).await?;

            let mut repl_outputs = Vec::new();
            for repl_code in extract_repl_blocks(&model_response) {
                let execution = self
                    .repl
                    .execute_code_with_callbacks(&repl_code, callbacks)?;
                let rendered_output = if execution.stdout.is_empty() {
                    execution.value.to_string()
                } else {
                    format!("{}{}", execution.stdout, execution.value)
                };
                repl_outputs.push(rendered_output);
            }

            let final_output = self.resolve_final_output(&model_response, callbacks)?;
            self.history.push(RlmIterationRecord {
                model_response,
                repl_outputs,
            });

            if let Some(final_output) = final_output {
                return Ok(RlmSubagentResult {
                    final_output,
                    iteration_count: self.iteration_count,
                    history: self.history.clone(),
                });
            }
        }

        Err(RlmSubagentError::IterationLimitExceeded {
            max_iterations: self.config.max_iterations,
        })
    }

    pub(crate) async fn run_loop_with_stub_callbacks(
        &mut self,
        prompt: &str,
        model: &mut impl RlmWorkerModel,
    ) -> Result<RlmSubagentResult, RlmSubagentError> {
        self.run_loop(prompt, model, &StubHostCallbacks).await
    }

    fn resolve_final_output(
        &mut self,
        model_response: &str,
        callbacks: &impl MontyHostCallbacks,
    ) -> Result<Option<String>, RlmSubagentError> {
        if let Some(final_literal) = extract_directive_arg(model_response, "FINAL") {
            return Ok(Some(final_literal));
        }

        if let Some(final_var_name) = extract_directive_arg(model_response, "FINAL_VAR") {
            let execution = self
                .repl
                .execute_code_with_callbacks(&final_var_name, callbacks)?;
            return Ok(Some(execution.value.to_string()));
        }

        Ok(None)
    }
}

fn extract_repl_blocks(model_response: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cursor = model_response;

    while let Some(fence_start) = cursor.find("```repl") {
        let after_start = &cursor[fence_start + "```repl".len()..];
        let Some(code_start) = after_start.find('\n') else {
            break;
        };
        let code_block = &after_start[code_start + 1..];
        let Some(fence_end) = code_block.find("```") else {
            break;
        };

        let code = code_block[..fence_end].trim();
        if !code.is_empty() {
            blocks.push(code.to_string());
        }

        cursor = &code_block[fence_end + 3..];
    }

    blocks
}

fn extract_directive_arg(model_response: &str, directive: &str) -> Option<String> {
    let token = format!("{directive}(");
    let start = model_response.find(&token)?;
    let tail = &model_response[start + token.len()..];
    let end = tail.find(')')?;
    Some(tail[..end].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::collections::VecDeque;

    struct MockModel {
        responses: VecDeque<String>,
    }

    impl MockModel {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.into_iter().map(str::to_string).collect(),
            }
        }
    }

    #[async_trait]
    impl RlmWorkerModel for MockModel {
        async fn next_step(
            &mut self,
            _prompt: &str,
            _history: &[RlmIterationRecord],
        ) -> Result<String, RlmSubagentError> {
            self.responses.pop_front().ok_or_else(|| {
                RlmSubagentError::Model("mock model ran out of responses".to_string())
            })
        }
    }

    #[test]
    fn config_enforces_minimum_depth_floor() {
        let error = RlmSubagentConfig::new(1, 4).expect_err("depth < 2 should fail");
        assert_eq!(
            error.to_string(),
            "max_depth must be at least 2 for subagent RLM recursion, got 1"
        );
    }

    #[tokio::test]
    async fn run_loop_returns_final_literal() -> Result<(), RlmSubagentError> {
        let config = RlmSubagentConfig::new(2, 4)?;
        let mut agent = RlmSubagent::with_new_repl(config, 1)?;
        let mut model = MockModel::new(vec!["FINAL(done)"]);

        let result = agent
            .run_loop_with_stub_callbacks("ignored prompt", &mut model)
            .await?;

        assert_eq!(result.final_output, "done");
        assert_eq!(result.iteration_count, 1);
        Ok(())
    }

    #[tokio::test]
    async fn run_loop_executes_repl_blocks_and_resolves_final_var() -> Result<(), RlmSubagentError>
    {
        let config = RlmSubagentConfig::new(2, 4)?;
        let mut agent = RlmSubagent::with_new_repl(config, 1)?;
        let mut model = MockModel::new(vec!["```repl\nvalue = 40 + 2\n```\nFINAL_VAR(value)"]);

        let result = agent
            .run_loop_with_stub_callbacks("ignored prompt", &mut model)
            .await?;

        assert_eq!(result.final_output, "42");
        assert_eq!(result.iteration_count, 1);
        Ok(())
    }

    #[tokio::test]
    async fn run_loop_tracks_multi_iteration_history() -> Result<(), RlmSubagentError> {
        let config = RlmSubagentConfig::new(2, 4)?;
        let mut agent = RlmSubagent::with_new_repl(config, 1)?;
        let mut model = MockModel::new(vec![
            "```repl\ncounter = 1\n```",
            "```repl\ncounter = counter + 1\n```\nFINAL_VAR(counter)",
        ]);

        let result = agent
            .run_loop_with_stub_callbacks("ignored prompt", &mut model)
            .await?;

        assert_eq!(result.iteration_count, 2);
        assert_eq!(result.final_output, "2");
        assert_eq!(result.history.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn run_loop_errors_when_iteration_budget_is_exhausted() -> Result<(), RlmSubagentError> {
        let config = RlmSubagentConfig::new(2, 1)?;
        let mut agent = RlmSubagent::with_new_repl(config, 1)?;
        let mut model = MockModel::new(vec!["no final directive"]);

        let error = agent
            .run_loop_with_stub_callbacks("ignored prompt", &mut model)
            .await
            .expect_err("missing FINAL should exhaust budget");

        assert_eq!(
            error.to_string(),
            "subagent loop exhausted max iterations (1) before FINAL/FINAL_VAR"
        );
        Ok(())
    }
}
