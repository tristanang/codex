#![cfg_attr(not(test), allow(dead_code))]

use monty::ExcType;
use monty::ExternalResult;
use monty::MontyException;
use monty::MontyObject;
use monty::MontyRepl;
use monty::NoLimitTracker;
use monty::OsFunction;
use monty::PrintWriter;
use monty::ReplProgress;
use monty::ReplSnapshot;
use monty::ReplStartError;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;
use thiserror::Error;
use tracing::debug;

const SCRIPT_NAME: &str = "<codex-rlm>";
const HOST_CALLBACK_FUNCTIONS: [&str; 5] = [
    "llm_query",
    "read_file",
    "write_file",
    "list_files",
    "shell",
];

#[derive(Debug, Default)]
pub(crate) struct SessionMutationGate {
    lock: RwLock<()>,
}

impl SessionMutationGate {
    pub(crate) fn with_read<T>(
        &self,
        operation: impl FnOnce() -> Result<T, MontyReplError>,
    ) -> Result<T, MontyReplError> {
        let wait_started = Instant::now();
        let guard = self
            .lock
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let waited = wait_started.elapsed();

        let hold_started = Instant::now();
        let result = operation();
        let held = hold_started.elapsed();
        debug!(
            wait_ms = waited.as_millis(),
            hold_ms = held.as_millis(),
            "rlm read callback gate"
        );
        drop(guard);

        result
    }

    pub(crate) fn with_write<T>(
        &self,
        operation: impl FnOnce() -> Result<T, MontyReplError>,
    ) -> Result<T, MontyReplError> {
        let wait_started = Instant::now();
        let guard = self
            .lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let waited = wait_started.elapsed();

        let hold_started = Instant::now();
        let result = operation();
        let held = hold_started.elapsed();
        debug!(
            wait_ms = waited.as_millis(),
            hold_ms = held.as_millis(),
            "rlm write callback gate"
        );
        drop(guard);

        result
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MontyExecutionResult {
    pub value: MontyObject,
    pub stdout: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MontyReplSnapshot {
    bytes: Vec<u8>,
}

impl MontyReplSnapshot {
    #[must_use]
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) trait MontyHostCallbacks {
    fn llm_query(&self, prompt: &str) -> Result<MontyObject, MontyReplError>;
    fn read_file(&self, path: &str) -> Result<MontyObject, MontyReplError>;
    fn write_file(&self, path: &str, content: &str) -> Result<MontyObject, MontyReplError>;
    fn list_files(&self, pattern: &str) -> Result<MontyObject, MontyReplError>;
    fn shell(&self, command: &str) -> Result<MontyObject, MontyReplError>;

    fn os_call(
        &self,
        function: OsFunction,
        _args: Vec<MontyObject>,
        _kwargs: Vec<(MontyObject, MontyObject)>,
    ) -> Result<ExternalResult, MontyReplError> {
        Err(MontyReplError::UnsupportedHostCallback(format!(
            "os callback {function:?} is not wired yet"
        )))
    }

    fn resolve_futures(
        &self,
        pending_call_ids: &[u32],
    ) -> Result<Vec<(u32, ExternalResult)>, MontyReplError> {
        if pending_call_ids.is_empty() {
            return Ok(Vec::new());
        }

        Err(MontyReplError::UnsupportedHostCallback(format!(
            "resolve_futures callback is not wired yet for call ids {pending_call_ids:?}"
        )))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct StubHostCallbacks;

impl MontyHostCallbacks for StubHostCallbacks {
    fn llm_query(&self, _prompt: &str) -> Result<MontyObject, MontyReplError> {
        Err(MontyReplError::UnsupportedHostCallback(
            "llm_query callback is not wired yet".to_string(),
        ))
    }

    fn read_file(&self, _path: &str) -> Result<MontyObject, MontyReplError> {
        Err(MontyReplError::UnsupportedHostCallback(
            "read_file callback is not wired yet".to_string(),
        ))
    }

    fn write_file(&self, _path: &str, _content: &str) -> Result<MontyObject, MontyReplError> {
        Err(MontyReplError::UnsupportedHostCallback(
            "write_file callback is not wired yet".to_string(),
        ))
    }

    fn list_files(&self, _pattern: &str) -> Result<MontyObject, MontyReplError> {
        Err(MontyReplError::UnsupportedHostCallback(
            "list_files callback is not wired yet".to_string(),
        ))
    }

    fn shell(&self, _command: &str) -> Result<MontyObject, MontyReplError> {
        Err(MontyReplError::UnsupportedHostCallback(
            "shell callback is not wired yet".to_string(),
        ))
    }
}

#[derive(Debug)]
pub(crate) struct MontyReplRuntime {
    repl: MontyRepl<NoLimitTracker>,
    mutation_gate: Arc<SessionMutationGate>,
}

#[derive(Debug, Error)]
pub(crate) enum MontyReplError {
    #[error("failed to initialize monty repl: {0}")]
    Initialization(String),
    #[error("monty repl execution failed: {0}")]
    Execution(String),
    #[error("failed to snapshot monty repl: {0}")]
    Snapshot(String),
    #[error("failed to restore monty repl: {0}")]
    Restore(String),
    #[error("unsupported host callback: {0}")]
    UnsupportedHostCallback(String),
    #[error("invalid host callback input: {0}")]
    InvalidHostCall(String),
}

impl MontyReplRuntime {
    pub(crate) fn new() -> Result<Self, MontyReplError> {
        Self::new_with_gate(Arc::new(SessionMutationGate::default()))
    }

    pub(crate) fn new_with_gate(
        mutation_gate: Arc<SessionMutationGate>,
    ) -> Result<Self, MontyReplError> {
        let repl = Self::initialize_repl()?;
        Ok(Self {
            repl,
            mutation_gate,
        })
    }

    pub(crate) fn restore(snapshot: &MontyReplSnapshot) -> Result<Self, MontyReplError> {
        Self::restore_with_gate(snapshot, Arc::new(SessionMutationGate::default()))
    }

    pub(crate) fn restore_with_gate(
        snapshot: &MontyReplSnapshot,
        mutation_gate: Arc<SessionMutationGate>,
    ) -> Result<Self, MontyReplError> {
        let repl = MontyRepl::load(snapshot.bytes())
            .map_err(|err| MontyReplError::Restore(err.to_string()))?;
        Ok(Self {
            repl,
            mutation_gate,
        })
    }

    pub(crate) fn snapshot(&self) -> Result<MontyReplSnapshot, MontyReplError> {
        let bytes = self
            .repl
            .dump()
            .map_err(|err| MontyReplError::Snapshot(err.to_string()))?;
        Ok(MontyReplSnapshot { bytes })
    }

    pub(crate) fn execute_code(
        &mut self,
        code: &str,
    ) -> Result<MontyExecutionResult, MontyReplError> {
        self.execute_code_with_callbacks(code, &StubHostCallbacks)
    }

    pub(crate) fn execute_code_with_callbacks(
        &mut self,
        code: &str,
        callbacks: &impl MontyHostCallbacks,
    ) -> Result<MontyExecutionResult, MontyReplError> {
        let mut print = PrintWriter::Collect(String::new());
        let current_repl = std::mem::replace(&mut self.repl, Self::initialize_repl()?);
        let mut progress = current_repl
            .start(code, &mut print)
            .map_err(|err| Self::handle_start_error(&mut self.repl, err))?;

        loop {
            match progress {
                ReplProgress::Complete { repl, value } => {
                    self.repl = repl;
                    let stdout = print.collected_output().unwrap_or_default().to_string();
                    return Ok(MontyExecutionResult { value, stdout });
                }
                ReplProgress::FunctionCall {
                    function_name,
                    args,
                    kwargs,
                    method_call,
                    state,
                    ..
                } => {
                    let result = if Self::is_mutating_callback(&function_name) {
                        self.mutation_gate.with_write(|| {
                            Self::dispatch_function_call(
                                callbacks,
                                &function_name,
                                args,
                                kwargs,
                                method_call,
                            )
                        })
                    } else {
                        self.mutation_gate.with_read(|| {
                            Self::dispatch_function_call(
                                callbacks,
                                &function_name,
                                args,
                                kwargs,
                                method_call,
                            )
                        })
                    }
                    .map(ExternalResult::Return);
                    progress = Self::resume_snapshot(state, result, &mut self.repl, &mut print)?;
                }
                ReplProgress::OsCall {
                    function,
                    args,
                    kwargs,
                    state,
                    ..
                } => {
                    let result = self
                        .mutation_gate
                        .with_write(|| callbacks.os_call(function, args, kwargs));
                    progress = Self::resume_snapshot(state, result, &mut self.repl, &mut print)?;
                }
                ReplProgress::ResolveFutures(state) => {
                    let pending_call_ids = state.pending_call_ids().to_vec();
                    let results = self
                        .mutation_gate
                        .with_read(|| callbacks.resolve_futures(&pending_call_ids))
                        .unwrap_or_else(|err| {
                            pending_call_ids
                                .into_iter()
                                .map(|call_id| {
                                    (
                                        call_id,
                                        ExternalResult::Error(MontyException::new(
                                            ExcType::RuntimeError,
                                            Some(err.to_string()),
                                        )),
                                    )
                                })
                                .collect()
                        });

                    progress = state
                        .resume(results, &mut print)
                        .map_err(|err| Self::handle_start_error(&mut self.repl, err))?;
                }
            }
        }
    }

    fn initialize_repl() -> Result<MontyRepl<NoLimitTracker>, MontyReplError> {
        let mut print = PrintWriter::Disabled;
        let external_functions = HOST_CALLBACK_FUNCTIONS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let (repl, _) = MontyRepl::new(
            String::new(),
            SCRIPT_NAME,
            Vec::new(),
            external_functions,
            Vec::new(),
            NoLimitTracker,
            &mut print,
        )
        .map_err(|err| MontyReplError::Initialization(err.to_string()))?;
        Ok(repl)
    }

    fn dispatch_function_call(
        callbacks: &impl MontyHostCallbacks,
        function_name: &str,
        args: Vec<MontyObject>,
        kwargs: Vec<(MontyObject, MontyObject)>,
        method_call: bool,
    ) -> Result<MontyObject, MontyReplError> {
        if method_call {
            return Err(MontyReplError::InvalidHostCall(format!(
                "method callback {function_name} is not supported"
            )));
        }
        if !kwargs.is_empty() {
            return Err(MontyReplError::InvalidHostCall(format!(
                "callback {function_name} does not support keyword args"
            )));
        }

        match function_name {
            "llm_query" => callbacks.llm_query(Self::required_string_arg(function_name, &args, 0)?),
            "read_file" => callbacks.read_file(Self::required_string_arg(function_name, &args, 0)?),
            "write_file" => callbacks.write_file(
                Self::required_string_arg(function_name, &args, 0)?,
                Self::required_string_arg(function_name, &args, 1)?,
            ),
            "list_files" => {
                callbacks.list_files(Self::required_string_arg(function_name, &args, 0)?)
            }
            "shell" => callbacks.shell(Self::required_string_arg(function_name, &args, 0)?),
            _ => Err(MontyReplError::UnsupportedHostCallback(format!(
                "unknown callback {function_name}"
            ))),
        }
    }

    fn is_mutating_callback(function_name: &str) -> bool {
        matches!(function_name, "write_file" | "shell")
    }

    fn required_string_arg<'a>(
        function_name: &str,
        args: &'a [MontyObject],
        index: usize,
    ) -> Result<&'a str, MontyReplError> {
        let argument = args.get(index).ok_or_else(|| {
            MontyReplError::InvalidHostCall(format!(
                "callback {function_name} expected argument at index {index}"
            ))
        })?;

        match argument {
            MontyObject::String(value) => Ok(value),
            _ => Err(MontyReplError::InvalidHostCall(format!(
                "callback {function_name} expected string argument at index {index}"
            ))),
        }
    }

    fn resume_snapshot(
        state: ReplSnapshot<NoLimitTracker>,
        callback_result: Result<ExternalResult, MontyReplError>,
        repl_slot: &mut MontyRepl<NoLimitTracker>,
        print: &mut PrintWriter<'_>,
    ) -> Result<ReplProgress<NoLimitTracker>, MontyReplError> {
        let result = match callback_result {
            Ok(value) => value,
            Err(err) => ExternalResult::Error(MontyException::new(
                ExcType::RuntimeError,
                Some(err.to_string()),
            )),
        };

        state
            .run(result, print)
            .map_err(|err| Self::handle_start_error(repl_slot, err))
    }

    fn handle_start_error(
        repl_slot: &mut MontyRepl<NoLimitTracker>,
        err: Box<ReplStartError<NoLimitTracker>>,
    ) -> MontyReplError {
        *repl_slot = err.repl;
        MontyReplError::Execution(err.error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    struct TestHostCallbacks;

    impl MontyHostCallbacks for TestHostCallbacks {
        fn llm_query(&self, prompt: &str) -> Result<MontyObject, MontyReplError> {
            Ok(MontyObject::String(format!("llm:{prompt}")))
        }

        fn read_file(&self, path: &str) -> Result<MontyObject, MontyReplError> {
            Ok(MontyObject::String(format!("{path}:contents")))
        }

        fn write_file(&self, path: &str, content: &str) -> Result<MontyObject, MontyReplError> {
            Ok(MontyObject::String(format!("wrote:{path}:{content}")))
        }

        fn list_files(&self, pattern: &str) -> Result<MontyObject, MontyReplError> {
            Ok(MontyObject::List(vec![MontyObject::String(
                pattern.to_string(),
            )]))
        }

        fn shell(&self, command: &str) -> Result<MontyObject, MontyReplError> {
            Ok(MontyObject::String(format!("shell:{command}")))
        }
    }

    #[test]
    fn execute_code_preserves_state_between_calls() -> Result<(), MontyReplError> {
        let mut repl = MontyReplRuntime::new()?;
        repl.execute_code("counter = 41")?;

        let result = repl.execute_code("counter + 1")?;

        assert_eq!(result.value, MontyObject::Int(42));
        assert_eq!(result.stdout, "");
        Ok(())
    }

    #[test]
    fn snapshot_restore_keeps_repl_state() -> Result<(), MontyReplError> {
        let mut repl = MontyReplRuntime::new()?;
        repl.execute_code("greeting = 'hello'")?;
        let snapshot = repl.snapshot()?;

        let mut restored = MontyReplRuntime::restore(&snapshot)?;
        let result = restored.execute_code("greeting + ' world'")?;

        assert_eq!(result.value, MontyObject::String("hello world".to_string()));
        Ok(())
    }

    #[test]
    fn execute_code_captures_print_output() -> Result<(), MontyReplError> {
        let mut repl = MontyReplRuntime::new()?;

        let result = repl.execute_code("print('hello from monty')")?;

        assert_eq!(result.value, MontyObject::None);
        assert_eq!(result.stdout, "hello from monty\n");
        Ok(())
    }

    #[test]
    fn execute_code_routes_host_callback_calls() -> Result<(), MontyReplError> {
        let mut repl = MontyReplRuntime::new()?;

        let result =
            repl.execute_code_with_callbacks("read_file('notes.txt')", &TestHostCallbacks)?;

        assert_eq!(
            result.value,
            MontyObject::String("notes.txt:contents".to_string())
        );
        Ok(())
    }

    #[test]
    fn callback_failures_do_not_reset_repl_state() -> Result<(), MontyReplError> {
        let mut repl = MontyReplRuntime::new()?;
        repl.execute_code("counter = 3")?;

        let error = repl
            .execute_code("read_file('notes.txt')")
            .expect_err("default callback should error");
        assert!(
            error
                .to_string()
                .contains("unsupported host callback: read_file callback is not wired yet")
        );

        let result = repl.execute_code("counter + 1")?;
        assert_eq!(result.value, MontyObject::Int(4));
        Ok(())
    }
}
