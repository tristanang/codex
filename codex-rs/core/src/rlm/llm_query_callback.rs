use crate::agent::control::SpawnAgentOptions;
use crate::agent::exceeds_thread_spawn_depth_limit;
use crate::agent::next_thread_spawn_depth;
use crate::agent::status::is_final;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::function_tool::FunctionCallError;
use crate::rlm::monty_repl::MontyReplError;
use crate::tools::handlers::multi_agents::SpawnAgentReturnMode;
use crate::tools::handlers::multi_agents::apply_spawn_agent_overrides;
use crate::tools::handlers::multi_agents::apply_spawn_agent_return_mode_overrides;
use crate::tools::handlers::multi_agents::build_agent_spawn_config;
use codex_protocol::ThreadId;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::user_input::UserInput;
use futures::future::join_all;
use monty::MontyObject;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
enum SchemaSpec {
    Any,
    Bool,
    String,
    Integer,
    Number,
    Null,
    Array(Box<SchemaSpec>),
    ObjectAny,
    ObjectExact(BTreeMap<String, SchemaSpec>),
}

pub(crate) async fn run_llm_query(
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    prompt: String,
    schema: MontyObject,
) -> Result<MontyObject, MontyReplError> {
    run_llm_query_with_runner(prompt, schema, {
        let session = Arc::clone(&session);
        let turn = Arc::clone(&turn);
        move |task_prompt| {
            let session = Arc::clone(&session);
            let turn = Arc::clone(&turn);
            async move { run_single_child_query(session, turn, task_prompt).await }
        }
    })
    .await
}

pub(crate) async fn run_llm_query_batched(
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    prompts: Vec<String>,
    schema: MontyObject,
) -> Result<MontyObject, MontyReplError> {
    run_llm_query_batched_with_query_fn(prompts, schema, {
        let session = Arc::clone(&session);
        let turn = Arc::clone(&turn);
        move |prompt, schema| {
            let session = Arc::clone(&session);
            let turn = Arc::clone(&turn);
            async move { run_llm_query(session, turn, prompt, schema).await }
        }
    })
    .await
}

async fn run_llm_query_with_runner<F, Fut>(
    prompt: String,
    schema: MontyObject,
    mut run_query: F,
) -> Result<MontyObject, MontyReplError>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<String, MontyReplError>>,
{
    let (schema_json, schema_spec) = parse_schema_argument(schema)?;
    let first_response = run_query(prompt.clone()).await?;
    let validated = match parse_and_validate_response(&first_response, &schema_spec) {
        Ok(value) => value,
        Err(first_error) => {
            let repair_prompt =
                build_repair_prompt(&prompt, &schema_json, &first_response, &first_error);
            let repaired_response = run_query(repair_prompt).await?;
            parse_and_validate_response(&repaired_response, &schema_spec).map_err(
                |second_error| {
                    MontyReplError::Execution(format!(
                        "llm_query validation failed after one repair attempt: {second_error}"
                    ))
                },
            )?
        }
    };

    json_to_monty_object(validated)
}

async fn run_llm_query_batched_with_query_fn<F, Fut>(
    prompts: Vec<String>,
    schema: MontyObject,
    run_query: F,
) -> Result<MontyObject, MontyReplError>
where
    F: Fn(String, MontyObject) -> Fut + Clone,
    Fut: Future<Output = Result<MontyObject, MontyReplError>>,
{
    if prompts.is_empty() {
        return Err(MontyReplError::InvalidHostCall(
            "llm_query_batched requires a non-empty prompts list".to_string(),
        ));
    }

    let mut tasks = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.into_iter().enumerate() {
        let run_query = run_query.clone();
        let schema = schema.clone();
        tasks.push(async move {
            run_query(prompt, schema).await.map_err(|err| {
                MontyReplError::Execution(format!("llm_query_batched item {index} failed: {err}"))
            })
        });
    }

    let mut results = Vec::new();
    for result in join_all(tasks).await {
        results.push(result?);
    }
    Ok(MontyObject::List(results))
}

async fn run_single_child_query(
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    prompt: String,
) -> Result<String, MontyReplError> {
    let child_depth = next_thread_spawn_depth(&turn.session_source);
    let max_depth = turn.config.agent_max_depth;
    if exceeds_thread_spawn_depth_limit(child_depth, max_depth) {
        return Err(MontyReplError::Execution(
            "Agent depth limit reached. Solve the task yourself.".to_string(),
        ));
    }

    let mut config =
        build_agent_spawn_config(&session.get_base_instructions().await, turn.as_ref())
            .map_err(function_call_to_monty_error)?;
    apply_spawn_agent_overrides(&mut config, child_depth);
    apply_spawn_agent_return_mode_overrides(&mut config, SpawnAgentReturnMode::ReturnValue);

    let thread_id = session
        .services
        .agent_control
        .spawn_agent_with_options(
            config,
            vec![UserInput::Text {
                text: prompt,
                text_elements: Vec::new(),
            }],
            Some(thread_spawn_source(session.conversation_id, child_depth)),
            SpawnAgentOptions::default(),
        )
        .await
        .map_err(|err| MontyReplError::Execution(format!("llm_query spawn failed: {err}")))?;

    let status_result = wait_for_final_status(session.clone(), thread_id).await;
    let _ = session
        .services
        .agent_control
        .shutdown_agent(thread_id)
        .await;
    let status = status_result?;
    match status {
        AgentStatus::Completed(Some(output)) => Ok(output),
        AgentStatus::Completed(None) => Err(MontyReplError::Execution(
            "llm_query child completed without a final message".to_string(),
        )),
        AgentStatus::Errored(err) => Err(MontyReplError::Execution(format!(
            "llm_query child failed: {err}"
        ))),
        AgentStatus::Shutdown => Err(MontyReplError::Execution(
            "llm_query child was shutdown before producing a value".to_string(),
        )),
        AgentStatus::NotFound => Err(MontyReplError::Execution(
            "llm_query child agent was not found".to_string(),
        )),
        AgentStatus::PendingInit | AgentStatus::Running => Err(MontyReplError::Execution(
            "llm_query child did not reach a final status".to_string(),
        )),
    }
}

async fn wait_for_final_status(
    session: Arc<Session>,
    thread_id: ThreadId,
) -> Result<AgentStatus, MontyReplError> {
    let mut status_rx = session
        .services
        .agent_control
        .subscribe_status(thread_id)
        .await
        .map_err(|err| {
            MontyReplError::Execution(format!(
                "failed to subscribe to child agent status {thread_id}: {err}"
            ))
        })?;
    let mut status = status_rx.borrow().clone();
    if is_final(&status) {
        return Ok(status);
    }
    loop {
        if status_rx.changed().await.is_err() {
            status = session.services.agent_control.get_status(thread_id).await;
            if is_final(&status) {
                return Ok(status);
            }
            return Err(MontyReplError::Execution(format!(
                "child agent status stream ended before completion for {thread_id}"
            )));
        }
        status = status_rx.borrow().clone();
        if is_final(&status) {
            return Ok(status);
        }
    }
}

fn parse_schema_argument(schema: MontyObject) -> Result<(Value, SchemaSpec), MontyReplError> {
    let schema_json = match schema {
        MontyObject::String(raw_schema) => {
            serde_json::from_str(&raw_schema).unwrap_or(Value::String(raw_schema))
        }
        other => monty_schema_to_json(other)?,
    };
    let schema_spec = parse_schema_spec(&schema_json)?;
    Ok((schema_json, schema_spec))
}

fn monty_schema_to_json(value: MontyObject) -> Result<Value, MontyReplError> {
    match value {
        MontyObject::None => Ok(Value::Null),
        MontyObject::Bool(value) => Ok(Value::Bool(value)),
        MontyObject::Int(value) => Ok(Value::Number(value.into())),
        MontyObject::Float(value) => serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| {
                MontyReplError::InvalidHostCall(
                    "llm_query schema contains a non-finite float".to_string(),
                )
            }),
        MontyObject::String(value) => Ok(Value::String(value)),
        MontyObject::List(items) | MontyObject::Tuple(items) => items
            .into_iter()
            .map(monty_schema_to_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        MontyObject::Dict(pairs) => {
            let mut object = serde_json::Map::new();
            for (key, value) in pairs {
                let MontyObject::String(key) = key else {
                    return Err(MontyReplError::InvalidHostCall(
                        "llm_query schema object keys must be strings".to_string(),
                    ));
                };
                object.insert(key, monty_schema_to_json(value)?);
            }
            Ok(Value::Object(object))
        }
        other => Err(MontyReplError::InvalidHostCall(format!(
            "llm_query schema contains unsupported value type `{other}`"
        ))),
    }
}

fn parse_schema_spec(schema: &Value) -> Result<SchemaSpec, MontyReplError> {
    match schema {
        Value::String(type_name) => parse_schema_type(type_name),
        Value::Object(object) => {
            if let Some(type_name) = object.get("type") {
                let Some(type_name) = type_name.as_str() else {
                    return Err(MontyReplError::InvalidHostCall(
                        "llm_query schema field `type` must be a string".to_string(),
                    ));
                };
                parse_typed_schema(type_name, object)
            } else {
                parse_object_properties_schema(object)
            }
        }
        Value::Array(items) => {
            if items.len() != 1 {
                return Err(MontyReplError::InvalidHostCall(
                    "llm_query schema list shorthand must contain exactly one item schema"
                        .to_string(),
                ));
            }
            let item_spec = parse_schema_spec(&items[0])?;
            Ok(SchemaSpec::Array(Box::new(item_spec)))
        }
        _ => Err(MontyReplError::InvalidHostCall(
            "llm_query schema must be a JSON object or a recognized type string".to_string(),
        )),
    }
}

fn parse_typed_schema(
    type_name: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<SchemaSpec, MontyReplError> {
    match normalize_schema_type(type_name) {
        "any" => Ok(SchemaSpec::Any),
        "bool" => Ok(SchemaSpec::Bool),
        "string" => Ok(SchemaSpec::String),
        "int" => Ok(SchemaSpec::Integer),
        "number" => Ok(SchemaSpec::Number),
        "null" => Ok(SchemaSpec::Null),
        "array" => {
            let item_spec = if let Some(item_schema) = object.get("items") {
                parse_schema_spec(item_schema)?
            } else {
                SchemaSpec::Any
            };
            Ok(SchemaSpec::Array(Box::new(item_spec)))
        }
        "object" => {
            if let Some(properties) = object.get("properties") {
                let Value::Object(properties) = properties else {
                    return Err(MontyReplError::InvalidHostCall(
                        "llm_query schema field `properties` must be an object".to_string(),
                    ));
                };
                parse_object_properties_schema(properties)
            } else {
                Ok(SchemaSpec::ObjectAny)
            }
        }
        unknown => Err(MontyReplError::InvalidHostCall(format!(
            "llm_query schema has unsupported type `{unknown}`"
        ))),
    }
}

fn parse_object_properties_schema(
    object: &serde_json::Map<String, Value>,
) -> Result<SchemaSpec, MontyReplError> {
    let mut properties = BTreeMap::new();
    for (key, value) in object {
        properties.insert(key.clone(), parse_schema_spec(value)?);
    }
    Ok(SchemaSpec::ObjectExact(properties))
}

fn parse_schema_type(type_name: &str) -> Result<SchemaSpec, MontyReplError> {
    match normalize_schema_type(type_name) {
        "any" => Ok(SchemaSpec::Any),
        "bool" => Ok(SchemaSpec::Bool),
        "string" => Ok(SchemaSpec::String),
        "int" => Ok(SchemaSpec::Integer),
        "number" => Ok(SchemaSpec::Number),
        "null" => Ok(SchemaSpec::Null),
        "array" => Ok(SchemaSpec::Array(Box::new(SchemaSpec::Any))),
        "object" => Ok(SchemaSpec::ObjectAny),
        unknown => Err(MontyReplError::InvalidHostCall(format!(
            "llm_query schema has unsupported type `{unknown}`"
        ))),
    }
}

fn normalize_schema_type(type_name: &str) -> &str {
    match type_name {
        "bool" | "boolean" => "bool",
        "str" | "string" => "string",
        "int" | "integer" => "int",
        "float" | "number" => "number",
        "list" | "array" => "array",
        "dict" | "object" | "map" => "object",
        "none" | "null" => "null",
        "*" | "any" | "json" => "any",
        other => other,
    }
}

fn parse_and_validate_response(raw_output: &str, schema: &SchemaSpec) -> Result<Value, String> {
    let parsed = parse_json_with_one_repair(raw_output)?;
    validate_schema_value(&parsed, schema, "$")?;
    Ok(parsed)
}

fn parse_json_with_one_repair(raw_output: &str) -> Result<Value, String> {
    match serde_json::from_str::<Value>(raw_output) {
        Ok(value) => Ok(value),
        Err(initial_error) => {
            let Some(repair_candidate) = json_repair_candidate(raw_output) else {
                return Err(format!("invalid JSON output: {initial_error}"));
            };
            serde_json::from_str::<Value>(&repair_candidate).map_err(|repair_error| {
                format!(
                    "invalid JSON output: initial parse failed ({initial_error}); repair parse failed ({repair_error})"
                )
            })
        }
    }
}

fn json_repair_candidate(raw_output: &str) -> Option<String> {
    let trimmed = raw_output.trim();
    if trimmed.starts_with("```")
        && let Some(first_newline) = trimmed.find('\n')
        && let Some(last_fence) = trimmed.rfind("```")
        && last_fence > first_newline
    {
        return Some(trimmed[first_newline + 1..last_fence].trim().to_string());
    }

    let object_candidate = trimmed
        .find('{')
        .zip(trimmed.rfind('}'))
        .filter(|(start, end)| start < end)
        .map(|(start, end)| trimmed[start..=end].trim().to_string());
    let array_candidate = trimmed
        .find('[')
        .zip(trimmed.rfind(']'))
        .filter(|(start, end)| start < end)
        .map(|(start, end)| trimmed[start..=end].trim().to_string());
    match (object_candidate, array_candidate) {
        (Some(object), Some(array)) => {
            if trimmed.find('{').unwrap_or(usize::MAX) < trimmed.find('[').unwrap_or(usize::MAX) {
                Some(object)
            } else {
                Some(array)
            }
        }
        (Some(object), None) => Some(object),
        (None, Some(array)) => Some(array),
        (None, None) => None,
    }
}

fn validate_schema_value(value: &Value, schema: &SchemaSpec, path: &str) -> Result<(), String> {
    match schema {
        SchemaSpec::Any => Ok(()),
        SchemaSpec::Bool => {
            if value.is_boolean() {
                Ok(())
            } else {
                Err(type_mismatch_error(path, "bool", value))
            }
        }
        SchemaSpec::String => {
            if value.is_string() {
                Ok(())
            } else {
                Err(type_mismatch_error(path, "string", value))
            }
        }
        SchemaSpec::Integer => {
            if value.is_i64() || value.is_u64() {
                Ok(())
            } else {
                Err(type_mismatch_error(path, "int", value))
            }
        }
        SchemaSpec::Number => {
            if value.is_number() {
                Ok(())
            } else {
                Err(type_mismatch_error(path, "number", value))
            }
        }
        SchemaSpec::Null => {
            if value.is_null() {
                Ok(())
            } else {
                Err(type_mismatch_error(path, "null", value))
            }
        }
        SchemaSpec::Array(item_spec) => {
            let Some(items) = value.as_array() else {
                return Err(type_mismatch_error(path, "array", value));
            };
            for (index, item) in items.iter().enumerate() {
                validate_schema_value(item, item_spec, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        SchemaSpec::ObjectAny => {
            if value.is_object() {
                Ok(())
            } else {
                Err(type_mismatch_error(path, "object", value))
            }
        }
        SchemaSpec::ObjectExact(properties) => {
            let Some(object) = value.as_object() else {
                return Err(type_mismatch_error(path, "object", value));
            };
            let mut missing_keys = Vec::new();
            for key in properties.keys() {
                if !object.contains_key(key) {
                    missing_keys.push(key.clone());
                }
            }
            if !missing_keys.is_empty() {
                return Err(format!("{path} missing keys: {}", missing_keys.join(", ")));
            }
            let mut extra_keys = Vec::new();
            for key in object.keys() {
                if !properties.contains_key(key) {
                    extra_keys.push(key.clone());
                }
            }
            if !extra_keys.is_empty() {
                return Err(format!("{path} has extra keys: {}", extra_keys.join(", ")));
            }
            for (key, property_schema) in properties {
                let property_value = &object[key];
                validate_schema_value(property_value, property_schema, &format!("{path}.{key}"))?;
            }
            Ok(())
        }
    }
}

fn type_mismatch_error(path: &str, expected: &str, actual: &Value) -> String {
    format!(
        "{path} expected {expected}, got {}",
        describe_json_value_type(actual)
    )
}

fn describe_json_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                "int"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn build_repair_prompt(
    original_prompt: &str,
    schema_json: &Value,
    invalid_output: &str,
    validation_error: &str,
) -> String {
    let schema_text =
        serde_json::to_string_pretty(schema_json).unwrap_or_else(|_| schema_json.to_string());
    format!(
        "Repair this response into valid JSON that matches the required schema.\n\n\
Original task:\n{original_prompt}\n\n\
Required schema:\n{schema_text}\n\n\
Validation error:\n{validation_error}\n\n\
Invalid output:\n{invalid_output}\n\n\
Return exactly one valid JSON value matching the schema. Do not include markdown or extra prose."
    )
}

fn thread_spawn_source(parent_thread_id: ThreadId, depth: i32) -> SessionSource {
    SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id,
        depth,
        agent_nickname: None,
        agent_role: None,
    })
}

fn json_to_monty_object(value: Value) -> Result<MontyObject, MontyReplError> {
    match value {
        Value::Null => Ok(MontyObject::None),
        Value::Bool(value) => Ok(MontyObject::Bool(value)),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                return Ok(MontyObject::Int(integer));
            }
            if let Some(integer) = value.as_u64()
                && let Ok(integer) = i64::try_from(integer)
            {
                return Ok(MontyObject::Int(integer));
            }
            value.as_f64().map(MontyObject::Float).ok_or_else(|| {
                MontyReplError::Execution(
                    "failed to convert llm_query number into a Monty object".to_string(),
                )
            })
        }
        Value::String(value) => Ok(MontyObject::String(value)),
        Value::Array(items) => items
            .into_iter()
            .map(json_to_monty_object)
            .collect::<Result<Vec<_>, _>>()
            .map(MontyObject::List),
        Value::Object(object) => {
            let mut pairs = Vec::with_capacity(object.len());
            for (key, value) in object {
                pairs.push((MontyObject::String(key), json_to_monty_object(value)?));
            }
            Ok(MontyObject::dict(pairs))
        }
    }
}

fn function_call_to_monty_error(err: FunctionCallError) -> MontyReplError {
    match err {
        FunctionCallError::RespondToModel(message) => {
            MontyReplError::Execution(format!("llm_query spawn setup failed: {message}"))
        }
        FunctionCallError::Fatal(message) => {
            MontyReplError::Execution(format!("llm_query spawn setup failed: {message}"))
        }
        FunctionCallError::MissingLocalShellCallId => MontyReplError::Execution(
            "llm_query spawn setup failed: missing local shell call id".to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex::make_session_and_context;
    use codex_protocol::protocol::SessionSource;
    use codex_protocol::protocol::SubAgentSource;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    fn schema_dict() -> MontyObject {
        MontyObject::dict(vec![
            (
                MontyObject::String("relevant".to_string()),
                MontyObject::String("bool".to_string()),
            ),
            (
                MontyObject::String("key_facts".to_string()),
                MontyObject::String("list".to_string()),
            ),
        ])
    }

    #[tokio::test]
    async fn llm_query_round_trips_valid_json_response() {
        let responses = Arc::new(Mutex::new(VecDeque::from([Ok(
            r#"{"relevant":true,"key_facts":["a"]}"#.to_string(),
        )])));
        let result = run_llm_query_with_runner("analyze".to_string(), schema_dict(), {
            let responses = Arc::clone(&responses);
            move |_prompt| {
                let responses = Arc::clone(&responses);
                async move {
                    responses
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .pop_front()
                        .expect("runner response")
                }
            }
        })
        .await
        .expect("llm_query should succeed");

        let result_json = monty_schema_to_json(result).expect("monty object should convert");
        assert_eq!(result_json, json!({"relevant": true, "key_facts": ["a"]}));
    }

    #[tokio::test]
    async fn llm_query_repairs_bad_json_once() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Ok("not-json".to_string()),
            Ok(r#"{"relevant":false,"key_facts":[]}"#.to_string()),
        ])));
        let prompts = Arc::new(Mutex::new(Vec::new()));

        let result = run_llm_query_with_runner("analyze".to_string(), schema_dict(), {
            let responses = Arc::clone(&responses);
            let prompts = Arc::clone(&prompts);
            move |prompt| {
                let responses = Arc::clone(&responses);
                let prompts = Arc::clone(&prompts);
                async move {
                    prompts
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(prompt);
                    responses
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .pop_front()
                        .expect("runner response")
                }
            }
        })
        .await
        .expect("llm_query should repair and succeed");

        let prompt_log = prompts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(prompt_log.len(), 2);
        assert!(prompt_log[1].contains("Validation error"));
        let result_json = monty_schema_to_json(result).expect("monty object should convert");
        assert_eq!(result_json, json!({"relevant": false, "key_facts": []}));
    }

    #[tokio::test]
    async fn llm_query_repairs_wrong_keys_once() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Ok(r#"{"relevant":true}"#.to_string()),
            Ok(r#"{"relevant":true,"key_facts":["x"]}"#.to_string()),
        ])));
        let prompts = Arc::new(Mutex::new(Vec::new()));

        let result = run_llm_query_with_runner("analyze".to_string(), schema_dict(), {
            let responses = Arc::clone(&responses);
            let prompts = Arc::clone(&prompts);
            move |prompt| {
                let responses = Arc::clone(&responses);
                let prompts = Arc::clone(&prompts);
                async move {
                    prompts
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(prompt);
                    responses
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .pop_front()
                        .expect("runner response")
                }
            }
        })
        .await
        .expect("llm_query should repair and succeed");

        let prompt_log = prompts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(prompt_log.len(), 2);
        assert!(prompt_log[1].contains("missing keys"));
        let result_json = monty_schema_to_json(result).expect("monty object should convert");
        assert_eq!(result_json, json!({"relevant": true, "key_facts": ["x"]}));
    }

    #[tokio::test]
    async fn llm_query_surfaces_error_after_repair_failure() {
        let responses = Arc::new(Mutex::new(VecDeque::from([
            Ok("not-json".to_string()),
            Ok("still-not-json".to_string()),
        ])));

        let err = run_llm_query_with_runner("analyze".to_string(), schema_dict(), {
            let responses = Arc::clone(&responses);
            move |_prompt| {
                let responses = Arc::clone(&responses);
                async move {
                    responses
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .pop_front()
                        .expect("runner response")
                }
            }
        })
        .await
        .expect_err("llm_query should fail after one repair attempt");

        assert!(
            err.to_string()
                .contains("llm_query validation failed after one repair attempt")
        );
    }

    #[tokio::test]
    async fn run_llm_query_enforces_depth_limit() {
        let (session, mut turn) = make_session_and_context().await;
        let max_depth = turn.config.agent_max_depth;
        turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
            parent_thread_id: session.conversation_id,
            depth: max_depth,
            agent_nickname: None,
            agent_role: None,
        });

        let err = run_llm_query(
            Arc::new(session),
            Arc::new(turn),
            "analyze".to_string(),
            schema_dict(),
        )
        .await
        .expect_err("depth exceeded should error");
        assert!(err.to_string().contains("Agent depth limit reached"));
    }

    #[tokio::test]
    async fn llm_query_batched_preserves_result_order() {
        let result = run_llm_query_batched_with_query_fn(
            vec!["first".to_string(), "second".to_string()],
            schema_dict(),
            |prompt, _schema| async move {
                let delay_ms = if prompt == "first" { 30 } else { 1 };
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                Ok(MontyObject::String(prompt))
            },
        )
        .await
        .expect("batched query should succeed");

        assert_eq!(
            result,
            MontyObject::List(vec![
                MontyObject::String("first".to_string()),
                MontyObject::String("second".to_string()),
            ])
        );
    }

    #[tokio::test]
    async fn llm_query_batched_surfaces_partial_failure() {
        let err = run_llm_query_batched_with_query_fn(
            vec!["ok".to_string(), "bad".to_string()],
            schema_dict(),
            |prompt, _schema| async move {
                if prompt == "bad" {
                    return Err(MontyReplError::Execution("boom".to_string()));
                }
                Ok(MontyObject::String(prompt))
            },
        )
        .await
        .expect_err("batched query should fail when one item fails");

        assert!(err.to_string().contains("llm_query_batched item 1 failed"));
    }
}
