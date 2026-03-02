use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::headers::build_conversation_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use tracing::warn;

/// Assembled request body plus headers for Chat Completions streaming calls.
pub struct ChatRequest {
    pub body: Value,
    pub headers: HeaderMap,
}

pub struct ChatRequestBuilder<'a> {
    model: &'a str,
    instructions: &'a str,
    input: &'a [ResponseItem],
    tools: &'a [Value],
    conversation_id: Option<String>,
    session_source: Option<SessionSource>,
}

impl<'a> ChatRequestBuilder<'a> {
    pub fn new(
        model: &'a str,
        instructions: &'a str,
        input: &'a [ResponseItem],
        tools: &'a [Value],
    ) -> Self {
        Self {
            model,
            instructions,
            input,
            tools,
            conversation_id: None,
            session_source: None,
        }
    }

    pub fn conversation_id(mut self, id: Option<String>) -> Self {
        self.conversation_id = id;
        self
    }

    pub fn session_source(mut self, source: Option<SessionSource>) -> Self {
        self.session_source = source;
        self
    }

    pub fn build(self, _provider: &Provider) -> Result<ChatRequest, ApiError> {
        let mut messages = Vec::<Value>::new();
        messages.push(json!({"role": "system", "content": self.instructions}));

        let input = self.input;
        let mut reasoning_by_anchor_index: HashMap<usize, String> = HashMap::new();
        let mut last_emitted_role: Option<&str> = None;
        for item in input {
            match item {
                ResponseItem::Message { role, .. } => last_emitted_role = Some(role.as_str()),
                ResponseItem::FunctionCall { .. }
                | ResponseItem::LocalShellCall { .. }
                | ResponseItem::CustomToolCall { .. } => last_emitted_role = Some("assistant"),
                ResponseItem::FunctionCallOutput { .. }
                | ResponseItem::CustomToolCallOutput { .. } => last_emitted_role = Some("tool"),
                ResponseItem::Reasoning { .. } | ResponseItem::Other => {}
                ResponseItem::WebSearchCall { .. } => {}
                ResponseItem::GhostSnapshot { .. } => {}
                ResponseItem::Compaction { .. } => {}
            }
        }

        let mut last_user_index: Option<usize> = None;
        for (idx, item) in input.iter().enumerate() {
            if let ResponseItem::Message { role, .. } = item
                && role == "user"
            {
                last_user_index = Some(idx);
            }
        }

        if !matches!(last_emitted_role, Some("user")) {
            for (idx, item) in input.iter().enumerate() {
                if let Some(u_idx) = last_user_index
                    && idx <= u_idx
                {
                    continue;
                }

                if let ResponseItem::Reasoning {
                    content: Some(items),
                    ..
                } = item
                {
                    let mut text = String::new();
                    for entry in items {
                        match entry {
                            ReasoningItemContent::ReasoningText { text: segment }
                            | ReasoningItemContent::Text { text: segment } => {
                                text.push_str(segment)
                            }
                        }
                    }
                    if text.trim().is_empty() {
                        continue;
                    }

                    let mut attached = false;
                    if idx > 0
                        && let ResponseItem::Message { role, .. } = &input[idx - 1]
                        && role == "assistant"
                    {
                        reasoning_by_anchor_index
                            .entry(idx - 1)
                            .and_modify(|v| v.push_str(&text))
                            .or_insert(text.clone());
                        attached = true;
                    }

                    if !attached && idx + 1 < input.len() {
                        match &input[idx + 1] {
                            ResponseItem::FunctionCall { .. }
                            | ResponseItem::LocalShellCall { .. } => {
                                reasoning_by_anchor_index
                                    .entry(idx + 1)
                                    .and_modify(|v| v.push_str(&text))
                                    .or_insert(text.clone());
                            }
                            ResponseItem::Message { role, .. } if role == "assistant" => {
                                reasoning_by_anchor_index
                                    .entry(idx + 1)
                                    .and_modify(|v| v.push_str(&text))
                                    .or_insert(text.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let mut last_assistant_text: Option<String> = None;

        for (idx, item) in input.iter().enumerate() {
            match item {
                ResponseItem::Message { role, content, .. } => {
                    let mut text = String::new();
                    let mut items: Vec<Value> = Vec::new();
                    let mut saw_image = false;

                    for c in content {
                        match c {
                            ContentItem::InputText { text: t }
                            | ContentItem::OutputText { text: t } => {
                                text.push_str(t);
                                items.push(json!({"type":"text","text": t}));
                            }
                            ContentItem::InputImage { image_url } => {
                                saw_image = true;
                                items.push(
                                    json!({"type":"image_url","image_url": {"url": image_url}}),
                                );
                            }
                        }
                    }

                    let merge_into_previous_tool_call_message = role == "assistant"
                        && matches!(
                            messages.last(),
                            Some(Value::Object(obj))
                                if obj.get("role").and_then(Value::as_str) == Some("assistant")
                                    && obj.get("tool_calls").is_some()
                        );

                    if role == "assistant" {
                        if !merge_into_previous_tool_call_message
                            && let Some(prev) = &last_assistant_text
                            && prev == &text
                        {
                            continue;
                        }
                        last_assistant_text = Some(text.clone());
                    }

                    let content_value = if role == "assistant" {
                        json!(text)
                    } else if saw_image {
                        json!(items)
                    } else {
                        json!(text)
                    };

                    if merge_into_previous_tool_call_message {
                        if let Some(Value::Object(obj)) = messages.last_mut() {
                            if let Some(content) = obj.get_mut("content") {
                                match content {
                                    Value::String(existing) => existing.push_str(&text),
                                    _ => *content = Value::String(text.clone()),
                                }
                            } else {
                                obj.insert("content".to_string(), Value::String(text.clone()));
                            }

                            if let Some(reasoning) = reasoning_by_anchor_index.get(&idx) {
                                if let Some(Value::String(existing)) = obj.get_mut("reasoning") {
                                    if !existing.is_empty() {
                                        existing.push('\n');
                                    }
                                    existing.push_str(reasoning);
                                } else {
                                    obj.insert(
                                        "reasoning".to_string(),
                                        Value::String(reasoning.clone()),
                                    );
                                }
                            }
                        }
                        continue;
                    }

                    let mut msg = json!({"role": role, "content": content_value});
                    if role == "assistant"
                        && let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                        && let Some(obj) = msg.as_object_mut()
                    {
                        obj.insert("reasoning".to_string(), json!(reasoning));
                    }
                    messages.push(msg);
                }
                ResponseItem::FunctionCall {
                    name,
                    arguments,
                    call_id,
                    ..
                } => {
                    let reasoning = reasoning_by_anchor_index.get(&idx).map(String::as_str);
                    let normalized_arguments =
                        normalize_tool_call_arguments(arguments, call_id, name);
                    let tool_call = json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": normalized_arguments,
                        }
                    });
                    push_tool_call_message(&mut messages, tool_call, reasoning);
                }
                ResponseItem::LocalShellCall {
                    id,
                    call_id,
                    status,
                    action,
                } => {
                    let reasoning = reasoning_by_anchor_index.get(&idx).map(String::as_str);
                    let resolved_id = call_id.clone().or_else(|| id.clone()).unwrap_or_default();
                    let tool_call = json!({
                        "id": resolved_id,
                        "type": "local_shell_call",
                        "status": status,
                        "action": action,
                    });
                    push_tool_call_message(&mut messages, tool_call, reasoning);
                }
                ResponseItem::FunctionCallOutput { call_id, output } => {
                    let content_value = if let Some(items) = output.content_items() {
                        let mapped: Vec<Value> = items
                            .iter()
                            .map(|it| match it {
                                FunctionCallOutputContentItem::InputText { text } => {
                                    json!({"type":"text","text": text})
                                }
                                FunctionCallOutputContentItem::InputImage { image_url } => {
                                    json!({"type":"image_url","image_url": {"url": image_url}})
                                }
                            })
                            .collect();
                        json!(mapped)
                    } else {
                        json!(output.text_content().unwrap_or_default())
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_value,
                    }));
                }
                ResponseItem::CustomToolCall {
                    call_id,
                    name,
                    input,
                    ..
                } => {
                    let tool_call = json!({
                        "id": call_id,
                        "type": "custom",
                        "custom": {
                            "name": name,
                            "input": input,
                        }
                    });
                    let reasoning = reasoning_by_anchor_index.get(&idx).map(String::as_str);
                    push_tool_call_message(&mut messages, tool_call, reasoning);
                }
                ResponseItem::CustomToolCallOutput { call_id, output } => {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output,
                    }));
                }
                ResponseItem::GhostSnapshot { .. } => {
                    continue;
                }
                ResponseItem::Reasoning { .. }
                | ResponseItem::WebSearchCall { .. }
                | ResponseItem::Other
                | ResponseItem::Compaction { .. } => {
                    continue;
                }
            }
        }

        let payload = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "tools": self.tools,
        });

        let mut headers = build_conversation_headers(self.conversation_id);
        if let Some(subagent) = subagent_header(&self.session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        Ok(ChatRequest {
            body: payload,
            headers,
        })
    }
}

fn push_tool_call_message(messages: &mut Vec<Value>, tool_call: Value, reasoning: Option<&str>) {
    // Chat Completions requires that tool calls are grouped into a single assistant message
    // (with `tool_calls: [...]`) followed by tool role responses. When the assistant produces
    // text *and* tool calls in the same turn, the Responses-API history stores them as separate
    // items (a Message followed by one or more FunctionCall items). We must merge them back
    // into a single Chat Completions assistant message so that providers that enforce strict
    // role alternation (e.g. Claude) don't reject the request.
    if let Some(Value::Object(obj)) = messages.last_mut()
        && obj.get("role").and_then(Value::as_str) == Some("assistant")
    {
        if let Some(tool_calls) = obj.get_mut("tool_calls").and_then(Value::as_array_mut) {
            tool_calls.push(tool_call);
        } else {
            obj.insert("tool_calls".to_string(), json!([tool_call]));
        }
        if let Some(reasoning) = reasoning {
            if let Some(Value::String(existing)) = obj.get_mut("reasoning") {
                if !existing.is_empty() {
                    existing.push('\n');
                }
                existing.push_str(reasoning);
            } else {
                obj.insert(
                    "reasoning".to_string(),
                    Value::String(reasoning.to_string()),
                );
            }
        }
        return;
    }

    let mut msg = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [tool_call],
    });
    if let Some(reasoning) = reasoning
        && let Some(obj) = msg.as_object_mut()
    {
        obj.insert("reasoning".to_string(), json!(reasoning));
    }
    messages.push(msg);
}

fn normalize_tool_call_arguments(arguments: &str, call_id: &str, name: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => value.to_string(),
        Err(err) => {
            warn!(
                ?err,
                call_id,
                name,
                "invalid tool call arguments in history; replacing with empty object"
            );
            "{}".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::RetryConfig;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::protocol::SessionSource;
    use codex_protocol::protocol::SubAgentSource;
    use http::HeaderValue;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    fn provider() -> Provider {
        Provider {
            name: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            query_params: None,
            headers: HeaderMap::new(),
            retry: RetryConfig {
                max_attempts: 1,
                base_delay: Duration::from_millis(10),
                retry_429: false,
                retry_5xx: true,
                retry_transport: true,
            },
            stream_idle_timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn attaches_conversation_and_subagent_headers() {
        let prompt_input = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "hi".to_string(),
            }],
            end_turn: None,
            phase: None,
        }];
        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[])
            .conversation_id(Some("conv-1".into()))
            .session_source(Some(SessionSource::SubAgent(SubAgentSource::Review)))
            .build(&provider())
            .expect("request");

        assert_eq!(
            req.headers.get("session_id"),
            Some(&HeaderValue::from_static("conv-1"))
        );
        assert_eq!(
            req.headers.get("x-openai-subagent"),
            Some(&HeaderValue::from_static("review"))
        );
    }

    #[test]
    fn groups_consecutive_tool_calls_into_a_single_assistant_message() {
        let prompt_input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "read these".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "read_file".to_string(),
                arguments: r#"{"path":"a.txt"}"#.to_string(),
                call_id: "call-a".to_string(),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "read_file".to_string(),
                arguments: r#"{"path":"b.txt"}"#.to_string(),
                call_id: "call-b".to_string(),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "read_file".to_string(),
                arguments: r#"{"path":"c.txt"}"#.to_string(),
                call_id: "call-c".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-a".to_string(),
                output: FunctionCallOutputPayload::from_text("A".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-b".to_string(),
                output: FunctionCallOutputPayload::from_text("B".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-c".to_string(),
                output: FunctionCallOutputPayload::from_text("C".to_string()),
            },
        ];

        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[])
            .build(&provider())
            .expect("request");

        let messages = req
            .body
            .get("messages")
            .and_then(|v| v.as_array())
            .expect("messages array");
        // system + user + assistant(tool_calls=[...]) + 3 tool outputs
        assert_eq!(messages.len(), 6);

        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");

        let tool_calls_msg = &messages[2];
        assert_eq!(tool_calls_msg["role"], "assistant");
        assert_eq!(tool_calls_msg["content"], serde_json::Value::Null);
        let tool_calls = tool_calls_msg["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls.len(), 3);
        assert_eq!(tool_calls[0]["id"], "call-a");
        assert_eq!(tool_calls[1]["id"], "call-b");
        assert_eq!(tool_calls[2]["id"], "call-c");

        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call-a");
        assert_eq!(messages[4]["role"], "tool");
        assert_eq!(messages[4]["tool_call_id"], "call-b");
        assert_eq!(messages[5]["role"], "tool");
        assert_eq!(messages[5]["tool_call_id"], "call-c");
    }

    #[test]
    fn merges_tool_call_into_preceding_assistant_text_message() {
        // When Claude responds with text AND a tool call in the same turn, the
        // Responses-API history stores them as a Message item followed by a
        // FunctionCall item. The builder must merge them into a single Chat
        // Completions assistant message so Claude's API (which requires strict
        // role alternation) doesn't reject the request.
        let prompt_input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "read the file".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "Let me read that file.".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                arguments: r#"{"cmd":"cat foo.rs"}"#.to_string(),
                call_id: "call-1".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output: FunctionCallOutputPayload::from_text("file contents".to_string()),
            },
        ];

        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[])
            .build(&provider())
            .expect("request");

        let messages = req
            .body
            .get("messages")
            .and_then(|v| v.as_array())
            .expect("messages array");
        // system + user + assistant(text + tool_calls) + tool output
        assert_eq!(messages.len(), 4);

        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");

        let assistant_msg = &messages[2];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(assistant_msg["content"], "Let me read that file.");
        let tool_calls = assistant_msg["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call-1");

        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call-1");
    }

    #[test]
    fn merges_assistant_text_into_preceding_tool_call_message() {
        // Some providers can emit a tool call item and then an assistant text
        // item in the same turn. We must merge them into one assistant message
        // so tool outputs still immediately follow the tool call.
        let prompt_input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "read the file".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                arguments: r#"{"cmd":"cat foo.rs"}"#.to_string(),
                call_id: "call-1".to_string(),
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "Let me read that file.".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-1".to_string(),
                output: FunctionCallOutputPayload::from_text("file contents".to_string()),
            },
        ];

        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[])
            .build(&provider())
            .expect("request");

        let messages = req
            .body
            .get("messages")
            .and_then(|v| v.as_array())
            .expect("messages array");
        // system + user + assistant(text + tool_calls) + tool output
        assert_eq!(messages.len(), 4);

        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");

        let assistant_msg = &messages[2];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(assistant_msg["content"], "Let me read that file.");
        let tool_calls = assistant_msg["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call-1");

        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call-1");
    }

    #[test]
    fn custom_tool_call_uses_call_id_not_item_id() {
        let prompt_input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "search".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::CustomToolCall {
                id: Some("item-999".to_string()),
                status: None,
                call_id: "toolu_vrtx_abc123".to_string(),
                name: "mcp_search".to_string(),
                input: r#"{"q":"test"}"#.to_string(),
            },
            ResponseItem::CustomToolCallOutput {
                call_id: "toolu_vrtx_abc123".to_string(),
                output: FunctionCallOutputPayload::from_text("results".to_string()),
            },
        ];

        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[])
            .build(&provider())
            .expect("request");

        let messages = req
            .body
            .get("messages")
            .and_then(|v| v.as_array())
            .expect("messages array");
        // system + user + assistant(tool_calls) + tool output
        assert_eq!(messages.len(), 4);

        let assistant_msg = &messages[2];
        let tool_calls = assistant_msg["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        // Must use call_id, not the item id
        assert_eq!(tool_calls[0]["id"], "toolu_vrtx_abc123");

        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "toolu_vrtx_abc123");
    }

    #[test]
    fn replaces_invalid_function_arguments_with_empty_object() {
        let prompt_input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "run tool".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "bad_tool".to_string(),
                arguments: "{\"broken\":".to_string(),
                call_id: "call-bad".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call-bad".to_string(),
                output: FunctionCallOutputPayload::from_text("done".to_string()),
            },
        ];

        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[])
            .build(&provider())
            .expect("request");

        let messages = req
            .body
            .get("messages")
            .and_then(|v| v.as_array())
            .expect("messages array");

        let assistant_msg = &messages[2];
        let tool_calls = assistant_msg["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls[0]["id"], "call-bad");
        assert_eq!(tool_calls[0]["function"]["arguments"], "{}");
    }
}
