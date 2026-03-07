use super::*;
use pretty_assertions::assert_eq;

#[test]
fn test_deserialize_ollama_model_provider_toml() {
    let azure_provider_toml = r#"
name = "Ollama"
base_url = "http://localhost:11434/v1"
        "#;
    let expected_provider = ModelProviderInfo {
        name: "Ollama".into(),
        base_url: Some("http://localhost:11434/v1".into()),
        env_key: None,
        env_key_instructions: None,
        experimental_bearer_token: None,
        wire_api: WireApi::Responses,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    };

    let provider: ModelProviderInfo = toml::from_str(azure_provider_toml).unwrap();
    assert_eq!(expected_provider, provider);
}

#[test]
fn test_deserialize_azure_model_provider_toml() {
    let azure_provider_toml = r#"
name = "Azure"
base_url = "https://xxxxx.openai.azure.com/openai"
env_key = "AZURE_OPENAI_API_KEY"
query_params = { api-version = "2025-04-01-preview" }
        "#;
    let expected_provider = ModelProviderInfo {
        name: "Azure".into(),
        base_url: Some("https://xxxxx.openai.azure.com/openai".into()),
        env_key: Some("AZURE_OPENAI_API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        wire_api: WireApi::Responses,
        query_params: Some(maplit::hashmap! {
            "api-version".to_string() => "2025-04-01-preview".to_string(),
        }),
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    };

    let provider: ModelProviderInfo = toml::from_str(azure_provider_toml).unwrap();
    assert_eq!(expected_provider, provider);
}

#[test]
fn test_deserialize_example_model_provider_toml() {
    let azure_provider_toml = r#"
name = "Example"
base_url = "https://example.com"
env_key = "API_KEY"
http_headers = { "X-Example-Header" = "example-value" }
env_http_headers = { "X-Example-Env-Header" = "EXAMPLE_ENV_VAR" }
        "#;
    let expected_provider = ModelProviderInfo {
        name: "Example".into(),
        base_url: Some("https://example.com".into()),
        env_key: Some("API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        wire_api: WireApi::Responses,
        query_params: None,
        http_headers: Some(maplit::hashmap! {
            "X-Example-Header".to_string() => "example-value".to_string(),
        }),
        env_http_headers: Some(maplit::hashmap! {
            "X-Example-Env-Header".to_string() => "EXAMPLE_ENV_VAR".to_string(),
        }),
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    };

    let provider: ModelProviderInfo = toml::from_str(azure_provider_toml).unwrap();
    assert_eq!(expected_provider, provider);
}

#[test]
fn test_deserialize_chat_wire_api_shows_helpful_error() {
    let provider_toml = r#"
name = "OpenAI using Chat Completions"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
wire_api = "chat"
        "#;

    let err = toml::from_str::<ModelProviderInfo>(provider_toml).unwrap_err();
    assert!(err.to_string().contains(CHAT_WIRE_API_REMOVED_ERROR));
}

#[test]
fn github_copilot_provider_uses_responses_and_expected_headers() {
    let provider = ModelProviderInfo::create_github_copilot_provider();

    assert_eq!(provider.name, GITHUB_COPILOT_PROVIDER_NAME);
    assert_eq!(
        provider.base_url,
        Some(GITHUB_COPILOT_DEFAULT_BASE_URL.to_string())
    );
    assert_eq!(provider.env_key, Some(GITHUB_COPILOT_TOKEN_ENV_VAR.to_string()));
    assert_eq!(provider.wire_api, WireApi::Responses);
    assert_eq!(provider.requires_openai_auth, false);
    assert_eq!(provider.supports_websockets, false);

    let headers = provider.http_headers.expect("copilot provider has headers");
    assert_eq!(
        headers.get(GITHUB_COPILOT_INTENT_HEADER_NAME),
        Some(&GITHUB_COPILOT_INTENT_HEADER_VALUE.to_string())
    );
    assert_eq!(
        headers.get(GITHUB_COPILOT_INITIATOR_HEADER_NAME),
        Some(&GITHUB_COPILOT_INITIATOR_HEADER_VALUE.to_string())
    );
    assert_eq!(
        headers
            .get("User-Agent")
            .expect("copilot provider has user-agent")
            .starts_with(GITHUB_COPILOT_USER_AGENT_PREFIX),
        true
    );
}

#[test]
fn built_in_providers_include_github_copilot() {
    let providers = built_in_model_providers(None);
    assert_eq!(providers.contains_key(GITHUB_COPILOT_PROVIDER_ID), true);
    assert_eq!(
        providers[GITHUB_COPILOT_PROVIDER_ID].wire_api,
        WireApi::Responses
    );
}

#[test]
fn github_copilot_detection_matches_common_provider_shapes() {
    let provider = ModelProviderInfo::create_github_copilot_provider();
    assert!(provider.is_github_copilot());

    let mut by_env =
        create_oss_provider_with_base_url("http://localhost:1234/v1", WireApi::Responses);
    by_env.env_key = Some("GITHUB_TOKEN".to_string());
    assert!(by_env.is_github_copilot());

    let mut by_base_url =
        create_oss_provider_with_base_url("https://copilot-api.example.com", WireApi::Responses);
    by_base_url.env_key = None;
    assert!(by_base_url.is_github_copilot());
}
