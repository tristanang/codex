use clap::Args;
use clap::Parser;
use clap::Subcommand;
use codex_core::config::find_codex_home;
use codex_core::github_copilot_auth;
use codex_core::github_copilot_auth::GithubCopilotAuth;
use reqwest::Client;
use reqwest::header::ACCEPT;
use reqwest::header::CONTENT_TYPE;
use reqwest::header::USER_AGENT;
use serde::Deserialize;
use std::io;
use std::path::Path;
use std::time::Duration;

const GITHUB_COPILOT_OAUTH_CLIENT_ID: &str = "Ov23li8tweQw6odWQebz";
const GITHUB_COPILOT_OAUTH_SCOPE: &str = "read:user";
const DEFAULT_GITHUB_DOMAIN: &str = "github.com";
const LOGIN_SUCCESS_MESSAGE: &str = "Successfully logged in";

#[derive(Debug, Clone, Copy)]
struct GithubOAuthPollingConfig {
    safety_margin_ms: u64,
    interval_floor_secs: u64,
    slow_down_floor_secs: u64,
}

const DEFAULT_GITHUB_OAUTH_POLLING_CONFIG: GithubOAuthPollingConfig = GithubOAuthPollingConfig {
    safety_margin_ms: 3_000,
    interval_floor_secs: 1,
    slow_down_floor_secs: 10,
};

#[derive(Debug, Deserialize)]
struct GitHubDeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct GitHubAccessTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
    interval: Option<u64>,
}

#[derive(Debug, Parser, PartialEq, Eq)]
#[command(name = "codex-provider-auth")]
#[command(about = "Provider authentication helpers")]
struct Cli {
    #[command(subcommand)]
    provider: ProviderCommand,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
enum ProviderCommand {
    /// GitHub provider authentication helpers.
    Github(GithubCommand),
}

#[derive(Debug, Args, PartialEq, Eq)]
struct GithubCommand {
    #[command(subcommand)]
    command: GithubSubcommand,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
enum GithubSubcommand {
    /// Login via GitHub device-code OAuth and save credentials.
    Login(GithubLoginArgs),

    /// Remove saved GitHub OAuth credentials.
    Logout,
}

#[derive(Debug, Args, PartialEq, Eq)]
struct GithubLoginArgs {
    /// GitHub Enterprise URL/domain (for example: github.example.com).
    #[arg(long = "enterprise-url", value_name = "URL")]
    enterprise_url: Option<String>,
}

#[tokio::main]
async fn main() {
    std::process::exit(run().await);
}

async fn run() -> i32 {
    let cli = Cli::parse();
    let codex_home = match find_codex_home() {
        Ok(codex_home) => codex_home,
        Err(err) => {
            eprintln!("Error resolving CODEX_HOME: {err}");
            return 1;
        }
    };

    match cli.provider {
        ProviderCommand::Github(github) => match github.command {
            GithubSubcommand::Login(args) => {
                match run_github_login(&codex_home, args.enterprise_url).await {
                    Ok(()) => {
                        eprintln!("{LOGIN_SUCCESS_MESSAGE}");
                        0
                    }
                    Err(err) => {
                        eprintln!("Error logging in with GitHub Copilot: {err}");
                        1
                    }
                }
            }
            GithubSubcommand::Logout => match run_github_logout(&codex_home) {
                Ok(true) => {
                    eprintln!("Removed GitHub Copilot credentials");
                    0
                }
                Ok(false) => {
                    eprintln!("No GitHub Copilot credentials found");
                    0
                }
                Err(err) => {
                    eprintln!("Error removing GitHub Copilot credentials: {err}");
                    1
                }
            },
        },
    }
}

async fn run_github_login(codex_home: &Path, enterprise_url: Option<String>) -> io::Result<()> {
    let enterprise_domain = match enterprise_url {
        Some(url) => Some(
            github_copilot_auth::normalize_enterprise_domain(&url).map_err(|err| {
                io::Error::new(err.kind(), format!("Invalid --enterprise-url: {err}"))
            })?,
        ),
        None => None,
    };

    let access_token = login_with_github_copilot_device_code(
        enterprise_domain.as_deref(),
        DEFAULT_GITHUB_OAUTH_POLLING_CONFIG,
    )
    .await?;

    let auth = GithubCopilotAuth {
        access_token,
        enterprise_domain,
    };
    github_copilot_auth::save(codex_home, &auth)
}

fn run_github_logout(codex_home: &Path) -> io::Result<bool> {
    github_copilot_auth::remove(codex_home)
}

fn github_oauth_urls(enterprise_domain: Option<&str>) -> (String, String) {
    let domain = enterprise_domain.unwrap_or(DEFAULT_GITHUB_DOMAIN);
    (
        format!("https://{domain}/login/device/code"),
        format!("https://{domain}/login/oauth/access_token"),
    )
}

async fn login_with_github_copilot_device_code(
    enterprise_domain: Option<&str>,
    polling_config: GithubOAuthPollingConfig,
) -> io::Result<String> {
    let (device_code_url, access_token_url) = github_oauth_urls(enterprise_domain);
    login_with_github_copilot_device_code_for_urls(
        &device_code_url,
        &access_token_url,
        polling_config,
    )
    .await
}

async fn login_with_github_copilot_device_code_for_urls(
    device_code_url: &str,
    access_token_url: &str,
    polling_config: GithubOAuthPollingConfig,
) -> io::Result<String> {
    let client = Client::new();
    let user_agent = format!("codex/{}", env!("CARGO_PKG_VERSION"));

    let device_response = client
        .post(device_code_url)
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json")
        .header(USER_AGENT, &user_agent)
        .json(&serde_json::json!({
            "client_id": GITHUB_COPILOT_OAUTH_CLIENT_ID,
            "scope": GITHUB_COPILOT_OAUTH_SCOPE,
        }))
        .send()
        .await
        .map_err(|err| {
            io::Error::other(format!("failed to start GitHub device-code login: {err}"))
        })?;

    if !device_response.status().is_success() {
        let status = device_response.status();
        let body = device_response.text().await.unwrap_or_default();
        return Err(io::Error::other(format!(
            "GitHub device-code login init failed ({status}): {}",
            body.trim()
        )));
    }

    let device_data: GitHubDeviceCodeResponse = device_response.json().await.map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse GitHub device-code response: {err}"),
        )
    })?;

    eprintln!("Complete GitHub Copilot login:");
    eprintln!("1. Open: {}", device_data.verification_uri);
    eprintln!("2. Enter code: {}", device_data.user_code);

    let mut poll_interval_ms = device_data
        .interval
        .unwrap_or(5)
        .max(polling_config.interval_floor_secs)
        * 1000
        + polling_config.safety_margin_ms;
    loop {
        tokio::time::sleep(Duration::from_millis(poll_interval_ms)).await;

        let token_response = client
            .post(access_token_url)
            .header(ACCEPT, "application/json")
            .header(CONTENT_TYPE, "application/json")
            .header(USER_AGENT, &user_agent)
            .json(&serde_json::json!({
                "client_id": GITHUB_COPILOT_OAUTH_CLIENT_ID,
                "device_code": device_data.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }))
            .send()
            .await
            .map_err(|err| io::Error::other(format!("failed to poll GitHub OAuth token: {err}")))?;

        if !token_response.status().is_success() {
            let status = token_response.status();
            let body = token_response.text().await.unwrap_or_default();
            return Err(io::Error::other(format!(
                "GitHub OAuth token exchange failed ({status}): {}",
                body.trim()
            )));
        }

        let token_data: GitHubAccessTokenResponse = token_response.json().await.map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse GitHub token response: {err}"),
            )
        })?;

        if let Some(access_token) = token_data.access_token {
            if access_token.trim().is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GitHub OAuth returned an empty access token",
                ));
            }
            return Ok(access_token);
        }

        match token_data.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                poll_interval_ms = token_data
                    .interval
                    .unwrap_or(10)
                    .max(polling_config.slow_down_floor_secs)
                    * 1000
                    + polling_config.safety_margin_ms;
                continue;
            }
            Some("expired_token") => {
                return Err(io::Error::other(
                    "GitHub device code expired before authorization completed",
                ));
            }
            Some("access_denied") => {
                return Err(io::Error::other("GitHub login was denied by the user"));
            }
            Some(other) => {
                let description = token_data.error_description.unwrap_or_default();
                return Err(io::Error::other(format!(
                    "GitHub OAuth login failed: {other} {description}"
                )));
            }
            None => {
                return Err(io::Error::other(
                    "GitHub OAuth response was missing both access_token and error",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::sync::Arc;
    use std::sync::Mutex;
    use tempfile::tempdir;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::Request;
    use wiremock::Respond;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::method;
    use wiremock::matchers::path;

    const FAST_POLLING: GithubOAuthPollingConfig = GithubOAuthPollingConfig {
        safety_margin_ms: 0,
        interval_floor_secs: 0,
        slow_down_floor_secs: 0,
    };

    #[derive(Clone)]
    struct SequentialResponder {
        templates: Arc<Mutex<Vec<ResponseTemplate>>>,
    }

    impl SequentialResponder {
        fn new(templates: Vec<ResponseTemplate>) -> Self {
            Self {
                templates: Arc::new(Mutex::new(templates)),
            }
        }
    }

    impl Respond for SequentialResponder {
        fn respond(&self, _: &Request) -> ResponseTemplate {
            let mut templates = self.templates.lock().expect("templates mutex");
            if templates.is_empty() {
                return ResponseTemplate::new(500)
                    .set_body_json(serde_json::json!({ "error": "exhausted sequence" }));
            }
            templates.remove(0)
        }
    }

    #[test]
    fn parses_github_login_with_enterprise_url() {
        let cli = Cli::try_parse_from([
            "codex-provider-auth",
            "github",
            "login",
            "--enterprise-url",
            "github.example.com",
        ])
        .expect("parse");

        assert_eq!(
            cli,
            Cli {
                provider: ProviderCommand::Github(GithubCommand {
                    command: GithubSubcommand::Login(GithubLoginArgs {
                        enterprise_url: Some("github.example.com".to_string()),
                    }),
                }),
            }
        );
    }

    #[test]
    fn parses_github_logout() {
        let logout =
            Cli::try_parse_from(["codex-provider-auth", "github", "logout"]).expect("logout parse");

        assert_eq!(
            logout,
            Cli {
                provider: ProviderCommand::Github(GithubCommand {
                    command: GithubSubcommand::Logout,
                }),
            }
        );
    }

    #[tokio::test]
    async fn login_flow_returns_access_token_on_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "device-code",
                "user_code": "USER-CODE",
                "verification_uri": "https://github.com/login/device",
                "interval": 0,
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "token-123",
            })))
            .mount(&server)
            .await;

        let token = login_with_github_copilot_device_code_for_urls(
            &format!("{}/login/device/code", server.uri()),
            &format!("{}/login/oauth/access_token", server.uri()),
            FAST_POLLING,
        )
        .await
        .expect("login success");

        assert_eq!(token, "token-123");
    }

    #[tokio::test]
    async fn login_flow_handles_pending_and_slow_down_before_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "device-code",
                "user_code": "USER-CODE",
                "verification_uri": "https://github.com/login/device",
                "interval": 0,
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(SequentialResponder::new(vec![
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "error": "authorization_pending",
                })),
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "error": "slow_down",
                    "interval": 0,
                })),
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "access_token": "token-after-retries",
                })),
            ]))
            .mount(&server)
            .await;

        let token = login_with_github_copilot_device_code_for_urls(
            &format!("{}/login/device/code", server.uri()),
            &format!("{}/login/oauth/access_token", server.uri()),
            FAST_POLLING,
        )
        .await
        .expect("login success after retries");

        assert_eq!(token, "token-after-retries");
    }

    #[tokio::test]
    async fn login_flow_returns_expired_token_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "device-code",
                "user_code": "USER-CODE",
                "verification_uri": "https://github.com/login/device",
                "interval": 0,
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "error": "expired_token",
            })))
            .mount(&server)
            .await;

        let err = login_with_github_copilot_device_code_for_urls(
            &format!("{}/login/device/code", server.uri()),
            &format!("{}/login/oauth/access_token", server.uri()),
            FAST_POLLING,
        )
        .await
        .expect_err("expired token should fail");

        assert!(
            err.to_string().contains("expired"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn login_flow_returns_access_denied_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "device-code",
                "user_code": "USER-CODE",
                "verification_uri": "https://github.com/login/device",
                "interval": 0,
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "error": "access_denied",
            })))
            .mount(&server)
            .await;

        let err = login_with_github_copilot_device_code_for_urls(
            &format!("{}/login/device/code", server.uri()),
            &format!("{}/login/oauth/access_token", server.uri()),
            FAST_POLLING,
        )
        .await
        .expect_err("access denied should fail");

        assert!(
            err.to_string().contains("denied"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn login_flow_returns_oauth_error_details() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "device-code",
                "user_code": "USER-CODE",
                "verification_uri": "https://github.com/login/device",
                "interval": 0,
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/login/oauth/access_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "error": "unsupported_grant_type",
                "error_description": "bad grant",
            })))
            .mount(&server)
            .await;

        let err = login_with_github_copilot_device_code_for_urls(
            &format!("{}/login/device/code", server.uri()),
            &format!("{}/login/oauth/access_token", server.uri()),
            FAST_POLLING,
        )
        .await
        .expect_err("oauth error should fail");

        assert!(
            err.to_string().contains("unsupported_grant_type bad grant"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn logout_is_idempotent() {
        let codex_home = tempdir().expect("tempdir");

        assert_eq!(run_github_logout(codex_home.path()).expect("logout"), false);

        github_copilot_auth::save(
            codex_home.path(),
            &GithubCopilotAuth {
                access_token: "token-123".to_string(),
                enterprise_domain: Some("enterprise.example.com".to_string()),
            },
        )
        .expect("save auth");

        assert_eq!(run_github_logout(codex_home.path()).expect("logout"), true);
        assert_eq!(run_github_logout(codex_home.path()).expect("logout"), false);
    }
}
