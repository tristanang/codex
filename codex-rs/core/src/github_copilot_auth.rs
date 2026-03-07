use crate::path_utils::write_atomically;
use serde::Deserialize;
use serde::Serialize;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use url::Url;

const GITHUB_COPILOT_AUTH_FILE: &str = "github-copilot-auth.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GithubCopilotAuth {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enterprise_domain: Option<String>,
}

pub fn enterprise_base_url(domain: &str) -> String {
    format!("https://copilot-api.{domain}")
}

pub fn normalize_enterprise_domain(input: &str) -> io::Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "enterprise URL/domain cannot be empty",
        ));
    }

    let parsed = if trimmed.contains("://") {
        Url::parse(trimmed)
    } else {
        Url::parse(&format!("https://{trimmed}"))
    }
    .map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid enterprise URL/domain: {err}"),
        )
    })?;

    let Some(host) = parsed.host_str() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "enterprise URL/domain must include a valid host",
        ));
    };

    let domain = host.trim().trim_end_matches('.').to_lowercase();
    if domain.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "enterprise URL/domain cannot resolve to an empty host",
        ));
    }
    Ok(domain)
}

fn auth_file_path(codex_home: &Path) -> PathBuf {
    codex_home.join(GITHUB_COPILOT_AUTH_FILE)
}

pub fn load(codex_home: &Path) -> io::Result<Option<GithubCopilotAuth>> {
    let path = auth_file_path(codex_home);
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(io::Error::new(
                err.kind(),
                format!("failed to read {}: {err}", path.display()),
            ));
        }
    };

    let record: GithubCopilotAuth = serde_json::from_str(&contents).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse {}: {err}", path.display()),
        )
    })?;

    if record.access_token.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} contains an empty access token", path.display()),
        ));
    }

    Ok(Some(record))
}

pub fn load_from_default_codex_home() -> Option<GithubCopilotAuth> {
    let codex_home = crate::config::find_codex_home().ok()?;
    load(&codex_home).ok().flatten()
}

pub fn save(codex_home: &Path, auth: &GithubCopilotAuth) -> io::Result<()> {
    if auth.access_token.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "access token cannot be empty",
        ));
    }

    let payload = serde_json::to_string_pretty(auth).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize GitHub Copilot auth data: {err}"),
        )
    })?;
    let path = auth_file_path(codex_home);
    write_atomically(&path, &(payload + "\n"))
}

pub fn remove(codex_home: &Path) -> io::Result<bool> {
    let path = auth_file_path(codex_home);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(io::Error::new(
            err.kind(),
            format!("failed to remove {}: {err}", path.display()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn normalize_enterprise_domain_accepts_domain_or_url() {
        assert_eq!(
            normalize_enterprise_domain("company.ghe.com").expect("normalize domain"),
            "company.ghe.com"
        );
        assert_eq!(
            normalize_enterprise_domain("https://Company.GHE.com/").expect("normalize URL"),
            "company.ghe.com"
        );
    }

    #[test]
    fn load_and_save_round_trip() {
        let tmp = tempdir().expect("tmpdir");
        let auth = GithubCopilotAuth {
            access_token: "test-token".to_string(),
            enterprise_domain: Some("enterprise.example.com".to_string()),
        };
        save(tmp.path(), &auth).expect("save");

        let loaded = load(tmp.path()).expect("load").expect("auth");
        assert_eq!(loaded, auth);
    }

    #[test]
    fn remove_is_idempotent() {
        let tmp = tempdir().expect("tmpdir");
        assert!(!remove(tmp.path()).expect("remove"));

        let auth = GithubCopilotAuth {
            access_token: "test-token".to_string(),
            enterprise_domain: None,
        };
        save(tmp.path(), &auth).expect("save");
        assert!(remove(tmp.path()).expect("remove"));
        assert!(!remove(tmp.path()).expect("remove"));
    }
}
