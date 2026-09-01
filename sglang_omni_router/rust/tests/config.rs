#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict configuration boundary tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sgl_omni_router::{Config, ConfigError, LogFormat};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-config-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated config test directory");
        Self(path)
    }

    fn write(&self, contents: &[u8]) -> PathBuf {
        let path = self.0.join("router.toml");
        fs::write(&path, contents).expect("write isolated config fixture");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _cleanup_result = fs::remove_dir_all(&self.0);
    }
}

fn valid_config(listen: &str, drain_timeout_ms: u64, filter: &str) -> String {
    format!(
        "schema_version = 1\n\n[server]\nlisten = \"{listen}\"\n\n[shutdown]\ndrain_timeout_ms = {drain_timeout_ms}\n\n[logging]\nformat = \"json\"\nfilter = \"{filter}\"\n"
    )
}

fn with_max_connections(config: String, max_connections: usize) -> String {
    config.replace(
        "listen = \"127.0.0.1:30000\"",
        &format!("listen = \"127.0.0.1:30000\"\nmax_connections = {max_connections}"),
    )
}

fn with_header_timeout(config: String, timeout_ms: u64) -> String {
    config.replace(
        "listen = \"127.0.0.1:30000\"",
        &format!("listen = \"127.0.0.1:30000\"\nheader_read_timeout_ms = {timeout_ms}"),
    )
}

fn load_bytes(contents: &[u8]) -> Result<Config, ConfigError> {
    let directory = TestDir::new();
    Config::load(&directory.write(contents))
}

#[test]
fn omitted_server_limits_use_bounded_defaults() {
    let config = load_bytes(valid_config("127.0.0.1:30000", 30_000, "info").as_bytes())
        .expect("complete strict configuration should be valid");
    assert_eq!(config.server.listen.to_string(), "127.0.0.1:30000");
    assert_eq!(config.server.max_connections, 1024);
    assert_eq!(config.server.header_read_timeout().as_millis(), 30_000);
    assert_eq!(config.shutdown.drain_timeout().as_millis(), 30_000);
}

#[test]
fn compact_logging_format_selects_compact_output() {
    let config = valid_config("127.0.0.1:30000", 30_000, "info")
        .replace("format = \"json\"", "format = \"compact\"");
    let config = load_bytes(config.as_bytes()).expect("compact logging format should be valid");
    assert_eq!(config.logging.format, LogFormat::Compact);
}

#[test]
fn validates_connection_cap_boundaries() {
    for max_connections in [1, 65_536, tokio::sync::Semaphore::MAX_PERMITS] {
        let config = with_max_connections(
            valid_config("127.0.0.1:30000", 30_000, "info"),
            max_connections,
        );
        assert!(load_bytes(config.as_bytes()).is_ok());
    }

    for max_connections in [0, tokio::sync::Semaphore::MAX_PERMITS + 1] {
        let config = with_max_connections(
            valid_config("127.0.0.1:30000", 30_000, "info"),
            max_connections,
        );
        assert!(load_bytes(config.as_bytes()).is_err());
    }
}

#[test]
fn rejects_unknown_duplicate_missing_and_unsupported_schema_fields() {
    let cases = [
        valid_config("127.0.0.1:30000", 30_000, "info").replace(
            "listen = \"127.0.0.1:30000\"",
            "listen = \"127.0.0.1:30000\"\nunknown = true",
        ),
        valid_config("127.0.0.1:30000", 30_000, "info").replace(
            "filter = \"info\"",
            "filter = \"info\"\nsecret = \"must-not-appear\"",
        ),
        valid_config("127.0.0.1:30000", 30_000, "info")
            + "\n[server]\nlisten = \"127.0.0.1:30001\"\n",
        "schema_version = 1\n".to_owned(),
        valid_config("127.0.0.1:30000", 30_000, "info")
            .replace("schema_version = 1", "schema_version = 2"),
    ];

    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn rejects_invalid_address_timeout_format_and_filter() {
    let cases = [
        valid_config("localhost:30000", 30_000, "info"),
        valid_config("127.0.0.1:30000", 0, "info"),
        valid_config("127.0.0.1:30000", 30_000, "[invalid"),
        valid_config("127.0.0.1:30000", 30_000, "info")
            .replace("format = \"json\"", "format = \"yaml\""),
        with_header_timeout(valid_config("127.0.0.1:30000", 30_000, "info"), 0),
    ];

    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn accepts_large_config_long_filter_and_long_drain_timeout() {
    let filter = (0..40)
        .map(|index| format!("target_{index}=debug"))
        .collect::<Vec<_>>()
        .join(",");
    assert!(filter.len() > 256);
    let mut config = valid_config("127.0.0.1:30000", 86_400_000, &filter);
    config.push_str(&format!("\n# {}\n", "padding".repeat(10_000)));
    assert!(config.len() > 64 * 1024);
    assert!(load_bytes(config.as_bytes()).is_ok());
}

#[test]
fn non_utf8_input_reports_path_without_echoing_contents() {
    let invalid_utf8 = [0xff, b's', b'e', b'c', b'r', b'e', b't'];
    let encoding_error = load_bytes(&invalid_utf8).expect_err("non-UTF-8 config must fail");
    assert!(matches!(encoding_error, ConfigError::Encoding { .. }));
    assert!(encoding_error.to_string().contains("router.toml"));
    assert!(!encoding_error.to_string().contains("secret"));
}

#[test]
fn read_errors_include_path_and_operating_system_cause() {
    let path = Path::new("/definitely-not-present/secret-router.toml");
    let error = Config::load(path).expect_err("missing config must fail");
    let message = error.to_string();
    assert!(message.contains(path.to_str().expect("test path is UTF-8")));
    match error {
        ConfigError::Read {
            path: error_path,
            source,
        } => {
            assert_eq!(error_path, path);
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("expected read error, got {other:?}"),
    }
}

#[test]
fn parse_errors_include_path_and_cause_without_echoing_contents() {
    let contents = b"schema_version = 1\nsecret_value = \"do-not-log\"\n";
    let error = load_bytes(contents).expect_err("unknown field must fail");
    let message = error.to_string();
    assert!(message.contains("router.toml"));
    assert!(message.contains("unknown field"));
    assert!(!message.contains("do-not-log"));
}
