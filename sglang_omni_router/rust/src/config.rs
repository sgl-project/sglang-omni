use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::error::ConfigError;

const DEFAULT_MAX_CONNECTIONS: usize = 1024;
const DEFAULT_HEADER_READ_TIMEOUT_MS: u64 = 30_000;
const SCHEMA_VERSION: u32 = 1;

/// Fully parsed and validated process configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    schema_version: u32,
    /// Listener configuration for router-local endpoints.
    pub server: ServerConfig,
    /// Graceful-shutdown limits.
    pub shutdown: ShutdownConfig,
    /// Structured diagnostic output configuration.
    pub logging: LoggingConfig,
}

/// Listener configuration for router-local endpoints.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address on which the router-local HTTP service listens.
    pub listen: SocketAddr,
    /// Maximum number of accepted client sockets.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    /// Deadline for receiving each initial or keep-alive HTTP/1 request head.
    #[serde(default = "default_header_read_timeout_ms")]
    header_read_timeout_ms: u64,
}

impl ServerConfig {
    /// Time allowed to receive one complete HTTP/1 request head.
    pub fn header_read_timeout(&self) -> Duration {
        Duration::from_millis(self.header_read_timeout_ms)
    }
}

/// Graceful-shutdown limits.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShutdownConfig {
    drain_timeout_ms: u64,
}

impl ShutdownConfig {
    /// Monotonic duration available for graceful server drain.
    pub fn drain_timeout(&self) -> Duration {
        Duration::from_millis(self.drain_timeout_ms)
    }
}

/// Structured diagnostic output configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Output encoding for structured diagnostics.
    pub format: LogFormat,
    /// Tracing filter expression. This value comes only from the config file.
    pub filter: String,
}

/// Supported diagnostic output encodings.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// One JSON object per event.
    Json,
    /// Compact human-readable events.
    Compact,
}

impl Config {
    /// Reads and validates one TOML file.
    ///
    /// Errors identify safe schema fields but never include file contents.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let bytes = fs::read(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|source| ConfigError::Encoding {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Self =
            toml::from_str(text).map_err(|source: toml::de::Error| ConfigError::Parse {
                path: path.to_path_buf(),
                message: source.message().to_owned(),
            })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ConfigError::InvalidField {
                field: "schema_version",
                reason: "unsupported version",
            });
        }
        if self.server.max_connections == 0
            || self.server.max_connections > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err(ConfigError::InvalidField {
                field: "server.max_connections",
                reason: "must fit the listener semaphore and be greater than zero",
            });
        }
        if self.server.header_read_timeout_ms == 0 {
            return Err(ConfigError::InvalidField {
                field: "server.header_read_timeout_ms",
                reason: "must be greater than zero",
            });
        }
        if tokio::time::Instant::now()
            .checked_add(self.server.header_read_timeout())
            .is_none()
        {
            return Err(ConfigError::InvalidField {
                field: "server.header_read_timeout_ms",
                reason: "cannot be represented by the monotonic clock",
            });
        }
        if self.shutdown.drain_timeout_ms == 0 {
            return Err(ConfigError::InvalidField {
                field: "shutdown.drain_timeout_ms",
                reason: "must be greater than zero",
            });
        }
        if tokio::time::Instant::now()
            .checked_add(self.shutdown.drain_timeout())
            .is_none()
        {
            return Err(ConfigError::InvalidField {
                field: "shutdown.drain_timeout_ms",
                reason: "cannot be represented by the monotonic clock",
            });
        }
        if self.logging.filter.is_empty() {
            return Err(ConfigError::InvalidField {
                field: "logging.filter",
                reason: "must not be empty",
            });
        }
        tracing_subscriber::EnvFilter::try_new(self.logging.filter.as_str()).map_err(|_| {
            ConfigError::InvalidField {
                field: "logging.filter",
                reason: "invalid filter expression",
            }
        })?;
        Ok(())
    }
}

const fn default_max_connections() -> usize {
    DEFAULT_MAX_CONNECTIONS
}

const fn default_header_read_timeout_ms() -> u64 {
    DEFAULT_HEADER_READ_TIMEOUT_MS
}
