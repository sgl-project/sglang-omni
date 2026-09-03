use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::error::ConfigError;
use crate::worker_pool::profile::{
    WorkerConfig, generation_cohort_is_homogeneous, validate_identifier, validate_workers,
};

const DEFAULT_STREAMED_REQUEST_MAX_BYTES: u64 = 536_870_912;
const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 1_800_000;
const DEFAULT_POOL_IDLE_TIMEOUT_MS: u64 = 90_000;
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
const DEFAULT_MAX_CONNECTIONS: usize = 1024;
const DEFAULT_HEADER_READ_TIMEOUT_MS: u64 = 30_000;
const SCHEMA_VERSION: u32 = 1;
const MAX_GLOBAL_ADMISSION: u32 = 1_000_000;
const MAX_CLASS_ADMISSION: u32 = 65_535;

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
    pub(crate) router: RouterConfig,
    pub(crate) admission: AdmissionConfig,
    pub(crate) health: HealthConfig,
    pub(crate) http_generation: HttpGenerationConfig,
    pub(crate) workers: Vec<WorkerConfig>,
}

/// Bounded transport and buffering policy for chat generation HTTP.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpGenerationConfig {
    pub(crate) trust_domain: String,
    pub(crate) streamed_request_max_bytes: u64,
    connect_timeout_ms: u64,
    request_timeout_ms: u64,
    pool_idle_timeout_ms: u64,
    pub(crate) pool_max_idle_per_host: usize,
}

impl Default for HttpGenerationConfig {
    fn default() -> Self {
        Self {
            trust_domain: String::from("local"),
            streamed_request_max_bytes: DEFAULT_STREAMED_REQUEST_MAX_BYTES,
            connect_timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            pool_idle_timeout_ms: DEFAULT_POOL_IDLE_TIMEOUT_MS,
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
        }
    }
}

impl HttpGenerationConfig {
    pub(crate) const fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    pub(crate) const fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub(crate) const fn pool_idle_timeout(&self) -> Duration {
        Duration::from_millis(self.pool_idle_timeout_ms)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RouterConfig {
    #[serde(default)]
    pub(crate) strategy: RoutingStrategy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RoutingStrategy {
    #[default]
    RoundRobin,
    LeastRequests,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdmissionConfig {
    pub(crate) global: u32,
    pub(crate) generation_http: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HealthConfig {
    interval_ms: u64,
    timeout_ms: u64,
    success_threshold: u8,
    failure_threshold: u8,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            interval_ms: 5_000,
            timeout_ms: 5_000,
            success_threshold: 2,
            failure_threshold: 3,
        }
    }
}

impl HealthConfig {
    pub(crate) fn interval(&self) -> Duration {
        Duration::from_millis(self.interval_ms)
    }

    pub(crate) fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    pub(crate) fn success_threshold(&self) -> u8 {
        self.success_threshold
    }

    pub(crate) fn failure_threshold(&self) -> u8 {
        self.failure_threshold
    }
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
        self.validate_admission()?;
        self.validate_health()?;
        validate_workers(&self.workers)?;
        self.validate_http_generation()?;
        Ok(())
    }

    fn validate_http_generation(&self) -> Result<(), ConfigError> {
        let generation = &self.http_generation;
        validate_identifier(&generation.trust_domain, "http_generation.trust_domain")?;
        if !(1..=4_294_967_296).contains(&generation.streamed_request_max_bytes) {
            return Err(ConfigError::invalid(
                "http_generation.streamed_request_max_bytes",
                "must be between 1 and 4294967296",
            ));
        }
        if !(1..=60_000).contains(&generation.connect_timeout_ms) {
            return Err(ConfigError::invalid(
                "http_generation.connect_timeout_ms",
                "must be between 1 and 60000",
            ));
        }
        if generation.request_timeout_ms < generation.connect_timeout_ms
            || generation.request_timeout_ms > 3_600_000
        {
            return Err(ConfigError::invalid(
                "http_generation.request_timeout_ms",
                "must be at least connect_timeout_ms and at most 3600000",
            ));
        }
        if !(1_000..=300_000).contains(&generation.pool_idle_timeout_ms) {
            return Err(ConfigError::invalid(
                "http_generation.pool_idle_timeout_ms",
                "must be between 1000 and 300000",
            ));
        }
        if !(1..=1_024).contains(&generation.pool_max_idle_per_host) {
            return Err(ConfigError::invalid(
                "http_generation.pool_max_idle_per_host",
                "must be between 1 and 1024",
            ));
        }
        let members = self
            .workers
            .iter()
            .filter(|worker| worker.trust_domain == generation.trust_domain)
            .map(|worker| {
                (
                    worker.default_model_id.as_str(),
                    worker.service_profiles.as_slice(),
                )
            });
        if !generation_cohort_is_homogeneous(members) {
            return Err(ConfigError::invalid(
                "http_generation.trust_domain",
                "must identify a homogeneous generation worker cohort",
            ));
        }
        Ok(())
    }

    fn validate_admission(&self) -> Result<(), ConfigError> {
        if !(1..=MAX_GLOBAL_ADMISSION).contains(&self.admission.global) {
            return Err(ConfigError::invalid(
                "admission.global",
                "must be between 1 and 1000000",
            ));
        }
        if !(1..=MAX_CLASS_ADMISSION).contains(&self.admission.generation_http)
            || self.admission.generation_http > self.admission.global
        {
            return Err(ConfigError::invalid(
                "admission",
                "class limits must be between 1 and 65535 and not exceed global",
            ));
        }
        Ok(())
    }

    fn validate_health(&self) -> Result<(), ConfigError> {
        if !(100..=300_000).contains(&self.health.interval_ms) {
            return Err(ConfigError::invalid(
                "health.interval_ms",
                "must be between 100 and 300000",
            ));
        }
        if self.health.timeout_ms < 10 || self.health.timeout_ms > self.health.interval_ms {
            return Err(ConfigError::invalid(
                "health.timeout_ms",
                "must be between 10 and interval_ms",
            ));
        }
        if !(1..=32).contains(&self.health.success_threshold)
            || !(1..=32).contains(&self.health.failure_threshold)
        {
            return Err(ConfigError::invalid(
                "health",
                "thresholds must be between 1 and 32",
            ));
        }
        Ok(())
    }
}

const fn default_max_connections() -> usize {
    DEFAULT_MAX_CONNECTIONS
}
const fn default_header_read_timeout_ms() -> u64 {
    DEFAULT_HEADER_READ_TIMEOUT_MS
}
