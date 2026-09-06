use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::error::ConfigError;
use crate::worker_pool::profile::{
    ServiceProfile, WorkerConfig, validate_identifier, validate_workers,
};

const DEFAULT_BUFFERED_REQUEST_MAX_BYTES: u64 = 8_388_608;
const DEFAULT_BUFFERED_REQUEST_TOTAL_BYTES: u64 = 268_435_456;
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
const DEFAULT_WS_CONNECT_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_WS_WORKER_SETUP_TIMEOUT_MS: u64 = 60_000;
const DEFAULT_WS_SPEECH_CONFIG_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_WS_CLOSE_TIMEOUT_MS: u64 = 5_000;

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
    #[serde(default)]
    pub(crate) http: HttpConfig,
    pub(crate) http_generation: Option<HttpGenerationConfig>,
    pub(crate) http_media: Option<HttpMediaConfig>,
    pub(crate) websocket: Option<WebsocketConfig>,
    pub(crate) workers: Vec<WorkerConfig>,
}

/// Bounded transport policy shared by the terminating WebSocket routes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct WebsocketConfig {
    pub(crate) speech: Option<WebsocketRouteConfig>,
    pub(crate) realtime: Option<WebsocketRouteConfig>,
    connect_timeout_ms: u64,
    worker_setup_timeout_ms: u64,
    speech_config_timeout_ms: u64,
    close_timeout_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WebsocketRouteConfig {
    pub(crate) trust_domain: String,
}

impl Default for WebsocketConfig {
    fn default() -> Self {
        Self {
            speech: None,
            realtime: None,
            connect_timeout_ms: DEFAULT_WS_CONNECT_TIMEOUT_MS,
            worker_setup_timeout_ms: DEFAULT_WS_WORKER_SETUP_TIMEOUT_MS,
            speech_config_timeout_ms: DEFAULT_WS_SPEECH_CONFIG_TIMEOUT_MS,
            close_timeout_ms: DEFAULT_WS_CLOSE_TIMEOUT_MS,
        }
    }
}

impl WebsocketConfig {
    pub(crate) const fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    pub(crate) const fn worker_setup_timeout(&self) -> Duration {
        Duration::from_millis(self.worker_setup_timeout_ms)
    }

    pub(crate) const fn speech_config_timeout(&self) -> Duration {
        Duration::from_millis(self.speech_config_timeout_ms)
    }

    pub(crate) const fn close_timeout(&self) -> Duration {
        Duration::from_millis(self.close_timeout_ms)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpConfig {
    pub(crate) buffered_request_total_bytes: u64,
    connect_timeout_ms: u64,
    pool_idle_timeout_ms: u64,
    pub(crate) pool_max_idle_per_host: usize,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            buffered_request_total_bytes: DEFAULT_BUFFERED_REQUEST_TOTAL_BYTES,
            connect_timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
            pool_idle_timeout_ms: DEFAULT_POOL_IDLE_TIMEOUT_MS,
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
        }
    }
}

impl HttpConfig {
    pub(crate) const fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    pub(crate) const fn pool_idle_timeout(&self) -> Duration {
        Duration::from_millis(self.pool_idle_timeout_ms)
    }

    pub(crate) fn buffered_total_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.buffered_request_total_bytes).map_err(|_| {
            ConfigError::invalid(
                "http.buffered_request_total_bytes",
                "cannot be represented on this platform",
            )
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpMediaConfig {
    pub(crate) routes: Vec<HttpMediaRoute>,
    pub(crate) trust_domain: String,
    pub(crate) buffered_request_max_bytes: u64,
    pub(crate) streamed_request_max_bytes: u64,
    request_timeout_ms: u64,
}

impl Default for HttpMediaConfig {
    fn default() -> Self {
        Self {
            routes: Vec::new(),
            trust_domain: String::from("local"),
            buffered_request_max_bytes: DEFAULT_BUFFERED_REQUEST_MAX_BYTES,
            streamed_request_max_bytes: DEFAULT_STREAMED_REQUEST_MAX_BYTES,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
        }
    }
}

impl HttpMediaConfig {
    pub(crate) const fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub(crate) fn buffered_max_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.buffered_request_max_bytes).map_err(|_| {
            ConfigError::invalid(
                "http_media.buffered_request_max_bytes",
                "cannot be represented on this platform",
            )
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HttpMediaRoute {
    Speech,
    SpeechBatch,
    Transcription,
    Translation,
}

impl HttpMediaRoute {
    pub(crate) const fn service_class(self) -> crate::worker_pool::profile::ServiceClass {
        use crate::worker_pool::profile::ServiceClass;
        match self {
            Self::Speech => ServiceClass::SpeechHttp,
            Self::SpeechBatch => ServiceClass::SpeechBatch,
            Self::Transcription | Self::Translation => ServiceClass::TranscriptionHttp,
        }
    }

    pub(crate) const fn speech_to_text_task(
        self,
    ) -> Option<crate::worker_pool::profile::SpeechToTextTask> {
        use crate::worker_pool::profile::SpeechToTextTask;
        match self {
            Self::Transcription => Some(SpeechToTextTask::Transcribe),
            Self::Translation => Some(SpeechToTextTask::Translate),
            Self::Speech | Self::SpeechBatch => None,
        }
    }

    pub(crate) fn matches_profile(
        self,
        profile: &crate::worker_pool::profile::ServiceProfile,
    ) -> bool {
        use crate::worker_pool::profile::ServiceProfile;
        match (self, profile) {
            (Self::Speech, ServiceProfile::SpeechHttp { .. })
            | (Self::SpeechBatch, ServiceProfile::SpeechBatch { .. }) => true,
            (
                Self::Transcription | Self::Translation,
                ServiceProfile::TranscriptionHttp { task, .. },
            ) => Some(*task) == self.speech_to_text_task(),
            _ => false,
        }
    }
}

/// Bounded transport and buffering policy for chat generation HTTP.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpGenerationConfig {
    pub(crate) trust_domain: String,
    pub(crate) buffered_request_max_bytes: u64,
    pub(crate) streamed_request_max_bytes: u64,
    request_timeout_ms: u64,
}

impl Default for HttpGenerationConfig {
    fn default() -> Self {
        Self {
            trust_domain: String::from("local"),
            buffered_request_max_bytes: DEFAULT_BUFFERED_REQUEST_MAX_BYTES,
            streamed_request_max_bytes: DEFAULT_STREAMED_REQUEST_MAX_BYTES,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
        }
    }
}

impl HttpGenerationConfig {
    pub(crate) const fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub(crate) fn buffered_max_usize(&self) -> Result<usize, ConfigError> {
        usize::try_from(self.buffered_request_max_bytes).map_err(|_| {
            ConfigError::invalid(
                "http_generation.buffered_request_max_bytes",
                "cannot be represented on this platform",
            )
        })
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
    pub(crate) generation_http: Option<u32>,
    pub(crate) speech_http: Option<u32>,
    pub(crate) speech_batch: Option<u32>,
    pub(crate) transcription_http: Option<u32>,
    pub(crate) speech_websocket: Option<u32>,
    pub(crate) realtime_websocket: Option<u32>,
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
        if self.http_generation.is_none() && self.http_media.is_none() && self.websocket.is_none() {
            return Err(ConfigError::invalid(
                "routes",
                "must configure at least one HTTP or WebSocket route",
            ));
        }
        self.validate_http()?;
        self.validate_http_generation()?;
        self.validate_http_media()?;
        self.validate_websocket()?;
        Ok(())
    }

    fn validate_http(&self) -> Result<(), ConfigError> {
        let largest_buffered_request = self
            .http_generation
            .as_ref()
            .map(|config| config.buffered_request_max_bytes)
            .into_iter()
            .chain(
                self.http_media
                    .as_ref()
                    .map(|config| config.buffered_request_max_bytes),
            )
            .max()
            .unwrap_or(0);
        if self.http.buffered_request_total_bytes < largest_buffered_request
            || self.http.buffered_request_total_bytes > 2_147_483_647
        {
            return Err(ConfigError::invalid(
                "http.buffered_request_total_bytes",
                "must cover every per-request buffer and be at most 2147483647",
            ));
        }
        let buffered_total = self.http.buffered_total_usize()?;
        if buffered_total > tokio::sync::Semaphore::MAX_PERMITS {
            return Err(ConfigError::invalid(
                "http.buffered_request_total_bytes",
                "exceeds the platform semaphore permit limit",
            ));
        }
        if !(1..=60_000).contains(&self.http.connect_timeout_ms) {
            return Err(ConfigError::invalid(
                "http.connect_timeout_ms",
                "must be between 1 and 60000",
            ));
        }
        if !(1_000..=300_000).contains(&self.http.pool_idle_timeout_ms) {
            return Err(ConfigError::invalid(
                "http.pool_idle_timeout_ms",
                "must be between 1000 and 300000",
            ));
        }
        if !(1..=1_024).contains(&self.http.pool_max_idle_per_host) {
            return Err(ConfigError::invalid(
                "http.pool_max_idle_per_host",
                "must be between 1 and 1024",
            ));
        }
        Ok(())
    }

    fn validate_websocket(&self) -> Result<(), ConfigError> {
        let Some(websocket) = self.websocket.as_ref() else {
            return Ok(());
        };
        if websocket.speech.is_none() && websocket.realtime.is_none() {
            return Err(ConfigError::invalid(
                "websocket",
                "must enable speech or realtime",
            ));
        }
        for (field, value) in [
            ("websocket.connect_timeout_ms", websocket.connect_timeout_ms),
            (
                "websocket.speech_config_timeout_ms",
                websocket.speech_config_timeout_ms,
            ),
            ("websocket.close_timeout_ms", websocket.close_timeout_ms),
        ] {
            if !(1..=60_000).contains(&value) {
                return Err(ConfigError::invalid(field, "must be between 1 and 60000"));
            }
        }
        if !(1..=3_600_000).contains(&websocket.worker_setup_timeout_ms) {
            return Err(ConfigError::invalid(
                "websocket.worker_setup_timeout_ms",
                "must be between 1 and 3600000",
            ));
        }
        if let Some(route) = websocket.speech.as_ref() {
            self.validate_websocket_route(
                route,
                crate::worker_pool::profile::ServiceClass::SpeechWebsocket,
                self.admission.speech_websocket,
                "websocket.speech",
            )?;
        }
        if let Some(route) = websocket.realtime.as_ref() {
            self.validate_websocket_route(
                route,
                crate::worker_pool::profile::ServiceClass::RealtimeWebsocket,
                self.admission.realtime_websocket,
                "websocket.realtime",
            )?;
        }
        Ok(())
    }

    fn validate_websocket_route(
        &self,
        route: &WebsocketRouteConfig,
        service: crate::worker_pool::profile::ServiceClass,
        admission: Option<u32>,
        field: &'static str,
    ) -> Result<(), ConfigError> {
        validate_identifier(&route.trust_domain, field)?;
        if admission.is_none() {
            return Err(ConfigError::invalid(
                "admission",
                "every enabled WebSocket route requires its class limit",
            ));
        }
        if !self.workers.iter().any(|worker| {
            worker.trust_domain == route.trust_domain
                && worker
                    .service_profiles
                    .iter()
                    .any(|profile| profile.service_class() == service)
        }) {
            return Err(ConfigError::invalid(
                field,
                "trust domain has no compatible configured worker",
            ));
        }
        Ok(())
    }

    fn validate_http_media(&self) -> Result<(), ConfigError> {
        let Some(media) = self.http_media.as_ref() else {
            return Ok(());
        };
        if media.routes.is_empty()
            || media
                .routes
                .iter()
                .enumerate()
                .any(|(index, route)| media.routes[..index].contains(route))
        {
            return Err(ConfigError::invalid(
                "http_media.routes",
                "must contain at least one route without duplicates",
            ));
        }
        validate_identifier(&media.trust_domain, "http_media.trust_domain")?;
        if !(1..=67_108_864).contains(&media.buffered_request_max_bytes) {
            return Err(ConfigError::invalid(
                "http_media.buffered_request_max_bytes",
                "must be between 1 and 67108864",
            ));
        }
        let _buffered_max = media.buffered_max_usize()?;
        if media.streamed_request_max_bytes < media.buffered_request_max_bytes
            || media.streamed_request_max_bytes > 4_294_967_296
        {
            return Err(ConfigError::invalid(
                "http_media.streamed_request_max_bytes",
                "must be at least the buffered limit and at most 4294967296",
            ));
        }
        if media.request_timeout_ms < self.http.connect_timeout_ms
            || media.request_timeout_ms > 3_600_000
        {
            return Err(ConfigError::invalid(
                "http_media.request_timeout_ms",
                "must be at least http.connect_timeout_ms and at most 3600000",
            ));
        }
        for route in &media.routes {
            let class_limit = match route {
                HttpMediaRoute::Speech => self.admission.speech_http,
                HttpMediaRoute::SpeechBatch => self.admission.speech_batch,
                HttpMediaRoute::Transcription | HttpMediaRoute::Translation => {
                    self.admission.transcription_http
                }
            };
            if class_limit.is_none() {
                return Err(ConfigError::invalid(
                    "admission",
                    "every enabled media route requires its class limit",
                ));
            }
            let available = self.workers.iter().any(|worker| {
                worker.trust_domain == media.trust_domain
                    && worker.service_profiles.iter().any(|profile| match route {
                        HttpMediaRoute::Speech => matches!(
                            profile,
                            crate::worker_pool::profile::ServiceProfile::SpeechHttp { .. }
                        ),
                        HttpMediaRoute::SpeechBatch => matches!(
                            profile,
                            crate::worker_pool::profile::ServiceProfile::SpeechBatch { .. }
                        ),
                        HttpMediaRoute::Transcription => matches!(
                            profile,
                            crate::worker_pool::profile::ServiceProfile::TranscriptionHttp {
                                task: crate::worker_pool::profile::SpeechToTextTask::Transcribe,
                                ..
                            }
                        ),
                        HttpMediaRoute::Translation => matches!(
                            profile,
                            crate::worker_pool::profile::ServiceProfile::TranscriptionHttp {
                                task: crate::worker_pool::profile::SpeechToTextTask::Translate,
                                ..
                            }
                        ),
                    })
            });
            if !available {
                return Err(ConfigError::invalid(
                    "http_media.routes",
                    "every enabled route requires a matching worker profile",
                ));
            }
        }
        Ok(())
    }

    fn validate_http_generation(&self) -> Result<(), ConfigError> {
        let Some(generation) = self.http_generation.as_ref() else {
            return Ok(());
        };
        if self.admission.generation_http.is_none() {
            return Err(ConfigError::invalid(
                "admission.generation_http",
                "is required while chat generation is enabled",
            ));
        }
        validate_identifier(&generation.trust_domain, "http_generation.trust_domain")?;
        if !(1..=67_108_864).contains(&generation.buffered_request_max_bytes) {
            return Err(ConfigError::invalid(
                "http_generation.buffered_request_max_bytes",
                "must be between 1 and 67108864",
            ));
        }
        let _buffered_max = generation.buffered_max_usize()?;
        if generation.streamed_request_max_bytes < generation.buffered_request_max_bytes
            || generation.streamed_request_max_bytes > 4_294_967_296
        {
            return Err(ConfigError::invalid(
                "http_generation.streamed_request_max_bytes",
                "must be at least the buffered limit and at most 4294967296",
            ));
        }
        if generation.request_timeout_ms < self.http.connect_timeout_ms
            || generation.request_timeout_ms > 3_600_000
        {
            return Err(ConfigError::invalid(
                "http_generation.request_timeout_ms",
                "must be at least http.connect_timeout_ms and at most 3600000",
            ));
        }
        if !self.workers.iter().any(|worker| {
            worker.trust_domain == generation.trust_domain
                && worker
                    .service_profiles
                    .iter()
                    .any(|profile| matches!(profile, ServiceProfile::GenerationHttp { .. }))
        }) {
            return Err(ConfigError::invalid(
                "http_generation.trust_domain",
                "must contain at least one generation worker",
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
        for limit in [
            self.admission.generation_http,
            self.admission.speech_http,
            self.admission.speech_batch,
            self.admission.transcription_http,
            self.admission.speech_websocket,
            self.admission.realtime_websocket,
        ]
        .into_iter()
        .flatten()
        {
            if !(1..=MAX_CLASS_ADMISSION).contains(&limit) {
                return Err(ConfigError::invalid(
                    "admission",
                    "configured class limits must be between 1 and 65535",
                ));
            }
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
