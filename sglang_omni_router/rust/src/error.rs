use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Strict configuration loading and validation failures.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The configuration file could not be opened or read.
    #[error("failed to read configuration {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// The configuration was not UTF-8.
    #[error("configuration {path} must be UTF-8: {source}")]
    Encoding {
        path: PathBuf,
        #[source]
        source: std::str::Utf8Error,
    },
    /// TOML syntax, duplicate fields, or unknown fields were invalid.
    #[error("failed to parse configuration {path}: {message}")]
    Parse { path: PathBuf, message: String },
    /// A parsed field violated a bounded semantic rule.
    #[error("invalid configuration field {field}: {reason}")]
    InvalidField {
        /// Stable schema field name.
        field: &'static str,
        /// Stable, non-sensitive validation reason.
        reason: &'static str,
    },
}

/// Top-level startup, runtime, and shutdown failures.
#[derive(Debug, Error)]
pub enum RouterError {
    /// Configuration failed before process infrastructure was created.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The validated log filter could not be reconstructed.
    #[error("failed to construct the configured logging filter: {source}")]
    LoggingFilter {
        /// Internal parser source, available to structured diagnostics.
        #[source]
        source: tracing_subscriber::filter::ParseError,
    },
    /// The process-global tracing subscriber was already initialized or failed.
    #[error("failed to initialize structured diagnostics: {source}")]
    TracingInit {
        /// Internal initialization source.
        #[source]
        source: tracing_subscriber::util::TryInitError,
    },
    /// The bounded Tokio runtime could not be created.
    #[error("failed to initialize the async runtime: {0}")]
    RuntimeBuild(#[source] io::Error),
    /// The configured listener could not be bound.
    #[error("failed to bind the configured listener: {0}")]
    Bind(#[source] io::Error),
    /// The process file-descriptor limit could not be raised or inspected.
    #[cfg(unix)]
    #[error("failed to prepare the process RLIMIT_NOFILE soft limit: {0}")]
    FileLimit(#[source] io::Error),
    /// The listener and configured accepted sockets cannot fit under RLIMIT_NOFILE.
    #[cfg(unix)]
    #[error(
        "server.max_connections ({max_connections}) plus the listener exceeds the RLIMIT_NOFILE soft limit ({soft_limit}); raise the soft limit or lower server.max_connections"
    )]
    InsufficientFileLimit {
        /// Configured accepted-socket ceiling.
        max_connections: usize,
        /// Process soft file-descriptor limit.
        soft_limit: u64,
    },
    /// The HTTP server stopped without a shutdown request.
    #[error("the local HTTP server stopped unexpectedly: {0}")]
    Server(#[source] io::Error),
    /// A server task panicked or otherwise failed to join.
    #[error("the local HTTP server task failed: {0}")]
    ServerTask(#[source] tokio::task::JoinError),
    /// Graceful drain exceeded the configured monotonic deadline.
    #[error("graceful shutdown exceeded its configured deadline")]
    DrainTimeout,
    /// A second signal forced an incomplete graceful drain.
    #[error("a second signal forced shutdown before graceful drain completed")]
    ForcedShutdown,
    /// The lifecycle owner observed an illegal or poisoned transition.
    #[error("the process lifecycle invariant failed")]
    Lifecycle,
    /// The graceful-shutdown notification owner disappeared unexpectedly.
    #[error("failed to notify the local HTTP server to drain")]
    ShutdownNotify,
    /// Signal observation could not be installed or completed.
    #[error("failed to observe process termination signals: {0}")]
    Signal(#[source] io::Error),
}

impl RouterError {
    /// Returns the stable process exit code for this failure.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Config(_) => 2,
            #[cfg(unix)]
            Self::FileLimit(_) | Self::InsufficientFileLimit { .. } => 1,
            Self::LoggingFilter { .. }
            | Self::TracingInit { .. }
            | Self::RuntimeBuild(_)
            | Self::Bind(_)
            | Self::Server(_)
            | Self::ServerTask(_)
            | Self::DrainTimeout
            | Self::ForcedShutdown
            | Self::Lifecycle
            | Self::ShutdownNotify
            | Self::Signal(_) => 1,
        }
    }
}
