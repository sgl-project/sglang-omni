use std::io;
use std::path::PathBuf;

use thiserror::Error;

use axum::body::Body;
use axum::http::header::{ALLOW, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Response, StatusCode};

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
    /// The generation data-plane client failed to build.
    #[error("failed to initialize the generation HTTP client")]
    GenerationClient(#[source] reqwest::Error),
    /// The isolated health client failed to build.
    #[error("failed to initialize the isolated health client")]
    HealthClient(#[source] reqwest::Error),
    /// Validated worker-pool configuration could not be reconstructed.
    #[error("the validated worker-pool invariant failed")]
    WorkerPoolInvariant,
    /// A health worker returned before process cancellation.
    #[error("an owned health task exited unexpectedly")]
    HealthTaskExited,
    /// A health worker panicked or otherwise failed to join.
    #[error("an owned health task failed: {0}")]
    HealthTask(#[source] tokio::task::JoinError),
}

impl RouterError {
    /// Returns the stable process exit code for this failure.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Config(_) => 2,
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
            | Self::Signal(_)
            | Self::GenerationClient(_)
            | Self::HealthClient(_)
            | Self::WorkerPoolInvariant
            | Self::HealthTaskExited
            | Self::HealthTask(_) => 1,
        }
    }
}

impl ConfigError {
    pub(crate) const fn invalid(field: &'static str, reason: &'static str) -> Self {
        Self::InvalidField { field, reason }
    }
}

/// Stable topology-free failures generated before response commitment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HttpFault {
    MalformedRequest,
    MethodNotAllowed,
    RequestTimeout,
    RequestBodyTooLarge,
    UnsupportedMediaType,
    UnsupportedContentEncoding,
    ExpectationFailed,
    NoCompatibleWorker,
    RouterOverloaded,
    InternalError,
    UpstreamProtocolError,
    RouterUnavailable,
    UpstreamTimeout,
    HttpVersionNotSupported,
}

impl HttpFault {
    const fn status(self) -> StatusCode {
        match self {
            Self::MalformedRequest => StatusCode::BAD_REQUEST,
            Self::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
            Self::RequestTimeout => StatusCode::REQUEST_TIMEOUT,
            Self::RequestBodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMediaType | Self::UnsupportedContentEncoding => {
                StatusCode::UNSUPPORTED_MEDIA_TYPE
            }
            Self::ExpectationFailed => StatusCode::EXPECTATION_FAILED,
            Self::NoCompatibleWorker => StatusCode::UNPROCESSABLE_ENTITY,
            Self::RouterOverloaded => StatusCode::TOO_MANY_REQUESTS,
            Self::InternalError => StatusCode::INTERNAL_SERVER_ERROR,
            Self::UpstreamProtocolError => StatusCode::BAD_GATEWAY,
            Self::RouterUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::UpstreamTimeout => StatusCode::GATEWAY_TIMEOUT,
            Self::HttpVersionNotSupported => StatusCode::HTTP_VERSION_NOT_SUPPORTED,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::MalformedRequest => "malformed_request",
            Self::MethodNotAllowed => "method_not_allowed",
            Self::RequestTimeout => "request_timeout",
            Self::RequestBodyTooLarge => "request_body_too_large",
            Self::UnsupportedMediaType => "unsupported_media_type",
            Self::UnsupportedContentEncoding => "unsupported_content_encoding",
            Self::ExpectationFailed => "expectation_failed",
            Self::NoCompatibleWorker => "no_compatible_worker",
            Self::RouterOverloaded => "router_overloaded",
            Self::InternalError => "internal_error",
            Self::UpstreamProtocolError => "upstream_protocol_error",
            Self::RouterUnavailable => "router_unavailable",
            Self::UpstreamTimeout => "upstream_timeout",
            Self::HttpVersionNotSupported => "http_version_not_supported",
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::MalformedRequest => "The request is malformed.",
            Self::MethodNotAllowed => "POST is required for this route.",
            Self::RequestTimeout => "The request body timed out.",
            Self::RequestBodyTooLarge => "The request body is too large.",
            Self::UnsupportedMediaType => "The content type is unsupported.",
            Self::UnsupportedContentEncoding => "The content encoding is unsupported.",
            Self::ExpectationFailed => "Request expectations are unsupported.",
            Self::NoCompatibleWorker => "No compatible worker is configured.",
            Self::RouterOverloaded => "The router is overloaded.",
            Self::InternalError => "The router encountered an internal error.",
            Self::UpstreamProtocolError => "The upstream response is invalid.",
            Self::RouterUnavailable => "The router is unavailable.",
            Self::UpstreamTimeout => "The upstream request timed out.",
            Self::HttpVersionNotSupported => "HTTP/1.1 is required.",
        }
    }

    fn error_type(self) -> &'static str {
        match self.status() {
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            status if status.is_client_error() => "invalid_request_error",
            _ => "server_error",
        }
    }

    pub(crate) fn into_response(self) -> Response<Body> {
        let body = format!(
            "{{\"error\":{{\"message\":\"{}\",\"type\":\"{}\",\"param\":null,\"code\":\"{}\"}}}}",
            self.message(),
            self.error_type(),
            self.code()
        );
        let content_length = body.len();
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = self.status();
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        response
            .headers_mut()
            .insert(CONTENT_LENGTH, HeaderValue::from(content_length));
        response
            .headers_mut()
            .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        if self == Self::MethodNotAllowed {
            response
                .headers_mut()
                .insert(ALLOW, HeaderValue::from_static("POST"));
        }
        response
    }
}
