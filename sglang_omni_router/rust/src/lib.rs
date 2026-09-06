//! Standalone SGLang-Omni Rust router.
//!
//! This crate owns strict startup configuration, a static worker pool, bounded
//! routing and health, byte-preserving chat and media HTTP relays, route-aware
//! readiness, and joined process shutdown.

mod config;
mod error;
mod http_generation;
mod http_media;
mod http_relay;
mod lifecycle;
mod request_id;
mod server;
mod shutdown;
mod speech_facts;
mod worker_pool;

use std::path::Path;

pub use config::{Config, LogFormat};
pub use error::{ConfigError, RouterError};

/// Loads a validated configuration and runs the service to a clean shutdown.
///
/// Configuration loading and tracing initialization occur before the Tokio
/// runtime is created. Runtime work owns one server task and joins it on every
/// shutdown path.
pub fn run(config_path: &Path) -> Result<(), RouterError> {
    let config = Config::load(config_path)?;
    init_tracing(&config)?;
    prepare_file_limit();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .thread_name("sgl-omni-router")
        .enable_io()
        .enable_time()
        .build()
        .map_err(RouterError::RuntimeBuild)?;
    runtime.block_on(server::serve(config))?;
    Ok(())
}

#[cfg(unix)]
fn prepare_file_limit() {
    const TARGET_NOFILE: u64 = 65_535;

    match rlimit::increase_nofile_limit(TARGET_NOFILE) {
        Ok(soft_limit) if soft_limit < TARGET_NOFILE => {
            tracing::warn!(
                soft_limit,
                target = TARGET_NOFILE,
                "RLIMIT_NOFILE remains below the recommended target; raise the process limit to support the configured concurrency"
            );
        }
        Ok(_soft_limit) => {}
        Err(error) => {
            tracing::warn!(
                %error,
                target = TARGET_NOFILE,
                "failed to raise RLIMIT_NOFILE; raise the process limit to support the configured concurrency"
            );
        }
    }
}

#[cfg(not(unix))]
fn prepare_file_limit() {}

fn init_tracing(config: &Config) -> Result<(), RouterError> {
    use tracing_subscriber::prelude::*;

    let filter = tracing_subscriber::EnvFilter::try_new(config.logging.filter.as_str())
        .map_err(|source| RouterError::LoggingFilter { source })?;
    let registry = tracing_subscriber::registry().with(filter);

    match config.logging.format {
        LogFormat::Json => registry
            .with(tracing_subscriber::fmt::layer().json())
            .try_init()
            .map_err(|source| RouterError::TracingInit { source }),
        LogFormat::Compact => registry
            .with(tracing_subscriber::fmt::layer().compact())
            .try_init()
            .map_err(|source| RouterError::TracingInit { source }),
    }
}
