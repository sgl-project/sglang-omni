use std::time::Duration;

use reqwest::redirect::Policy;
use reqwest::{Client, Url};

use super::profile::WorkerConfig;

/// One immutable worker target derived from a validated URL authority.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedTarget {
    base_url: Url,
    health_url: Url,
}

impl ResolvedTarget {
    pub(super) fn from_worker(worker: &WorkerConfig) -> Option<Self> {
        Self::from_parts(worker.base_url.as_str(), worker.health_path.as_str())
    }

    pub(super) fn from_parts(base_url: &str, health_path: &str) -> Option<Self> {
        let base_url = Url::parse(base_url).ok()?;
        if !matches!(base_url.scheme(), "http" | "https")
            || !base_url.username().is_empty()
            || base_url.password().is_some()
            || base_url.query().is_some()
            || base_url.fragment().is_some()
            || base_url.path() != "/"
            || base_url.port() == Some(0)
        {
            return None;
        }
        let host = base_url.host_str()?;
        if host.ends_with('.') {
            return None;
        }
        base_url.port_or_known_default()?;
        let mut health_url = base_url.clone();
        health_url.set_path(health_path);
        Some(Self {
            base_url,
            health_url,
        })
    }

    pub(crate) fn base_url(&self) -> &Url {
        &self.base_url
    }

    pub(super) fn health_url(&self) -> &Url {
        &self.health_url
    }
}

pub(super) fn build_health_client(
    timeout: Duration,
    interval: Duration,
) -> Result<Client, reqwest::Error> {
    Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .retry(reqwest::retry::never())
        .http1_only()
        .connect_timeout(timeout)
        .timeout(timeout)
        .pool_idle_timeout(Some(interval.saturating_mul(2)))
        .pool_max_idle_per_host(1)
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .build()
}

pub(super) fn build_generation_client(
    connect_timeout: Duration,
    pool_idle_timeout: Duration,
    pool_max_idle_per_host: usize,
) -> Result<Client, reqwest::Error> {
    Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .retry(reqwest::retry::never())
        .http1_only()
        .connect_timeout(connect_timeout)
        .pool_idle_timeout(Some(pool_idle_timeout))
        .pool_max_idle_per_host(pool_max_idle_per_host)
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .build()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::ResolvedTarget;

    #[test]
    fn worker_origins_are_strict_and_dns_names_are_valid() {
        let invalid = [
            "ftp://worker.invalid:8000/",
            "http://user:secret@worker.invalid:8000/",
            "http://worker.invalid:8000/chat",
            "http://worker.invalid:8000/?worker=secret",
            "http://worker.invalid:8000/#secret",
            "http://worker.invalid:0/",
            "http://worker.invalid.:8000/",
        ];
        for base_url in invalid {
            assert!(ResolvedTarget::from_parts(base_url, "/health").is_none());
        }

        let hostname = ResolvedTarget::from_parts("HTTP://WORKER.INVALID:80/", "/health")
            .expect("valid DNS worker target");
        assert_eq!(hostname.base_url().as_str(), "http://worker.invalid/");
        assert_eq!(
            hostname.health_url().as_str(),
            "http://worker.invalid/health"
        );

        assert!(ResolvedTarget::from_parts("https://127.0.0.1/", "/health").is_some());
        assert!(ResolvedTarget::from_parts("http://[::1]:18080/", "/health").is_some());
    }
}
