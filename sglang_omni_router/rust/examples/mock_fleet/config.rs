use std::collections::HashSet;
use std::net::SocketAddr;

use serde::Deserialize;
use serde_json::Value;

pub(super) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Behavior {
    pub first_packet_delay_ms: u64,
    pub chunk_interval_ms: u64,
    pub chunks: usize,
    pub chunk_bytes: usize,
    pub health_status: u16,
    pub response_status: u16,
    pub disconnect_after_chunks: Option<usize>,
    pub max_request_bytes: usize,
}

impl Default for Behavior {
    fn default() -> Self {
        Self {
            first_packet_delay_ms: 0,
            chunk_interval_ms: 0,
            chunks: 4,
            chunk_bytes: 4096,
            health_status: 200,
            response_status: 200,
            disconnect_after_chunks: None,
            max_request_bytes: 8 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Worker {
    pub id: String,
    pub trust_domain: String,
    pub default_model: String,
    pub health_path: String,
    pub profiles: Vec<Value>,
    pub behavior: Behavior,
    pub voice_owner: bool,
}

pub(super) struct Fleet {
    pub document: toml::Value,
    pub workers: Vec<Worker>,
    pub listen: SocketAddr,
}

impl Fleet {
    pub(super) fn parse(contents: &str) -> Result<Self> {
        let document: toml::Value = toml::from_str(contents)?;
        let listen: SocketAddr = document
            .get("server")
            .and_then(|server| server.get("listen"))
            .and_then(toml::Value::as_str)
            .ok_or("server.listen is required")?
            .parse()?;
        if !listen.ip().is_loopback() || listen.port() == 0 {
            return Err("mock fleet requires a loopback router address with a nonzero port".into());
        }
        let owner = document
            .get("router")
            .and_then(|router| router.get("voice_owner_worker_id"))
            .and_then(toml::Value::as_str);
        let rows = document
            .get("workers")
            .and_then(toml::Value::as_array)
            .ok_or("workers must be a nonempty array")?;
        if rows.is_empty() || rows.len() > 64 {
            return Err("mock fleet requires 1..=64 workers".into());
        }
        let mut identities = HashSet::new();
        let mut workers = Vec::new();
        for row in rows {
            let id = row
                .get("worker_id")
                .and_then(toml::Value::as_str)
                .ok_or("worker_id required")?;
            if !identities.insert(id)
                || id.is_empty()
                || !id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
            {
                return Err(
                    "worker IDs must be unique nonempty ASCII alphanumeric/-_. strings".into(),
                );
            }
            let behavior: Behavior = row
                .get("mock")
                .cloned()
                .map(toml::Value::try_into)
                .transpose()?
                .unwrap_or_default();
            if behavior.chunks == 0
                || behavior.chunks > 4096
                || behavior.chunk_bytes == 0
                || behavior.chunk_bytes > 1024 * 1024
                || !behavior.chunk_bytes.is_multiple_of(2)
                || behavior.chunks.saturating_mul(behavior.chunk_bytes) > 64 * 1024 * 1024
                || behavior.max_request_bytes == 0
                || behavior.max_request_bytes > 64 * 1024 * 1024
                || behavior.first_packet_delay_ms > 60_000
                || behavior.chunk_interval_ms > 60_000
                || !(200..=599).contains(&behavior.health_status)
                || !(400..=599).contains(&behavior.response_status)
                    && behavior.response_status != 200
            {
                return Err(format!("invalid or excessive mock behavior limits for {id}").into());
            }
            let profiles: Vec<Value> = serde_json::from_value(serde_json::to_value(
                row.get("service_profiles")
                    .ok_or("service_profiles required")?,
            )?)?;
            if profiles.is_empty() {
                return Err("service_profiles cannot be empty".into());
            }
            for profile in &profiles {
                let service = profile["service"].as_str().ok_or("service required")?;
                if ![
                    "generation_http",
                    "speech_http",
                    "speech_batch",
                    "transcription_http",
                    "speech_websocket",
                    "realtime_websocket",
                ]
                .contains(&service)
                {
                    return Err(format!("unsupported mock service: {service}").into());
                }
                for field in ["response_formats", "chat_audio_formats"] {
                    if let Some(formats) = profile[field].as_array() {
                        for format in formats {
                            if !matches!(
                                format.as_str(),
                                Some("pcm" | "wav" | "json" | "verbose_json" | "text" | "sse")
                            ) {
                                return Err(format!("mock does not encode {format}; use PCM/WAV or supported transcription formats").into());
                            }
                        }
                    }
                }
            }
            workers.push(Worker {
                id: id.to_owned(),
                trust_domain: row
                    .get("trust_domain")
                    .and_then(toml::Value::as_str)
                    .ok_or("trust_domain required")?
                    .to_owned(),
                default_model: row
                    .get("default_model_id")
                    .and_then(toml::Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                health_path: row
                    .get("health_path")
                    .and_then(toml::Value::as_str)
                    .unwrap_or("/health")
                    .to_owned(),
                profiles,
                behavior,
                voice_owner: owner == Some(id),
            });
        }
        Ok(Self {
            document,
            workers,
            listen,
        })
    }

    pub(super) fn router_config(&self, urls: &[String]) -> Result<String> {
        if urls.len() != self.workers.len() {
            return Err("worker endpoint count does not match configuration".into());
        }
        let mut document = self.document.clone();
        let rows = document
            .get_mut("workers")
            .and_then(toml::Value::as_array_mut)
            .ok_or("workers missing")?;
        for (row, url) in rows.iter_mut().zip(urls) {
            let table = row.as_table_mut().ok_or("worker must be a table")?;
            table.remove("mock");
            table.insert("base_url".to_owned(), toml::Value::String(url.clone()));
        }
        Ok(toml::to_string_pretty(&document)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
schema_version = 1
[server]
listen = "127.0.0.1:30000"
[[workers]]
worker_id = "text-1"
base_url = "http://127.0.0.1:8000/"
trust_domain = "local"
default_model_id = "text"
[workers.mock]
first_packet_delay_ms = 5
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["text"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["typed_parts"]
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming", "streaming"]
"#;

    #[test]
    fn generated_config_preserves_profiles_but_removes_mock_settings() -> Result<()> {
        let fleet = Fleet::parse(CONFIG)?;
        assert_eq!(fleet.workers[0].behavior.first_packet_delay_ms, 5);
        let generated = fleet.router_config(&["http://127.0.0.1:43210/".to_owned()])?;
        let parsed: toml::Value = toml::from_str(&generated)?;
        assert!(parsed["workers"][0].get("mock").is_none());
        assert_eq!(
            parsed["workers"][0]["base_url"].as_str(),
            Some("http://127.0.0.1:43210/")
        );
        assert_eq!(
            parsed["workers"][0]["service_profiles"],
            fleet.document["workers"][0]["service_profiles"]
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_behavior_and_non_loopback_listener() {
        assert!(
            Fleet::parse(&CONFIG.replace("first_packet_delay_ms", "misspelled_delay")).is_err()
        );
        assert!(Fleet::parse(&CONFIG.replace("127.0.0.1:30000", "0.0.0.0:30000")).is_err());
    }

    #[test]
    fn rejects_duplicate_worker_ids_and_invalid_behavior_limits() {
        let workers = CONFIG.split("[[workers]]").nth(1).unwrap_or_default();
        assert!(Fleet::parse(&format!("{CONFIG}\n[[workers]]{workers}")).is_err());
        assert!(
            Fleet::parse(&CONFIG.replace("first_packet_delay_ms = 5", "chunk_bytes = 0")).is_err()
        );
    }
}
