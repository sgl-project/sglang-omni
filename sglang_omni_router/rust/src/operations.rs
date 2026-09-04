use std::collections::BTreeSet;
use std::fmt::Write as _;

use axum::body::Body;
use axum::http::header::{CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Response, StatusCode};
use bytes::Bytes;
use serde::Serialize;

use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::lifecycle::State as LifecycleState;
use crate::worker_pool::{CapacityClass, OperationsSnapshot};

const JSON_CONTENT_TYPE: &str = "application/json";
const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";
const LIFECYCLE_STATES: [&str; 5] = ["starting", "serving", "draining", "stopped", "failed"];
const HEALTH_STATES: [&str; 3] = ["unknown", "healthy", "unhealthy"];

/// Immutable model inventory plus scrape-time operations rendering.
pub(crate) struct Operations {
    models: Bytes,
}

impl Operations {
    pub(crate) fn build(config: &Config) -> Result<Self, RouterError> {
        let profile_ids = config.workers.iter().flat_map(|worker| {
            worker
                .service_profiles
                .iter()
                .filter_map(|profile| profile.model_ids())
        });
        let defaults = config
            .workers
            .iter()
            .filter_map(|worker| worker.default_model_id.as_deref());
        Ok(Self {
            models: render_model_sources(profile_ids, defaults)?,
        })
    }

    pub(crate) fn models_response(&self) -> Response<Body> {
        response(StatusCode::OK, JSON_CONTENT_TYPE, self.models.clone())
    }

    pub(crate) fn metrics_response(
        &self,
        lifecycle: LifecycleState,
        ready: bool,
        snapshot: &OperationsSnapshot,
    ) -> Response<Body> {
        response(
            StatusCode::OK,
            METRICS_CONTENT_TYPE,
            Bytes::from(render_metrics(lifecycle, ready, snapshot)),
        )
    }

    pub(crate) fn diagnostics_response(
        &self,
        lifecycle: LifecycleState,
        ready: bool,
        snapshot: &OperationsSnapshot,
    ) -> Result<Response<Body>, HttpFault> {
        let diagnostics = Diagnostics::from_snapshot(lifecycle, ready, snapshot);
        let bytes = serde_json::to_vec(&diagnostics).map_err(|_| HttpFault::InternalError)?;
        Ok(response(
            StatusCode::OK,
            JSON_CONTENT_TYPE,
            Bytes::from(bytes),
        ))
    }
}

fn response(status: StatusCode, content_type: &'static str, bytes: Bytes) -> Response<Body> {
    let content_length = bytes.len();
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
        .headers_mut()
        .insert(CONTENT_LENGTH, HeaderValue::from(content_length));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

#[derive(Serialize)]
struct ModelList<'a> {
    object: &'static str,
    data: Vec<ModelCard<'a>>,
}

#[derive(Serialize)]
struct ModelCard<'a> {
    id: &'a str,
    object: &'static str,
    created: u8,
    owned_by: &'static str,
    permission: [ModelPermission; 1],
    root: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct ModelPermission {
    id: &'static str,
    object: &'static str,
    allow_create_engine: bool,
    allow_sampling: bool,
    allow_logprobs: bool,
}

fn render_model_sources<'a>(
    profile_ids: impl Iterator<Item = &'a [String]>,
    defaults: impl Iterator<Item = &'a str>,
) -> Result<Bytes, RouterError> {
    render_models(profile_ids.flatten().map(String::as_str).chain(defaults))
}

fn render_models<'a>(ids: impl Iterator<Item = &'a str>) -> Result<Bytes, RouterError> {
    let ids: BTreeSet<_> = ids.collect();
    let permission = ModelPermission {
        id: "modelperm-default",
        object: "model_permission",
        allow_create_engine: false,
        allow_sampling: true,
        allow_logprobs: true,
    };
    let models = ModelList {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelCard {
                id,
                object: "model",
                created: 0,
                owned_by: "sglang-omni",
                permission: [permission],
                root: id,
            })
            .collect(),
    };
    serde_json::to_vec(&models)
        .map(Bytes::from)
        .map_err(|_| RouterError::WorkerPoolInvariant)
}

fn render_metrics(lifecycle: LifecycleState, ready: bool, snapshot: &OperationsSnapshot) -> String {
    let mut output = String::new();
    output.push_str("# HELP sglang_omni_router_lifecycle Router lifecycle state.\n");
    output.push_str("# TYPE sglang_omni_router_lifecycle gauge\n");
    for state in LIFECYCLE_STATES {
        let value = u8::from(state == lifecycle.label());
        let _ = writeln!(
            output,
            "sglang_omni_router_lifecycle{{state=\"{state}\"}} {value}"
        );
    }
    output.push_str("# HELP sglang_omni_router_ready Router readiness state.\n");
    output.push_str("# TYPE sglang_omni_router_ready gauge\n");
    let _ = writeln!(output, "sglang_omni_router_ready {}", u8::from(ready));

    output.push_str("# HELP sglang_omni_router_workers_by_health Workers by health state.\n");
    output.push_str("# TYPE sglang_omni_router_workers_by_health gauge\n");
    for health in HEALTH_STATES {
        let count = snapshot
            .workers
            .iter()
            .filter(|worker| worker.health.label() == health)
            .count();
        let _ = writeln!(
            output,
            "sglang_omni_router_workers_by_health{{health=\"{health}\"}} {count}"
        );
    }

    output.push_str("# HELP sglang_omni_router_workers_routable Routable workers.\n");
    output.push_str("# TYPE sglang_omni_router_workers_routable gauge\n");
    let routable = snapshot
        .workers
        .iter()
        .filter(|worker| worker.routable)
        .count();
    let _ = writeln!(output, "sglang_omni_router_workers_routable {routable}");

    render_admission_metrics(&mut output, snapshot);
    render_worker_metrics(&mut output, snapshot);
    output
}

fn render_admission_metrics(output: &mut String, snapshot: &OperationsSnapshot) {
    output.push_str(
        "# HELP sglang_omni_router_admission_limit Configured admission limit in class-specific units.\n",
    );
    output.push_str("# TYPE sglang_omni_router_admission_limit gauge\n");
    for entry in &snapshot.admission {
        let _ = writeln!(
            output,
            "sglang_omni_router_admission_limit{{class=\"{}\"}} {}",
            entry.class.label(),
            entry.limit
        );
    }
    output.push_str(
        "# HELP sglang_omni_router_admission_in_flight Current admission usage in class-specific units.\n",
    );
    output.push_str("# TYPE sglang_omni_router_admission_in_flight gauge\n");
    for entry in &snapshot.admission {
        let _ = writeln!(
            output,
            "sglang_omni_router_admission_in_flight{{class=\"{}\"}} {}",
            entry.class.label(),
            entry.in_flight
        );
    }
}

fn render_worker_metrics(output: &mut String, snapshot: &OperationsSnapshot) {
    let active_requests: usize = snapshot
        .workers
        .iter()
        .map(|worker| worker.active_requests)
        .sum();
    output.push_str(
        "# HELP sglang_omni_router_worker_active_requests Aggregate active worker load; speech batches are item-weighted.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_active_requests gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_worker_active_requests {active_requests}"
    );

    let classes = [
        CapacityClass::SpeechWebsocket,
        CapacityClass::RealtimeWebsocket,
    ];
    let mut limits = [0_usize; 2];
    let mut in_flight = [0_usize; 2];
    for worker in &snapshot.workers {
        for capacity in &worker.session_capacity {
            if let Some(index) = classes.iter().position(|class| *class == capacity.class) {
                limits[index] += capacity.limit;
                in_flight[index] += capacity.in_flight;
            }
        }
    }
    output.push_str(
        "# HELP sglang_omni_router_worker_capacity_limit Aggregate configured worker capacity.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_capacity_limit gauge\n");
    for (index, class) in classes.into_iter().enumerate() {
        let _ = writeln!(
            output,
            "sglang_omni_router_worker_capacity_limit{{class=\"{}\"}} {}",
            class.label(),
            limits[index]
        );
    }
    output.push_str(
        "# HELP sglang_omni_router_worker_capacity_in_flight Aggregate current worker capacity use.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_capacity_in_flight gauge\n");
    for (index, class) in classes.into_iter().enumerate() {
        let _ = writeln!(
            output,
            "sglang_omni_router_worker_capacity_in_flight{{class=\"{}\"}} {}",
            class.label(),
            in_flight[index]
        );
    }
}

#[derive(Serialize)]
struct Diagnostics<'a> {
    lifecycle: &'static str,
    ready: bool,
    admission: Vec<DiagnosticCapacity>,
    workers: Vec<DiagnosticWorker<'a>>,
}

#[derive(Serialize)]
struct DiagnosticCapacity {
    class: &'static str,
    limit: usize,
    in_flight: usize,
}

#[derive(Serialize)]
struct DiagnosticWorker<'a> {
    worker_id: &'a str,
    registration_ordinal: usize,
    health: &'static str,
    routable: bool,
    active_requests: usize,
    capacity: Vec<DiagnosticCapacity>,
}

impl<'a> Diagnostics<'a> {
    fn from_snapshot(
        lifecycle: LifecycleState,
        ready: bool,
        snapshot: &'a OperationsSnapshot,
    ) -> Self {
        Self {
            lifecycle: lifecycle.label(),
            ready,
            admission: snapshot
                .admission
                .iter()
                .map(|entry| DiagnosticCapacity {
                    class: entry.class.label(),
                    limit: entry.limit,
                    in_flight: entry.in_flight,
                })
                .collect(),
            workers: snapshot
                .workers
                .iter()
                .map(|worker| DiagnosticWorker {
                    worker_id: &worker.worker_id,
                    registration_ordinal: worker.registration_ordinal,
                    health: worker.health.label(),
                    routable: worker.routable,
                    active_requests: worker.active_requests,
                    capacity: worker
                        .session_capacity
                        .iter()
                        .map(|entry| DiagnosticCapacity {
                            class: entry.class.label(),
                            limit: entry.limit,
                            in_flight: entry.in_flight,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::collections::BTreeSet;

    use crate::lifecycle::State as LifecycleState;
    use crate::worker_pool::{
        AdmissionClass, AdmissionSnapshot, CapacityClass, OperationsSnapshot,
        SessionCapacitySnapshot, WorkerHealth, WorkerSnapshot,
    };

    use super::{Diagnostics, render_metrics, render_model_sources, render_models};

    fn admission(class: AdmissionClass, limit: usize, in_flight: usize) -> AdmissionSnapshot {
        AdmissionSnapshot {
            class,
            limit,
            in_flight,
        }
    }

    fn capacity(class: CapacityClass, limit: usize, in_flight: usize) -> SessionCapacitySnapshot {
        SessionCapacitySnapshot {
            class,
            limit,
            in_flight,
        }
    }

    fn representative_snapshot() -> OperationsSnapshot {
        OperationsSnapshot {
            admission: [
                admission(AdmissionClass::Global, 100, 0),
                admission(
                    AdmissionClass::Service(CapacityClass::GenerationHttp),
                    101,
                    1,
                ),
                admission(AdmissionClass::Service(CapacityClass::SpeechHttp), 102, 2),
                admission(AdmissionClass::Service(CapacityClass::SpeechBatch), 103, 3),
                admission(
                    AdmissionClass::Service(CapacityClass::TranscriptionHttp),
                    104,
                    4,
                ),
                admission(
                    AdmissionClass::Service(CapacityClass::SpeechWebsocket),
                    105,
                    5,
                ),
                admission(
                    AdmissionClass::Service(CapacityClass::RealtimeWebsocket),
                    106,
                    6,
                ),
            ],
            workers: vec![
                WorkerSnapshot {
                    worker_id: String::from("worker-a"),
                    registration_ordinal: 0,
                    health: WorkerHealth::Unknown,
                    routable: false,
                    active_requests: 3,
                    session_capacity: vec![
                        capacity(CapacityClass::SpeechWebsocket, 10, 0),
                        capacity(CapacityClass::RealtimeWebsocket, 11, 1),
                    ],
                },
                WorkerSnapshot {
                    worker_id: String::from("worker-b"),
                    registration_ordinal: 1,
                    health: WorkerHealth::Healthy,
                    routable: true,
                    active_requests: 1,
                    session_capacity: vec![capacity(CapacityClass::RealtimeWebsocket, 2, 1)],
                },
            ],
        }
    }

    #[test]
    fn model_bytes_are_exact_sorted_and_deduplicated() {
        let empty = render_models(std::iter::empty()).expect("serialize empty model list");
        assert_eq!(empty.as_ref(), br#"{"object":"list","data":[]}"#);

        let bytes = render_models(["zeta", "alpha", "zeta"].into_iter())
            .expect("serialize fixed model schema");
        assert_eq!(
            bytes.as_ref(),
            br#"{"object":"list","data":[{"id":"alpha","object":"model","created":0,"owned_by":"sglang-omni","permission":[{"id":"modelperm-default","object":"model_permission","allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true}],"root":"alpha"},{"id":"zeta","object":"model","created":0,"owned_by":"sglang-omni","permission":[{"id":"modelperm-default","object":"model_permission","allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true}],"root":"zeta"}]}"#
        );

        let first = vec![String::from("zeta"), String::from("shared")];
        let second = vec![String::from("alpha"), String::from("shared")];
        let union = render_model_sources(
            [first.as_slice(), second.as_slice()].into_iter(),
            ["realtime-only", "alpha"].into_iter(),
        )
        .expect("serialize exact model union");
        let value: serde_json::Value =
            serde_json::from_slice(&union).expect("parse canonical model JSON");
        let ids: Vec<_> = value["data"]
            .as_array()
            .expect("model data array")
            .iter()
            .map(|card| card["id"].as_str().expect("model id"))
            .collect();
        assert_eq!(ids, ["alpha", "realtime-only", "shared", "zeta"]);
    }

    #[test]
    fn metrics_text_is_complete_exact_and_fixed_order() {
        let rendered = render_metrics(LifecycleState::Serving, true, &representative_snapshot());
        assert_eq!(
            rendered,
            concat!(
                "# HELP sglang_omni_router_lifecycle Router lifecycle state.\n",
                "# TYPE sglang_omni_router_lifecycle gauge\n",
                "sglang_omni_router_lifecycle{state=\"starting\"} 0\n",
                "sglang_omni_router_lifecycle{state=\"serving\"} 1\n",
                "sglang_omni_router_lifecycle{state=\"draining\"} 0\n",
                "sglang_omni_router_lifecycle{state=\"stopped\"} 0\n",
                "sglang_omni_router_lifecycle{state=\"failed\"} 0\n",
                "# HELP sglang_omni_router_ready Router readiness state.\n",
                "# TYPE sglang_omni_router_ready gauge\n",
                "sglang_omni_router_ready 1\n",
                "# HELP sglang_omni_router_workers_by_health Workers by health state.\n",
                "# TYPE sglang_omni_router_workers_by_health gauge\n",
                "sglang_omni_router_workers_by_health{health=\"unknown\"} 1\n",
                "sglang_omni_router_workers_by_health{health=\"healthy\"} 1\n",
                "sglang_omni_router_workers_by_health{health=\"unhealthy\"} 0\n",
                "# HELP sglang_omni_router_workers_routable Routable workers.\n",
                "# TYPE sglang_omni_router_workers_routable gauge\n",
                "sglang_omni_router_workers_routable 1\n",
                "# HELP sglang_omni_router_admission_limit Configured admission limit in class-specific units.\n",
                "# TYPE sglang_omni_router_admission_limit gauge\n",
                "sglang_omni_router_admission_limit{class=\"global\"} 100\n",
                "sglang_omni_router_admission_limit{class=\"generation_http\"} 101\n",
                "sglang_omni_router_admission_limit{class=\"speech_http\"} 102\n",
                "sglang_omni_router_admission_limit{class=\"speech_batch\"} 103\n",
                "sglang_omni_router_admission_limit{class=\"transcription_http\"} 104\n",
                "sglang_omni_router_admission_limit{class=\"speech_websocket\"} 105\n",
                "sglang_omni_router_admission_limit{class=\"realtime_websocket\"} 106\n",
                "# HELP sglang_omni_router_admission_in_flight Current admission usage in class-specific units.\n",
                "# TYPE sglang_omni_router_admission_in_flight gauge\n",
                "sglang_omni_router_admission_in_flight{class=\"global\"} 0\n",
                "sglang_omni_router_admission_in_flight{class=\"generation_http\"} 1\n",
                "sglang_omni_router_admission_in_flight{class=\"speech_http\"} 2\n",
                "sglang_omni_router_admission_in_flight{class=\"speech_batch\"} 3\n",
                "sglang_omni_router_admission_in_flight{class=\"transcription_http\"} 4\n",
                "sglang_omni_router_admission_in_flight{class=\"speech_websocket\"} 5\n",
                "sglang_omni_router_admission_in_flight{class=\"realtime_websocket\"} 6\n",
                "# HELP sglang_omni_router_worker_active_requests Aggregate active worker load; speech batches are item-weighted.\n",
                "# TYPE sglang_omni_router_worker_active_requests gauge\n",
                "sglang_omni_router_worker_active_requests 4\n",
                "# HELP sglang_omni_router_worker_capacity_limit Aggregate configured worker capacity.\n",
                "# TYPE sglang_omni_router_worker_capacity_limit gauge\n",
                "sglang_omni_router_worker_capacity_limit{class=\"speech_websocket\"} 10\n",
                "sglang_omni_router_worker_capacity_limit{class=\"realtime_websocket\"} 13\n",
                "# HELP sglang_omni_router_worker_capacity_in_flight Aggregate current worker capacity use.\n",
                "# TYPE sglang_omni_router_worker_capacity_in_flight gauge\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"speech_websocket\"} 0\n",
                "sglang_omni_router_worker_capacity_in_flight{class=\"realtime_websocket\"} 2\n",
            )
        );
    }

    #[test]
    fn maximum_diagnostics_are_bounded_ordered_and_redacted() {
        let admission = representative_snapshot().admission;
        let workers = (0..256)
            .map(|registration_ordinal| WorkerSnapshot {
                worker_id: format!("worker-{registration_ordinal:03}"),
                registration_ordinal,
                health: match registration_ordinal % 3 {
                    0 => WorkerHealth::Unknown,
                    1 => WorkerHealth::Healthy,
                    _ => WorkerHealth::Unhealthy,
                },
                routable: registration_ordinal % 2 == 0,
                active_requests: registration_ordinal,
                session_capacity: vec![
                    capacity(CapacityClass::SpeechWebsocket, 1, 0),
                    capacity(CapacityClass::RealtimeWebsocket, 2, 1),
                ],
            })
            .collect();
        let snapshot = OperationsSnapshot { admission, workers };
        let bytes = serde_json::to_vec(&Diagnostics::from_snapshot(
            LifecycleState::Draining,
            false,
            &snapshot,
        ))
        .expect("serialize maximum diagnostics");
        assert!(bytes.len() < 256 * 1_024);

        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse diagnostics JSON");
        let keys: BTreeSet<_> = value
            .as_object()
            .expect("diagnostics object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            BTreeSet::from(["admission", "lifecycle", "ready", "workers"])
        );
        assert_eq!(value["workers"][0]["registration_ordinal"], 0);
        assert_eq!(value["workers"][255]["registration_ordinal"], 255);
        let text = String::from_utf8(bytes).expect("diagnostics are UTF-8");
        for forbidden in ["base_url", "trust_domain", "health_path", "request_id"] {
            assert!(!text.contains(forbidden));
        }
    }
}
