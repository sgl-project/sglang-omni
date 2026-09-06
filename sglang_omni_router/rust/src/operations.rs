use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;

use axum::body::Body;
use axum::http::header::{CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Response, StatusCode};
use bytes::Bytes;
use serde::Serialize;

use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::lifecycle::State as LifecycleState;
use crate::metrics::{HttpRoute, Rejection, RouterMetrics, StatusClass};
use crate::worker_pool::{
    OperationsSnapshot, ProbeOutcome, ProbeSnapshot, SESSION_CAPACITY_CLASSES, WorkerHealth,
};

const JSON_CONTENT_TYPE: &str = "application/json";
const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ResourceUsage {
    limit: usize,
    in_use: usize,
}

impl ResourceUsage {
    pub(crate) const fn new(limit: usize, in_use: usize) -> Self {
        Self { limit, in_use }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ResourceSnapshot {
    listener_slots: ResourceUsage,
    buffered_request_bytes: ResourceUsage,
    classification_slots: ResourceUsage,
    websocket_sessions_registered: usize,
}

impl ResourceSnapshot {
    pub(crate) const fn new(
        listener_slots: ResourceUsage,
        buffered_request_bytes: ResourceUsage,
        classification_slots: ResourceUsage,
        websocket_sessions_registered: usize,
    ) -> Self {
        Self {
            listener_slots,
            buffered_request_bytes,
            classification_slots,
            websocket_sessions_registered,
        }
    }
}

/// Immutable model inventory plus scrape-time operations rendering.
pub(crate) struct Operations {
    models: Bytes,
    metrics: Arc<RouterMetrics>,
}

impl Operations {
    pub(crate) fn build(config: &Config, metrics: Arc<RouterMetrics>) -> Result<Self, RouterError> {
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
            metrics,
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
        resources: &ResourceSnapshot,
    ) -> Response<Body> {
        response(
            StatusCode::OK,
            METRICS_CONTENT_TYPE,
            Bytes::from(render_metrics(
                lifecycle,
                ready,
                snapshot,
                resources,
                &self.metrics,
            )),
        )
    }

    pub(crate) fn diagnostics_response(
        &self,
        lifecycle: LifecycleState,
        ready: bool,
        snapshot: &OperationsSnapshot,
        resources: &ResourceSnapshot,
    ) -> Result<Response<Body>, HttpFault> {
        let diagnostics = Diagnostics::from_snapshot(lifecycle, ready, snapshot, resources);
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

fn render_metrics(
    lifecycle: LifecycleState,
    ready: bool,
    snapshot: &OperationsSnapshot,
    resources: &ResourceSnapshot,
    metrics: &RouterMetrics,
) -> String {
    let mut output = String::new();
    output.push_str("# HELP sglang_omni_router_lifecycle Router lifecycle state.\n");
    output.push_str("# TYPE sglang_omni_router_lifecycle gauge\n");
    for state in LifecycleState::ALL {
        let value = u8::from(state == lifecycle);
        let _ = writeln!(
            output,
            "sglang_omni_router_lifecycle{{state=\"{}\"}} {value}",
            state.label()
        );
    }
    output.push_str("# HELP sglang_omni_router_ready Router readiness state.\n");
    output.push_str("# TYPE sglang_omni_router_ready gauge\n");
    let _ = writeln!(output, "sglang_omni_router_ready {}", u8::from(ready));

    render_request_metrics(&mut output, metrics);

    output.push_str("# HELP sglang_omni_router_workers_by_health Workers by health state.\n");
    output.push_str("# TYPE sglang_omni_router_workers_by_health gauge\n");
    for health in WorkerHealth::ALL {
        let count = snapshot
            .workers
            .iter()
            .filter(|worker| worker.health == health)
            .count();
        let _ = writeln!(
            output,
            "sglang_omni_router_workers_by_health{{health=\"{}\"}} {count}",
            health.label()
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

    render_probe_metrics(&mut output, snapshot);
    render_admission_metrics(&mut output, snapshot);
    render_worker_metrics(&mut output, snapshot);
    render_resource_metrics(&mut output, resources);
    output
}

fn render_request_metrics(output: &mut String, metrics: &RouterMetrics) {
    output.push_str("# HELP sglang_omni_router_http_requests_total HTTP requests received.\n");
    output.push_str("# TYPE sglang_omni_router_http_requests_total counter\n");
    for route in HttpRoute::ALL {
        let count = metrics.requests(route);
        if count != 0 {
            let _ = writeln!(
                output,
                "sglang_omni_router_http_requests_total{{route=\"{}\"}} {count}",
                route.label()
            );
        }
    }

    output.push_str(
        "# HELP sglang_omni_router_http_response_headers_total HTTP response headers by status class.\n",
    );
    output.push_str("# TYPE sglang_omni_router_http_response_headers_total counter\n");
    for route in HttpRoute::ALL {
        for status in StatusClass::ALL {
            let count = metrics.responses(route, status);
            if count != 0 {
                let _ = writeln!(
                    output,
                    "sglang_omni_router_http_response_headers_total{{route=\"{}\",status=\"{}\"}} {count}",
                    route.label(),
                    status.label()
                );
            }
        }
    }

    output.push_str("# HELP sglang_omni_router_http_faults_total Router-generated HTTP faults.\n");
    output.push_str("# TYPE sglang_omni_router_http_faults_total counter\n");
    for route in HttpRoute::ALL {
        for fault in HttpFault::ALL {
            let count = metrics.faults(route, fault);
            if count != 0 {
                let _ = writeln!(
                    output,
                    "sglang_omni_router_http_faults_total{{route=\"{}\",code=\"{}\"}} {count}",
                    route.label(),
                    fault.code()
                );
            }
        }
    }

    output.push_str(
        "# HELP sglang_omni_router_rejections_total Requests rejected by a saturated bounded resource.\n",
    );
    output.push_str("# TYPE sglang_omni_router_rejections_total counter\n");
    for rejection in Rejection::ALL {
        let count = metrics.rejections(rejection);
        if count != 0 {
            let _ = writeln!(
                output,
                "sglang_omni_router_rejections_total{{resource=\"{}\"}} {count}",
                rejection.label()
            );
        }
    }

    output.push_str(
        "# HELP sglang_omni_router_http_relay_failures_total Upstream body failures after response commitment.\n",
    );
    output.push_str("# TYPE sglang_omni_router_http_relay_failures_total counter\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_http_relay_failures_total {}",
        metrics.relay_failures()
    );
}

fn render_probe_metrics(output: &mut String, snapshot: &OperationsSnapshot) {
    output.push_str("# HELP sglang_omni_router_worker_probes_total Worker health probes.\n");
    output.push_str("# TYPE sglang_omni_router_worker_probes_total counter\n");
    for outcome in ProbeOutcome::OBSERVED {
        let count: u64 = snapshot
            .workers
            .iter()
            .map(|worker| match outcome {
                ProbeOutcome::Success => worker.probe.successes,
                ProbeOutcome::HttpFailure => worker.probe.http_failures,
                ProbeOutcome::TransportFailure => worker.probe.transport_failures,
                ProbeOutcome::Pending => 0,
            })
            .sum();
        let _ = writeln!(
            output,
            "sglang_omni_router_worker_probes_total{{outcome=\"{}\"}} {count}",
            outcome.label()
        );
    }
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

    let mut limits = [0_usize; 2];
    let mut in_flight = [0_usize; 2];
    for worker in &snapshot.workers {
        for capacity in &worker.session_capacity {
            if let Some(index) = SESSION_CAPACITY_CLASSES
                .iter()
                .position(|class| *class == capacity.class)
            {
                limits[index] += capacity.limit;
                in_flight[index] += capacity.in_flight;
            }
        }
    }
    output.push_str(
        "# HELP sglang_omni_router_worker_capacity_limit Aggregate configured worker capacity.\n",
    );
    output.push_str("# TYPE sglang_omni_router_worker_capacity_limit gauge\n");
    for (index, class) in SESSION_CAPACITY_CLASSES.into_iter().enumerate() {
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
    for (index, class) in SESSION_CAPACITY_CLASSES.into_iter().enumerate() {
        let _ = writeln!(
            output,
            "sglang_omni_router_worker_capacity_in_flight{{class=\"{}\"}} {}",
            class.label(),
            in_flight[index]
        );
    }
}

fn render_resource_metrics(output: &mut String, resources: &ResourceSnapshot) {
    output.push_str(
        "# HELP sglang_omni_router_listener_slots_limit Configured accepted-socket limit.\n",
    );
    output.push_str("# TYPE sglang_omni_router_listener_slots_limit gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_listener_slots_limit {}",
        resources.listener_slots.limit
    );
    output.push_str(
        "# HELP sglang_omni_router_listener_slots_reserved Reserved accepted-socket slots, including the pending accept reservation.\n",
    );
    output.push_str("# TYPE sglang_omni_router_listener_slots_reserved gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_listener_slots_reserved {}",
        resources.listener_slots.in_use
    );

    output.push_str(
        "# HELP sglang_omni_router_buffered_request_bytes_limit Configured aggregate buffered-request byte budget.\n",
    );
    output.push_str("# TYPE sglang_omni_router_buffered_request_bytes_limit gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_buffered_request_bytes_limit {}",
        resources.buffered_request_bytes.limit
    );
    output.push_str(
        "# HELP sglang_omni_router_buffered_request_bytes_reserved Reserved buffered-request bytes.\n",
    );
    output.push_str("# TYPE sglang_omni_router_buffered_request_bytes_reserved gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_buffered_request_bytes_reserved {}",
        resources.buffered_request_bytes.in_use
    );

    output.push_str(
        "# HELP sglang_omni_router_classification_slots_limit Configured CPU-parallel classification slots.\n",
    );
    output.push_str("# TYPE sglang_omni_router_classification_slots_limit gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_classification_slots_limit {}",
        resources.classification_slots.limit
    );
    output.push_str(
        "# HELP sglang_omni_router_classification_slots_in_use Active classification slots.\n",
    );
    output.push_str("# TYPE sglang_omni_router_classification_slots_in_use gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_classification_slots_in_use {}",
        resources.classification_slots.in_use
    );

    output.push_str(
        "# HELP sglang_omni_router_websocket_sessions_registered Registered WebSocket callbacks retained for shutdown.\n",
    );
    output.push_str("# TYPE sglang_omni_router_websocket_sessions_registered gauge\n");
    let _ = writeln!(
        output,
        "sglang_omni_router_websocket_sessions_registered {}",
        resources.websocket_sessions_registered
    );
}

#[derive(Serialize)]
struct Diagnostics<'a> {
    lifecycle: &'static str,
    ready: bool,
    resources: ResourceSnapshot,
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
    voice_owner: bool,
    health: &'static str,
    probe: DiagnosticProbe,
    routable: bool,
    active_requests: usize,
    capacity: Vec<DiagnosticCapacity>,
}

#[derive(Serialize)]
struct DiagnosticProbe {
    outcome: &'static str,
    status: Option<u16>,
    checked_at_unix_ms: Option<u64>,
    consecutive_successes: u8,
    consecutive_failures: u8,
    successes: u64,
    http_failures: u64,
    transport_failures: u64,
}

impl From<ProbeSnapshot> for DiagnosticProbe {
    fn from(probe: ProbeSnapshot) -> Self {
        Self {
            outcome: probe.outcome.label(),
            status: probe.status,
            checked_at_unix_ms: probe.checked_at_unix_ms,
            consecutive_successes: probe.consecutive_successes,
            consecutive_failures: probe.consecutive_failures,
            successes: probe.successes,
            http_failures: probe.http_failures,
            transport_failures: probe.transport_failures,
        }
    }
}

impl<'a> Diagnostics<'a> {
    fn from_snapshot(
        lifecycle: LifecycleState,
        ready: bool,
        snapshot: &'a OperationsSnapshot,
        resources: &ResourceSnapshot,
    ) -> Self {
        Self {
            lifecycle: lifecycle.label(),
            ready,
            resources: *resources,
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
                    voice_owner: worker.voice_owner,
                    health: worker.health.label(),
                    probe: worker.probe.into(),
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

    use axum::http::{Response, StatusCode};

    use crate::error::HttpFault;
    use crate::lifecycle::State as LifecycleState;
    use crate::metrics::{HttpRoute, Rejection, RouterMetrics};
    use crate::worker_pool::{
        AdmissionClass, AdmissionSnapshot, CapacityClass, OperationsSnapshot, ProbeOutcome,
        ProbeSnapshot, SessionCapacitySnapshot, WorkerHealth, WorkerSnapshot,
    };

    use super::{
        Diagnostics, ResourceSnapshot, ResourceUsage, render_metrics, render_model_sources,
        render_models,
    };

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

    fn probe(outcome: ProbeOutcome, successes: u64, failures: u64) -> ProbeSnapshot {
        ProbeSnapshot {
            outcome,
            status: Some(if matches!(outcome, ProbeOutcome::Success) {
                200
            } else {
                503
            }),
            checked_at_unix_ms: Some(1_000),
            consecutive_successes: u8::from(matches!(outcome, ProbeOutcome::Success)),
            consecutive_failures: u8::from(!matches!(outcome, ProbeOutcome::Success)),
            successes,
            http_failures: failures,
            transport_failures: 0,
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
                    voice_owner: true,
                    health: WorkerHealth::Unknown,
                    probe: probe(ProbeOutcome::HttpFailure, 2, 3),
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
                    voice_owner: false,
                    health: WorkerHealth::Healthy,
                    probe: probe(ProbeOutcome::Success, 4, 1),
                    routable: true,
                    active_requests: 1,
                    session_capacity: vec![capacity(CapacityClass::RealtimeWebsocket, 2, 1)],
                },
            ],
        }
    }

    const fn resources() -> ResourceSnapshot {
        ResourceSnapshot::new(
            ResourceUsage::new(1024, 3),
            ResourceUsage::new(268_435_456, 4096),
            ResourceUsage::new(8, 2),
            1,
        )
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
        let metrics = RouterMetrics::new();
        let rendered = render_metrics(
            LifecycleState::Serving,
            true,
            &representative_snapshot(),
            &resources(),
            &metrics,
        );
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
                "# HELP sglang_omni_router_http_requests_total HTTP requests received.\n",
                "# TYPE sglang_omni_router_http_requests_total counter\n",
                "# HELP sglang_omni_router_http_response_headers_total HTTP response headers by status class.\n",
                "# TYPE sglang_omni_router_http_response_headers_total counter\n",
                "# HELP sglang_omni_router_http_faults_total Router-generated HTTP faults.\n",
                "# TYPE sglang_omni_router_http_faults_total counter\n",
                "# HELP sglang_omni_router_rejections_total Requests rejected by a saturated bounded resource.\n",
                "# TYPE sglang_omni_router_rejections_total counter\n",
                "# HELP sglang_omni_router_http_relay_failures_total Upstream body failures after response commitment.\n",
                "# TYPE sglang_omni_router_http_relay_failures_total counter\n",
                "sglang_omni_router_http_relay_failures_total 0\n",
                "# HELP sglang_omni_router_workers_by_health Workers by health state.\n",
                "# TYPE sglang_omni_router_workers_by_health gauge\n",
                "sglang_omni_router_workers_by_health{health=\"unknown\"} 1\n",
                "sglang_omni_router_workers_by_health{health=\"healthy\"} 1\n",
                "sglang_omni_router_workers_by_health{health=\"unhealthy\"} 0\n",
                "# HELP sglang_omni_router_workers_routable Routable workers.\n",
                "# TYPE sglang_omni_router_workers_routable gauge\n",
                "sglang_omni_router_workers_routable 1\n",
                "# HELP sglang_omni_router_worker_probes_total Worker health probes.\n",
                "# TYPE sglang_omni_router_worker_probes_total counter\n",
                "sglang_omni_router_worker_probes_total{outcome=\"success\"} 6\n",
                "sglang_omni_router_worker_probes_total{outcome=\"http_failure\"} 4\n",
                "sglang_omni_router_worker_probes_total{outcome=\"transport_failure\"} 0\n",
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
                "# HELP sglang_omni_router_listener_slots_limit Configured accepted-socket limit.\n",
                "# TYPE sglang_omni_router_listener_slots_limit gauge\n",
                "sglang_omni_router_listener_slots_limit 1024\n",
                "# HELP sglang_omni_router_listener_slots_reserved Reserved accepted-socket slots, including the pending accept reservation.\n",
                "# TYPE sglang_omni_router_listener_slots_reserved gauge\n",
                "sglang_omni_router_listener_slots_reserved 3\n",
                "# HELP sglang_omni_router_buffered_request_bytes_limit Configured aggregate buffered-request byte budget.\n",
                "# TYPE sglang_omni_router_buffered_request_bytes_limit gauge\n",
                "sglang_omni_router_buffered_request_bytes_limit 268435456\n",
                "# HELP sglang_omni_router_buffered_request_bytes_reserved Reserved buffered-request bytes.\n",
                "# TYPE sglang_omni_router_buffered_request_bytes_reserved gauge\n",
                "sglang_omni_router_buffered_request_bytes_reserved 4096\n",
                "# HELP sglang_omni_router_classification_slots_limit Configured CPU-parallel classification slots.\n",
                "# TYPE sglang_omni_router_classification_slots_limit gauge\n",
                "sglang_omni_router_classification_slots_limit 8\n",
                "# HELP sglang_omni_router_classification_slots_in_use Active classification slots.\n",
                "# TYPE sglang_omni_router_classification_slots_in_use gauge\n",
                "sglang_omni_router_classification_slots_in_use 2\n",
                "# HELP sglang_omni_router_websocket_sessions_registered Registered WebSocket callbacks retained for shutdown.\n",
                "# TYPE sglang_omni_router_websocket_sessions_registered gauge\n",
                "sglang_omni_router_websocket_sessions_registered 1\n",
            )
        );
    }

    #[test]
    fn counter_metrics_use_fixed_route_fault_and_resource_labels() {
        let metrics = RouterMetrics::new();
        metrics.record_request(HttpRoute::Speech);
        let mut response = Response::new(());
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        response
            .extensions_mut()
            .insert(HttpFault::RouterOverloaded);
        metrics.record_response(HttpRoute::Speech, &response);
        metrics.record_rejection(Rejection::SpeechAdmission);
        metrics.record_relay_failure();

        let rendered = render_metrics(
            LifecycleState::Serving,
            true,
            &representative_snapshot(),
            &resources(),
            &metrics,
        );
        for sample in [
            "sglang_omni_router_http_requests_total{route=\"speech\"} 1\n",
            "sglang_omni_router_http_response_headers_total{route=\"speech\",status=\"4xx\"} 1\n",
            "sglang_omni_router_http_faults_total{route=\"speech\",code=\"router_overloaded\"} 1\n",
            "sglang_omni_router_rejections_total{resource=\"admission_speech_http\"} 1\n",
            "sglang_omni_router_http_relay_failures_total 1\n",
        ] {
            assert!(rendered.contains(sample), "missing metric sample: {sample}");
        }
    }

    #[test]
    fn maximum_diagnostics_are_bounded_ordered_and_redacted() {
        let admission = representative_snapshot().admission;
        let workers = (0..256)
            .map(|registration_ordinal| WorkerSnapshot {
                worker_id: format!("worker-{registration_ordinal:03}"),
                registration_ordinal,
                voice_owner: registration_ordinal == 0,
                health: match registration_ordinal % 3 {
                    0 => WorkerHealth::Unknown,
                    1 => WorkerHealth::Healthy,
                    _ => WorkerHealth::Unhealthy,
                },
                probe: probe(ProbeOutcome::Success, 1, 0),
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
            &resources(),
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
            BTreeSet::from(["admission", "lifecycle", "ready", "resources", "workers"])
        );
        assert_eq!(value["lifecycle"], "draining");
        assert_eq!(value["ready"], false);
        assert_eq!(value["workers"][0]["voice_owner"], true);
        assert_eq!(value["workers"][1]["voice_owner"], false);
        assert_eq!(value["workers"][0]["probe"]["outcome"], "success");
        assert_eq!(value["workers"][0]["registration_ordinal"], 0);
        assert_eq!(value["workers"][255]["registration_ordinal"], 255);
        let text = String::from_utf8(bytes).expect("diagnostics are UTF-8");
        for forbidden in ["base_url", "trust_domain", "health_path", "request_id"] {
            assert!(!text.contains(forbidden));
        }
    }
}
