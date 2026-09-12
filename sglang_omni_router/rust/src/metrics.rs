use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::http::{Response, StatusCode};

use crate::error::HttpFault;
use crate::worker_pool::CapacityClass;

pub(crate) const DURATION_BUCKETS: [DurationBucket; 26] = [
    DurationBucket::new(10, "0.00001"),
    DurationBucket::new(25, "0.000025"),
    DurationBucket::new(50, "0.00005"),
    DurationBucket::new(100, "0.0001"),
    DurationBucket::new(250, "0.00025"),
    DurationBucket::new(500, "0.0005"),
    DurationBucket::new(1_000, "0.001"),
    DurationBucket::new(5_000, "0.005"),
    DurationBucket::new(10_000, "0.01"),
    DurationBucket::new(25_000, "0.025"),
    DurationBucket::new(50_000, "0.05"),
    DurationBucket::new(100_000, "0.1"),
    DurationBucket::new(250_000, "0.25"),
    DurationBucket::new(500_000, "0.5"),
    DurationBucket::new(1_000_000, "1"),
    DurationBucket::new(2_500_000, "2.5"),
    DurationBucket::new(5_000_000, "5"),
    DurationBucket::new(10_000_000, "10"),
    DurationBucket::new(15_000_000, "15"),
    DurationBucket::new(30_000_000, "30"),
    DurationBucket::new(45_000_000, "45"),
    DurationBucket::new(60_000_000, "60"),
    DurationBucket::new(90_000_000, "90"),
    DurationBucket::new(120_000_000, "120"),
    DurationBucket::new(180_000_000, "180"),
    DurationBucket::new(240_000_000, "240"),
];
pub(crate) const DURATION_BUCKET_COUNT: usize = DURATION_BUCKETS.len() + 1;

#[derive(Clone, Copy)]
pub(crate) struct DurationBucket {
    pub(crate) upper_micros: u64,
    pub(crate) label: &'static str,
}

impl DurationBucket {
    const fn new(upper_micros: u64, label: &'static str) -> Self {
        Self {
            upper_micros,
            label,
        }
    }
}

pub(crate) struct DurationHistogramSnapshot {
    pub(crate) buckets: [u64; DURATION_BUCKET_COUNT],
    pub(crate) sum_micros: u64,
}

impl DurationHistogramSnapshot {
    pub(crate) fn count(&self) -> u64 {
        self.buckets.iter().copied().fold(0, u64::saturating_add)
    }
}

struct DurationHistogram {
    buckets: [AtomicU64; DURATION_BUCKET_COUNT],
    sum_micros: AtomicU64,
}

impl DurationHistogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            sum_micros: AtomicU64::new(0),
        }
    }

    fn observe(&self, duration: Duration) {
        let micros = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        let index = DURATION_BUCKETS.partition_point(|bucket| bucket.upper_micros < micros);
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        let _updated = self
            .sum_micros
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |sum| {
                Some(sum.saturating_add(micros))
            });
    }

    fn snapshot(&self) -> DurationHistogramSnapshot {
        DurationHistogramSnapshot {
            buckets: std::array::from_fn(|index| self.buckets[index].load(Ordering::Relaxed)),
            sum_micros: self.sum_micros.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum HttpRoute {
    Live,
    Ready,
    Models,
    Metrics,
    Diagnostics,
    Chat,
    Speech,
    SpeechBatch,
    Transcription,
    Translation,
    VoiceCollection,
    VoiceItem,
    SpeechWebsocket,
    RealtimeWebsocket,
    Unknown,
}

impl HttpRoute {
    pub(crate) const ALL: [Self; 15] = [
        Self::Live,
        Self::Ready,
        Self::Models,
        Self::Metrics,
        Self::Diagnostics,
        Self::Chat,
        Self::Speech,
        Self::SpeechBatch,
        Self::Transcription,
        Self::Translation,
        Self::VoiceCollection,
        Self::VoiceItem,
        Self::SpeechWebsocket,
        Self::RealtimeWebsocket,
        Self::Unknown,
    ];

    pub(crate) fn from_path(path: &str) -> Self {
        match path {
            "/live" => Self::Live,
            "/ready" => Self::Ready,
            "/v1/models" => Self::Models,
            "/metrics" => Self::Metrics,
            "/diagnostics" => Self::Diagnostics,
            "/v1/chat/completions" => Self::Chat,
            "/v1/audio/speech" => Self::Speech,
            "/v1/audio/speech/batch" => Self::SpeechBatch,
            "/v1/audio/transcriptions" => Self::Transcription,
            "/v1/audio/translations" => Self::Translation,
            "/v1/audio/voices" => Self::VoiceCollection,
            "/v1/audio/speech/stream" => Self::SpeechWebsocket,
            "/v1/realtime" => Self::RealtimeWebsocket,
            path if voice_item(path) => Self::VoiceItem,
            _ => Self::Unknown,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Ready => "ready",
            Self::Models => "models",
            Self::Metrics => "metrics",
            Self::Diagnostics => "diagnostics",
            Self::Chat => "chat",
            Self::Speech => "speech",
            Self::SpeechBatch => "speech_batch",
            Self::Transcription => "transcription",
            Self::Translation => "translation",
            Self::VoiceCollection => "voice_collection",
            Self::VoiceItem => "voice_item",
            Self::SpeechWebsocket => "speech_websocket",
            Self::RealtimeWebsocket => "realtime_websocket",
            Self::Unknown => "unknown",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

fn voice_item(path: &str) -> bool {
    path.strip_prefix("/v1/audio/voices/")
        .is_some_and(|name| !name.is_empty() && !name.contains('/'))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum StatusClass {
    Informational,
    Success,
    Redirection,
    ClientError,
    ServerError,
    Other,
}

impl StatusClass {
    pub(crate) const ALL: [Self; 6] = [
        Self::Informational,
        Self::Success,
        Self::Redirection,
        Self::ClientError,
        Self::ServerError,
        Self::Other,
    ];

    fn from_status(status: StatusCode) -> Self {
        match status.as_u16() / 100 {
            1 => Self::Informational,
            2 => Self::Success,
            3 => Self::Redirection,
            4 => Self::ClientError,
            5 => Self::ServerError,
            _ => Self::Other,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Informational => "1xx",
            Self::Success => "2xx",
            Self::Redirection => "3xx",
            Self::ClientError => "4xx",
            Self::ServerError => "5xx",
            Self::Other => "other",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum Rejection {
    GlobalAdmission,
    GenerationAdmission,
    SpeechAdmission,
    SpeechBatchAdmission,
    TranscriptionAdmission,
    SpeechWebsocketAdmission,
    RealtimeWebsocketAdmission,
    BufferedRequestBytes,
    SpeechWebsocketWorker,
    RealtimeWebsocketWorker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum ClassificationKind {
    Chat,
    Speech,
    SpeechBatch,
    Transcription,
    Translation,
    SpeechWebsocket,
}

impl ClassificationKind {
    pub(crate) const ALL: [Self; 6] = [
        Self::Chat,
        Self::Speech,
        Self::SpeechBatch,
        Self::Transcription,
        Self::Translation,
        Self::SpeechWebsocket,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Speech => "speech",
            Self::SpeechBatch => "speech_batch",
            Self::Transcription => "transcription",
            Self::Translation => "translation",
            Self::SpeechWebsocket => "speech_websocket",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum ClassificationPhase {
    SlotWait,
    ExecutorWait,
    Execution,
}

impl ClassificationPhase {
    pub(crate) const ALL: [Self; 3] = [Self::SlotWait, Self::ExecutorWait, Self::Execution];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::SlotWait => "slot_wait",
            Self::ExecutorWait => "executor_wait",
            Self::Execution => "execution",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum ClassificationOutcome {
    Success,
    Error,
    Timeout,
    Cancelled,
}

impl ClassificationOutcome {
    pub(crate) const ALL: [Self; 4] = [Self::Success, Self::Error, Self::Timeout, Self::Cancelled];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum WebsocketProtocol {
    Speech,
    Realtime,
}

impl WebsocketProtocol {
    pub(crate) const ALL: [Self; 2] = [Self::Speech, Self::Realtime];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::Realtime => "realtime",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum WebsocketPhase {
    Setup,
    Relay,
}

impl WebsocketPhase {
    pub(crate) const ALL: [Self; 2] = [Self::Setup, Self::Relay];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Relay => "relay",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum WebsocketTermination {
    ClientClose,
    ClientDisconnect,
    ClientProtocolError,
    WorkerClose,
    WorkerDisconnect,
    WorkerProtocolError,
    ConfigurationError,
    ConfigurationTimeout,
    ClassificationError,
    ClassificationTimeout,
    DispatchError,
    ConnectError,
    ConnectTimeout,
    WorkerSetupError,
    WorkerSetupTimeout,
    Draining,
    ForcedShutdown,
    Cancelled,
    Internal,
}

impl WebsocketTermination {
    pub(crate) const ALL: [Self; 19] = [
        Self::ClientClose,
        Self::ClientDisconnect,
        Self::ClientProtocolError,
        Self::WorkerClose,
        Self::WorkerDisconnect,
        Self::WorkerProtocolError,
        Self::ConfigurationError,
        Self::ConfigurationTimeout,
        Self::ClassificationError,
        Self::ClassificationTimeout,
        Self::DispatchError,
        Self::ConnectError,
        Self::ConnectTimeout,
        Self::WorkerSetupError,
        Self::WorkerSetupTimeout,
        Self::Draining,
        Self::ForcedShutdown,
        Self::Cancelled,
        Self::Internal,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::ClientClose => "client_close",
            Self::ClientDisconnect => "client_disconnect",
            Self::ClientProtocolError => "client_protocol_error",
            Self::WorkerClose => "worker_close",
            Self::WorkerDisconnect => "worker_disconnect",
            Self::WorkerProtocolError => "worker_protocol_error",
            Self::ConfigurationError => "configuration_error",
            Self::ConfigurationTimeout => "configuration_timeout",
            Self::ClassificationError => "classification_error",
            Self::ClassificationTimeout => "classification_timeout",
            Self::DispatchError => "dispatch_error",
            Self::ConnectError => "connect_error",
            Self::ConnectTimeout => "connect_timeout",
            Self::WorkerSetupError => "worker_setup_error",
            Self::WorkerSetupTimeout => "worker_setup_timeout",
            Self::Draining => "draining",
            Self::ForcedShutdown => "forced_shutdown",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum HttpBodyTermination {
    Complete,
    UpstreamError,
    Dropped,
}

impl HttpBodyTermination {
    pub(crate) const ALL: [Self; 3] = [Self::Complete, Self::UpstreamError, Self::Dropped];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::UpstreamError => "upstream_error",
            Self::Dropped => "dropped",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

impl Rejection {
    pub(crate) const ALL: [Self; 10] = [
        Self::GlobalAdmission,
        Self::GenerationAdmission,
        Self::SpeechAdmission,
        Self::SpeechBatchAdmission,
        Self::TranscriptionAdmission,
        Self::SpeechWebsocketAdmission,
        Self::RealtimeWebsocketAdmission,
        Self::BufferedRequestBytes,
        Self::SpeechWebsocketWorker,
        Self::RealtimeWebsocketWorker,
    ];

    pub(crate) const fn admission(class: CapacityClass) -> Self {
        match class {
            CapacityClass::GenerationHttp => Self::GenerationAdmission,
            CapacityClass::SpeechHttp => Self::SpeechAdmission,
            CapacityClass::SpeechBatch => Self::SpeechBatchAdmission,
            CapacityClass::TranscriptionHttp => Self::TranscriptionAdmission,
            CapacityClass::SpeechWebsocket => Self::SpeechWebsocketAdmission,
            CapacityClass::RealtimeWebsocket => Self::RealtimeWebsocketAdmission,
        }
    }

    pub(crate) const fn worker(class: CapacityClass) -> Option<Self> {
        match class {
            CapacityClass::SpeechWebsocket => Some(Self::SpeechWebsocketWorker),
            CapacityClass::RealtimeWebsocket => Some(Self::RealtimeWebsocketWorker),
            CapacityClass::GenerationHttp
            | CapacityClass::SpeechHttp
            | CapacityClass::SpeechBatch
            | CapacityClass::TranscriptionHttp => None,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::GlobalAdmission => "admission_global",
            Self::GenerationAdmission => "admission_generation_http",
            Self::SpeechAdmission => "admission_speech_http",
            Self::SpeechBatchAdmission => "admission_speech_batch",
            Self::TranscriptionAdmission => "admission_transcription_http",
            Self::SpeechWebsocketAdmission => "admission_speech_websocket",
            Self::RealtimeWebsocketAdmission => "admission_realtime_websocket",
            Self::BufferedRequestBytes => "buffered_request_bytes",
            Self::SpeechWebsocketWorker => "worker_speech_websocket",
            Self::RealtimeWebsocketWorker => "worker_realtime_websocket",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

pub(crate) struct RouterMetrics {
    requests: [AtomicU64; HttpRoute::ALL.len()],
    responses: [[AtomicU64; StatusClass::ALL.len()]; HttpRoute::ALL.len()],
    response_header_durations: [DurationHistogram; HttpRoute::ALL.len()],
    cancelled_before_headers: [AtomicU64; HttpRoute::ALL.len()],
    classification_durations:
        [[DurationHistogram; ClassificationPhase::ALL.len()]; ClassificationKind::ALL.len()],
    classification_outcomes:
        [[AtomicU64; ClassificationOutcome::ALL.len()]; ClassificationKind::ALL.len()],
    websocket_terminations: [[[AtomicU64; WebsocketTermination::ALL.len()];
        WebsocketPhase::ALL.len()]; WebsocketProtocol::ALL.len()],
    http_body_terminations: [AtomicU64; HttpBodyTermination::ALL.len()],
    faults: [[AtomicU64; HttpFault::ALL.len()]; HttpRoute::ALL.len()],
    rejections: [AtomicU64; Rejection::ALL.len()],
    relay_failures: AtomicU64,
}

impl RouterMetrics {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            requests: std::array::from_fn(|_| AtomicU64::new(0)),
            responses: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
            response_header_durations: std::array::from_fn(|_| DurationHistogram::new()),
            cancelled_before_headers: std::array::from_fn(|_| AtomicU64::new(0)),
            classification_durations: std::array::from_fn(|_| {
                std::array::from_fn(|_| DurationHistogram::new())
            }),
            classification_outcomes: std::array::from_fn(|_| {
                std::array::from_fn(|_| AtomicU64::new(0))
            }),
            websocket_terminations: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0)))
            }),
            http_body_terminations: std::array::from_fn(|_| AtomicU64::new(0)),
            faults: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
            rejections: std::array::from_fn(|_| AtomicU64::new(0)),
            relay_failures: AtomicU64::new(0),
        })
    }

    pub(crate) fn record_request(&self, route: HttpRoute) {
        self.requests[route.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_response<B>(&self, route: HttpRoute, response: &Response<B>) {
        let status = StatusClass::from_status(response.status());
        self.responses[route.index()][status.index()].fetch_add(1, Ordering::Relaxed);
        if let Some(fault) = response.extensions().get::<HttpFault>() {
            self.faults[route.index()][fault.index()].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_response_header_duration(&self, route: HttpRoute, duration: Duration) {
        self.response_header_durations[route.index()].observe(duration);
    }

    pub(crate) fn record_cancelled_before_headers(&self, route: HttpRoute) {
        self.cancelled_before_headers[route.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_classification_duration(
        &self,
        kind: ClassificationKind,
        phase: ClassificationPhase,
        duration: Duration,
    ) {
        self.classification_durations[kind.index()][phase.index()].observe(duration);
    }

    pub(crate) fn record_classification_outcome(
        &self,
        kind: ClassificationKind,
        outcome: ClassificationOutcome,
    ) {
        self.classification_outcomes[kind.index()][outcome.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_rejection(&self, rejection: Rejection) {
        self.rejections[rejection.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_websocket_termination(
        &self,
        protocol: WebsocketProtocol,
        phase: WebsocketPhase,
        termination: WebsocketTermination,
    ) {
        self.websocket_terminations[protocol.index()][phase.index()][termination.index()]
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_http_body_termination(&self, termination: HttpBodyTermination) {
        self.http_body_terminations[termination.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_relay_failure(&self) {
        self.relay_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn requests(&self, route: HttpRoute) -> u64 {
        self.requests[route.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn responses(&self, route: HttpRoute, status: StatusClass) -> u64 {
        self.responses[route.index()][status.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn response_header_duration(&self, route: HttpRoute) -> DurationHistogramSnapshot {
        self.response_header_durations[route.index()].snapshot()
    }

    pub(crate) fn cancelled_before_headers(&self, route: HttpRoute) -> u64 {
        self.cancelled_before_headers[route.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn classification_duration(
        &self,
        kind: ClassificationKind,
        phase: ClassificationPhase,
    ) -> DurationHistogramSnapshot {
        self.classification_durations[kind.index()][phase.index()].snapshot()
    }

    pub(crate) fn classification_outcome(
        &self,
        kind: ClassificationKind,
        outcome: ClassificationOutcome,
    ) -> u64 {
        self.classification_outcomes[kind.index()][outcome.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn faults(&self, route: HttpRoute, fault: HttpFault) -> u64 {
        self.faults[route.index()][fault.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn rejections(&self, rejection: Rejection) -> u64 {
        self.rejections[rejection.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn websocket_terminations(
        &self,
        protocol: WebsocketProtocol,
        phase: WebsocketPhase,
        termination: WebsocketTermination,
    ) -> u64 {
        self.websocket_terminations[protocol.index()][phase.index()][termination.index()]
            .load(Ordering::Relaxed)
    }

    pub(crate) fn http_body_terminations(&self, termination: HttpBodyTermination) -> u64 {
        self.http_body_terminations[termination.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn relay_failures(&self) -> u64 {
        self.relay_failures.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::{Response, StatusCode};

    use super::{
        ClassificationKind, ClassificationOutcome, ClassificationPhase, HttpBodyTermination,
        HttpRoute, Rejection, RouterMetrics, StatusClass, WebsocketPhase, WebsocketProtocol,
        WebsocketTermination,
    };
    use crate::error::HttpFault;

    #[test]
    fn paths_use_fixed_route_labels() {
        assert_eq!(HttpRoute::from_path("/live"), HttpRoute::Live);
        assert_eq!(
            HttpRoute::from_path("/v1/audio/voices/alice"),
            HttpRoute::VoiceItem
        );
        assert_eq!(
            HttpRoute::from_path("/v1/audio/voices/alice/extra"),
            HttpRoute::Unknown
        );
        assert_eq!(
            HttpRoute::from_path("/unbounded/client/path"),
            HttpRoute::Unknown
        );
    }

    #[test]
    fn duration_histogram_uses_inclusive_bounds_and_an_overflow_bucket() {
        let histogram = super::DurationHistogram::new();
        histogram.observe(Duration::from_micros(10));
        histogram.observe(Duration::from_micros(11));
        histogram.observe(Duration::from_micros(240_000_001));

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.buckets[0], 1);
        assert_eq!(snapshot.buckets[1], 1);
        assert_eq!(snapshot.buckets[super::DURATION_BUCKETS.len()], 1);
        assert_eq!(snapshot.count(), 3);
    }

    #[test]
    fn counters_have_one_owner_and_fixed_indices() {
        let metrics = RouterMetrics::new();
        metrics.record_request(HttpRoute::Speech);
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        response
            .extensions_mut()
            .insert(HttpFault::RouterOverloaded);
        metrics.record_response(HttpRoute::Speech, &response);
        metrics.record_response_header_duration(HttpRoute::Speech, Duration::from_micros(750));
        metrics.record_cancelled_before_headers(HttpRoute::Chat);
        metrics.record_classification_duration(
            ClassificationKind::Speech,
            ClassificationPhase::Execution,
            Duration::from_micros(750),
        );
        metrics.record_classification_outcome(
            ClassificationKind::Speech,
            ClassificationOutcome::Success,
        );
        metrics.record_websocket_termination(
            WebsocketProtocol::Speech,
            WebsocketPhase::Relay,
            WebsocketTermination::WorkerClose,
        );
        metrics.record_http_body_termination(HttpBodyTermination::Complete);
        metrics.record_rejection(Rejection::SpeechAdmission);
        metrics.record_relay_failure();

        assert_eq!(metrics.requests(HttpRoute::Speech), 1);
        assert_eq!(
            metrics.responses(HttpRoute::Speech, StatusClass::ClientError),
            1
        );
        assert_eq!(
            metrics.faults(HttpRoute::Speech, HttpFault::RouterOverloaded),
            1
        );
        let duration = metrics.response_header_duration(HttpRoute::Speech);
        assert_eq!(duration.count(), 1);
        assert_eq!(duration.sum_micros, 750);
        assert_eq!(duration.buckets[6], 1);
        assert_eq!(metrics.cancelled_before_headers(HttpRoute::Chat), 1);
        let classification = metrics
            .classification_duration(ClassificationKind::Speech, ClassificationPhase::Execution);
        assert_eq!(classification.count(), 1);
        assert_eq!(classification.sum_micros, 750);
        assert_eq!(classification.buckets[6], 1);
        assert_eq!(
            metrics
                .classification_outcome(ClassificationKind::Speech, ClassificationOutcome::Success),
            1
        );
        assert_eq!(
            metrics.websocket_terminations(
                WebsocketProtocol::Speech,
                WebsocketPhase::Relay,
                WebsocketTermination::WorkerClose,
            ),
            1
        );
        assert_eq!(
            metrics.http_body_terminations(HttpBodyTermination::Complete),
            1
        );
        assert_eq!(metrics.rejections(Rejection::SpeechAdmission), 1);
        assert_eq!(metrics.relay_failures(), 1);
    }

    #[test]
    fn metric_enum_tables_match_their_atomic_indices() {
        for (index, route) in HttpRoute::ALL.into_iter().enumerate() {
            assert_eq!(route.index(), index);
        }
        for (index, status) in StatusClass::ALL.into_iter().enumerate() {
            assert_eq!(status.index(), index);
        }
        for (index, rejection) in Rejection::ALL.into_iter().enumerate() {
            assert_eq!(rejection.index(), index);
        }
        for (index, kind) in ClassificationKind::ALL.into_iter().enumerate() {
            assert_eq!(kind.index(), index);
        }
        for (index, phase) in ClassificationPhase::ALL.into_iter().enumerate() {
            assert_eq!(phase.index(), index);
        }
        for (index, outcome) in ClassificationOutcome::ALL.into_iter().enumerate() {
            assert_eq!(outcome.index(), index);
        }
        for (index, protocol) in WebsocketProtocol::ALL.into_iter().enumerate() {
            assert_eq!(protocol.index(), index);
        }
        for (index, phase) in WebsocketPhase::ALL.into_iter().enumerate() {
            assert_eq!(phase.index(), index);
        }
        for (index, termination) in WebsocketTermination::ALL.into_iter().enumerate() {
            assert_eq!(termination.index(), index);
        }
        for (index, termination) in HttpBodyTermination::ALL.into_iter().enumerate() {
            assert_eq!(termination.index(), index);
        }
        for (index, fault) in HttpFault::ALL.into_iter().enumerate() {
            assert_eq!(fault.index(), index);
        }

        let extension_status = match StatusCode::from_u16(600) {
            Ok(status) => status,
            Err(error) => panic!("valid extension status: {error}"),
        };
        assert_eq!(
            StatusClass::from_status(extension_status),
            StatusClass::Other
        );
    }
}
