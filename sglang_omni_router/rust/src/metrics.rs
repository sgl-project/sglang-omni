use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::http::{Response, StatusCode};

use crate::error::HttpFault;
use crate::worker_pool::CapacityClass;

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
    faults: [[AtomicU64; HttpFault::ALL.len()]; HttpRoute::ALL.len()],
    rejections: [AtomicU64; Rejection::ALL.len()],
    relay_failures: AtomicU64,
}

impl RouterMetrics {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            requests: std::array::from_fn(|_| AtomicU64::new(0)),
            responses: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
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

    pub(crate) fn record_rejection(&self, rejection: Rejection) {
        self.rejections[rejection.index()].fetch_add(1, Ordering::Relaxed);
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

    pub(crate) fn faults(&self, route: HttpRoute, fault: HttpFault) -> u64 {
        self.faults[route.index()][fault.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn rejections(&self, rejection: Rejection) -> u64 {
        self.rejections[rejection.index()].load(Ordering::Relaxed)
    }

    pub(crate) fn relay_failures(&self) -> u64 {
        self.relay_failures.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use axum::body::Body;
    use axum::http::{Response, StatusCode};

    use super::{HttpRoute, Rejection, RouterMetrics, StatusClass};
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
    fn counters_have_one_owner_and_fixed_indices() {
        let metrics = RouterMetrics::new();
        metrics.record_request(HttpRoute::Speech);
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        response
            .extensions_mut()
            .insert(HttpFault::RouterOverloaded);
        metrics.record_response(HttpRoute::Speech, &response);
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
