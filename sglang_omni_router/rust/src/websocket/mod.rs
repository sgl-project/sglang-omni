use std::error::Error as _;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, Query, State};
use axum::http::header::ORIGIN;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer as _};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message as UpstreamMessage;
use tokio_tungstenite::tungstenite::error::CapacityError;

use crate::classification::ClassificationExecutor;
use crate::config::{Config, WebsocketConfig};
use crate::error::HttpFault;
use crate::request_id::CanonicalRequestId;
use crate::speech_facts::{
    ScalarFactSeed, SpeechFields, named_voice as classify_named_voice,
    read_field as read_shared_speech_field, read_stream as read_speech_stream, reference_forms,
    response_format as classify_response_format, task as classify_task,
};
use crate::worker_pool::{
    AdmissionError, CapacityClass, DispatchError, ModelSelection, ProfileRequirement,
    RouteRequirement, ServiceClass, SpeechResponseFormat, StreamMode, TrustDomain, WorkerPool,
};

mod session;
mod upstream;

pub(crate) use session::SessionTracker;
use session::{DrainState, PendingSession, RelayProtocol, SessionSupervisor, close_message};

pub(crate) const SPEECH_PATH: &str = "/v1/audio/speech/stream";
pub(crate) const REALTIME_PATH: &str = "/v1/realtime";
const MAX_MESSAGE_BYTES: usize = 16 * 1_024 * 1_024;

#[derive(Deserialize)]
struct RealtimeQuery {
    model: Option<String>,
}

fn downstream_text_to_upstream(
    text: axum::extract::ws::Utf8Bytes,
) -> Result<tokio_tungstenite::tungstenite::Utf8Bytes, ()> {
    let bytes: bytes::Bytes = text.into();
    bytes.try_into().map_err(|_| ())
}

fn upstream_text_to_downstream(
    text: tokio_tungstenite::tungstenite::Utf8Bytes,
) -> Result<axum::extract::ws::Utf8Bytes, ()> {
    let bytes: bytes::Bytes = text.into();
    bytes.try_into().map_err(|_| ())
}

/// Immutable terminating WebSocket route state.
pub(crate) struct WebsocketGateway {
    pool: Arc<WorkerPool>,
    policy: WebsocketConfig,
    speech: Option<TrustDomain>,
    realtime: Option<TrustDomain>,
    classifier: Arc<ClassificationExecutor>,
    tracker: SessionTracker,
}

impl WebsocketGateway {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
        classifier: Arc<ClassificationExecutor>,
        tracker: SessionTracker,
    ) -> Option<Arc<Self>> {
        let policy = config.websocket.clone()?;
        Some(Arc::new(Self {
            pool,
            speech: policy
                .speech
                .as_ref()
                .map(|route| TrustDomain::new(route.trust_domain.clone())),
            realtime: policy
                .realtime
                .as_ref()
                .map(|route| TrustDomain::new(route.trust_domain.clone())),
            classifier,
            policy,
            tracker,
        }))
    }

    pub(crate) const fn speech_enabled(&self) -> bool {
        self.speech.is_some()
    }

    pub(crate) const fn realtime_enabled(&self) -> bool {
        self.realtime.is_some()
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.speech.as_ref().is_none_or(|trust| {
            self.pool
                .service_ready(trust, ServiceClass::SpeechWebsocket)
        }) && self.realtime.as_ref().is_none_or(|trust| {
            self.pool
                .service_ready(trust, ServiceClass::RealtimeWebsocket)
        })
    }
}

pub(crate) async fn speech(
    State(gateway): State<Arc<WebsocketGateway>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    uri: Uri,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(trust) = gateway.speech.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let origin = headers.get(ORIGIN).cloned();
    let upstream_headers = upstream::HandshakeHeaders::new(request_id.into_header_value(), origin);
    let admission = match gateway.pool.try_admit(CapacityClass::SpeechWebsocket, 1) {
        Ok(admission) => admission,
        Err(error) => return admission_fault(error).into_response(),
    };
    let Some(registration) = gateway.tracker.register() else {
        return HttpFault::RouterUnavailable.into_response();
    };
    let pending = PendingSession::new(registration, admission);
    let upstream_path = uri.path().to_owned();
    let upstream_query = uri.query().map(str::to_owned);
    upgrade
        .max_message_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            run_speech(
                socket,
                gateway,
                pending,
                trust,
                upstream_path,
                upstream_query,
                upstream_headers,
            )
            .await;
        })
        .into_response()
}

async fn run_speech(
    mut downstream: WebSocket,
    gateway: Arc<WebsocketGateway>,
    pending: PendingSession,
    trust: TrustDomain,
    upstream_path: String,
    upstream_query: Option<String>,
    upstream_headers: upstream::HandshakeHeaders,
) {
    let mut drain = pending.drain_receiver();
    let config_deadline = Instant::now() + gateway.policy.speech_config_timeout();
    let config_text = match setup_until(
        &mut drain,
        config_deadline,
        receive_speech_config(&mut downstream),
    )
    .await
    {
        Ok(Ok(text)) => text,
        Ok(Err(close)) => {
            send_setup_close(&mut downstream, &gateway.policy, &mut drain, close).await;
            return;
        }
        Err(termination) => {
            close_for_setup_termination(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                close_message(1008, "session.config timeout"),
            )
            .await;
            return;
        }
    };
    let classification_deadline = Instant::now() + gateway.policy.worker_setup_timeout();
    let classify_trust = trust.clone();
    let classified = setup_until(
        &mut drain,
        classification_deadline,
        gateway
            .classifier
            .classify(classification_deadline, move || {
                let requirement = classify_speech(config_text.as_bytes(), &classify_trust);
                Ok((config_text, requirement))
            }),
    )
    .await;
    let (config_text, requirement) = match classified {
        Ok(Ok((text, Ok(requirement)))) => (text, requirement),
        Ok(Ok((_text, Err(())))) => {
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1008, "invalid session.config"),
            )
            .await;
            return;
        }
        Ok(Err(_fault)) => {
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "internal setup failure"),
            )
            .await;
            return;
        }
        Err(termination) => {
            close_for_setup_termination(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                close_message(1011, "internal setup failure"),
            )
            .await;
            return;
        }
    };
    let upstream_config = match downstream_text_to_upstream(config_text) {
        Ok(text) => text,
        Err(()) => {
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "internal setup failure"),
            )
            .await;
            return;
        }
    };
    let (registration, admission) = pending.into_admitted();
    let lease = match gateway.pool.dispatch_session(admission, &requirement) {
        Ok(lease) => lease,
        Err(error) => {
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                dispatch_close(error),
            )
            .await;
            return;
        }
    };
    let connect_deadline = Instant::now() + gateway.policy.connect_timeout();
    let connected = setup_until(
        &mut drain,
        connect_deadline,
        upstream::connect(
            lease.target(),
            &upstream_path,
            upstream_query.as_deref(),
            &upstream_headers,
            connect_deadline,
        ),
    )
    .await;
    let upstream = match connected {
        Ok(Ok(upstream)) => upstream,
        Ok(Err(_)) => {
            lease.request_immediate_probe();
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
        Err(termination) => {
            if termination == SetupTermination::Deadline {
                lease.request_immediate_probe();
            }
            close_for_setup_termination(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
    };
    let mut supervisor = SessionSupervisor::from_admitted(registration, lease, upstream);
    let worker_deadline = Instant::now() + gateway.policy.worker_setup_timeout();
    let sent = {
        let upstream = supervisor.upstream_mut();
        setup_until(
            &mut drain,
            worker_deadline,
            upstream.send(UpstreamMessage::Text(upstream_config)),
        )
        .await
    };
    match sent {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            supervisor.request_immediate_probe();
            supervisor
                .close_setup(
                    &mut downstream,
                    1011,
                    "upstream setup failure",
                    &gateway.policy,
                    &mut drain,
                )
                .await;
            return;
        }
        Err(termination) => {
            close_supervised_setup_termination(
                supervisor,
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                1011,
                "upstream setup failure",
            )
            .await;
            return;
        }
    }
    let Some((next_supervisor, next_downstream, first)) = supervisor
        .wait_for_worker_event(
            downstream,
            worker_deadline,
            &gateway.policy,
            RelayProtocol::Speech,
            "upstream setup failure",
        )
        .await
    else {
        return;
    };
    supervisor = next_supervisor;
    downstream = next_downstream;
    let text = match first {
        Some(Ok(UpstreamMessage::Text(text))) if is_speech_setup_event(text.as_bytes()) => text,
        _ => {
            supervisor.request_immediate_probe();
            supervisor
                .close_setup(
                    &mut downstream,
                    1011,
                    "upstream setup failure",
                    &gateway.policy,
                    &mut drain,
                )
                .await;
            return;
        }
    };
    let text = match upstream_text_to_downstream(text) {
        Ok(text) => text,
        Err(()) => {
            supervisor.request_immediate_probe();
            supervisor
                .close_setup(
                    &mut downstream,
                    1011,
                    "upstream setup failure",
                    &gateway.policy,
                    &mut drain,
                )
                .await;
            return;
        }
    };
    match setup_until(
        &mut drain,
        worker_deadline,
        downstream.send(Message::Text(text)),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            supervisor
                .close_upstream_after_client_loss(&mut downstream, &gateway.policy, &mut drain)
                .await;
            return;
        }
        Err(termination) => {
            close_supervised_setup_termination(
                supervisor,
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                1011,
                "upstream setup failure",
            )
            .await;
            return;
        }
    }
    supervisor
        .relay(downstream, &gateway.policy, RelayProtocol::Speech)
        .await;
}

pub(crate) async fn realtime(
    State(gateway): State<Arc<WebsocketGateway>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    uri: Uri,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(trust) = gateway.realtime.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let origin = headers.get(ORIGIN).cloned();
    let upstream_headers = upstream::HandshakeHeaders::new(request_id.into_header_value(), origin);
    let model = match realtime_model(&uri) {
        Ok(model) => model,
        Err(fault) => return fault.into_response(),
    };
    let admission = match gateway.pool.try_admit(CapacityClass::RealtimeWebsocket, 1) {
        Ok(admission) => admission,
        Err(error) => return admission_fault(error).into_response(),
    };
    let Some(registration) = gateway.tracker.register() else {
        return HttpFault::RouterUnavailable.into_response();
    };
    let mut drain = registration.drain_receiver();
    let requirement = RouteRequirement::new(ProfileRequirement::RealtimeWebsocket { model }, trust);
    let lease = match gateway.pool.dispatch_session(admission, &requirement) {
        Ok(lease) => lease,
        Err(error) => {
            drop(registration);
            return dispatch_fault(error).into_response();
        }
    };
    let connect_deadline = Instant::now() + gateway.policy.connect_timeout();
    let connected = setup_until(
        &mut drain,
        connect_deadline,
        upstream::connect(
            lease.target(),
            uri.path(),
            uri.query(),
            &upstream_headers,
            connect_deadline,
        ),
    )
    .await;
    let upstream = match connected {
        Ok(Ok(upstream)) => upstream,
        Ok(Err(_)) => {
            lease.request_immediate_probe();
            drop(lease);
            drop(registration);
            return HttpFault::RouterUnavailable.into_response();
        }
        Err(termination) => {
            if termination == SetupTermination::Deadline {
                lease.request_immediate_probe();
            }
            drop(lease);
            drop(registration);
            return HttpFault::RouterUnavailable.into_response();
        }
    };
    let supervisor = SessionSupervisor::from_admitted(registration, lease, upstream);
    upgrade
        .max_message_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            run_realtime(socket, gateway, supervisor).await;
        })
        .into_response()
}

async fn run_realtime(
    mut downstream: WebSocket,
    gateway: Arc<WebsocketGateway>,
    mut supervisor: SessionSupervisor,
) {
    let mut drain = supervisor.drain_receiver();
    let worker_deadline = Instant::now() + gateway.policy.worker_setup_timeout();
    let Some((next_supervisor, next_downstream, first)) = supervisor
        .wait_for_worker_event(
            downstream,
            worker_deadline,
            &gateway.policy,
            RelayProtocol::Realtime,
            "invalid session.created",
        )
        .await
    else {
        return;
    };
    supervisor = next_supervisor;
    downstream = next_downstream;
    let text = match first {
        Some(Ok(UpstreamMessage::Text(text)))
            if parse_event_kind(text.as_bytes()) == Some(EventKind::SessionCreated) =>
        {
            text
        }
        _ => {
            supervisor.request_immediate_probe();
            supervisor
                .close_setup(
                    &mut downstream,
                    1011,
                    "invalid session.created",
                    &gateway.policy,
                    &mut drain,
                )
                .await;
            return;
        }
    };
    let text = match upstream_text_to_downstream(text) {
        Ok(text) => text,
        Err(()) => {
            supervisor.request_immediate_probe();
            supervisor
                .close_setup(
                    &mut downstream,
                    1011,
                    "invalid session.created",
                    &gateway.policy,
                    &mut drain,
                )
                .await;
            return;
        }
    };
    match setup_until(
        &mut drain,
        worker_deadline,
        downstream.send(Message::Text(text)),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            supervisor
                .close_upstream_after_client_loss(&mut downstream, &gateway.policy, &mut drain)
                .await;
            return;
        }
        Err(termination) => {
            close_supervised_setup_termination(
                supervisor,
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                1011,
                "invalid session.created",
            )
            .await;
            return;
        }
    }
    supervisor
        .relay(downstream, &gateway.policy, RelayProtocol::Realtime)
        .await;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SetupTermination {
    Deadline,
    Drain(DrainState),
}

async fn setup_until<T>(
    drain: &mut watch::Receiver<DrainState>,
    deadline: Instant,
    operation: impl Future<Output = T>,
) -> Result<T, SetupTermination> {
    let initial = *drain.borrow();
    if initial != DrainState::Serving {
        return Err(SetupTermination::Drain(initial));
    }
    if Instant::now() >= deadline {
        return Err(SetupTermination::Deadline);
    }
    tokio::pin!(operation);
    tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline) => Err(SetupTermination::Deadline),
        result = &mut operation => Ok(result),
        changed = drain.changed() => {
            if changed.is_err() {
                Err(SetupTermination::Drain(DrainState::Forced))
            } else {
                Err(SetupTermination::Drain(*drain.borrow()))
            }
        }
    }
}

async fn close_for_setup_termination(
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
    drain: &mut watch::Receiver<DrainState>,
    termination: SetupTermination,
    timeout_close: Message,
) {
    match termination {
        SetupTermination::Deadline => {
            send_setup_close(downstream, policy, drain, timeout_close).await;
        }
        SetupTermination::Drain(state) => {
            close_setup_for_drain(downstream, policy, drain, state).await;
        }
    }
}

async fn close_supervised_setup_termination(
    supervisor: SessionSupervisor,
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
    drain: &mut watch::Receiver<DrainState>,
    termination: SetupTermination,
    timeout_code: u16,
    timeout_reason: &'static str,
) {
    match termination {
        SetupTermination::Deadline => {
            supervisor
                .close_setup(downstream, timeout_code, timeout_reason, policy, drain)
                .await;
        }
        SetupTermination::Drain(DrainState::Draining) => {
            supervisor
                .close_setup(downstream, 1012, "service restart", policy, drain)
                .await;
        }
        SetupTermination::Drain(DrainState::Serving | DrainState::Forced) => {}
    }
}

async fn close_setup_for_drain(
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
    drain: &mut watch::Receiver<DrainState>,
    state: DrainState,
) {
    if state == DrainState::Draining {
        send_setup_close(
            downstream,
            policy,
            drain,
            close_message(1012, "service restart"),
        )
        .await;
    }
}

async fn send_setup_close(
    downstream: &mut WebSocket,
    policy: &WebsocketConfig,
    drain: &mut watch::Receiver<DrainState>,
    message: Message,
) {
    tokio::select! {
        _ = tokio::time::timeout(policy.close_timeout(), downstream.send(message)) => {}
        _ = drain.wait_for(|state| *state == DrainState::Forced) => {}
    }
}

async fn receive_speech_config(
    downstream: &mut WebSocket,
) -> Result<axum::extract::ws::Utf8Bytes, Message> {
    loop {
        match downstream.next().await {
            Some(Ok(Message::Text(text))) => return Ok(text),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
            Some(Ok(Message::Binary(_) | Message::Close(_))) => {
                return Err(close_message(1008, "invalid session.config"));
            }
            Some(Err(error)) if websocket_message_too_large(&error) => {
                return Err(close_message(1009, "session.config too large"));
            }
            Some(Err(_)) | None => return Err(close_message(1008, "invalid session.config")),
        }
    }
}

fn websocket_message_too_large(error: &axum::Error) -> bool {
    error
        .source()
        .and_then(|source| source.downcast_ref::<tokio_tungstenite::tungstenite::Error>())
        .is_some_and(|error| {
            matches!(
                error,
                tokio_tungstenite::tungstenite::Error::Capacity(
                    CapacityError::MessageTooLong { .. }
                )
            )
        })
}

fn admission_fault(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
        AdmissionError::Draining => HttpFault::RouterUnavailable,
    }
}

fn dispatch_fault(error: DispatchError) -> HttpFault {
    match error {
        DispatchError::AmbiguousModel => HttpFault::AmbiguousModel,
        DispatchError::NoEligibleProfile => HttpFault::NoCompatibleWorker,
        DispatchError::Overloaded => HttpFault::RouterOverloaded,
        DispatchError::Unavailable => HttpFault::RouterUnavailable,
        DispatchError::Internal => HttpFault::InternalError,
    }
}

fn dispatch_close(error: DispatchError) -> Message {
    match error {
        DispatchError::Overloaded | DispatchError::Unavailable => {
            close_message(1013, "route unavailable")
        }
        DispatchError::NoEligibleProfile => close_message(1008, "no compatible worker"),
        DispatchError::AmbiguousModel => close_message(1008, "explicit model required"),
        DispatchError::Internal => close_message(1011, "internal setup failure"),
    }
}

fn classify_speech(bytes: &[u8], trust: &TrustDomain) -> Result<RouteRequirement, ()> {
    let fields = parse_speech_config(bytes)?;
    let model = match fields.model.clone().flatten() {
        Some(model) if !model.is_empty() => ModelSelection::Explicit(model),
        Some(_) | None => ModelSelection::UnresolvedDefault,
    };
    speech_requirement(fields, model, trust)
}

fn speech_requirement(
    fields: SpeechFields,
    model: ModelSelection,
    trust: &TrustDomain,
) -> Result<RouteRequirement, ()> {
    let mut format = classify_response_format(
        fields
            .response_format
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("pcm"),
    );
    let stream_mode = if fields
        .stream
        .as_ref()
        .and_then(|value| *value)
        .unwrap_or(false)
    {
        if format != Some(SpeechResponseFormat::Pcm) {
            format = None;
        }
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    let task = fields
        .task
        .as_ref()
        .and_then(Option::as_deref)
        .and_then(classify_task);
    let mut references = reference_forms(&fields);
    let named_voice = classify_named_voice(&fields, &references);
    if named_voice {
        references.clear();
    }
    Ok(RouteRequirement::new(
        ProfileRequirement::SpeechWebsocket {
            model,
            response_format: format,
            stream_mode,
            task,
            reference_forms: references,
            named_voice,
        },
        trust.clone(),
    ))
}

fn parse_speech_config(bytes: &[u8]) -> Result<SpeechFields, ()> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = ConfigEnvelopeSeed
        .deserialize(&mut deserializer)
        .map_err(|_| ())?;
    deserializer.end().map_err(|_| ())?;
    Ok(parsed)
}

struct ConfigEnvelopeSeed;

impl<'de> DeserializeSeed<'de> for ConfigEnvelopeSeed {
    type Value = SpeechFields;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(ConfigEnvelopeVisitor)
    }
}

struct ConfigEnvelopeVisitor;

impl<'de> Visitor<'de> for ConfigEnvelopeVisitor {
    type Value = SpeechFields;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a session.config object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut nested_seen = false;
        let mut nested = None;
        let mut flat = SpeechFields::default();
        while let Some(key) = map.next_key::<String>()? {
            if key == "session" {
                nested_seen = true;
                nested = map.next_value_seed(NullableSpeechFieldsSeed)?;
            } else {
                read_speech_field(&key, &mut map, &mut flat)?;
            }
        }
        if nested_seen {
            Ok(nested.unwrap_or(flat))
        } else {
            Ok(flat)
        }
    }
}

struct NullableSpeechFieldsSeed;

impl<'de> DeserializeSeed<'de> for NullableSpeechFieldsSeed {
    type Value = Option<SpeechFields>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct NullableSpeechFieldsVisitor;

        impl<'de> Visitor<'de> for NullableSpeechFieldsVisitor {
            type Value = Option<SpeechFields>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech session object or worker-owned value")
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                SpeechFieldsVisitor.visit_map(map).map(Some)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<IgnoredAny>()?.is_some() {}
                Ok(None)
            }
        }

        deserializer.deserialize_any(NullableSpeechFieldsVisitor)
    }
}

struct SpeechFieldsVisitor;

impl<'de> Visitor<'de> for SpeechFieldsVisitor {
    type Value = SpeechFields;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a speech session object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = SpeechFields::default();
        while let Some(key) = map.next_key::<String>()? {
            read_speech_field(&key, &mut map, &mut fields)?;
        }
        Ok(fields)
    }
}

fn read_speech_field<'de, A>(
    key: &str,
    map: &mut A,
    fields: &mut SpeechFields,
) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
{
    if key == "stream_audio" {
        return read_speech_stream(map, fields);
    }
    if !read_shared_speech_field(key, map, fields)? {
        let _ignored = map.next_value::<IgnoredAny>()?;
    }
    Ok(())
}

fn realtime_model(uri: &Uri) -> Result<Option<String>, HttpFault> {
    let Query(query) =
        Query::<RealtimeQuery>::try_from_uri(uri).map_err(|_| HttpFault::MalformedRequest)?;
    Ok(query.model.filter(|model| !model.is_empty()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EventKind {
    SessionCreated,
    SessionConfigured,
    Error,
    Other,
}

fn parse_event_kind(bytes: &[u8]) -> Option<EventKind> {
    struct EventVisitor;
    impl<'de> Visitor<'de> for EventVisitor {
        type Value = Option<EventKind>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a realtime event object")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut event_type = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "type" {
                    event_type = map.next_value_seed(ScalarFactSeed)?.into_string();
                } else {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
            let Some(event_type) = event_type else {
                return Ok(None);
            };
            Ok(Some(match event_type.as_str() {
                "session.created" => EventKind::SessionCreated,
                "session.configured" => EventKind::SessionConfigured,
                "error" => EventKind::Error,
                _ => EventKind::Other,
            }))
        }
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let event = deserializer.deserialize_map(EventVisitor).ok()?;
    deserializer.end().ok()?;
    event
}

fn is_speech_setup_event(bytes: &[u8]) -> bool {
    matches!(
        parse_event_kind(bytes),
        Some(EventKind::SessionConfigured | EventKind::Error)
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::time::Duration;

    use axum::http::Uri;
    use tokio::sync::watch;

    use crate::classification::ClassificationExecutor;
    use crate::worker_pool::{
        ModelSelection, ProfileRequirement, ReferenceForm, SpeechResponseFormat, SpeechTask,
        StreamMode, TrustDomain,
    };

    use super::{
        DrainState, EventKind, MAX_MESSAGE_BYTES, SetupTermination, is_speech_setup_event,
        parse_event_kind, parse_speech_config, realtime_model, reference_forms, setup_until,
        speech_requirement, websocket_message_too_large,
    };

    #[tokio::test]
    async fn speech_setup_deadline_bounds_blocking_classification() {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let (_drain_sender, mut drain) = watch::channel(DrainState::Serving);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(25);

        let result = setup_until(
            &mut drain,
            deadline,
            ClassificationExecutor::for_test(1).classify(deadline, move || {
                entered_tx.send(()).expect("classifier started");
                release_rx.recv().expect("release classifier");
                Ok(())
            }),
        )
        .await;

        entered_rx.await.expect("classifier entered");
        assert!(matches!(result, Err(SetupTermination::Deadline)));
        release_tx.send(()).expect("release classifier");
    }

    #[test]
    fn speech_config_accepts_flat_and_nested_and_uses_last_routing_value() {
        assert!(parse_speech_config(br#"{"type":"session.config","model":"tts"}"#).is_ok());
        assert!(
            parse_speech_config(br#"{"type":"session.config","session":{"model":"tts"}}"#).is_ok()
        );
        let fields = parse_speech_config(br#"{"type":"session.config","model":"a","model":"b"}"#);
        assert_eq!(
            fields
                .expect("duplicate routing fields use the last value")
                .model
                .flatten()
                .as_deref(),
            Some("b")
        );
        assert!(parse_speech_config(br#"{"type":"input.text"}"#).is_ok());
    }

    #[test]
    fn nested_speech_config_ignores_all_flat_fallback_fields() {
        for bytes in [
            br#"{"type":"session.config","model":7,"model":{"ignored":true},"session":{"model":"nested"}}"#
                .as_slice(),
            br#"{"type":"session.config","session":{"model":"nested"},"stream_audio":"invalid","stream_audio":false}"#
                .as_slice(),
        ] {
            let fields = parse_speech_config(bytes).expect("nested session is authoritative");
            assert_eq!(
                fields.model.as_ref().and_then(Option::as_deref),
                Some("nested")
            );
        }
    }

    #[test]
    fn absent_nested_facts_keep_tolerant_flat_fallback_and_last_wins() {
        let invalid_model =
            parse_speech_config(br#"{"type":"session.config","model":7,"session":null}"#)
                .expect("worker-owned invalid model remains routable");
        assert_eq!(invalid_model.model, Some(None));
        let fields = parse_speech_config(
            br#"{"type":"session.config","session":null,"model":"a","model":"b"}"#,
        );
        assert_eq!(
            fields
                .expect("flat routing fields use the last value")
                .model
                .flatten()
                .as_deref(),
            Some("b")
        );
    }

    #[test]
    fn duplicate_session_and_event_fields_use_the_last_value() {
        let fields = parse_speech_config(
            br#"{"type":"session.config","session":{"model":"a","model":"b"}}"#,
        );
        assert_eq!(
            fields
                .expect("nested routing fields use the last value")
                .model
                .flatten()
                .as_deref(),
            Some("b")
        );
        let duplicate_session = parse_speech_config(
            br#"{"type":"session.config","session":{},"session":{"model":"last"}}"#,
        );
        assert_eq!(
            duplicate_session
                .expect("last session object is authoritative")
                .model
                .flatten()
                .as_deref(),
            Some("last")
        );
        let malformed_then_valid = parse_speech_config(
            br#"{"type":"session.config","session":"invalid","session":{"model":7,"model":"last"}}"#,
        );
        assert_eq!(
            malformed_then_valid
                .expect("only the last session and nested routing value is authoritative")
                .model
                .flatten()
                .as_deref(),
            Some("last")
        );
        assert!(
            parse_speech_config(br#"{"type":"input.text","type":"session.config","session":{}}"#)
                .is_ok()
        );
        assert!(
            parse_speech_config(
                br#"{"type":"session.config","model":"flat","session":"not-an-object"}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn near_limit_reference_audio_retains_only_routing_facts() {
        let audio = "A".repeat(MAX_MESSAGE_BYTES - 1_024);
        let direct = format!(
            r#"{{"type":"session.config","session":{{"model":"tts","ref_audio":"{audio}"}}}}"#
        );
        let direct = parse_speech_config(direct.as_bytes()).expect("near-limit direct reference");
        assert_eq!(reference_forms(&direct), &[ReferenceForm::Direct]);

        let listed = format!(
            r#"{{"type":"session.config","session":{{"model":"tts","references":[{{"audio":"{audio}"}}]}}}}"#
        );
        let listed = parse_speech_config(listed.as_bytes()).expect("near-limit list reference");
        assert_eq!(reference_forms(&listed), &[ReferenceForm::List]);
    }

    #[test]
    fn speech_requirement_preserves_every_selection_dimension() {
        let fields = parse_speech_config(
            br#"{"type":"session.config","model":"tts","response_format":"pcm","stream_audio":true,"task_type":"Base","voice":"named","ref_audio":"direct","references":[{"audio":"list"},{"vq_codes":[1]}],"split_granularity":"clause"}"#,
        )
        .expect("valid mixed speech configuration");
        let requirement = speech_requirement(
            fields,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("valid speech requirement");
        let ProfileRequirement::SpeechWebsocket {
            model,
            response_format,
            stream_mode,
            task,
            reference_forms,
            named_voice,
            ..
        } = requirement.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!(model.expected_model_id(), Some("tts"));
        assert_eq!(*response_format, Some(SpeechResponseFormat::Pcm));
        assert_eq!(*stream_mode, StreamMode::Streaming);
        assert_eq!(*task, Some(SpeechTask::VoiceClone));
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes
            ]
        );
        assert!(
            !named_voice,
            "explicit references avoid named voice routing"
        );

        let encoded_stream = parse_speech_config(
            br#"{"type":"session.config","response_format":"mp3","stream_audio":true}"#,
        )
        .expect("routing fields parse before relationship validation");
        let encoded_stream = speech_requirement(
            encoded_stream,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("worker owns unsupported response validation");
        let ProfileRequirement::SpeechWebsocket {
            response_format,
            task,
            ..
        } = encoded_stream.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!(*response_format, None);
        assert_eq!(*task, None);

        let base = parse_speech_config(
            br#"{"type":"session.config","task_type":"Base","ref_audio":"reference"}"#,
        )
        .expect("Base speech configuration");
        let base = speech_requirement(
            base,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("Base speech requirement");
        let ProfileRequirement::SpeechWebsocket { task, .. } = base.profile() else {
            panic!("speech websocket requirement")
        };
        assert_eq!(*task, Some(SpeechTask::VoiceClone));

        let custom = parse_speech_config(
            br#"{"type":"session.config","task_type":"CustomVoice","voice":"Vivian"}"#,
        )
        .expect("CustomVoice speech configuration");
        let custom = speech_requirement(
            custom,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("CustomVoice speech requirement");
        let ProfileRequirement::SpeechWebsocket { task, .. } = custom.profile() else {
            panic!("speech websocket requirement")
        };
        assert_eq!(*task, Some(SpeechTask::TextToSpeech));

        let unknown = parse_speech_config(
            br#"{"type":"session.config","response_format":"future","task_type":"future"}"#,
        )
        .expect("unknown worker-owned values remain routable");
        let unknown = speech_requirement(
            unknown,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("worker owns unknown enum validation");
        let ProfileRequirement::SpeechWebsocket {
            response_format,
            task,
            ..
        } = unknown.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!((*response_format, *task), (None, None));
    }

    #[test]
    fn speech_reference_aliases_follow_direct_omni_precedence_without_rejection() {
        let fields = parse_speech_config(
            br#"{"type":"session.config","references":[{"audio_path":null,"ref_audio":"first","audio":"second","data":"third"}]}"#,
        )
        .expect("valid direct reference aliases");
        let requirement = speech_requirement(
            fields,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("valid speech requirement");
        let ProfileRequirement::SpeechWebsocket {
            reference_forms, ..
        } = requirement.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!(reference_forms, &[ReferenceForm::List]);
    }

    #[test]
    fn named_speech_config_uses_voice_as_its_reference_requirement() {
        let fields =
            parse_speech_config(br#"{"type":"session.config","model":"tts","voice":"named"}"#)
                .expect("valid named speech configuration");
        let requirement = speech_requirement(
            fields,
            ModelSelection::Explicit(String::from("tts")),
            &TrustDomain::new(String::from("local")),
        )
        .expect("valid speech requirement");
        let ProfileRequirement::SpeechWebsocket {
            reference_forms,
            named_voice,
            ..
        } = requirement.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert!(reference_forms.is_empty());
        assert!(*named_voice);
    }

    #[test]
    fn realtime_query_is_an_optional_worker_requirement() {
        let explicit = |query: &str| {
            let uri: Uri = format!("/v1/realtime?{query}")
                .parse()
                .expect("valid test URI");
            realtime_model(&uri)
        };
        assert_eq!(explicit("model=omni"), Ok(Some(String::from("omni"))));
        assert_eq!(
            explicit("unknown=first&mo%64el=qwen%2FOmni"),
            Ok(Some(String::from("qwen/Omni")))
        );
        let absent: Uri = "/v1/realtime?unknown=retained"
            .parse()
            .expect("valid absent-model URI");
        assert_eq!(realtime_model(&absent), Ok(None));
        assert_eq!(explicit("model=%"), Ok(Some(String::from("%"))));
        assert_eq!(explicit("model=%FF"), Ok(Some(String::from("�"))));
        assert_eq!(explicit("model="), Ok(None));
        assert!(explicit("model=a&model=b").is_err());
    }

    #[test]
    fn realtime_first_event_is_duplicate_aware() {
        assert_eq!(
            parse_event_kind(br#"{"type":"session.created","session":{}}"#),
            Some(EventKind::SessionCreated)
        );
        assert_eq!(
            parse_event_kind(br#"{"type":"session.created","type":"other"}"#),
            Some(EventKind::Other)
        );
        assert_eq!(
            parse_event_kind(br#"{"type":7,"type":"session.created"}"#),
            Some(EventKind::SessionCreated)
        );
        assert_eq!(
            parse_event_kind(br#"{"type":"session.created","type":7}"#),
            None
        );
        assert_eq!(
            parse_event_kind(br#"{"type":"session.updated"}"#),
            Some(EventKind::Other)
        );
        assert!(is_speech_setup_event(
            br#"{"type":"error","message":"worker validation"}"#
        ));
        assert!(!is_speech_setup_event(br#"{"type":"audio.start"}"#));
    }

    #[test]
    fn axum_capacity_error_is_recognized_as_message_too_large() {
        let error = axum::Error::new(tokio_tungstenite::tungstenite::Error::Capacity(
            tokio_tungstenite::tungstenite::error::CapacityError::MessageTooLong {
                size: 11,
                max_size: 10,
            },
        ));
        assert!(websocket_message_too_large(&error));
    }
}
