use std::error::Error as _;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, Query, State};
use axum::http::header::{CONNECTION, UPGRADE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, Version};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer as _};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message as UpstreamMessage;
use tokio_tungstenite::tungstenite::error::CapacityError;
use tracing::error;

use crate::classification;
use crate::config::{Config, WebsocketConfig};
use crate::error::HttpFault;
use crate::request_id::CanonicalRequestId;
use crate::speech_facts::{
    SpeechFields, managed_voice as classify_managed_voice, read_field as read_shared_speech_field,
    reference_forms, response_format as classify_response_format, task as classify_task,
};
use crate::worker_pool::{
    AdmissionError, CapacityClass, DefaultModelResolution, DispatchError, ModelSelection,
    ProfileRequirement, RealtimeProtocol, RouteRequirement, ServiceClass, SpeechResponseFormat,
    StreamMode, TrustDomain, WorkerPool, valid_model_id,
};

mod session;
mod upstream;

pub(crate) use session::SessionTracker;
use session::{DrainState, PendingSession, RelayProtocol, SessionSupervisor, close_message};

pub(crate) const SPEECH_PATH: &str = "/v1/audio/speech/stream";
pub(crate) const REALTIME_PATH: &str = "/v1/realtime";

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
    tracker: SessionTracker,
}

impl WebsocketGateway {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
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
    version: Version,
    uri: Uri,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(trust) = gateway.speech.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let origin = match validate_upgrade(version, &uri, &headers, SPEECH_PATH, &gateway.policy) {
        Ok(origin) => origin,
        Err(()) => return HttpFault::MalformedRequest.into_response(),
    };
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
        .max_frame_size(gateway.policy.frame_max_bytes)
        .max_message_size(gateway.policy.speech_config_max_bytes)
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
        receive_speech_config(&mut downstream, &gateway.policy),
    )
    .await
    {
        Ok(Ok(text)) => text,
        Ok(Err(close)) => {
            drop(pending);
            send_setup_close(&mut downstream, &gateway.policy, &mut drain, close).await;
            return;
        }
        Err(termination) => {
            drop(pending);
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
    let setup_deadline = Instant::now() + gateway.policy.setup_timeout();
    let classify_pool = Arc::clone(&gateway.pool);
    let classify_trust = trust.clone();
    let classified = setup_until(
        &mut drain,
        setup_deadline,
        classification::run(move || {
            let requirement =
                classify_speech(config_text.as_bytes(), &classify_pool, &classify_trust);
            (config_text, requirement)
        }),
    )
    .await;
    let (config_text, requirement) = match classified {
        Ok(Ok((text, Ok(requirement)))) => (text, requirement),
        Ok(Ok((_text, Err(())))) => {
            drop(pending);
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1008, "invalid session.config"),
            )
            .await;
            return;
        }
        Ok(Err(source)) => {
            error!(error = %source, "classification task failed");
            drop(pending);
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
            drop(pending);
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
            drop(pending);
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
    let lease = match gateway.pool.dispatch(admission, &requirement) {
        Ok(lease) => lease,
        Err(error) => {
            drop(registration);
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
    let connected = setup_until(
        &mut drain,
        setup_deadline,
        upstream::connect(
            lease.target(),
            &upstream_path,
            upstream_query.as_deref(),
            &upstream_headers,
            &gateway.policy,
            setup_deadline,
        ),
    )
    .await;
    let upstream = match connected {
        Ok(Ok(upstream)) => upstream,
        Ok(Err(_)) => {
            lease.request_immediate_probe();
            drop(lease);
            drop(registration);
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
            drop(lease);
            drop(registration);
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
    let sent = {
        let upstream = supervisor.upstream_mut();
        setup_until(
            &mut drain,
            setup_deadline,
            upstream.send(UpstreamMessage::Text(upstream_config)),
        )
        .await
    };
    match sent {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            supervisor.request_immediate_probe();
            drop(supervisor);
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
                supervisor.request_immediate_probe();
            }
            drop(supervisor);
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
    }
    let first = {
        let upstream = supervisor.upstream_mut();
        setup_until(
            &mut drain,
            setup_deadline,
            next_worker_application(upstream),
        )
        .await
    };
    let text = match first {
        Ok(Some(Ok(UpstreamMessage::Text(text)))) if is_speech_setup_event(text.as_bytes()) => text,
        Ok(_) => {
            supervisor.request_immediate_probe();
            drop(supervisor);
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
                supervisor.request_immediate_probe();
            }
            drop(supervisor);
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
    let text = match upstream_text_to_downstream(text) {
        Ok(text) => text,
        Err(()) => {
            supervisor.request_immediate_probe();
            drop(supervisor);
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "upstream setup failure"),
            )
            .await;
            return;
        }
    };
    match setup_until(
        &mut drain,
        setup_deadline,
        downstream.send(Message::Text(text)),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            drop(supervisor);
            return;
        }
        Err(termination) => {
            drop(supervisor);
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
    }
    supervisor
        .relay(downstream, &gateway.policy, RelayProtocol::Speech)
        .await;
}

pub(crate) async fn realtime(
    State(gateway): State<Arc<WebsocketGateway>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    version: Version,
    uri: Uri,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(trust) = gateway.realtime.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let origin = match validate_upgrade(version, &uri, &headers, REALTIME_PATH, &gateway.policy) {
        Ok(origin) => origin,
        Err(()) => return HttpFault::MalformedRequest.into_response(),
    };
    let upstream_headers = upstream::HandshakeHeaders::new(request_id.into_header_value(), origin);
    let model = match realtime_model(&uri, || {
        gateway
            .pool
            .resolve_default_model_id(&trust, ServiceClass::RealtimeWebsocket, None)
    }) {
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
    let setup_deadline = Instant::now() + gateway.policy.setup_timeout();
    let requirement = RouteRequirement::new(
        ProfileRequirement::RealtimeWebsocket {
            protocol: RealtimeProtocol::OpenaiRealtimeV1,
            expected_default_model_id: model,
        },
        trust,
    );
    let lease = match gateway.pool.dispatch(admission, &requirement) {
        Ok(lease) => lease,
        Err(error) => {
            drop(registration);
            return dispatch_fault(error).into_response();
        }
    };
    let connected = setup_until(
        &mut drain,
        setup_deadline,
        upstream::connect(
            lease.target(),
            uri.path(),
            uri.query(),
            &upstream_headers,
            &gateway.policy,
            setup_deadline,
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
        .max_frame_size(gateway.policy.frame_max_bytes)
        .max_message_size(gateway.policy.realtime_message_max_bytes)
        .on_upgrade(move |socket| async move {
            run_realtime(socket, gateway, supervisor, setup_deadline).await;
        })
        .into_response()
}

async fn run_realtime(
    mut downstream: WebSocket,
    gateway: Arc<WebsocketGateway>,
    mut supervisor: SessionSupervisor,
    setup_deadline: Instant,
) {
    let mut drain = supervisor.drain_receiver();
    let first = {
        let upstream = supervisor.upstream_mut();
        setup_until(
            &mut drain,
            setup_deadline,
            next_worker_application(upstream),
        )
        .await
    };
    let text = match first {
        Ok(Some(Ok(UpstreamMessage::Text(text))))
            if text.len() <= gateway.policy.worker_message_max_bytes
                && parse_event_kind(text.as_bytes()) == Some(EventKind::SessionCreated) =>
        {
            text
        }
        Ok(_) => {
            supervisor.request_immediate_probe();
            drop(supervisor);
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "invalid session.created"),
            )
            .await;
            return;
        }
        Err(termination) => {
            if termination == SetupTermination::Deadline {
                supervisor.request_immediate_probe();
            }
            drop(supervisor);
            close_for_setup_termination(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                close_message(1011, "invalid session.created"),
            )
            .await;
            return;
        }
    };
    let text = match upstream_text_to_downstream(text) {
        Ok(text) => text,
        Err(()) => {
            supervisor.request_immediate_probe();
            drop(supervisor);
            send_setup_close(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                close_message(1011, "invalid session.created"),
            )
            .await;
            return;
        }
    };
    match setup_until(
        &mut drain,
        setup_deadline,
        downstream.send(Message::Text(text)),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            drop(supervisor);
            return;
        }
        Err(termination) => {
            drop(supervisor);
            close_for_setup_termination(
                &mut downstream,
                &gateway.policy,
                &mut drain,
                termination,
                close_message(1011, "invalid session.created"),
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
    policy: &WebsocketConfig,
) -> Result<axum::extract::ws::Utf8Bytes, Message> {
    match downstream.next().await {
        Some(Ok(Message::Text(text))) if text.len() <= policy.speech_config_max_bytes => Ok(text),
        Some(Ok(Message::Text(_))) => Err(close_message(1009, "session.config too large")),
        Some(Ok(Message::Binary(bytes))) if bytes.len() > policy.speech_config_max_bytes => {
            Err(close_message(1009, "session.config too large"))
        }
        Some(Ok(Message::Binary(_))) => Err(close_message(1008, "invalid session.config")),
        Some(Err(error)) if websocket_message_too_large(&error) => {
            Err(close_message(1009, "session.config too large"))
        }
        _ => Err(close_message(1008, "invalid session.config")),
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

async fn next_worker_application(
    upstream: &mut upstream::UpstreamSocket,
) -> Option<Result<UpstreamMessage, tokio_tungstenite::tungstenite::Error>> {
    loop {
        match upstream.next().await {
            Some(Ok(
                UpstreamMessage::Ping(_) | UpstreamMessage::Pong(_) | UpstreamMessage::Frame(_),
            )) => {}
            other => return other,
        }
    }
}

fn validate_upgrade(
    version: Version,
    uri: &Uri,
    headers: &HeaderMap,
    path: &str,
    policy: &WebsocketConfig,
) -> Result<Option<HeaderValue>, ()> {
    let path_and_query = uri.path_and_query().ok_or(())?;
    if version != Version::HTTP_11
        || uri.path() != path
        || path_and_query.as_str().len() > policy.uri_max_bytes
        || headers.len() > policy.header_max_fields
        || headers.contains_key("sec-websocket-protocol")
        || headers.contains_key("sec-websocket-extensions")
    {
        return Err(());
    }
    if !connection_requests_upgrade(headers)? || !is_exact_websocket_upgrade(headers)? {
        return Err(());
    }
    let total = headers.iter().try_fold(0_usize, |total, (name, value)| {
        total
            .checked_add(name.as_str().len())?
            .checked_add(value.as_bytes().len())
    });
    if total.is_none_or(|total| total > policy.header_max_bytes) {
        return Err(());
    }
    let mut origins = headers.get_all("origin").iter();
    let origin = origins.next().cloned();
    if origins.next().is_some() {
        return Err(());
    }
    if let Some(origin) = origin.as_ref() {
        let origin = origin.to_str().map_err(|_| ())?;
        if origin != "null" {
            let parsed: Uri = origin.parse().map_err(|_| ())?;
            if parsed.scheme().is_none()
                || parsed.authority().is_none()
                || parsed.path() != "/"
                || parsed.query().is_some()
            {
                return Err(());
            }
        }
    }
    Ok(origin)
}

fn connection_requests_upgrade(headers: &HeaderMap) -> Result<bool, ()> {
    let mut found = false;
    let mut count = 0_usize;
    for value in headers.get_all(CONNECTION) {
        let value = value.to_str().map_err(|_| ())?;
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() || HeaderName::from_bytes(token.as_bytes()).is_err() {
                return Err(());
            }
            count = count.checked_add(1).ok_or(())?;
            found |= token.eq_ignore_ascii_case("upgrade");
        }
    }
    Ok(count != 0 && found)
}

fn is_exact_websocket_upgrade(headers: &HeaderMap) -> Result<bool, ()> {
    let mut count = 0_usize;
    for value in headers.get_all(UPGRADE) {
        let value = value.to_str().map_err(|_| ())?;
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty()
                || HeaderName::from_bytes(token.as_bytes()).is_err()
                || !token.eq_ignore_ascii_case("websocket")
            {
                return Err(());
            }
            count = count.checked_add(1).ok_or(())?;
        }
    }
    Ok(count == 1)
}

fn admission_fault(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
        AdmissionError::Draining => HttpFault::RouterUnavailable,
    }
}

fn dispatch_fault(error: DispatchError) -> HttpFault {
    match error {
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
        DispatchError::Internal => close_message(1011, "internal setup failure"),
    }
}

fn classify_speech(
    bytes: &[u8],
    pool: &WorkerPool,
    trust: &TrustDomain,
) -> Result<RouteRequirement, ()> {
    let fields = parse_speech_config(bytes)?;
    let model = match fields.model.clone().flatten() {
        Some(model) if !model.is_empty() => ModelSelection::Explicit(model),
        Some(_) | None => {
            match pool.resolve_default_model_id(trust, ServiceClass::SpeechWebsocket, None) {
                DefaultModelResolution::Unique(model) => ModelSelection::WorkerDefault {
                    expected_model_id: model.to_owned(),
                },
                DefaultModelResolution::NoService | DefaultModelResolution::Ambiguous => {
                    return Err(());
                }
            }
        }
    };
    speech_requirement(fields, model, trust)
}

fn speech_requirement(
    fields: SpeechFields,
    model: ModelSelection,
    trust: &TrustDomain,
) -> Result<RouteRequirement, ()> {
    let format = classify_response_format(
        fields
            .response_format
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("pcm"),
    )
    .ok_or(())?;
    let stream_mode = if fields
        .stream
        .as_ref()
        .and_then(|value| *value)
        .unwrap_or(false)
    {
        if format != SpeechResponseFormat::Pcm {
            return Err(());
        }
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    let task = classify_task(
        fields
            .task
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("Base"),
    )
    .ok_or(())?;
    let references = reference_forms(&fields);
    let managed_voice = classify_managed_voice(&fields, &references);
    Ok(RouteRequirement::new(
        ProfileRequirement::SpeechWebsocket {
            model,
            response_format: format,
            stream_mode,
            task,
            reference_forms: references,
            managed_voice,
        },
        trust.clone(),
    ))
}

fn parse_speech_config(bytes: &[u8]) -> Result<SpeechFields, ()> {
    let source = select_speech_config_source(bytes)?;
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = ConfigEnvelopeSeed(source)
        .deserialize(&mut deserializer)
        .map_err(|_| ())?;
    deserializer.end().map_err(|_| ())?;
    parsed
}

#[derive(Clone, Copy)]
enum SpeechConfigSource {
    Flat,
    Nested,
}

fn select_speech_config_source(bytes: &[u8]) -> Result<SpeechConfigSource, ()> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let envelope = <ConfigSourceEnvelope as serde::Deserialize>::deserialize(&mut deserializer)
        .map_err(|_| ())?;
    deserializer.end().map_err(|_| ())?;
    if envelope.message_type.as_deref() != Some("session.config") {
        return Err(());
    }
    Ok(if envelope.session.is_some() {
        SpeechConfigSource::Nested
    } else {
        SpeechConfigSource::Flat
    })
}

#[derive(serde::Deserialize)]
struct ConfigSourceEnvelope {
    #[serde(rename = "type")]
    message_type: Option<String>,
    session: Option<IgnoredSessionObject>,
}

#[derive(serde::Deserialize)]
struct IgnoredSessionObject {}

struct ConfigEnvelopeSeed(SpeechConfigSource);

impl<'de> DeserializeSeed<'de> for ConfigEnvelopeSeed {
    type Value = Result<SpeechFields, ()>;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(ConfigEnvelopeVisitor(self.0))
    }
}

struct ConfigEnvelopeVisitor(SpeechConfigSource);

impl<'de> Visitor<'de> for ConfigEnvelopeVisitor {
    type Value = Result<SpeechFields, ()>;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a session.config object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut nested = None;
        let mut flat = SpeechFields::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "session" => set_once(
                    &mut nested,
                    map.next_value_seed(NullableSpeechFieldsSeed)?,
                    "session",
                )?,
                _ if matches!(self.0, SpeechConfigSource::Flat) => {
                    read_speech_field(&key, &mut map, &mut flat)?;
                }
                _ => {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
        }
        match self.0 {
            SpeechConfigSource::Flat => Ok(Ok(flat)),
            SpeechConfigSource::Nested => Ok(nested.flatten().ok_or(())),
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
        struct OptionalVisitor;
        impl<'de> Visitor<'de> for OptionalVisitor {
            type Value = Option<SpeechFields>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("null or a session object")
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_map(SpeechFieldsVisitor).map(Some)
            }
        }
        deserializer.deserialize_option(OptionalVisitor)
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
        return set_once(&mut fields.stream, map.next_value()?, "stream_audio");
    }
    if !read_shared_speech_field(key, map, fields)? {
        let _ignored = map.next_value::<IgnoredAny>()?;
    }
    Ok(())
}

fn realtime_model<'a>(
    uri: &Uri,
    resolve_default: impl FnOnce() -> DefaultModelResolution<'a>,
) -> Result<String, HttpFault> {
    validate_query_encoding(uri.query().unwrap_or_default())?;
    let Query(query) =
        Query::<RealtimeQuery>::try_from_uri(uri).map_err(|_| HttpFault::MalformedRequest)?;
    match query.model {
        Some(model) if valid_model_id(&model) => Ok(model),
        Some(_) => Err(HttpFault::MalformedRequest),
        None => match resolve_default() {
            DefaultModelResolution::Unique(model) => Ok(model.to_owned()),
            DefaultModelResolution::Ambiguous => Err(HttpFault::AmbiguousModel),
            DefaultModelResolution::NoService => Err(HttpFault::RouterUnavailable),
        },
    }
}

fn validate_query_encoding(query: &str) -> Result<(), HttpFault> {
    let mut decoded = Vec::with_capacity(query.len());
    for component in query.split('&').flat_map(|pair| pair.splitn(2, '=')) {
        decoded.clear();
        let bytes = component.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'%' {
                let high = bytes.get(index + 1).and_then(|byte| hex_value(*byte));
                let low = bytes.get(index + 2).and_then(|byte| hex_value(*byte));
                let (Some(high), Some(low)) = (high, low) else {
                    return Err(HttpFault::MalformedRequest);
                };
                decoded.push((high << 4) | low);
                index += 3;
            } else {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
        std::str::from_utf8(&decoded).map_err(|_| HttpFault::MalformedRequest)?;
    }
    Ok(())
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
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
                    set_once(&mut event_type, map.next_value::<Option<String>>()?, "type")?;
                } else {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(event_type
                .flatten()
                .map(|event_type| match event_type.as_str() {
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

fn set_once<E, T>(slot: &mut Option<T>, value: T, field: &'static str) -> Result<(), E>
where
    E: de::Error,
{
    if slot.is_some() {
        return Err(E::duplicate_field(field));
    }
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::http::header::{CONNECTION, UPGRADE};
    use axum::http::{HeaderMap, HeaderValue, Uri, Version};
    use tokio::sync::{Semaphore, watch};

    use crate::classification;
    use crate::config::WebsocketConfig;
    use crate::error::HttpFault;
    use crate::worker_pool::{
        DefaultModelResolution, ModelSelection, ProfileRequirement, ReferenceForm,
        SpeechResponseFormat, SpeechTask, StreamMode, TrustDomain,
    };

    use super::{
        DrainState, EventKind, SetupTermination, is_speech_setup_event, parse_event_kind,
        parse_speech_config, realtime_model, setup_until, speech_requirement, validate_upgrade,
        websocket_message_too_large,
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
            classification::run(move || {
                entered_tx.send(()).expect("classifier started");
                release_rx.recv().expect("release classifier");
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
        assert!(parse_speech_config(br#"{"type":"input.text"}"#).is_err());
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
    fn null_session_keeps_flat_fallback_strict_and_last_wins() {
        assert!(
            parse_speech_config(br#"{"type":"session.config","model":7,"session":null}"#).is_err()
        );
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
    fn nested_session_retains_envelope_and_active_field_validation() {
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
        assert!(
            parse_speech_config(br#"{"type":"session.config","session":{},"session":{}}"#).is_err()
        );
        assert!(
            parse_speech_config(
                br#"{"type":"session.config","type":"session.config","session":{}}"#
            )
            .is_err()
        );
        assert!(
            parse_speech_config(br#"{"type":"session.config","session":"not-an-object"}"#).is_err()
        );
    }

    #[test]
    fn speech_requirement_preserves_every_selection_dimension() {
        let fields = parse_speech_config(
            br#"{"type":"session.config","model":"tts","response_format":"pcm","stream_audio":true,"task_type":"CustomVoice","voice":"named","ref_audio":"direct","references":[{"audio":"list"},{"vq_codes":[1]}],"split_granularity":"clause"}"#,
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
            managed_voice,
            ..
        } = requirement.profile()
        else {
            panic!("speech websocket requirement")
        };
        assert_eq!(model.model_id(), "tts");
        assert_eq!(*response_format, SpeechResponseFormat::Pcm);
        assert_eq!(*stream_mode, StreamMode::Streaming);
        assert_eq!(*task, SpeechTask::VoiceClone);
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes
            ]
        );
        assert!(
            !managed_voice,
            "explicit references avoid managed voice routing"
        );

        let encoded_stream = parse_speech_config(
            br#"{"type":"session.config","response_format":"mp3","stream_audio":true}"#,
        )
        .expect("routing fields parse before relationship validation");
        assert!(
            speech_requirement(
                encoded_stream,
                ModelSelection::Explicit(String::from("tts")),
                &TrustDomain::new(String::from("local")),
            )
            .is_err()
        );
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
    fn realtime_query_is_strict_and_default_aware() {
        let explicit = |query: &str| {
            let uri: Uri = format!("/v1/realtime?{query}")
                .parse()
                .expect("valid test URI");
            realtime_model(&uri, || DefaultModelResolution::NoService)
        };
        assert_eq!(explicit("model=omni"), Ok(String::from("omni")));
        assert_eq!(
            explicit("unknown=first&mo%64el=qwen%2FOmni"),
            Ok(String::from("qwen/Omni"))
        );
        let absent: Uri = "/v1/realtime?unknown=retained"
            .parse()
            .expect("valid absent-model URI");
        assert_eq!(
            realtime_model(&absent, || DefaultModelResolution::Unique("common")),
            Ok(String::from("common"))
        );
        for query in ["model=a&model=b", "model=", "model=%", "model=%FF"] {
            assert_eq!(explicit(query), Err(HttpFault::MalformedRequest));
        }
    }

    #[test]
    fn realtime_first_event_is_duplicate_aware() {
        assert_eq!(
            parse_event_kind(br#"{"type":"session.created","session":{}}"#),
            Some(EventKind::SessionCreated)
        );
        assert_eq!(
            parse_event_kind(br#"{"type":"session.created","type":"other"}"#),
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

    #[test]
    fn handshake_tokens_are_exact_not_axum_substrings() {
        let uri: Uri = "/v1/realtime".parse().expect("valid route URI");
        let policy = WebsocketConfig::default();
        let mut headers = HeaderMap::new();
        headers.insert(CONNECTION, HeaderValue::from_static("keep-alive, Upgrade"));
        headers.insert(UPGRADE, HeaderValue::from_static("websocket"));
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_ok()
        );

        let origin = HeaderValue::from_static("https://Client.Example:8443");
        headers.insert("origin", origin.clone());
        let validated = validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy)
            .expect("valid upgrade")
            .expect("validated origin");
        assert_eq!(validated.as_bytes(), origin.as_bytes());

        headers.insert(
            "origin",
            HeaderValue::from_static("https://client.example?tenant=one"),
        );
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_err()
        );
        headers.remove("origin");

        headers.insert(CONNECTION, HeaderValue::from_static("xupgrade"));
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_err()
        );

        headers.insert(CONNECTION, HeaderValue::from_static("upgrade"));
        headers.append(UPGRADE, HeaderValue::from_static("websocket"));
        assert!(
            validate_upgrade(Version::HTTP_11, &uri, &headers, "/v1/realtime", &policy).is_err()
        );
    }
}
