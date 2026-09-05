use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::ws::{
    CloseCode, CloseFrame as DownstreamClose, Message as DownstreamMessage, WebSocket,
};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Error as UpstreamError;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode as UpstreamCloseCode;
use tokio_tungstenite::tungstenite::protocol::{
    CloseFrame as UpstreamClose, Message as UpstreamMessage,
};

use crate::config::WebsocketConfig;
use crate::worker_pool::{AdmissionLease, RequestLease};

use super::upstream::UpstreamSocket;

type DownstreamSink = SplitSink<WebSocket, DownstreamMessage>;
type DownstreamStream = SplitStream<WebSocket>;
type UpstreamSink = SplitSink<UpstreamSocket, UpstreamMessage>;
type UpstreamStream = SplitStream<UpstreamSocket>;
pub(super) type WorkerEvent = Option<Result<UpstreamMessage, UpstreamError>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DrainState {
    Serving,
    Draining,
    Forced,
}

enum SetupOutcome {
    Worker(WorkerEvent),
    Terminal(RelayTerminal),
}

struct TrackerState {
    draining: bool,
    active: usize,
}

struct TrackerInner {
    state: Mutex<TrackerState>,
    empty: Notify,
    drain: watch::Sender<DrainState>,
}

impl TrackerInner {
    fn state(&self) -> MutexGuard<'_, TrackerState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// Drain-visible ownership for upgraded WebSocket callbacks.
#[derive(Clone)]
pub(crate) struct SessionTracker {
    inner: Arc<TrackerInner>,
}

pub(super) struct SessionRegistration {
    tracker: SessionTracker,
    drain: watch::Receiver<DrainState>,
}

impl SessionTracker {
    pub(crate) fn new() -> Self {
        let (drain, _receiver) = watch::channel(DrainState::Serving);
        Self {
            inner: Arc::new(TrackerInner {
                state: Mutex::new(TrackerState {
                    draining: false,
                    active: 0,
                }),
                empty: Notify::new(),
                drain,
            }),
        }
    }

    pub(super) fn register(&self) -> Option<SessionRegistration> {
        let mut state = self.inner.state();
        if state.draining {
            return None;
        }
        state.active += 1;
        Some(SessionRegistration {
            tracker: self.clone(),
            drain: self.inner.drain.subscribe(),
        })
    }

    pub(crate) fn drain(&self) {
        self.inner.state().draining = true;
        self.inner.drain.send_replace(DrainState::Draining);
    }

    pub(crate) fn force(&self) {
        self.inner.state().draining = true;
        self.inner.drain.send_replace(DrainState::Forced);
    }

    pub(crate) async fn wait_empty(&self) {
        loop {
            let notified = self.inner.empty.notified();
            if self.inner.state().active == 0 {
                return;
            }
            notified.await;
        }
    }
}

impl Drop for SessionRegistration {
    fn drop(&mut self) {
        let mut state = self.tracker.inner.state();
        state.active -= 1;
        if state.active == 0 {
            self.tracker.inner.empty.notify_waiters();
        }
    }
}

impl SessionRegistration {
    pub(super) fn drain_receiver(&self) -> watch::Receiver<DrainState> {
        self.drain.clone()
    }
}

pub(super) struct PendingSession {
    registration: SessionRegistration,
    admission: AdmissionLease,
}

impl PendingSession {
    pub(super) fn new(registration: SessionRegistration, admission: AdmissionLease) -> Self {
        Self {
            registration,
            admission,
        }
    }

    pub(super) fn into_admitted(self) -> (SessionRegistration, AdmissionLease) {
        (self.registration, self.admission)
    }

    pub(super) fn drain_receiver(&self) -> watch::Receiver<DrainState> {
        self.registration.drain_receiver()
    }
}

/// Holds session ownership until relay or bounded setup close work is complete.
pub(super) struct SessionSupervisor {
    registration: SessionRegistration,
    lease: RequestLease,
    upstream: UpstreamSocket,
}

enum RelayTerminal {
    ClientClose(Option<DownstreamClose>),
    WorkerClose(Option<UpstreamClose>),
    ClientGone,
    WorkerGone,
    ClientViolation {
        code: CloseCode,
        reason: &'static str,
    },
    WorkerViolation {
        code: CloseCode,
        reason: &'static str,
    },
    Draining,
    Forced,
    SetupDeadline {
        code: CloseCode,
        reason: &'static str,
    },
}

#[derive(Clone, Copy)]
pub(super) enum RelayProtocol {
    Speech,
    Realtime,
}

impl RelayProtocol {
    const fn accepts_client_binary(self) -> bool {
        matches!(self, Self::Speech)
    }

    const fn accepts_worker_binary(self) -> bool {
        matches!(self, Self::Speech)
    }
}

#[derive(Clone, Copy)]
enum ClosePlan {
    Both,
    DownstreamOnly,
    UpstreamOnly,
}

impl SessionSupervisor {
    pub(super) fn from_admitted(
        registration: SessionRegistration,
        lease: RequestLease,
        upstream: UpstreamSocket,
    ) -> Self {
        Self {
            registration,
            lease,
            upstream,
        }
    }

    pub(super) fn upstream_mut(&mut self) -> &mut UpstreamSocket {
        &mut self.upstream
    }

    pub(super) fn drain_receiver(&self) -> watch::Receiver<DrainState> {
        self.registration.drain_receiver()
    }

    pub(super) fn request_immediate_probe(&self) {
        self.lease.request_immediate_probe();
    }

    pub(super) async fn close_setup(
        mut self,
        downstream: &mut WebSocket,
        code: CloseCode,
        reason: &'static str,
        policy: &WebsocketConfig,
        drain: &mut watch::Receiver<DrainState>,
    ) {
        terminate_both(
            downstream,
            &mut self.upstream,
            code,
            reason,
            policy.close_timeout(),
            drain,
        )
        .await;
    }

    pub(super) async fn close_upstream_after_client_loss(
        mut self,
        downstream: &mut WebSocket,
        policy: &WebsocketConfig,
        drain: &mut watch::Receiver<DrainState>,
    ) {
        close_upstream(
            downstream,
            &mut self.upstream,
            policy.close_timeout(),
            drain,
        )
        .await;
    }

    pub(super) async fn relay(
        self,
        downstream: WebSocket,
        policy: &WebsocketConfig,
        protocol: RelayProtocol,
    ) {
        let Self {
            registration,
            lease,
            upstream,
        } = self;
        let mut drain = registration.drain.clone();
        let (mut downstream_sink, mut downstream_stream) = downstream.split();
        let (mut upstream_sink, mut upstream_stream) = upstream.split();

        let initial_drain = *drain.borrow();
        let terminal = if initial_drain == DrainState::Forced {
            RelayTerminal::Forced
        } else if initial_drain == DrainState::Draining {
            RelayTerminal::Draining
        } else {
            let client_to_worker =
                client_to_worker(&mut downstream_stream, &mut upstream_sink, protocol);
            let worker_to_client =
                worker_to_client(&mut upstream_stream, &mut downstream_sink, protocol);
            tokio::pin!(client_to_worker, worker_to_client);
            tokio::select! {
                biased;
                terminal = &mut client_to_worker => terminal,
                terminal = &mut worker_to_client => terminal,
                changed = drain.changed() => {
                    if changed.is_err() || *drain.borrow() == DrainState::Forced {
                        RelayTerminal::Forced
                    } else {
                        RelayTerminal::Draining
                    }
                }
            }
        };

        let Ok(mut downstream) = downstream_sink.reunite(downstream_stream) else {
            return;
        };
        let Ok(upstream) = upstream_sink.reunite(upstream_stream) else {
            return;
        };

        Self {
            registration,
            lease,
            upstream,
        }
        .finish_terminal(&mut downstream, terminal, policy, &mut drain)
        .await;
    }

    pub(super) async fn wait_for_worker_event(
        self,
        downstream: WebSocket,
        deadline: Instant,
        policy: &WebsocketConfig,
        protocol: RelayProtocol,
        timeout_reason: &'static str,
    ) -> Option<(Self, WebSocket, WorkerEvent)> {
        let Self {
            registration,
            lease,
            upstream,
        } = self;
        let mut drain = registration.drain.clone();
        let (downstream_sink, mut downstream_stream) = downstream.split();
        let (mut upstream_sink, mut upstream_stream) = upstream.split();

        let initial_drain = *drain.borrow();
        let outcome = if initial_drain == DrainState::Forced {
            SetupOutcome::Terminal(RelayTerminal::Forced)
        } else if initial_drain == DrainState::Draining {
            SetupOutcome::Terminal(RelayTerminal::Draining)
        } else {
            loop {
                tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(deadline) => {
                        break SetupOutcome::Terminal(RelayTerminal::SetupDeadline {
                            code: 1011,
                            reason: timeout_reason,
                        });
                    }
                    changed = drain.changed() => {
                        let terminal = if changed.is_err()
                            || *drain.borrow() == DrainState::Forced
                        {
                            RelayTerminal::Forced
                        } else {
                            RelayTerminal::Draining
                        };
                        break SetupOutcome::Terminal(terminal);
                    }
                    event = next_worker_application(&mut upstream_stream) => {
                        break SetupOutcome::Worker(event);
                    }
                    message = downstream_stream.next() => {
                        let forward = forward_client_message(message, &mut upstream_sink, protocol);
                        tokio::pin!(forward);
                        let terminal = tokio::select! {
                            biased;
                            _ = tokio::time::sleep_until(deadline) => {
                                Some(RelayTerminal::SetupDeadline {
                                    code: 1011,
                                    reason: timeout_reason,
                                })
                            }
                            changed = drain.changed() => {
                                Some(if changed.is_err()
                                    || *drain.borrow() == DrainState::Forced
                                {
                                    RelayTerminal::Forced
                                } else {
                                    RelayTerminal::Draining
                                })
                            }
                            terminal = &mut forward => terminal,
                        };
                        if let Some(terminal) = terminal {
                            break SetupOutcome::Terminal(terminal);
                        }
                    }
                }
            }
        };

        let Ok(downstream) = downstream_sink.reunite(downstream_stream) else {
            return None;
        };
        let Ok(upstream) = upstream_sink.reunite(upstream_stream) else {
            return None;
        };
        let supervisor = Self {
            registration,
            lease,
            upstream,
        };
        match outcome {
            SetupOutcome::Worker(event) => Some((supervisor, downstream, event)),
            SetupOutcome::Terminal(terminal) => {
                let mut downstream = downstream;
                supervisor
                    .finish_terminal(&mut downstream, terminal, policy, &mut drain)
                    .await;
                None
            }
        }
    }

    async fn finish_terminal(
        mut self,
        downstream: &mut WebSocket,
        terminal: RelayTerminal,
        policy: &WebsocketConfig,
        drain: &mut watch::Receiver<DrainState>,
    ) {
        let upstream = &mut self.upstream;

        match terminal {
            RelayTerminal::ClientClose(frame) => {
                let converted = frame.and_then(downstream_close_to_upstream);
                bounded_close(
                    downstream,
                    upstream,
                    None,
                    Some(UpstreamMessage::Close(converted)),
                    ClosePlan::Both,
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
            RelayTerminal::WorkerClose(frame) => {
                let converted = frame.and_then(upstream_close_to_downstream);
                bounded_close(
                    downstream,
                    upstream,
                    Some(DownstreamMessage::Close(converted)),
                    None,
                    ClosePlan::Both,
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
            RelayTerminal::ClientGone => {
                close_upstream(downstream, upstream, policy.close_timeout(), drain).await;
            }
            RelayTerminal::WorkerGone => {
                self.lease.request_immediate_probe();
                close_downstream(
                    downstream,
                    upstream,
                    1011,
                    "upstream connection lost",
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
            RelayTerminal::ClientViolation { code, reason } => {
                terminate_both(
                    downstream,
                    upstream,
                    code,
                    reason,
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
            RelayTerminal::WorkerViolation { code, reason } => {
                self.lease.request_immediate_probe();
                terminate_both(
                    downstream,
                    upstream,
                    code,
                    reason,
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
            RelayTerminal::Draining => {
                terminate_both(
                    downstream,
                    upstream,
                    1012,
                    "service restart",
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
            RelayTerminal::Forced => {}
            RelayTerminal::SetupDeadline { code, reason } => {
                terminate_both(
                    downstream,
                    upstream,
                    code,
                    reason,
                    policy.close_timeout(),
                    drain,
                )
                .await;
            }
        }
    }
}

async fn next_worker_application(upstream: &mut UpstreamStream) -> WorkerEvent {
    loop {
        match upstream.next().await {
            Some(Ok(
                UpstreamMessage::Ping(_) | UpstreamMessage::Pong(_) | UpstreamMessage::Frame(_),
            )) => {}
            other => return other,
        }
    }
}

async fn client_to_worker(
    downstream: &mut DownstreamStream,
    upstream: &mut UpstreamSink,
    protocol: RelayProtocol,
) -> RelayTerminal {
    loop {
        if let Some(terminal) =
            forward_client_message(downstream.next().await, upstream, protocol).await
        {
            return terminal;
        }
    }
}

async fn forward_client_message(
    message: Option<Result<DownstreamMessage, axum::Error>>,
    upstream: &mut UpstreamSink,
    protocol: RelayProtocol,
) -> Option<RelayTerminal> {
    match message {
        Some(Ok(DownstreamMessage::Text(text))) => {
            let Ok(text) = super::downstream_text_to_upstream(text) else {
                return Some(RelayTerminal::ClientViolation {
                    code: 1007,
                    reason: "invalid text payload",
                });
            };
            upstream
                .send(UpstreamMessage::Text(text))
                .await
                .err()
                .map(|_| RelayTerminal::WorkerGone)
        }
        Some(Ok(DownstreamMessage::Binary(bytes))) if protocol.accepts_client_binary() => upstream
            .send(UpstreamMessage::Binary(bytes))
            .await
            .err()
            .map(|_| RelayTerminal::WorkerGone),
        Some(Ok(DownstreamMessage::Binary(_))) => Some(RelayTerminal::ClientViolation {
            code: 1003,
            reason: "binary messages are unsupported",
        }),
        Some(Ok(DownstreamMessage::Close(frame))) => Some(RelayTerminal::ClientClose(frame)),
        Some(Ok(DownstreamMessage::Ping(_) | DownstreamMessage::Pong(_))) => None,
        Some(Err(error)) if super::websocket_message_too_large(&error) => {
            Some(RelayTerminal::ClientViolation {
                code: 1009,
                reason: "message too large",
            })
        }
        Some(Err(_)) | None => Some(RelayTerminal::ClientGone),
    }
}

async fn worker_to_client(
    upstream: &mut UpstreamStream,
    downstream: &mut DownstreamSink,
    protocol: RelayProtocol,
) -> RelayTerminal {
    loop {
        match upstream.next().await {
            Some(Ok(UpstreamMessage::Text(text))) => {
                let Ok(text) = super::upstream_text_to_downstream(text) else {
                    return RelayTerminal::WorkerViolation {
                        code: 1011,
                        reason: "upstream protocol error",
                    };
                };
                if downstream
                    .send(DownstreamMessage::Text(text))
                    .await
                    .is_err()
                {
                    return RelayTerminal::ClientGone;
                }
            }
            Some(Ok(UpstreamMessage::Binary(bytes))) if protocol.accepts_worker_binary() => {
                if downstream
                    .send(DownstreamMessage::Binary(bytes))
                    .await
                    .is_err()
                {
                    return RelayTerminal::ClientGone;
                }
            }
            Some(Ok(UpstreamMessage::Binary(_))) => {
                return RelayTerminal::WorkerViolation {
                    code: 1011,
                    reason: "upstream protocol error",
                };
            }
            Some(Ok(UpstreamMessage::Close(frame))) => {
                return RelayTerminal::WorkerClose(frame);
            }
            Some(Ok(
                UpstreamMessage::Ping(_) | UpstreamMessage::Pong(_) | UpstreamMessage::Frame(_),
            )) => {}
            Some(Err(_)) | None => return RelayTerminal::WorkerGone,
        }
    }
}

pub(super) fn close_message(code: CloseCode, reason: &'static str) -> DownstreamMessage {
    DownstreamMessage::Close(Some(DownstreamClose {
        code,
        reason: reason.into(),
    }))
}

async fn terminate_both(
    downstream: &mut WebSocket,
    upstream: &mut UpstreamSocket,
    code: CloseCode,
    reason: &'static str,
    timeout: std::time::Duration,
    drain: &mut watch::Receiver<DrainState>,
) {
    bounded_close(
        downstream,
        upstream,
        Some(close_message(code, reason)),
        Some(UpstreamMessage::Close(Some(UpstreamClose {
            code: UpstreamCloseCode::from(code),
            reason: reason.into(),
        }))),
        ClosePlan::Both,
        timeout,
        drain,
    )
    .await;
}

async fn close_downstream(
    downstream: &mut WebSocket,
    upstream: &mut UpstreamSocket,
    code: CloseCode,
    reason: &'static str,
    timeout: std::time::Duration,
    drain: &mut watch::Receiver<DrainState>,
) {
    bounded_close(
        downstream,
        upstream,
        Some(close_message(code, reason)),
        None,
        ClosePlan::DownstreamOnly,
        timeout,
        drain,
    )
    .await;
}

async fn close_upstream(
    downstream: &mut WebSocket,
    upstream: &mut UpstreamSocket,
    timeout: std::time::Duration,
    drain: &mut watch::Receiver<DrainState>,
) {
    bounded_close(
        downstream,
        upstream,
        None,
        Some(UpstreamMessage::Close(Some(UpstreamClose {
            code: UpstreamCloseCode::Normal,
            reason: "".into(),
        }))),
        ClosePlan::UpstreamOnly,
        timeout,
        drain,
    )
    .await;
}

async fn bounded_close(
    downstream: &mut WebSocket,
    upstream: &mut UpstreamSocket,
    downstream_close: Option<DownstreamMessage>,
    upstream_close: Option<UpstreamMessage>,
    plan: ClosePlan,
    timeout: std::time::Duration,
    drain: &mut watch::Receiver<DrainState>,
) {
    let closing = async {
        match (downstream_close, upstream_close) {
            (Some(downstream_message), Some(upstream_message)) => {
                let (_downstream, _upstream) = tokio::join!(
                    downstream.send(downstream_message),
                    upstream.send(upstream_message),
                );
            }
            (Some(message), None) => {
                let _ignored = downstream.send(message).await;
            }
            (None, Some(message)) => {
                let _ignored = upstream.send(message).await;
            }
            (None, None) => {}
        }
        match plan {
            ClosePlan::Both => {
                let (_downstream, _upstream) = tokio::join!(
                    wait_downstream_terminal(downstream),
                    wait_upstream_terminal(upstream),
                );
            }
            ClosePlan::DownstreamOnly => wait_downstream_terminal(downstream).await,
            ClosePlan::UpstreamOnly => wait_upstream_terminal(upstream).await,
        }
    };
    tokio::select! {
        _ = tokio::time::timeout(timeout, closing) => {}
        _ = drain.wait_for(|state| *state == DrainState::Forced) => {}
    }
}

async fn wait_downstream_terminal(downstream: &mut WebSocket) {
    while let Some(message) = downstream.next().await {
        if matches!(message, Ok(DownstreamMessage::Close(_)) | Err(_)) {
            return;
        }
    }
}

async fn wait_upstream_terminal(upstream: &mut UpstreamSocket) {
    while let Some(message) = upstream.next().await {
        if matches!(message, Ok(UpstreamMessage::Close(_)) | Err(_)) {
            return;
        }
    }
}

fn downstream_close_to_upstream(frame: DownstreamClose) -> Option<UpstreamClose> {
    (frame.reason.len() <= 123).then(|| UpstreamClose {
        code: UpstreamCloseCode::from(frame.code),
        reason: frame.reason.as_str().into(),
    })
}

fn upstream_close_to_downstream(frame: UpstreamClose) -> Option<DownstreamClose> {
    (frame.reason.len() <= 123).then(|| DownstreamClose {
        code: CloseCode::from(frame.code),
        reason: frame.reason.as_str().into(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::time::Duration;

    use super::{DrainState, SessionTracker};

    #[tokio::test]
    async fn tracker_registration_is_exact_once_and_drain_visible() {
        let tracker = SessionTracker::new();
        let registration = tracker.register().expect("register serving session");
        tracker.drain();
        assert!(tracker.register().is_none());
        assert_eq!(*registration.drain.borrow(), DrainState::Draining);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), tracker.wait_empty())
                .await
                .is_err()
        );
        drop(registration);
        tokio::time::timeout(Duration::from_secs(1), tracker.wait_empty())
            .await
            .expect("tracker reaches zero after one drop");
    }

    #[tokio::test]
    async fn forced_drain_is_observed_by_every_registered_session() {
        let tracker = SessionTracker::new();
        let first = tracker.register().expect("first registration");
        let second = tracker.register().expect("second registration");
        tracker.drain();
        tracker.force();
        assert_eq!(*first.drain.borrow(), DrainState::Forced);
        assert_eq!(*second.drain.borrow(), DrainState::Forced);
        drop((first, second));
        tracker.wait_empty().await;
    }
}
