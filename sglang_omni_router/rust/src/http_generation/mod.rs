mod classify;
mod headers;
mod request_body;
mod response_body;

use std::future::poll_fn;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, Response, Version};
use bytes::{Bytes, BytesMut};
use http_body::Body as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::error;

use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::request_id::{CanonicalRequestId, REQUEST_ID_HEADER};
use crate::worker_pool::{AdmissionError, DispatchError, RequestLease, TrustDomain, WorkerPool};

use classify::classify;
use headers::{canonical_content_type, sanitize_response, validate_request};
use request_body::{BufferedBody, DirectRequestBody, SharedUploadState, UploadState};
use response_body::DirectResponseBody;

pub(crate) const CHAT_PATH: &str = "/v1/chat/completions";

pub(crate) struct HttpGeneration {
    pool: Arc<WorkerPool>,
    client: reqwest::Client,
    trust: TrustDomain,
    buffered_max: u64,
    streamed_max: u64,
    buffered_budget: Arc<Semaphore>,
    classification_slots: Arc<Semaphore>,
    request_timeout: std::time::Duration,
}

impl HttpGeneration {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
        classification_slots: Arc<Semaphore>,
    ) -> Result<Arc<Self>, RouterError> {
        let http_generation = &config.http_generation;
        let buffered_total = http_generation
            .buffered_total_usize()
            .map_err(RouterError::Config)?;
        let client = pool.generation_client();
        Ok(Arc::new(Self {
            client,
            pool,
            trust: TrustDomain::new(http_generation.trust_domain.clone()),
            buffered_max: http_generation.buffered_request_max_bytes,
            streamed_max: http_generation.streamed_request_max_bytes,
            buffered_budget: Arc::new(Semaphore::new(buffered_total)),
            classification_slots,
            request_timeout: http_generation.request_timeout(),
        }))
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.pool.generation_http_ready(&self.trust)
    }
}

pub(crate) async fn chat(
    State(generation): State<Arc<HttpGeneration>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    match handle(generation, request, request_id.into_header_value()).await {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle(
    generation: Arc<HttpGeneration>,
    request: Request<Body>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    if request.method() != Method::POST {
        return Err(HttpFault::MethodNotAllowed);
    }
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    if request.uri().path() != CHAT_PATH || request.uri().query().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    let deadline = tokio::time::Instant::now() + generation.request_timeout;
    let framing = validate_request(request.headers())?;
    let proof = generation
        .pool
        .content_blind_generation_http(&generation.trust);
    let maximum = if proof.is_some() {
        generation.streamed_max
    } else {
        generation.buffered_max
    };
    if framing.content_length > maximum {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let admission = generation.pool.try_admit().map_err(map_admission)?;

    if let Some(proof) = proof {
        let length = framing.content_length;
        let lease = proof.dispatch(admission).map_err(map_dispatch)?;
        return relay_direct(
            generation,
            request.into_body(),
            length,
            lease,
            request_id,
            deadline,
        )
        .await;
    }

    let reserve = framing.content_length;
    let budget = reserve_budget(&generation.buffered_budget, reserve)?;
    let bytes = read_buffered(
        request.into_body(),
        Some(framing.content_length),
        generation.buffered_max,
        deadline,
    )
    .await?;
    let classify_pool = Arc::clone(&generation.pool);
    let classify_trust = generation.trust.clone();
    let (bytes, budget, classified) = classify_blocking(
        Arc::clone(&generation.classification_slots),
        deadline,
        move || {
            let classified = classify(&bytes, &classify_pool, &classify_trust)?;
            Ok((bytes, budget, classified))
        },
    )
    .await?;
    let lease = generation
        .pool
        .dispatch(admission, &classified.requirement)
        .map_err(map_dispatch)?;
    relay_buffered(generation, bytes, budget, lease, request_id, deadline).await
}

fn reserve_budget(
    semaphore: &Arc<Semaphore>,
    bytes: u64,
) -> Result<OwnedSemaphorePermit, HttpFault> {
    let permits = u32::try_from(bytes).map_err(|_| HttpFault::InternalError)?;
    Arc::clone(semaphore)
        .try_acquire_many_owned(permits)
        .map_err(|_| HttpFault::RouterOverloaded)
}

async fn read_buffered(
    mut body: Body,
    expected: Option<u64>,
    maximum: u64,
    deadline: tokio::time::Instant,
) -> Result<Bytes, HttpFault> {
    let initial = initial_buffer_capacity(expected);
    let mut output = BytesMut::with_capacity(initial);
    let mut observed = 0_u64;
    let deadline_timer = tokio::time::sleep_until(deadline);
    tokio::pin!(deadline_timer);
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(HttpFault::RequestTimeout);
        }
        let frame = tokio::select! {
            biased;
            () = &mut deadline_timer => return Err(HttpFault::RequestTimeout),
            frame = poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx)) => frame,
        };
        match frame {
            Some(Ok(frame)) => match frame.into_data() {
                Ok(data) => {
                    let length =
                        u64::try_from(data.len()).map_err(|_| HttpFault::RequestBodyTooLarge)?;
                    observed = observed
                        .checked_add(length)
                        .ok_or(HttpFault::RequestBodyTooLarge)?;
                    if observed > maximum || expected.is_some_and(|value| observed > value) {
                        return Err(HttpFault::RequestBodyTooLarge);
                    }
                    output.extend_from_slice(&data);
                }
                Err(_trailers) => return Err(HttpFault::MalformedRequest),
            },
            Some(Err(_source)) => return Err(HttpFault::MalformedRequest),
            None => break,
        }
    }
    if expected.is_some_and(|value| observed != value) {
        return Err(HttpFault::MalformedRequest);
    }
    Ok(output.freeze())
}

fn initial_buffer_capacity(expected: Option<u64>) -> usize {
    expected.unwrap_or(0).min(usize::MAX as u64) as usize
}

async fn relay_buffered(
    generation: Arc<HttpGeneration>,
    bytes: Bytes,
    budget: OwnedSemaphorePermit,
    lease: RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let length = u64::try_from(bytes.len()).map_err(|_| HttpFault::InternalError)?;
    let body = reqwest::Body::wrap(BufferedBody::new(bytes, budget));
    send_once(
        generation,
        OutgoingBody {
            body,
            length,
            upload: None,
        },
        lease,
        request_id,
        deadline,
    )
    .await
}

async fn relay_direct(
    generation: Arc<HttpGeneration>,
    body: Body,
    length: u64,
    lease: RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let state = SharedUploadState::for_length(length);
    let direct = DirectRequestBody::new(body, length, state.clone());
    send_once(
        generation,
        OutgoingBody {
            body: reqwest::Body::wrap(direct),
            length,
            upload: Some(state),
        },
        lease,
        request_id,
        deadline,
    )
    .await
}

struct OutgoingBody {
    body: reqwest::Body,
    length: u64,
    upload: Option<SharedUploadState>,
}

async fn send_once(
    generation: Arc<HttpGeneration>,
    outgoing: OutgoingBody,
    lease: RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let mut url = lease.target().base_url().clone();
    url.set_path(CHAT_PATH);
    url.set_query(None);
    let request = generation
        .client
        .post(url)
        .header(CONTENT_TYPE, canonical_content_type())
        .header(CONTENT_LENGTH, outgoing.length)
        .header(REQUEST_ID_HEADER, request_id)
        .body(outgoing.body);
    let _attempt = authorize_upstream_attempt_at(
        deadline,
        outgoing.upload.as_ref(),
        tokio::time::Instant::now(),
    )?;
    let sent = tokio::select! {
        biased;
        result = request.send() => result,
        () = tokio::time::sleep_until(deadline) => {
            let outcome = deadline_outcome(outgoing.upload.as_ref());
            if outcome.request_probe {
                lease.request_immediate_probe();
            }
            return Err(outcome.fault);
        }
    };
    let response = match sent {
        Ok(response) => response,
        Err(_source) => {
            let fault = selected_send_fault(&outgoing.upload)?;
            if fault == HttpFault::UpstreamProtocolError {
                lease.request_immediate_probe();
            }
            return Err(fault);
        }
    };
    if let Err(fault) = require_completed_upload(&outgoing.upload) {
        if fault == HttpFault::UpstreamProtocolError {
            tracing::warn!(
                worker = %lease.target().base_url(),
                "worker responded before the request upload completed"
            );
        }
        return Err(fault);
    }
    let response: axum::http::Response<reqwest::Body> = response.into();
    let (parts, body) = response.into_parts();
    let headers = match sanitize_response(parts.status, &parts.headers) {
        Ok(headers) => headers,
        Err(fault) => {
            lease.request_immediate_probe();
            return Err(fault);
        }
    };
    let relay = DirectResponseBody::new(body, lease);
    let mut downstream = Response::new(Body::new(relay));
    *downstream.status_mut() = parts.status;
    *downstream.headers_mut() = headers;
    Ok(downstream)
}

fn require_completed_upload(upload: &Option<SharedUploadState>) -> Result<(), HttpFault> {
    match upload
        .as_ref()
        .map(SharedUploadState::snapshot)
        .transpose()?
    {
        Some(UploadState::Incomplete) => Err(HttpFault::UpstreamProtocolError),
        Some(UploadState::Failed(fault)) => Err(fault),
        Some(UploadState::Complete) | None => Ok(()),
    }
}

fn selected_send_fault(upload: &Option<SharedUploadState>) -> Result<HttpFault, HttpFault> {
    match upload
        .as_ref()
        .map(SharedUploadState::snapshot)
        .transpose()?
    {
        Some(UploadState::Failed(fault)) => Ok(fault),
        Some(UploadState::Incomplete | UploadState::Complete) | None => {
            Ok(HttpFault::UpstreamProtocolError)
        }
    }
}

fn deadline_fault(upload: Option<&SharedUploadState>) -> HttpFault {
    match upload.map(SharedUploadState::snapshot).transpose() {
        Err(fault) => fault,
        Ok(state) => match state {
            Some(UploadState::Incomplete)
            | Some(UploadState::Failed(HttpFault::RequestTimeout)) => HttpFault::RequestTimeout,
            Some(UploadState::Failed(fault)) => fault,
            Some(UploadState::Complete) | None => HttpFault::UpstreamTimeout,
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeadlineOutcome {
    fault: HttpFault,
    request_probe: bool,
}

fn deadline_outcome(upload: Option<&SharedUploadState>) -> DeadlineOutcome {
    let fault = deadline_fault(upload);
    DeadlineOutcome {
        fault,
        request_probe: fault == HttpFault::UpstreamTimeout,
    }
}

fn finish_buffered_classification<T>(
    classified: Result<T, HttpFault>,
    deadline: tokio::time::Instant,
    now: tokio::time::Instant,
) -> Result<T, HttpFault> {
    check_precommit_deadline_at(deadline, None, now)?;
    classified
}

async fn classify_blocking<T>(
    slots: Arc<Semaphore>,
    deadline: tokio::time::Instant,
    operation: impl FnOnce() -> Result<T, HttpFault> + Send + 'static,
) -> Result<T, HttpFault>
where
    T: Send + 'static,
{
    check_precommit_deadline_at(deadline, None, tokio::time::Instant::now())?;
    let slot = tokio::select! {
        biased;
        () = tokio::time::sleep_until(deadline) => return Err(HttpFault::UpstreamTimeout),
        result = slots.acquire_owned() => result.map_err(|_| HttpFault::InternalError)?,
    };
    let mut task = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        check_precommit_deadline_at(deadline, None, tokio::time::Instant::now())?;
        operation()
    });
    let classified = tokio::select! {
        biased;
        () = tokio::time::sleep_until(deadline) => {
            task.abort();
            return Err(HttpFault::UpstreamTimeout);
        }
        result = &mut task => result,
    }
    .map_err(|source| {
        error!(error = %source, "classification task failed");
        HttpFault::InternalError
    })?;
    finish_buffered_classification(classified, deadline, tokio::time::Instant::now())
}

struct UpstreamAttemptAuthorization;

fn authorize_upstream_attempt_at(
    deadline: tokio::time::Instant,
    upload: Option<&SharedUploadState>,
    now: tokio::time::Instant,
) -> Result<UpstreamAttemptAuthorization, HttpFault> {
    check_precommit_deadline_at(deadline, upload, now)?;
    Ok(UpstreamAttemptAuthorization)
}

fn check_precommit_deadline_at(
    deadline: tokio::time::Instant,
    upload: Option<&SharedUploadState>,
    now: tokio::time::Instant,
) -> Result<(), HttpFault> {
    if now >= deadline {
        Err(deadline_fault(upload))
    } else {
        Ok(())
    }
}

const fn map_admission(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Draining => HttpFault::RouterUnavailable,
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
    }
}

const fn map_dispatch(error: DispatchError) -> HttpFault {
    match error {
        DispatchError::NoEligibleProfile => HttpFault::NoCompatibleWorker,
        DispatchError::Unavailable => HttpFault::RouterUnavailable,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::cell::Cell;
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use axum::body::Body;
    use bytes::Bytes;
    use http_body::Frame;
    use tokio::sync::Semaphore;

    use super::{
        HttpFault, SharedUploadState, UploadState, authorize_upstream_attempt_at, deadline_outcome,
        finish_buffered_classification, initial_buffer_capacity, read_buffered,
        require_completed_upload, reserve_budget, selected_send_fault,
    };

    struct AlwaysReady;

    impl http_body::Body for AlwaysReady {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"x")))))
        }
    }

    #[tokio::test]
    async fn buffered_deadline_preempts_perpetually_ready_frames() {
        let result = read_buffered(
            Body::new(AlwaysReady),
            None,
            1_024,
            tokio::time::Instant::now(),
        )
        .await;
        assert_eq!(result, Err(HttpFault::RequestTimeout));
    }

    #[test]
    fn unknown_length_reserves_logical_budget_without_eager_physical_capacity() {
        assert_eq!(initial_buffer_capacity(None), 0);
        assert_eq!(initial_buffer_capacity(Some(64)), 64);
    }

    #[test]
    fn buffered_byte_budget_rejects_aggregate_exhaustion() {
        let budget = Arc::new(Semaphore::new(8));
        let held = reserve_budget(&budget, 6).expect("reserve the first buffered request");
        assert_eq!(
            reserve_budget(&budget, 3).err(),
            Some(HttpFault::RouterOverloaded)
        );
        drop(held);
        assert!(reserve_budget(&budget, 8).is_ok());
    }

    #[test]
    fn post_classification_and_pre_attempt_deadline_gates_are_deterministic() {
        let start = tokio::time::Instant::now();
        let deadline = start + Duration::from_millis(1);
        assert_eq!(
            finish_buffered_classification::<()>(Ok(()), deadline, deadline),
            Err(HttpFault::UpstreamTimeout)
        );
        assert_eq!(
            finish_buffered_classification::<()>(
                Err(HttpFault::MalformedRequest),
                deadline,
                deadline,
            ),
            Err(HttpFault::UpstreamTimeout),
            "the absolute post-EOF deadline is authoritative over a synchronous parse result"
        );

        finish_buffered_classification(Ok(()), deadline, start)
            .expect("classification completed before the injected deadline");
        let attempts = Cell::new(0_u8);
        let result =
            authorize_upstream_attempt_at(deadline, None, deadline).map(|_authorization| {
                attempts.set(attempts.get() + 1);
            });
        assert_eq!(result, Err(HttpFault::UpstreamTimeout));
        assert_eq!(attempts.get(), 0, "an expired request cannot start a send");

        let upload = SharedUploadState::new(UploadState::Incomplete);
        assert!(matches!(
            authorize_upstream_attempt_at(deadline, Some(&upload), deadline),
            Err(HttpFault::RequestTimeout)
        ));
        upload
            .publish(UploadState::Complete)
            .expect("update test upload state");
        assert!(matches!(
            authorize_upstream_attempt_at(deadline, Some(&upload), deadline),
            Err(HttpFault::UpstreamTimeout)
        ));
    }

    #[test]
    fn selected_send_result_is_not_remapped_by_wall_clock_time() {
        assert_eq!(
            selected_send_fault(&None),
            Ok(HttpFault::UpstreamProtocolError)
        );
        let complete = Some(SharedUploadState::new(UploadState::Complete));
        assert_eq!(
            selected_send_fault(&complete),
            Ok(HttpFault::UpstreamProtocolError)
        );
        let failed = Some(SharedUploadState::new(UploadState::Failed(
            HttpFault::RequestTimeout,
        )));
        assert_eq!(selected_send_fault(&failed), Ok(HttpFault::RequestTimeout));
    }

    #[test]
    fn completed_upstream_response_requires_a_completed_upload() {
        let incomplete = Some(SharedUploadState::new(UploadState::Incomplete));

        assert_eq!(
            require_completed_upload(&incomplete),
            Err(HttpFault::UpstreamProtocolError)
        );
        let complete = Some(SharedUploadState::new(UploadState::Complete));
        assert_eq!(require_completed_upload(&complete), Ok(()));
        let failed = Some(SharedUploadState::new(UploadState::Failed(
            HttpFault::MalformedRequest,
        )));
        assert_eq!(
            require_completed_upload(&failed),
            Err(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn only_worker_owned_precommit_timeouts_request_an_immediate_probe() {
        let complete = SharedUploadState::new(UploadState::Complete);
        let incomplete = SharedUploadState::new(UploadState::Incomplete);
        let client_timeout = SharedUploadState::new(UploadState::Failed(HttpFault::RequestTimeout));
        let body_fault = SharedUploadState::new(UploadState::Failed(HttpFault::MalformedRequest));

        for upload in [None, Some(&complete)] {
            let outcome = deadline_outcome(upload);
            assert_eq!(outcome.fault, HttpFault::UpstreamTimeout);
            assert!(outcome.request_probe);
        }
        for (upload, expected) in [
            (Some(&incomplete), HttpFault::RequestTimeout),
            (Some(&client_timeout), HttpFault::RequestTimeout),
            (Some(&body_fault), HttpFault::MalformedRequest),
        ] {
            let outcome = deadline_outcome(upload);
            assert_eq!(outcome.fault, expected);
            assert!(!outcome.request_probe);
        }
    }

    #[tokio::test]
    async fn blocking_classification_does_not_stall_the_reactor() {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let classifier = tokio::spawn(async move {
            super::classify_blocking(
                Arc::new(Semaphore::new(1)),
                tokio::time::Instant::now() + Duration::from_secs(1),
                move || {
                    entered_tx.send(()).expect("announce classifier entry");
                    release_rx.recv().expect("release blocking classifier");
                    Ok(())
                },
            )
            .await
        });
        entered_rx.await.expect("classifier entered");

        let (reactor_tx, reactor_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            reactor_tx.send(()).expect("mark reactor progress");
        });
        tokio::time::timeout(Duration::from_secs(1), reactor_rx)
            .await
            .expect("reactor progressed while classifier blocked")
            .expect("reactor marker delivered");
        release_tx.send(()).expect("release classifier");
        assert_eq!(
            classifier.await.expect("join classification fixture"),
            Ok(())
        );
    }

    #[tokio::test]
    async fn classification_waits_for_an_execution_slot() {
        let slots = Arc::new(Semaphore::new(1));
        let held = Arc::clone(&slots)
            .try_acquire_owned()
            .expect("hold classification slot");
        let (entered_tx, mut entered_rx) = tokio::sync::oneshot::channel();
        let classifier = tokio::spawn(super::classify_blocking(
            Arc::clone(&slots),
            tokio::time::Instant::now() + Duration::from_secs(1),
            move || {
                entered_tx.send(()).expect("announce classifier entry");
                Ok(())
            },
        ));

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut entered_rx)
                .await
                .is_err()
        );
        drop(held);
        entered_rx.await.expect("classifier entered after release");
        assert_eq!(
            classifier.await.expect("join classification fixture"),
            Ok(())
        );
    }

    #[tokio::test]
    async fn classification_deadline_includes_execution_slot_wait() {
        let slots = Arc::new(Semaphore::new(1));
        let _held = Arc::clone(&slots)
            .try_acquire_owned()
            .expect("hold classification slot");
        let ran = Arc::new(AtomicBool::new(false));
        let ran_in_task = Arc::clone(&ran);

        let result = super::classify_blocking(
            slots,
            tokio::time::Instant::now() + Duration::from_millis(20),
            move || {
                ran_in_task.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .await;

        assert_eq!(result, Err(HttpFault::UpstreamTimeout));
        assert!(!ran.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn timed_out_running_classification_retains_owned_resources() {
        let slots = Arc::new(Semaphore::new(1));
        let budget = Arc::new(Semaphore::new(1));
        let budget_permit = Arc::clone(&budget)
            .try_acquire_owned()
            .expect("reserve buffered memory");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let classifier = tokio::spawn(super::classify_blocking(
            Arc::clone(&slots),
            tokio::time::Instant::now() + Duration::from_millis(50),
            move || {
                let _budget = budget_permit;
                entered_tx.send(()).expect("announce classifier entry");
                release_rx.recv().expect("release blocking classifier");
                Ok(())
            },
        ));
        entered_rx.await.expect("classifier entered");

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), classifier)
                .await
                .expect("deadline returned promptly")
                .expect("join classification fixture"),
            Err(HttpFault::UpstreamTimeout)
        );
        assert_eq!(slots.available_permits(), 0);
        assert_eq!(budget.available_permits(), 0);

        release_tx.send(()).expect("release classifier");
        tokio::time::timeout(Duration::from_secs(1), async {
            while slots.available_permits() == 0 || budget.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking closure eventually releases its resources");
        assert_eq!(slots.available_permits(), 1);
        assert_eq!(budget.available_permits(), 1);
    }

    #[tokio::test]
    async fn cancelled_waiter_keeps_buffer_budget_until_classification_finishes() {
        let budget = Arc::new(Semaphore::new(1));
        let budget_permit = Arc::clone(&budget)
            .try_acquire_owned()
            .expect("reserve buffered memory");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let classifier = tokio::spawn(async move {
            super::classify_blocking(
                Arc::new(Semaphore::new(1)),
                tokio::time::Instant::now() + Duration::from_secs(1),
                move || {
                    let _budget = budget_permit;
                    entered_tx.send(()).expect("announce classifier entry");
                    release_rx.recv().expect("release blocking classifier");
                    Ok(())
                },
            )
            .await
        });
        entered_rx.await.expect("classifier entered");
        classifier.abort();
        assert!(
            classifier
                .await
                .expect_err("waiter is cancelled")
                .is_cancelled()
        );
        assert_eq!(budget.available_permits(), 0);

        release_tx.send(()).expect("release classifier");
        tokio::time::timeout(Duration::from_secs(1), async {
            while budget.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking closure eventually releases buffered memory");
        assert_eq!(budget.available_permits(), 1);
    }
}
