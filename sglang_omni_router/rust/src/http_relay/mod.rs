mod headers;
mod request_body;
mod response_body;

use std::future::poll_fn;
use std::sync::Arc;

use axum::body::Body;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use bytes::{Bytes, BytesMut};
use http_body::Body as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::error;

use crate::error::HttpFault;
use crate::request_id::REQUEST_ID_HEADER;
use crate::worker_pool::{AdmissionError, DispatchError, RequestLease};

use request_body::{BufferedBody, DirectRequestBody};
pub(crate) use request_body::{SharedUploadState, UploadState};
use response_body::DirectResponseBody;

pub(crate) use headers::{
    RequestEnvelope, is_request_media_type, request_content_type, sanitize_response_headers,
    validate_request_envelope,
};

pub(crate) struct HttpRelay {
    client: reqwest::Client,
    buffered_budget: Arc<Semaphore>,
    classification_slots: Arc<Semaphore>,
}

pub(crate) struct BufferedUpload {
    pub(crate) bytes: Bytes,
    budget: Option<OwnedSemaphorePermit>,
}

pub(crate) struct OutgoingRequest {
    path: &'static str,
    content_type: HeaderValue,
    body: reqwest::Body,
    content_length: Option<u64>,
    upload: Option<SharedUploadState>,
}

impl HttpRelay {
    pub(crate) fn new(client: reqwest::Client, buffered_budget: usize) -> Arc<Self> {
        Arc::new(Self {
            client,
            buffered_budget: Arc::new(Semaphore::new(buffered_budget)),
            classification_slots: Arc::new(Semaphore::new(
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
            )),
        })
    }

    pub(crate) async fn classify<T>(
        &self,
        deadline: tokio::time::Instant,
        operation: impl FnOnce() -> Result<T, HttpFault> + Send + 'static,
    ) -> Result<T, HttpFault>
    where
        T: Send + 'static,
    {
        classify_blocking(Arc::clone(&self.classification_slots), deadline, operation).await
    }

    pub(crate) async fn read_buffered(
        &self,
        mut body: Body,
        expected: Option<u64>,
        maximum: u64,
        deadline: tokio::time::Instant,
    ) -> Result<BufferedUpload, HttpFault> {
        let mut budget = match expected {
            Some(bytes) => reserve_budget(&self.buffered_budget, bytes)?,
            None => None,
        };
        let mut output = BytesMut::with_capacity(initial_buffer_capacity(expected));
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
                        let length = u64::try_from(data.len())
                            .map_err(|_| HttpFault::RequestBodyTooLarge)?;
                        observed = observed
                            .checked_add(length)
                            .ok_or(HttpFault::RequestBodyTooLarge)?;
                        if observed > maximum || expected.is_some_and(|value| observed > value) {
                            return Err(HttpFault::RequestBodyTooLarge);
                        }
                        if expected.is_none() && length != 0 {
                            merge_budget(
                                &mut budget,
                                reserve_budget(&self.buffered_budget, length)?,
                            );
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
        Ok(BufferedUpload {
            bytes: output.freeze(),
            budget,
        })
    }

    pub(crate) async fn send(
        self: Arc<Self>,
        outgoing: OutgoingRequest,
        lease: RequestLease,
        request_id: HeaderValue,
        deadline: tokio::time::Instant,
        sanitize: impl FnOnce(StatusCode, &HeaderMap) -> Result<HeaderMap, HttpFault>,
    ) -> Result<Response<Body>, HttpFault> {
        check_precommit_deadline_at(
            deadline,
            outgoing.upload.as_ref(),
            tokio::time::Instant::now(),
        )?;
        let mut url = lease.target().base_url().clone();
        url.set_path(outgoing.path);
        url.set_query(None);
        let mut request = self
            .client
            .post(url)
            .header(CONTENT_TYPE, outgoing.content_type)
            .header(REQUEST_ID_HEADER, request_id);
        if let Some(length) = outgoing.content_length {
            request = request.header(CONTENT_LENGTH, length);
        }
        let request = request.body(outgoing.body);
        let sent = tokio::select! {
            biased;
            result = request.send() => result,
            () = tokio::time::sleep_until(deadline) => {
                let fault = deadline_fault(outgoing.upload.as_ref());
                if fault == HttpFault::UpstreamTimeout {
                    lease.request_immediate_probe();
                }
                return Err(fault);
            }
        };
        let response = match sent {
            Ok(response) => response,
            Err(_source) => {
                let fault = upload_fault(outgoing.upload.as_ref())?
                    .unwrap_or(HttpFault::UpstreamProtocolError);
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
        let headers = match sanitize(parts.status, &parts.headers) {
            Ok(headers) => headers,
            Err(fault) => {
                drop(body);
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
}

impl OutgoingRequest {
    pub(crate) fn buffered(
        path: &'static str,
        content_type: HeaderValue,
        upload: BufferedUpload,
    ) -> Result<Self, HttpFault> {
        let content_length =
            u64::try_from(upload.bytes.len()).map_err(|_| HttpFault::InternalError)?;
        Ok(Self {
            path,
            content_type,
            body: reqwest::Body::wrap(BufferedBody::new(upload.bytes, upload.budget)),
            content_length: Some(content_length),
            upload: None,
        })
    }

    pub(crate) fn direct(
        path: &'static str,
        content_type: HeaderValue,
        body: Body,
        expected: Option<u64>,
        maximum: u64,
    ) -> Self {
        let state = SharedUploadState::new(UploadState::Incomplete);
        let direct = DirectRequestBody::new(body, expected, maximum, state.clone());
        Self {
            path,
            content_type,
            body: reqwest::Body::wrap(direct),
            content_length: expected,
            upload: Some(state),
        }
    }
}

fn reserve_budget(
    semaphore: &Arc<Semaphore>,
    bytes: u64,
) -> Result<Option<OwnedSemaphorePermit>, HttpFault> {
    if bytes == 0 {
        return Ok(None);
    }
    let permits = u32::try_from(bytes).map_err(|_| HttpFault::InternalError)?;
    Arc::clone(semaphore)
        .try_acquire_many_owned(permits)
        .map(Some)
        .map_err(|_| HttpFault::RouterOverloaded)
}

fn merge_budget(
    accumulated: &mut Option<OwnedSemaphorePermit>,
    acquired: Option<OwnedSemaphorePermit>,
) {
    let Some(acquired) = acquired else {
        return;
    };
    match accumulated {
        Some(permit) => permit.merge(acquired),
        None => *accumulated = Some(acquired),
    }
}

fn initial_buffer_capacity(expected: Option<u64>) -> usize {
    expected.unwrap_or(0).min(usize::MAX as u64) as usize
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
    check_precommit_deadline_at(deadline, None, tokio::time::Instant::now())?;
    classified
}

pub(crate) fn snapshot_upload(state: &SharedUploadState) -> Result<UploadState, HttpFault> {
    state.snapshot()
}

fn require_completed_upload(upload: &Option<SharedUploadState>) -> Result<(), HttpFault> {
    match upload.as_ref().map(snapshot_upload).transpose()? {
        Some(UploadState::Incomplete) => Err(HttpFault::UpstreamProtocolError),
        Some(UploadState::Failed(fault)) => Err(fault),
        Some(UploadState::Complete) | None => Ok(()),
    }
}

fn upload_fault(upload: Option<&SharedUploadState>) -> Result<Option<HttpFault>, HttpFault> {
    upload
        .map(snapshot_upload)
        .transpose()
        .map(|state| match state {
            Some(UploadState::Failed(fault)) => Some(fault),
            Some(UploadState::Incomplete | UploadState::Complete) | None => None,
        })
}

fn deadline_fault(upload: Option<&SharedUploadState>) -> HttpFault {
    match upload.map(snapshot_upload).transpose() {
        Err(fault) => fault,
        Ok(Some(UploadState::Incomplete)) => HttpFault::RequestTimeout,
        Ok(Some(UploadState::Failed(fault))) => fault,
        Ok(Some(UploadState::Complete) | None) => HttpFault::UpstreamTimeout,
    }
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

pub(crate) const fn map_admission(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Draining => HttpFault::RouterUnavailable,
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
    }
}

pub(crate) const fn map_dispatch(error: DispatchError) -> HttpFault {
    match error {
        DispatchError::AmbiguousModel => HttpFault::AmbiguousModel,
        DispatchError::NoEligibleProfile => HttpFault::NoCompatibleWorker,
        DispatchError::Unavailable => HttpFault::RouterUnavailable,
        DispatchError::Internal => HttpFault::InternalError,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
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
        HttpFault, HttpRelay, SharedUploadState, UploadState, check_precommit_deadline_at,
        deadline_fault, initial_buffer_capacity, require_completed_upload, upload_fault,
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

    fn relay(budget: usize) -> Arc<HttpRelay> {
        HttpRelay::new(reqwest::Client::new(), budget)
    }

    fn relay_with_slots(budget: usize, slots: usize) -> Arc<HttpRelay> {
        Arc::new(HttpRelay {
            client: reqwest::Client::new(),
            buffered_budget: Arc::new(Semaphore::new(budget)),
            classification_slots: Arc::new(Semaphore::new(slots)),
        })
    }

    #[tokio::test]
    async fn buffered_deadline_preempts_perpetually_ready_frames() {
        let result = relay(1_024)
            .read_buffered(
                Body::new(AlwaysReady),
                None,
                1_024,
                tokio::time::Instant::now(),
            )
            .await;
        assert!(matches!(result, Err(HttpFault::RequestTimeout)));
    }

    #[tokio::test]
    async fn known_and_chunked_uploads_share_exact_buffer_budget() {
        let relay = relay(8);
        let held = relay
            .read_buffered(
                Body::from("123456"),
                Some(6),
                8,
                tokio::time::Instant::now() + Duration::from_secs(1),
            )
            .await
            .expect("reserve known upload");
        let rejected = relay
            .read_buffered(
                Body::from("123"),
                None,
                8,
                tokio::time::Instant::now() + Duration::from_secs(1),
            )
            .await;
        assert!(matches!(rejected, Err(HttpFault::RouterOverloaded)));
        drop(held);

        let chunked = relay
            .read_buffered(
                Body::from("12345678"),
                None,
                8,
                tokio::time::Instant::now() + Duration::from_secs(1),
            )
            .await
            .expect("incrementally reserve complete chunked upload");
        assert_eq!(chunked.bytes, "12345678");
        assert_eq!(relay.buffered_budget.available_permits(), 0);
        drop(chunked);
        assert_eq!(relay.buffered_budget.available_permits(), 8);
    }

    #[test]
    fn buffer_capacity_and_precommit_faults_are_deterministic() {
        assert_eq!(initial_buffer_capacity(None), 0);
        assert_eq!(initial_buffer_capacity(Some(64)), 64);

        let now = tokio::time::Instant::now();
        assert_eq!(
            check_precommit_deadline_at(now, None, now),
            Err(HttpFault::UpstreamTimeout)
        );
        let upload = SharedUploadState::new(UploadState::Incomplete);
        assert_eq!(
            check_precommit_deadline_at(now, Some(&upload), now),
            Err(HttpFault::RequestTimeout)
        );
        assert_eq!(upload_fault(Some(&upload)), Ok(None));
        assert_eq!(
            require_completed_upload(&Some(upload.clone())),
            Err(HttpFault::UpstreamProtocolError)
        );
        upload
            .publish(UploadState::Complete)
            .expect("update upload state");
        assert_eq!(require_completed_upload(&Some(upload.clone())), Ok(()));
        assert_eq!(deadline_fault(Some(&upload)), HttpFault::UpstreamTimeout);
        upload
            .publish(UploadState::Failed(HttpFault::MalformedRequest))
            .expect("update upload state");
        assert_eq!(
            upload_fault(Some(&upload)),
            Ok(Some(HttpFault::MalformedRequest))
        );
    }

    #[tokio::test]
    async fn blocking_classification_does_not_stall_the_reactor() {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let relay = relay_with_slots(1, 1);
        let classifier = tokio::spawn(async move {
            relay
                .classify(
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
        tokio::time::timeout(Duration::from_secs(1), tokio::task::yield_now())
            .await
            .expect("reactor progressed while classifier blocked");
        release_tx.send(()).expect("release classifier");
        assert_eq!(classifier.await.expect("join classifier"), Ok(()));
    }

    #[tokio::test]
    async fn classification_waits_for_an_execution_slot() {
        let relay = relay_with_slots(1, 1);
        let held = Arc::clone(&relay.classification_slots)
            .try_acquire_owned()
            .expect("hold classification slot");
        let (entered_tx, mut entered_rx) = tokio::sync::oneshot::channel();
        let classifier = tokio::spawn({
            let relay = Arc::clone(&relay);
            async move {
                relay
                    .classify(
                        tokio::time::Instant::now() + Duration::from_secs(1),
                        move || {
                            entered_tx.send(()).expect("announce classifier entry");
                            Ok(())
                        },
                    )
                    .await
            }
        });

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut entered_rx)
                .await
                .is_err()
        );
        drop(held);
        entered_rx.await.expect("classifier entered after release");
        assert_eq!(classifier.await.expect("join classifier"), Ok(()));
    }

    #[tokio::test]
    async fn classification_deadline_includes_execution_slot_wait() {
        let relay = relay_with_slots(1, 1);
        let _held = Arc::clone(&relay.classification_slots)
            .try_acquire_owned()
            .expect("hold classification slot");
        let ran = Arc::new(AtomicBool::new(false));
        let ran_in_task = Arc::clone(&ran);

        let result = relay
            .classify(
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
        let relay = relay_with_slots(1, 1);
        let budget_permit = Arc::clone(&relay.buffered_budget)
            .try_acquire_owned()
            .expect("reserve buffered memory");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let classifier = tokio::spawn({
            let relay = Arc::clone(&relay);
            async move {
                relay
                    .classify(
                        tokio::time::Instant::now() + Duration::from_millis(50),
                        move || {
                            let _budget = budget_permit;
                            entered_tx.send(()).expect("announce classifier entry");
                            release_rx.recv().expect("release blocking classifier");
                            Ok(())
                        },
                    )
                    .await
            }
        });
        entered_rx.await.expect("classifier entered");

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), classifier)
                .await
                .expect("deadline returned promptly")
                .expect("join classification fixture"),
            Err(HttpFault::UpstreamTimeout)
        );
        assert_eq!(relay.classification_slots.available_permits(), 0);
        assert_eq!(relay.buffered_budget.available_permits(), 0);

        release_tx.send(()).expect("release classifier");
        tokio::time::timeout(Duration::from_secs(1), async {
            while relay.classification_slots.available_permits() == 0
                || relay.buffered_budget.available_permits() == 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking closure eventually releases its resources");
        assert_eq!(relay.classification_slots.available_permits(), 1);
        assert_eq!(relay.buffered_budget.available_permits(), 1);
    }

    #[tokio::test]
    async fn cancelled_waiter_retains_buffer_budget_until_classification_finishes() {
        let budget = Arc::new(Semaphore::new(1));
        let budget_permit = Arc::clone(&budget)
            .try_acquire_owned()
            .expect("reserve buffered memory");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let relay = relay_with_slots(1, 1);
        let classifier = tokio::spawn(async move {
            relay
                .classify(
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
        assert!(classifier.await.expect_err("cancel waiter").is_cancelled());
        assert_eq!(budget.available_permits(), 0);
        release_tx.send(()).expect("release classifier");
        tokio::time::timeout(Duration::from_secs(1), async {
            while budget.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("classifier releases buffered memory");
        assert_eq!(budget.available_permits(), 1);
    }
}
