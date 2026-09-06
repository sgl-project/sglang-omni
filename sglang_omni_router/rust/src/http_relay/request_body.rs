use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::body::Body;
use bytes::Bytes;
use http_body::{Frame, SizeHint};
use sync_wrapper::SyncWrapper;
use thiserror::Error;
use tokio::sync::OwnedSemaphorePermit;

use crate::error::HttpFault;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UploadState {
    Incomplete,
    Failed(HttpFault),
    Complete,
}

#[derive(Clone)]
pub(crate) struct SharedUploadState {
    inner: Arc<Mutex<UploadState>>,
}

impl SharedUploadState {
    pub(crate) fn new(state: UploadState) -> Self {
        Self {
            inner: Arc::new(Mutex::new(state)),
        }
    }

    pub(crate) fn snapshot(&self) -> Result<UploadState, HttpFault> {
        self.inner
            .lock()
            .map(|state| *state)
            .map_err(|_| HttpFault::InternalError)
    }

    pub(crate) fn publish(&self, state: UploadState) -> Result<(), HttpFault> {
        *self.inner.lock().map_err(|_| HttpFault::InternalError)? = state;
        Ok(())
    }
}

struct BudgetedBytes {
    data: Bytes,
    _budget: OwnedSemaphorePermit,
}

impl AsRef<[u8]> for BudgetedBytes {
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref()
    }
}

/// One already-classified request body whose shared byte budget follows the
/// payload into the upstream transport.
pub(crate) struct BufferedBody {
    data: Option<Bytes>,
}

impl BufferedBody {
    pub(crate) fn new(data: Bytes, budget: Option<OwnedSemaphorePermit>) -> Self {
        let data = match budget {
            Some(budget) => Bytes::from_owner(BudgetedBytes {
                data,
                _budget: budget,
            }),
            None => data,
        };
        Self { data: Some(data) }
    }
}

impl http_body::Body for BufferedBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Poll::Ready(self.data.take().map(|data| Ok(Frame::data(data))))
    }

    fn is_end_stream(&self) -> bool {
        self.data.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        let remaining = self.data.as_ref().map_or(0, |data| data.len() as u64);
        SizeHint::with_exact(remaining)
    }
}

#[derive(Debug, Error)]
#[error("request upload failed")]
pub(crate) struct UploadError;

pub(crate) struct DirectRequestBody {
    inner: SyncWrapper<Body>,
    expected: Option<u64>,
    maximum: u64,
    observed: u64,
    final_frame_returned: bool,
    state: SharedUploadState,
    terminal: bool,
}

impl DirectRequestBody {
    pub(crate) fn new(
        body: Body,
        expected: Option<u64>,
        maximum: u64,
        state: SharedUploadState,
    ) -> Self {
        Self {
            inner: SyncWrapper::new(body),
            expected,
            maximum,
            observed: 0,
            final_frame_returned: false,
            state,
            terminal: false,
        }
    }

    fn fail(&mut self, fault: HttpFault) -> Poll<Option<Result<Frame<Bytes>, UploadError>>> {
        self.terminal = true;
        let _published = self.state.publish(UploadState::Failed(fault));
        Poll::Ready(Some(Err(UploadError)))
    }
}

impl http_body::Body for DirectRequestBody {
    type Data = Bytes;
    type Error = UploadError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        if self.final_frame_returned {
            self.terminal = true;
            return Poll::Ready(None);
        }
        match Pin::new(self.inner.get_mut()).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(data) => {
                    if data.is_empty() {
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    let Ok(length) = u64::try_from(data.len()) else {
                        return self.fail(HttpFault::RequestBodyTooLarge);
                    };
                    let Some(observed) = self.observed.checked_add(length) else {
                        return self.fail(HttpFault::RequestBodyTooLarge);
                    };
                    if observed > self.maximum
                        || self.expected.is_some_and(|expected| observed > expected)
                    {
                        return self.fail(HttpFault::RequestBodyTooLarge);
                    }
                    self.observed = observed;
                    if self.expected == Some(observed) {
                        // Axum/Hyper owns fixed-length wire framing; do not hold this frame for a synthetic EOF.
                        if self.state.publish(UploadState::Complete).is_err() {
                            return self.fail(HttpFault::InternalError);
                        }
                        self.final_frame_returned = true;
                    }
                    Poll::Ready(Some(Ok(Frame::data(data))))
                }
                Err(_trailers) => self.fail(HttpFault::MalformedRequest),
            },
            Poll::Ready(Some(Err(_source))) => self.fail(HttpFault::MalformedRequest),
            Poll::Ready(None) => {
                self.terminal = true;
                if self
                    .expected
                    .is_some_and(|expected| self.observed != expected)
                {
                    return self.fail(HttpFault::MalformedRequest);
                }
                if self.state.publish(UploadState::Complete).is_err() {
                    return self.fail(HttpFault::InternalError);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.terminal
    }

    fn size_hint(&self) -> SizeHint {
        self.expected.map_or_else(SizeHint::default, |expected| {
            SizeHint::with_exact(expected.saturating_sub(self.observed))
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::future::poll_fn;
    use std::io;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};
    use std::thread;
    use std::time::Duration;

    use super::{BufferedBody, DirectRequestBody, HttpFault, SharedUploadState, UploadState};
    use axum::body::Body;
    use bytes::{Bytes, BytesMut};
    use http_body::{Body as _, Frame};
    use tokio::sync::Semaphore;

    struct Frames(VecDeque<Result<Frame<Bytes>, io::Error>>);

    impl http_body::Body for Frames {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    struct CountedFrames {
        frames: VecDeque<Result<Frame<Bytes>, io::Error>>,
        polls: Arc<AtomicUsize>,
    }

    impl http_body::Body for CountedFrames {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            self.polls.fetch_add(1, Ordering::Relaxed);
            Poll::Ready(self.frames.pop_front())
        }
    }

    async fn drive(body: Body, expected: u64) -> (Bytes, UploadState) {
        let state = SharedUploadState::new(UploadState::Incomplete);
        let mut direct = DirectRequestBody::new(body, Some(expected), expected, state.clone());
        let mut output = BytesMut::new();
        while let Some(Ok(frame)) = poll_fn(|cx| Pin::new(&mut direct).poll_frame(cx)).await {
            output.extend_from_slice(
                &frame
                    .into_data()
                    .expect("direct body only returns data frames"),
            );
        }
        let terminal = state.snapshot().expect("read upload state");
        (output.freeze(), terminal)
    }

    #[tokio::test]
    async fn direct_upload_preserves_frames_and_completes_at_exact_length() {
        let frames = Frames(VecDeque::from([
            Ok(Frame::data(Bytes::from_static(b"ab"))),
            Ok(Frame::data(Bytes::from_static(b"c"))),
        ]));
        let (output, state) = drive(Body::new(frames), 3).await;
        assert_eq!(output, Bytes::from_static(b"abc"));
        assert_eq!(state, UploadState::Complete);

        let (_, short) = drive(Body::from("ab"), 3).await;
        assert_eq!(short, UploadState::Failed(HttpFault::MalformedRequest));
        let (_, long) = drive(Body::from("abcd"), 3).await;
        assert_eq!(long, UploadState::Failed(HttpFault::RequestBodyTooLarge));
    }

    #[test]
    fn buffered_upload_budget_follows_the_last_payload_reference() {
        let budget = Arc::new(Semaphore::new(4));
        let permit = Arc::clone(&budget)
            .try_acquire_many_owned(4)
            .expect("reserve buffered payload");
        let mut body = BufferedBody::new(Bytes::from_static(b"data"), Some(permit));
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);

        let frame = Pin::new(&mut body).poll_frame(&mut context);
        assert!(matches!(&frame, Poll::Ready(Some(Ok(_)))));
        let Poll::Ready(Some(Ok(frame))) = frame else {
            return;
        };
        let payload = frame.into_data().expect("buffered data frame");
        let slice = payload.slice(1..);
        drop(body);
        assert_eq!(budget.available_permits(), 0);

        drop(payload);
        assert_eq!(budget.available_permits(), 0);
        drop(slice);
        assert_eq!(budget.available_permits(), 4);
    }

    #[tokio::test]
    async fn stalled_upstream_retains_buffered_budget_until_send_is_cancelled() {
        const PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stalled upstream");
        let address = listener
            .local_addr()
            .expect("read stalled upstream address");
        let (body_seen_tx, body_seen_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _peer) = listener.accept().expect("accept stalled upload");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("bound stalled upload read");
            let mut request = Vec::new();
            let mut chunk = [0_u8; 4096];
            loop {
                let count = stream.read(&mut chunk).expect("read stalled upload");
                assert_ne!(count, 0, "stalled upload closed before its body");
                request.extend_from_slice(&chunk[..count]);
                if request
                    .windows(4)
                    .position(|part| part == b"\r\n\r\n")
                    .is_some_and(|head_end| request.len() > head_end + 4)
                {
                    let _sent = body_seen_tx.send(());
                    break;
                }
            }
            release_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("release stalled upstream");
        });

        let budget = Arc::new(Semaphore::new(PAYLOAD_BYTES));
        let permits = u32::try_from(PAYLOAD_BYTES).expect("payload fits semaphore permit count");
        let permit = Arc::clone(&budget)
            .try_acquire_many_owned(permits)
            .expect("reserve complete buffered payload");
        let body = BufferedBody::new(Bytes::from(vec![0_u8; PAYLOAD_BYTES]), Some(permit));
        let client = reqwest::Client::builder()
            .http1_only()
            .build()
            .expect("build stalled upstream client");
        let send = tokio::spawn(async move {
            client
                .post(format!("http://{address}/upload"))
                .header("content-length", PAYLOAD_BYTES)
                .body(reqwest::Body::wrap(body))
                .send()
                .await
        });

        tokio::time::timeout(Duration::from_secs(5), body_seen_rx)
            .await
            .expect("upstream observed request body")
            .expect("stalled upstream stayed available");
        let held_while_queued = budget.available_permits() == 0
            && Arc::clone(&budget).try_acquire_many_owned(permits).is_err();

        send.abort();
        let _cancelled = send.await;
        release_tx.send(()).expect("release stalled upstream");
        server.join().expect("join stalled upstream");
        tokio::time::timeout(Duration::from_secs(5), async {
            while budget.available_permits() != PAYLOAD_BYTES {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled upload released its budget");

        assert!(held_while_queued);
    }

    #[tokio::test]
    async fn direct_upload_rejects_trailers_and_body_errors() {
        let trailers = Frames(VecDeque::from([Ok(Frame::trailers(
            axum::http::HeaderMap::new(),
        ))]));
        let (_, trailer_state) = drive(Body::new(trailers), 0).await;
        assert_eq!(
            trailer_state,
            UploadState::Failed(HttpFault::MalformedRequest)
        );
        let errors = Frames(VecDeque::from([Err(io::Error::other("fixture"))]));
        let (_, error_state) = drive(Body::new(errors), 0).await;
        assert_eq!(
            error_state,
            UploadState::Failed(HttpFault::MalformedRequest)
        );
    }

    #[tokio::test]
    async fn direct_upload_skips_empty_data_and_completes_at_exact_length() {
        let polls = Arc::new(AtomicUsize::new(0));
        let state = SharedUploadState::new(UploadState::Incomplete);
        let mut direct = DirectRequestBody::new(
            Body::new(CountedFrames {
                frames: VecDeque::from([
                    Ok(Frame::data(Bytes::new())),
                    Ok(Frame::data(Bytes::from_static(b"ab"))),
                    Ok(Frame::data(Bytes::new())),
                    Ok(Frame::data(Bytes::new())),
                ]),
                polls: Arc::clone(&polls),
            }),
            Some(2),
            2,
            state.clone(),
        );
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);

        assert!(Pin::new(&mut direct).poll_frame(&mut context).is_pending());
        assert_eq!(polls.load(Ordering::Relaxed), 1);
        assert_eq!(
            state.snapshot().expect("read cooperative upload state"),
            UploadState::Incomplete
        );
        let final_frame = match Pin::new(&mut direct).poll_frame(&mut context) {
            Poll::Ready(Some(Ok(frame))) => frame.into_data().ok(),
            Poll::Pending | Poll::Ready(None) | Poll::Ready(Some(Err(_))) => None,
        }
        .expect("exact declared length publishes the final data frame");
        assert_eq!(final_frame, Bytes::from_static(b"ab"));
        assert_eq!(polls.load(Ordering::Relaxed), 2);
        assert_eq!(
            state.snapshot().expect("read completed upload state"),
            UploadState::Complete
        );
        assert!(matches!(
            Pin::new(&mut direct).poll_frame(&mut context),
            Poll::Ready(None)
        ));
        assert_eq!(polls.load(Ordering::Relaxed), 2);

        let empty_frames = Frames(VecDeque::from([
            Ok(Frame::data(Bytes::new())),
            Ok(Frame::data(Bytes::new())),
        ]));
        let empty_state = SharedUploadState::new(UploadState::Incomplete);
        let mut empty =
            DirectRequestBody::new(Body::new(empty_frames), Some(0), 0, empty_state.clone());
        assert!(Pin::new(&mut empty).poll_frame(&mut context).is_pending());
        assert!(Pin::new(&mut empty).poll_frame(&mut context).is_pending());
        assert!(matches!(
            Pin::new(&mut empty).poll_frame(&mut context),
            Poll::Ready(None)
        ));
        assert_eq!(
            empty_state
                .snapshot()
                .expect("read zero-length upload state"),
            UploadState::Complete
        );
    }

    #[tokio::test]
    async fn reqwest_can_reuse_a_connection_after_the_direct_upload_terminal() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind pool fixture");
        let address = listener.local_addr().expect("read pool fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _peer) = listener.accept().expect("accept pooled connection");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound pooled read");
            for _ in 0..2 {
                let mut request = Vec::new();
                let mut chunk = [0_u8; 1024];
                while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                    let count = stream.read(&mut chunk).expect("read pooled request head");
                    assert_ne!(count, 0, "pooled connection closed before next request");
                    request.extend_from_slice(&chunk[..count]);
                }
                while !request.ends_with(b"{}") {
                    let count = stream.read(&mut chunk).expect("read pooled request body");
                    assert_ne!(count, 0, "pooled request body closed early");
                    request.extend_from_slice(&chunk[..count]);
                }
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}",
                    )
                    .expect("write pooled response");
            }
        });
        let client = reqwest::Client::builder()
            .http1_only()
            .build()
            .expect("build pool client");
        for _ in 0..2 {
            let state = SharedUploadState::new(UploadState::Incomplete);
            let body = DirectRequestBody::new(Body::from("{}"), Some(2), 2, state);
            let response = client
                .post(format!("http://{address}/v1/chat/completions"))
                .header("content-length", 2)
                .body(reqwest::Body::wrap(body))
                .send()
                .await
                .expect("send pooled direct body");
            assert_eq!(response.bytes().await.expect("consume pooled body"), "{}");
        }
        server.join().expect("join pooled fixture");
    }
}
