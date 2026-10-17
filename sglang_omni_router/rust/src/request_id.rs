use std::hash::{BuildHasher, RandomState};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Response};
use axum::middleware::Next;
use bytes::Bytes;
use http_body::{Frame, SizeHint};

use crate::error::HttpFault;
use crate::http_relay::RelayedResponse;
use crate::metrics::{HttpRoute, RouterMetrics};

pub(crate) const REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_REQUEST_ID_BYTES: usize = 128;

/// Canonical request identity established once at the outer service boundary.
#[derive(Clone)]
pub(crate) struct CanonicalRequestId(HeaderValue);

impl CanonicalRequestId {
    pub(crate) fn into_header_value(self) -> HeaderValue {
        self.0
    }
}

/// Process-wide canonical request-ID authority.
pub(crate) struct RequestIds {
    prefix: String,
    sequence: AtomicU64,
}

pub(crate) struct RequestBoundary {
    request_ids: Arc<RequestIds>,
    metrics: Arc<RouterMetrics>,
}

struct BoundaryObservation<'a> {
    metrics: &'a RouterMetrics,
    route: HttpRoute,
    started: Instant,
    completed: bool,
}

struct FirstPayloadBody {
    inner: Body,
    metrics: Arc<RouterMetrics>,
    route: HttpRoute,
    request_started: Option<Instant>,
}

impl http_body::Body for FirstPayloadBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let frame = Pin::new(&mut self.inner).poll_frame(context);
        if let Poll::Ready(Some(Ok(frame))) = &frame
            && frame.data_ref().is_some_and(|bytes| !bytes.is_empty())
            && let Some(request_started) = self.request_started.take()
        {
            self.metrics
                .record_first_payload_duration(self.route, request_started.elapsed());
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        http_body::Body::is_end_stream(&self.inner)
    }

    fn size_hint(&self) -> SizeHint {
        http_body::Body::size_hint(&self.inner)
    }
}

impl<'a> BoundaryObservation<'a> {
    fn new(metrics: &'a RouterMetrics, route: HttpRoute) -> Self {
        Self {
            metrics,
            route,
            started: Instant::now(),
            completed: false,
        }
    }

    fn complete<B>(&mut self, response: &Response<B>) {
        self.metrics.record_response(self.route, response);
        self.metrics
            .record_response_header_duration(self.route, self.started.elapsed());
        self.completed = true;
    }
}

impl Drop for BoundaryObservation<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.metrics.record_cancelled_before_headers(self.route);
        }
    }
}

impl RequestBoundary {
    pub(crate) fn new(request_ids: Arc<RequestIds>, metrics: Arc<RouterMetrics>) -> Arc<Self> {
        Arc::new(Self {
            request_ids,
            metrics,
        })
    }
}

impl RequestIds {
    pub(crate) fn new() -> Arc<Self> {
        let process_id = std::process::id();
        let nonce = RandomState::new().hash_one(process_id);
        Arc::new(Self {
            prefix: format!("sglang-omni-{process_id}-{nonce:016x}"),
            sequence: AtomicU64::new(0),
        })
    }

    fn canonicalize(&self, headers: &HeaderMap) -> Option<(CanonicalRequestId, bool)> {
        let mut values = headers.get_all(REQUEST_ID_HEADER).iter();
        match (values.next(), values.next()) {
            (Some(value), None) if valid(value) => Some((CanonicalRequestId(value.clone()), true)),
            (None, None) => self.generate().map(|request_id| (request_id, true)),
            _ => self.generate().map(|request_id| (request_id, false)),
        }
    }

    fn generate(&self) -> Option<CanonicalRequestId> {
        let sequence = self
            .sequence
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .ok()?;
        let value = HeaderValue::from_str(&format!("{}-{sequence}", self.prefix)).ok()?;
        Some(CanonicalRequestId(value))
    }
}

pub(crate) async fn canonicalize(
    State(boundary): State<Arc<RequestBoundary>>,
    mut request: Request,
    next: Next,
) -> Response<Body> {
    let route = HttpRoute::from_path(request.uri().path());
    let mut observation = BoundaryObservation::new(&boundary.metrics, route);
    boundary.metrics.record_request(route);
    let mut response = match boundary.request_ids.canonicalize(request.headers()) {
        None => HttpFault::InternalError.into_response(),
        Some((request_id, false)) => {
            let mut response = HttpFault::MalformedRequest.into_response();
            response
                .headers_mut()
                .insert(REQUEST_ID_HEADER, request_id.0);
            response
        }
        Some((request_id, true)) => {
            request
                .headers_mut()
                .insert(REQUEST_ID_HEADER, request_id.0.clone());
            request.extensions_mut().insert(request_id.clone());
            let mut response = next.run(request).await;
            response
                .headers_mut()
                .insert(REQUEST_ID_HEADER, request_id.0);
            response
        }
    };
    observation.complete(&response);
    if response
        .extensions_mut()
        .remove::<RelayedResponse>()
        .is_some()
    {
        response = response.map(|inner| {
            Body::new(FirstPayloadBody {
                inner,
                metrics: Arc::clone(&boundary.metrics),
                route,
                request_started: Some(observation.started),
            })
        });
    }
    response
}

fn valid(value: &HeaderValue) -> bool {
    valid_request_id_bytes(value.as_bytes())
}

pub(crate) fn valid_request_id_bytes(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && bytes.len() <= MAX_REQUEST_ID_BYTES
        && bytes.iter().all(|byte| matches!(byte, 0x21..=0x7e))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::collections::{HashSet, VecDeque};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::task::{Context, Poll, Waker};
    use std::time::{Duration, Instant};

    use axum::body::Body;
    use axum::http::{HeaderMap, HeaderValue, Response};
    use bytes::Bytes;
    use http_body::{Body as _, Frame};

    use super::{BoundaryObservation, FirstPayloadBody, RequestIds, valid};
    use crate::metrics::{HttpRoute, RouterMetrics};

    #[test]
    fn missing_valid_and_invalid_ids_have_one_authority() {
        let ids = RequestIds::new();
        let (generated, accepted) = ids
            .canonicalize(&HeaderMap::new())
            .expect("sequence remains available");
        assert!(accepted);
        let generated = generated.into_header_value();
        let generated = generated.to_str().expect("generated ID is visible ASCII");
        let fields: Vec<_> = generated.split('-').collect();
        assert_eq!(fields[..2], ["sglang", "omni"]);
        assert_eq!(fields[2], std::process::id().to_string());
        assert_eq!(fields[3].len(), 16);
        assert!(fields[3].bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(fields[4], "0");

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("caller,visible;1"));
        let (preserved, accepted) = ids
            .canonicalize(&headers)
            .expect("sequence remains available");
        assert!(accepted);
        assert_eq!(preserved.into_header_value(), "caller,visible;1");

        headers.append("x-request-id", HeaderValue::from_static("caller-2"));
        let (_replacement, accepted) = ids
            .canonicalize(&headers)
            .expect("sequence remains available");
        assert!(!accepted);

        assert!(!valid(&HeaderValue::from_static("")));
        assert!(!valid(&HeaderValue::from_static("has space")));
        let oversized = HeaderValue::from_str(&"x".repeat(129)).expect("valid header syntax");
        assert!(!valid(&oversized));
    }

    #[test]
    fn generated_ids_are_unique_under_concurrency_and_exhaustion_does_not_wrap() {
        let ids = RequestIds::new();
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let ids = Arc::clone(&ids);
                std::thread::spawn(move || {
                    (0..64)
                        .map(|_| {
                            ids.generate()
                                .expect("sequence remains available")
                                .into_header_value()
                                .to_str()
                                .expect("generated ID is visible ASCII")
                                .to_owned()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let generated: HashSet<_> = handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("join generator thread"))
            .collect();
        assert_eq!(generated.len(), 1_024);

        let exhausted = RequestIds {
            prefix: String::from("test"),
            sequence: AtomicU64::new(u64::MAX),
        };
        assert!(exhausted.generate().is_none());
        assert!(exhausted.generate().is_none());
    }

    #[test]
    fn boundary_observation_distinguishes_response_headers_from_cancellation() {
        let metrics = RouterMetrics::new();
        {
            let _cancelled = BoundaryObservation::new(&metrics, HttpRoute::Chat);
        }
        assert_eq!(metrics.cancelled_before_headers(HttpRoute::Chat), 1);
        assert_eq!(metrics.response_header_duration(HttpRoute::Chat).count(), 0);

        {
            let mut completed = BoundaryObservation::new(&metrics, HttpRoute::Chat);
            completed.complete(&Response::new(Body::empty()));
        }
        assert_eq!(metrics.cancelled_before_headers(HttpRoute::Chat), 1);
        assert_eq!(metrics.response_header_duration(HttpRoute::Chat).count(), 1);
    }

    struct Frames(VecDeque<Result<Frame<Bytes>, std::io::Error>>);

    impl http_body::Body for Frames {
        type Data = Bytes;
        type Error = std::io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    #[test]
    fn first_payload_observation_preserves_frames_and_counts_nonempty_data_once() {
        let metrics = RouterMetrics::new();
        let mut trailers = HeaderMap::new();
        trailers.insert("x-trailer", HeaderValue::from_static("preserved"));
        let mut body = FirstPayloadBody {
            inner: Body::new(Frames(VecDeque::from([
                Ok(Frame::data(Bytes::new())),
                Ok(Frame::trailers(trailers.clone())),
                Ok(Frame::data(Bytes::from_static(b"one"))),
                Ok(Frame::data(Bytes::from_static(b"two"))),
                Err(std::io::Error::other("body failure")),
            ]))),
            metrics: Arc::clone(&metrics),
            route: HttpRoute::Chat,
            request_started: Some(Instant::now() - Duration::from_secs(2)),
        };
        let mut context = Context::from_waker(Waker::noop());
        for (index, expected_count) in [0, 0, 1, 1, 1].into_iter().enumerate() {
            let Poll::Ready(Some(frame)) = Pin::new(&mut body).poll_frame(&mut context) else {
                panic!("expected a ready frame");
            };
            match index {
                0 => assert_eq!(frame.expect("empty data").data_ref(), Some(&Bytes::new())),
                1 => assert_eq!(
                    frame.expect("trailers").into_trailers().expect("trailers"),
                    trailers
                ),
                2 | 3 => assert_eq!(
                    frame.expect("payload").into_data().expect("data"),
                    if index == 2 { "one" } else { "two" }
                ),
                _ => assert_eq!(frame.expect_err("body error").to_string(), "body failure"),
            }
            assert_eq!(
                metrics.first_payload_duration(HttpRoute::Chat).count(),
                expected_count
            );
        }
        drop(body);
        assert_eq!(metrics.first_payload_duration(HttpRoute::Chat).count(), 1);
        assert!(metrics.first_payload_duration(HttpRoute::Chat).sum_micros >= 2_000_000);
        assert_eq!(metrics.first_payload_duration(HttpRoute::Speech).count(), 0);
    }

    #[test]
    fn no_payload_means_no_sample_and_body_hints_are_preserved() {
        let error = Frames(VecDeque::from([Err(std::io::Error::other(
            "before payload",
        ))]));
        let trailers = Frames(VecDeque::from([Ok(Frame::trailers(HeaderMap::new()))]));
        for (inner, should_poll) in [
            (Body::empty(), true),
            (Body::from("unpolled"), false),
            (Body::new(error), true),
            (Body::new(trailers), true),
        ] {
            let metrics = RouterMetrics::new();
            let expected_size = inner.size_hint().exact();
            let expected_end = inner.is_end_stream();
            let mut body = FirstPayloadBody {
                inner,
                metrics: Arc::clone(&metrics),
                route: HttpRoute::Chat,
                request_started: Some(Instant::now()),
            };
            assert_eq!(body.size_hint().exact(), expected_size);
            assert_eq!(body.is_end_stream(), expected_end);
            if should_poll {
                let frame = Pin::new(&mut body).poll_frame(&mut Context::from_waker(Waker::noop()));
                assert!(!matches!(frame, Poll::Ready(Some(Ok(frame))) if frame.is_data()));
            }
            drop(body);
            assert_eq!(metrics.first_payload_duration(HttpRoute::Chat).count(), 0);
        }
    }
}
