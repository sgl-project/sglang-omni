use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Response};
use axum::middleware::Next;

use crate::error::HttpFault;

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
    State(request_ids): State<Arc<RequestIds>>,
    mut request: Request,
    next: Next,
) -> Response<Body> {
    let Some((request_id, accepted)) = request_ids.canonicalize(request.headers()) else {
        return HttpFault::InternalError.into_response();
    };
    if !accepted {
        let mut response = HttpFault::MalformedRequest.into_response();
        response
            .headers_mut()
            .insert(REQUEST_ID_HEADER, request_id.0);
        return response;
    }

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

fn valid(value: &HeaderValue) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_REQUEST_ID_BYTES
        && bytes.iter().all(|byte| matches!(byte, 0x21..=0x7e))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use axum::http::{HeaderMap, HeaderValue};

    use super::{RequestIds, valid};

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
}
