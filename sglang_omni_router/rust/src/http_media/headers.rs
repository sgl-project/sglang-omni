use axum::http::HeaderMap;
use axum::http::HeaderValue;

use crate::error::HttpFault;
use crate::http_relay::{is_request_media_type, request_content_type, validate_request_envelope};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RequestKind {
    Json,
    Multipart,
}

pub(super) struct RequestFraming {
    pub(super) content_length: Option<u64>,
    pub(super) content_type: HeaderValue,
    pub(super) boundary: Option<Vec<u8>>,
    pub(super) route_model: Option<String>,
    pub(super) route_stream: Option<bool>,
}

pub(super) fn validate_request(
    headers: &HeaderMap,
    kind: RequestKind,
) -> Result<RequestFraming, HttpFault> {
    let content_type = request_content_type(headers)?;
    let text = content_type
        .to_str()
        .map_err(|_| HttpFault::UnsupportedMediaType)?;
    let boundary = match kind {
        RequestKind::Json => {
            if !is_request_media_type(text, "application/json") {
                return Err(HttpFault::UnsupportedMediaType);
            }
            None
        }
        RequestKind::Multipart => Some(
            multipart_boundary(text)
                .ok_or(HttpFault::UnsupportedMediaType)?
                .into_bytes(),
        ),
    };
    let envelope = validate_request_envelope(headers)?;
    let route_model = one_route_header(headers, "x-sglang-omni-route-model")?
        .map(str::trim)
        .map(str::to_owned);
    if route_model.as_ref().is_some_and(String::is_empty) {
        return Err(HttpFault::MalformedRequest);
    }
    let route_stream = one_route_header(headers, "x-sglang-omni-route-stream")?
        .map(str::trim)
        .map(|value| {
            if value.eq_ignore_ascii_case("true") {
                Ok(true)
            } else if value.eq_ignore_ascii_case("false") {
                Ok(false)
            } else {
                Err(HttpFault::MalformedRequest)
            }
        })
        .transpose()?;
    Ok(RequestFraming {
        content_length: envelope.content_length,
        content_type,
        boundary,
        route_model,
        route_stream,
    })
}

fn one_route_header<'a>(
    headers: &'a HeaderMap,
    name: &'static str,
) -> Result<Option<&'a str>, HttpFault> {
    let mut values = headers.get_all(name).iter();
    let first = values.next();
    if values.next().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    first
        .map(|value| {
            let text = value.to_str().map_err(|_| HttpFault::MalformedRequest)?;
            if text.len() > 4_096 {
                return Err(HttpFault::MalformedRequest);
            }
            Ok(text)
        })
        .transpose()
}

fn multipart_boundary(value: &str) -> Option<String> {
    let mut parts = value.split(';');
    if !parts
        .next()?
        .trim()
        .eq_ignore_ascii_case("multipart/form-data")
    {
        return None;
    }
    let mut boundary = None;
    let mut charset_seen = false;
    for parameter in parts {
        let (name, raw) = parameter.trim().split_once('=')?;
        let name = name.trim();
        if name.eq_ignore_ascii_case("charset") {
            if charset_seen {
                return None;
            }
            charset_seen = true;
            let raw = raw.trim();
            let charset = if raw.starts_with('"') {
                raw.strip_prefix('"')?.strip_suffix('"')?
            } else {
                raw
            };
            if !charset.eq_ignore_ascii_case("utf-8") {
                return None;
            }
            continue;
        }
        if !name.eq_ignore_ascii_case("boundary") || boundary.is_some() {
            return None;
        }
        let raw = raw.trim();
        let quoted = raw.starts_with('"');
        let parsed = if quoted {
            raw.strip_prefix('"')?.strip_suffix('"')?
        } else {
            if !raw.bytes().all(is_boundary_byte) {
                return None;
            }
            raw
        };
        if parsed.is_empty()
            || parsed.len() > 70
            || !parsed.is_ascii()
            || parsed.as_bytes().last() == Some(&b' ')
            || !parsed
                .bytes()
                .all(|byte| is_boundary_byte(byte) || quoted && byte == b' ')
        {
            return None;
        }
        boundary = Some(parsed.to_owned());
    }
    boundary
}

fn is_boundary_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"'()+_,-./:=?".contains(&byte)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use axum::http::header::{CONNECTION, CONTENT_LENGTH, CONTENT_TYPE};
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use crate::http_relay::sanitize_response_headers as sanitize_response;

    use super::{HttpFault, RequestKind, validate_request};

    fn json_headers(content_type: &[u8]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_bytes(content_type).expect("representable content type fixture"),
        );
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("2"));
        headers
    }

    fn pcm_headers(include_metadata: bool) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("audio/pcm"));
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("4"));
        if include_metadata {
            headers.insert("x-sample-rate", HeaderValue::from_static("24000"));
            headers.insert("x-channels", HeaderValue::from_static("1"));
            headers.insert("x-bit-depth", HeaderValue::from_static("16"));
        }
        headers
    }

    #[test]
    fn json_content_type_accepts_worker_owned_parameters() {
        for value in [
            b"application/json".as_slice(),
            b"application/json; charset=utf-8".as_slice(),
            b"application/json; charset=\"UTF-8\"".as_slice(),
            b"application/json; charset=utf-8; charset=utf-8".as_slice(),
            b"application/json; version=1".as_slice(),
        ] {
            validate_request(&json_headers(value), RequestKind::Json)
                .expect("valid JSON content type");
        }

        for value in [
            b"application/json; charset=\"utf-8".as_slice(),
            b"application/json;".as_slice(),
            b"application/json; charset=\"utf-8\"junk".as_slice(),
        ] {
            assert_eq!(
                validate_request(&json_headers(value), RequestKind::Json).err(),
                Some(HttpFault::UnsupportedMediaType),
                "malformed JSON content type {value:?}"
            );
        }
    }

    #[test]
    fn accepts_quoted_and_unquoted_multipart_boundaries() {
        for (value, expected) in [
            ("multipart/form-data; boundary=abc-123", "abc-123"),
            ("multipart/form-data; boundary=\"abc-123\"", "abc-123"),
            ("multipart/form-data; boundary=\"abc def\"", "abc def"),
            (
                "multipart/form-data; charset=utf-8; boundary=abc-123",
                "abc-123",
            ),
            (
                "multipart/form-data; boundary=abc-123; charset=\"UTF-8\"",
                "abc-123",
            ),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static(value));
            headers.insert(CONTENT_LENGTH, HeaderValue::from_static("10"));
            let framing = validate_request(&headers, RequestKind::Multipart)
                .expect("valid multipart request headers");
            assert_eq!(framing.boundary.as_deref(), Some(expected.as_bytes()));
        }

        for value in [
            "multipart/form-data; boundary=abc def",
            "multipart/form-data; boundary=\"abc def \"",
            "multipart/form-data; boundary=\"abc[def\"",
            "multipart/form-data; boundary=\"abc\\def\"",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static(value));
            headers.insert(CONTENT_LENGTH, HeaderValue::from_static("10"));
            assert_eq!(
                validate_request(&headers, RequestKind::Multipart).err(),
                Some(HttpFault::UnsupportedMediaType)
            );
        }
        let mut control = HeaderMap::new();
        control.insert(
            CONTENT_TYPE,
            HeaderValue::from_bytes(b"multipart/form-data; boundary=\"abc\tdef\"")
                .expect("horizontal tab is representable in a field value"),
        );
        control.insert(CONTENT_LENGTH, HeaderValue::from_static("10"));
        assert_eq!(
            validate_request(&control, RequestKind::Multipart).err(),
            Some(HttpFault::UnsupportedMediaType)
        );
    }

    #[test]
    fn route_model_and_stream_headers_are_bounded_singular_facts() {
        let mut headers = json_headers(b"application/json");
        headers.insert(
            "x-sglang-omni-route-model",
            HeaderValue::from_static(" asr "),
        );
        headers.insert(
            "x-sglang-omni-route-stream",
            HeaderValue::from_static("TrUe"),
        );
        let framing = validate_request(&headers, RequestKind::Json).expect("route facts");
        assert_eq!(framing.route_model.as_deref(), Some("asr"));
        assert_eq!(framing.route_stream, Some(true));
        headers.append(
            "x-sglang-omni-route-stream",
            HeaderValue::from_static("true"),
        );
        assert_eq!(
            validate_request(&headers, RequestKind::Json).err(),
            Some(HttpFault::MalformedRequest)
        );
        headers.remove("x-sglang-omni-route-stream");
        headers.insert(
            "x-sglang-omni-route-stream",
            HeaderValue::from_static("yes"),
        );
        assert_eq!(
            validate_request(&headers, RequestKind::Json).err(),
            Some(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn preserves_worker_media_type_and_rejects_redirects() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("audio/x-private"));
        assert_eq!(
            sanitize_response(StatusCode::FOUND, &headers),
            Err(HttpFault::UpstreamProtocolError)
        );
        let sanitized = sanitize_response(StatusCode::OK, &headers)
            .expect("worker-owned response type is relayable");
        assert_eq!(
            sanitized.get(CONTENT_TYPE),
            Some(&HeaderValue::from_static("audio/x-private"))
        );
    }

    #[test]
    fn speech_preserves_finish_reason_unless_connection_nominated() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
        headers.insert("x-finish-reason", HeaderValue::from_static("length"));

        let sanitized = sanitize_response(StatusCode::OK, &headers).expect("speech finish reason");
        assert_eq!(
            sanitized.get("x-finish-reason"),
            Some(&HeaderValue::from_static("length"))
        );

        headers.insert(CONNECTION, HeaderValue::from_static("x-finish-reason"));
        let sanitized = sanitize_response(StatusCode::OK, &headers)
            .expect("connection-nominated speech metadata");
        assert!(!sanitized.contains_key("x-finish-reason"));
    }

    #[test]
    fn pcm_metadata_is_optional_and_worker_owned() {
        let complete_source = pcm_headers(true);
        let complete = sanitize_response(StatusCode::OK, &complete_source)
            .expect("streaming PCM complete metadata");
        assert_eq!(
            complete.get("x-sample-rate"),
            complete_source.get("x-sample-rate")
        );

        assert!(sanitize_response(StatusCode::OK, &pcm_headers(false)).is_ok());
        for worker_owned in ["", "0", "-1", "+1", "1.0", " 1"] {
            let mut headers = pcm_headers(true);
            headers.insert(
                "x-bit-depth",
                HeaderValue::from_str(worker_owned).expect("valid header syntax"),
            );
            let sanitized = sanitize_response(StatusCode::OK, &headers)
                .expect("worker-owned PCM metadata is relayable");
            assert_eq!(sanitized.get("x-bit-depth"), headers.get("x-bit-depth"));
        }
        let mut duplicate = pcm_headers(true);
        duplicate.append("x-sample-rate", HeaderValue::from_static("48000"));
        let sanitized = sanitize_response(StatusCode::OK, &duplicate)
            .expect("duplicate worker metadata is relayable");
        assert_eq!(sanitized.get_all("x-sample-rate").iter().count(), 2);
        let mut nominated = pcm_headers(true);
        nominated.insert(CONNECTION, HeaderValue::from_static("x-sample-rate"));
        let sanitized = sanitize_response(StatusCode::OK, &nominated)
            .expect("connection-nominated metadata is stripped");
        assert!(!sanitized.contains_key("x-sample-rate"));
    }

    #[test]
    fn non_pcm_response_preserves_worker_metadata() {
        let mut exact = pcm_headers(true);
        exact.insert(CONTENT_TYPE, HeaderValue::from_static("audio/opus"));
        let sanitized = sanitize_response(StatusCode::OK, &exact).expect("non-PCM response");
        for name in ["x-sample-rate", "x-channels", "x-bit-depth"] {
            assert_eq!(sanitized.get(name), exact.get(name));
        }
    }
}
