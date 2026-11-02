use std::collections::HashSet;

use axum::http::header::{
    CONNECTION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, EXPECT, HeaderName, HeaderValue,
    TRAILER, TRANSFER_ENCODING,
};
use axum::http::{HeaderMap, StatusCode};

use crate::error::HttpFault;
use crate::request_id::REQUEST_ID_HEADER;

pub(crate) struct RequestEnvelope {
    pub(crate) content_length: Option<u64>,
    pub(crate) transfer_framed: bool,
}

pub(crate) fn request_content_type(headers: &HeaderMap) -> Result<HeaderValue, HttpFault> {
    let mut values = headers.get_all(CONTENT_TYPE).iter();
    let Some(value) = values.next() else {
        return Err(HttpFault::UnsupportedMediaType);
    };
    if values.next().is_some() {
        return Err(HttpFault::UnsupportedMediaType);
    }
    Ok(value.clone())
}

pub(crate) fn validate_request_envelope(headers: &HeaderMap) -> Result<RequestEnvelope, HttpFault> {
    let mut encodings = headers.get_all(CONTENT_ENCODING).iter();
    let encoding = encodings.next();
    if encodings.next().is_some()
        || encoding.is_some_and(|value| {
            !value
                .to_str()
                .is_ok_and(|text| text.eq_ignore_ascii_case("identity"))
        })
    {
        return Err(HttpFault::UnsupportedContentEncoding);
    }
    let mut expectations = headers.get_all(EXPECT).iter();
    if let Some(expectation) = expectations.next()
        && (!expectation.as_bytes().eq_ignore_ascii_case(b"100-continue")
            || expectations.next().is_some())
    {
        return Err(HttpFault::ExpectationFailed);
    }
    if headers.contains_key(TRAILER) {
        return Err(HttpFault::MalformedRequest);
    }
    let transfer_framed = headers.contains_key(TRANSFER_ENCODING);
    let mut lengths = headers.get_all(CONTENT_LENGTH).iter();
    let length = lengths.next();
    if lengths.next().is_some() || (transfer_framed && length.is_some()) {
        return Err(HttpFault::MalformedRequest);
    }
    let content_length = length
        .map(|value| parse_content_length(value).ok_or(HttpFault::MalformedRequest))
        .transpose()?;
    Ok(RequestEnvelope {
        content_length,
        transfer_framed,
    })
}

pub(crate) fn sanitize_response_headers(
    status: StatusCode,
    source: &HeaderMap,
) -> Result<HeaderMap, HttpFault> {
    let connection = connection_tokens(source)?;
    let chunked = response_is_chunked(source)?;
    if !(status.is_success() || status.is_client_error() || status.is_server_error()) {
        return Err(HttpFault::UpstreamProtocolError);
    }
    let mut types = source.get_all(CONTENT_TYPE).iter();
    let content_type = types.next();
    let content_length = if chunked {
        None
    } else {
        let mut lengths = source.get_all(CONTENT_LENGTH).iter();
        let content_length = lengths.next();
        if lengths.next().is_some() {
            return Err(HttpFault::UpstreamProtocolError);
        }
        content_length
    };
    if types.next().is_some() {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = content_type
        && !value.to_str().is_ok_and(valid_generic_content_type)
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    if let Some(value) = content_length
        && parse_content_length(value).is_none()
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    let mut result = HeaderMap::new();
    for (name, value) in source {
        if strip_response_header(name, &connection) || (chunked && name == CONTENT_LENGTH) {
            continue;
        }
        result.append(name.clone(), value.clone());
    }
    Ok(result)
}

fn strip_response_header(name: &HeaderName, connection: &HashSet<String>) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) || name.as_str() == REQUEST_ID_HEADER
        || connection.contains(name.as_str())
}

fn response_is_chunked(headers: &HeaderMap) -> Result<bool, HttpFault> {
    let mut values = headers.get_all(TRANSFER_ENCODING).iter();
    let Some(value) = values.next() else {
        return Ok(false);
    };
    if values.next().is_some()
        || !value
            .to_str()
            .is_ok_and(|value| value.trim().eq_ignore_ascii_case("chunked"))
    {
        return Err(HttpFault::UpstreamProtocolError);
    }
    Ok(true)
}

pub(crate) fn connection_tokens(headers: &HeaderMap) -> Result<HashSet<String>, HttpFault> {
    let mut result = HashSet::new();
    for value in headers.get_all(CONNECTION) {
        let value = value
            .to_str()
            .map_err(|_| HttpFault::UpstreamProtocolError)?;
        for token in value.split(',') {
            let token = token.trim();
            let name = HeaderName::from_bytes(token.as_bytes())
                .map_err(|_| HttpFault::UpstreamProtocolError)?;
            result.insert(name.as_str().to_owned());
        }
    }
    Ok(result)
}

pub(crate) fn parse_content_length(value: &HeaderValue) -> Option<u64> {
    let text = value.to_str().ok()?;
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

pub(crate) fn is_request_media_type(value: &str, expected: &str) -> bool {
    parse_content_type(value).is_some_and(|media| media.eq_ignore_ascii_case(expected))
}

pub(crate) fn valid_generic_content_type(value: &str) -> bool {
    parse_content_type(value).is_some()
}

fn parse_content_type(value: &str) -> Option<&str> {
    let bytes = value.as_bytes();
    let mut cursor = skip_ows(bytes, 0);
    let media_start = cursor;
    cursor = parse_token(bytes, cursor)?;
    if bytes.get(cursor) != Some(&b'/') {
        return None;
    }
    cursor = parse_token(bytes, cursor + 1)?;
    let media_end = cursor;

    loop {
        cursor = skip_ows(bytes, cursor);
        if cursor == bytes.len() {
            return Some(&value[media_start..media_end]);
        }
        if bytes.get(cursor) != Some(&b';') {
            return None;
        }
        cursor = skip_ows(bytes, cursor + 1);
        cursor = parse_token(bytes, cursor)?;
        if bytes.get(cursor) != Some(&b'=') {
            return None;
        }
        cursor += 1;
        if bytes.get(cursor) == Some(&b'"') {
            cursor = quoted_end(bytes, cursor + 1)?;
        } else {
            cursor = parse_token(bytes, cursor)?;
        }
    }
}

fn parse_token(bytes: &[u8], start: usize) -> Option<usize> {
    let mut cursor = start;
    while bytes.get(cursor).is_some_and(|byte| is_tchar(*byte)) {
        cursor += 1;
    }
    (cursor > start).then_some(cursor)
}

const fn is_tchar(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn quoted_end(bytes: &[u8], mut cursor: usize) -> Option<usize> {
    while let Some(&byte) = bytes.get(cursor) {
        match byte {
            b'"' => return Some(cursor + 1),
            b'\\' => {
                cursor += 1;
                let escaped = *bytes.get(cursor)?;
                if !is_quoted_pair_byte(escaped) {
                    return None;
                }
            }
            byte if !is_qdtext(byte) => return None,
            _ => {}
        }
        cursor += 1;
    }
    None
}

const fn is_qdtext(byte: u8) -> bool {
    matches!(byte, b'\t' | b' ' | b'!' | b'#'..=b'[' | b']'..=b'~')
}

const fn is_quoted_pair_byte(byte: u8) -> bool {
    matches!(byte, b'\t' | b' '..=b'~')
}

fn skip_ows(bytes: &[u8], mut cursor: usize) -> usize {
    while bytes
        .get(cursor)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        cursor += 1;
    }
    cursor
}
