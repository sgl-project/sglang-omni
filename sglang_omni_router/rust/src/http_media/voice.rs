use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, Path, State};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::config::VOICE_UPLOAD_BODY_MAX_BYTES;
use crate::error::HttpFault;
use crate::http_relay::{
    BufferedUpload, OutgoingRequest, map_admission, map_dispatch, sanitize_response_headers,
};
use crate::request_id::CanonicalRequestId;

use super::HttpMedia;
use super::headers::{RequestKind, validate_bodyless_request, validate_request};

pub(crate) async fn collection(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    let request_id = request_id.into_header_value();
    let result = if request.method() == Method::GET {
        handle_bodyless(media, request, Method::GET, None, request_id).await
    } else if request.method() == Method::POST {
        handle_upload(media, request, request_id).await
    } else {
        return HttpFault::MethodNotAllowed
            .into_response_with_allow(HeaderValue::from_static("GET, POST"));
    };
    outcome(result)
}

pub(crate) async fn item(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    Path(name): Path<String>,
    request: Request<Body>,
) -> Response<Body> {
    if request.method() != Method::DELETE {
        return HttpFault::MethodNotAllowed
            .into_response_with_allow(HeaderValue::from_static("DELETE"));
    }
    let result = handle_bodyless(
        media,
        request,
        Method::DELETE,
        Some(name),
        request_id.into_header_value(),
    )
    .await;
    outcome(result)
}

fn outcome(result: Result<Response<Body>, HttpFault>) -> Response<Body> {
    match result {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle_bodyless(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    method: Method,
    name: Option<String>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    validate_common(&request)?;
    validate_bodyless_request(request.headers())?;
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let envelope = media
        .pool
        .try_admit_envelope()
        .map_err(map_admission)?;
    let lease = media
        .pool
        .dispatch_voice_control(envelope)
        .map_err(map_dispatch)?;
    send_once(
        media,
        method,
        request.uri().query(),
        name.as_deref(),
        None,
        lease,
        request_id,
        deadline,
    )
    .await
}

async fn handle_upload(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    validate_common(&request)?;
    let framing = validate_request(request.headers(), RequestKind::Multipart)?;
    if framing
        .content_length
        .is_some_and(|length| length > VOICE_UPLOAD_BODY_MAX_BYTES)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let envelope = media
        .pool
        .try_admit_envelope()
        .map_err(map_admission)?;
    let query = request.uri().query().map(str::to_owned);
    let upload = media
        .relay
        .read_buffered(
            request.into_body(),
            framing.content_length,
            VOICE_UPLOAD_BODY_MAX_BYTES,
            deadline,
        )
        .await?;
    let lease = media
        .pool
        .dispatch_voice_control(envelope)
        .map_err(map_dispatch)?;
    send_once(
        media,
        Method::POST,
        query.as_deref(),
        None,
        Some((framing.content_type, upload)),
        lease,
        request_id,
        deadline,
    )
    .await
}

fn validate_common(request: &Request<Body>) -> Result<(), HttpFault> {
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn send_once(
    media: Arc<HttpMedia>,
    method: Method,
    query: Option<&str>,
    name: Option<&str>,
    upload: Option<(HeaderValue, BufferedUpload)>,
    lease: crate::worker_pool::RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    if tokio::time::Instant::now() >= deadline {
        return Err(HttpFault::UpstreamTimeout);
    }
    let mut path = vec![
        String::from("v1"),
        String::from("audio"),
        String::from("voices"),
    ];
    if let Some(name) = name {
        path.push(name.to_owned());
    }
    let (content_type, upload) = upload
        .map(|(content_type, upload)| (Some(content_type), Some(upload)))
        .unwrap_or((None, None));
    let outgoing =
        OutgoingRequest::control(method, path, query.map(str::to_owned), content_type, upload)?;
    Arc::clone(&media.relay)
        .send(
            outgoing,
            lease,
            request_id,
            deadline,
            sanitize_response_headers,
        )
        .await
}
