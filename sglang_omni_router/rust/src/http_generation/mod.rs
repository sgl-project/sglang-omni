mod classify;
mod headers;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::http_relay::{
    HttpRelay, OutgoingRequest, map_admission, map_dispatch, sanitize_response_headers,
};
use crate::request_id::CanonicalRequestId;
use crate::worker_pool::{CapacityClass, TrustDomain, WorkerPool};

use classify::classify;
use headers::{canonical_content_type, validate_request};

pub(crate) const CHAT_PATH: &str = "/v1/chat/completions";

pub(crate) struct HttpGeneration {
    pool: Arc<WorkerPool>,
    relay: Arc<HttpRelay>,
    trust: TrustDomain,
    buffered_max: u64,
    streamed_max: u64,
    request_timeout: std::time::Duration,
}

impl HttpGeneration {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
        relay: Arc<HttpRelay>,
    ) -> Result<Option<Arc<Self>>, RouterError> {
        let Some(http_generation) = config.http_generation.as_ref() else {
            return Ok(None);
        };
        Ok(Some(Arc::new(Self {
            pool,
            relay,
            trust: TrustDomain::new(http_generation.trust_domain.clone()),
            buffered_max: http_generation.buffered_request_max_bytes,
            streamed_max: http_generation.streamed_request_max_bytes,
            request_timeout: http_generation.request_timeout(),
        })))
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
    if framing
        .content_length
        .is_some_and(|length| length > maximum)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let admission = generation
        .pool
        .try_admit(CapacityClass::GenerationHttp, 1)
        .map_err(map_admission)?;

    if let Some(proof) = proof {
        let lease = proof.dispatch(admission).map_err(map_dispatch)?;
        let outgoing = OutgoingRequest::direct(
            CHAT_PATH,
            canonical_content_type(),
            request.into_body(),
            framing.content_length,
            generation.streamed_max,
        );
        return Arc::clone(&generation.relay)
            .send(
                outgoing,
                lease,
                request_id,
                deadline,
                sanitize_response_headers,
            )
            .await;
    }

    let upload = generation
        .relay
        .read_buffered(
            request.into_body(),
            framing.content_length,
            generation.buffered_max,
            deadline,
        )
        .await?;
    let classify_trust = generation.trust.clone();
    let (upload, classified) = generation
        .relay
        .classify(deadline, move || {
            let classified = classify(&upload.bytes, &classify_trust)?;
            Ok((upload, classified))
        })
        .await?;
    let lease = generation
        .pool
        .dispatch(admission, &classified.requirement)
        .map_err(map_dispatch)?;
    let outgoing = OutgoingRequest::buffered(CHAT_PATH, canonical_content_type(), upload)?;
    Arc::clone(&generation.relay)
        .send(
            outgoing,
            lease,
            request_id,
            deadline,
            sanitize_response_headers,
        )
        .await
}
