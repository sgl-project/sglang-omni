mod classify;
mod headers;
mod multipart;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::config::{Config, HttpMediaRoute};
use crate::error::{HttpFault, RouterError};
use crate::http_relay::{
    HttpRelay, OutgoingRequest, map_admission, map_dispatch, sanitize_response_headers,
};
use crate::request_id::CanonicalRequestId;
use crate::worker_pool::{CapacityClass, TrustDomain, WorkerPool};

use classify::Classified;
use headers::RequestKind;

const SPEECH_PATH: &str = "/v1/audio/speech";
const BATCH_PATH: &str = "/v1/audio/speech/batch";
const TRANSCRIPTION_PATH: &str = "/v1/audio/transcriptions";
const TRANSLATION_PATH: &str = "/v1/audio/translations";

pub(crate) struct HttpMedia {
    pool: Arc<WorkerPool>,
    relay: Arc<HttpRelay>,
    trust: TrustDomain,
    enabled_routes: Box<[HttpMediaRoute]>,
    buffered_max: u64,
    streamed_max: u64,
    request_timeout: std::time::Duration,
}

impl HttpMediaRoute {
    pub(crate) const fn path(self) -> &'static str {
        match self {
            Self::Speech => SPEECH_PATH,
            Self::SpeechBatch => BATCH_PATH,
            Self::Transcription => TRANSCRIPTION_PATH,
            Self::Translation => TRANSLATION_PATH,
        }
    }

    const fn capacity(self) -> CapacityClass {
        match self {
            Self::Speech => CapacityClass::SpeechHttp,
            Self::SpeechBatch => CapacityClass::SpeechBatch,
            Self::Transcription | Self::Translation => CapacityClass::TranscriptionHttp,
        }
    }

    const fn request_kind(self) -> RequestKind {
        match self {
            Self::Speech | Self::SpeechBatch => RequestKind::Json,
            Self::Transcription | Self::Translation => RequestKind::Multipart,
        }
    }
}

impl HttpMedia {
    pub(crate) fn build(
        config: &Config,
        pool: Arc<WorkerPool>,
        relay: Arc<HttpRelay>,
    ) -> Result<Option<Arc<Self>>, RouterError> {
        let Some(media) = config.http_media.as_ref() else {
            return Ok(None);
        };
        Ok(Some(Arc::new(Self {
            pool,
            relay,
            trust: TrustDomain::new(media.trust_domain.clone()),
            enabled_routes: media.routes.clone().into_boxed_slice(),
            buffered_max: media.buffered_request_max_bytes,
            streamed_max: media.streamed_request_max_bytes,
            request_timeout: media.request_timeout(),
        })))
    }

    pub(crate) fn enables(&self, route: HttpMediaRoute) -> bool {
        self.enabled_routes.contains(&route)
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.pool
            .media_http_ready(&self.trust, &self.enabled_routes)
    }
}

pub(crate) async fn speech(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::Speech, request_id).await
}

pub(crate) async fn batch(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::SpeechBatch, request_id).await
}

pub(crate) async fn transcription(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::Transcription, request_id).await
}

pub(crate) async fn translation(
    State(media): State<Arc<HttpMedia>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    outcome(media, request, HttpMediaRoute::Translation, request_id).await
}

async fn outcome(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    route: HttpMediaRoute,
    request_id: CanonicalRequestId,
) -> Response<Body> {
    match handle(media, request, route, request_id.into_header_value()).await {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle(
    media: Arc<HttpMedia>,
    request: Request<Body>,
    route: HttpMediaRoute,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    if request.method() != Method::POST {
        return Err(HttpFault::MethodNotAllowed);
    }
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    if request.uri().path() != route.path() || request.uri().query().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    let deadline = tokio::time::Instant::now() + media.request_timeout;
    let framing = headers::validate_request(request.headers(), route.request_kind())?;
    let direct_proof = if route != HttpMediaRoute::SpeechBatch
        && framing.route_model.is_none()
        && framing.route_stream.is_none()
        && framing
            .content_length
            .is_none_or(|length| length <= media.streamed_max)
    {
        media.pool.content_blind_media_http(&media.trust, route)
    } else {
        None
    };
    let maximum = if direct_proof.is_some() {
        media.streamed_max
    } else {
        media.buffered_max
    };
    if framing
        .content_length
        .is_some_and(|length| length > maximum)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let mut envelope = Some(media.pool.try_admit_envelope().map_err(map_admission)?);
    let mut admission = if route == HttpMediaRoute::SpeechBatch {
        None
    } else {
        Some(
            media
                .pool
                .try_admit_class(
                    envelope.take().ok_or(HttpFault::InternalError)?,
                    route.capacity(),
                    1,
                )
                .map_err(map_admission)?,
        )
    };
    if let Some(proof) = direct_proof {
        let lease = proof
            .dispatch(admission.take().ok_or(HttpFault::InternalError)?)
            .map_err(map_dispatch)?;
        let outgoing = OutgoingRequest::direct(
            route.path(),
            framing.content_type,
            request.into_body(),
            framing.content_length,
            media.streamed_max,
        );
        return Arc::clone(&media.relay)
            .send(
                outgoing,
                lease,
                request_id,
                deadline,
                sanitize_response_headers,
            )
            .await;
    }
    let upload = media
        .relay
        .read_buffered(
            request.into_body(),
            framing.content_length,
            media.buffered_max,
            deadline,
        )
        .await?;
    let content_type = framing.content_type;
    let boundary = framing.boundary;
    let route_model = framing.route_model;
    let route_stream = framing.route_stream;
    let classify_trust = media.trust.clone();
    let (upload, classified) = media
        .relay
        .classify(deadline, move || {
            let classified = classify(
                route,
                &upload.bytes,
                boundary.as_deref(),
                route_model.as_deref(),
                route_stream,
                &classify_trust,
            )?;
            Ok((upload, classified))
        })
        .await?;
    if route == HttpMediaRoute::SpeechBatch
        && !media
            .pool
            .supports_speech_batch_size(&media.trust, classified.credits)
    {
        return Err(HttpFault::NoCompatibleWorker);
    }
    let admission = match admission {
        Some(admission) => admission,
        None => media
            .pool
            .try_admit_class(
                envelope.take().ok_or(HttpFault::InternalError)?,
                route.capacity(),
                classified.credits,
            )
            .map_err(map_admission)?,
    };
    let lease = media
        .pool
        .dispatch(admission, &classified.requirement)
        .map_err(map_dispatch)?;
    let outgoing = OutgoingRequest::buffered(route.path(), content_type, upload)?;
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

fn classify(
    route: HttpMediaRoute,
    bytes: &[u8],
    boundary: Option<&[u8]>,
    route_model: Option<&str>,
    route_stream: Option<bool>,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    match route {
        HttpMediaRoute::Speech => {
            classify::speech_with_hints(bytes, route_model, route_stream, trust)
        }
        HttpMediaRoute::SpeechBatch => {
            classify::batch_with_hints(bytes, route_model, route_stream, trust)
        }
        HttpMediaRoute::Transcription => classify::transcription_with_hints(
            bytes,
            boundary.ok_or(HttpFault::UnsupportedMediaType)?,
            route_model,
            route_stream,
            trust,
        ),
        HttpMediaRoute::Translation => classify::translation_with_hints(
            bytes,
            boundary.ok_or(HttpFault::UnsupportedMediaType)?,
            route_model,
            route_stream,
            trust,
        ),
    }
}
