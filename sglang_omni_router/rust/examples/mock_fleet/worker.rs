use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Query, Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::config::{Behavior, Result, Worker};

pub(super) fn contains(profile: &Value, field: &str, value: &str) -> bool {
    profile[field]
        .as_array()
        .is_some_and(|values| values.iter().any(|item| item.as_str() == Some(value)))
}

pub(super) fn supports(worker: &Worker, service: &str, request: &Value) -> bool {
    worker
        .profiles
        .iter()
        .any(|profile| profile_matches(worker, profile, service, request))
}

fn uses_uploaded_voice(worker: &Worker, service: &str, request: &Value) -> bool {
    worker.profiles.iter().any(|profile| {
        profile["voice_name_policy"] == "uploaded"
            && profile_matches(worker, profile, service, request)
    })
}

fn profile_matches(worker: &Worker, profile: &Value, service: &str, request: &Value) -> bool {
    if profile["service"] != service {
        return false;
    }
    if service == "realtime_websocket" {
        return request["model"]
            .as_str()
            .is_none_or(|model| model == worker.default_model);
    }
    let model = request["model"].as_str().unwrap_or(&worker.default_model);
    if !contains(profile, "model_ids", model) {
        return false;
    }
    let stream_field = if service == "speech_websocket" {
        "stream_audio"
    } else {
        "stream"
    };
    let stream_mode = if request[stream_field].as_bool().unwrap_or(false) {
        "streaming"
    } else {
        "non_streaming"
    };
    if service != "speech_batch" && !contains(profile, "stream_modes", stream_mode) {
        return false;
    }
    match service {
        "generation_http" => generation_matches(profile, request),
        "transcription_http" => {
            profile["task"] == request["task"]
                && contains(
                    profile,
                    "response_formats",
                    request["response_format"].as_str().unwrap_or("json"),
                )
        }
        "speech_batch" => request["items"].as_array().is_some_and(|items| {
            !items.is_empty()
                && items.len() as u64 <= profile["max_batch_size"].as_u64().unwrap_or(0)
                && items.iter().all(|item| {
                    let mut merged = request.clone();
                    if let (Some(target), Some(source)) = (merged.as_object_mut(), item.as_object())
                    {
                        target.extend(source.clone());
                    } else {
                        return false;
                    }
                    contains(
                        profile,
                        "model_ids",
                        merged["model"].as_str().unwrap_or(&worker.default_model),
                    ) && speech_matches(profile, &merged)
                })
        }),
        "speech_http" | "speech_websocket" => speech_matches(profile, request),
        _ => false,
    }
}

fn generation_matches(profile: &Value, request: &Value) -> bool {
    let mut inputs = BTreeSet::new();
    for (field, modality) in [
        ("images", "image"),
        ("audios", "audio"),
        ("videos", "video"),
    ] {
        if request[field]
            .as_array()
            .is_some_and(|items| !items.is_empty())
        {
            if !contains(profile, "media_placements", "top_level") {
                return false;
            }
            inputs.insert(modality);
        }
    }
    if let Some(messages) = request["messages"].as_array() {
        for message in messages {
            match &message["content"] {
                Value::String(_) => {
                    if !contains(profile, "message_content_forms", "string") {
                        return false;
                    }
                    inputs.insert("text");
                }
                Value::Array(parts) => {
                    if !contains(profile, "message_content_forms", "typed_parts") {
                        return false;
                    }
                    for part in parts {
                        let modality = match part["type"].as_str() {
                            Some("text") => "text",
                            Some("image" | "image_url") => "image",
                            Some("input_audio" | "audio_url") => "audio",
                            Some("video" | "video_url") => "video",
                            _ => return false,
                        };
                        if modality != "text"
                            && !contains(profile, "media_placements", "typed_parts")
                        {
                            return false;
                        }
                        inputs.insert(modality);
                    }
                }
                Value::Null => {}
                _ => return false,
            }
        }
    }
    if !inputs
        .iter()
        .all(|input| contains(profile, "input_modalities", input))
    {
        return false;
    }
    let outputs = request["modalities"]
        .as_array()
        .filter(|values| !values.is_empty());
    if let Some(outputs) = outputs {
        if !outputs.iter().all(|output| {
            output
                .as_str()
                .is_some_and(|value| contains(profile, "output_modalities", value))
        }) {
            return false;
        }
        if outputs.iter().any(|output| output == "audio") {
            return contains(
                profile,
                "chat_audio_formats",
                request["audio"]["format"].as_str().unwrap_or("wav"),
            );
        }
        true
    } else {
        contains(profile, "output_modalities", "text")
    }
}

fn speech_matches(profile: &Value, request: &Value) -> bool {
    let format = request["response_format"].as_str().unwrap_or("wav");
    if !contains(profile, "response_formats", format) {
        return false;
    }
    if let Some(task) = request["task_type"].as_str() {
        let task = match task.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
            "base" => "voice_clone",
            "customvoice" => "text_to_speech",
            "voicedesign" => "voice_design",
            _ => return false,
        };
        if !contains(profile, "tasks", task) {
            return false;
        }
    }
    let mut references = Vec::new();
    if request["ref_audio"].is_string() {
        references.push("direct");
    }
    if let Some(items) = request["references"]
        .as_array()
        .filter(|items| !items.is_empty())
    {
        for item in items {
            references.push(if item.get("vq_codes").is_some() {
                "vq_codes"
            } else {
                "list"
            });
        }
    }
    if references.is_empty() {
        references.push("none");
    }
    references
        .iter()
        .all(|reference| contains(profile, "reference_forms", reference))
}

struct MockState {
    worker: Worker,
    accepted: AtomicU64,
    rejected: AtomicU64,
    health: AtomicU16,
    status: AtomicU16,
    voices: Mutex<BTreeSet<String>>,
    counts: Mutex<BTreeMap<String, u64>>,
}

pub(super) fn app(worker: Worker) -> Router {
    let limit = worker.behavior.max_request_bytes;
    let state = Arc::new(MockState {
        health: AtomicU16::new(worker.behavior.health_status),
        status: AtomicU16::new(worker.behavior.response_status),
        worker,
        accepted: AtomicU64::new(0),
        rejected: AtomicU64::new(0),
        voices: Mutex::new(BTreeSet::new()),
        counts: Mutex::new(BTreeMap::new()),
    });
    Router::new()
        .route("/v1/audio/speech/stream", axum::routing::get(speech_socket))
        .route("/v1/realtime", axum::routing::get(realtime_socket))
        .fallback(handle)
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
}

async fn speech_socket(State(state): State<Arc<MockState>>, upgrade: WebSocketUpgrade) -> Response {
    if !state
        .worker
        .profiles
        .iter()
        .any(|profile| profile["service"] == "speech_websocket")
    {
        return fault(
            &state,
            StatusCode::UNPROCESSABLE_ENTITY,
            "speech sessions unsupported",
        );
    }
    let session_state = Arc::clone(&state);
    decorate(
        upgrade
            .max_message_size(16 * 1024 * 1024)
            .on_upgrade(move |socket| session(socket, session_state, false)),
        &state,
        "speech_websocket",
        None,
    )
}

async fn realtime_socket(
    State(state): State<Arc<MockState>>,
    Query(query): Query<BTreeMap<String, String>>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let mut request = json!({});
    if let Some(model) = query.get("model") {
        request["model"] = json!(model);
    }
    if !supports(&state.worker, "realtime_websocket", &request) {
        return fault(
            &state,
            StatusCode::UNPROCESSABLE_ENTITY,
            "realtime model unsupported",
        );
    }
    let session_state = Arc::clone(&state);
    decorate(
        upgrade
            .max_message_size(16 * 1024 * 1024)
            .on_upgrade(move |socket| session(socket, session_state, true)),
        &state,
        "realtime_websocket",
        None,
    )
}

async fn session(mut socket: WebSocket, state: Arc<MockState>, realtime: bool) {
    let mut response_format = "pcm".to_owned();
    let mut streaming = true;
    if !realtime {
        let initial = tokio::time::timeout(Duration::from_secs(5), socket.next()).await;
        let valid = if let Ok(Some(Ok(Message::Text(text)))) = initial {
            if let Ok(body) = serde_json::from_str::<Value>(&text) {
                response_format = body["response_format"].as_str().unwrap_or("wav").to_owned();
                streaming = body["stream_audio"].as_bool().unwrap_or(false);
                let name = body["voice"].as_str().or_else(|| body["speaker"].as_str());
                let uploaded = uses_uploaded_voice(&state.worker, "speech_websocket", &body);
                let known_voice = match name {
                    Some(name) if uploaded => state.voices.lock().await.contains(name),
                    _ => true,
                };
                known_voice
                    && body["type"] == "session.config"
                    && supports(&state.worker, "speech_websocket", &body)
            } else {
                false
            }
        } else {
            false
        };
        if !valid {
            state.rejected.fetch_add(1, Ordering::Relaxed);
            let _sent = socket.send(Message::Text(json!({"type":"error","message":"unsupported mock session configuration","mock":identity(&state.worker)}).to_string().into())).await;
            let _closed = socket.close().await;
            return;
        }
    }
    state.accepted.fetch_add(1, Ordering::Relaxed);
    let service = if realtime {
        "realtime_websocket"
    } else {
        "speech_websocket"
    };
    *state
        .counts
        .lock()
        .await
        .entry(service.to_owned())
        .or_default() += 1;
    let configured = json!({"type":if realtime {"session.created"} else {"session.configured"},"session":{"id":"mock-session","model":state.worker.default_model},"mock":identity(&state.worker)});
    if socket
        .send(Message::Text(configured.to_string().into()))
        .await
        .is_err()
    {
        return;
    }
    while let Ok(Some(Ok(message))) =
        tokio::time::timeout(Duration::from_secs(60), socket.next()).await
    {
        match message {
            Message::Text(text) => {
                let Ok(event) = serde_json::from_str::<Value>(&text) else {
                    break;
                };
                let kind = event["type"].as_str().unwrap_or_default();
                if matches!(kind, "response.create" | "input.text" | "text") {
                    let behavior = &state.worker.behavior;
                    if behavior.first_packet_delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(behavior.first_packet_delay_ms))
                            .await;
                    }
                    let payload = audio_bytes(behavior, &response_format);
                    let chunk_bytes = if streaming {
                        behavior.chunk_bytes
                    } else {
                        payload.len()
                    };
                    for (index, audio) in payload.chunks(chunk_bytes).enumerate() {
                        if behavior.disconnect_after_chunks == Some(index) {
                            return;
                        }
                        if index > 0 && behavior.chunk_interval_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(behavior.chunk_interval_ms))
                                .await;
                        }
                        let message = if realtime {
                            Message::Text(json!({"type":"response.audio.delta","delta":STANDARD.encode(audio),"mock":identity(&state.worker)}).to_string().into())
                        } else {
                            Message::Binary(Bytes::copy_from_slice(audio))
                        };
                        if socket.send(message).await.is_err() {
                            return;
                        }
                    }
                    if socket.send(Message::Text(json!({"type":if realtime {"response.done"} else {"audio.done"},"mock":identity(&state.worker)}).to_string().into())).await.is_err() { return; }
                } else {
                    let event = json!({"type":if kind == "response.cancel" {"response.done"} else {"mock.ack"},"mock":identity(&state.worker)});
                    if socket
                        .send(Message::Text(event.to_string().into()))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
            Message::Close(frame) => {
                let _closed = socket.send(Message::Close(frame)).await;
                return;
            }
            Message::Ping(payload) => {
                if socket.send(Message::Pong(payload)).await.is_err() {
                    return;
                }
            }
            Message::Binary(_) | Message::Pong(_) => {}
        }
    }
    let _closed = socket.close().await;
}

fn identity(worker: &Worker) -> Value {
    json!({"worker_id":worker.id,"default_model":worker.default_model,"profiles":worker.profiles})
}

fn decorate(
    mut response: Response,
    state: &MockState,
    service: &str,
    request_id: Option<HeaderValue>,
) -> Response {
    if let Ok(value) = HeaderValue::from_str(&state.worker.id) {
        response.headers_mut().insert("x-mock-worker", value);
    }
    if let Ok(value) = HeaderValue::from_str(service) {
        response.headers_mut().insert("x-mock-service", value);
    }
    let modalities: BTreeSet<&str> = state
        .worker
        .profiles
        .iter()
        .filter_map(|profile| profile["input_modalities"].as_array())
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if let Ok(value) = HeaderValue::from_str(&modalities.into_iter().collect::<Vec<_>>().join(","))
    {
        response.headers_mut().insert("x-mock-modalities", value);
    }
    if let Some(value) = request_id {
        response.headers_mut().insert("x-mock-request-id", value);
    }
    response
}

fn fault(state: &MockState, code: StatusCode, message: &str) -> Response {
    state.rejected.fetch_add(1, Ordering::Relaxed);
    decorate((code, Json(json!({"error":{"message":message,"type":"mock_rejection"},"mock":identity(&state.worker)}))).into_response(), state, "rejection", None)
}

async fn parse_body(request: Request, limit: usize) -> Result<Value> {
    if request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"))
    {
        let mut multipart = Multipart::from_request(request, &()).await?;
        let mut fields = serde_json::Map::new();
        let mut size = 0_usize;
        while let Some(field) = multipart.next_field().await? {
            let name = field
                .name()
                .ok_or("multipart field name required")?
                .to_owned();
            let is_file = field.file_name().is_some();
            let bytes = field.bytes().await?;
            size = size.saturating_add(bytes.len());
            if size > limit {
                return Err("multipart body too large".into());
            }
            if fields.contains_key(&name) {
                return Err("duplicate multipart field".into());
            }
            fields.insert(
                name.clone(),
                if is_file {
                    json!({"bytes":bytes.len()})
                } else if name == "stream" {
                    Value::Bool(std::str::from_utf8(&bytes)?.parse()?)
                } else {
                    Value::String(String::from_utf8(bytes.to_vec())?)
                },
            );
        }
        Ok(Value::Object(fields))
    } else {
        let bytes = to_bytes(request.into_body(), limit).await?;
        if bytes.is_empty() {
            Ok(json!({}))
        } else {
            Ok(serde_json::from_slice(&bytes)?)
        }
    }
}

async fn handle(State(state): State<Arc<MockState>>, request: Request) -> Response {
    let path = request.uri().path().to_owned();
    let method = request.method().clone();
    let query = request.uri().query().unwrap_or_default().to_owned();
    let request_id = request.headers().get("x-request-id").cloned();
    if path == state.worker.health_path {
        return StatusCode::from_u16(state.health.load(Ordering::Relaxed))
            .unwrap_or(StatusCode::SERVICE_UNAVAILABLE)
            .into_response();
    }
    if path == "/__mock/stats" && method == "GET" {
        return Json(json!({"mock":identity(&state.worker),"accepted":state.accepted.load(Ordering::Relaxed),
            "rejected":state.rejected.load(Ordering::Relaxed),"services":*state.counts.lock().await})).into_response();
    }
    if path == "/__mock/control" && method == "POST" {
        return match parse_body(request, 4096).await {
            Ok(body) => {
                for (field, target) in [
                    ("health_status", &state.health),
                    ("response_status", &state.status),
                ] {
                    if let Some(value) = body.get(field) {
                        let Some(code) = value.as_u64().filter(|code| (200..=599).contains(code))
                        else {
                            return fault(
                                &state,
                                StatusCode::BAD_REQUEST,
                                "invalid control status",
                            );
                        };
                        target.store(code as u16, Ordering::Relaxed);
                    }
                }
                Json(json!({"ok":true})).into_response()
            }
            Err(error) => fault(&state, StatusCode::BAD_REQUEST, &error.to_string()),
        };
    }
    let service = match path.as_str() {
        "/v1/chat/completions" => "generation_http",
        "/v1/audio/speech" => "speech_http",
        "/v1/audio/speech/batch" => "speech_batch",
        "/v1/audio/transcriptions" | "/v1/audio/translations" => "transcription_http",
        "/v1/audio/voices" => "voice_control",
        path if path.starts_with("/v1/audio/voices/") => "voice_control",
        _ => return fault(&state, StatusCode::NOT_FOUND, "unknown mock route"),
    };
    if service != "voice_control" && method != "POST" {
        return fault(&state, StatusCode::METHOD_NOT_ALLOWED, "POST required");
    }
    let mut body = match parse_body(request, state.worker.behavior.max_request_bytes).await {
        Ok(body) => body,
        Err(error) => return fault(&state, StatusCode::BAD_REQUEST, &error.to_string()),
    };
    if !body.is_object() {
        return fault(&state, StatusCode::BAD_REQUEST, "object required");
    }
    if service == "transcription_http" {
        body["task"] = json!(if path.ends_with("translations") {
            "translate"
        } else {
            "transcribe"
        });
        if body.get("file").is_none() {
            return fault(&state, StatusCode::BAD_REQUEST, "multipart file required");
        }
    }
    if service == "voice_control" {
        if !state.worker.voice_owner {
            return fault(&state, StatusCode::NOT_FOUND, "not a voice owner");
        }
    } else if !supports(&state.worker, service, &body) {
        return fault(
            &state,
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported model, modality, task, format or stream mode",
        );
    }
    if let Some(name) = body["voice"].as_str().or_else(|| body["speaker"].as_str()) {
        let uploaded = uses_uploaded_voice(&state.worker, service, &body);
        if uploaded && !state.voices.lock().await.contains(name) {
            return fault(&state, StatusCode::NOT_FOUND, "unknown uploaded voice");
        }
    }
    state.accepted.fetch_add(1, Ordering::Relaxed);
    *state
        .counts
        .lock()
        .await
        .entry(service.to_owned())
        .or_default() += 1;
    let status = state.status.load(Ordering::Relaxed);
    if status != 200 {
        return fault(
            &state,
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            "injected upstream failure",
        );
    }
    let behavior = &state.worker.behavior;
    let response = match service {
        "generation_http" => chat(&state.worker, &body),
        "speech_http" => audio(behavior, body["response_format"].as_str().unwrap_or("wav")),
        "speech_batch" => {
            let results: Vec<Value> = body["items"].as_array().into_iter().flatten().enumerate().map(|(index, item)| {
                let format = item["response_format"].as_str().or(body["response_format"].as_str()).unwrap_or("wav");
                json!({"index":index,"audio":STANDARD.encode(audio_bytes(behavior, format)),"format":format,"mock":identity(&state.worker)})
            }).collect();
            Json(json!({"results":results,"mock":identity(&state.worker)})).into_response()
        }
        "transcription_http" => {
            let text = format!("mock transcription by {}", state.worker.id);
            let format = body["response_format"].as_str().unwrap_or("json");
            if body["stream"] == true || format == "sse" {
                streamed(
                    vec![
                        Bytes::from(format!(
                            "data: {}\n\n",
                            json!({"type":"transcript.text.delta","delta":text})
                        )),
                        Bytes::from_static(b"data: [DONE]\n\n"),
                    ],
                    "text/event-stream",
                    behavior,
                )
            } else if format == "text" {
                ([("content-type", "text/plain")], text).into_response()
            } else {
                Json(json!({"text":text,"mock":identity(&state.worker)})).into_response()
            }
        }
        "voice_control" => {
            let mut voices = state.voices.lock().await;
            if method == "GET" {
                if query == "names_only=true" {
                    Json(json!({"voices":*voices})).into_response()
                } else {
                    Json(json!({"voices":*voices,"mock":identity(&state.worker)})).into_response()
                }
            } else if method == "POST" {
                let Some(name) = body["name"]
                    .as_str()
                    .filter(|name| !name.is_empty() && name.len() <= 128)
                else {
                    return fault(&state, StatusCode::BAD_REQUEST, "voice name required");
                };
                if voices.len() >= 1024 {
                    return fault(
                        &state,
                        StatusCode::TOO_MANY_REQUESTS,
                        "mock voice capacity exceeded",
                    );
                }
                voices.insert(name.to_owned());
                Json(json!({"name":name,"mock":identity(&state.worker)})).into_response()
            } else if method == "DELETE" {
                let name = path.strip_prefix("/v1/audio/voices/").unwrap_or_default();
                if !voices.remove(name) {
                    return fault(&state, StatusCode::NOT_FOUND, "unknown voice");
                }
                Json(json!({"deleted":true,"name":name})).into_response()
            } else {
                return fault(
                    &state,
                    StatusCode::METHOD_NOT_ALLOWED,
                    "unsupported voice method",
                );
            }
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    };
    decorate(response, &state, service, request_id)
}

fn chat(worker: &Worker, request: &Value) -> Response {
    let model = request["model"].as_str().unwrap_or(&worker.default_model);
    let has_audio = request["modalities"]
        .as_array()
        .is_some_and(|values| values.iter().any(|value| value == "audio"));
    let format = request["audio"]["format"].as_str().unwrap_or("wav");
    let behavior = &worker.behavior;
    if request["stream"] == true {
        let mut chunks = Vec::new();
        let audio = if has_audio {
            audio_bytes(behavior, format)
        } else {
            Vec::new()
        };
        for index in 0..behavior.chunks {
            let delta = if has_audio {
                let start = index * audio.len() / behavior.chunks;
                let end = (index + 1) * audio.len() / behavior.chunks;
                json!({"audio":{"data":STANDARD.encode(&audio[start..end])}})
            } else {
                json!({"content":format!("{}:{}",worker.id,"x".repeat(behavior.chunk_bytes))})
            };
            chunks.push(Bytes::from(format!("data: {}\n\n", json!({"id":"mock-chat","object":"chat.completion.chunk","model":model,"choices":[{"index":0,"delta":delta,"finish_reason":null}],"mock":identity(worker)}))));
        }
        chunks.push(Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n",json!({"id":"mock-chat","object":"chat.completion.chunk","model":model,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}))));
        streamed(chunks, "text/event-stream", behavior)
    } else {
        let mut message = json!({"role":"assistant","content":format!("{}:{}", worker.id, "x".repeat(behavior.chunk_bytes * behavior.chunks))});
        if has_audio {
            message["audio"] = json!({"id":"mock-audio","data":STANDARD.encode(audio_bytes(behavior, format)),"transcript":"mock","expires_at":0});
        }
        let body = json!({"id":"mock-chat","object":"chat.completion","model":model,"choices":[{"index":0,"message":message,"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"mock":identity(worker)});
        streamed(
            vec![Bytes::from(body.to_string())],
            "application/json",
            behavior,
        )
    }
}

fn audio_bytes(behavior: &Behavior, format: &str) -> Vec<u8> {
    let pcm = vec![0_u8; behavior.chunk_bytes * behavior.chunks];
    if format == "wav" { wav(&pcm) } else { pcm }
}

fn audio(behavior: &Behavior, format: &str) -> Response {
    let bytes = audio_bytes(behavior, format);
    let chunks = bytes
        .chunks(behavior.chunk_bytes)
        .map(Bytes::copy_from_slice)
        .collect();
    let mut response = streamed(
        chunks,
        if format == "wav" {
            "audio/wav"
        } else {
            "audio/pcm"
        },
        behavior,
    );
    response
        .headers_mut()
        .insert("x-audio-sample-rate", HeaderValue::from_static("24000"));
    response
        .headers_mut()
        .insert("x-audio-channels", HeaderValue::from_static("1"));
    response
}

pub(super) fn wav(pcm: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(44 + pcm.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&24000_u32.to_le_bytes());
    bytes.extend_from_slice(&48000_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    bytes.extend_from_slice(pcm);
    bytes
}

fn streamed(chunks: Vec<Bytes>, content_type: &'static str, behavior: &Behavior) -> Response {
    let behavior = behavior.clone();
    let stream = futures_util::stream::unfold(
        (chunks.into_iter(), 0_usize, behavior),
        |(mut chunks, index, behavior)| async move {
            if behavior.disconnect_after_chunks == Some(index) {
                return Some((
                    Err(std::io::Error::other("injected mock stream failure")),
                    (Vec::new().into_iter(), index + 1, behavior),
                ));
            }
            let chunk = chunks.next()?;
            let delay = if index == 0 {
                behavior.first_packet_delay_ms
            } else {
                behavior.chunk_interval_ms
            };
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            Some((
                Ok::<_, std::io::Error>(chunk),
                (chunks, index + 1, behavior),
            ))
        },
    );
    ([("content-type", content_type)], Body::from_stream(stream)).into_response()
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn unrelated_uploaded_profile_does_not_reject_preset_voice() -> crate::config::Result<()>
    {
        let mut worker = text_worker();
        worker.profiles = vec![
            json!({"service":"speech_http","model_ids":["text"],"response_formats":["pcm"],"stream_modes":["non_streaming"],"tasks":["text_to_speech"],"reference_forms":["none"],"voice_name_policy":"preset"}),
        ];
        let mut uploaded = worker.profiles[0].clone();
        uploaded["model_ids"] = json!(["another-model"]);
        uploaded["voice_name_policy"] = json!("uploaded");
        worker.profiles.push(uploaded);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move { axum::serve(listener, app(worker)).await });
        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/audio/speech"))
            .json(&json!({"model":"text","response_format":"pcm","voice":"preset-name"}))
            .send()
            .await?;
        let status = response.status();
        task.abort();
        let _stopped = task.await;
        assert_eq!(status, 200);
        Ok(())
    }

    #[tokio::test]
    async fn wav_websocket_returns_wav_not_pcm() -> crate::config::Result<()> {
        let mut worker = text_worker();
        worker.profiles = vec![
            json!({"service":"speech_websocket","model_ids":["text"],"response_formats":["wav"],"stream_modes":["non_streaming"],"tasks":["text_to_speech"],"reference_forms":["none"],"voice_name_policy":"preset"}),
        ];
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move { axum::serve(listener, app(worker)).await });
        let (mut socket, _) =
            tokio_tungstenite::connect_async(format!("ws://{address}/v1/audio/speech/stream"))
                .await?;
        socket.send(tokio_tungstenite::tungstenite::Message::Text(json!({"type":"session.config","model":"text","response_format":"wav","stream_audio":false}).to_string().into())).await?;
        let _setup = socket.next().await.ok_or("setup missing")??;
        socket
            .send(tokio_tungstenite::tungstenite::Message::Text(
                json!({"type":"input.text","text":"test"})
                    .to_string()
                    .into(),
            ))
            .await?;
        let audio = socket.next().await.ok_or("audio missing")??.into_data();
        task.abort();
        let _stopped = task.await;
        assert!(audio.starts_with(b"RIFF"));
        Ok(())
    }

    #[test]
    fn speech_rejects_unsupported_task_reference_and_format() {
        let mut worker = text_worker();
        worker.profiles = vec![
            json!({"service":"speech_http","model_ids":["text"],"response_formats":["pcm"],"stream_modes":["streaming"],"tasks":["text_to_speech"],"reference_forms":["none"],"voice_name_policy":"preset"}),
        ];
        let mut body =
            json!({"model":"text","stream":true,"response_format":"pcm","task_type":"CustomVoice"});
        assert!(supports(&worker, "speech_http", &body));
        body["response_format"] = json!("mp3");
        assert!(!supports(&worker, "speech_http", &body));
        body["response_format"] = json!("pcm");
        body["ref_audio"] = json!("reference.wav");
        assert!(!supports(&worker, "speech_http", &body));
        body.as_object_mut()
            .map(|object| object.remove("ref_audio"));
        body["task_type"] = json!("Base");
        assert!(!supports(&worker, "speech_http", &body));
    }

    use super::*;
    use crate::config::{Behavior, Worker};
    use serde_json::json;

    fn text_worker() -> Worker {
        Worker {
            id: "text-1".to_owned(),
            trust_domain: "local".to_owned(),
            default_model: "text".to_owned(),
            health_path: "/health".to_owned(),
            behavior: Behavior::default(),
            voice_owner: false,
            profiles: vec![json!({
                "service":"generation_http", "model_ids":["text"],
                "message_content_forms":["string","typed_parts"], "media_placements":["typed_parts"],
                "input_modalities":["text"], "output_modalities":["text"],
                "chat_audio_formats":[], "stream_modes":["non_streaming","streaming"]
            })],
        }
    }

    #[test]
    fn text_worker_rejects_wrong_model_and_image_requests() {
        let worker = text_worker();
        assert!(supports(
            &worker,
            "generation_http",
            &json!({"model":"text","messages":[{"content":"hello"}]})
        ));
        assert!(!supports(
            &worker,
            "generation_http",
            &json!({"model":"other","messages":[{"content":"hello"}]})
        ));
        assert!(!supports(
            &worker,
            "generation_http",
            &json!({"model":"text","messages":[{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]})
        ));
    }

    #[test]
    fn cannot_combine_capabilities_from_different_profile_rows() {
        let mut worker = text_worker();
        let mut vision = worker.profiles[0].clone();
        vision["input_modalities"] = json!(["image"]);
        vision["stream_modes"] = json!(["non_streaming"]);
        worker.profiles.push(vision);
        assert!(!supports(
            &worker,
            "generation_http",
            &json!({"model":"text","stream":true,"messages":[{"content":[{"type":"image_url"}]}]})
        ));
    }

    #[test]
    fn wav_is_real_pcm_with_consistent_lengths() {
        let bytes = wav(&[0; 128]);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 172);
        assert_eq!(&bytes[40..44], &128_u32.to_le_bytes());
    }

    #[tokio::test]
    async fn websocket_setup_pins_identity_and_rejects_wrong_model() -> crate::config::Result<()> {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;
        let mut worker = text_worker();
        worker.profiles = vec![
            json!({"service":"speech_websocket","model_ids":["text"],"response_formats":["pcm"],"stream_modes":["streaming"],"tasks":["text_to_speech"],"reference_forms":["none"],"voice_name_policy":"preset"}),
        ];
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move { axum::serve(listener, app(worker)).await });
        let (mut socket, _) =
            tokio_tungstenite::connect_async(format!("ws://{address}/v1/audio/speech/stream"))
                .await?;
        socket.send(Message::Text(json!({"type":"session.config","model":"text","response_format":"pcm","stream_audio":true}).to_string().into())).await?;
        let event = socket.next().await.ok_or("missing setup event")??;
        let event: serde_json::Value = serde_json::from_str(event.to_text()?)?;
        assert_eq!(event["type"], "session.configured");
        assert_eq!(event["mock"]["worker_id"], "text-1");
        socket.close(None).await?;
        let (mut socket, _) =
            tokio_tungstenite::connect_async(format!("ws://{address}/v1/audio/speech/stream"))
                .await?;
        socket.send(Message::Text(json!({"type":"session.config","model":"wrong","response_format":"pcm","stream_audio":true}).to_string().into())).await?;
        let event = socket.next().await.ok_or("missing error event")??;
        let event: serde_json::Value = serde_json::from_str(event.to_text()?)?;
        assert_eq!(event["type"], "error");
        task.abort();
        let _stopped = task.await;
        Ok(())
    }

    #[tokio::test]
    async fn socket_worker_identifies_itself_and_rejects_unsupported_input()
    -> crate::config::Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move { axum::serve(listener, app(text_worker())).await });
        let client = reqwest::Client::new();
        let good = client
            .post(format!("http://{address}/v1/chat/completions"))
            .json(&json!({"model":"text","messages":[{"content":"hello"}]}))
            .send()
            .await?;
        assert_eq!(good.status(), 200);
        assert_eq!(good.headers()["x-mock-worker"], "text-1");
        assert!(
            good.json::<serde_json::Value>().await?["choices"][0]["message"]["content"]
                .as_str()
                .is_some()
        );
        let bad = client
            .post(format!("http://{address}/v1/chat/completions"))
            .json(&json!({"model":"text","images":["mock"]}))
            .send()
            .await?;
        assert_eq!(bad.status(), 422);
        task.abort();
        let _stopped = task.await;
        Ok(())
    }
}
