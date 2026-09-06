#![allow(clippy::expect_used, clippy::panic)]

//! Real-socket ordering and exact-replay tests for terminating WebSockets.

use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri};
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as ClientMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-websocket-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create websocket test directory");
        Self(path)
    }

    fn config(&self, contents: &str) -> PathBuf {
        let path = self.0.join("router.toml");
        fs::write(&path, contents).expect("write websocket router config");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _removed = fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _killed = self.0.kill();
        let _waited = self.0.wait();
    }
}

#[derive(Clone)]
struct WorkerState {
    speech_config: Arc<Mutex<Option<String>>>,
    speech_request_id: Arc<Mutex<Option<String>>>,
    speech_close_code: Arc<AtomicUsize>,
    realtime_path: Arc<Mutex<Option<String>>>,
    realtime_release: Arc<Notify>,
    realtime_control: Arc<Notify>,
}

#[derive(Clone)]
struct SetupDeadlineWorkerState {
    speech_attempts: Arc<AtomicUsize>,
    realtime_attempts: Arc<AtomicUsize>,
    speech_early_frames: Arc<AtomicUsize>,
    realtime_early_frames: Arc<AtomicUsize>,
    speech_close_code: Arc<AtomicUsize>,
    realtime_close_code: Arc<AtomicUsize>,
    health_probes: Arc<AtomicUsize>,
}

const REALTIME_FLOOD: &str = r#"{"type":"test.flood"}"#;
const REALTIME_CONTROL: &str = r#"{"type":"response.cancel"}"#;

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn setup_deadline_health(State(state): State<SetupDeadlineWorkerState>) -> StatusCode {
    state.health_probes.fetch_add(1, Ordering::Relaxed);
    StatusCode::OK
}

async fn speech_worker(
    State(state): State<WorkerState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    *state.speech_request_id.lock().await = headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    upgrade.on_upgrade(move |mut socket| async move {
        if let Some(Ok(Message::Text(text))) = socket.next().await {
            *state.speech_config.lock().await = Some(text.to_string());
            let setup_event = if !text.contains(r#""type":"session.config""#)
                || text.contains(r#""response_format":"future""#)
                || text.contains(r#""task_type":"future""#)
            {
                r#"{"type":"error","message":"invalid worker configuration"}"#
            } else {
                r#"{"type":"session.configured","worker":"pinned"}"#
            };
            let _sent = socket
                .send(Message::Text(setup_event.into()))
                .await;
            if text.contains(r#""active":true"#) {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let _sent = socket.send(Message::Binary(vec![9].into())).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
                let _sent = socket.send(Message::Binary(vec![10].into())).await;
            }
            while let Some(message) = socket.next().await {
                match message {
                    Ok(Message::Text(text)) => {
                        if socket.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Ok(Message::Binary(_)) => {
                        let _sent = socket
                            .send(Message::Text(
                                r#"{"type":"error","message":"speech WebSocket client messages must be text frames"}"#.into(),
                            ))
                            .await;
                    }
                    Ok(Message::Close(frame)) => {
                        if let Some(frame) = &frame {
                            state
                                .speech_close_code
                                .store(usize::from(frame.code), Ordering::Relaxed);
                        }
                        let _closed = socket.send(Message::Close(frame)).await;
                        break;
                    }
                    Ok(Message::Ping(_) | Message::Pong(_)) => {}
                    Err(_) => break,
                }
            }
        }
    })
}

async fn realtime_worker(
    State(state): State<WorkerState>,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    *state.realtime_path.lock().await = Some(uri.to_string());
    state.realtime_release.notified().await;
    upgrade.on_upgrade(move |socket| async move {
        let (mut sink, mut stream) = socket.split();
        let _sent = sink
            .send(Message::Text(
                r#"{"type":"session.created","session":{"model":"omni"}}"#.into(),
            ))
            .await;
        while let Some(message) = stream.next().await {
            match message {
                Ok(Message::Text(text)) if text.as_str() == REALTIME_FLOOD => {
                    let payload = axum::extract::ws::Utf8Bytes::from("x".repeat(64 * 1024));
                    let flood = async {
                        loop {
                            if sink.send(Message::Text(payload.clone())).await.is_err() {
                                return;
                            }
                        }
                    };
                    let control = async {
                        while let Some(message) = stream.next().await {
                            match message {
                                Ok(Message::Text(text)) if text.as_str() == REALTIME_CONTROL => {
                                    state.realtime_control.notify_one();
                                    return;
                                }
                                Ok(Message::Close(_)) | Err(_) => return,
                                _ => {}
                            }
                        }
                    };
                    tokio::pin!(flood, control);
                    tokio::select! {
                        () = &mut flood => {}
                        () = &mut control => {}
                    }
                    return;
                }
                Ok(Message::Text(text)) => {
                    if sink.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
                Ok(Message::Close(_)) => break,
                Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {}
                Err(_) => break,
            }
        }
    })
}

async fn setup_deadline_speech_worker(
    State(state): State<SetupDeadlineWorkerState>,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    upgrade.on_upgrade(move |mut socket| async move {
        let Some(Ok(Message::Text(_config))) = socket.next().await else {
            return;
        };
        state.speech_attempts.fetch_add(1, Ordering::Relaxed);
        if uri.query() == Some("case=stall") {
            while let Some(message) = socket.next().await {
                if let Ok(Message::Close(frame)) = message {
                    state.speech_close_code.store(
                        frame.as_ref().map_or(1000, |frame| usize::from(frame.code)),
                        Ordering::Relaxed,
                    );
                    let _closed = socket.send(Message::Close(frame)).await;
                    break;
                }
            }
            return;
        }
        let early = uri.query() == Some("case=early");
        if early {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if socket
            .send(Message::Text(
                r#"{"type":"session.configured","worker":"reused"}"#.into(),
            ))
            .await
            .is_err()
        {
            return;
        }
        if early {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let Some(Ok(message @ (Message::Text(_) | Message::Binary(_)))) = socket.next().await
            else {
                return;
            };
            state.speech_early_frames.fetch_add(1, Ordering::Relaxed);
            let bytes = match message {
                Message::Text(text) => text.len(),
                Message::Binary(bytes) => bytes.len(),
                _ => unreachable!(),
            };
            if socket
                .send(Message::Text(
                    format!(r#"{{"type":"early.received","bytes":{bytes}}}"#).into(),
                ))
                .await
                .is_err()
            {
                return;
            }
        }
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Close(frame)) => {
                    let _closed = socket.send(Message::Close(frame)).await;
                    return;
                }
                Ok(_) => {}
                Err(_) => return,
            }
        }
    })
}

async fn setup_deadline_realtime_worker(
    State(state): State<SetupDeadlineWorkerState>,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    upgrade.on_upgrade(move |mut socket| async move {
        state.realtime_attempts.fetch_add(1, Ordering::Relaxed);
        if uri.query() == Some("case=stall") {
            while let Some(message) = socket.next().await {
                if let Ok(Message::Close(frame)) = message {
                    state.realtime_close_code.store(
                        frame.as_ref().map_or(1000, |frame| usize::from(frame.code)),
                        Ordering::Relaxed,
                    );
                    let _closed = socket.send(Message::Close(frame)).await;
                    break;
                }
            }
            return;
        }
        let early = uri.query() == Some("case=early");
        if early {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if socket
            .send(Message::Text(
                r#"{"type":"session.created","session":{"model":"omni"}}"#.into(),
            ))
            .await
            .is_err()
        {
            return;
        }
        if early {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let Some(Ok(Message::Text(text))) = socket.next().await else {
                return;
            };
            state.realtime_early_frames.fetch_add(1, Ordering::Relaxed);
            let bytes = text.len();
            if socket
                .send(Message::Text(
                    format!(r#"{{"type":"early.received","bytes":{bytes}}}"#).into(),
                ))
                .await
                .is_err()
            {
                return;
            }
        }
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Close(frame)) => {
                    let _closed = socket.send(Message::Close(frame)).await;
                    return;
                }
                Ok(_) => {}
                Err(_) => return,
            }
        }
    })
}

fn router_config(router: SocketAddr, worker: SocketAddr) -> String {
    format!(
        r#"schema_version = 1

[server]
listen = "{router}"

[shutdown]
drain_timeout_ms = 5000

[logging]
format = "json"
filter = "error"

[router]
strategy = "round_robin"

[admission]
global = 8
speech_websocket = 1
realtime_websocket = 1

[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1

[websocket]
connect_timeout_ms = 5000
worker_setup_timeout_ms = 5000

[websocket.speech]
trust_domain = "local"

[websocket.realtime]
trust_domain = "local"

[[workers]]
worker_id = "worker"
base_url = "http://{worker}"
trust_domain = "local"
default_model_id = "omni"

[workers.capacity]
speech_websocket = 1
realtime_websocket = 1

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["omni"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
voice_name_policy = "preset"

[[workers.service_profiles]]
service = "realtime_websocket"
"#
    )
}

async fn connect_with_retry(
    url: &str,
) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match connect_async(url).await {
            Ok((socket, _)) => return socket,
            Err(_) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(error) => panic!("router websocket did not become available: {error}"),
        }
    }
}

async fn wait_for_worker_attempt(attempts: &AtomicUsize, expected: usize) {
    tokio::time::timeout(Duration::from_secs(1), async {
        while attempts.load(Ordering::Relaxed) < expected {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("worker reached the bounded setup stage");
}

async fn wait_ready(address: SocketAddr) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build readiness client");
    loop {
        if client
            .get(format!("http://{address}/ready"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "router did not become ready"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn speech_exact_replay_and_realtime_precommit_and_server_first_ordering() {
    let worker_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind worker fixture");
    let worker_address = worker_listener.local_addr().expect("worker address");
    let router_probe = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve router port");
    let router_address = router_probe.local_addr().expect("router address");
    drop(router_probe);
    let state = WorkerState {
        speech_config: Arc::new(Mutex::new(None)),
        speech_request_id: Arc::new(Mutex::new(None)),
        speech_close_code: Arc::new(AtomicUsize::new(0)),
        realtime_path: Arc::new(Mutex::new(None)),
        realtime_release: Arc::new(Notify::new()),
        realtime_control: Arc::new(Notify::new()),
    };
    let worker_app = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/speech/stream", get(speech_worker))
        .route("/v1/realtime", get(realtime_worker))
        .with_state(state.clone());
    let worker_task = tokio::spawn(async move {
        axum::serve(worker_listener, worker_app)
            .await
            .expect("serve worker fixture");
    });
    let directory = TestDir::new();
    let config = directory.config(&router_config(router_address, worker_address));
    let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(config)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start router process");
    let mut child = ChildGuard(child);

    wait_ready(router_address).await;

    let speech_url = format!("ws://{router_address}/v1/audio/speech/stream");
    let mut rejected_speech = connect_with_retry(&speech_url).await;
    rejected_speech
        .send(ClientMessage::Text(
            r#"{"type":"input.text","text":"too early"}"#.into(),
        ))
        .await
        .expect("send invalid initial speech event");
    let worker_error = rejected_speech
        .next()
        .await
        .expect("worker validation event")
        .expect("valid worker validation event")
        .into_text()
        .expect("worker validation text");
    assert!(worker_error.contains("invalid worker configuration"));
    rejected_speech
        .close(None)
        .await
        .expect("close rejected speech session");
    let _closed = rejected_speech.next().await;
    drop(rejected_speech);

    let mut speech_request = speech_url
        .as_str()
        .into_client_request()
        .expect("build speech request");
    speech_request
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_static("speech-request-1"));
    speech_request.headers_mut().insert(
        "sec-websocket-extensions",
        HeaderValue::from_static("permessage-deflate; client_max_window_bits"),
    );
    let (mut speech, speech_response) = connect_async(speech_request)
        .await
        .expect("connect speech with caller request ID");
    assert_eq!(
        speech_response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some("speech-request-1")
    );
    assert!(
        !speech_response
            .headers()
            .contains_key("sec-websocket-extensions"),
        "unsupported extension offers are declined rather than negotiated"
    );
    let exact =
        r#"{"type":"session.config","model":"omni","response_format":"pcm","stream_audio":true}"#;
    speech
        .send(ClientMessage::Text(exact.into()))
        .await
        .expect("send speech configuration after downstream 101");
    let configured = speech
        .next()
        .await
        .expect("configured event")
        .expect("valid event");
    assert_eq!(
        configured.into_text().expect("configured text"),
        r#"{"type":"session.configured","worker":"pinned"}"#
    );
    assert_eq!(
        state.speech_request_id.lock().await.as_deref(),
        Some("speech-request-1")
    );
    assert_eq!(state.speech_config.lock().await.as_deref(), Some(exact));
    speech
        .send(ClientMessage::Binary(vec![1, 2, 3].into()))
        .await
        .expect("send recoverable binary input");
    let recoverable = speech
        .next()
        .await
        .expect("recoverable response")
        .expect("valid response");
    assert!(
        recoverable
            .into_text()
            .expect("error text")
            .contains("text frames")
    );
    let _closed = speech.close(None).await;
    drop(speech);

    let mut next_speech = connect_with_retry(&speech_url).await;
    next_speech
        .send(ClientMessage::Ping(vec![1, 2, 3].into()))
        .await
        .expect("send hop-local control frame before configuration");
    next_speech
        .send(ClientMessage::Text(exact.into()))
        .await
        .expect("send configuration after prior permit release");
    loop {
        match next_speech.next().await {
            Some(Ok(ClientMessage::Ping(_) | ClientMessage::Pong(_))) => {}
            Some(Ok(ClientMessage::Text(_))) => break,
            other => panic!("expected configured event after control frame, got {other:?}"),
        }
    }
    let _closed = next_speech.close(None).await;
    drop(next_speech);

    let mut worker_validated = connect_with_retry(&speech_url).await;
    worker_validated
        .send(ClientMessage::Text(
            r#"{"type":"session.config","model":"omni","response_format":"future","task_type":"future"}"#.into(),
        ))
        .await
        .expect("send worker-owned configuration values");
    let worker_error = worker_validated
        .next()
        .await
        .expect("worker-owned validation event")
        .expect("valid worker-owned validation event")
        .into_text()
        .expect("worker-owned validation text");
    assert!(worker_error.contains("invalid worker configuration"));
    worker_validated
        .close(None)
        .await
        .expect("close worker-validated session");
    let _closed = worker_validated.next().await;
    drop(worker_validated);

    let mut active_speech = connect_with_retry(&speech_url).await;
    let active = r#"{"type":"session.config","model":"omni","active":true}"#;
    active_speech
        .send(ClientMessage::Text(active.into()))
        .await
        .expect("send active-worker speech configuration");
    assert!(matches!(
        active_speech.next().await,
        Some(Ok(ClientMessage::Text(_)))
    ));
    for expected in [vec![9], vec![10]] {
        let frame = tokio::time::timeout(Duration::from_secs(1), active_speech.next())
            .await
            .expect("silent client continues receiving worker output")
            .expect("active worker frame")
            .expect("valid active worker frame");
        assert_eq!(frame.into_data(), expected);
    }
    drop(active_speech);
    wait_for_worker_attempt(&state.speech_close_code, 1000).await;

    let exact_realtime_path =
        "/v1/realtime?unknown=first&model=%6F%6D%6E%69&unknown=second%2fvalue";
    let realtime_url = format!("ws://{router_address}{exact_realtime_path}");
    let connect_task = tokio::spawn(async move { connect_async(realtime_url).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !connect_task.is_finished(),
        "downstream 101 must await upstream handshake"
    );
    state.realtime_release.notify_one();
    let (mut realtime, _) = connect_task
        .await
        .expect("join realtime connect")
        .expect("complete realtime downstream handshake");
    let created = realtime
        .next()
        .await
        .expect("session.created")
        .expect("valid event");
    assert_eq!(
        created.into_text().expect("session.created text"),
        r#"{"type":"session.created","session":{"model":"omni"}}"#
    );
    assert_eq!(
        state.realtime_path.lock().await.as_deref(),
        Some(exact_realtime_path)
    );
    let update = r#"{"type":"session.update","session":{"model":"reflected"}}"#;
    let cancel = r#"{"type":"response.cancel","event_id":"ordered"}"#;
    realtime
        .send(ClientMessage::Text(update.into()))
        .await
        .expect("send realtime model-bearing update");
    realtime
        .send(ClientMessage::Text(cancel.into()))
        .await
        .expect("send ordered realtime control");
    assert_eq!(
        realtime
            .next()
            .await
            .expect("echoed update")
            .expect("valid echoed update")
            .into_text()
            .expect("text update"),
        update
    );
    assert_eq!(
        realtime
            .next()
            .await
            .expect("echoed control")
            .expect("valid echoed control")
            .into_text()
            .expect("text control"),
        cancel
    );
    let _closed = realtime.close(None).await;
    drop(realtime);

    let flood_url = format!("ws://{router_address}/v1/realtime");
    let flood_connect = tokio::spawn(async move { connect_with_retry(&flood_url).await });
    state.realtime_release.notify_one();
    let mut flood_client = flood_connect.await.expect("join flood connection");
    assert!(matches!(
        flood_client.next().await,
        Some(Ok(ClientMessage::Text(_)))
    ));
    flood_client
        .send(ClientMessage::Text(REALTIME_FLOOD.into()))
        .await
        .expect("start sustained worker output");
    tokio::time::sleep(Duration::from_millis(50)).await;
    flood_client
        .send(ClientMessage::Text(REALTIME_CONTROL.into()))
        .await
        .expect("send control while downstream output is unread");
    tokio::time::timeout(Duration::from_secs(2), state.realtime_control.notified())
        .await
        .expect("client-to-worker direction remains live under downstream backpressure");
    drop(flood_client);

    #[cfg(unix)]
    {
        let mut draining = connect_with_retry(&speech_url).await;
        draining
            .send(ClientMessage::Text(exact.into()))
            .await
            .expect("configure session held through process drain");
        assert!(matches!(
            draining.next().await,
            Some(Ok(ClientMessage::Text(_)))
        ));
        let signal = Command::new("kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .expect("send router drain signal");
        assert!(signal.success());
        let close = tokio::time::timeout(Duration::from_secs(2), draining.next())
            .await
            .expect("drain closes active WebSocket")
            .expect("drain close frame")
            .expect("valid drain close frame");
        assert!(matches!(
            close,
            ClientMessage::Close(Some(frame)) if u16::from(frame.code) == 1012
        ));
        drop(draining);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(status) = child.0.try_wait().expect("poll drained router") {
                assert!(status.success());
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "router retained a WebSocket session after drain"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    worker_task.abort();
    let _joined = worker_task.await;
}

#[tokio::test]
async fn setup_deadline_releases_stalled_speech_and_realtime_capacity() {
    let worker_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind setup-deadline worker fixture");
    let worker_address = worker_listener.local_addr().expect("worker address");
    let router_probe =
        std::net::TcpListener::bind("127.0.0.1:0").expect("reserve setup-deadline router port");
    let router_address = router_probe.local_addr().expect("router address");
    drop(router_probe);
    let state = SetupDeadlineWorkerState {
        speech_attempts: Arc::new(AtomicUsize::new(0)),
        realtime_attempts: Arc::new(AtomicUsize::new(0)),
        speech_early_frames: Arc::new(AtomicUsize::new(0)),
        realtime_early_frames: Arc::new(AtomicUsize::new(0)),
        speech_close_code: Arc::new(AtomicUsize::new(0)),
        realtime_close_code: Arc::new(AtomicUsize::new(0)),
        health_probes: Arc::new(AtomicUsize::new(0)),
    };
    let worker_app = Router::new()
        .route("/health", get(setup_deadline_health))
        .route("/v1/audio/speech/stream", get(setup_deadline_speech_worker))
        .route("/v1/realtime", get(setup_deadline_realtime_worker))
        .with_state(state.clone());
    let worker_task = tokio::spawn(async move {
        axum::serve(worker_listener, worker_app)
            .await
            .expect("serve setup-deadline worker fixture");
    });
    let directory = TestDir::new();
    let config = directory.config(
        &router_config(router_address, worker_address)
            .replace("interval_ms = 100", "interval_ms = 60000")
            .replace("connect_timeout_ms = 5000", "connect_timeout_ms = 100")
            .replace(
                "worker_setup_timeout_ms = 5000",
                "worker_setup_timeout_ms = 2000",
            ),
    );
    let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(config)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start setup-deadline router");
    let _child = ChildGuard(child);
    wait_ready(router_address).await;
    let initial_health_probes = state.health_probes.load(Ordering::Relaxed);

    let speech_url = format!("ws://{router_address}/v1/audio/speech/stream");
    let speech_config = r#"{"type":"session.config","model":"omni","response_format":"pcm"}"#;
    let mut early_speech = connect_with_retry(&format!("{speech_url}?case=early")).await;
    early_speech
        .send(ClientMessage::Text(speech_config.into()))
        .await
        .expect("send early-frame speech configuration");
    early_speech
        .send(ClientMessage::Binary(vec![7; 15 * 1_024 * 1_024].into()))
        .await
        .expect("send speech frame before session.configured");
    let configured = tokio::time::timeout(Duration::from_secs(5), early_speech.next())
        .await
        .expect("early speech frame reaches the worker during setup")
        .expect("speech configured event")
        .expect("valid speech configured event")
        .into_text()
        .expect("speech configured text");
    assert_eq!(
        configured,
        r#"{"type":"session.configured","worker":"reused"}"#
    );
    let received = tokio::time::timeout(Duration::from_secs(5), early_speech.next())
        .await
        .expect("worker receives the complete early speech frame")
        .expect("speech early-frame acknowledgement")
        .expect("valid speech early-frame acknowledgement")
        .into_text()
        .expect("speech early-frame acknowledgement text");
    assert_eq!(received, r#"{"type":"early.received","bytes":15728640}"#);
    assert_eq!(state.speech_early_frames.load(Ordering::Relaxed), 1);
    early_speech.close(None).await.expect("close early speech");
    drop(early_speech);

    state.speech_close_code.store(0, Ordering::Relaxed);
    let mut disconnected_speech = connect_with_retry(&format!("{speech_url}?case=stall")).await;
    disconnected_speech
        .send(ClientMessage::Text(speech_config.into()))
        .await
        .expect("send configuration before speech disconnect");
    wait_for_worker_attempt(&state.speech_attempts, 2).await;
    disconnected_speech
        .close(None)
        .await
        .expect("disconnect during speech setup");
    drop(disconnected_speech);
    wait_for_worker_attempt(&state.speech_close_code, 1000).await;

    let mut reused_speech = tokio::time::timeout(Duration::from_secs(1), async {
        let mut socket = connect_with_retry(&speech_url).await;
        socket
            .send(ClientMessage::Text(speech_config.into()))
            .await
            .expect("send configuration after speech disconnect");
        let configured = socket
            .next()
            .await
            .expect("speech configured event")
            .expect("valid speech configured event")
            .into_text()
            .expect("speech configured text");
        assert_eq!(
            configured,
            r#"{"type":"session.configured","worker":"reused"}"#
        );
        socket
    })
    .await
    .expect("speech capacity is reusable before the two-second setup deadline");
    reused_speech
        .close(None)
        .await
        .expect("close reused speech");
    drop(reused_speech);

    state.speech_close_code.store(0, Ordering::Relaxed);
    let mut stalled_speech = connect_with_retry(&format!("{speech_url}?case=stall")).await;
    stalled_speech
        .send(ClientMessage::Text(speech_config.into()))
        .await
        .expect("send configuration to stalled speech worker");
    wait_for_worker_attempt(&state.speech_attempts, 4).await;
    let speech_close = tokio::time::timeout(Duration::from_secs(3), stalled_speech.next())
        .await
        .expect("setup deadline closes downstream speech")
        .expect("speech close frame")
        .expect("valid speech close frame");
    assert!(matches!(
        speech_close,
        ClientMessage::Close(Some(frame)) if u16::from(frame.code) == 1011
    ));
    wait_for_worker_attempt(&state.speech_close_code, 1011).await;
    assert_eq!(
        state.health_probes.load(Ordering::Relaxed),
        initial_health_probes,
        "a worker application timeout must not trigger a health probe"
    );
    drop(stalled_speech);

    let mut after_timeout_speech = tokio::time::timeout(Duration::from_secs(1), async {
        let mut socket = connect_with_retry(&speech_url).await;
        socket
            .send(ClientMessage::Text(speech_config.into()))
            .await
            .expect("send configuration after speech setup timeout");
        let configured = socket
            .next()
            .await
            .expect("speech configured event")
            .expect("valid speech configured event")
            .into_text()
            .expect("speech configured text");
        assert_eq!(
            configured,
            r#"{"type":"session.configured","worker":"reused"}"#
        );
        socket
    })
    .await
    .expect("speech capacity is reusable after the setup deadline");
    after_timeout_speech
        .close(None)
        .await
        .expect("close speech after timeout");
    drop(after_timeout_speech);
    assert_eq!(state.speech_attempts.load(Ordering::Relaxed), 5);

    let realtime_url = format!("ws://{router_address}/v1/realtime");
    let mut early_realtime = connect_with_retry(&format!("{realtime_url}?case=early")).await;
    let early_realtime_payload = "x".repeat(15 * 1_024 * 1_024);
    early_realtime
        .send(ClientMessage::Text(early_realtime_payload.into()))
        .await
        .expect("send realtime frame before session.created");
    let created = tokio::time::timeout(Duration::from_secs(5), early_realtime.next())
        .await
        .expect("early realtime frame reaches the worker during setup")
        .expect("realtime created event")
        .expect("valid realtime created event")
        .into_text()
        .expect("realtime created text");
    assert_eq!(
        created,
        r#"{"type":"session.created","session":{"model":"omni"}}"#
    );
    let received = tokio::time::timeout(Duration::from_secs(5), early_realtime.next())
        .await
        .expect("worker receives the complete early realtime frame")
        .expect("realtime early-frame acknowledgement")
        .expect("valid realtime early-frame acknowledgement")
        .into_text()
        .expect("realtime early-frame acknowledgement text");
    assert_eq!(received, r#"{"type":"early.received","bytes":15728640}"#);
    assert_eq!(state.realtime_early_frames.load(Ordering::Relaxed), 1);
    early_realtime
        .close(None)
        .await
        .expect("close early realtime");
    drop(early_realtime);

    state.realtime_close_code.store(0, Ordering::Relaxed);
    let mut disconnected_realtime = connect_with_retry(&format!("{realtime_url}?case=stall")).await;
    wait_for_worker_attempt(&state.realtime_attempts, 2).await;
    disconnected_realtime
        .close(None)
        .await
        .expect("disconnect during realtime setup");
    drop(disconnected_realtime);
    wait_for_worker_attempt(&state.realtime_close_code, 1000).await;

    let mut reused_realtime = tokio::time::timeout(Duration::from_secs(1), async {
        let mut socket = connect_with_retry(&realtime_url).await;
        let created = socket
            .next()
            .await
            .expect("realtime created event")
            .expect("valid realtime created event")
            .into_text()
            .expect("realtime created text");
        assert_eq!(
            created,
            r#"{"type":"session.created","session":{"model":"omni"}}"#
        );
        socket
    })
    .await
    .expect("realtime capacity is reusable before the two-second setup deadline");
    reused_realtime
        .close(None)
        .await
        .expect("close reused realtime");
    drop(reused_realtime);

    state.realtime_close_code.store(0, Ordering::Relaxed);
    let mut stalled_realtime = connect_with_retry(&format!("{realtime_url}?case=stall")).await;
    wait_for_worker_attempt(&state.realtime_attempts, 4).await;
    let realtime_close = tokio::time::timeout(Duration::from_secs(3), stalled_realtime.next())
        .await
        .expect("setup deadline closes downstream realtime")
        .expect("realtime close frame")
        .expect("valid realtime close frame");
    assert!(matches!(
        realtime_close,
        ClientMessage::Close(Some(frame)) if u16::from(frame.code) == 1011
    ));
    wait_for_worker_attempt(&state.realtime_close_code, 1011).await;
    assert_eq!(
        state.health_probes.load(Ordering::Relaxed),
        initial_health_probes,
        "a worker application timeout must not trigger a health probe"
    );
    drop(stalled_realtime);

    let mut after_timeout_realtime = tokio::time::timeout(Duration::from_secs(1), async {
        let mut socket = connect_with_retry(&realtime_url).await;
        let created = socket
            .next()
            .await
            .expect("realtime created event")
            .expect("valid realtime created event")
            .into_text()
            .expect("realtime created text");
        assert_eq!(
            created,
            r#"{"type":"session.created","session":{"model":"omni"}}"#
        );
        socket
    })
    .await
    .expect("realtime capacity is reusable after the setup deadline");
    after_timeout_realtime
        .close(None)
        .await
        .expect("close realtime after timeout");
    drop(after_timeout_realtime);
    assert_eq!(state.realtime_attempts.load(Ordering::Relaxed), 5);

    worker_task.abort();
    let _joined = worker_task.await;
}

#[derive(Clone)]
struct HeterogeneousWorkerState {
    model: &'static str,
    handshakes: Arc<AtomicUsize>,
    paths: Arc<Mutex<Vec<String>>>,
}

async fn heterogeneous_speech_worker(
    State(state): State<HeterogeneousWorkerState>,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    upgrade.on_upgrade(move |mut socket| async move {
        let Some(Ok(Message::Text(_config))) = socket.next().await else {
            return;
        };
        let configured = format!(
            r#"{{"type":"session.configured","worker":"{}"}}"#,
            state.model
        );
        if socket.send(Message::Text(configured.into())).await.is_err() {
            return;
        }
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Close(frame)) => {
                    let _closed = socket.send(Message::Close(frame)).await;
                    return;
                }
                Ok(Message::Text(_) | Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {
                }
                Err(_) => return,
            }
        }
    })
}

async fn heterogeneous_realtime_worker(
    State(state): State<HeterogeneousWorkerState>,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    state.handshakes.fetch_add(1, Ordering::Relaxed);
    state.paths.lock().await.push(uri.to_string());
    upgrade.on_upgrade(move |mut socket| async move {
        let created = format!(
            r#"{{"type":"session.created","session":{{"model":"{}"}}}}"#,
            state.model
        );
        if socket.send(Message::Text(created.into())).await.is_err() {
            return;
        }
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Text(_)) => {
                    let selected = format!(r#"{{"type":"test.worker","model":"{}"}}"#, state.model);
                    if socket.send(Message::Text(selected.into())).await.is_err() {
                        return;
                    }
                }
                Ok(Message::Close(frame)) => {
                    let _closed = socket.send(Message::Close(frame)).await;
                    return;
                }
                Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {}
                Err(_) => return,
            }
        }
    })
}

fn heterogeneous_router_config(router: SocketAddr, alpha: SocketAddr, beta: SocketAddr) -> String {
    let worker = |id: &str, model: &str, address: SocketAddr, task: &str, references: &str| {
        format!(
            r#"
[[workers]]
worker_id = "{id}"
base_url = "http://{address}/"
trust_domain = "local"
default_model_id = "{model}"

[workers.capacity]
speech_websocket = 1
realtime_websocket = 1

[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["{model}", "tts"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["{task}"]
reference_forms = ["{references}"]
voice_name_policy = "preset"

[[workers.service_profiles]]
service = "realtime_websocket"
"#
        )
    };
    format!(
        r#"schema_version = 1

[server]
listen = "{router}"

[shutdown]
drain_timeout_ms = 5000

[logging]
format = "json"
filter = "error"

[router]
strategy = "round_robin"

[admission]
global = 2
speech_websocket = 2
realtime_websocket = 2

[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1

[websocket.speech]
trust_domain = "local"

[websocket.realtime]
trust_domain = "local"
{}{}
"#,
        worker("alpha", "omni-alpha", alpha, "text_to_speech", "none"),
        worker("beta", "omni-beta", beta, "voice_clone", "direct")
    )
}

#[tokio::test]
async fn heterogeneous_websocket_routing_follows_request_facts() {
    let alpha_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind alpha worker");
    let beta_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind beta worker");
    let alpha_address = alpha_listener.local_addr().expect("alpha address");
    let beta_address = beta_listener.local_addr().expect("beta address");
    let handshakes = Arc::new(AtomicUsize::new(0));
    let alpha_paths = Arc::new(Mutex::new(Vec::new()));
    let beta_paths = Arc::new(Mutex::new(Vec::new()));
    let alpha_state = HeterogeneousWorkerState {
        model: "omni-alpha",
        handshakes: Arc::clone(&handshakes),
        paths: Arc::clone(&alpha_paths),
    };
    let beta_state = HeterogeneousWorkerState {
        model: "omni-beta",
        handshakes: Arc::clone(&handshakes),
        paths: Arc::clone(&beta_paths),
    };
    let alpha_task = tokio::spawn(async move {
        axum::serve(
            alpha_listener,
            Router::new()
                .route("/health", get(health))
                .route("/v1/audio/speech/stream", get(heterogeneous_speech_worker))
                .route("/v1/realtime", get(heterogeneous_realtime_worker))
                .with_state(alpha_state),
        )
        .await
        .expect("serve alpha worker");
    });
    let beta_task = tokio::spawn(async move {
        axum::serve(
            beta_listener,
            Router::new()
                .route("/health", get(health))
                .route("/v1/audio/speech/stream", get(heterogeneous_speech_worker))
                .route("/v1/realtime", get(heterogeneous_realtime_worker))
                .with_state(beta_state),
        )
        .await
        .expect("serve beta worker");
    });

    let router_probe =
        std::net::TcpListener::bind("127.0.0.1:0").expect("reserve heterogeneous router port");
    let router_address = router_probe.local_addr().expect("router address");
    drop(router_probe);
    let directory = TestDir::new();
    let config = directory.config(&heterogeneous_router_config(
        router_address,
        alpha_address,
        beta_address,
    ));
    let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(config)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start heterogeneous router");
    let _child = ChildGuard(child);
    wait_ready(router_address).await;

    let speech_url = format!("ws://{router_address}/v1/audio/speech/stream");
    let (mut custom, _) = connect_async(&speech_url)
        .await
        .expect("connect text-only speech session");
    custom
        .send(ClientMessage::Text(
            r#"{"type":"session.config","model":"tts","voice":"default"}"#.into(),
        ))
        .await
        .expect("send text-only speech configuration");
    let configured = custom
        .next()
        .await
        .expect("CustomVoice configured event")
        .expect("valid CustomVoice configured event")
        .into_text()
        .expect("CustomVoice configured text");
    assert!(configured.contains(r#""worker":"omni-alpha""#));
    custom.close(None).await.expect("close CustomVoice session");
    let _closed = custom.next().await;
    drop(custom);

    let (mut base, _) = connect_async(&speech_url)
        .await
        .expect("connect voice-clone speech session");
    base.send(ClientMessage::Text(
        r#"{"type":"session.config","model":"tts","task_type":"Base","ref_audio":"reference"}"#
            .into(),
    ))
    .await
    .expect("send Base speech configuration");
    let configured = base
        .next()
        .await
        .expect("Base configured event")
        .expect("valid Base configured event")
        .into_text()
        .expect("Base configured text");
    assert!(configured.contains(r#""worker":"omni-beta""#));
    base.close(None).await.expect("close Base session");
    let _closed = base.next().await;
    drop(base);

    let (mut unspecified, _) = connect_async(format!("ws://{router_address}/v1/realtime"))
        .await
        .expect("select a realtime worker without a model requirement");
    let created = unspecified
        .next()
        .await
        .expect("unscoped session.created")
        .expect("valid unscoped event")
        .into_text()
        .expect("unscoped event text");
    assert!(created.contains(r#""model":"omni-alpha""#));
    unspecified
        .close(None)
        .await
        .expect("close unscoped session");
    let _closed = unspecified.next().await;
    drop(unspecified);

    let beta_path = "/v1/realtime?trace=first&model=omni%2Dbeta&trace=second%2fvalue";
    let (mut beta, _) = connect_async(format!("ws://{router_address}{beta_path}"))
        .await
        .expect("select beta worker");
    let created = beta
        .next()
        .await
        .expect("beta session.created")
        .expect("valid beta event")
        .into_text()
        .expect("beta event text");
    assert!(created.contains(r#""model":"omni-beta""#));
    beta.send(ClientMessage::Text(
        r#"{"type":"session.update","session":{"model":"omni-alpha"}}"#.into(),
    ))
    .await
    .expect("send later model-bearing event");
    let pinned = beta
        .next()
        .await
        .expect("pinned response")
        .expect("valid pinned response")
        .into_text()
        .expect("pinned response text");
    assert!(pinned.contains(r#""model":"omni-beta""#));
    assert_eq!(alpha_paths.lock().await.as_slice(), ["/v1/realtime"]);
    assert_eq!(beta_paths.lock().await.as_slice(), [beta_path]);
    beta.close(None).await.expect("close beta session");
    let _closed = beta.next().await;
    drop(beta);

    let failure = connect_async(format!("ws://{router_address}/v1/realtime?model=unknown"))
        .await
        .expect_err("an unknown explicit model must not fall back");
    assert!(matches!(
        failure,
        tokio_tungstenite::tungstenite::Error::Http(response)
            if response.status() == StatusCode::UNPROCESSABLE_ENTITY
    ));
    assert_eq!(alpha_paths.lock().await.as_slice(), ["/v1/realtime"]);
    assert_eq!(handshakes.load(Ordering::Relaxed), 2);

    alpha_task.abort();
    beta_task.abort();
    let _alpha = alpha_task.await;
    let _beta = beta_task.await;
}
