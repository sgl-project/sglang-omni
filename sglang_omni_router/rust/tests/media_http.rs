#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Real-socket media request classification, exact-byte forwarding, and relay proof.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static SOCKET_LOCK: Mutex<()> = Mutex::new(());
const DEADLINE: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MediaRoute {
    Speech,
    SpeechBatch,
    Transcription,
    Translation,
}

impl MediaRoute {
    const ALL: [Self; 4] = [
        Self::Speech,
        Self::SpeechBatch,
        Self::Transcription,
        Self::Translation,
    ];

    const fn route_name(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::SpeechBatch => "speech_batch",
            Self::Transcription => "transcription",
            Self::Translation => "translation",
        }
    }

    const fn path(self) -> &'static str {
        match self {
            Self::Speech => "/v1/audio/speech",
            Self::SpeechBatch => "/v1/audio/speech/batch",
            Self::Transcription => "/v1/audio/transcriptions",
            Self::Translation => "/v1/audio/translations",
        }
    }
}

const MEDIA_ROUTE_SUBSETS: [&[MediaRoute]; 5] = [
    &[MediaRoute::Speech],
    &[MediaRoute::SpeechBatch],
    &[MediaRoute::Transcription],
    &[MediaRoute::Translation],
    &MediaRoute::ALL,
];

#[derive(Clone, Debug)]
struct Captured {
    path: String,
    content_type: String,
    body: Vec<u8>,
    request_id: Option<String>,
    route_model: Option<String>,
    route_stream: Option<String>,
}

struct Worker {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    health_completed: Arc<(Mutex<bool>, Condvar)>,
    captured: Arc<Mutex<Vec<Captured>>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind media worker fixture");
        let address = listener.local_addr().expect("read worker address");
        let stop = Arc::new(AtomicBool::new(false));
        let health_completed = Arc::new((Mutex::new(false), Condvar::new()));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = Arc::clone(&stop);
        let thread_health_completed = Arc::clone(&health_completed);
        let thread_captured = Arc::clone(&captured);
        let thread = thread::spawn(move || {
            let mut connections = Vec::new();
            loop {
                let (stream, _) = listener.accept().expect("accept media worker request");
                if thread_stop.load(Ordering::Acquire) {
                    break;
                }
                let captures = Arc::clone(&thread_captured);
                let health_completed = Arc::clone(&thread_health_completed);
                connections.push(thread::spawn(move || {
                    handle_connection(stream, captures, health_completed)
                }));
            }
            for connection in connections {
                connection.join().expect("join media worker connection");
            }
        });
        Self {
            address,
            stop,
            health_completed,
            captured,
            thread: Some(thread),
        }
    }

    fn captures(&self) -> Vec<Captured> {
        self.captured.lock().expect("read media captures").clone()
    }

    fn wait_health_completed(&self) {
        let deadline = Instant::now() + DEADLINE;
        let (completed, changed) = &*self.health_completed;
        let mut completed = completed.lock().expect("lock health completion");
        while !*completed {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("worker health probe did not complete");
            }
            let waited = changed
                .wait_timeout(completed, remaining)
                .expect("wait for health completion");
            completed = waited.0;
            if waited.1.timed_out() && !*completed {
                panic!("worker health probe did not complete");
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _wake = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let _joined = thread.join();
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    captured: Arc<Mutex<Vec<Captured>>>,
    health_completed: Arc<(Mutex<bool>, Condvar)>,
) {
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound worker read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound worker write");
    loop {
        let Some((head, mut body)) = read_head(&mut stream) else {
            return;
        };
        let request_line = head.lines().next().unwrap_or_default();
        if request_line.starts_with("GET /health ") {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write health response");
            stream.flush().expect("flush health response");
            drop(stream);
            let (completed, changed) = &*health_completed;
            *completed.lock().expect("publish health completion") = true;
            changed.notify_all();
            return;
        }
        let path = request_line
            .split_ascii_whitespace()
            .nth(1)
            .expect("request path")
            .to_owned();
        let expected = header_value(&head, "content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .expect("router sends canonical content length");
        while body.len() < expected {
            let mut chunk = [0_u8; 4096];
            let count = stream.read(&mut chunk).expect("read worker request body");
            if count == 0 {
                return;
            }
            body.extend_from_slice(&chunk[..count]);
        }
        body.truncate(expected);
        let content_type = header_value(&head, "content-type")
            .expect("router sends content type")
            .to_owned();
        captured
            .lock()
            .expect("record media request")
            .push(Captured {
                path: path.clone(),
                content_type,
                body: body.clone(),
                request_id: header_value(&head, "x-request-id").map(str::to_owned),
                route_model: header_value(&head, "x-sglang-omni-route-model").map(str::to_owned),
                route_stream: header_value(&head, "x-sglang-omni-route-stream").map(str::to_owned),
            });
        let response: &[u8] = match path.as_str() {
        "/v1/audio/speech" if contains_bytes(&body, b"\"stream\":true") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nContent-Length: 4\r\nX-Sample-Rate: 24000\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n\r\nPCM!"
        }
        "/v1/audio/speech" if contains_bytes(&body, b"\"response_format\":\"opus\"") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/opus\r\nContent-Length: 4\r\n\r\nOPUS"
        }
        "/v1/audio/speech" if contains_bytes(&body, b"\"response_format\":\"pcm\"") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nContent-Length: 4\r\n\r\nPCM0"
        }
        "/v1/audio/speech" => b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: 4\r\nContent-Disposition: attachment; filename=\"speech.wav\"\r\nX-Prompt-Tokens: 7\r\nX-Completion-Tokens: 9\r\nX-Engine-Time: 0.25\r\nX-Finish-Reason: stop\r\n\r\nWAVE",
        "/v1/audio/speech/batch" => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 23\r\n\r\n{\"results\":[0,1],\"n\":2}",
        "/v1/audio/transcriptions"
            if body
                .windows(4)
                .any(|window| window.eq_ignore_ascii_case(b"true")) =>
        {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Request-Id: transcription-public\r\nContent-Length: 65\r\n\r\nevent: transcript.text.delta\ndata: {\"delta\":\"hi\"}\n\ndata: [DONE]\n\n"
        }
        "/v1/audio/transcriptions" if contains_bytes(&body, b"srt") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 35\r\n\r\n1\n00:00:00,000 --> 00:00:01,000\nhi\n"
        }
        "/v1/audio/transcriptions" if contains_bytes(&body, b"vtt") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 41\r\n\r\nWEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n"
        }
        "/v1/audio/transcriptions" => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Request-Id: transcription-public\r\nContent-Length: 13\r\n\r\n{\"text\":\"hi\"}",
        "/v1/audio/translations" if contains_bytes(&body, b"name=\"stream\"") =>
        {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 65\r\n\r\nevent: transcript.text.delta\ndata: {\"delta\":\"hi\"}\n\ndata: [DONE]\n\n"
        }
        "/v1/audio/translations" if contains_bytes(&body, b"srt") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 35\r\n\r\n1\n00:00:00,000 --> 00:00:01,000\nhi\n"
        }
        "/v1/audio/translations" if contains_bytes(&body, b"vtt") => {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 41\r\n\r\nWEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n"
        }
        "/v1/audio/translations" => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"text\":\"hi\"}",
        _ => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
        };
        stream.write_all(response).expect("write media response");
    }
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    directory: PathBuf,
}

impl RouterProcess {
    fn start(routes: &[MediaRoute], workers: &[(&Worker, bool)]) -> Self {
        let worker_refs = workers
            .iter()
            .map(|(worker, _alternate)| *worker)
            .collect::<Vec<_>>();
        Self::start_with(&worker_refs, |address| config(address, routes, workers))
    }

    fn start_with(workers: &[&Worker], build_config: impl FnOnce(SocketAddr) -> String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve router port");
        let address = listener.local_addr().expect("read router port");
        drop(listener);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-router-media-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create media router directory");
        let config_path = directory.join("router.toml");
        fs::write(&config_path, build_config(address)).expect("write media router config");
        let binary = env!("CARGO_BIN_EXE_sgl-omni-router");
        let child = Command::new(binary)
            .arg("--config")
            .arg(&config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start media router");
        let process = Self {
            child,
            address,
            directory,
        };
        process.wait_ready();
        for worker in workers {
            worker.wait_health_completed();
        }
        process
    }

    fn wait_ready(&self) {
        let deadline = Instant::now() + DEADLINE;
        while Instant::now() < deadline {
            if let Ok(response) = request(self.address, "GET", "/ready", None, b"")
                && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("media router did not become ready");
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _killed = self.child.kill();
        let _waited = self.child.wait();
        let _cleanup = fs::remove_dir_all(&self.directory);
    }
}

fn config(address: SocketAddr, routes: &[MediaRoute], workers: &[(&Worker, bool)]) -> String {
    let mut output = config_header(address, routes);
    let all_speech_tasks = "[\"text_to_speech\", \"voice_clone\", \"voice_design\"]";
    for (index, (worker, alternate)) in workers.iter().enumerate() {
        let speech_formats = if *alternate {
            "[\"mp3\"]"
        } else {
            "[\"mp3\", \"opus\", \"aac\", \"flac\", \"wav\"]"
        };
        let speech_to_text_formats = if *alternate {
            "[\"json\", \"text\", \"verbose_json\", \"sse\"]"
        } else {
            "[\"json\", \"text\", \"verbose_json\", \"srt\", \"vtt\", \"sse\"]"
        };
        output.push_str(&format!(
            "\n[[workers]]\nworker_id = \"worker-{index}\"\nbase_url = \"http://{}\"\ntrust_domain = \"local\"\ndefault_model_id = \"tts\"\n",
            worker.address
        ));
        for route in routes {
            output.push('\n');
            output.push_str(&media_profile(
                *route,
                speech_formats,
                speech_to_text_formats,
                all_speech_tasks,
            ));
        }
    }
    output
}

fn config_header(address: SocketAddr, routes: &[MediaRoute]) -> String {
    let enabled_routes = routes
        .iter()
        .map(|route| format!("\"{}\"", route.route_name()))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "schema_version = 1\n\n[server]\nlisten = \"{address}\"\nmax_connections = 64\n\n[shutdown]\ndrain_timeout_ms = 5000\n\n[logging]\nformat = \"json\"\nfilter = \"error\"\n\n[router]\nstrategy = \"round_robin\"\n\n[admission]\nglobal = 32\nspeech_http = 8\ntranscription_http = 8\nspeech_batch = 16\n\n[health]\ninterval_ms = 1000\ntimeout_ms = 500\nsuccess_threshold = 1\nfailure_threshold = 3\n\n[http]\nbuffered_request_total_bytes = 4096\nconnect_timeout_ms = 1000\npool_idle_timeout_ms = 1000\npool_max_idle_per_host = 4\n\n[http_media]\nroutes = [{enabled_routes}]\ntrust_domain = \"local\"\nbuffered_request_max_bytes = 1024\nstreamed_request_max_bytes = 8192\nrequest_timeout_ms = 5000\n"
    )
}

fn media_profile(
    route: MediaRoute,
    speech_formats: &str,
    speech_to_text_formats: &str,
    speech_tasks: &str,
) -> String {
    match route {
        MediaRoute::Speech => format!(
            "[[workers.service_profiles]]\nservice = \"speech_http\"\nmodel_ids = [\"tts\"]\nresponse_formats = {speech_formats}\nstream_modes = [\"non_streaming\"]\ntasks = {speech_tasks}\nreference_forms = [\"none\", \"direct\", \"list\", \"vq_codes\"]\nvoice_name_policy = \"preset\"\n\n[[workers.service_profiles]]\nservice = \"speech_http\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"pcm\"]\nstream_modes = [\"non_streaming\", \"streaming\"]\ntasks = {speech_tasks}\nreference_forms = [\"none\", \"direct\", \"list\", \"vq_codes\"]\nvoice_name_policy = \"preset\"\n"
        ),
        MediaRoute::SpeechBatch => format!(
            "[[workers.service_profiles]]\nservice = \"speech_batch\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"mp3\", \"opus\", \"aac\", \"flac\", \"wav\", \"pcm\"]\ntasks = {speech_tasks}\nreference_forms = [\"none\", \"direct\", \"list\", \"vq_codes\"]\nvoice_name_policy = \"preset\"\nmax_batch_size = 16\n",
        ),
        MediaRoute::Transcription => format!(
            "[[workers.service_profiles]]\nservice = \"transcription_http\"\nmodel_ids = [\"tts\", \"asr\"]\ntask = \"transcribe\"\nresponse_formats = {speech_to_text_formats}\nstream_modes = [\"non_streaming\", \"streaming\"]\n",
        ),
        MediaRoute::Translation => format!(
            "[[workers.service_profiles]]\nservice = \"transcription_http\"\nmodel_ids = [\"tts\", \"asr\"]\ntask = \"translate\"\nresponse_formats = {speech_to_text_formats}\nstream_modes = [\"non_streaming\", \"streaming\"]\n",
        ),
    }
}

fn qwen_task_config(address: SocketAddr, custom: &Worker, base: &Worker) -> String {
    let routes = [MediaRoute::Speech];
    let mut output = config_header(address, &routes);
    for (index, (worker, task, references, voice_policy)) in [
        (custom, "text_to_speech", "[\"none\"]", "preset"),
        (base, "voice_clone", "[\"direct\", \"list\"]", "uploaded"),
    ]
    .into_iter()
    .enumerate()
    {
        output.push_str(&format!(
            "\n[[workers]]\nworker_id = \"qwen-{index}\"\nbase_url = \"http://{}\"\ntrust_domain = \"local\"\ndefault_model_id = \"tts\"\n\n[[workers.service_profiles]]\nservice = \"speech_http\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"wav\"]\nstream_modes = [\"non_streaming\"]\ntasks = [\"{task}\"]\nreference_forms = {references}\nvoice_name_policy = \"{voice_policy}\"\n",
            worker.address
        ));
    }
    output
}

#[test]
fn every_media_subset_is_ready_and_registers_only_enabled_routes() {
    let _guard = socket_guard();
    for routes in MEDIA_ROUTE_SUBSETS {
        let worker = Worker::start();
        let router = RouterProcess::start(routes, &[(&worker, false)]);
        for route in MediaRoute::ALL {
            let (content_type, body) = match route {
                MediaRoute::Speech => (
                    "application/json",
                    br#"{"model":"tts","input":"hello","response_format":"wav"}"#.to_vec(),
                ),
                MediaRoute::SpeechBatch => (
                    "application/json",
                    br#"{"model":"tts","response_format":"wav","items":[{"input":"hello"}]}"#
                        .to_vec(),
                ),
                MediaRoute::Transcription | MediaRoute::Translation => (
                    "multipart/form-data; boundary=media-boundary",
                    multipart_body(false, 32),
                ),
            };
            let prior = worker.captures().len();
            let (content_type, body) = if routes.contains(&route) {
                (Some(content_type), body.as_slice())
            } else {
                (None, &[][..])
            };
            let response = request(router.address, "POST", route.path(), content_type, body)
                .expect("media subset route response");
            if routes.contains(&route) {
                assert!(
                    response.starts_with(b"HTTP/1.1 200"),
                    "enabled route {route:?} failed for {routes:?}: {}",
                    String::from_utf8_lossy(&response)
                );
                let captures = worker.captures();
                assert_eq!(captures.len(), prior + 1);
                assert_eq!(
                    captures.last().map(|capture| capture.path.as_str()),
                    Some(route.path())
                );
            } else {
                assert!(
                    response.starts_with(b"HTTP/1.1 404"),
                    "disabled route {route:?} was registered for {routes:?}: {}",
                    String::from_utf8_lossy(&response)
                );
                assert_eq!(worker.captures().len(), prior);
            }
        }
    }
}

#[test]
fn media_only_process_becomes_ready_without_installing_chat() {
    let _guard = socket_guard();
    let worker = Worker::start();
    let router = RouterProcess::start(&[MediaRoute::Transcription], &[(&worker, false)]);
    let response = request(
        router.address,
        "POST",
        "/v1/chat/completions",
        Some("application/json"),
        br#"{"model":"omni","messages":[]}"#,
    )
    .expect("media-only chat response");
    assert!(response.starts_with(b"HTTP/1.1 404"));
    assert!(worker.captures().is_empty());
}

#[test]
fn relays_all_media_routes_with_exact_bytes_headers_and_large_direct_uploads() {
    let _guard = socket_guard();
    let worker = Worker::start();
    let router = RouterProcess::start(&MediaRoute::ALL, &[(&worker, false)]);
    let relay = |path, content_type, body| roundtrip(&worker, &router, path, content_type, body);
    let speech = br#"{"model":"tts","input":"hello","response_format":"wav","task_type":"CustomVoice","ref_audio":"data:audio/wav;base64,AA==","references":[{"audio":"x"},{"vq_codes":[1,2]}]}"#;
    let (response, capture) = relay("/v1/audio/speech", "application/json", speech);
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected speech response: {}",
        String::from_utf8_lossy(&response)
    );
    assert!(header(&response, "content-disposition").is_some());
    assert_eq!(header(&response, "x-finish-reason"), Some("stop"));
    assert_eq!(response_body(&response), b"WAVE");
    assert_eq!(capture.body, speech);
    let generated_id = header(&response, "x-request-id").expect("generated media request ID");
    assert!(generated_id.starts_with("sglang-omni-"));
    assert_eq!(capture.request_id.as_deref(), Some(generated_id));

    let opus = br#"{"model":"tts","input":"hello","response_format":"opus"}"#;
    let (response, capture) = relay("/v1/audio/speech", "application/json", opus);
    assert_eq!(header(&response, "content-type"), Some("audio/opus"));
    assert_eq!(response_body(&response), b"OPUS");
    assert_eq!(capture.body, opus);

    let completed_pcm = br#"{"model":"tts","input":"hello","response_format":"pcm"}"#;
    let (response, capture) = relay("/v1/audio/speech", "application/json", completed_pcm);
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(header(&response, "content-type"), Some("audio/pcm"));
    assert!(header(&response, "x-sample-rate").is_none());
    assert!(header(&response, "x-channels").is_none());
    assert!(header(&response, "x-bit-depth").is_none());
    assert_eq!(response_body(&response), b"PCM0");
    assert_eq!(capture.body, completed_pcm);

    let streaming_pcm = br#"{"model":"tts","input":"hello","response_format":"pcm","stream":true}"#;
    let (response, capture) = relay("/v1/audio/speech", "application/json", streaming_pcm);
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected PCM response: {}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(header(&response, "x-sample-rate"), Some("24000"));
    assert_eq!(response_body(&response), b"PCM!");
    assert_eq!(capture.body, streaming_pcm);

    let batch = br#"{"model":"tts","response_format":"wav","items":[{"input":"a"},{"input":"b","response_format":"mp3","task_type":"VoiceDesign","ref_audio":"x"}]}"#;
    let (response, capture) = relay("/v1/audio/speech/batch", "application/json", batch);
    assert_eq!(response_body(&response), b"{\"results\":[0,1],\"n\":2}");
    assert_eq!(capture.body, batch);

    let mixed_batch = br#"{"items":[{"input":"bad","model":7,"response_format":"mp3"},{"input":"good","model":"tts","response_format":"wav"}]}"#;
    let (response, capture) = relay("/v1/audio/speech/batch", "application/json", mixed_batch);
    assert_eq!(response_body(&response), b"{\"results\":[0,1],\"n\":2}");
    assert_eq!(capture.body, mixed_batch);

    let multipart = multipart_body(false, 32);
    let (response, capture) = roundtrip_with_extra_headers(
        &worker,
        &router,
        "/v1/audio/transcriptions",
        "multipart/form-data; boundary=\"media-boundary\"",
        "x-request-id: caller-media-1\r\n",
        &multipart,
    );
    assert_eq!(header(&response, "x-request-id"), Some("caller-media-1"));
    assert_eq!(capture.request_id.as_deref(), Some("caller-media-1"));
    assert_eq!(response_body(&response), b"{\"text\":\"hi\"}");
    assert_eq!(capture.body, multipart);
    assert_eq!(
        capture.content_type,
        "multipart/form-data; boundary=\"media-boundary\""
    );

    let streaming_multipart = multipart_body(true, 32);
    let (response, capture) = roundtrip(
        &worker,
        &router,
        "/v1/audio/transcriptions",
        "multipart/form-data; boundary=media-boundary",
        &streaming_multipart,
    );
    assert!(
        capture
            .body
            .windows(4)
            .any(|window| window.eq_ignore_ascii_case(b"true")),
        "streaming multipart bytes reached the worker"
    );
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "unexpected SSE response: {}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(header(&response, "content-type"), Some("text/event-stream"));
    assert!(response_body(&response).ends_with(b"data: [DONE]\n\n"));
    assert_eq!(capture.body, streaming_multipart);

    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        for format in ["srt", "vtt"] {
            let subtitle = multipart_body_with_format(format, false, 32);
            let (response, capture) = roundtrip(
                &worker,
                &router,
                path,
                "multipart/form-data; boundary=media-boundary",
                &subtitle,
            );
            assert!(response.starts_with(b"HTTP/1.1 200"));
            assert_eq!(
                header(&response, "content-type"),
                Some("text/plain; charset=utf-8")
            );
            assert_eq!(capture.body, subtitle);
        }
    }

    let translated_stream = multipart_body_with_format("text", true, 32);
    let (response, capture) = roundtrip(
        &worker,
        &router,
        "/v1/audio/translations",
        "multipart/form-data; boundary=media-boundary",
        &translated_stream,
    );
    assert_eq!(header(&response, "content-type"), Some("text/event-stream"));
    assert!(response_body(&response).ends_with(b"data: [DONE]\n\n"));
    assert_eq!(capture.body, translated_stream);

    let mut large_speech = vec![b' '; 2_048];
    large_speech[..2].copy_from_slice(b"{}");
    let (response, capture) = relay("/v1/audio/speech", "application/json", &large_speech);
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(capture.body, large_speech);

    let large_batch = format!(
        r#"{{"model":"tts","items":[{{"input":"{}"}}]}}"#,
        "b".repeat(2_048)
    )
    .into_bytes();
    let prior = worker.captures().len();
    let response = expect_rejected_request(
        router.address,
        "/v1/audio/speech/batch",
        "application/json",
        large_batch.len(),
    )
    .expect("oversized batch response");
    assert!(response.starts_with(b"HTTP/1.1 413"));
    assert_eq!(worker.captures().len(), prior);

    let large_multipart = multipart_body(false, 2_048);
    let (response, capture) = roundtrip(
        &worker,
        &router,
        "/v1/audio/transcriptions",
        "multipart/form-data; boundary=media-boundary",
        &large_multipart,
    );
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(capture.body, large_multipart);
}

#[test]
fn qwen_task_types_select_the_matching_worker_profile() {
    let _guard = socket_guard();
    let custom = Worker::start();
    let base = Worker::start();
    let router = RouterProcess::start_with(&[&custom, &base], |address| {
        qwen_task_config(address, &custom, &base)
    });

    let custom_request =
        br#"{"model":"tts","input":"hello","task_type":"CustomVoice","voice":"Vivian"}"#;
    let response = request(
        router.address,
        "POST",
        MediaRoute::Speech.path(),
        Some("application/json"),
        custom_request,
    )
    .expect("CustomVoice response");
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(custom.captures().len(), 1);
    assert!(base.captures().is_empty());

    let base_request = br#"{"model":"tts","input":"hello","task_type":"Base","ref_audio":"data:audio/wav;base64,AA=="}"#;
    let response = request(
        router.address,
        "POST",
        MediaRoute::Speech.path(),
        Some("application/json"),
        base_request,
    )
    .expect("Base response");
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(base.captures().len(), 1);
    assert_eq!(custom.captures().len(), 1);
}

fn roundtrip(
    worker: &Worker,
    router: &RouterProcess,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> (Vec<u8>, Captured) {
    roundtrip_with_extra_headers(worker, router, path, content_type, "", body)
}

fn roundtrip_with_extra_headers(
    worker: &Worker,
    router: &RouterProcess,
    path: &str,
    content_type: &str,
    extra_headers: &str,
    body: &[u8],
) -> (Vec<u8>, Captured) {
    let prior = worker.captures().len();
    let response = request_with_extra_headers(
        router.address,
        "POST",
        path,
        Some(content_type),
        extra_headers,
        body,
    )
    .expect("media roundtrip response");
    let captures = worker.captures();
    let capture = captures
        .get(prior)
        .filter(|capture| capture.path == path)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "missing {path} capture for {} bytes; captures: {captures:?}; response: {}",
                body.len(),
                String::from_utf8_lossy(&response)
            )
        });
    (response, capture)
}

#[test]
fn oversized_heterogeneous_request_is_rejected_without_upstream_send() {
    let _guard = socket_guard();
    let first = Worker::start();
    let second = Worker::start();
    let router = RouterProcess::start(&MediaRoute::ALL, &[(&first, false), (&second, true)]);
    let body = vec![b'x'; 2_048];
    let response = request_with_extra_headers(
        router.address,
        "POST",
        "/v1/audio/speech",
        Some("application/json"),
        "x-request-id: caller-media-error\r\n",
        &body,
    )
    .expect("heterogeneous oversized request response");
    assert!(response.starts_with(b"HTTP/1.1 413"));
    assert_eq!(
        header(&response, "x-request-id"),
        Some("caller-media-error")
    );

    let response = request_with_extra_headers(
        router.address,
        "POST",
        "/v1/audio/speech",
        Some("application/json"),
        "x-request-id: one\r\nx-request-id: two\r\n",
        br#"{}"#,
    )
    .expect("duplicate media request-ID response");
    assert!(response.starts_with(b"HTTP/1.1 400"));
    let head = std::str::from_utf8(&response).expect("media error response is ASCII");
    let replacement = header(&response, "x-request-id").expect("canonical replacement ID");
    assert!(replacement.starts_with("sglang-omni-"));
    assert_eq!(
        head.to_ascii_lowercase().matches("x-request-id:").count(),
        1
    );
    assert!(first.captures().is_empty());
    assert!(second.captures().is_empty());
}

#[test]
fn speech_batch_above_every_worker_limit_is_not_capacity_pressure() {
    let _guard = socket_guard();
    let worker = Worker::start();
    let router = RouterProcess::start(&[MediaRoute::SpeechBatch], &[(&worker, false)]);
    let items = std::iter::repeat_n(r#"{"input":"x"}"#, 17)
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(r#"{{"model":"tts","items":[{items}]}}"#);
    let response = request(
        router.address,
        "POST",
        "/v1/audio/speech/batch",
        Some("application/json"),
        body.as_bytes(),
    )
    .expect("oversized batch response");
    assert!(response.starts_with(b"HTTP/1.1 422"));
    assert!(worker.captures().is_empty());
}

#[test]
fn media_accepts_chunked_uploads_and_standard_continue() {
    let _guard = socket_guard();
    let first = Worker::start();
    let second = Worker::start();
    let router = RouterProcess::start(&[MediaRoute::Speech], &[(&first, false), (&second, true)]);
    let body = br#"{"model":"tts","input":"chunked","response_format":"wav"}"#;
    let response = chunked_request(
        router.address,
        "/v1/audio/speech",
        "application/json",
        &[&body[..17], &body[17..]],
    )
    .expect("chunked media response");
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert!(
        first
            .captures()
            .into_iter()
            .chain(second.captures())
            .any(|capture| capture.body == body)
    );

    let mut stream = TcpStream::connect_timeout(&router.address, Duration::from_millis(200))
        .expect("connect continue client");
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound continue read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound continue write");
    write!(
        stream,
        "POST /v1/audio/speech HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nExpect: 100-continue\r\nConnection: close\r\n\r\n",
        router.address,
        body.len()
    )
    .expect("write continue request head");
    stream.flush().expect("flush continue request head");
    let (head, pending) = read_head(&mut stream).expect("read continue response");
    assert!(head.starts_with("HTTP/1.1 100 Continue"));
    assert!(pending.is_empty());
    stream.write_all(body).expect("write continued body");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("read final response");
    assert!(response.starts_with(b"HTTP/1.1 200"));
}

#[test]
fn homogeneous_media_round_robin_reaches_both_workers_over_real_sockets() {
    let _guard = socket_guard();
    let first = Worker::start();
    let second = Worker::start();
    let router = RouterProcess::start(&[MediaRoute::Speech], &[(&first, false), (&second, false)]);
    let body = br#"{"model":"tts","input":"hello","response_format":"wav"}"#;
    let mut identity_body = br#"{"model":"tts","input":""#.to_vec();
    identity_body.extend(std::iter::repeat_n(b'a', 1_500));
    identity_body.extend_from_slice(br#"","response_format":"wav"}"#);

    let response = request_with_extra_headers(
        router.address,
        "POST",
        MediaRoute::Speech.path(),
        Some("application/json"),
        "content-encoding: identity\r\n",
        &identity_body,
    )
    .expect("identity-encoded homogeneous media response");
    assert!(response.starts_with(b"HTTP/1.1 200"));

    for _ in 0..3 {
        let response = request(
            router.address,
            "POST",
            MediaRoute::Speech.path(),
            Some("application/json"),
            body,
        )
        .expect("homogeneous media response");
        assert!(response.starts_with(b"HTTP/1.1 200"));
    }

    assert_eq!(first.captures().len(), 2);
    assert_eq!(second.captures().len(), 2);
    let captures = first
        .captures()
        .into_iter()
        .chain(second.captures())
        .collect::<Vec<_>>();
    assert_eq!(
        captures
            .iter()
            .filter(|capture| capture.body == identity_body)
            .count(),
        1
    );
    assert_eq!(
        captures
            .iter()
            .filter(|capture| capture.body == body)
            .count(),
        3
    );
}

#[test]
fn heterogeneous_media_classification_selects_only_a_correlated_capable_worker() {
    let _guard = socket_guard();
    let capable = Worker::start();
    let limited = Worker::start();
    let router = RouterProcess::start(&MediaRoute::ALL, &[(&capable, false), (&limited, true)]);
    let translated = multipart_body_with_format("srt", false, 32);
    let response = request(
        router.address,
        "POST",
        "/v1/audio/translations",
        Some("multipart/form-data; boundary=media-boundary"),
        &translated,
    )
    .expect("heterogeneous translation response");
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(capable.captures().len(), 1);
    assert!(limited.captures().is_empty());

    let hinted = multipart_body_with_format("text", true, 32);
    let response = request_with_extra_headers(
        router.address,
        "POST",
        "/v1/audio/translations",
        Some("multipart/form-data; boundary=media-boundary"),
        "x-sglang-omni-route-model: tts\r\nx-sglang-omni-route-stream: true\r\n",
        &hinted,
    )
    .expect("route assertion response");
    assert_eq!(header(&response, "content-type"), Some("text/event-stream"));
    let captures = capable
        .captures()
        .into_iter()
        .chain(limited.captures())
        .collect::<Vec<_>>();
    let capture = captures
        .iter()
        .find(|capture| capture.body == hinted)
        .expect("asserted translation capture");
    assert!(capture.route_model.is_none());
    assert!(capture.route_stream.is_none());

    let header_only_true = multipart_body_with_format("text", false, 32);
    let response = request_with_extra_headers(
        router.address,
        "POST",
        "/v1/audio/translations",
        Some("multipart/form-data; boundary=media-boundary"),
        "x-sglang-omni-route-stream: true\r\n",
        &header_only_true,
    )
    .expect("header-only stream assertion response");
    assert!(response.starts_with(b"HTTP/1.1 400"));
    assert_eq!(capable.captures().len() + limited.captures().len(), 2);

    let model_conflict = br#"{"model":"tts","input":"x","response_format":"wav"}"#;
    let response = request_with_extra_headers(
        router.address,
        "POST",
        "/v1/audio/speech",
        Some("application/json"),
        "x-sglang-omni-route-model: other\r\n",
        model_conflict,
    )
    .expect("route model conflict response");
    assert!(response.starts_with(b"HTTP/1.1 400"));
    assert_eq!(capable.captures().len() + limited.captures().len(), 2);

    let conflicting = multipart_body_with_format("text", true, 32);
    let response = request_with_extra_headers(
        router.address,
        "POST",
        "/v1/audio/translations",
        Some("multipart/form-data; boundary=media-boundary"),
        "x-sglang-omni-route-stream: false\r\n",
        &conflicting,
    )
    .expect("route-header conflict response");
    assert!(response.starts_with(b"HTTP/1.1 400"));
    assert_eq!(capable.captures().len() + limited.captures().len(), 2);

    let invalid_streaming_speech =
        br#"{"model":"tts","input":"x","stream":true,"response_format":"wav"}"#;
    let response = request(
        router.address,
        "POST",
        "/v1/audio/speech",
        Some("application/json"),
        invalid_streaming_speech,
    )
    .expect("invalid streaming speech response");
    assert!(response.starts_with(b"HTTP/1.1 400"));
    assert_eq!(capable.captures().len() + limited.captures().len(), 2);

    let invalid = multipart_body_with_format("srt", true, 32);
    let response = request(
        router.address,
        "POST",
        "/v1/audio/translations",
        Some("multipart/form-data; boundary=media-boundary"),
        &invalid,
    )
    .expect("invalid streaming subtitle response");
    assert!(response.starts_with(b"HTTP/1.1 400"));
    assert_eq!(capable.captures().len() + limited.captures().len(), 2);
}

fn multipart_body(stream: bool, file_size: usize) -> Vec<u8> {
    multipart_body_with_format("json", stream, file_size)
}

fn multipart_body_with_format(format: &str, stream: bool, file_size: usize) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(b"--media-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF");
    output.extend(std::iter::repeat_n(b'a', file_size));
    output.extend_from_slice(b"\r\n--media-boundary-not-a-delimiter\r\n");
    output.extend_from_slice(
        b"--media-boundary\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n",
    );
    output.extend_from_slice(format.as_bytes());
    output.extend_from_slice(b"\r\n");
    if stream {
        output.extend_from_slice(
            b"--media-boundary\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n",
        );
    }
    output.extend_from_slice(b"--media-boundary--\r\n");
    output
}

fn socket_guard() -> MutexGuard<'static, ()> {
    SOCKET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
) -> std::io::Result<Vec<u8>> {
    request_with_extra_headers(address, method, path, content_type, "", body)
}

fn request_with_extra_headers(
    address: SocketAddr,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    extra_headers: &str,
    body: &[u8],
) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    let content = content_type.map_or_else(String::new, |value| {
        format!(
            "Content-Type: {value}\r\nContent-Length: {}\r\n",
            body.len()
        )
    });
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\n{content}{extra_headers}Connection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn expect_rejected_request(
    address: SocketAddr,
    path: &str,
    content_type: &str,
    content_length: usize,
) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: {content_type}\r\nContent-Length: {content_length}\r\nExpect: 100-continue\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn chunked_request(
    address: SocketAddr,
    path: &str,
    content_type: &str,
    chunks: &[&[u8]],
) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )?;
    for chunk in chunks {
        write!(stream, "{:X}\r\n", chunk.len())?;
        stream.write_all(chunk)?;
        stream.write_all(b"\r\n")?;
    }
    stream.write_all(b"0\r\n\r\n")?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn read_head(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let body = bytes.split_off(index + 4);
            bytes.truncate(index + 4);
            return String::from_utf8(bytes).ok().map(|head| (head, body));
        }
    }
}

fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

fn header<'a>(response: &'a [u8], name: &str) -> Option<&'a str> {
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&response[..split]).ok()?;
    header_value(head, name)
}

fn response_body(response: &[u8]) -> &[u8] {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .and_then(|index| response.get(index + 4..))
        .unwrap_or_default()
}
