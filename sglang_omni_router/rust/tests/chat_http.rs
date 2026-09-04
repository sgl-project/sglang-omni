#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Real-socket proof for direct chat relay, replica routing, and health filtering.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static SOCKET_TEST_LOCK: Mutex<()> = Mutex::new(());
const DEADLINE: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
struct Captured {
    head: String,
    body: Vec<u8>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkerBehavior {
    ConsumeRequest,
    RespondAfterHeaders(&'static [u8]),
    IgnoreRequest,
}

struct Worker {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    health_requests: Arc<AtomicUsize>,
    captured: Arc<Mutex<Vec<Captured>>>,
    thread: Option<JoinHandle<()>>,
    _guard: Rc<MutexGuard<'static, ()>>,
}

impl Worker {
    fn start() -> Self {
        let guard = Rc::new(
            SOCKET_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        Self::start_with_guard(guard, WorkerBehavior::ConsumeRequest)
    }

    fn start_early_response(response: &'static [u8]) -> Self {
        let guard = Rc::new(
            SOCKET_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        Self::start_with_guard(guard, WorkerBehavior::RespondAfterHeaders(response))
    }

    fn start_stalled_upload() -> Self {
        let guard = Rc::new(
            SOCKET_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        Self::start_with_guard(guard, WorkerBehavior::IgnoreRequest)
    }

    fn start_pair() -> (Self, Self) {
        let guard = Rc::new(
            SOCKET_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        (
            Self::start_with_guard(Rc::clone(&guard), WorkerBehavior::ConsumeRequest),
            Self::start_with_guard(guard, WorkerBehavior::ConsumeRequest),
        )
    }

    fn start_with_guard(guard: Rc<MutexGuard<'static, ()>>, behavior: WorkerBehavior) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind worker fixture");
        listener
            .set_nonblocking(true)
            .expect("set worker nonblocking");
        let address = listener.local_addr().expect("read worker address");
        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(true));
        let health_requests = Arc::new(AtomicUsize::new(0));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = Arc::clone(&stop);
        let thread_healthy = Arc::clone(&healthy);
        let thread_health_requests = Arc::clone(&health_requests);
        let thread_captured = Arc::clone(&captured);
        let thread = thread::spawn(move || {
            let mut connections = Vec::new();
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _peer)) if !thread_stop.load(Ordering::Acquire) => {
                        let captures = Arc::clone(&thread_captured);
                        let healthy = Arc::clone(&thread_healthy);
                        let health_requests = Arc::clone(&thread_health_requests);
                        connections.push(thread::spawn(move || {
                            serve_connection(stream, captures, healthy, health_requests, behavior);
                        }));
                    }
                    Ok((_stream, _peer)) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("accept worker fixture: {error}"),
                }
                let mut index = 0;
                while index < connections.len() {
                    if connections[index].is_finished() {
                        let joined = connections.swap_remove(index);
                        let _result = joined.join();
                    } else {
                        index += 1;
                    }
                }
            }
            for connection in connections {
                let _result = connection.join();
            }
        });
        Self {
            address,
            stop,
            healthy,
            health_requests,
            captured,
            thread: Some(thread),
            _guard: guard,
        }
    }

    fn captures(&self) -> Vec<Captured> {
        self.captured.lock().expect("read captures").clone()
    }

    fn wait_for_requests(&self, count: usize) {
        let deadline = Instant::now() + DEADLINE;
        while self.captures().len() < count {
            assert!(Instant::now() < deadline, "worker did not receive request");
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn set_healthy(&self, healthy: bool) -> usize {
        self.healthy.store(healthy, Ordering::Release);
        self.health_requests.load(Ordering::Acquire)
    }

    fn wait_for_health_requests(&self, count: usize) {
        let deadline = Instant::now() + DEADLINE;
        while self.health_requests.load(Ordering::Acquire) < count {
            assert!(
                Instant::now() < deadline,
                "worker did not receive health probe"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _wake = TcpStream::connect(self.address);
        if let Some(thread) = self.thread.take() {
            let _result = thread.join();
        }
    }
}

fn read_request_head(stream: &mut TcpStream) -> Option<String> {
    let mut bytes = Vec::with_capacity(1024);
    let mut chunk = [0_u8; 1024];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let split = bytes.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    String::from_utf8(bytes[..split].to_vec()).ok()
}

fn serve_connection(
    mut stream: TcpStream,
    captured: Arc<Mutex<Vec<Captured>>>,
    healthy: Arc<AtomicBool>,
    health_requests: Arc<AtomicUsize>,
    behavior: WorkerBehavior,
) {
    stream
        .set_nonblocking(false)
        .expect("set worker connection blocking");
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound worker read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound worker write");
    if behavior != WorkerBehavior::ConsumeRequest {
        if let Some(head) = read_request_head(&mut stream) {
            if head.starts_with("GET /health HTTP/1.1") {
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                health_requests.fetch_add(1, Ordering::AcqRel);
            } else {
                captured.lock().expect("record request").push(Captured {
                    head,
                    body: Vec::new(),
                });
                match behavior {
                    WorkerBehavior::RespondAfterHeaders(response) => {
                        write_response(&mut stream, response);
                    }
                    WorkerBehavior::IgnoreRequest => thread::sleep(Duration::from_secs(3)),
                    WorkerBehavior::ConsumeRequest => unreachable!(),
                }
            }
        }
        return;
    }
    while let Some((head, body)) = read_request(&mut stream) {
        if head.starts_with("GET /health HTTP/1.1") {
            if healthy.load(Ordering::Acquire) {
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
                );
            } else {
                write_response(
                    &mut stream,
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
                );
            }
            health_requests.fetch_add(1, Ordering::AcqRel);
            continue;
        }
        captured.lock().expect("record request").push(Captured {
            head,
            body: body.clone(),
        });
        match body.as_slice() {
            b"not-json" => write_response(
                &mut stream,
                b"HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: application/json\r\nContent-Length: 22\r\nConnection: close\r\n\r\n{\"worker\":\"malformed\"}",
            ),
            b"unsupported-model" => write_response(
                &mut stream,
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 24\r\nConnection: close\r\n\r\n{\"worker\":\"unsupported\"}",
            ),
            b"slow" => {
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\nB\r\ndata: one\n\n\r\n",
                );
                thread::sleep(Duration::from_millis(750));
                write_response(&mut stream, b"E\r\ndata: [DONE]\n\n\r\n0\r\n\r\n");
            }
            b"timeout" => {
                thread::sleep(Duration::from_millis(750));
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
                );
            }
            b"reset" => return,
            b"mid-body-reset" => {
                write_response(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\nB\r\ndata: one\n\n\r\n",
                );
                return;
            }
            b"te-cl" => write_response(
                &mut stream,
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\nConnection: close\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
            ),
            b"te-compound" => write_response(
                &mut stream,
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: gzip, chunked\r\nConnection: close\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
            ),
            _ => write_response(
                &mut stream,
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nCache-Control: private\r\nCache-Control: max-age=0\r\nSet-Cookie: worker=1\r\nRetry-After: 1\r\nX-Request-ID: worker-id\r\nConnection: keep-alive\r\n\r\n{\"ok\":1}",
            ),
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut bytes = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 1024];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let split = bytes.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    let mut body = bytes.split_off(split);
    let head = String::from_utf8(bytes).ok()?;
    if let Some(length) = header(&head, "content-length").and_then(|value| value.parse().ok()) {
        while body.len() < length {
            let count = stream.read(&mut chunk).ok()?;
            if count == 0 {
                return None;
            }
            body.extend_from_slice(&chunk[..count]);
        }
        body.truncate(length);
    } else if header(&head, "transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        while !body.windows(5).any(|window| window == b"0\r\n\r\n") {
            let count = stream.read(&mut chunk).ok()?;
            if count == 0 {
                return None;
            }
            body.extend_from_slice(&chunk[..count]);
        }
        body = decode_chunks(&body)?;
    } else if head.starts_with("GET /health HTTP/1.1") {
        body.clear();
    } else {
        return None;
    }
    Some((head, body))
}

fn decode_chunks(encoded: &[u8]) -> Option<Vec<u8>> {
    let mut cursor = 0;
    let mut decoded = Vec::new();
    loop {
        let line_end = encoded[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")?
            + cursor;
        let length =
            usize::from_str_radix(std::str::from_utf8(&encoded[cursor..line_end]).ok()?, 16)
                .ok()?;
        cursor = line_end + 2;
        if length == 0 {
            return Some(decoded);
        }
        let end = cursor.checked_add(length)?;
        decoded.extend_from_slice(encoded.get(cursor..end)?);
        if encoded.get(end..end + 2)? != b"\r\n" {
            return None;
        }
        cursor = end + 2;
    }
}

fn write_response(stream: &mut TcpStream, bytes: &[u8]) {
    let _result = stream.write_all(bytes);
    let _result = stream.flush();
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.split("\r\n").skip(1).find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    directory: PathBuf,
}

#[derive(Clone, Copy)]
enum GenerationProfile {
    Text,
    TypedImage,
}

impl GenerationProfile {
    const fn manifest_fields(self) -> &'static str {
        match self {
            Self::Text => {
                "message_content_forms = [\"string\"]\nmedia_placements = []\ninput_modalities = [\"text\"]"
            }
            Self::TypedImage => {
                "message_content_forms = [\"string\", \"typed_parts\"]\nmedia_placements = [\"typed_parts\"]\ninput_modalities = [\"text\", \"image\"]"
            }
        }
    }
}

impl RouterProcess {
    fn start(worker: SocketAddr, global: u32, timeout_ms: u64, hostname: bool) -> Self {
        Self::start_configured(
            &[("worker-a", worker, hostname, GenerationProfile::Text)],
            global,
            timeout_ms,
            "round_robin",
            1_048_576,
        )
    }

    fn start_workers(
        workers: &[(&str, SocketAddr, bool, GenerationProfile)],
        global: u32,
        timeout_ms: u64,
    ) -> Self {
        Self::start_configured(workers, global, timeout_ms, "round_robin", 1_048_576)
    }

    fn start_configured(
        workers: &[(&str, SocketAddr, bool, GenerationProfile)],
        global: u32,
        timeout_ms: u64,
        strategy: &str,
        streamed_request_max_bytes: u64,
    ) -> Self {
        let workers: Vec<_> = workers
            .iter()
            .map(|(worker_id, address, hostname, profile)| {
                (*worker_id, *address, *hostname, *profile, "omni")
            })
            .collect();
        Self::start_configured_models(
            &workers,
            global,
            timeout_ms,
            strategy,
            streamed_request_max_bytes,
        )
    }

    fn start_configured_models(
        workers: &[(&str, SocketAddr, bool, GenerationProfile, &str)],
        global: u32,
        timeout_ms: u64,
        strategy: &str,
        streamed_request_max_bytes: u64,
    ) -> Self {
        let reservation = TcpListener::bind("127.0.0.1:0").expect("reserve router address");
        let address = reservation.local_addr().expect("read router address");
        drop(reservation);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-core-http-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create router test directory");
        let config = directory.join("router.toml");
        let mut worker_config = String::new();
        for (worker_id, worker, hostname, profile, default_model_id) in workers {
            let base_url = if *hostname {
                format!("http://localhost:{}/", worker.port())
            } else {
                format!("http://{worker}/")
            };
            let profile_fields = profile.manifest_fields();
            worker_config.push_str(&format!(
                "\n[[workers]]\nworker_id = \"{worker_id}\"\nbase_url = \"{base_url}\"\ntrust_domain = \"local\"\ndefault_model_id = \"{default_model_id}\"\nhealth_path = \"/health\"\n\n[[workers.service_profiles]]\nservice = \"generation_http\"\nmodel_ids = [\"{default_model_id}\"]\n{profile_fields}\noutput_modalities = [\"text\"]\nchat_audio_formats = []\nstream_modes = [\"non_streaming\", \"streaming\"]\n"
            ));
        }
        fs::write(
            &config,
            format!(
                "schema_version = 1\n\n[server]\nlisten = \"{address}\"\nmax_connections = 128\n\n[shutdown]\ndrain_timeout_ms = 2000\n\n[logging]\nformat = \"json\"\nfilter = \"info\"\n\n[router]\nstrategy = \"{strategy}\"\n\n[admission]\nglobal = {global}\ngeneration_http = {global}\n\n[health]\ninterval_ms = 100\ntimeout_ms = 50\nsuccess_threshold = 1\nfailure_threshold = 1\n\n[http_generation]\ntrust_domain = \"local\"\nbuffered_request_max_bytes = 1048576\nbuffered_request_total_bytes = 2097152\nstreamed_request_max_bytes = {streamed_request_max_bytes}\nconnect_timeout_ms = 100\nrequest_timeout_ms = {timeout_ms}\npool_idle_timeout_ms = 30000\npool_max_idle_per_host = 8\n{worker_config}"
            ),
        )
        .expect("write router config");
        let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn router");
        let mut process = Self {
            child,
            address,
            directory,
        };
        process.wait_live();
        process.wait_ready();
        process
    }

    fn wait_live(&mut self) {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Ok(response) = raw_request(
                self.address,
                b"GET /live HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            ) && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            if let Some(status) = self.child.try_wait().expect("poll router") {
                panic!("router exited before liveness: {status}");
            }
            assert!(Instant::now() < deadline, "router did not become live");
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_ready(&mut self) {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Ok(response) = raw_request(
                self.address,
                b"GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            ) && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            if let Some(status) = self.child.try_wait().expect("poll router readiness") {
                panic!("router exited before readiness: {status}");
            }
            assert!(Instant::now() < deadline, "router did not become ready");
            thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _result = Command::new("kill")
                .arg("-TERM")
                .arg(self.child.id().to_string())
                .status();
            let _result = self.child.wait();
        }
        let _result = fs::remove_dir_all(&self.directory);
    }
}

fn raw_request(address: SocketAddr, request: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(500))?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    stream.write_all(request)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn post(address: SocketAddr, body: &[u8], request_id: Option<&str>) -> Vec<u8> {
    let request_id =
        request_id.map_or_else(String::new, |value| format!("X-Request-ID: {value}\r\n"));
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{request_id}Connection: close\r\n\r\n",
        body.len()
    );
    let mut request = head.into_bytes();
    request.extend_from_slice(body);
    raw_request(address, &request).expect("complete routed request")
}

fn post_when_admission_releases(address: SocketAddr) -> Vec<u8> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        let response = post(address, b"{}", None);
        if status(&response) != 429 {
            return response;
        }
        assert!(
            Instant::now() < deadline,
            "router admission remained reserved"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

fn status(response: &[u8]) -> u16 {
    let line_end = response
        .windows(2)
        .position(|window| window == b"\r\n")
        .expect("response status line");
    std::str::from_utf8(&response[..line_end])
        .expect("ASCII response status")
        .split_whitespace()
        .nth(1)
        .expect("response status code")
        .parse()
        .expect("numeric response status")
}

fn response_head(response: &[u8]) -> &str {
    let end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("complete response head");
    std::str::from_utf8(&response[..end]).expect("ASCII response head")
}

#[test]
fn worker_owns_body_semantics_and_receives_exact_bytes_and_request_id() {
    for (body, expected_status, marker) in [
        (&b"not-json"[..], 422, &b"malformed"[..]),
        (&b"unsupported-model"[..], 400, &b"unsupported"[..]),
    ] {
        let worker = Worker::start();
        let router = RouterProcess::start(worker.address, 8, 2_000, true);
        let response = post(router.address, body, Some("caller-id"));
        assert_eq!(
            status(&response),
            expected_status,
            "response={} captures={:?}",
            String::from_utf8_lossy(&response),
            worker.captures()
        );
        assert!(response.windows(marker.len()).any(|part| part == marker));
        assert!(
            response_head(&response)
                .to_ascii_lowercase()
                .contains("x-request-id: caller-id")
        );
        let captures = worker.captures();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].body, body);
        assert_eq!(header(&captures[0].head, "x-request-id"), Some("caller-id"));
        drop(router);
        drop(worker);
    }
}

#[test]
fn small_and_large_homogeneous_bodies_use_the_same_direct_worker_path() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 8, 2_000, false);
    let small = b"{}".to_vec();
    let mut large = Vec::with_capacity(70_000);
    large.extend_from_slice(b"{\"messages\":[{\"role\":\"user\",\"content\":\"");
    large.resize(69_998, b'x');
    large.extend_from_slice(b"\"}]}");

    assert_eq!(status(&post(router.address, &small, None)), 200);
    assert_eq!(status(&post(router.address, &large, None)), 200);
    worker.wait_for_requests(2);
    let captures = worker.captures();
    assert_eq!(captures[0].body, small);
    assert_eq!(captures[1].body, large);
}

#[test]
fn strict_envelopes_fail_before_dispatch_and_missing_ids_are_generated() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 8, 2_000, false);
    let valid = post(router.address, b"{}", None);
    assert_eq!(status(&valid), 200);
    let generated =
        header(response_head(&valid), "x-request-id").expect("generated downstream request ID");
    worker.wait_for_requests(1);
    assert_eq!(
        header(&worker.captures()[0].head, "x-request-id"),
        Some(generated)
    );

    for (request, expected_status, expected_id) in [
        (
            b"GET /live HTTP/1.1\r\nHost: localhost\r\nX-Request-ID: live-id\r\nConnection: close\r\n\r\n".as_slice(),
            200,
            Some("live-id"),
        ),
        (
            b"GET /missing HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".as_slice(),
            404,
            None,
        ),
    ] {
        let response = raw_request(router.address, request).expect("canonical route response");
        assert_eq!(status(&response), expected_status);
        let response_id =
            header(response_head(&response), "x-request-id").expect("canonical response ID");
        if let Some(expected_id) = expected_id {
            assert_eq!(response_id, expected_id);
        } else {
            assert!(response_id.starts_with("sglang-omni-"));
        }
    }

    let get = raw_request(
        router.address,
        b"GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nX-Request-ID: method-id\r\nConnection: close\r\n\r\n",
    )
    .expect("canonical method response");
    assert_eq!(status(&get), 405);
    let method_code = b"\"method_not_allowed\"";
    assert!(
        get.windows(method_code.len())
            .any(|part| part == method_code)
    );
    assert_eq!(
        header(response_head(&get), "x-request-id"),
        Some("method-id")
    );
    assert_eq!(
        header(response_head(&get), "content-type"),
        Some("application/json")
    );
    for field in [
        b"\"message\"".as_slice(),
        b"\"type\":\"invalid_request_error\"".as_slice(),
        b"\"param\":null".as_slice(),
        b"\"code\":\"method_not_allowed\"".as_slice(),
    ] {
        assert!(get.windows(field.len()).any(|part| part == field));
    }

    for (request, expected) in [
        (b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\n{}\r\n0\r\n\r\n".as_slice(), 400),
    ] {
        let response = raw_request(router.address, request).expect("router envelope response");
        assert_eq!(status(&response), expected);
    }

    for request in [
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nX-Request-ID: one\r\nX-Request-ID: two\r\nConnection: close\r\n\r\n{}".as_slice(),
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nX-Request-ID: has space\r\nConnection: close\r\n\r\n{}".as_slice(),
    ] {
        let response = raw_request(router.address, request).expect("invalid request-ID response");
        assert_eq!(status(&response), 400);
        let head = response_head(&response);
        let replacement = header(head, "x-request-id").expect("generated canonical replacement");
        assert!(replacement.starts_with("sglang-omni-"));
        assert!(replacement.len() <= 128);
        assert_eq!(
            head.to_ascii_lowercase().matches("x-request-id:").count(),
            1
        );
    }
    thread::sleep(Duration::from_millis(20));
    assert_eq!(worker.captures().len(), 1);
}

#[test]
fn standard_continue_expectation_is_terminated_locally_and_not_forwarded() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 8, 2_000, false);
    let mut client = TcpStream::connect(router.address).expect("connect expect client");
    client
        .set_read_timeout(Some(DEADLINE))
        .expect("bound expect response");
    client
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nExpect: 100-Continue\r\nConnection: close\r\n\r\n",
        )
        .expect("write expect request head");

    let mut interim = [0_u8; 64];
    let count = client.read(&mut interim).expect("read continue response");
    assert!(interim[..count].starts_with(b"HTTP/1.1 100 Continue\r\n\r\n"));
    client.write_all(b"{}").expect("write expected body");
    let mut response = Vec::new();
    client
        .read_to_end(&mut response)
        .expect("read final expect response");
    assert_eq!(status(&response), 200);

    worker.wait_for_requests(1);
    assert!(header(&worker.captures()[0].head, "expect").is_none());

    let rejected = raw_request(
        router.address,
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nExpect: custom\r\nConnection: close\r\n\r\n{}",
    )
    .expect("read unsupported expectation response");
    assert_eq!(status(&rejected), 417);
}

#[test]
fn relay_holds_admission_and_is_not_cut_off_after_commitment() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);

    let address = router.address;
    let slow = thread::spawn(move || post(address, b"slow", Some("slow-id")));
    worker.wait_for_requests(1);
    let oversized = raw_request(
        router.address,
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1048577\r\nConnection: close\r\n\r\n",
    )
    .expect("oversized response while admission is full");
    assert_eq!(status(&oversized), 413);
    let overloaded = post(router.address, b"{}", None);
    assert_eq!(status(&overloaded), 429);

    let slow_response = slow.join().expect("join slow client");
    assert_eq!(status(&slow_response), 200);
    assert!(slow_response.windows(6).any(|part| part == b"[DONE]"));

    drop(router);
    drop(worker);
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);
    let first = post(router.address, b"{}", None);
    assert_eq!(status(&first), 200);
    let head = response_head(&first).to_ascii_lowercase();
    assert_eq!(head.matches("cache-control:").count(), 2);
    assert!(head.contains("set-cookie: worker=1"));
    assert!(head.contains("retry-after: 1"));
    assert!(!head.contains("x-request-id: worker-id"));
    assert_eq!(head.matches("x-request-id:").count(), 1);
}

#[test]
fn homogeneous_replicas_rotate_and_unhealthy_workers_are_filtered() {
    let (first, second) = Worker::start_pair();
    let router = RouterProcess::start_workers(
        &[
            ("worker-a", first.address, false, GenerationProfile::Text),
            ("worker-b", second.address, false, GenerationProfile::Text),
        ],
        8,
        2_000,
    );

    for _ in 0..6 {
        assert_eq!(status(&post(router.address, b"{}", None)), 200);
    }
    first.wait_for_requests(3);
    second.wait_for_requests(3);
    assert_eq!(first.captures().len(), 3);
    assert_eq!(second.captures().len(), 3);

    let prior_health = first.set_healthy(false);
    first.wait_for_health_requests(prior_health + 2);
    let first_before = first.captures().len();
    let second_before = second.captures().len();
    for _ in 0..4 {
        assert_eq!(status(&post(router.address, b"{}", None)), 200);
    }
    assert_eq!(first.captures().len(), first_before);
    assert_eq!(second.captures().len(), second_before + 4);
}

#[test]
fn least_requests_prefers_the_less_occupied_replica() {
    let (first, second) = Worker::start_pair();
    let router = RouterProcess::start_configured(
        &[
            ("worker-a", first.address, false, GenerationProfile::Text),
            ("worker-b", second.address, false, GenerationProfile::Text),
        ],
        4,
        2_000,
        "least_requests",
        1_048_576,
    );

    let address = router.address;
    let slow = thread::spawn(move || post(address, b"slow", None));
    let deadline = Instant::now() + DEADLINE;
    while first.captures().is_empty() && second.captures().is_empty() {
        assert!(
            Instant::now() < deadline,
            "neither worker received the slow request"
        );
        thread::sleep(Duration::from_millis(2));
    }
    let (occupied, available) = if first.captures().len() == 1 {
        (&first, &second)
    } else {
        (&second, &first)
    };
    assert_eq!(status(&post(router.address, b"{}", None)), 200);
    assert_eq!(status(&post(router.address, b"{}", None)), 200);
    available.wait_for_requests(2);
    assert_eq!(occupied.captures().len(), 1);
    assert_eq!(available.captures().len(), 2);
    assert_eq!(status(&slow.join().expect("join slow client")), 200);
}

#[test]
fn heterogeneous_typed_image_request_reaches_only_the_compatible_worker() {
    let (text, image) = Worker::start_pair();
    let router = RouterProcess::start_workers(
        &[
            ("worker-a", text.address, false, GenerationProfile::Text),
            (
                "worker-b",
                image.address,
                false,
                GenerationProfile::TypedImage,
            ),
        ],
        8,
        2_000,
    );
    let body = br#"{"model":"omni","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}"#;

    assert_eq!(status(&post(router.address, body, Some("image-id"))), 200);
    image.wait_for_requests(1);
    assert!(text.captures().is_empty());
    let captures = image.captures();
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].body, body);
    assert_eq!(header(&captures[0].head, "x-request-id"), Some("image-id"));
}

#[test]
fn heterogeneous_classification_preserves_worker_owned_request_semantics() {
    let (text, image) = Worker::start_pair();
    let router = RouterProcess::start_workers(
        &[
            ("worker-a", text.address, false, GenerationProfile::Text),
            (
                "worker-b",
                image.address,
                false,
                GenerationProfile::TypedImage,
            ),
        ],
        8,
        2_000,
    );
    let requests = [
        br#"{"messages":[]}"#.as_slice(),
        br#"{"messages":[{"content":[{"type":"vendor_part"}]}]}"#.as_slice(),
        br#"{"messages":[{"content":"x"}],"stream":"true"}"#.as_slice(),
        br#"{"messages":[{"content":"x"}],"modalities":["video","text","text"],"audio":{"format":"vendor"}}"#.as_slice(),
        br#"{"messages":[{"content":"old"}],"messages":[{"content":[{"type":"image"}]}]}"#.as_slice(),
    ];

    for request in requests {
        assert_eq!(
            status(&post(router.address, request, None)),
            200,
            "{request:?}"
        );
    }

    let mut captures = text.captures();
    captures.extend(image.captures());
    assert_eq!(captures.len(), requests.len());
    for request in requests {
        assert!(
            captures.iter().any(|capture| capture.body == request),
            "request bytes changed: {request:?}"
        );
    }
}

#[test]
fn heterogeneous_limits_and_unsupported_profiles_fail_before_dispatch() {
    let (text, image) = Worker::start_pair();
    let router = RouterProcess::start_workers(
        &[
            ("worker-a", text.address, false, GenerationProfile::Text),
            (
                "worker-b",
                image.address,
                false,
                GenerationProfile::TypedImage,
            ),
        ],
        8,
        2_000,
    );

    let oversized = raw_request(
        router.address,
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1048577\r\nConnection: close\r\n\r\n",
    )
    .expect("oversized classified response");
    assert_eq!(status(&oversized), 413);

    let unsupported = br#"{"model":"omni","messages":[{"content":[{"type":"audio_url"}]}]}"#;
    assert_eq!(status(&post(router.address, unsupported, None)), 422);
    assert!(text.captures().is_empty());
    assert!(image.captures().is_empty());
}

#[test]
fn heterogeneous_near_limit_body_is_forwarded_byte_for_byte() {
    let (text, image) = Worker::start_pair();
    let router = RouterProcess::start_workers(
        &[
            ("worker-a", text.address, false, GenerationProfile::Text),
            (
                "worker-b",
                image.address,
                false,
                GenerationProfile::TypedImage,
            ),
        ],
        8,
        2_000,
    );
    let prefix = br#"{"model":"omni","messages":[{"content":[{"type":"image_url"}]}],"vendor":""#;
    let suffix = br#""}"#;
    let mut body = Vec::with_capacity(1_048_576);
    body.extend_from_slice(prefix);
    body.resize(1_048_576 - suffix.len(), b'x');
    body.extend_from_slice(suffix);

    assert_eq!(body.len(), 1_048_576);
    assert_eq!(status(&post(router.address, &body, None)), 200);
    image.wait_for_requests(1);
    assert_eq!(image.captures()[0].body, body);
    assert!(text.captures().is_empty());
}

#[test]
fn heterogeneous_default_model_ambiguity_requires_an_explicit_model() {
    let (first, second) = Worker::start_pair();
    let router = RouterProcess::start_configured_models(
        &[
            (
                "worker-a",
                first.address,
                false,
                GenerationProfile::Text,
                "omni",
            ),
            (
                "worker-b",
                second.address,
                false,
                GenerationProfile::Text,
                "other",
            ),
        ],
        8,
        2_000,
        "round_robin",
        1_048_576,
    );

    assert_eq!(
        status(&post(
            router.address,
            br#"{"messages":[{"content":"x"}]}"#,
            None
        )),
        400
    );
    assert!(first.captures().is_empty());
    assert!(second.captures().is_empty());
    assert_eq!(
        status(&post(
            router.address,
            br#"{"model":"omni","messages":[{"content":"x"}]}"#,
            None,
        )),
        200
    );
    first.wait_for_requests(1);
    assert_eq!(first.captures().len(), 1);
    assert!(second.captures().is_empty());
}

#[test]
fn precommit_timeout_and_upstream_reset_are_bounded_and_release_admission() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);

    let timeout = post(router.address, b"timeout", None);
    assert_eq!(
        status(&timeout),
        504,
        "{} captures={:?}",
        String::from_utf8_lossy(&timeout),
        worker.captures()
    );
    let recovered = post_when_admission_releases(router.address);
    assert_ne!(
        status(&recovered),
        429,
        "precommit timeout retained the sole admission permit"
    );

    drop(router);
    drop(worker);
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 500, false);
    let reset = post(router.address, b"reset", None);
    assert_eq!(status(&reset), 502);
}

#[test]
fn early_upload_eof_and_downstream_disconnect_release_admission() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 2_000, false);

    let mut short = TcpStream::connect(router.address).expect("connect short upload");
    short
        .set_read_timeout(Some(DEADLINE))
        .expect("bound short response");
    short
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 10\r\nConnection: close\r\n\r\n{}",
        )
        .expect("write short upload");
    short
        .shutdown(std::net::Shutdown::Write)
        .expect("finish short client write");
    let mut short_response = Vec::new();
    short
        .read_to_end(&mut short_response)
        .expect("read short-upload response");
    assert_eq!(status(&short_response), 400);

    let mut disconnect = TcpStream::connect(router.address).expect("connect disconnect client");
    disconnect
        .set_read_timeout(Some(DEADLINE))
        .expect("bound disconnect response");
    disconnect
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 4\r\nConnection: close\r\n\r\nslow",
        )
        .expect("write disconnect request");
    worker.wait_for_requests(1);
    let mut prefix = [0_u8; 256];
    let count = disconnect.read(&mut prefix).expect("read committed prefix");
    assert_ne!(count, 0);
    drop(disconnect);

    let released = post_when_admission_releases(router.address);
    assert_ne!(
        status(&released),
        429,
        "downstream drop retained the sole admission permit"
    );
}

fn assert_early_response_is_rejected(response: &'static [u8]) {
    let worker = Worker::start_early_response(response);
    let router = RouterProcess::start(worker.address, 1, 2_000, false);
    let mut client = TcpStream::connect(router.address).expect("connect streaming upload");
    client
        .set_read_timeout(Some(DEADLINE))
        .expect("bound early-response client read");
    client
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1048576\r\nConnection: close\r\n\r\n{",
        )
        .expect("write partial streaming upload");

    let mut response = Vec::new();
    client
        .read_to_end(&mut response)
        .expect("read early upstream response");

    assert_eq!(status(&response), 502);
    assert!(
        !response
            .windows(b"early".len())
            .any(|part| part == b"early")
    );

    let released = post_when_admission_releases(router.address);
    assert_ne!(status(&released), 429);
}

#[test]
fn early_upstream_response_is_rejected_before_upload_completion() {
    assert_early_response_is_rejected(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
    );
    assert_early_response_is_rejected(
        b"HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: application/json\r\nContent-Length: 18\r\nConnection: close\r\n\r\n{\"early\":\"reject\"}",
    );
}

#[test]
fn outer_deadline_bounds_a_backpressured_upload() {
    let worker = Worker::start_stalled_upload();
    let router = RouterProcess::start_configured(
        &[("worker-a", worker.address, false, GenerationProfile::Text)],
        1,
        300,
        "round_robin",
        64 * 1024 * 1024,
    );
    let reader = TcpStream::connect(router.address).expect("connect stalled upload");
    let mut writer = reader.try_clone().expect("clone stalled-upload client");
    writer
        .set_write_timeout(Some(DEADLINE))
        .expect("bound stalled-upload write");
    let upload = thread::spawn(move || {
        writer
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 67108864\r\nConnection: close\r\n\r\n",
            )
            .expect("write stalled-upload head");
        let chunk = [b' '; 64 * 1024];
        for _ in 0..1024 {
            if writer.write_all(&chunk).is_err() {
                break;
            }
        }
    });

    worker.wait_for_requests(1);

    let release_started = Instant::now();
    let released = post_when_admission_releases(router.address);
    assert_eq!(status(&released), 504);
    assert!(
        release_started.elapsed() < Duration::from_secs(2),
        "capacity was released only when the worker closed the request"
    );
    drop(reader);
    upload.join().expect("join stalled-upload writer");
}

#[test]
fn upstream_failure_after_sse_commitment_releases_capacity() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 2_000, false);

    let response = post(router.address, b"mid-body-reset", None);
    assert_eq!(status(&response), 200);
    assert!(response.windows(9).any(|part| part == b"data: one"));
    assert!(!response.windows(6).any(|part| part == b"[DONE]"));

    let recovered = post_when_admission_releases(router.address);
    assert_ne!(status(&recovered), 429);
}

#[test]
fn upstream_chunked_framing_is_normalized_before_commitment() {
    let worker = Worker::start();
    let router = RouterProcess::start(worker.address, 1, 2_000, false);

    let decoded = post(router.address, b"te-cl", None);
    assert_eq!(status(&decoded), 200);
    assert!(decoded.windows(2).any(|part| part == b"{}"));
    assert!(!response_head(&decoded).contains("content-length: 0"));

    let unsupported = post(router.address, b"te-compound", None);
    assert_eq!(status(&unsupported), 502);
}

#[test]
fn graceful_drain_waits_for_a_committed_relay() {
    let worker = Worker::start();
    let mut router = RouterProcess::start(worker.address, 1, 2_000, false);
    let address = router.address;
    let slow = thread::spawn(move || post(address, b"slow", None));
    worker.wait_for_requests(1);

    let signaled = Command::new("kill")
        .arg("-TERM")
        .arg(router.child.id().to_string())
        .status()
        .expect("signal router");
    assert!(signaled.success());
    let response = slow.join().expect("join draining relay");
    assert_eq!(status(&response), 200);
    assert!(response.windows(6).any(|part| part == b"[DONE]"));
    assert!(router.child.wait().expect("wait for router").success());
}
