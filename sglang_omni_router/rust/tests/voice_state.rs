#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic)]

//! Exact-owner voice CRUD, relay, and upload-ordering socket proof.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static SOCKET_LOCK: Mutex<()> = Mutex::new(());
const DEADLINE: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
struct Captured {
    method: String,
    path: String,
    content_type: Option<String>,
    request_id: Option<String>,
    authorization: Option<String>,
    route_model: Option<String>,
    route_stream: Option<String>,
    custom: Option<String>,
    body: Vec<u8>,
}

struct Worker {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    captured: Arc<Mutex<Vec<Captured>>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind voice worker");
        let address = listener.local_addr().expect("read voice worker address");
        let stop = Arc::new(AtomicBool::new(false));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = Arc::clone(&stop);
        let thread_captured = Arc::clone(&captured);
        let thread = thread::spawn(move || {
            let mut connections = Vec::new();
            loop {
                let (stream, _) = listener.accept().expect("accept voice worker request");
                if thread_stop.load(Ordering::Acquire) {
                    break;
                }
                let captured = Arc::clone(&thread_captured);
                connections.push(thread::spawn(move || handle_connection(stream, captured)));
            }
            for connection in connections {
                connection.join().expect("join voice worker connection");
            }
        });
        Self {
            address,
            stop,
            captured,
            thread: Some(thread),
        }
    }

    fn captures(&self) -> Vec<Captured> {
        self.captured.lock().expect("read voice captures").clone()
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

fn handle_connection(mut stream: TcpStream, captured: Arc<Mutex<Vec<Captured>>>) {
    stream
        .set_read_timeout(Some(DEADLINE))
        .expect("bound voice worker read");
    stream
        .set_write_timeout(Some(DEADLINE))
        .expect("bound voice worker write");
    let Some((head, mut body)) = read_head(&mut stream) else {
        return;
    };
    let request_line = head.lines().next().unwrap_or_default();
    if request_line.starts_with("GET /health ") {
        write_response(&mut stream, "200 OK", &[], b"");
        return;
    }
    let mut parts = request_line.split_ascii_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    let expected = header_value(&head, "content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    while body.len() < expected {
        let mut chunk = [0_u8; 4096];
        let count = stream.read(&mut chunk).expect("read voice request body");
        if count == 0 {
            return;
        }
        body.extend_from_slice(&chunk[..count]);
    }
    body.truncate(expected);
    captured
        .lock()
        .expect("record voice request")
        .push(Captured {
            method: method.clone(),
            path: path.clone(),
            content_type: header_value(&head, "content-type").map(str::to_owned),
            request_id: header_value(&head, "x-request-id").map(str::to_owned),
            authorization: header_value(&head, "authorization").map(str::to_owned),
            route_model: header_value(&head, "x-sglang-omni-route-model").map(str::to_owned),
            route_stream: header_value(&head, "x-sglang-omni-route-stream").map(str::to_owned),
            custom: header_value(&head, "x-custom-downstream").map(str::to_owned),
            body,
        });
    let response = match (method.as_str(), path.as_str()) {
        ("GET", "/v1/audio/voices?names_only=true") => {
            br#"{"uploaded_voice_names":["sample"]}"#.as_slice()
        }
        ("POST", "/v1/audio/voices") => br#"{"name":"sample","success":true}"#.as_slice(),
        ("DELETE", "/v1/audio/voices/name%20one") => {
            br#"{"success":true,"message":"deleted"}"#.as_slice()
        }
        _ => br#"{"error":"not found"}"#.as_slice(),
    };
    let status = if response == br#"{"error":"not found"}"# {
        "404 Not Found"
    } else {
        "200 OK"
    };
    write_response(
        &mut stream,
        status,
        &[
            ("Content-Type", "application/json"),
            ("X-Request-Id", "worker-must-not-win"),
        ],
        response,
    );
}

fn write_response(stream: &mut TcpStream, status: &str, headers: &[(&str, &str)], body: &[u8]) {
    let mut response = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        response.push_str(name);
        response.push_str(": ");
        response.push_str(value);
        response.push_str("\r\n");
    }
    response.push_str("\r\n");
    stream
        .write_all(response.as_bytes())
        .expect("write voice response head");
    stream.write_all(body).expect("write voice response body");
}

struct RouterProcess {
    child: Child,
    address: SocketAddr,
    directory: PathBuf,
}

impl RouterProcess {
    fn start(owner: &Worker, non_owner: &Worker) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve voice router port");
        let address = listener.local_addr().expect("read voice router address");
        drop(listener);
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "sgl-omni-router-voice-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create voice test directory");
        let config_path = directory.join("router.toml");
        fs::write(&config_path, config(address, owner, non_owner))
            .expect("write voice router config");
        let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(&config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start voice router");
        let process = Self {
            child,
            address,
            directory,
        };
        process.wait_ready();
        process
    }

    fn wait_ready(&self) {
        let deadline = Instant::now() + DEADLINE;
        while Instant::now() < deadline {
            if let Ok(response) = request(self.address, "GET", "/ready", &[], b"")
                && response.starts_with(b"HTTP/1.1 200")
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("voice router did not become ready");
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _killed = self.child.kill();
        let _waited = self.child.wait();
        let _cleanup = fs::remove_dir_all(&self.directory);
    }
}

fn config(address: SocketAddr, owner: &Worker, non_owner: &Worker) -> String {
    let mut output = format!(
        "schema_version = 1\n\n[server]\nlisten = \"{address}\"\nmax_connections = 32\n\n[shutdown]\ndrain_timeout_ms = 5000\n\n[logging]\nformat = \"json\"\nfilter = \"error\"\n\n[router]\nstrategy = \"least_requests\"\nvoice_owner_worker_id = \"owner\"\n\n[admission]\nglobal = 8\n\n[health]\ninterval_ms = 1000\ntimeout_ms = 500\nsuccess_threshold = 1\nfailure_threshold = 3\n\n[http]\nbuffered_request_total_bytes = 16777216\nconnect_timeout_ms = 1000\npool_idle_timeout_ms = 30000\npool_max_idle_per_host = 8\n"
    );
    for (id, worker) in [("owner", owner), ("non-owner", non_owner)] {
        let managed_voice = id == "owner";
        output.push_str(&format!(
            "\n[[workers]]\nworker_id = \"{id}\"\nbase_url = \"http://{}\"\ntrust_domain = \"local\"\nhealth_path = \"/health\"\ndefault_model_id = \"tts\"\n\n[[workers.service_profiles]]\nservice = \"speech_http\"\nmodel_ids = [\"tts\"]\nresponse_formats = [\"wav\"]\nstream_modes = [\"non_streaming\"]\ntasks = [\"text_to_speech\"]\nreference_forms = [\"none\"]\nmanaged_voice = {managed_voice}\n",
            worker.address,
        ));
    }
    output
}

#[test]
fn exact_owner_voice_crud_preserves_contract_and_upload_ordering() {
    let _guard = SOCKET_LOCK.lock().expect("serialize socket test");
    let owner = Worker::start();
    let non_owner = Worker::start();
    let router = RouterProcess::start(&owner, &non_owner);

    let common = [
        ("X-Request-Id", "caller-voice-id"),
        ("Authorization", "Bearer must-not-forward"),
        ("X-Sglang-Omni-Route-Model", "ignored"),
        ("X-Sglang-Omni-Route-Stream", "false"),
        ("X-Custom-Downstream", "must-not-forward"),
    ];
    let list = request(
        router.address,
        "GET",
        "/v1/audio/voices?names_only=true",
        &common,
        b"",
    )
    .expect("list voice response");
    assert!(list.starts_with(b"HTTP/1.1 200"));
    assert_eq!(
        response_header(&list, "x-request-id"),
        Some("caller-voice-id")
    );
    assert!(list.ends_with(br#"{"uploaded_voice_names":["sample"]}"#));

    let prefix = b"--voice-boundary\r\nContent-Disposition: form-data; name=\"audio_sample\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\n";
    let suffix = b"\r\n--voice-boundary--\r\n";
    let mut body = vec![b'x'; 10_551_296];
    body[..prefix.len()].copy_from_slice(prefix);
    let suffix_start = body.len() - suffix.len();
    body[suffix_start..].copy_from_slice(suffix);
    let upload_headers = [
        (
            "Content-Type",
            "multipart/form-data; boundary=voice-boundary",
        ),
        ("X-Request-Id", "upload-id"),
        ("X-Sglang-Omni-Route-Model", "ignored"),
        ("X-Sglang-Omni-Route-Stream", "false"),
    ];
    let upload = request(
        router.address,
        "POST",
        "/v1/audio/voices",
        &upload_headers,
        &body,
    )
    .expect("upload voice response");
    assert!(upload.starts_with(b"HTTP/1.1 200"));
    assert_eq!(response_header(&upload, "x-request-id"), Some("upload-id"));

    let delete = request(
        router.address,
        "DELETE",
        "/v1/audio/voices/name%20one",
        &[("X-Request-Id", "delete-id")],
        b"",
    )
    .expect("delete voice response");
    assert!(delete.starts_with(b"HTTP/1.1 200"));
    assert_eq!(response_header(&delete, "x-request-id"), Some("delete-id"));

    let before_method_rejections = owner.captures().len();
    let head = request(
        router.address,
        "HEAD",
        "/v1/audio/voices",
        &[("X-Request-Id", "head-id")],
        b"",
    )
    .expect("voice HEAD response");
    assert!(head.starts_with(b"HTTP/1.1 405"));
    assert_eq!(response_header(&head, "x-request-id"), Some("head-id"));
    assert_eq!(response_header(&head, "allow"), Some("GET, POST"));

    let unsupported = request(
        router.address,
        "PUT",
        "/v1/audio/voices/name%20one",
        &[("X-Request-Id", "put-id")],
        b"",
    )
    .expect("unsupported voice method response");
    assert!(unsupported.starts_with(b"HTTP/1.1 405"));
    assert_eq!(
        response_header(&unsupported, "x-request-id"),
        Some("put-id")
    );
    assert_eq!(response_header(&unsupported, "allow"), Some("DELETE"));
    assert_eq!(owner.captures().len(), before_method_rejections);

    let chat = request(
        router.address,
        "POST",
        "/v1/chat/completions",
        &[("Content-Type", "application/json")],
        b"{}",
    )
    .expect("voice-only chat response");
    assert!(chat.starts_with(b"HTTP/1.1 404"));

    let before_rejections = owner.captures().len();
    let oversized =
        request_with_declared_length(router.address, "/v1/audio/voices", 10_551_297, b"")
            .expect("oversized voice response");
    assert!(oversized.starts_with(b"HTTP/1.1 413"));
    assert_eq!(owner.captures().len(), before_rejections);

    let mut slow = TcpStream::connect(router.address).expect("open slow voice upload");
    slow.set_write_timeout(Some(DEADLINE))
        .expect("bound slow upload write");
    slow.write_all(
        b"POST /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=slow\r\nContent-Length: 64\r\nConnection: close\r\n\r\nx",
    )
    .expect("start slow upload");
    thread::sleep(Duration::from_millis(100));
    let during_upload = request(
        router.address,
        "GET",
        "/v1/audio/voices?names_only=true",
        &[("X-Request-Id", "parallel-id")],
        b"",
    )
    .expect("parallel list response");
    assert!(during_upload.starts_with(b"HTTP/1.1 200"));
    drop(slow);

    let captures = owner.captures();
    assert_eq!(captures.len(), before_rejections + 1);
    assert_eq!(captures[0].method, "GET");
    assert_eq!(captures[0].path, "/v1/audio/voices?names_only=true");
    assert_eq!(captures[0].request_id.as_deref(), Some("caller-voice-id"));
    assert!(captures[0].authorization.is_none());
    assert!(captures[0].route_model.is_none());
    assert!(captures[0].route_stream.is_none());
    assert!(captures[0].custom.is_none());
    assert_eq!(captures[1].method, "POST");
    assert_eq!(captures[1].path, "/v1/audio/voices");
    assert_eq!(
        captures[1].content_type.as_deref(),
        Some("multipart/form-data; boundary=voice-boundary")
    );
    assert_eq!(captures[1].body, body);
    assert_eq!(captures[1].request_id.as_deref(), Some("upload-id"));
    assert!(captures[1].route_model.is_none());
    assert!(captures[1].route_stream.is_none());
    assert_eq!(captures[2].method, "DELETE");
    assert_eq!(captures[2].path, "/v1/audio/voices/name%20one");
    assert_eq!(captures[3].request_id.as_deref(), Some("parallel-id"));
    assert!(non_owner.captures().is_empty());
}

fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<Vec<u8>> {
    request_with_length(address, method, path, headers, body.len(), body)
}

fn request_with_declared_length(
    address: SocketAddr,
    path: &str,
    length: usize,
    body: &[u8],
) -> std::io::Result<Vec<u8>> {
    request_with_length(
        address,
        "POST",
        path,
        &[("Content-Type", "multipart/form-data; boundary=voice")],
        length,
        body,
    )
}

fn request_with_length(
    address: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    length: usize,
    body: &[u8],
) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect(address)?;
    stream.set_read_timeout(Some(DEADLINE))?;
    stream.set_write_timeout(Some(DEADLINE))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {length}\r\nConnection: close\r\n"
    )?;
    for (name, value) in headers {
        write!(stream, "{name}: {value}\r\n")?;
    }
    stream.write_all(b"\r\n")?;
    stream.write_all(body)?;
    stream.flush()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

fn read_head(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 2048];
    loop {
        let count = stream.read(&mut buffer).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let body = bytes.split_off(index + 4);
            bytes.truncate(index);
            return String::from_utf8(bytes).ok().map(|head| (head, body));
        }
        if bytes.len() > 64 * 1024 {
            return None;
        }
    }
}

fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn response_header<'a>(response: &'a [u8], name: &str) -> Option<&'a str> {
    let boundary = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&response[..boundary]).ok()?;
    header_value(head, name)
}
