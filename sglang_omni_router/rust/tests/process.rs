#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Real child-process, socket, route, and signal tests.

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
static PROCESS_LOCK: Mutex<()> = Mutex::new(());
const PROCESS_DEADLINE: Duration = Duration::from_secs(10);
const MAX_LOCAL_RESPONSE_BYTES: usize = 4 * 1024;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-process-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated process test directory");
        Self(path)
    }

    fn config(
        &self,
        address: SocketAddr,
        max_connections: usize,
        drain_timeout_ms: u64,
    ) -> PathBuf {
        self.config_with_timeouts(address, max_connections, 300, drain_timeout_ms)
    }

    fn config_with_timeouts(
        &self,
        address: SocketAddr,
        max_connections: usize,
        header_read_timeout_ms: u64,
        drain_timeout_ms: u64,
    ) -> PathBuf {
        let path = self.0.join("router.toml");
        let contents = format!(
            "schema_version = 1\n\n[server]\nlisten = \"{address}\"\nmax_connections = {max_connections}\nheader_read_timeout_ms = {header_read_timeout_ms}\n\n[shutdown]\ndrain_timeout_ms = {drain_timeout_ms}\n\n[logging]\nformat = \"json\"\nfilter = \"info\"\n\n[router]\nstrategy = \"round_robin\"\n\n[admission]\nglobal = 128\ngeneration_http = 64\n\n[health]\ninterval_ms = 100\ntimeout_ms = 50\nsuccess_threshold = 1\nfailure_threshold = 1\n\n[http_generation]\ntrust_domain = \"local\"\nstreamed_request_max_bytes = 1048576\nconnect_timeout_ms = 1000\nrequest_timeout_ms = 5000\npool_idle_timeout_ms = 30000\npool_max_idle_per_host = 8\n\n[[workers]]\nworker_id = \"worker-a\"\nbase_url = \"http://127.0.0.1:1/\"\ntrust_domain = \"local\"\ndefault_model_id = \"omni\"\nhealth_path = \"/health\"\n\n[[workers.service_profiles]]\nservice = \"generation_http\"\nmodel_ids = [\"omni\"]\nmessage_content_forms = [\"string\"]\nmedia_placements = []\ninput_modalities = [\"text\"]\noutput_modalities = [\"text\"]\nchat_audio_formats = []\nstream_modes = [\"non_streaming\"]\n"
        );
        fs::write(&path, contents).expect("write isolated process config");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _cleanup_result = fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Child);

impl ChildGuard {
    fn spawn(config: &PathBuf) -> Self {
        let child = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn production router binary");
        Self(child)
    }

    fn spawn_with_nofile(config: &PathBuf, soft_limit: u64) -> Self {
        let child = Command::new("sh")
            .arg("-c")
            .arg("ulimit -n \"$1\" && exec \"$2\" --config \"$3\"")
            .arg("sh")
            .arg(soft_limit.to_string())
            .arg(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn router with bounded RLIMIT_NOFILE");
        Self(child)
    }

    fn spawn_with_soft_nofile(config: &PathBuf, soft_limit: u64) -> Self {
        let child = Command::new("sh")
            .arg("-c")
            .arg("ulimit -S -n \"$1\" && exec \"$2\" --config \"$3\"")
            .arg("sh")
            .arg(soft_limit.to_string())
            .arg(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn router with a low RLIMIT_NOFILE soft limit");
        Self(child)
    }

    fn id(&self) -> u32 {
        self.0.id()
    }

    fn wait(&mut self, deadline: Duration) -> ExitStatus {
        let end = Instant::now() + deadline;
        loop {
            if let Some(status) = self.0.try_wait().expect("poll child status") {
                return status;
            }
            assert!(Instant::now() < end, "child did not exit before deadline");
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn assert_running(&mut self) {
        if let Some(status) = self.0.try_wait().expect("poll child while starting") {
            let mut stderr = String::new();
            if let Some(mut pipe) = self.0.stderr.take() {
                pipe.read_to_string(&mut stderr)
                    .expect("read failed child diagnostic");
            }
            panic!("router exited during startup with {status}: {stderr}");
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        match self.0.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                let _kill_result = self.0.kill();
                let _wait_result = self.0.wait();
            }
        }
    }
}

fn unused_address() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve ephemeral address");
    listener.local_addr().expect("read ephemeral address")
}

fn signal(process_id: u32, name: &str) {
    let status = Command::new("kill")
        .arg(name)
        .arg(process_id.to_string())
        .status()
        .expect("invoke platform signal utility");
    assert!(status.success(), "signal utility failed");
}

fn rapid_distinct_signals(process_id: u32) {
    let start = Arc::new(Barrier::new(3));
    thread::scope(|scope| {
        for name in ["-TERM", "-INT"] {
            let thread_start = Arc::clone(&start);
            scope.spawn(move || {
                thread_start.wait();
                signal(process_id, name);
            });
        }
        start.wait();
    });
}

fn request(address: SocketAddr, method: &str, path: &str) -> String {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(250))
        .expect("connect to serving router");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set bounded response timeout");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .expect("write HTTP request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read bounded local HTTP response");
    response
}

fn confirmed_keep_alive_connection(address: SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(250))
        .expect("connect confirmed keep-alive client");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set bounded keep-alive read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .expect("set bounded keep-alive write timeout");
    stream
        .write_all(b"GET /live HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n")
        .expect("write keep-alive proof request");

    let mut response = Vec::new();
    let header_end = loop {
        if let Some(offset) = response.windows(4).position(|window| window == b"\r\n\r\n") {
            break offset;
        }
        assert!(
            response.len() < MAX_LOCAL_RESPONSE_BYTES,
            "local response headers exceeded the test bound"
        );
        let mut buffer = [0_u8; 512];
        let read = stream
            .read(&mut buffer)
            .expect("read bounded keep-alive response headers");
        assert_ne!(read, 0, "server closed before completing response headers");
        response.extend_from_slice(&buffer[..read]);
    };

    let headers =
        std::str::from_utf8(&response[..header_end]).expect("local response headers must be UTF-8");
    let mut lines = headers.split("\r\n");
    assert_eq!(lines.next(), Some("HTTP/1.1 200 OK"));
    let mut content_length = None;
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .expect("local response header must contain a colon");
        if name.eq_ignore_ascii_case("content-length") {
            assert!(content_length.is_none(), "duplicate Content-Length header");
            content_length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .expect("Content-Length must be a decimal integer"),
            );
        }
    }
    let content_length = content_length.expect("local response must have Content-Length");
    let body_start = header_end + 4;
    let response_end = body_start
        .checked_add(content_length)
        .expect("local response length must not overflow");
    assert!(
        response_end <= MAX_LOCAL_RESPONSE_BYTES,
        "local response exceeded the test bound"
    );
    assert!(
        response.len() <= response_end,
        "local response contained bytes beyond its declared body"
    );
    while response.len() < response_end {
        let mut buffer = [0_u8; 512];
        let remaining = response_end - response.len();
        let read_limit = remaining.min(buffer.len());
        let read = stream
            .read(&mut buffer[..read_limit])
            .expect("read bounded keep-alive response body");
        assert_ne!(read, 0, "server closed before completing response body");
        response.extend_from_slice(&buffer[..read]);
    }
    assert_eq!(&response[body_start..response_end], b"live\n");

    stream
}

fn active_partial_connection(address: SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect(address).expect("open active HTTP connection");
    stream
        .write_all(b"GET /live HTTP/1.1\r\nHost: localhost\r\n")
        .expect("write incomplete request");
    stream
}

fn request_blocked_by_connection_cap(address: SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect(address).expect("connect capacity-waiting client");
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .expect("set bounded blocked-response timeout");
    stream
        .write_all(b"GET /live HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .expect("write complete capacity-waiting request");
    let mut byte = [0_u8; 1];
    let blocked = stream
        .read(&mut byte)
        .expect_err("capped request must not receive a response");
    assert!(matches!(
        blocked.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    ));
    stream
}

fn process_lock() -> MutexGuard<'static, ()> {
    PROCESS_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait_until_live(address: SocketAddr, child: &mut ChildGuard) {
    let end = Instant::now() + PROCESS_DEADLINE;
    let mut last_observation = String::from("no connection attempt completed");
    loop {
        if let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(50)) {
            stream
                .set_read_timeout(Some(Duration::from_millis(100)))
                .expect("set startup probe timeout");
            let _write_result = stream
                .write_all(b"GET /live HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
            let mut response = String::new();
            let read_result = stream.read_to_string(&mut response);
            if response.starts_with("HTTP/1.1 200") && response.ends_with("live\n") {
                return;
            }
            last_observation = format!("read={read_result:?}, response={response:?}");
        }
        child.assert_running();
        assert!(
            Instant::now() < end,
            "router never became live: {last_observation}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn help_version_and_check_config_have_exact_process_outcomes() {
    let _process_guard = process_lock();
    let help = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(help.status.success());
    let help_text = String::from_utf8(help.stdout).expect("help is UTF-8");
    assert!(help_text.contains("--config <PATH>"));
    assert!(help_text.contains("--check-config"));

    let version = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--version")
        .output()
        .expect("run --version");
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).expect("version is UTF-8"),
        "sgl-omni-router 0.1.0\n"
    );

    let directory = TestDir::new();
    let occupied = TcpListener::bind("127.0.0.1:0").expect("occupy ephemeral listener");
    let config = directory.config(
        occupied.local_addr().expect("read occupied address"),
        1024,
        1_000,
    );
    let checked = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(config)
        .arg("--check-config")
        .output()
        .expect("run --check-config");
    assert!(checked.status.success());
    assert_eq!(checked.stdout, b"configuration valid\n");
}

#[test]
fn check_config_ignores_the_process_file_limit() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let config = directory.config(unused_address(), 1024, 1_000);
    let checked = Command::new("sh")
        .arg("-c")
        .arg("ulimit -n \"$1\" && exec \"$2\" --config \"$3\" --check-config")
        .arg("sh")
        .arg("32")
        .arg(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg(config)
        .output()
        .expect("check config with an impossible process file limit");

    assert!(checked.status.success());
    assert_eq!(checked.stdout, b"configuration valid\n");
}

#[test]
fn invalid_connection_caps_fail_check_config_with_exit_two() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();

    for max_connections in [0, tokio::sync::Semaphore::MAX_PERMITS + 1] {
        let config = directory.config(address, max_connections, 1_000);
        let checked = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
            .arg("--config")
            .arg(config)
            .arg("--check-config")
            .output()
            .expect("run --check-config with invalid connection cap");
        assert_eq!(checked.status.code(), Some(2));
    }
}

#[test]
fn low_hard_nofile_limit_warns_and_serves() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config(address, 128, 1_000);
    let mut child = ChildGuard::spawn_with_nofile(&config, 192);

    wait_until_live(address, &mut child);
    signal(child.id(), "-TERM");
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(0));
    let mut diagnostics = String::new();
    child
        .0
        .stdout
        .take()
        .expect("file-limit log stream remains available")
        .read_to_string(&mut diagnostics)
        .expect("read file-limit log");
    child
        .0
        .stderr
        .take()
        .expect("file-limit warning stream remains available")
        .read_to_string(&mut diagnostics)
        .expect("read file-limit warning");
    assert!(
        diagnostics.contains("RLIMIT_NOFILE remains below the recommended target"),
        "missing file-limit warning: {diagnostics}"
    );
}

#[test]
fn low_soft_nofile_limit_is_raised_before_serving() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config(address, 128, 1_000);
    let mut child = ChildGuard::spawn_with_soft_nofile(&config, 32);

    wait_until_live(address, &mut child);
    signal(child.id(), "-TERM");
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(0));
}

#[test]
fn invalid_cli_and_config_exit_two_without_disclosing_contents() {
    let _process_guard = process_lock();
    let cli = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .output()
        .expect("run invalid CLI");
    assert_eq!(cli.status.code(), Some(2));

    let directory = TestDir::new();
    let path = directory.0.join("secret-path.toml");
    fs::write(&path, "schema_version = 1\nsecret_value = \"do-not-log\"\n")
        .expect("write invalid config");
    let invalid = Command::new(env!("CARGO_BIN_EXE_sgl-omni-router"))
        .arg("--config")
        .arg(&path)
        .output()
        .expect("run invalid config");
    assert_eq!(invalid.status.code(), Some(2));
    let stderr = String::from_utf8(invalid.stderr).expect("diagnostic is UTF-8");
    assert!(!stderr.contains("do-not-log"));
    assert!(stderr.contains("secret-path.toml"));
}

#[test]
fn serves_only_exact_local_liveness_route_and_shuts_down_cleanly() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config(address, 1024, 2_000);
    let mut child = ChildGuard::spawn(&config);
    wait_until_live(address, &mut child);

    let live = request(address, "GET", "/live");
    assert!(live.starts_with("HTTP/1.1 200"));
    assert!(live.ends_with("live\n"));
    for (method, path, status) in [
        ("POST", "/live", "405"),
        ("HEAD", "/live", "405"),
        ("GET", "/live/", "404"),
        ("GET", "/ready", "503"),
        ("PUT", "/ready", "405"),
        ("GET", "/ready/", "404"),
        ("GET", "/health", "404"),
        ("GET", "/metrics", "404"),
        ("GET", "/v1/models", "404"),
        ("POST", "/generate", "404"),
    ] {
        let response = request(address, method, path);
        assert!(
            response.starts_with(&format!("HTTP/1.1 {status}")),
            "unexpected {method} {path} response: {response}"
        );
    }

    signal(child.id(), "-TERM");
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(0));
    let reused = TcpListener::bind(address).expect("listener is reusable after joined shutdown");
    drop(reused);
}

#[test]
fn bind_failure_exits_one_without_disturbing_the_existing_listener() {
    let _process_guard = process_lock();
    let occupied = TcpListener::bind("127.0.0.1:0").expect("occupy ephemeral listener");
    let address = occupied.local_addr().expect("read occupied address");
    let directory = TestDir::new();
    let config = directory.config(address, 1024, 1_000);
    let mut child = ChildGuard::spawn(&config);
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(1));
    let mut stderr = String::new();
    child
        .0
        .stderr
        .take()
        .expect("bind failure stderr remains available")
        .read_to_string(&mut stderr)
        .expect("read bind failure diagnostic");
    assert!(stderr.contains("Address already in use"));
    assert_eq!(
        occupied.local_addr().expect("existing listener remains"),
        address
    );
}

#[test]
fn rapid_distinct_signals_force_process_exit_with_an_active_connection() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config(address, 1, 5_000);
    let mut child = ChildGuard::spawn(&config);
    wait_until_live(address, &mut child);

    let partial = active_partial_connection(address);
    let queued = request_blocked_by_connection_cap(address);
    let forced_start = Instant::now();
    rapid_distinct_signals(child.id());
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(1));
    assert!(
        forced_start.elapsed() < Duration::from_secs(4),
        "rapid second signal was lost and drain reached its five-second deadline"
    );
    drop(partial);
    drop(queued);
    let reused =
        TcpListener::bind(address).expect("listener is reusable after forced process teardown");
    drop(reused);
}

#[test]
fn idle_keep_alive_connections_release_capacity_at_header_deadline() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config(address, 2, 2_000);
    let mut child = ChildGuard::spawn(&config);
    wait_until_live(address, &mut child);

    let held = (0..2)
        .map(|_| confirmed_keep_alive_connection(address))
        .collect::<Vec<_>>();

    let mut waiting = request_blocked_by_connection_cap(address);

    waiting
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set bounded resumed-response timeout");
    let mut response = String::new();
    waiting
        .read_to_string(&mut response)
        .expect("read response after bounded idle reclamation");
    assert!(response.starts_with("HTTP/1.1 200"));
    assert!(response.ends_with("live\n"));

    drop(held);
    signal(child.id(), "-TERM");
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(0));
}

#[test]
fn graceful_shutdown_closes_partial_connection_and_releases_capacity() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config(address, 1, 2_000);
    let mut child = ChildGuard::spawn(&config);
    wait_until_live(address, &mut child);

    let partial = active_partial_connection(address);
    let queued = request_blocked_by_connection_cap(address);
    signal(child.id(), "-TERM");
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(0));
    drop(partial);
    drop(queued);
    let reused = TcpListener::bind(address).expect("listener is reusable after capped drain");
    drop(reused);
}

#[test]
fn graceful_shutdown_stops_accepting_before_connections_finish() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config_with_timeouts(address, 1, 1_000, 3_000);
    let mut child = ChildGuard::spawn(&config);
    wait_until_live(address, &mut child);

    let partial = active_partial_connection(address);
    signal(child.id(), "-TERM");

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        match TcpStream::connect_timeout(&address, Duration::from_millis(20)) {
            Err(error) if error.kind() == std::io::ErrorKind::ConnectionRefused => break,
            Ok(stream) => drop(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::ConnectionReset
                ) => {}
            Err(error) => panic!("unexpected connection result during drain: {error}"),
        }
        child.assert_running();
        assert!(
            Instant::now() < deadline,
            "listener remained open after graceful shutdown started"
        );
        thread::sleep(Duration::from_millis(10));
    }
    child.assert_running();

    drop(partial);
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(0));
}

#[test]
fn drain_deadline_aborts_a_partial_connection() {
    let _process_guard = process_lock();
    let directory = TestDir::new();
    let address = unused_address();
    let config = directory.config_with_timeouts(address, 1, 5_000, 100);
    let mut child = ChildGuard::spawn(&config);
    wait_until_live(address, &mut child);

    let partial = active_partial_connection(address);
    signal(child.id(), "-TERM");
    assert_eq!(child.wait(PROCESS_DEADLINE).code(), Some(1));
    drop(partial);
    let reused = TcpListener::bind(address).expect("listener is reusable after forced drain");
    drop(reused);
}
