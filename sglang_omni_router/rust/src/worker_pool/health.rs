use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use tokio::sync::watch;
use tokio::task::{JoinError, JoinSet};

use super::WorkerRecord;

/// Health observed by the worker probe loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum WorkerHealth {
    Unknown = 0,
    Healthy = 1,
    Unhealthy = 2,
}

impl WorkerHealth {
    fn from_atomic(value: u8) -> Self {
        match value {
            1 => Self::Healthy,
            2 => Self::Unhealthy,
            _ => Self::Unknown,
        }
    }
}

/// Atomic health cell. Release/Acquire publishes only the state transition;
/// capability remains immutable and load is owned by request leases.
pub(super) struct AtomicHealth(AtomicU8);

impl AtomicHealth {
    pub(super) const fn unknown() -> Self {
        Self(AtomicU8::new(WorkerHealth::Unknown as u8))
    }

    pub(super) fn load(&self) -> WorkerHealth {
        WorkerHealth::from_atomic(self.0.load(Ordering::Acquire))
    }

    pub(super) fn store(&self, state: WorkerHealth) {
        self.0.store(state as u8, Ordering::Release);
    }
}

pub(crate) struct HealthSupervisor {
    shutdown: watch::Sender<bool>,
    tasks: JoinSet<()>,
}

impl HealthSupervisor {
    pub(super) fn start(
        records: &[Arc<WorkerRecord>],
        client: reqwest::Client,
        interval: Duration,
        success_threshold: u8,
        failure_threshold: u8,
    ) -> Self {
        let (shutdown, receiver) = watch::channel(false);
        let mut tasks = JoinSet::new();
        for record in records {
            tasks.spawn(run_worker_health(
                Arc::clone(record),
                client.clone(),
                receiver.clone(),
                interval,
                success_threshold,
                failure_threshold,
            ));
        }
        Self { shutdown, tasks }
    }

    pub(crate) fn cancel(&self) {
        let _receivers = self.shutdown.send(true);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub(crate) async fn join_next(&mut self) -> Option<Result<(), JoinError>> {
        self.tasks.join_next().await
    }

    pub(crate) async fn abort_and_join_all(&mut self) -> Vec<JoinError> {
        self.tasks.abort_all();
        let mut failures = Vec::new();
        while let Some(result) = self.tasks.join_next().await {
            if let Err(source) = result
                && !source.is_cancelled()
            {
                failures.push(source);
            }
        }
        failures
    }

    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        let (shutdown, _receiver) = watch::channel(false);
        Self {
            shutdown,
            tasks: JoinSet::new(),
        }
    }
}

async fn run_worker_health(
    record: Arc<WorkerRecord>,
    client: reqwest::Client,
    mut shutdown: watch::Receiver<bool>,
    interval: Duration,
    success_threshold: u8,
    failure_threshold: u8,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut tracker = ProbeTracker::new(success_threshold, failure_threshold);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => return,
            () = record.immediate_probe.notified() => {}
            _ = ticker.tick() => {}
        }

        let response = tokio::select! {
            biased;
            _ = shutdown.changed() => return,
            result = client.get(record.target.health_url().clone()).send() => {
                result
            }
        };
        let success = response
            .as_ref()
            .is_ok_and(|response| response.status().is_success());
        let previous = record.health.load();
        let next = tracker.observe(success);
        record.health.store(next);
        if previous != next {
            tracing::info!(
                worker_id = record.worker_id.as_str(),
                health = ?next,
                "worker health changed"
            );
        }

        if let Ok(mut response) = response {
            loop {
                tokio::select! {
                    biased;
                    _ = shutdown.changed() => return,
                    chunk = response.chunk() => {
                        if !matches!(chunk, Ok(Some(_))) {
                            break;
                        }
                    }
                }
            }
        }
    }
}

struct ProbeTracker {
    successes: u8,
    failures: u8,
    success_threshold: u8,
    failure_threshold: u8,
    state: WorkerHealth,
}

impl ProbeTracker {
    fn new(success_threshold: u8, failure_threshold: u8) -> Self {
        Self {
            successes: 0,
            failures: 0,
            success_threshold,
            failure_threshold,
            state: WorkerHealth::Unknown,
        }
    }

    fn observe(&mut self, success: bool) -> WorkerHealth {
        if success {
            self.failures = 0;
            self.successes = self.successes.saturating_add(1);
            if self.successes >= self.success_threshold {
                self.state = WorkerHealth::Healthy;
            }
        } else {
            self.successes = 0;
            self.failures = self.failures.saturating_add(1);
            if self.failures >= self.failure_threshold {
                self.state = WorkerHealth::Unhealthy;
            }
        }
        self.state
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use tokio::sync::Notify;

    use super::{AtomicHealth, HealthSupervisor, ProbeTracker, WorkerHealth};
    use crate::worker_pool::WorkerRecord;
    use crate::worker_pool::profile::{
        InputModality, MessageContentForm, OutputModality, RegistrationId, ServiceProfile,
        StreamMode, TrustDomain, WorkerId,
    };
    use crate::worker_pool::resolver::{ResolvedTarget, build_health_client};

    fn test_target(address: SocketAddr) -> ResolvedTarget {
        ResolvedTarget::from_parts(&format!("http://{address}/"), "/health")
            .expect("test target must build")
    }

    fn test_record(ordinal: usize, address: SocketAddr) -> Arc<WorkerRecord> {
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("worker-{ordinal}")),
            default_model_id: Some(String::from("omni")),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: test_target(address),
            trust_domain: TrustDomain::new(String::from("local")),
            profiles: vec![ServiceProfile::GenerationHttp {
                model_ids: vec![String::from("omni")],
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                chat_audio_formats: Vec::new(),
                stream_modes: vec![StreamMode::NonStreaming],
            }],
            active_requests: AtomicUsize::new(0),
            session_capacity: [None, None],
            health: AtomicHealth::unknown(),
            immediate_probe: Notify::new(),
        })
    }

    fn test_client(timeout: Duration) -> reqwest::Client {
        build_health_client(timeout, Duration::from_secs(60))
            .expect("test health client must build")
    }

    fn read_request_head(stream: &mut TcpStream) {
        let mut request = Vec::with_capacity(256);
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0_u8];
            stream
                .read_exact(&mut byte)
                .expect("read complete health request head");
            request.push(byte[0]);
            assert!(request.len() <= 8_192);
        }
    }

    #[test]
    fn hysteresis_preserves_unknown_and_requires_consecutive_observations() {
        let mut tracker = ProbeTracker::new(2, 3);
        assert_eq!(tracker.observe(true), WorkerHealth::Unknown);
        assert_eq!(tracker.observe(false), WorkerHealth::Unknown);
        assert_eq!(tracker.observe(false), WorkerHealth::Unknown);
        assert_eq!(tracker.observe(false), WorkerHealth::Unhealthy);
        assert_eq!(tracker.observe(true), WorkerHealth::Unhealthy);
        assert_eq!(tracker.observe(true), WorkerHealth::Healthy);
        assert_eq!(tracker.observe(false), WorkerHealth::Healthy);
    }

    #[tokio::test]
    async fn status_is_published_before_response_body_finishes() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind health fixture");
        let address = listener.local_addr().expect("read health fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept health probe");
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .expect("bound health request read");
            let mut request = [0_u8; 1024];
            let _bytes = stream.read(&mut request).expect("read health request");
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100000000\r\nConnection: close\r\n\r\n",
                )
                .expect("write health headers");
            thread::sleep(Duration::from_millis(300));
        });
        let record = test_record(0, address);
        let client = test_client(Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while record.health.load() != WorkerHealth::Healthy
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(record.health.load(), WorkerHealth::Healthy);
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(()))));
        server.join().expect("join health fixture server");
    }

    #[tokio::test]
    async fn completed_health_bodies_allow_connection_reuse() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind reuse fixture");
        let address = listener.local_addr().expect("read reuse fixture address");
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = Arc::clone(&requests);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept health connection");
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .expect("bound reused health connection");
            for _ in 0..2 {
                read_request_head(&mut stream);
                server_requests.fetch_add(1, Ordering::AcqRel);
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                    .expect("write reused health response");
            }
        });
        let record = test_record(0, address);
        let client = test_client(Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while record.health.load() != WorkerHealth::Healthy {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("initial health probe must complete");
        record.immediate_probe.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while requests.load(Ordering::Acquire) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("second probe must reuse the health connection");

        server.join().expect("join reuse fixture server");
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(()))));
    }

    #[tokio::test]
    async fn probe_timeout_is_observed_as_a_failed_health_result() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind timeout fixture");
        let address = listener.local_addr().expect("read timeout fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept timed health probe");
            let mut request = [0_u8; 1024];
            let _bytes = stream.read(&mut request).expect("read timed health probe");
            thread::sleep(Duration::from_millis(200));
        });
        let record = test_record(0, address);
        let client = test_client(Duration::from_millis(50));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_millis(150);
        while record.health.load() != WorkerHealth::Unhealthy
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(record.health.load(), WorkerHealth::Unhealthy);
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(()))));
        server.join().expect("join timeout fixture server");
    }

    #[tokio::test]
    async fn non_success_status_is_a_failed_health_result() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind non-2xx fixture");
        let address = listener.local_addr().expect("read non-2xx fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept non-2xx probe");
            let mut request = [0_u8; 1024];
            let _bytes = stream.read(&mut request).expect("read non-2xx probe");
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("write non-2xx response");
        });
        let record = test_record(0, address);
        let client = test_client(Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while record.health.load() != WorkerHealth::Unhealthy {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("non-2xx observation must complete");
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(()))));
        server.join().expect("join non-2xx fixture server");
    }

    #[tokio::test]
    async fn one_worker_never_overlaps_and_stalled_probe_cancels_and_joins() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stalled fixture");
        let address = listener.local_addr().expect("read stalled fixture address");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let overlap = Arc::new(AtomicBool::new(false));
        let server_started = Arc::clone(&started);
        let server_release = Arc::clone(&release);
        let server_overlap = Arc::clone(&overlap);
        let server = thread::spawn(move || {
            let (mut stalled, _) = listener.accept().expect("accept stalled probe");
            let mut request = [0_u8; 1024];
            let _bytes = stalled.read(&mut request).expect("read stalled probe");
            listener
                .set_nonblocking(true)
                .expect("set stalled listener nonblocking");
            server_started.store(true, Ordering::Release);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !server_release.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((_second, _)) => server_overlap.store(true, Ordering::Release),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("stalled fixture accept failed: {error}"),
                }
                thread::yield_now();
            }
            assert!(server_release.load(Ordering::Acquire));
            drop(stalled);
        });
        let record = test_record(0, address);
        let client = test_client(Duration::from_secs(5));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while !started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stalled probe must start");
        for _ in 0..32 {
            record.immediate_probe.notify_one();
            tokio::task::yield_now().await;
        }
        assert!(!overlap.load(Ordering::Acquire));

        supervisor.cancel();
        let joined = tokio::time::timeout(Duration::from_millis(500), supervisor.join_next())
            .await
            .expect("stalled probe cancellation must be bounded");
        assert!(matches!(joined, Some(Ok(()))));
        release.store(true, Ordering::Release);
        server.join().expect("join stalled fixture server");
        assert!(!overlap.load(Ordering::Acquire));
        assert!(supervisor.is_empty());
    }

    #[tokio::test]
    async fn immediate_requests_coalesce_to_one_pending_probe() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind coalescing fixture");
        let address = listener
            .local_addr()
            .expect("read coalescing fixture address");
        let count = Arc::new(AtomicUsize::new(0));
        let first_probe_active = Arc::new(AtomicU8::new(0));
        let server_count = Arc::clone(&count);
        let server_first_probe_active = Arc::clone(&first_probe_active);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept first health probe");
            server_count.fetch_add(1, Ordering::AcqRel);
            let mut request = [0_u8; 1024];
            let _bytes = stream
                .read(&mut request)
                .expect("read first coalesced probe");
            server_first_probe_active.store(1, Ordering::Release);
            thread::sleep(Duration::from_millis(250));
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write first coalesced health response");
            server_first_probe_active.store(0, Ordering::Release);
            listener
                .set_nonblocking(true)
                .expect("set coalescing fixture nonblocking");
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        server_count.fetch_add(1, Ordering::AcqRel);
                        stream
                            .set_nonblocking(false)
                            .expect("set second coalesced probe blocking");
                        let _bytes = stream
                            .read(&mut request)
                            .expect("read second coalesced probe");
                        stream
                            .write_all(
                                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .expect("write second coalesced health response");
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("coalescing fixture accept failed: {error}"),
                }
            }
            let extra_deadline = std::time::Instant::now() + Duration::from_millis(150);
            while std::time::Instant::now() < extra_deadline {
                match listener.accept() {
                    Ok((_stream, _)) => {
                        server_count.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("coalescing fixture extra accept failed: {error}"),
                }
            }
            assert_eq!(server_count.load(Ordering::Acquire), 2);
        });
        let record = test_record(0, address);
        let client = test_client(Duration::from_secs(1));
        let mut supervisor = HealthSupervisor::start(
            &[Arc::clone(&record)],
            client,
            Duration::from_secs(60),
            1,
            1,
        );
        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while first_probe_active.load(Ordering::Acquire) == 0
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(first_probe_active.load(Ordering::Acquire), 1);
        assert_eq!(count.load(Ordering::Acquire), 1);
        for _ in 0..32 {
            record.immediate_probe.notify_one();
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while count.load(Ordering::Acquire) < 2 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        server.join().expect("join coalescing health fixture");
        assert_eq!(count.load(Ordering::Acquire), 2);
        supervisor.cancel();
        assert!(matches!(supervisor.join_next().await, Some(Ok(()))));
    }

    #[tokio::test]
    async fn stalled_worker_does_not_delay_other_workers() {
        let stalled_listener =
            TcpListener::bind("127.0.0.1:0").expect("bind stalled worker fixture");
        let stalled_address = stalled_listener
            .local_addr()
            .expect("read stalled worker fixture address");
        let healthy_listener =
            TcpListener::bind("127.0.0.1:0").expect("bind healthy worker fixture");
        let healthy_address = healthy_listener
            .local_addr()
            .expect("read healthy worker fixture address");
        let stalled_started = Arc::new(AtomicBool::new(false));
        let stalled_release = Arc::new(AtomicBool::new(false));

        let server_started = Arc::clone(&stalled_started);
        let server_release = Arc::clone(&stalled_release);
        let stalled_server = thread::spawn(move || {
            let (mut stream, _) = stalled_listener
                .accept()
                .expect("accept stalled worker probe");
            let mut request = [0_u8; 1024];
            let _bytes = stream
                .read(&mut request)
                .expect("read stalled worker probe");
            server_started.store(true, Ordering::Release);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !server_release.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                thread::yield_now();
            }
            assert!(server_release.load(Ordering::Acquire));
        });

        let healthy_server_started = Arc::clone(&stalled_started);
        let healthy_server = thread::spawn(move || {
            let (mut stream, _) = healthy_listener
                .accept()
                .expect("accept healthy worker probe");
            let mut request = [0_u8; 1024];
            let _bytes = stream
                .read(&mut request)
                .expect("read healthy worker probe");
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !healthy_server_started.load(Ordering::Acquire)
                && std::time::Instant::now() < deadline
            {
                thread::yield_now();
            }
            assert!(healthy_server_started.load(Ordering::Acquire));
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write healthy worker response");
        });

        let records = [
            test_record(0, stalled_address),
            test_record(1, healthy_address),
        ];
        let client = test_client(Duration::from_secs(5));
        let mut supervisor =
            HealthSupervisor::start(&records, client, Duration::from_secs(60), 1, 1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while records[1].health.load() != WorkerHealth::Healthy {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("healthy worker probe must not wait for stalled worker");
        assert!(stalled_started.load(Ordering::Acquire));
        assert_eq!(records[0].health.load(), WorkerHealth::Unknown);

        supervisor.cancel();
        while !supervisor.is_empty() {
            assert!(matches!(supervisor.join_next().await, Some(Ok(()))));
        }
        stalled_release.store(true, Ordering::Release);
        stalled_server.join().expect("join stalled worker fixture");
        healthy_server.join().expect("join healthy worker fixture");
    }
}
