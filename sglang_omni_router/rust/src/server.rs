use std::io;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use tokio::sync::{oneshot, watch};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{error, info, trace};

use crate::config::Config;
use crate::error::RouterError;
use crate::lifecycle::Lifecycle;
use crate::shutdown;

mod bounded_listener;

use bounded_listener::BoundedTcpListener;

const LIVE_BODY: &str = "live\n";

pub(crate) async fn serve(config: Config) -> Result<(), RouterError> {
    let lifecycle = Arc::new(Lifecycle::starting());
    let mut signal_observer = shutdown::SignalObserver::install().map_err(RouterError::Signal)?;
    let app = route_table(Arc::clone(&lifecycle));
    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .map_err(RouterError::Bind)?;
    let listener = BoundedTcpListener::new(listener, config.server.max_connections);
    let (shutdown_sender, shutdown_receiver) = oneshot::channel::<()>();

    lifecycle.enter_serving()?;
    info!(state = "serving", "local service started");

    let mut server_task = tokio::spawn(async move {
        serve_http(
            listener,
            app,
            config.server.header_read_timeout(),
            shutdown_receiver,
        )
        .await
    });

    let first_signal = tokio::select! {
        biased;
        task_result = &mut server_task => {
            lifecycle.enter_failed()?;
            return unexpected_server_exit(task_result);
        }
        signal_result = signal_observer.next() => match signal_result {
            Ok(signal) => signal,
            Err(source) => {
                abort_and_join(&mut server_task).await?;
                lifecycle.enter_failed()?;
                return Err(RouterError::Signal(source));
            }
        },
    };

    if lifecycle.enter_draining().is_err() {
        abort_and_join(&mut server_task).await?;
        lifecycle.enter_failed()?;
        return Err(RouterError::Lifecycle);
    }
    info!(state = "draining", reason = ?first_signal, "graceful shutdown started");
    if shutdown_sender.send(()).is_err() {
        abort_and_join(&mut server_task).await?;
        lifecycle.enter_failed()?;
        return Err(RouterError::ShutdownNotify);
    }

    let drain_timeout = config.shutdown.drain_timeout();
    let drain_result = tokio::select! {
        biased;
        task_result = &mut server_task => finish_graceful(task_result, &lifecycle),
        second_signal = signal_observer.next() => {
            match second_signal {
                Ok(signal) => {
                    error!(state = "draining", reason = ?signal, "second signal forced shutdown");
                    abort_and_join(&mut server_task).await?;
                    lifecycle.enter_failed()?;
                    Err(RouterError::ForcedShutdown)
                }
                Err(source) => {
                    abort_and_join(&mut server_task).await?;
                    lifecycle.enter_failed()?;
                    Err(RouterError::Signal(source))
                }
            }
        }
        () = tokio::time::sleep(drain_timeout) => {
            error!(state = "draining", "graceful shutdown deadline elapsed");
            abort_and_join(&mut server_task).await?;
            lifecycle.enter_failed()?;
            Err(RouterError::DrainTimeout)
        }
    };

    if drain_result.is_ok() {
        info!(
            state = "stopped",
            remaining_tasks = 0_u8,
            "shutdown complete"
        );
    }
    drain_result
}

async fn serve_http(
    listener: BoundedTcpListener,
    app: Router,
    header_read_timeout: std::time::Duration,
    mut shutdown: oneshot::Receiver<()>,
) -> io::Result<()> {
    let (connection_shutdown, _) = watch::channel(());
    let mut connections = JoinSet::new();

    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => break,
            joined = connections.join_next(), if !connections.is_empty() => {
                if let Some(result) = joined {
                    result.map_err(io::Error::other)?;
                }
            }
            (io, peer) = accept_connection(&listener) => {
                let app = app.clone();
                let mut shutdown = connection_shutdown.subscribe();
                connections.spawn(async move {
                    let mut builder = http1::Builder::new();
                    builder
                        .timer(TokioTimer::new())
                        .header_read_timeout(header_read_timeout);
                    let connection = builder
                        .serve_connection(TokioIo::new(io), TowerToHyperService::new(app))
                        .with_upgrades();
                    tokio::pin!(connection);
                    tokio::select! {
                        result = connection.as_mut() => {
                            if let Err(error) = result {
                                trace!(%peer, %error, "client connection closed with an HTTP error");
                            }
                        }
                        _ = shutdown.changed() => {
                            connection.as_mut().graceful_shutdown();
                            if let Err(error) = connection.await {
                                trace!(%peer, %error, "client connection closed with an HTTP error");
                            }
                        }
                    }
                });
            }
        }
    }

    drop(listener);
    let _notified = connection_shutdown.send(());
    while let Some(joined) = connections.join_next().await {
        joined.map_err(io::Error::other)?;
    }
    Ok(())
}

async fn accept_connection(
    listener: &BoundedTcpListener,
) -> (bounded_listener::ConnectionIo, std::net::SocketAddr) {
    loop {
        match listener.accept().await {
            Ok(connection) => return connection,
            Err(error) => handle_accept_error(&error).await,
        }
    }
}

async fn handle_accept_error(error: &io::Error) {
    if matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        trace!(%error, "client connection failed during accept");
        return;
    }

    error!(%error, "client accept failed; retrying");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

fn route_table(lifecycle: Arc<Lifecycle>) -> Router {
    Router::new()
        .route("/live", get(live).head(reject_head))
        .with_state(lifecycle)
}

async fn live(State(lifecycle): State<Arc<Lifecycle>>) -> (StatusCode, &'static str) {
    if lifecycle.is_live() {
        (StatusCode::OK, LIVE_BODY)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not live\n")
    }
}

async fn reject_head() -> StatusCode {
    StatusCode::METHOD_NOT_ALLOWED
}

fn unexpected_server_exit(
    task_result: Result<Result<(), std::io::Error>, tokio::task::JoinError>,
) -> Result<(), RouterError> {
    match task_result {
        Ok(Ok(())) => Err(RouterError::Lifecycle),
        Ok(Err(source)) => Err(RouterError::Server(source)),
        Err(source) => Err(RouterError::ServerTask(source)),
    }
}

fn finish_graceful(
    task_result: Result<Result<(), std::io::Error>, tokio::task::JoinError>,
    lifecycle: &Lifecycle,
) -> Result<(), RouterError> {
    match task_result {
        Ok(Ok(())) => lifecycle.enter_stopped(),
        Ok(Err(source)) => {
            lifecycle.enter_failed()?;
            Err(RouterError::Server(source))
        }
        Err(source) => {
            lifecycle.enter_failed()?;
            Err(RouterError::ServerTask(source))
        }
    }
}

async fn abort_and_join(
    server_task: &mut JoinHandle<std::io::Result<()>>,
) -> Result<(), RouterError> {
    server_task.abort();
    match server_task.await {
        Err(join_error) if join_error.is_cancelled() => Ok(()),
        Err(join_error) => Err(RouterError::ServerTask(join_error)),
        Ok(Ok(())) => Ok(()),
        Ok(Err(source)) => Err(RouterError::Server(source)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use std::io;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use axum::body::Body;
    use axum::http::{Request, Response, StatusCode};
    use axum::routing::get;
    use tokio::net::TcpStream;
    use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot};
    use tokio::task::JoinHandle;

    use super::{BoundedTcpListener, Router, serve_http};

    const TEST_TIMEOUT: Duration = Duration::from_millis(100);

    async fn start(
        app: Router,
        capacity: usize,
        header_timeout: Duration,
    ) -> (
        SocketAddr,
        Arc<Semaphore>,
        oneshot::Sender<()>,
        JoinHandle<io::Result<()>>,
    ) {
        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind isolated listener");
        let address = tcp.local_addr().expect("read isolated listener address");
        let listener = BoundedTcpListener::new(tcp, capacity);
        let permits = listener.permit_pool();
        let (shutdown_sender, shutdown_receiver) = oneshot::channel();
        let task = tokio::spawn(serve_http(listener, app, header_timeout, shutdown_receiver));
        (address, permits, shutdown_sender, task)
    }

    async fn write_all(stream: &TcpStream, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            stream.writable().await.expect("wait for writable socket");
            match stream.try_write(bytes) {
                Ok(written) => bytes = &bytes[written..],
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("write test socket: {error}"),
            }
        }
    }

    async fn read_to_eof(stream: &TcpStream) -> Vec<u8> {
        tokio::time::timeout(Duration::from_secs(1), async {
            let mut response = Vec::new();
            loop {
                stream.readable().await.expect("wait for readable socket");
                let mut buffer = [0_u8; 512];
                match stream.try_read(&mut buffer) {
                    Ok(0) => return response,
                    Ok(read) => response.extend_from_slice(&buffer[..read]),
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("read test socket: {error}"),
                }
            }
        })
        .await
        .expect("response should complete")
    }

    async fn request(address: SocketAddr) -> Vec<u8> {
        let stream = TcpStream::connect(address).await.expect("connect client");
        write_all(
            &stream,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        read_to_eof(&stream).await
    }

    async fn wait_for_available(permits: &Semaphore, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while permits.available_permits() != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permit count should converge");
    }

    async fn assert_capacity_recovers(client_bytes: &[u8]) {
        let app = Router::new().route("/", get(|| async { "ok" }));
        let (address, permits, shutdown, server) = start(app, 1, TEST_TIMEOUT).await;
        let held = TcpStream::connect(address)
            .await
            .expect("connect held client");
        write_all(&held, client_bytes).await;
        wait_for_available(&permits, 0).await;

        let response = request(address).await;
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"ok"));

        drop(held);
        shutdown.send(()).expect("notify test shutdown");
        server
            .await
            .expect("server task should join")
            .expect("server should stop cleanly");
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn initial_header_timeout_releases_capacity() {
        assert_capacity_recovers(b"").await;
    }

    #[tokio::test]
    async fn partial_header_timeout_releases_capacity() {
        assert_capacity_recovers(b"GET / HTTP/1.1\r\nHost: localhost\r\n").await;
    }

    #[tokio::test]
    async fn idle_keep_alive_timeout_releases_capacity() {
        let app = Router::new().route("/", get(|| async { "ok" }));
        let (address, permits, shutdown, server) = start(app, 1, TEST_TIMEOUT).await;
        let held = TcpStream::connect(address)
            .await
            .expect("connect keep-alive client");
        write_all(
            &held,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n",
        )
        .await;
        let mut first_response = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !first_response.ends_with(b"ok") {
                held.readable().await.expect("wait for keep-alive response");
                let mut buffer = [0_u8; 512];
                match held.try_read(&mut buffer) {
                    Ok(0) => panic!("keep-alive closed before its response"),
                    Ok(read) => first_response.extend_from_slice(&buffer[..read]),
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("read keep-alive response: {error}"),
                }
            }
        })
        .await
        .expect("keep-alive response should complete");
        wait_for_available(&permits, 0).await;

        let response = request(address).await;
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"ok"));

        drop(held);
        shutdown.send(()).expect("notify test shutdown");
        server
            .await
            .expect("server task should join")
            .expect("server should stop cleanly");
    }

    #[tokio::test]
    async fn header_timeout_does_not_limit_an_active_handler() {
        let app = Router::new().route(
            "/",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(150)).await;
                "ok"
            }),
        );
        let (address, permits, shutdown, server) = start(app, 1, TEST_TIMEOUT).await;
        let started = Instant::now();
        let response = request(address).await;
        assert!(started.elapsed() >= Duration::from_millis(150));
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"ok"));

        shutdown.send(()).expect("notify test shutdown");
        server
            .await
            .expect("server task should join")
            .expect("server should stop cleanly");
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn eof_malformed_input_and_disconnect_release_capacity() {
        let app = Router::new().route("/", get(|| async { "ok" }));
        let (address, permits, shutdown, server) = start(app, 1, Duration::from_secs(1)).await;

        let eof = TcpStream::connect(address)
            .await
            .expect("connect EOF client");
        drop(eof);
        assert!(request(address).await.starts_with(b"HTTP/1.1 200"));

        let malformed = TcpStream::connect(address)
            .await
            .expect("connect malformed client");
        write_all(&malformed, b"not-http\r\n\r\n").await;
        assert!(request(address).await.starts_with(b"HTTP/1.1 200"));

        let disconnected = TcpStream::connect(address)
            .await
            .expect("connect disconnecting client");
        write_all(&disconnected, b"GET / HTTP/1.1\r\nHost: localhost\r\n").await;
        drop(disconnected);
        assert!(request(address).await.starts_with(b"HTTP/1.1 200"));
        drop(malformed);

        shutdown.send(()).expect("notify test shutdown");
        server
            .await
            .expect("server task should join")
            .expect("server should stop cleanly");
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn graceful_shutdown_waits_for_active_handler_and_conserves_permit() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let app = Router::new().route(
            "/",
            get({
                let entered = Arc::clone(&entered);
                let release = Arc::clone(&release);
                move || {
                    let entered = Arc::clone(&entered);
                    let release = Arc::clone(&release);
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        "ok"
                    }
                }
            }),
        );
        let (address, permits, shutdown, mut server) = start(app, 1, Duration::from_secs(1)).await;
        let stream = TcpStream::connect(address).await.expect("connect client");
        write_all(
            &stream,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .expect("handler should start");
        assert_eq!(permits.available_permits(), 0);

        shutdown.send(()).expect("notify graceful shutdown");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut server)
                .await
                .is_err()
        );
        release.notify_one();
        let response = read_to_eof(&stream).await;
        assert!(response.ends_with(b"ok"));
        server
            .await
            .expect("server task should join")
            .expect("server should stop cleanly");
        assert_eq!(permits.available_permits(), 1);
    }

    async fn upgrade(
        mut request: Request<Body>,
        upgraded: Arc<Notify>,
        release: Arc<Semaphore>,
    ) -> Response<Body> {
        let pending = hyper::upgrade::on(&mut request);
        tokio::spawn(async move {
            let io = pending.await.expect("complete HTTP upgrade");
            upgraded.notify_one();
            let _hold: OwnedSemaphorePermit = release
                .acquire_owned()
                .await
                .expect("upgrade release semaphore remains open");
            drop(io);
        });
        Response::builder()
            .status(StatusCode::SWITCHING_PROTOCOLS)
            .header("connection", "upgrade")
            .header("upgrade", "test")
            .body(Body::empty())
            .expect("build upgrade response")
    }

    #[tokio::test]
    async fn upgraded_transport_owns_permit_after_http_task_finishes() {
        let upgraded = Arc::new(Notify::new());
        let release = Arc::new(Semaphore::new(0));
        let app = Router::new().route(
            "/",
            get({
                let upgraded = Arc::clone(&upgraded);
                let release = Arc::clone(&release);
                move |request| upgrade(request, Arc::clone(&upgraded), Arc::clone(&release))
            }),
        );
        let (address, permits, shutdown, server) = start(app, 1, Duration::from_secs(1)).await;
        let stream = TcpStream::connect(address).await.expect("connect client");
        write_all(
            &stream,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: upgrade\r\nUpgrade: test\r\n\r\n",
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), upgraded.notified())
            .await
            .expect("upgrade should complete");
        assert_eq!(permits.available_permits(), 0);

        shutdown.send(()).expect("notify graceful shutdown");
        server
            .await
            .expect("server task should join")
            .expect("server should stop cleanly");
        assert_eq!(permits.available_permits(), 0);

        release.add_permits(1);
        wait_for_available(&permits, 1).await;
        drop(stream);
    }
}
