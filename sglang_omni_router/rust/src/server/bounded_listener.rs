use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::trace;

/// TCP listener that bounds accepted client sockets.
pub(super) struct BoundedTcpListener {
    listener: TcpListener,
    permits: Arc<Semaphore>,
}

impl BoundedTcpListener {
    /// Wraps a bound TCP listener with `max_connections` permits.
    pub(super) fn new(listener: TcpListener, max_connections: usize) -> Self {
        Self {
            listener,
            permits: Arc::new(Semaphore::new(max_connections)),
        }
    }

    /// Accepts one socket after reserving its lifetime permit.
    pub(super) async fn accept(&self) -> io::Result<(ConnectionIo, SocketAddr)> {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| io::Error::other("connection permit pool closed"))?;
        let (stream, address) = self.listener.accept().await?;
        if let Err(error) = stream.set_nodelay(true) {
            trace!(%error, "failed to enable TCP_NODELAY on client connection");
        }
        Ok((ConnectionIo::new(stream, permit), address))
    }

    #[cfg(test)]
    pub(super) fn permit_pool(&self) -> Arc<Semaphore> {
        Arc::clone(&self.permits)
    }
}

/// Accepted socket whose lifetime owns one listener-capacity permit.
pub(super) struct ConnectionIo {
    stream: TcpStream,
    _permit: OwnedSemaphorePermit,
}

impl ConnectionIo {
    fn new(stream: TcpStream, permit: OwnedSemaphorePermit) -> Self {
        Self {
            stream,
            _permit: permit,
        }
    }
}

impl AsyncRead for ConnectionIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(context, buffer)
    }
}

impl AsyncWrite for ConnectionIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.stream).poll_write(context, buffer)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.stream).poll_flush(context)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.stream).poll_shutdown(context)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffers: &[io::IoSlice<'_>],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.stream).poll_write_vectored(context, buffers)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

    use std::sync::Arc;
    use std::time::Duration;

    use tokio::net::TcpStream;

    use super::BoundedTcpListener;

    async fn listener(capacity: usize) -> (BoundedTcpListener, std::net::SocketAddr) {
        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind isolated listener");
        let address = tcp.local_addr().expect("read isolated listener address");
        (BoundedTcpListener::new(tcp, capacity), address)
    }

    #[tokio::test]
    async fn bounds_accepted_wrappers_and_wakes_after_drop() {
        let (listener, address) = listener(1).await;
        let _first_client = TcpStream::connect(address)
            .await
            .expect("connect first client");
        let (first_io, _) = listener.accept().await.expect("accept first client");
        assert_eq!(listener.permits.available_permits(), 0);

        let _second_client = TcpStream::connect(address)
            .await
            .expect("connect queued client");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err(),
            "second wrapper must wait while the only permit is owned"
        );

        drop(first_io);
        let (second_io, _) = tokio::time::timeout(Duration::from_secs(1), listener.accept())
            .await
            .expect("waiting accept should wake after wrapper drop")
            .expect("accept second client");
        assert_eq!(listener.permits.available_permits(), 0);
        drop(second_io);
        assert_eq!(listener.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn accepted_sockets_enable_tcp_nodelay() {
        let (listener, address) = listener(1).await;
        let _client = TcpStream::connect(address)
            .await
            .expect("connect isolated client");

        let (connection, _) = listener.accept().await.expect("accept client");

        assert!(
            connection.stream.nodelay().expect("read TCP_NODELAY"),
            "accepted client sockets must disable Nagle's algorithm"
        );
    }

    #[tokio::test]
    async fn cancelled_accept_returns_its_pre_accept_permit() {
        let (listener, _address) = listener(1).await;

        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err(),
            "accept without a client should remain pending"
        );
        assert_eq!(listener.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn wrapper_drop_is_exact_once_and_nonblocking() {
        let (listener, address) = listener(1).await;
        let _client = TcpStream::connect(address)
            .await
            .expect("connect isolated client");
        let (io, _) = listener.accept().await.expect("accept client");
        let permits = Arc::clone(&listener.permits);

        tokio::time::timeout(Duration::from_millis(50), async move { drop(io) })
            .await
            .expect("wrapper drop must not block");
        let permit = Arc::clone(&permits)
            .try_acquire_owned()
            .expect("drop must return exactly one permit");
        assert!(Arc::clone(&permits).try_acquire_owned().is_err());
        drop(permit);
        assert_eq!(permits.available_permits(), 1);
    }
}
