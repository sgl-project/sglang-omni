use axum::http::header::ORIGIN;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use tokio::net::TcpStream;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig as TungsteniteConfig;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, client_async_tls_with_config};

use crate::request_id::REQUEST_ID_HEADER;
use crate::worker_pool::{ConnectTarget, ResolvedTarget};

pub(super) type UpstreamSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub(super) struct HandshakeHeaders {
    request_id: HeaderValue,
    origin: Option<HeaderValue>,
}

impl HandshakeHeaders {
    pub(super) fn new(request_id: HeaderValue, origin: Option<HeaderValue>) -> Self {
        Self { request_id, origin }
    }

    fn apply(&self, destination: &mut HeaderMap) {
        destination.insert(REQUEST_ID_HEADER, self.request_id.clone());
        if let Some(origin) = self.origin.as_ref() {
            destination.insert(ORIGIN, origin.clone());
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConnectError {
    InvalidRequest,
    Connect,
    SetupTimeout,
    Handshake,
    Negotiation,
}

pub(super) async fn connect(
    target: &ResolvedTarget,
    path: &str,
    query: Option<&str>,
    headers: &HandshakeHeaders,
    setup_deadline: Instant,
) -> Result<UpstreamSocket, ConnectError> {
    let uri = target
        .websocket_uri(path, query)
        .ok_or(ConnectError::InvalidRequest)?;
    let mut request = uri
        .as_str()
        .into_client_request()
        .map_err(|_source| ConnectError::InvalidRequest)?;
    headers.apply(request.headers_mut());

    let connect_target = target
        .connect_target()
        .ok_or(ConnectError::InvalidRequest)?;
    let tcp = tokio::time::timeout_at(setup_deadline, async move {
        match connect_target {
            ConnectTarget::Socket(address) => TcpStream::connect(address).await,
            ConnectTarget::Host(host, port) => TcpStream::connect((host, port)).await,
        }
    })
    .await
    .map_err(|_elapsed| ConnectError::SetupTimeout)?
    .map_err(|_source| ConnectError::Connect)?;
    tcp.set_nodelay(true)
        .map_err(|_source| ConnectError::Connect)?;
    let config = TungsteniteConfig::default().max_message_size(Some(super::MAX_MESSAGE_BYTES));
    let (mut socket, response) = tokio::time::timeout_at(
        setup_deadline,
        client_async_tls_with_config(request, tcp, Some(config), None),
    )
    .await
    .map_err(|_elapsed| ConnectError::SetupTimeout)?
    .map_err(|_source| ConnectError::Handshake)?;
    if response.status() != StatusCode::SWITCHING_PROTOCOLS
        || response.headers().contains_key("sec-websocket-protocol")
        || response.headers().contains_key("sec-websocket-extensions")
    {
        let _ignored = socket.close(None).await;
        return Err(ConnectError::Negotiation);
    }
    Ok(socket)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::result_large_err)]
mod tests {
    use axum::http::header::{HOST, ORIGIN};
    use axum::http::{HeaderMap, HeaderValue};
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_hdr_async;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

    use crate::config::WebsocketConfig;
    use crate::worker_pool::ResolvedTarget;

    use super::{HandshakeHeaders, REQUEST_ID_HEADER, connect};

    #[test]
    fn typed_handshake_headers_apply_exact_allowlist() {
        let headers = HandshakeHeaders::new(
            HeaderValue::from_static("request-1"),
            Some(HeaderValue::from_static("https://Client.Example:8443")),
        );
        let mut destination = HeaderMap::new();

        headers.apply(&mut destination);

        assert_eq!(destination.len(), 2);
        assert_eq!(
            destination
                .get(ORIGIN)
                .and_then(|value| value.to_str().ok()),
            Some("https://Client.Example:8443")
        );
        assert_eq!(
            destination
                .get(REQUEST_ID_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("request-1")
        );

        let headers = HandshakeHeaders::new(HeaderValue::from_static("request-2"), None);
        let mut destination = HeaderMap::new();
        headers.apply(&mut destination);
        assert_eq!(destination.len(), 1);
        assert!(!destination.contains_key(ORIGIN));
    }

    #[tokio::test]
    async fn dns_target_preserves_authority_query_and_headers() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind pinned websocket fixture");
        let address = listener.local_addr().expect("fixture address");
        let (observed_sender, observed_receiver) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept pinned websocket");
            let mut sender = Some(observed_sender);
            let mut socket =
                accept_hdr_async(stream, move |request: &Request, response: Response| {
                    let observed = (
                        request.uri().to_string(),
                        request
                            .headers()
                            .get(HOST)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        request
                            .headers()
                            .get(REQUEST_ID_HEADER)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        request
                            .headers()
                            .get(ORIGIN)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                    );
                    if let Some(sender) = sender.take() {
                        let _sent = sender.send(observed);
                    }
                    Ok(response)
                })
                .await
                .expect("accept websocket handshake");
            let _closed = socket.close(None).await;
        });
        let target =
            ResolvedTarget::from_parts(&format!("http://localhost:{}/", address.port()), "/health")
                .expect("valid DNS hostname target");
        let headers = HandshakeHeaders::new(
            HeaderValue::from_static("request-2"),
            Some(HeaderValue::from_static("https://Client.Example:8443")),
        );

        let policy = WebsocketConfig::default();
        let setup_deadline = tokio::time::Instant::now() + policy.connect_timeout();
        let mut socket = connect(
            &target,
            "/v1/realtime",
            Some("model=%69gnored"),
            &headers,
            setup_deadline,
        )
        .await
        .expect("connect through DNS");
        let observed = observed_receiver.await.expect("observe handshake");
        assert_eq!(observed.0, "/v1/realtime?model=%69gnored");
        assert_eq!(
            observed.1.as_deref(),
            Some(format!("localhost:{}", address.port()).as_str())
        );
        assert_eq!(observed.2.as_deref(), Some("request-2"));
        assert_eq!(observed.3.as_deref(), Some("https://Client.Example:8443"));
        let _closed = socket.close(None).await;
        server.await.expect("join pinned websocket fixture");
    }

    #[tokio::test]
    async fn ipv6_literal_connects_and_preserves_authority() {
        let listener = TcpListener::bind("[::1]:0")
            .await
            .expect("bind IPv6 websocket fixture");
        let address = listener.local_addr().expect("fixture address");
        let (observed_sender, observed_receiver) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept IPv6 websocket");
            let mut sender = Some(observed_sender);
            let mut socket =
                accept_hdr_async(stream, move |request: &Request, response: Response| {
                    if let Some(sender) = sender.take() {
                        let host = request
                            .headers()
                            .get(HOST)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned);
                        let _sent = sender.send(host);
                    }
                    Ok(response)
                })
                .await
                .expect("accept IPv6 websocket handshake");
            let _closed = socket.close(None).await;
        });
        let target =
            ResolvedTarget::from_parts(&format!("http://[::1]:{}/", address.port()), "/health")
                .expect("valid IPv6 target");
        let headers = HandshakeHeaders::new(HeaderValue::from_static("request-3"), None);

        let policy = WebsocketConfig::default();
        let setup_deadline = tokio::time::Instant::now() + policy.connect_timeout();
        let mut socket = connect(&target, "/v1/realtime", None, &headers, setup_deadline)
            .await
            .expect("connect through IPv6 literal");
        let observed = observed_receiver.await.expect("observe IPv6 handshake");
        assert_eq!(
            observed.as_deref(),
            Some(format!("[::1]:{}", address.port()).as_str())
        );
        let _closed = socket.close(None).await;
        server.await.expect("join IPv6 websocket fixture");
    }
}
