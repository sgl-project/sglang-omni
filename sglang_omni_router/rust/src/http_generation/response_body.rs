use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Frame, SizeHint};
use thiserror::Error;

use crate::worker_pool::RequestLease;

#[derive(Debug, Error)]
#[error("upstream response body terminated")]
pub(crate) struct RelayError;

/// Direct upstream response body whose terminal owner retains admission.
pub(crate) struct DirectResponseBody {
    inner: Option<reqwest::Body>,
    lease: Option<RequestLease>,
    terminal: bool,
}

impl DirectResponseBody {
    pub(crate) fn new(inner: reqwest::Body, lease: RequestLease) -> Self {
        Self {
            inner: Some(inner),
            lease: Some(lease),
            terminal: false,
        }
    }

    fn terminalize(&mut self, upstream_failure: bool) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if upstream_failure && let Some(lease) = self.lease.as_ref() {
            lease.request_immediate_probe();
        }
        drop(self.inner.take());
        drop(self.lease.take());
    }

    fn fail(&mut self, upstream_failure: bool) -> Poll<Option<Result<Frame<Bytes>, RelayError>>> {
        self.terminalize(upstream_failure);
        Poll::Ready(Some(Err(RelayError)))
    }
}

impl Drop for DirectResponseBody {
    fn drop(&mut self) {
        self.terminalize(false);
    }
}

impl http_body::Body for DirectResponseBody {
    type Data = Bytes;
    type Error = RelayError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        let Some(inner) = self.inner.as_mut() else {
            return self.fail(true);
        };
        match Pin::new(inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(data) => Poll::Ready(Some(Ok(Frame::data(data)))),
                Err(_trailers) => self.fail(true),
            },
            Poll::Ready(Some(Err(_source))) => self.fail(true),
            Poll::Ready(None) => {
                self.terminalize(false);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.terminal
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}
