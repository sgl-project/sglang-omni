use std::io;

/// A bounded, non-sensitive reason for entering graceful drain.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SignalKind {
    Interrupt,
    #[cfg(unix)]
    Terminate,
}

#[cfg(unix)]
/// Persistent Unix signal streams installed before listener binding.
pub(crate) struct SignalObserver {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl SignalObserver {
    pub(crate) fn install() -> io::Result<Self> {
        use tokio::signal::unix::{SignalKind as TokioSignalKind, signal};

        Ok(Self {
            interrupt: signal(TokioSignalKind::interrupt())?,
            terminate: signal(TokioSignalKind::terminate())?,
        })
    }

    /// Receives one signal while retaining both streams for the next call.
    pub(crate) async fn next(&mut self) -> io::Result<SignalKind> {
        tokio::select! {
            signal = self.interrupt.recv() => received(signal, SignalKind::Interrupt),
            signal = self.terminate.recv() => received(signal, SignalKind::Terminate),
        }
    }
}

#[cfg(unix)]
fn received(signal: Option<()>, kind: SignalKind) -> io::Result<SignalKind> {
    signal.map(|()| kind).ok_or_else(closed_signal_stream)
}

#[cfg(windows)]
/// Persistent Windows Ctrl-C stream installed before listener binding.
pub(crate) struct SignalObserver {
    interrupt: tokio::signal::windows::CtrlC,
}

#[cfg(windows)]
impl SignalObserver {
    pub(crate) fn install() -> io::Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::windows::ctrl_c()?,
        })
    }

    /// Receives the next Ctrl-C notification without replacing the stream.
    pub(crate) async fn next(&mut self) -> io::Result<SignalKind> {
        self.interrupt
            .recv()
            .await
            .map(|()| SignalKind::Interrupt)
            .ok_or_else(closed_signal_stream)
    }
}

#[cfg(not(any(unix, windows)))]
/// Fail-closed signal owner for unsupported process targets.
pub(crate) struct SignalObserver;

#[cfg(not(any(unix, windows)))]
impl SignalObserver {
    pub(crate) fn install() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "process signal observation is unsupported on this target",
        ))
    }

    pub(crate) async fn next(&mut self) -> io::Result<SignalKind> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "process signal observation is unsupported on this target",
        ))
    }
}

#[cfg(any(unix, windows))]
fn closed_signal_stream() -> io::Error {
    io::Error::new(
        io::ErrorKind::BrokenPipe,
        "process signal stream closed unexpectedly",
    )
}
