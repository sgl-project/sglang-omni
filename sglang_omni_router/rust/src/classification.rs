use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::error;

use crate::error::HttpFault;

pub(crate) struct ClassificationExecutor {
    slots: Arc<Semaphore>,
}

impl ClassificationExecutor {
    pub(crate) fn new() -> Arc<Self> {
        Self::with_slots(
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        )
    }

    fn with_slots(slots: usize) -> Arc<Self> {
        Arc::new(Self {
            slots: Arc::new(Semaphore::new(slots)),
        })
    }

    pub(crate) async fn classify<T>(
        &self,
        deadline: Instant,
        operation: impl FnOnce() -> Result<T, HttpFault> + Send + 'static,
    ) -> Result<T, HttpFault>
    where
        T: Send + 'static,
    {
        ensure_before(deadline)?;
        let slot = tokio::select! {
            biased;
            () = tokio::time::sleep_until(deadline) => return Err(HttpFault::UpstreamTimeout),
            result = Arc::clone(&self.slots).acquire_owned() => {
                result.map_err(|_| HttpFault::InternalError)?
            }
        };
        let mut task = tokio::task::spawn_blocking(move || {
            let _slot = slot;
            ensure_before(deadline)?;
            operation()
        });
        let classified = tokio::select! {
            biased;
            () = tokio::time::sleep_until(deadline) => {
                task.abort();
                return Err(HttpFault::UpstreamTimeout);
            }
            result = &mut task => result,
        }
        .map_err(|source| {
            error!(error = %source, "classification task failed");
            HttpFault::InternalError
        })?;
        ensure_before(deadline)?;
        classified
    }

    #[cfg(test)]
    pub(crate) fn for_test(slots: usize) -> Arc<Self> {
        Self::with_slots(slots)
    }

    #[cfg(test)]
    pub(crate) fn try_hold_slot(
        &self,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::TryAcquireError> {
        Arc::clone(&self.slots).try_acquire_owned()
    }

    #[cfg(test)]
    pub(crate) fn available_slots(&self) -> usize {
        self.slots.available_permits()
    }
}

fn ensure_before(deadline: Instant) -> Result<(), HttpFault> {
    if Instant::now() >= deadline {
        Err(HttpFault::UpstreamTimeout)
    } else {
        Ok(())
    }
}
