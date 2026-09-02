use std::sync::Arc;

use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::{ResolvedTarget, WorkerRecord};

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    #[error("router is draining")]
    Draining,
    #[error("router admission is full")]
    Overloaded,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum DispatchError {
    #[error("matching workers are unavailable")]
    Unavailable,
}

/// Global and generation-class ingress ownership, released exactly once.
pub(crate) struct AdmissionLease {
    _generation: OwnedSemaphorePermit,
    _global: OwnedSemaphorePermit,
}

/// Active worker load retained through response termination.
struct WorkerLoadGuard {
    registration: Arc<WorkerRecord>,
}

impl WorkerLoadGuard {
    fn new(registration: Arc<WorkerRecord>) -> Self {
        registration.increment_load();
        Self { registration }
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.registration.decrement_load();
    }
}

/// Admission and worker load retained through response termination.
pub(crate) struct RequestLease {
    _admission: AdmissionLease,
    load: WorkerLoadGuard,
}

impl RequestLease {
    pub(super) fn new(admission: AdmissionLease, registration: Arc<WorkerRecord>) -> Self {
        Self {
            _admission: admission,
            load: WorkerLoadGuard::new(registration),
        }
    }

    pub(crate) fn target(&self) -> &ResolvedTarget {
        &self.load.registration.target
    }

    pub(crate) fn request_immediate_probe(&self) {
        self.load.registration.immediate_probe.notify_one();
    }

    #[cfg(test)]
    pub(super) fn registration_ordinal(&self) -> usize {
        self.load.registration.registration_id.startup_ordinal()
    }
}

pub(super) struct AdmissionController {
    global: Arc<Semaphore>,
    generation: Arc<Semaphore>,
}

impl AdmissionController {
    pub(super) fn new(global: usize, generation: usize) -> Self {
        Self {
            global: Arc::new(Semaphore::new(global)),
            generation: Arc::new(Semaphore::new(generation)),
        }
    }

    pub(super) fn try_admit(&self) -> Result<AdmissionLease, AdmissionError> {
        let global = Arc::clone(&self.global)
            .try_acquire_owned()
            .map_err(|error| match error {
                TryAcquireError::Closed => AdmissionError::Draining,
                TryAcquireError::NoPermits => AdmissionError::Overloaded,
            })?;
        let generation = Arc::clone(&self.generation)
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Overloaded)?;
        Ok(AdmissionLease {
            _generation: generation,
            _global: global,
        })
    }

    pub(super) fn close(&self) {
        self.global.close();
    }

    #[cfg(test)]
    pub(super) fn available(&self) -> (usize, usize) {
        (
            self.global.available_permits(),
            self.generation.available_permits(),
        )
    }
}
