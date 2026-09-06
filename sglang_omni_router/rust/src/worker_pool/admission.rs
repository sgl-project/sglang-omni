use std::sync::Arc;

use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::profile::CAPACITY_CLASS_COUNT;
use super::{CapacityClass, ResolvedTarget, WorkerRecord};

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    #[error("router is draining")]
    Draining,
    #[error("router admission is full")]
    Overloaded,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum DispatchError {
    #[error("compatible workers have ambiguous default models")]
    AmbiguousModel,
    #[error("no configured profile matches the request")]
    NoEligibleProfile,
    #[error("matching workers are unavailable")]
    Unavailable,
    #[error("matching worker session capacity is full")]
    Overloaded,
    #[error("router dispatch invariant failed")]
    Internal,
}

/// Global-envelope and route-class ingress ownership, released exactly once.
pub(crate) struct AdmissionLease {
    class: CapacityClass,
    credits: usize,
    _class: OwnedSemaphorePermit,
    _envelope: EnvelopeLease,
}

pub(crate) struct EnvelopeLease {
    _global: OwnedSemaphorePermit,
}

impl AdmissionLease {
    pub(super) const fn class(&self) -> CapacityClass {
        self.class
    }
}

/// Active worker load retained through response termination.
struct WorkerLoadGuard {
    registration: Arc<WorkerRecord>,
    weight: usize,
}

impl WorkerLoadGuard {
    fn new(registration: Arc<WorkerRecord>, weight: usize) -> Self {
        registration.increment_load(weight);
        Self {
            registration,
            weight,
        }
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.registration.decrement_load(self.weight);
    }
}

/// Admission and weighted worker load retained through response termination.
pub(crate) struct RequestLease {
    _admission: AdmissionLease,
    _capacity: Option<OwnedSemaphorePermit>,
    load: WorkerLoadGuard,
}

impl RequestLease {
    pub(super) fn new(admission: AdmissionLease, registration: Arc<WorkerRecord>) -> Self {
        let weight = admission.credits;
        Self {
            _admission: admission,
            _capacity: None,
            load: WorkerLoadGuard::new(registration, weight),
        }
    }

    pub(super) fn new_session(
        admission: AdmissionLease,
        capacity: OwnedSemaphorePermit,
        registration: Arc<WorkerRecord>,
    ) -> Self {
        let weight = admission.credits;
        Self {
            _admission: admission,
            _capacity: Some(capacity),
            load: WorkerLoadGuard::new(registration, weight),
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
    classes: [Option<Arc<Semaphore>>; CAPACITY_CLASS_COUNT],
}

impl AdmissionController {
    pub(super) fn new(global: usize, limits: [Option<usize>; CAPACITY_CLASS_COUNT]) -> Self {
        Self {
            global: Arc::new(Semaphore::new(global)),
            classes: limits.map(|limit| limit.map(|value| Arc::new(Semaphore::new(value)))),
        }
    }

    pub(super) fn try_admit(
        &self,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        let envelope = self.try_admit_envelope()?;
        self.try_admit_class(envelope, class, credits)
    }

    pub(super) fn try_admit_envelope(&self) -> Result<EnvelopeLease, AdmissionError> {
        let global = Arc::clone(&self.global)
            .try_acquire_owned()
            .map_err(|error| match error {
                TryAcquireError::Closed => AdmissionError::Draining,
                TryAcquireError::NoPermits => AdmissionError::Overloaded,
            })?;
        Ok(EnvelopeLease { _global: global })
    }

    pub(super) fn try_admit_class(
        &self,
        envelope: EnvelopeLease,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        let class_semaphore = self.classes[class.index()]
            .as_ref()
            .ok_or(AdmissionError::Overloaded)?;
        let class_permit = Arc::clone(class_semaphore)
            .try_acquire_many_owned(credits)
            .map_err(|_| AdmissionError::Overloaded)?;
        let credits = usize::try_from(credits).map_err(|_| AdmissionError::Overloaded)?;
        Ok(AdmissionLease {
            class,
            credits,
            _class: class_permit,
            _envelope: envelope,
        })
    }

    pub(super) fn close(&self) {
        self.global.close();
    }

    #[cfg(test)]
    pub(super) fn available(&self) -> (usize, [Option<usize>; CAPACITY_CLASS_COUNT]) {
        let classes = std::array::from_fn(|index| {
            self.classes[index]
                .as_ref()
                .map(|semaphore| semaphore.available_permits())
        });
        (self.global.available_permits(), classes)
    }
}
