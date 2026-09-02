use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::config::RoutingStrategy;

/// Generation policy state. Least-request observation, ordering, and exact
/// reservation occur while the short unit-valued guard is held.
pub(super) struct Selector {
    strategy: RoutingStrategy,
    cursor: AtomicU64,
    least_requests: Mutex<()>,
}

impl Selector {
    pub(super) fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            cursor: AtomicU64::new(0),
            least_requests: Mutex::new(()),
        }
    }

    pub(super) fn least_requests_guard(&self) -> Option<MutexGuard<'_, ()>> {
        if self.strategy != RoutingStrategy::LeastRequests {
            return None;
        }
        Some(
            self.least_requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    pub(super) const fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }

    pub(super) fn start(&self, pool_size: usize) -> usize {
        if pool_size == 0 {
            return 0;
        }
        let sequence = self.cursor.fetch_add(1, Ordering::Relaxed);
        u64::try_from(pool_size)
            .ok()
            .and_then(|size| usize::try_from(sequence % size).ok())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::Selector;
    use crate::config::RoutingStrategy;

    #[test]
    fn least_requests_always_serializes_while_round_robin_remains_lock_free() {
        let least_requests = Selector::new(RoutingStrategy::LeastRequests);
        let round_robin = Selector::new(RoutingStrategy::RoundRobin);

        assert!(least_requests.least_requests_guard().is_some());
        assert!(round_robin.least_requests_guard().is_none());
    }
}
