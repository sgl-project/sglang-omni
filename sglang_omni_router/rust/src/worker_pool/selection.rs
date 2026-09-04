use std::sync::{Mutex, MutexGuard};

use crate::config::RoutingStrategy;

use super::profile::MAX_WORKERS;

/// Selection and reservation share this lock so least-request callers cannot
/// choose from the same load observation.
pub(super) struct Selector {
    strategy: RoutingStrategy,
    recency: Mutex<RecencyOrder>,
}

impl Selector {
    pub(super) fn new(strategy: RoutingStrategy, worker_count: usize) -> Self {
        debug_assert!(worker_count <= MAX_WORKERS);
        Self {
            strategy,
            recency: Mutex::new(RecencyOrder {
                ordinals: std::array::from_fn(|ordinal| ordinal),
                worker_count,
            }),
        }
    }

    pub(super) fn lock(&self) -> SelectorGuard<'_> {
        SelectorGuard {
            recency: self
                .recency
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        }
    }

    pub(super) const fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }
}

struct RecencyOrder {
    ordinals: [usize; MAX_WORKERS],
    worker_count: usize,
}

pub(super) struct SelectorGuard<'a> {
    recency: MutexGuard<'a, RecencyOrder>,
}

impl SelectorGuard<'_> {
    pub(super) fn select(&mut self, candidates: &[bool; MAX_WORKERS]) -> Option<usize> {
        let position = self.recency.ordinals[..self.recency.worker_count]
            .iter()
            .position(|ordinal| candidates[*ordinal])?;
        let selected = self.recency.ordinals[position];

        // Moving the selected ordinal behind every other worker prevents
        // changing candidate subsets from sharing a modulo phase.
        let worker_count = self.recency.worker_count;
        self.recency.ordinals[position..worker_count].rotate_left(1);
        Some(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_WORKERS, Selector};
    use crate::config::RoutingStrategy;

    #[test]
    fn selected_worker_moves_behind_the_other_candidates() {
        let selector = Selector::new(RoutingStrategy::RoundRobin, 4);
        let mut candidates = [false; MAX_WORKERS];
        candidates[0] = true;
        candidates[2] = true;
        let mut guard = selector.lock();

        assert_eq!(guard.select(&candidates), Some(0));
        assert_eq!(guard.select(&candidates), Some(2));
        assert_eq!(guard.select(&candidates), Some(0));
    }
}
