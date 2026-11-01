use std::borrow::Borrow;
use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::error;

use crate::error::HttpFault;
use crate::metrics::{
    ClassificationKind, ClassificationOutcome, ClassificationPhase, RouterMetrics,
};

pub(crate) struct ClassificationExecutor {
    limit: usize,
    slots: Arc<Semaphore>,
    metrics: Arc<RouterMetrics>,
}

impl ClassificationExecutor {
    pub(crate) fn new(metrics: Arc<RouterMetrics>) -> Arc<Self> {
        Self::with_slots(
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
            metrics,
        )
    }

    fn with_slots(slots: usize, metrics: Arc<RouterMetrics>) -> Arc<Self> {
        Arc::new(Self {
            limit: slots,
            slots: Arc::new(Semaphore::new(slots)),
            metrics,
        })
    }

    pub(crate) fn usage(&self) -> (usize, usize) {
        (self.limit, self.limit - self.slots.available_permits())
    }

    pub(crate) async fn classify<T>(
        &self,
        kind: ClassificationKind,
        deadline: Instant,
        operation: impl FnOnce() -> Result<T, HttpFault> + Send + 'static,
    ) -> Result<T, HttpFault>
    where
        T: Send + 'static,
    {
        let mut call = CallObservation::new(&self.metrics, kind);
        let mut slot_wait =
            PhaseObservation::new(self.metrics.as_ref(), kind, ClassificationPhase::SlotWait);
        if let Err(fault) = ensure_before(deadline) {
            call.finish(ClassificationOutcome::Timeout);
            return Err(fault);
        }
        let slot = tokio::select! {
            biased;
            () = tokio::time::sleep_until(deadline) => {
                call.finish(ClassificationOutcome::Timeout);
                return Err(HttpFault::UpstreamTimeout);
            }
            result = Arc::clone(&self.slots).acquire_owned() => {
                match result {
                    Ok(slot) => slot,
                    Err(_) => {
                        call.finish(ClassificationOutcome::Error);
                        return Err(HttpFault::InternalError);
                    }
                }
            }
        };
        slot_wait.finish();
        let executor_wait = PhaseObservation::new(
            Arc::clone(&self.metrics),
            kind,
            ClassificationPhase::ExecutorWait,
        );
        let mut task = tokio::task::spawn_blocking(move || {
            let _slot = slot;
            let mut executor_wait = executor_wait;
            executor_wait.finish();
            let _execution = PhaseObservation::new(
                executor_wait.metrics.as_ref(),
                kind,
                ClassificationPhase::Execution,
            );
            ensure_before(deadline)?;
            operation()
        });
        let classified = tokio::select! {
            biased;
            () = tokio::time::sleep_until(deadline) => {
                task.abort();
                call.finish(ClassificationOutcome::Timeout);
                return Err(HttpFault::UpstreamTimeout);
            }
            result = &mut task => result,
        }
        .map_err(|source| {
            error!(error = %source, "classification task failed");
            HttpFault::InternalError
        });
        let classified = match classified {
            Ok(classified) => complete_before(deadline, classified),
            Err(fault) => Err(fault),
        };
        let outcome = match &classified {
            Ok(_) => ClassificationOutcome::Success,
            Err(HttpFault::UpstreamTimeout) => ClassificationOutcome::Timeout,
            Err(_) => ClassificationOutcome::Error,
        };
        call.finish(outcome);
        classified
    }

    #[cfg(test)]
    pub(crate) fn for_test(slots: usize) -> Arc<Self> {
        Self::with_slots(slots, RouterMetrics::new())
    }

    #[cfg(test)]
    pub(crate) fn for_test_with_metrics(slots: usize, metrics: Arc<RouterMetrics>) -> Arc<Self> {
        Self::with_slots(slots, metrics)
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

struct PhaseObservation<M: Borrow<RouterMetrics>> {
    metrics: M,
    kind: ClassificationKind,
    phase: ClassificationPhase,
    started: Instant,
    completed: bool,
}

impl<M: Borrow<RouterMetrics>> PhaseObservation<M> {
    fn new(metrics: M, kind: ClassificationKind, phase: ClassificationPhase) -> Self {
        Self {
            metrics,
            kind,
            phase,
            started: Instant::now(),
            completed: false,
        }
    }

    fn finish(&mut self) {
        self.observe();
    }

    fn observe(&mut self) {
        if self.completed {
            return;
        }
        self.completed = true;
        self.metrics.borrow().record_classification_duration(
            self.kind,
            self.phase,
            self.started.elapsed(),
        );
    }
}

impl<M: Borrow<RouterMetrics>> Drop for PhaseObservation<M> {
    fn drop(&mut self) {
        self.observe();
    }
}

struct CallObservation<'a> {
    metrics: &'a RouterMetrics,
    kind: ClassificationKind,
    completed: bool,
}

impl<'a> CallObservation<'a> {
    fn new(metrics: &'a RouterMetrics, kind: ClassificationKind) -> Self {
        Self {
            metrics,
            kind,
            completed: false,
        }
    }

    fn finish(&mut self, outcome: ClassificationOutcome) {
        if self.completed {
            return;
        }
        self.completed = true;
        self.metrics
            .record_classification_outcome(self.kind, outcome);
    }
}

impl Drop for CallObservation<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.metrics
                .record_classification_outcome(self.kind, ClassificationOutcome::Cancelled);
        }
    }
}

fn ensure_before(deadline: Instant) -> Result<(), HttpFault> {
    if Instant::now() >= deadline {
        Err(HttpFault::UpstreamTimeout)
    } else {
        Ok(())
    }
}

fn complete_before<T>(deadline: Instant, result: Result<T, HttpFault>) -> Result<T, HttpFault> {
    ensure_before(deadline).and(result)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::{ClassificationExecutor, complete_before};
    use crate::error::HttpFault;
    use crate::metrics::{
        ClassificationKind, ClassificationOutcome, ClassificationPhase, RouterMetrics,
    };

    #[tokio::test(start_paused = true)]
    async fn elapsed_deadline_precedes_operation_error() {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(20);
        tokio::time::advance(Duration::from_millis(21)).await;

        assert_eq!(
            complete_before::<()>(deadline, Err(HttpFault::MalformedRequest)),
            Err(HttpFault::UpstreamTimeout)
        );
    }

    async fn occupy_blocking_executor(
        executor: Arc<ClassificationExecutor>,
    ) -> (
        tokio::task::JoinHandle<Result<(), HttpFault>>,
        std::sync::mpsc::SyncSender<()>,
    ) {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let running = tokio::spawn(async move {
            executor
                .classify(
                    ClassificationKind::Chat,
                    tokio::time::Instant::now() + Duration::from_secs(1),
                    move || {
                        entered_tx.send(()).expect("classification started");
                        release_rx.recv().expect("release classification");
                        Ok(())
                    },
                )
                .await
        });
        entered_rx.await.expect("blocking executor occupied");
        (running, release_tx)
    }

    #[tokio::test]
    async fn records_success_and_error_with_all_execution_phases() {
        for (result, outcome) in [
            (Ok(()), ClassificationOutcome::Success),
            (
                Err(HttpFault::MalformedRequest),
                ClassificationOutcome::Error,
            ),
        ] {
            let metrics = RouterMetrics::new();
            let executor = ClassificationExecutor::for_test_with_metrics(1, Arc::clone(&metrics));
            assert_eq!(
                executor
                    .classify(
                        ClassificationKind::Chat,
                        tokio::time::Instant::now() + Duration::from_secs(1),
                        move || result,
                    )
                    .await,
                result
            );
            assert_eq!(
                metrics.classification_outcome(ClassificationKind::Chat, outcome),
                1
            );
            for phase in ClassificationPhase::ALL {
                assert_eq!(
                    metrics
                        .classification_duration(ClassificationKind::Chat, phase)
                        .count(),
                    1
                );
            }
        }
    }

    #[tokio::test]
    async fn slot_timeout_records_wait_without_execution() {
        let metrics = RouterMetrics::new();
        let executor = ClassificationExecutor::for_test_with_metrics(1, Arc::clone(&metrics));
        let _held = executor.try_hold_slot().expect("hold classification slot");

        let result = executor
            .classify(
                ClassificationKind::Speech,
                tokio::time::Instant::now() + Duration::from_millis(20),
                || Ok(()),
            )
            .await;

        assert_eq!(result, Err(HttpFault::UpstreamTimeout));
        assert_eq!(
            metrics
                .classification_outcome(ClassificationKind::Speech, ClassificationOutcome::Timeout),
            1
        );
        assert_eq!(
            metrics
                .classification_duration(ClassificationKind::Speech, ClassificationPhase::SlotWait)
                .count(),
            1
        );
        for phase in [
            ClassificationPhase::ExecutorWait,
            ClassificationPhase::Execution,
        ] {
            assert_eq!(
                metrics
                    .classification_duration(ClassificationKind::Speech, phase)
                    .count(),
                0
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn running_timeout_records_one_terminal_outcome() {
        let metrics = RouterMetrics::new();
        let executor = ClassificationExecutor::for_test_with_metrics(1, Arc::clone(&metrics));
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let task = tokio::spawn({
            let executor = Arc::clone(&executor);
            async move {
                executor
                    .classify(
                        ClassificationKind::SpeechBatch,
                        tokio::time::Instant::now() + Duration::from_millis(20),
                        move || {
                            entered_tx.send(()).expect("classification started");
                            release_rx.recv().expect("release classification");
                            Ok(())
                        },
                    )
                    .await
            }
        });
        tokio::task::yield_now().await;
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("classification entered");
        tokio::time::advance(Duration::from_millis(21)).await;

        assert_eq!(
            task.await.expect("join classification"),
            Err(HttpFault::UpstreamTimeout)
        );
        assert_eq!(
            metrics.classification_outcome(
                ClassificationKind::SpeechBatch,
                ClassificationOutcome::Timeout
            ),
            1
        );
        tokio::time::resume();
        release_tx.send(()).expect("release classification");
        tokio::time::timeout(Duration::from_secs(1), async {
            while metrics
                .classification_duration(
                    ClassificationKind::SpeechBatch,
                    ClassificationPhase::Execution,
                )
                .count()
                == 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("classification completed");
        assert_eq!(
            metrics.classification_outcome(
                ClassificationKind::SpeechBatch,
                ClassificationOutcome::Success
            ),
            0
        );
    }

    #[tokio::test]
    async fn caller_cancellation_records_cancelled_once() {
        let metrics = RouterMetrics::new();
        let executor = ClassificationExecutor::for_test_with_metrics(1, Arc::clone(&metrics));
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let task = tokio::spawn({
            let executor = Arc::clone(&executor);
            async move {
                executor
                    .classify(
                        ClassificationKind::Translation,
                        tokio::time::Instant::now() + Duration::from_secs(1),
                        move || {
                            entered_tx.send(()).expect("classification started");
                            release_rx.recv().expect("release classification");
                            Ok(())
                        },
                    )
                    .await
            }
        });
        entered_rx.await.expect("classification entered");
        task.abort();
        assert!(
            task.await
                .expect_err("cancel classification")
                .is_cancelled()
        );
        assert_eq!(
            metrics.classification_outcome(
                ClassificationKind::Translation,
                ClassificationOutcome::Cancelled
            ),
            1
        );
        release_tx.send(()).expect("release classification");
    }

    #[test]
    fn queued_timeout_and_cancellation_retain_exact_phase_ownership() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(1)
            .build()
            .expect("build bounded blocking runtime");
        runtime.block_on(async {
            let metrics = RouterMetrics::new();
            let executor = ClassificationExecutor::for_test_with_metrics(2, Arc::clone(&metrics));
            let (running, release_tx) = occupy_blocking_executor(Arc::clone(&executor)).await;

            assert_eq!(
                executor
                    .classify(
                        ClassificationKind::Translation,
                        tokio::time::Instant::now() + Duration::from_millis(20),
                        || Ok(()),
                    )
                    .await,
                Err(HttpFault::UpstreamTimeout)
            );
            assert_eq!(
                metrics.classification_outcome(
                    ClassificationKind::Translation,
                    ClassificationOutcome::Timeout
                ),
                1
            );
            assert_eq!(
                metrics
                    .classification_duration(
                        ClassificationKind::Translation,
                        ClassificationPhase::ExecutorWait,
                    )
                    .count(),
                0
            );
            assert_eq!(
                metrics
                    .classification_duration(
                        ClassificationKind::Translation,
                        ClassificationPhase::Execution,
                    )
                    .count(),
                0
            );

            release_tx.send(()).expect("release running classification");
            assert_eq!(running.await.expect("join running classification"), Ok(()));
            tokio::time::timeout(Duration::from_secs(1), async {
                while executor.available_slots() != 2 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("timed-out queued task released its slot");
            assert_eq!(
                metrics
                    .classification_duration(
                        ClassificationKind::Translation,
                        ClassificationPhase::ExecutorWait,
                    )
                    .count(),
                1
            );

            let (running, release_tx) = occupy_blocking_executor(Arc::clone(&executor)).await;

            let cancelled = tokio::spawn({
                let executor = Arc::clone(&executor);
                async move {
                    executor
                        .classify(
                            ClassificationKind::Speech,
                            tokio::time::Instant::now() + Duration::from_secs(1),
                            || Ok(()),
                        )
                        .await
                }
            });
            tokio::time::timeout(Duration::from_secs(1), async {
                while executor.available_slots() != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("classification queued behind blocking executor");
            cancelled.abort();
            assert!(
                cancelled
                    .await
                    .expect_err("cancel queued classification")
                    .is_cancelled()
            );
            assert_eq!(
                metrics.classification_outcome(
                    ClassificationKind::Speech,
                    ClassificationOutcome::Cancelled
                ),
                1
            );

            release_tx
                .send(())
                .expect("release second running classification");
            assert_eq!(running.await.expect("join running classification"), Ok(()));
            tokio::time::timeout(Duration::from_secs(1), async {
                while executor.available_slots() != 2 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("cancelled caller's queued work completed");
            assert_eq!(
                metrics
                    .classification_duration(
                        ClassificationKind::Speech,
                        ClassificationPhase::ExecutorWait,
                    )
                    .count(),
                1
            );
            assert_eq!(
                metrics
                    .classification_duration(
                        ClassificationKind::Speech,
                        ClassificationPhase::Execution,
                    )
                    .count(),
                1
            );
            assert_eq!(
                metrics.classification_outcome(
                    ClassificationKind::Speech,
                    ClassificationOutcome::Success
                ),
                0
            );
        });
    }
}
