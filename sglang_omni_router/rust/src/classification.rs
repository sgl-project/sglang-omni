/// Runs one finite classifier off the async runtime.
pub(crate) async fn run<T>(
    operation: impl FnOnce() -> T + Send + 'static,
) -> Result<T, tokio::task::JoinError>
where
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation).await
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    #[tokio::test]
    async fn cancelled_waiter_does_not_release_owned_resources_while_work_runs() {
        let payload_budget = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let payload = std::sync::Arc::clone(&payload_budget)
            .try_acquire_owned()
            .expect("reserve payload ownership");
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let waiter = tokio::spawn(async move {
            super::run(move || {
                let _payload = payload;
                entered_tx.send(()).expect("announce classifier entry");
                release_rx.recv().expect("release blocking classifier");
            })
            .await
        });
        entered_rx.await.expect("classifier entered");

        waiter.abort();
        assert!(
            waiter
                .await
                .expect_err("waiter is cancelled")
                .is_cancelled()
        );
        assert_eq!(payload_budget.available_permits(), 0);

        release_tx.send(()).expect("release classifier");
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while payload_budget.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking closure eventually releases payload");
        assert_eq!(payload_budget.available_permits(), 1);
    }
}
