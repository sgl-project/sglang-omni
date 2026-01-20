# Omni Engine Design Notes

Short decisions and constraints for the OmniEngine v2 design.

## Decisions

- Streaming outputs deferred; store simple output lists in `SchedulerRequest.data` for now.
- BatchPlanner owns request selection; resource allocation happens inside `select_requests()`.
- InputPreparer and OutputProcessor receive full `SchedulerOutput` for per-request params.
- Stages are independent; a separate coordinator handles cross-stage flow.
- AR prefill/decode separation deferred; a dedicated PD model runner will handle that later.

## Near-Term TODOs

- Replace `SimpleResourceManager` with a paged KV cache manager for AR.
- Add streaming outputs with backpressure and cancellation semantics.
- Add step_id validation if/when multiple in-flight batches are enabled.
- Revisit AR fairness if decode-first hurts new request latency.
