# RL Admin Control

SGLang-Omni exposes a small administrative API for inference-side RL workflows.
The contract follows the SGLang and Miles control surface while preserving the
Omni pipeline boundary:

```text
Worker HTTP API -> Client -> Coordinator -> Stage -> Scheduler -> ModelWorker
```

The control plane carries only metadata and small result summaries. Tensor
payloads and bulk checkpoint data must be moved through disk, a distributed
group, or another data plane.

## Authentication

Admin endpoints are unauthenticated by default for backward compatibility. They
require `Authorization: Bearer <key>` when either of these is set:

- `admin_api_key` passed to the worker `create_app(...)`
- `SGLANG_OMNI_ADMIN_KEY` in the environment

Send administrative requests directly to the worker endpoints. The Omni Router does not expose these APIs or broadcast administrative requests.

## Worker Endpoints

The worker server supports:

- `GET|POST /model_info`
- `POST /pause_generation`
- `POST /continue_generation`
- `POST /update_weights_from_disk`
- `POST /update_weights_from_tensor`
- `POST /init_weights_update_group`
- `POST /destroy_weights_update_group`
- `POST /update_weights_from_distributed`
- `GET|POST /weights_checker`

`/update_weights_from_disk` is the primary implemented update path. It pauses
the target scheduler, optionally aborts active requests, calls the underlying
SGLang model runner update method, optionally flushes cache, and resumes unless
`keep_pause=true`. From-disk updates run on the scheduler thread. If active
requests are present, the update is rejected unless the request sets
`abort_all_requests=true` or generation was already paused with `mode=retract`.

`/init_weights_update_group` and `/destroy_weights_update_group` manage the
SGLang/Miles distributed update process group. `/update_weights_from_distributed`
then sends metadata (`names`, `dtypes`, `shapes`, `group_name`, and optional
`load_format` / `weight_version`) through the admin control plane while the
actual tensors move over the distributed group. The distributed update path uses
the same scheduler-thread lifecycle as disk updates: active requests must be
aborted or safely retracted, cache is flushed by default, and the visible
`weight_version` is updated after a successful runner update. If the distributed
update fails, the scheduler remains paused because SGLang may have partially
updated the model weights; recover by reloading or otherwise repairing the
worker before calling `continue_generation`.

`/update_weights_from_tensor` is still reserved for a future tensor data-plane
integration and returns HTTP 501 from the worker HTTP API. Once
enabled, a failed tensor update leaves the scheduler paused because tensor
loading is non-transactional; repair the worker before calling
`continue_generation`.

## Stage and TP Behavior

The Coordinator sends one admin operation to each target stage and waits for
stage results. For TP stages, rank 0 fans the operation out to follower ranks,
collects one result per rank, and returns a stage-level aggregate result with
`rank_results`.

Stages without an admin-capable scheduler return a successful skipped result so
mixed pipelines can broadcast model info or pause commands without failing on
pre/post-processing stages.

## Weight Checker

`/weights_checker` supports `snapshot`, `reset_tensors`, `compare`, and
`checksum`. The Omni checker computes strict SHA256 digests from each tensor's
name, dtype, shape, and raw bytes, then derives a per-rank checksum from the
sorted tensor digests. Full-model SHA256 checks block inference on that worker
until the digest completes.
