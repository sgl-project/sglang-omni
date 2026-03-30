# SGLang Integration Design

This page explains how `sglang-omni` integrates with SGLang for autoregressive decoding, focusing on the pieces that matter for maintenance: where the integration sits, which files are active, and how requests are mapped into SGLang batches and model execution.

This is not a full copy of the upstream SGLang server. It is a targeted integration that reuses SGLang's model runner, paged KV cache, and batching utilities inside the `sglang-omni` runtime.

## Role in the System

The SGLang integration is one implementation of the AR engine used by the `omni` runtime.

At a high level, the execution path is:

`CLI / Python entrypoint` → `ServerArgs` → `create_sglang_ar_engine()` → `OmniEngine` → `Scheduler` → `SGLang ModelWorker` → model forward

That split keeps responsibilities clean:

- `sglang_backend` adapts SGLang configuration and runtime primitives
- `engines.omni.factory` assembles those pieces into the generic `OmniEngine`
- `engines.omni.runtime.sglang_ar` bridges between `sglang-omni` scheduler abstractions and SGLang request / batch objects

If you are tracing actual behavior, start from `create_sglang_ar_engine()`

## Key Files

For the current integration, these are the files that matter most.

| File | Role |
| --- | --- |
| `sglang_omni/engines/ar/sglang_backend/args.py` | Parses CLI args, filters unsupported flags, and builds SGLang `ServerArgs` |
| `sglang_omni/engines/ar/sglang_backend/model_runner.py` | Wraps upstream SGLang `ModelRunner` and registers `sglang_omni.models` as an external model package |
| `sglang_omni/engines/ar/sglang_backend/model_worker.py` | Builds model config, initializes the runner, exposes memory pools, and runs batch forward passes |
| `sglang_omni/engines/ar/sglang_backend/scheduler/prefill.py` | Manages waiting requests and builds prefill / chunked-prefill batches |
| `sglang_omni/engines/ar/sglang_backend/scheduler/decode.py` | Manages decode-side running batches and KV-cache-driven retraction |
| `sglang_omni/engines/ar/sglang_backend/scheduler/cache.py` | Creates the tree cache used for prefix reuse and chunked prefill |
| `sglang_omni/engines/omni/factory.py` | Assembles the SGLang-backed `OmniEngine` |
| `sglang_omni/engines/omni/runtime/sglang_ar.py` | Implements the bridge between generic scheduler requests and SGLang-native request, batch, and output handling |

If you only open two files, `factory.py` and `runtime/sglang_ar.py` usually give the most accurate picture of the live path.

## `prepare_sgl_server_args()` vs `create_sglang_ar_engine()`

This is the most important distinction in the integration code.

### `prepare_sgl_server_args(argv)`

`prepare_sgl_server_args()` only prepares SGLang configuration.

It:

- adds upstream SGLang CLI flags through `ServerArgs.add_cli_args()`
- adds `sglang-omni` backend-specific flags such as `--disabled-args`
- strips disabled or unsupported arguments
- converts the parsed namespace into SGLang `ServerArgs`

It does **not**:

- load the model
- initialize CUDA workers
- create memory pools or KV cache
- assemble an `OmniEngine`

Use it when you need a validated `ServerArgs` object.

### `create_sglang_ar_engine(server_args, gpu_id=0)`

`create_sglang_ar_engine()` is the actual runtime assembly path.

It:

- creates a `ModelWorker`
- gets request-token and token-KV memory pools from SGLang
- creates the tree cache
- creates prefill and decode managers
- wraps them with `SGLangBatchPlanner`, `SGLangResourceManager`, and `SGLangIterationController`
- creates an `OmniEngine` whose model execution is backed by SGLang

Use it when you want the integrated AR engine, not just the config object.

## Configuration Mapping

The integration does not define a separate backend config schema. It reuses SGLang's `ServerArgs` and only adds a thin filtering layer.

### Disabled and unsupported arguments

`args.py` currently supports `--disabled-args`, which removes named arguments from the parsed namespace before `ServerArgs.from_cli_args()` runs.

It also hard-disables some upstream options that are not yet implemented or intentionally disabled in `sglang-omni`, including:

- `enable_mixed_chunk`
- `enable_dynamic_chunking`

That means the integration is intentionally narrower than the full upstream SGLang server surface.

### Model registration

Before the upstream `ModelRunner` is initialized, `SGLModelRunner` sets:

- `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_omni.models`

This is the hook that allows SGLang to discover `sglang-omni` custom model definitions through its normal model registry path.

## Engine Assembly

The live SGLang-backed engine is assembled in `create_sglang_ar_engine()`.

The object graph is roughly:

`ServerArgs` → `ModelWorker` → memory pools / cache → `PrefillManager` + `DecodeManager` → `SGLangBatchPlanner` → generic `Scheduler` → `SGLangModelRunner`

### Model worker

`ModelWorker` is the thin wrapper around SGLang's execution stack.

It:

- builds `ModelConfig` from `ServerArgs`
- initializes `SGLModelRunner`
- exposes the request-token pool and KV allocator
- broadcasts and sets the random seed across tensor-parallel ranks
- converts a forward pass into SGLang's `GenerationBatchResult`

### Cache and memory pools

The worker exposes two important memory structures:

- `req_to_token_pool`
- `token_to_kv_pool_allocator`

`create_tree_cache()` then creates either:

- `RadixCache`, which is the default path
- `ChunkCache`, only when chunked prefill is enabled and radix cache is disabled

Those objects are shared by prefill scheduling, decode scheduling, and request cleanup.

## Request and Batch Mapping

The key adaptation layer is `engines.omni.runtime.sglang_ar`.

The generic omni scheduler works with `SchedulerRequest`, `SchedulerOutput`, and `RequestOutput`. The SGLang bridge stores the SGLang-native request object inside `SGLangARRequestData.req` and keeps a `req_id_map` from SGLang request id to scheduler request.

### Waiting requests into prefill

`SGLangBatchPlanner.select_requests()` is the main scheduling entry point.

It:

- syncs newly waiting scheduler requests into `PrefillManager.waiting_queue`
- prunes finished or inactive request state
- computes the remaining request budget from `max_running_requests`
- first tries to build a prefill batch
- falls back to decode scheduling when no prefill batch is available

This preserves the standard continuous-batching shape:

`waiting queue` → prefill / chunked prefill → running decode batch

### Chunked prefill

`PrefillManager` keeps an explicit `chunked_req` pointer for unfinished long-prefill requests.

When chunked prefill is active, it:

- continues unfinished chunks before admitting more waiting requests
- caches unfinished state in the tree cache
- builds a new `ScheduleBatch`
- marks requests with `is_chunked`

The stream adapter in `factory.py` suppresses token emission for intermediate chunked-prefill steps, so partial prefill chunks do not surface as user-visible output.

### Decode and retraction

`DecodeManager` owns the decode-side running batch.

Its main job is to:

- check whether decode can continue with current KV-cache capacity
- retract requests when the KV cache is full
- return retracted requests back through `on_retract`, which currently requeues them into `PrefillManager`
- prepare the running batch for the next decode step

This means cache pressure is handled inside the SGLang side of the integration rather than by the generic omni scheduler.

## Model Execution

`SGLangModelRunner` replaces the generic AR model runner for this engine.

The execution path is roughly:

`ScheduleBatch` → `get_model_worker_batch()` → `ForwardBatch.init_new()` → SGLang / omni forward path → next token ids → `RequestOutput`

### Standard token generation

For ordinary text-only requests, the model runner uses the underlying SGLang `ModelWorker` path and returns one token id per scheduled request.

### Multimodal embedding injection

For requests carrying `omni_model_inputs`, `SGLangModelRunner` can bypass plain token embedding lookup and inject:

- image embeddings
- video embeddings
- audio embeddings
- deepstack visual embeddings

The runner matches placeholder token ids in `input_ids`, replaces those positions with prepared embeddings, and then calls the inner model with `input_embeds` and optional deepstack inputs.

That is the main reason this integration exists as a custom runner instead of using upstream SGLang unmodified.

### Output processing

`SGLangOutputProcessor` converts the model output into per-request `RequestOutput` objects.

`SGLangIterationController` then:

- appends emitted token ids to each SGLang request
- updates finished state through `req.check_finished()`
- caches unfinished decode state when needed
- suppresses user-visible output while a chunked-prefill request is still being completed

## Current Boundaries

This integration is functional but intentionally incomplete relative to upstream SGLang.

The important current boundaries are:

- dynamic chunking and mixed chunking are explicitly disabled
- overlap scheduling is not implemented
- the older `sglang_backend/scheduler/scheduler.py` is a partial skeleton and is not the main live path
- the current setup assumes a narrow tensor-parallel configuration in the local wrapper code
- several scheduler-side TODOs remain around abort handling, metrics, and standalone scheduler loop wiring

When documenting or extending behavior, prefer the factory-driven runtime path over the standalone scheduler skeleton unless you are specifically reviving that older design.
