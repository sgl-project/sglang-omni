# Cosmos3 Super: shared native adapters

This integration extends the shared Cosmos3 generation and Reasoner adapters
to reproducible Super deployments. It provides candidate multi-GPU configs and
CPU contract coverage. GPU correctness, quality, memory fit, and recovery are
not yet qualified.

| Phase 4 requirement | Current boundary |
| --- | --- |
| Super geometry | Preserve checkpoint metadata and delegate loading to native SGLang; CPU contract tested |
| Transactional multi-GPU loading/recovery | Omni placement, returned-owner rollback and cleanup tested; partial native construction rollback requires the foundation's native prerequisites |
| Independent modality qualification | SDK routing and 11 separate GPU smoke cases ready; H100 execution and quality/parity results pending |

## Architecture

The generation path calls native SGLang `DiffGenerator`; the understanding path
calls native SRT `Engine` and `OpenAIServingChat`. Omni owns stage placement,
request transport, and serving integration. Native SGLang owns model geometry,
weight loading, preprocessing, parallel execution, and caches.

Super's published root config uses `Cosmos3ForConditionalGeneration`, which
already selects the shared Cosmos3 pipeline. The adapter passes its resolved
checkpoint directory to the native loader without rewriting architecture or
processor metadata. Edge's checkpoint normalization is a separate concern.

See the [family roadmap](https://github.com/sgl-project/sglang-omni/issues/1864),
[interleaving RFC](https://github.com/sgl-project/sglang-omni/issues/2183), and
[Super checkpoint metadata](https://huggingface.co/nvidia/Cosmos3-Super/blob/main/config.json).

## Prerequisites

Use the generation, media, and Reasoner foundation layers from
- [#2050](https://github.com/sgl-project/sglang-omni/pull/2050)
- [#2048](https://github.com/sgl-project/sglang-omni/pull/2048)
- [#2049](https://github.com/sgl-project/sglang-omni/pull/2049)

in that order.

Their native SGLang prerequisites are also required: Cosmos3 family loading,
SDK cancellation and saved-output propagation, worker-failure-aware HTTP RPC,
and transactional native startup/cleanup. Installing an arbitrary SGLang release
does not establish these contracts. Record the exact native revision and any
patches used in every GPU campaign.

If native construction raises before returning an Engine or DiffGenerator,
native SGLang must release its partially created workers and transports. Omni
can shut down a returned runtime when subsequent adapter construction fails,
but cannot clean resources whose ownership was never returned to it. Recovery
must create a fresh serving owner after cleanup has completed.

## Checkpoint identity

Both standalone adapters accept a local checkpoint directory or Omni's
`nvidia/Cosmos3-Super@<revision>` syntax. Resolve a full commit SHA before a
qualification run. Local directories work without a Hub download.

For HTTP generation, frontend setup resolves the checkpoint once. It stores
that snapshot in `stages.generation.factory.model_path` for the SDK worker,
along with their shared native ports. This prevents a moving Hub branch from
selecting different weights in the frontend and worker. `config.model_path`
retains the original specification; the default served name omits the Hub
revision. An explicit `served_model_name` takes precedence.

## Candidate GPU configurations

The example configs are starting points, **not measured hardware requirements**:

| Config | Native topology | Omni processes |
| --- | --- | --- |
| `examples/configs/cosmos3_super_generation.yaml` | FSDP weight sharding across four GPUs; native execution defaults | One generation owner |
| `examples/configs/cosmos3_super_reasoner.yaml` | Four-way SRT tensor parallelism | One Reasoner owner |

Like [Edge #2107](https://github.com/sgl-project/sglang-omni/pull/2107), the shared
adapter forwards placement and native options without choosing a sequence or
CFG parallelism policy. Edge's reported single-DGX-Spark configuration uses
TP=1 and transformer offload. Super's provisional generation recipe adds
four-GPU FSDP for weight memory; it requires independent H100 validation.

CFG remains available through native request parameters such as `guidance_scale`
and `negative_prompt`. CFG parallelism controls where its branches execute;
leaving that server option at the native default does not disable guidance.

Run these deployments separately; both examples reserve GPUs 0–3.
Adapt the allocation and all native parallel degrees together for the target
machine. Memory fit depends on dtype, weight residency, KV budget, resolution,
and concurrency. Sequence parallelism alone does not imply sharded weights.

Two similarly named settings have different owners:

- `stages.<name>.tp_size: 1`: Omni launches one owning stage process.
- `runtime_gpu_ids`: the complete GPU allocation reserved for native workers.
- `factory.server_args_overrides.tp_size`: native tensor-parallel degree.

Native generation validates its parallel topology. SRT placement must use one
positive GPU-index stride. Keep native generation RPC deadlines unset: a timeout
must not release request media while native work is still using it. Startup
timeout is a separate budget and can be increased for a large checkpoint.

With compatible native prerequisites installed:

```bash
SGLANG_OMNI_STARTUP_TIMEOUT=1800 sgl-omni serve \
  --config examples/configs/cosmos3_super_generation.yaml \
  --model-path /models/Cosmos3-Super --host 127.0.0.1 --port 8000
```

For understanding, use `cosmos3_super_reasoner.yaml` in the same command.
The generation deployment mounts native media routes. The Reasoner deployment
serves `/v1/chat/completions`. Neither example enables interleaving.

## SDK modality routing

The SDK accepts native sampling fields in `GenerateRequest(prompt={...}, stream=False)`.
The [GPU tests](../../tests/integration/cosmos3/test_super_gpu.py) contain complete
startup, request, validation, and shutdown examples.

| Task | Native request fields | Output |
| --- | --- | --- |
| T2I / T2V | `prompt`, `num_frames` (1 for an image) | Saved image/video |
| I2V | Add `image_path` | Saved video |
| V2V continuation | Add `video_path`, `condition_frame_indexes`, `condition_video_keep` | Saved video |
| Video with sound | Add `sound_duration` in seconds | Video with an audio stream |
| Action policy | `action_mode="policy"`, `image_path`, domain settings | Action JSON |
| Inverse dynamics | `action_mode="inverse_dynamics"`, `video_path`, domain settings | Action JSON |
| Forward dynamics | `action_mode="forward_dynamics"`, `image_path`, `action`, domain settings | Saved video |

Action policy/inverse requests call native `generate_action()`; forward dynamics
calls native `generate()`, following the distinction in Edge #2107. The SDK uses
native sampling fields directly, including `num_frames` (action horizon + 1),
`domain_name` or `domain_id`, and `raw_action_dim`. Explicit sampling values are
preserved by the native SDK. For example, the action tests use 17 frames, an AV
domain with 9 action channels, `guidance_scale=1.0`, and disable system/duration
templates. They use synthetic fixtures to exercise routing, not physical control
or trajectory quality. See NVIDIA's [action cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/README.md)
for real workload examples.

Prompt-free action requests are accepted. Policy/inverse results use the existing
`chunk.media` contract: `{"path": ".../action.json", "modality": "action"}`. The JSON
uses native `action_generation_response()`, including action values and shape.
Native action generation currently has no SDK cancellation-event argument: an
active action call must settle before cancelled output is removed. Successful
output survives delivery; failed or undelivered output is cleaned up.

HTTP requests continue through the mounted native routes and their native
schemas, including action observation envelopes. They do not use the SDK's
sampling-dictionary conversion. This contribution adds no duplicate HTTP router.

## Qualification gates

### Run the opt-in H100 tests

Install this checkout and its compatible native prerequisites on the GPU node,
including `pytest`, `pytest-asyncio`, `psutil`, Pillow, NumPy, and PyAV. Run from
the repository root. These tests are skipped unless explicitly enabled:

```bash
export COSMOS3_SUPER_RUN_GPU=1
export COSMOS3_SUPER_MODEL_PATH=/models/Cosmos3-Super
export COSMOS3_SUPER_CHECKPOINT_REVISION='REPLACE_WITH_CHECKPOINT_COMMIT'
export COSMOS3_SUPER_NATIVE_REVISION='REPLACE_WITH_NATIVE_COMMIT_AND_PATCH_ID'
export COSMOS3_SUPER_GPU_IDS=0,1,2,3
python -m pytest -q tests/integration/cosmos3/test_super_gpu.py
```

There are 11 independently selectable cases, each with its own startup, shutdown,
and report. Run sequentially on the reserved GPUs (do not use pytest-xdist).

- Generation: `t2i`, `t2v`, `i2v`, `v2v`, `sound`, `policy`, `inverse_dynamics`, `forward_dynamics`.
- Reasoner: `text`, `image`, `video`.

Select a case with, for example:

```bash
python -m pytest -q 'tests/integration/cosmos3/test_super_gpu.py::test_super_generation[sound]'
python -m pytest -q 'tests/integration/cosmos3/test_super_gpu.py::test_super_reasoner[video]'
```

Visual checks decode files and verify dimensions/frame counts. Sound additionally
requires a decodable, finite, non-silent audio stream of the requested duration.
Action checks require native JSON with the expected horizon/channel dimensions
and finite values. Reasoner checks require a nonempty response, saved locally for
inspection. Every case checks owned-process cleanup; T2I also tests serving after
a fresh startup. Four denoising steps keep these plumbing checks; they do not
establish semantic quality, task accuracy, or numerical parity. The fixtures use
`adjust_frames=False` to keep workload size fixed across GPU counts while retaining
native temporal-VAE validation.

Repeat with `COSMOS3_SUPER_GPU_IDS=0,1,2,3,4,5,6,7`, then `0,1`, then `0`.
These are independent campaigns, not automatic fallbacks after an OOM. GPU IDs
are relative to the process's visible devices. The generation test selects
FSDP with native execution defaults for multiple GPUs and transformer offload for one
GPU. A one-GPU Reasoner uses a larger static-memory fraction. None of these
recipes guarantees memory fit; the one-GPU generation path also needs ample
host RAM. Multi-node execution is outside this contribution.

`COSMOS3_SUPER_NATIVE_OVERRIDES` can point to a local JSON file with `generation`
and/or `reasoner` mappings to adjust residency or native topology. The report
records these options, so review them for private paths before sharing.
`COSMOS3_SUPER_STARTUP_TIMEOUT` defaults to 1800 seconds;
`COSMOS3_SUPER_REQUEST_TIMEOUT` defaults to 900 seconds.

Each test writes a timestamped JSON report under `results/cosmos3-super/` (or
`COSMOS3_SUPER_REPORT_DIR`). It records revisions, device information, requested native options,
timings, output sizes, and cleanup counts. Raw media remains in pytest's temporary
directory. A failure report is not benchmark evidence; logs and assertions still
need inspection. Running these cases on H100, GPU fault injection, HTTP integration,
semantic/numerical parity, and performance distributions remain separate gates.

### Acceptance checklist

1. Freeze Omni, native SGLang, checkpoint, and reference implementation revisions.
2. Verify loading on every rank and compare resolved native geometry to the
   checkpoint. Check text/image/video Reasoner parity separately from generation.
3. Qualify T2I, T2V, I2V, V2V, sound, and action independently. Compare decoded
   outputs and task-specific quality against matched native runs.
4. Inject partial-rank startup failure, startup timeout, active cancellation,
   and worker death on an isolated test deployment. Confirm no owned processes,
   ports, or request files remain before a fresh startup succeeds.
5. Measure cold startup, warm latency distributions, throughput, per-rank peak
   memory, and concurrency separately. Record exact requests, seeds, warmups,
   repetitions, topology, and hardware. Do not infer performance from one smoke
   request or from a Nano/Edge result.

Keep raw inputs and outputs on the private GPU machine. Export only the approved
aggregate results, revision manifest, and reproduction configuration. Pin hashes
for private fixtures if needed; do not commit their contents to reproduce a run.

Super Image2Video and Super Text2Image need separate checkpoint recipes and
qualification in the subsequent contribution. Their results must not inherit
base-Super acceptance merely because they use the same adapters.
