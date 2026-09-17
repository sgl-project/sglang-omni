# Cosmos3 Super: shared native adapters

Base Super uses the shared Cosmos3 generation and Reasoner adapters. Native
SGLang owns checkpoint geometry, weights, preprocessing, sampling, GPU workers,
and caches; Omni owns stage placement, request conversion, and result delivery.
The specialized Super Image2Video/Text2Image checkpoints are a separate contribution.

This builds on [#2050](https://github.com/sgl-project/sglang-omni/pull/2050),
[#2048](https://github.com/sgl-project/sglang-omni/pull/2048), and
[#2049](https://github.com/sgl-project/sglang-omni/pull/2049).
Use their compatible native prerequisites for cancellation, saved outputs,
worker-failure-aware HTTP RPC, and startup cleanup. Omni can shut down a returned
native owner if adapter setup fails; native construction must clean up workers
it creates before returning. Partial-rank startup recovery still needs GPU testing.

## Launch

Use a local checkpoint or `nvidia/Cosmos3-Super@<revision>`. The adapters preserve
checkpoint metadata and served names. HTTP setup resolves one snapshot for both
the frontend and generation worker, preventing a moving revision from diverging.

No separate download step is required: the example configs pin
`nvidia/Cosmos3-Super@<revision>`, and at first launch the stage resolver
(`sglang_omni/utils/checkpoint.py:resolve_checkpoint`) calls `snapshot_download`
for that exact revision, exactly like standard SGLang/vLLM serving. Export
`HF_TOKEN` (the repo is gated) and, if you want the weights on a specific disk,
point `HF_HOME`/`HF_HUB_CACHE` at it before serving. Passing `--model-path` a
local snapshot dir skips the download entirely.

```bash
SGLANG_OMNI_STARTUP_TIMEOUT=1800 HF_TOKEN=... sgl-omni serve \
  --config examples/configs/cosmos3_super_generation.yaml \
  --host 127.0.0.1 --port 8000
```

An explicit prefetch is **optional** — useful only to land weights on a fast
local SSD (`hf download nvidia/Cosmos3-Super@<revision> --local-dir ...`) or to
fail fast on a missing token before GPUs are allocated. It is not a prerequisite
for serving.

For understanding, substitute `cosmos3_super_reasoner.yaml`. Run them separately;
both default to GPUs 0–1. Generation mounts native media routes; Reasoner serves
`/v1/chat/completions`. Neither configuration enables interleaved generation.

The two-GPU allocation is the tested default on this hardware; the layout is
**provisional**, not a measured memory requirement:

- `runtime_gpu_ids` reserves the native workers' GPUs; stage `tp_size: 1` means one Omni owner.
- Generation enables FSDP weight sharding (`hsdp_shard_dim: 2`) and forwards native execution options, following [Edge #2107](https://github.com/sgl-project/sglang-omni/pull/2107). It does not hard-code Ulysses or CFG parallelism.
- Reasoner uses native SRT TP=2. Native `server_args_overrides.tp_size` is distinct from Omni's stage process count.
- To scale up (e.g. TP=4 on four GPUs), widen `runtime_gpu_ids` and raise the native `tp_size`/`hsdp_shard_dim` to match.

CFG remains available through request parameters such as `guidance_scale` and
`negative_prompt`; choosing where its branches execute is a separate setting.

## SDK requests

Pass native sampling fields in `GenerateRequest(prompt={...}, stream=False)`.
Complete startup/request/shutdown examples are in the
[GPU tests](../../tests/integration/cosmos3/test_super_gpu.py).

| Task | Main native fields | Output |
| --- | --- | --- |
| T2I / T2V | `prompt`, `num_frames` (1 for an image) | Image/video |
| I2V | Add `image_path` | Video |
| V2V continuation | `video_path`, `condition_frame_indexes`, `condition_video_keep` | Video |
| Video with sound | Add `sound_duration` in seconds | Video with audio |
| Policy / inverse dynamics | `action_mode`, image/video input, domain settings | Action JSON |
| Forward dynamics | `action_mode="forward_dynamics"`, `image_path`, `action`, domain settings | Video |

Policy/inverse requests may omit the prompt and call native `generate_action()`.
Specify `num_frames` as action horizon + 1, and supply native domain settings
(`domain_name` or `domain_id`, and `raw_action_dim`). Results use
`chunk.media=[{"path": ".../action.json", "modality": "action"}]` with native
`action_generation_response()` formatting. Forward dynamics stays on `generate()`.
Active action calls must settle before cancelled output is removed because the
native action SDK has no cancellation-event argument. HTTP uses the mounted
native endpoint schemas, including their observation envelopes.

## Run on H100

Install the compatible native runtime plus pytest, pytest-asyncio, psutil, Pillow,
NumPy, and PyAV. Run cases sequentially on one node; do not use pytest-xdist.

```bash
export COSMOS3_SUPER_RUN_GPU=1
export COSMOS3_SUPER_MODEL_PATH=/models/Cosmos3-Super
export COSMOS3_SUPER_CHECKPOINT_REVISION='<checkpoint commit>'
export COSMOS3_SUPER_NATIVE_REVISION='<native commit and patch identifier>'
export COSMOS3_SUPER_GPU_IDS=0,1
python -m pytest -q tests/integration/cosmos3/test_super_gpu.py \
  --junitxml=results/cosmos3-super.xml --durations=0
```

There are **11 independent smoke cases**: generation `t2i`, `t2v`, `i2v`, `v2v`,
`sound`, `policy`, `inverse_dynamics`, `forward_dynamics`; Reasoner `text`, `image`,
`video`. Select one by its pytest node ID, for example:

```bash
python -m pytest -q 'tests/integration/cosmos3/test_super_gpu.py::test_super_generation[sound]'
```

The tests decode visual outputs, check audio presence/duration, validate finite
action arrays and dimensions, require nonempty Reasoner responses, and check
owned-process cleanup. T2I also serves after a fresh startup. Synthetic inputs
and four denoising steps establish plumbing only; quality/parity and GPU failure
recovery remain unqualified. Pytest durations include startup and cleanup and are
not inference latency benchmarks. JUnit records case outcomes and revision/GPU
metadata. Raw media and Reasoner responses remain in pytest's temporary directory.

Repeat with 1, 4, or 8 GPU IDs in separate campaigns. The tests load the example
YAMLs and adjust their allocation: generation uses FSDP with DiT layerwise offload
(the committed generation YAML keeps 24 of 128 layers resident for multi-GPU; a
single GPU keeps one layer), and Reasoner uses native TP equal to the allocation.
Memory fit is unqualified; one-GPU offload also needs sufficient host RAM.
Startup/request timeouts default to 1800/900 seconds, overridable with
`COSMOS3_SUPER_STARTUP_TIMEOUT` and `COSMOS3_SUPER_REQUEST_TIMEOUT`.
Review reports before sharing; keep private inputs and outputs on the GPU machine.
Use different JUnit filenames to retain separate campaigns.
