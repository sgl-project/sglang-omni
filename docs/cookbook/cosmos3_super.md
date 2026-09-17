# Cosmos3-Super

Base Super uses the shared Cosmos3 adapters: Omni handles requests and worker ownership; native SGLang loads the model and runs inference. The specialized Super Image2Video and Text2Image checkpoints are separate work.

## Configs

Both configs pin `nvidia/Cosmos3-Super@fe77b66696d645f663b8f27e942b3b43e4629e23` and use GPUs 0–1. Run them separately on a two-GPU machine.

| Config | Use | Native settings |
| --- | --- | --- |
| [cosmos3_super_generation.yaml](../../examples/configs/cosmos3_super_generation.yaml) | Images, video, sound, actions | Two native workers; TP=1; transformer layerwise offload with 24 resident layers per stack |
| [cosmos3_super_reasoner.yaml](../../examples/configs/cosmos3_super_reasoner.yaml) | Text/image/video understanding | TP=2; 32,768-token context; memory fraction 0.6; up to 8 concurrent requests |

`runtime_gpu_ids` selects the GPUs. Stage `tp_size: 1` means one Omni owner; `factory.server_args_overrides` goes to the native runtime. Generation leaves Ulysses and CFG placement at native defaults. Request-time `guidance_scale` and `negative_prompt` still control CFG.

### Why generation offloads weights

The automatic residency setup ran out of memory on two H100 80GB GPUs. The recipe enables SGLang's existing layerwise offload: weights move from host memory to GPU as needed. `layerwise_resident_layers: {transformer: 24}` keeps 24 layers in **each** 64-layer stack (`gen_layers` and `language_model.layers`), 48 total. This trades latency for lower GPU memory use; host RAM and transfer bandwidth also matter.

Although the YAML enables FSDP, the pinned native runtime only applies it to resident components. The offloaded transformer is **not FSDP-sharded**. The failed automatic setup (on SGLang 0.5.19) selected component offload, so it does not prove that explicit resident FSDP cannot fit two GPUs. That comparison is pending.

The two-GPU campaign used this setting for 720p generation. Its reported 189-frame I2V check is longer than the saved 121-frame CI outputs; that run's memory log is not in the artifact bundle. Treat 24 as the tested recipe setting, not a memory guarantee for arbitrary requests or concurrency.

## Runtime and launch

The recorded SDK generation and Reasoner runs used native SGLang `4e9e407d3720045d59cae185c05f33649f4e544e`, `sglang-kernel==0.4.7`, and `flashinfer-python==0.6.18`. Released SGLang 0.5.19 did not register the Super
Reasoner architecture.

Export `HF_TOKEN` for the gated checkpoint. The pinned snapshot downloads on first launch; optionally set `HF_HOME` to choose the cache disk. From the repo root:

```bash
SGLANG_OMNI_STARTUP_TIMEOUT=1800 sgl-omni serve \
  --config examples/configs/cosmos3_super_reasoner.yaml \
  --host 127.0.0.1 --port 8000
```

This serves `/v1/chat/completions`. Add `--model-path /models/Cosmos3-Super` to use an existing local snapshot instead of the pinned Hub download.

**Generation HTTP has an additional native prerequisite.** The benchmark commit lacks `AsyncSchedulerClient.initialize(..., worker_failure=...)`; Omni rejects it before startup. The [media foundation #2048](https://github.com/sgl-project/sglang-omni/pull/2048) requires that API and propagation of `app.state.scheduler_failure`. Once those native lifecycle changes are installed, launch the media routes with:

```bash
SGLANG_OMNI_STARTUP_TIMEOUT=1800 sgl-omni serve \
  --config examples/configs/cosmos3_super_generation.yaml \
  --host 127.0.0.1 --port 8000
```

The SDK generation tests below bypass HTTP setup and ran on the recorded commit. They do not establish HTTP compatibility or partial-rank startup recovery. Neither config enables interleaved generation.

## Reproduce the GPU checks

Use the runtime above, plus pytest, pytest-asyncio, psutil, Pillow, NumPy and PyAV.
Run sequentially, without pytest-xdist:

```bash
export COSMOS3_SUPER_RUN_GPU=1

# Generation YAML → Omni SDK → native generator; seven modes, one shared server.
python -m pytest -s tests/test_model/test_cosmos3_super_generator_ci.py

# Reasoner YAML → HTTP server → shared benchmark scorer.
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
python -m benchmarks.dataset.prepare --dataset videomme-ci-50
python -m pytest -s tests/test_model/test_cosmos3_super_reasoner_ci.py

# Small synthetic inputs: eight generation modes (including policy), three
# Reasoner modes, plus Reasoner cancellation, stage death and fresh restart.
python -m pytest -s tests/integration/cosmos3/test_super_gpu.py
```

The generation CI uses bundled inputs (plus a custom T2I prompt) and 35 steps; T2V/I2V/sound/V2V run at 1280×720, 121 frames. It checks decoded outputs and action shapes. The smoke suite uses four steps and synthetic inputs. These checks do not measure visual quality or action accuracy against a reference.

The [saved two-H100 results](https://github.com/thekevinli/sglang-benchmark-artifacts/tree/473829277ba26b35bf2f46f3584749d907b43fd2/pr-2201-cosmos3-super-adapters/2gpu) include MMMU 30/50 and VideoMME 28/50, both with zero failed requests. The Reasoner lifecycle case passed, with resource-tracker warnings about semaphore and shared-memory cleanup. Native rank failure and partial startup remain untested.

Only two GPUs are qualified here. For a future four-GPU run, change `runtime_gpu_ids` to `[0, 1, 2, 3]` and generation `hsdp_shard_dim` to 4, or Reasoner native `tp_size` to 4. Keep generation TP=1 and Omni stage TP=1. To test resident FSDP, set `component_residency: {transformer: resident}` and remove `layerwise_resident_layers`. This is an unqualified alternative on any GPU count; measure memory before adopting it. Merely adding GPUs while keeping layerwise offload does not shard the transformer's weights on this runtime.
