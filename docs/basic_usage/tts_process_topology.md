# TTS Process Topology

`StageConfig.process` is the only source of truth for process topology. Stages
that name the same process share one OS process; stages with different names run
in different processes. There is no CLI override — a deployment that needs a
different topology declares it in YAML.

For example, a config can make vocoder isolation persistent:

```yaml
stages:
  - name: vocoder
    process: vocoder
```

Or it can keep the vocoder in a shared process:

```yaml
stages:
  - name: vocoder
    process: pipeline
```

## Changing Placement Without Editing the Model Config

A YAML config or a dotted override sets `process` on individual stages. The
following reproduces the topology the built-in Higgs-TTS config already
declares:

```bash
python -m sglang_omni.cli serve \
  --model-path bosonai/higgs-tts-3-4b \
  --stages.preprocessing.process tts_frontend \
  --stages.audio_encoder.process tts_frontend
```

```text
tts_frontend : preprocessing, audio_encoder
pipeline     : tts_engine
vocoder      : vocoder
```

Running a stage alone means giving it a process name nothing else uses.

## How a Topology Is Validated

The compiler enumerates every cross-process edge of the final topology from
`next`, `stream_to`, and `wait_for`, and applies the model's correctness
contract:

- `process_local_edges()` — which handoffs must stay inside one process because
  the payload does not carry required process-local state, or because the model
  retains an established compatibility boundary. Edges are splittable by
  default. The exception is declared per **edge**, not per stage, because grouping
  `preprocessing` with `audio_encoder` leaves their shared handoff local while
  still permitting `audio_encoder -> tts_engine` to cross processes.

The contract is checked once while compiling the config, including for edges
that tensor parallelism creates by putting a TP stage in its own process.

## Applicability by Model

| Model | Process-local edges |
| --- | --- |
| Higgs-TTS | — |
| FishAudio S2-Pro | — |
| Voxtral TTS | `preprocessing -> tts_generation` — compatibility guard preserving the previous process-split allowlist |
| Ming-Omni-TTS | `preprocessing -> reference_encode`, `reference_encode -> tts_engine` — compatibility guards preserving the previous process-split allowlist |
| MOSS-TTS Local (single-GPU) | `preprocessing -> tts_engine` — preprocessing publishes into a process-local `PreparedRequestQueue` the AR stage pops |
| MOSS-TTS Local (split) | all pipeline edges; placement declares GPU 0 while the codec runs on `cuda:1` |
| Qwen3-TTS | `preprocessing -> tts_engine` — prepared requests live in `_PREPROCESSING_CONTEXT` / `_PREPARED_REQUESTS`, read in-process by the AR engine builder |
| MOSS-TTS Delay | `preprocessing -> tts_engine` — same process-local `PreparedRequestQueue` handoff |
| Audar-TTS | — |
| Fun-CosyVoice3 | `preprocessing -> tts_engine` — prepared requests live in `_PREPROCESSING_CONTEXT` / `_PREPARED_REQUESTS`, read in-process by the AR engine builder |
| Zonos2 | — |

Higgs-TTS already groups `preprocessing` and `audio_encoder` in a
`tts_frontend` process and places `vocoder` in its own process by default.
Redeclaring either placement is a no-op; the stages can still be fully separated
or regrouped under another process name.

Audar-TTS and Zonos2 carry stage state in `StagePayload.data`, so neither
declares a process-local edge.

Voxtral carries the preprocessing output (`input_ids`, `voice`, and generation
limits) in `StagePayload.data` before `tts_generation`. Ming does the same for
its preprocessing fields, then serializes the reference encoder's `spk_emb` and
`prompt_latent` tensors with the `typed_tensor` wire codec before `tts_engine`.
The unit suite verifies those wire contracts in separate spawned workers through
the production control plane and SHM relay. The model configs still reject these
splits so this change does not expand the previously supported topology surface;
they can be enabled separately after model-level rollout validation.

## Process Replicas

`PipelineConfig.processes` gives a Process more than one instance. A replica
copies the whole Process, so members never end up in different replicas:

```yaml
processes:
  vocoder:
    num_replicas: 2
    replica_devices: [1, 2]
```

Each request picks one replica per replicated Process at admission and keeps it
for its lifetime. See
[`config.md`](../developer_reference/config.md) for the naming, device, and
memory-fraction rules.

## Resource and Performance Trade-offs

Splitting a stage out creates another OS process and usually another CUDA
context. It can improve throughput by overlapping vocoder scheduling and GPU
work with generation, but it also changes IPC and serialization paths, can
increase idle VRAM, and may duplicate process-local caches or runtime state.
Grouping stages that share a cache or a local handoff keeps that cost down,
which is what a shared process name expresses.

When multiple processes share one GPU, all affected GPU stages must declare
compatible `runtime.resources.total_gpu_memory_fraction` values, and their total
must fit the placement limit. A model may opt out of that requirement with
`require_memory_fraction_for_colocation: false` only when the sharing is declared
entirely by `StageConfig.gpu`. Final placement from `replica_devices` always
requires fractions, including with `num_replicas: 1`. Explicitly configured
fractions still count toward the placement limit. Fractions must be declared
directly on the stages; compiling a process topology does not infer or rewrite
them.

These fractions are placement-accounting declarations, not proof of an
allocator-enforced runtime limit. A factory receives
`total_gpu_memory_fraction` only when its signature accepts that argument, and
an SGLang `mem_fraction_static` override can represent a different runtime
value. Keep runtime overrides consistent with the placement declaration.
Unsafe declared same-GPU topologies are rejected before startup.

Performance depends on the model, hardware, concurrency, request shape, and
streaming mode. Measure the target workload before making a topology change
persistent in model or YAML configuration.
