# TTS Process Topology

`StageConfig.process` is the source of truth for process topology. Omitting a
CLI placement option preserves the topology declared by the selected model or
YAML config.

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

## Temporary Placement Overrides

Two CLI options change process placement without editing the source config.

`--isolate-stage STAGE` puts one stage in a dedicated process:

```bash
python -m sglang_omni.cli serve \
  --model-path MODEL \
  --isolate-stage vocoder
```

`--stage-process STAGE=PROCESS` reads left to right: place `STAGE` in
`PROCESS`. Repeating one process name colocates stages in it. For example, the
following explicit override produces the topology already declared by the
built-in Higgs-TTS config:

```bash
python -m sglang_omni.cli serve \
  --model-path bosonai/higgs-tts-3-4b \
  --stage-process preprocessing=tts_frontend \
  --stage-process audio_encoder=tts_frontend
```

```text
tts_frontend : preprocessing, audio_encoder
pipeline     : tts_engine
vocoder      : vocoder
```

`--isolate-stage X` is shorthand for `--stage-process X=X`, so it can only ever
produce a singleton. Use `--stage-process` for grouped topologies. Naming the
same resolved stage more than once, or naming it in both options, is an error.
Both options are repeatable, and literal stage names take precedence over role
aliases — the stable `vocoder` role resolves to Ming-Omni-TTS's model-specific
`audio_decode` stage.

Requesting a stage the model already runs alone is an idempotent no-op: the
declared topology and every declared fraction are returned unchanged.

## How a Placement Request Is Validated

Moving a stage asks two independent questions, and a model answers them with two
separate declarations:

- `process_safe_edges()` — does the handoff stay correct once it crosses a
  process boundary? Declared per **edge**, not per stage, because correctness
  depends on which handoff is split rather than on which stage moved. Grouping
  `preprocessing` with `audio_encoder` leaves their shared handoff local and
  only crosses `audio_encoder -> tts_engine`, so it is checked against that edge
  alone. Being a non-TP stage is not sufficient, because some stages exchange
  state through process-local registries a second process cannot read.
- `process_edge_resources()` — which GPU memory fractions to apply when an
  override makes that edge cross processes on a GPU already in use. A
  recommendation, not a capability. An edge with no recommendation is still
  splittable when the config already declares fractions or nothing else shares
  its GPU; otherwise placement validation names the stages whose fractions are
  missing.

An unsupported handoff is reported before a missing fraction, because declaring
fractions would not make that split correct.

## Applicability by Model

| Model | Process-safe edges | Recommended fractions | Unsupported edges |
| --- | --- | --- | --- |
| Higgs-TTS | `preprocessing -> audio_encoder`, `audio_encoder -> tts_engine`, `tts_engine -> vocoder` | none needed; the config already declares 0.03 / 0.85 / 0.10 | — |
| FishAudio S2-Pro | `preprocessing -> tts_engine`, `tts_engine -> vocoder` | `tts_engine -> vocoder` | — |
| Voxtral TTS | `tts_generation -> vocoder` | `tts_generation -> vocoder` | `preprocessing -> tts_generation` |
| Ming-Omni-TTS | `tts_engine -> audio_decode` | `tts_engine -> audio_decode` | `preprocessing -> reference_encode`, `reference_encode -> tts_engine` |
| MOSS-TTS Local (single-GPU) | `tts_engine -> vocoder` | `tts_engine -> vocoder` | `preprocessing -> tts_engine` — preprocessing publishes into a process-local `PreparedRequestQueue` the AR stage pops |
| MOSS-TTS Local (split) | none | — | all; placement declares GPU 0 while the codec runs on `cuda:1`, so the colocated fractions do not describe this topology |
| Qwen3-TTS | `tts_engine -> vocoder` | `tts_engine -> vocoder` | `preprocessing -> tts_engine` — prepared requests live in `_PREPROCESSING_CONTEXT` / `_PREPARED_REQUESTS`, read in-process by the AR engine builder |
| MOSS-TTS Delay | `tts_engine -> vocoder` | `tts_engine -> vocoder` | `preprocessing -> tts_engine` — same process-local `PreparedRequestQueue` handoff |
| Audar-TTS | `preprocessing -> reference_encoder`, `reference_encoder -> tts_engine`, `tts_engine -> vocoder` | none yet — declare fractions before splitting | — |
| Fun-CosyVoice3 | `tts_engine -> vocoder` | `tts_engine -> vocoder` | `preprocessing -> tts_engine` — prepared requests live in `_PREPROCESSING_CONTEXT` / `_PREPARED_REQUESTS`, read in-process by the AR engine builder |
| Zonos2 | `preprocessing -> speaker_encode`, `speaker_encode -> tts_engine`, `tts_engine -> vocoder` | none yet — declare fractions before splitting | — |

Higgs-TTS already groups `preprocessing` and `audio_encoder` in a
`tts_frontend` process and places `vocoder` in its own process by default.
Reapplying either placement is a no-op; the options can still fully separate
the frontend stages or regroup them under another process name.

Audar-TTS and Zonos2 are declared safe from stage state carried entirely in
`StagePayload.data`, but neither has benchmark coverage yet and neither ships
recommended fractions, so a split on a shared GPU fails with the missing-fraction
error until the operator declares
`runtime.resources.total_gpu_memory_fraction` for every stage on that GPU.

## Resource and Performance Trade-offs

Splitting a stage out creates another OS process and usually another CUDA
context. It can improve throughput by overlapping vocoder scheduling and GPU
work with generation, but it also changes IPC and serialization paths, can
increase idle VRAM, and may duplicate process-local caches or runtime state.
Grouping stages that share a cache or a local handoff keeps that cost down,
which is what `--stage-process` expresses and `--isolate-stage` cannot.

When multiple processes share one GPU, all affected GPU stages must declare
compatible `runtime.resources.total_gpu_memory_fraction` values, and their total
must fit the placement limit. Supported models apply recommended fractions to
the copied config only when a placement option is present, preserving explicitly
configured fractions. An explicit value below the recommendation emits a
warning, and conflicting recommendations for one stage are rejected. Omitting
both options therefore leaves the declared process topology and the default
placement totals unchanged.

These fractions are placement-accounting declarations, not proof of an
allocator-enforced runtime limit. A factory receives
`total_gpu_memory_fraction` only when its signature accepts that argument, and
an SGLang `mem_fraction_static` override can represent a different runtime
value. Keep runtime overrides consistent with the placement declaration.
Unsafe declared same-GPU topologies are rejected before startup.

Performance depends on the model, hardware, concurrency, request shape, and
streaming mode. Measure the target workload before making isolation persistent
in model or YAML configuration.
