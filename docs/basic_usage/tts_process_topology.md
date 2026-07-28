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

## Temporary Isolation Override

Use `--isolate-stage` to test or deploy a temporary override without editing
the source config:

```bash
python -m sglang_omni.cli serve \
  --model-path MODEL \
  --isolate-stage vocoder
```

The option is repeatable. Literal stage names take precedence over role aliases.
The stable `vocoder` role resolves to Ming-Omni-TTS's model-specific
`audio_decode` stage.

Isolating a stage in a dedicated process asks two independent questions, and a
model answers them with two separate declarations:

- `process_isolation_stages()` — does the boundary stay correct across an OS
  process? Being a non-TP stage is not sufficient, because some stages exchange
  state through process-local registries that a second process cannot read.
- `isolation_stage_resources()` — which GPU memory fractions should be applied
  when isolation creates another process on a GPU that is already in use? This
  is a recommendation, not a capability. A process-safe stage with no
  recommendation is still isolatable when the config already declares fractions
  or when nothing else shares its GPU; otherwise placement validation names the
  stages whose fractions are missing.

Requesting a stage the model already runs alone is an idempotent no-op: the
declared topology and every declared fraction are returned unchanged.

## Applicability by Model

| Model | Process-safe stages | Recommended fractions | Not process-safe |
| --- | --- | --- | --- |
| Higgs-TTS | `audio_encoder`, `vocoder` | none needed; the config already declares 0.03 / 0.85 / 0.10 | `preprocessing` (not evaluated) |
| FishAudio S2-Pro | `preprocessing`, `vocoder` | `vocoder` | `tts_engine` |
| Voxtral TTS | `vocoder` | `vocoder` | `preprocessing`, `tts_generation` |
| Ming-Omni-TTS | `audio_decode` | `audio_decode` | `preprocessing`, `reference_encode`, `tts_engine` |
| MOSS-TTS Local (single-GPU) | `vocoder` | `vocoder` | `preprocessing` — publishes into a process-local `PreparedRequestQueue` the AR stage pops |
| MOSS-TTS Local (split) | none | — | placement declares GPU 0 while the codec runs on `cuda:1`, so the colocated fractions do not describe this topology |
| Qwen3-TTS | `vocoder` | `vocoder` | `preprocessing` — stores prepared requests in `_PREPROCESSING_CONTEXT` / `_PREPARED_REQUESTS`, read in-process by the AR engine builder |
| MOSS-TTS Delay | `vocoder` | `vocoder` | `preprocessing` — same process-local `PreparedRequestQueue` handoff |
| Audar-TTS | `reference_encoder`, `vocoder` | none yet — declare fractions before isolating | `preprocessing` (not evaluated) |
| Zonos2 | `speaker_encode`, `vocoder` | none yet — declare fractions before isolating | `preprocessing` (not evaluated) |

Higgs-TTS already places `vocoder` in its own process by default, so isolating
it is a no-op; `audio_encoder` is the boundary the flag actually moves.

Audar-TTS and Zonos2 are declared process-safe from stage state that is carried
entirely in `StagePayload.data`, but neither has benchmark coverage yet and
neither ships recommended fractions. Isolating one of their stages on a shared
GPU therefore fails with the missing-fraction error until the operator declares
`runtime.resources.total_gpu_memory_fraction` for every stage on that GPU.

## Resource and Performance Trade-offs

Isolation creates another OS process and usually another CUDA context. It can
improve throughput by overlapping vocoder scheduling and GPU work with
generation, but it also changes IPC and serialization paths, can increase idle
VRAM, and may duplicate process-local caches or runtime state.

When multiple processes share one GPU, all affected GPU stages must declare
compatible `runtime.resources.total_gpu_memory_fraction` values, and their total
must fit the placement limit. Supported models apply recommended fractions to
the copied config only when `--isolate-stage` is present, preserving explicitly
configured fractions. Omitting the option therefore leaves both the declared
process topology and the default placement totals unchanged.

These fractions are placement-accounting declarations, not proof of an
allocator-enforced runtime limit. A factory receives
`total_gpu_memory_fraction` only when its signature accepts that argument, and
an SGLang `mem_fraction_static` override can represent a different runtime
value. Keep runtime overrides consistent with the placement declaration.
Unsafe declared same-GPU topologies are rejected before startup.

Performance depends on the model, hardware, concurrency, request shape, and
streaming mode. Measure the target workload before making isolation persistent
in model or YAML configuration.
