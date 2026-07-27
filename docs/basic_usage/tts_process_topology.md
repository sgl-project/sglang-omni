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

Only stages that the selected model marks as process-safe can be isolated.
Being a non-TP stage is not sufficient because some stages exchange state
through process-local registries. MOSS-TTS Local's single-GPU variant,
Ming-Omni-TTS, FishAudio S2-Pro, and Voxtral TTS currently support vocoder
isolation. MOSS-TTS Local split mode and Qwen3-TTS do not declare process-safe
isolation stages, so the server rejects those overrides.

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
