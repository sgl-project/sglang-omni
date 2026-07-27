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

The current model configs preserve their established colocated vocoder
topologies. MOSS-TTS Local's colocated variant, Ming-Omni-TTS, FishAudio S2-Pro,
and Voxtral TTS declare compatible GPU-memory fractions for optional same-GPU
vocoder isolation. MOSS-TTS Local split mode and Qwen3-TTS do not declare a
complete same-GPU isolation contract, so the server rejects that override with
a resource-contract error.

## Resource and Performance Trade-offs

Isolation creates another OS process and usually another CUDA context. It can
improve throughput by overlapping vocoder scheduling and GPU work with
generation, but it also changes IPC and serialization paths, can increase idle
VRAM, and may duplicate process-local caches or runtime state.

When multiple processes share one GPU, all affected GPU stages must declare
compatible `runtime.resources.total_gpu_memory_fraction` values, and their total
must fit the placement limit. These values are placement-accounting limits; the
declared total is unchanged when the CLI moves a stage into another process
group. Unsafe same-GPU overrides are rejected before topology startup.

Performance depends on the model, hardware, concurrency, request shape, and
streaming mode. Measure the target workload before making isolation persistent
in model or YAML configuration.
