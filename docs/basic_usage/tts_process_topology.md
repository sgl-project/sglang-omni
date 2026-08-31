# TTS Process Topology

`StageConfig.process` is the source of truth for process topology. It is plain
per-stage configuration: the model's config class declares the default, and a
config file or a dotted CLI flag overrides it like any other stage field.
Omitting it preserves the topology declared by the selected model or YAML
config.

A config file can make vocoder isolation persistent:

```yaml
stages:
  vocoder:
    process: vocoder
```

Or keep the vocoder in a shared process:

```yaml
stages:
  vocoder:
    process: pipeline
```

## Changing Placement at Launch

The same field is set from the command line with the dotted spelling, without
editing the source config:

```bash
# put the vocoder in its own process
python -m sglang_omni.cli serve \
  --model-path MODEL \
  --vocoder.process vocoder
```

Repeating one process name colocates stages in it. The following override
produces the topology already declared by the built-in Higgs-TTS config:

```bash
python -m sglang_omni.cli serve \
  --model-path bosonai/higgs-tts-3-4b \
  --preprocessing.process tts_frontend \
  --audio_encoder.process tts_frontend
```

```text
tts_frontend : preprocessing, audio_encoder
pipeline     : tts_engine
vocoder      : vocoder
```

Setting a stage to the process it already runs in is an idempotent no-op.
Writing one stage's `process` twice with different values is refused as a
conflict, like any other doubly-written path.

## How a Placement Is Validated

Process topology is validated by the placement and topology planners before
startup:

- Every non-TP stage must declare a `process`; TP stages derive one process
  per rank.
- A process group may span CPU stages and stages on at most one GPU.
- When multiple process groups share one GPU, every GPU stage involved must
  declare `gpu_memory_fraction`, and the per-GPU total must fit
  `placement.max_total_gpu_memory_fraction_per_gpu`. Validation names the
  stages whose fractions are missing.
- TP rank process names must not collide with other process groups.

Not every handoff tolerates a process boundary: some stages exchange state
through process-local registries a second process cannot read (for example,
MOSS-TTS pipelines hand prepared requests from preprocessing to the AR engine
through a process-local queue, and Qwen3-TTS keeps prepared requests in
process-local module state). Splitting such an edge fails at serving time
rather than at config validation — keep those stages in one process as the
shipped configs do.

Ming-Omni-TTS carries preprocessing fields in `StagePayload.data` and serializes the reference encoder's `spk_emb` and `prompt_latent` tensors with the `typed_tensor` wire codec. Both `preprocessing -> reference_encode` and `reference_encode -> tts_engine` can therefore cross process boundaries.

## Resource and Performance Trade-offs

Splitting a stage out creates another OS process and usually another CUDA
context. It can improve throughput by overlapping vocoder scheduling and GPU
work with generation, but it also changes IPC and serialization paths, can
increase idle VRAM, and may duplicate process-local caches or runtime state.
Grouping stages that share a cache or a local handoff keeps that cost down.

When multiple processes share one GPU, all affected GPU stages must declare
compatible `gpu_memory_fraction` values, and their total must fit the
placement limit. These fractions are placement-accounting declarations, not
proof of an allocator-enforced runtime limit: a factory receives
`total_gpu_memory_fraction` only when its signature accepts that argument,
and an `engine.mem_fraction_static` override can represent a different
runtime value. Keep the two consistent.

Performance depends on the model, hardware, concurrency, request shape, and
streaming mode. Measure the target workload before making isolation persistent
in model or YAML configuration.
