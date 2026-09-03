# Deterministic inference

Deterministic inference means defining which parts of a model's output remain
stable when the same request is repeated. A fixed sampling seed alone does not
guarantee identical output across batch shapes, concurrency, devices, kernels,
or pipeline topology.

## Reproducibility levels

Document the strongest contract that has actually been validated:

| Level | Contract |
|---|---|
| Seeded sampling | The random token draws repeat under the same execution shape |
| Request-level reproducibility | A request repeats under a documented server configuration |
| Batch invariance | Output is unchanged when runtime batch composition changes |
| Byte identity | Encoded or PCM output bytes are identical under the stated conditions |

Always record the checkpoint revision, request, seed, software commit,
hardware, precision, launch configuration, concurrency, and output comparison.

## Current opt-in mode

Qwen3-TTS Base is the current pipeline with an explicit engine-wide
deterministic mode. Enable it in the pipeline configuration:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-1.7B-Base
enable_deterministic_inference: true
```

Both the 0.6B and 1.7B Base checkpoints expose the mode. The checked
[runtime batch-invariance test](https://github.com/sgl-project/sglang-omni/blob/main/tests/test_model/test_qwen3_tts_batch_invariance.py)
defaults to the 0.6B Base checkpoint and verifies that the same prompt,
reference audio, reference transcript, and seed produce byte-identical PCM at
runtime batch sizes 1 and 8. Qualify other checkpoint, hardware, precision, and
topology combinations separately before applying the same evidence to them.

## Performance tradeoff

Qwen3-TTS deterministic mode changes execution policy to remove known sources
of batch-dependent output:

- reference preprocessing is serialized;
- Talker compilation is disabled;
- vocoder decoding is serialized when needed for stable ordering;
- both the initial and follow-up vocoder CUDA Graph paths are disabled.

These changes reduce throughput, so the mode is opt-in. Do not enable it in a
recommended configuration without stating the performance cost.

## Other seeded models

Some models provide request-scoped seeds without an engine-wide deterministic
mode. Their cookbooks define the narrower contract. For example, MOSS-TTS Local
documents seeded request reproducibility, while default-mode Qwen3-TTS does not
claim batch-invariant PCM.

Do not generalize one model's result to a different checkpoint, precision,
GPU, kernel, or topology.

## Validate a claim

1. Pin every source-of-truth revision and the full launch configuration.
2. Repeat identical requests at concurrency 1 and under mixed batches.
3. Compare the artifact appropriate to the claim: tokens, decoded samples,
   PCM bytes, or encoded files.
4. Report every mismatch and failed request; do not average them away.
5. Measure throughput and latency separately so the reproducibility cost is
   visible.

## Related documentation

- [Qwen3-TTS cookbook](../../cookbook/qwen3_tts.md)
- [MOSS-TTS Local cookbook](../../cookbook/moss_tts_local.md)
- [MPS/DP weight-sharing qualification](../../basic_usage/mps_dp.md)
- [Benchmark methodology](../../benchmarks/methodology.md)
