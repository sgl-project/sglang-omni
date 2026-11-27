# Drama 3 B200 expert configurations

These optional Triton 3.8.0 configurations target the block-FP8 Drama 3 experts on NVIDIA B200. The up and down projections have separate files; copy or select the directory containing both files. Other GPU models and Triton versions require separate tuning.

From the repository root, select the configurations before starting the server:

```bash
export SGLANG_MOE_CONFIG_DIR="$PWD/examples/configs/drama3_b200_moe"
sgl-omni serve --model-path "$DRAMA3_CHECKPOINT" \
  --vocoder.factory.codec_path "$DRAMA3_CODEC" \
  --tts_engine.engine.moe_runner_backend triton
```

For a comparison against the installed SGLang defaults, unset SGLANG_MOE_CONFIG_DIR and restart the server. Keep the checkpoint, capacity, sampling, graph settings, decoder, warmup and request corpus identical. Changes to quantizer upsampling or sampler/tower support belong to separate comparisons.

The baseline subdirectory preserves an earlier non-TMA tuning, not the current upstream defaults. Select that directory only when explicitly reproducing the earlier tuning comparison; do not label its results as the upstream baseline.

Configuration selection does not establish numerical equivalence, end-to-end speedup or audio-quality equivalence. Record the installed SGLang version and compare up/down outputs, service throughput, first-audio latency and playback stalls before choosing deployment settings.
