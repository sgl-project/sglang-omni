# AuK migration reference

These four requests compare the native pipeline with the Python MLX implementation
at [`ca1e2f23`](https://github.com/xuchen-w/sglang-omni/commit/ca1e2f2332db9a03c31cf3a766c8bd4f3fecaf56).
`source.json` pins its source files and the Seed-TTS-Eval dataset revision.
The generator checks source and reference-audio hashes before loading models.

Use the frozen checkout's Python environment, with `mlx==0.32.3`, and point
`PYTHONPATH` at that checkout. The validated environment used PyTorch 2.13.0 and
Transformers 5.12.1. The prepared model comes from `tools/prepare_auk.py` and is
shared by both implementations. No model weights or Python server are installed
by this comparison utility.

Run from the Seed-TTS-Eval dataset directory containing `en/` and `zh/`:

```bash
export MLX_ENABLE_TF32=0
PYTHONPATH="$REFERENCE_SOURCE" "$REFERENCE_PYTHON" \
  "$OMNI_SOURCE/sglang_omni_mlx/native/ci/generate_auk_reference.py" \
  --model-path "$PREPARED_MODEL" \
  --requests "$OMNI_SOURCE/sglang_omni_mlx/native/ci/auk_reference/requests.jsonl" \
  --output-directory "$OUTPUT/reference"

"$NATIVE_RUNTIME/bin/auk_synthesize" \
  --model-path "$PREPARED_MODEL" \
  --requests "$OMNI_SOURCE/sglang_omni_mlx/native/ci/auk_reference/requests.jsonl" \
  --output-directory "$OUTPUT/native" --dtype bfloat16

for case in en-text en-clone zh-text zh-clone; do
  cmp "$OUTPUT/reference/$case.wav" "$OUTPUT/native/$case.wav"
done
```

Set `PREPARED_MODEL` to AuK, then AuK-Flash, and use fresh output directories.
Repeat the commands into separate directories to check request reproducibility.
The comparison uses identical instructions, seeds, model sampling recipes, and
24 kHz reference WAVs. Both WAV writers clip and truncate FP32 audio to PCM16;
the Python generator preserves the former pipeline's CPU tensor boundaries.
Run the two implementations sequentially on the same device and MLX version.

This is a migration reference. It is not a run of Tencent's separate MLX branch
or an MPS quality benchmark. Tensor-level DiT, VAE and conditioner comparisons
exercise numerical precision separately from the endpoint PCM16 comparison.
For other input sample rates, the former VAE frontend uses Torch resampling while
the native frontend uses SoXR; identical seeds alone do not imply identical audio.

## HTTP benchmark

`benchmark_cases.json` contains the first 20 English and 20 Chinese cases from
the dataset revision in `source.json`. Copy it into the dataset directory so
its reference paths resolve, then run the same client against each server:

```bash
cp "$OMNI_SOURCE/sglang_omni_mlx/native/ci/auk_reference/benchmark_cases.json" \
  "$SEED_TTS_ROOT/benchmark_cases.json"
python "$OMNI_SOURCE/sglang_omni_mlx/native/ci/benchmark_auk.py" \
  --cases "$SEED_TTS_ROOT/benchmark_cases.json" --url http://127.0.0.1:8000 \
  --backend native --variant AuK-Flash --server-pid "$SERVER_PID" \
  --output "$OUTPUT"
```

Run one server at a time. Each client invocation sends every case once at
concurrency 1. Start a fresh server and use a new output directory for each
trial; change the backend label and model variant as needed.
