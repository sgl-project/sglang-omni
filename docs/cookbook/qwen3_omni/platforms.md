# Qwen3-Omni Platform Deployment

[Back to the Qwen3-Omni quick start](../qwen3_omni.md).

For the single-H200 BF16 quick start, see [Server Configuration](../qwen3_omni.md#server-configuration). Run all launch commands from the repository root.

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Text-Only Deployment

Text-only mode runs the thinker pipeline on a single GPU. It accepts multi-modal input (text, image, audio) and produces text output only.

### Launch a Text-Only Server

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

Multimodal preprocessing (tokenization, image/video/audio feature extraction) runs serially by default. Add `--preprocessing.factory.max_concurrency 4` to run it on a thread pool when CPU preprocessing is the bottleneck under concurrent load. Threaded preprocessing changes how requests arrive at the thinker, so greedy outputs of the bf16 MoE can differ from the serial default; keep the default when comparing accuracy across runs.

Each preprocessing worker decodes a request's videos on parallel decoders that share min(16, the stage's CPUs divided by its workers) FFmpeg threads, split evenly among the request's videos. Set `--preprocessing.factory.video_decode_threads N` to choose that count. A value at or below the video reader's own threads (`TORCHCODEC_NUM_THREADS`, 8 by default) decodes each video on the reader alone.

For MMSU-style audio-input / text-output benchmarks with short requests, use the fused text-path config so the full text path stays inside one worker process:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_mmsu.yaml \
  --text-only \
  --port 8008
```

## Speech Deployment

Speech mode runs the full eight-stage pipeline on one or more GPUs. It produces both text (from the thinker) and audio (from the talker) output.

### Launch a Speech Server

Speech mode can run as a colocated one-GPU worker using the colocated config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

Use `examples/configs/qwen3_omni_colocated_h200.yaml` on single-H200 workers.

For manual multi-GPU placement, use the example script:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008
```

For the standard speech pipeline with default GPU placement, omit `--text-only`:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

## Single-GPU FP8 on H100/H20

SGLang-Omni can also serve native FP8 Qwen3-Omni checkpoints. Native FP8 uses the checkpoint quantization config when loading the thinker and talker AR stages, while keeping the [shared Omni request format](../../basic_usage/omni.md#general-usage).

For one-GPU H100/H20 colocated launch, use the FP8 colocated config:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

The config file contains the FP8 checkpoint path: `marksverdhei/Qwen3-Omni-30B-A3B-FP8`. You can still pass `--model-path` to override the config value.

The FP8 path keeps dense FP8 GEMM on SGLang `auto` and defaults native FP8 MoE to CUTLASS when supported. For Qwen3-Omni pipeline launches, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` is set as a default unless the operator has already set that environment variable. This disables SGLang's all-M DeepGEMM precompile session while keeping DeepGEMM available for dense FP8 GEMMs.

To opt back into SGLang's all-M DeepGEMM precompile behavior:

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

## Single-GPU AutoRound INT4 Thinker on H100/H20

SGLang-Omni also supports AutoRound INT4 quantized Qwen3-Omni checkpoints. AutoRound uses a 4-bit quantization scheme with group size 128, significantly reducing memory footprint compared to BF16 or FP8.

The public AutoRound checkpoint quantizes the thinker transformer layers. In speech mode, the talker and code2wav stages load as BF16 from the same checkpoint. For one-GPU H100/H20 colocated launch, use the colocated config with the AutoRound checkpoint:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --model-name qwen3-omni \
  --model-path Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --port 8008
```

AutoRound quantization provides:

- **~50% memory reduction** compared to BF16 (from ~60GB to ~30GB)
- **~25% memory reduction** compared to FP8 (from ~40GB to ~30GB)
- **Accuracy at ultra-low bit widths**: maintains high accuracy even at 2–4 bits, requiring minimal tuning effort thanks to its sign-gradient descent optimization.

## Realtime Sessions

### Realtime Speech with Server-Side Turn Detection

The speech pipeline also streams spoken responses over `/v1/realtime` with `--enable-realtime`, with server-side turn detection and barge-in. Its protocol is in {ref}`Realtime on the speech pipeline <realtime-on-the-speech-pipeline>`; the Session graph that most realtime clients use is in [Turn-Based: Qwen3-Omni](../../developer_reference/full_duplex_guide.md#turn-based-qwen3-omni).

### Qwen3-Omni persistent session stages

The session profile serves Qwen3-Omni as a turn-based realtime voice assistant on `/v1/realtime`: a VAD node ends each user turn and the model answers it with the conversation so far. Launching on H100 or H200, settings, the protocol and benchmarking are in [Full-Duplex Guide](../../developer_reference/full_duplex_guide.md#turn-based-qwen3-omni).
