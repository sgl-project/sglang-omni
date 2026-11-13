---
orphan: true
---

# Qwen3-TTS Platform Deployment

[Back to the Qwen3-TTS quick start](../qwen3_tts.md).

For NVIDIA GPUs, use the [quick start](../qwen3_tts.md#prerequisites). For Ascend, first follow [NPU installation](../../other_platforms/installation_npu.md). For Intel GPUs, see the [XPU recipe](../../other_platforms/installation_xpu.md#qwen3-tts-text-to-speech-single-xpu).

## Ascend NPU baseline

All four NPU examples enable Talker graphs, asynchronous vocoder decoding, and incremental codec graphs. Custom NPU configurations use synchronous decoding by default.

Graph capture and replay have been validated on Ascend 910 with SGLang 0.5.21, PyTorch 2.10.0, and torch_npu 2.10.0. The 0.6B Base configuration has been validated with 16 concurrent requests; the other examples keep `max_running_requests: 1`.

```bash
# 0.6B Base
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b_npu.yaml \
  --port 8000
```

Use `qwen3_tts_1_7b_npu.yaml`, `qwen3_tts_0_6b_customvoice_npu.yaml`, or `qwen3_tts_1_7b_voicedesign_npu.yaml` for the other supported checkpoints.

To disable graph capture, pass `--tts_engine.engine.disable_cuda_graph true` for the Talker and `--vocoder.factory.incremental_codec_cuda_graph false` for the vocoder. Use `--vocoder.factory.async_decode false` for synchronous decoding.

The 0.6B and 1.7B files intentionally have separate `mem_fraction_static` starting values. Calibrate configurations that retain the single-request baseline on the target NPU before increasing `max_running_requests` or any vocoder batch limit.

## Apple Silicon (Native MLX)

`qwen3_tts_server` from the native MLX runtime (`sglang_omni_mlx/native`) serves the CustomVoice `/v1/audio/speech` API on an Apple Silicon Mac, from the bf16 [`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16) checkpoint, as one C++ process without CUDA, PyTorch or Python.

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime

hf download mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16 \
  --revision 6415d95f88be018ff9e46813119dc3bc12261328 --local-dir qwen3-tts-0.6b-customvoice

~/omni-runtime/bin/qwen3_tts_server --model-path qwen3-tts-0.6b-customvoice \
  --model-name Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8000
```

The [CustomVoice quick-start request](../qwen3_tts.md#synthesizing-speech) works with the native model ID substituted, with the same speakers, `language` values, sampling defaults (`max_new_tokens` 2048), `stage_params.tts_engine` overrides and prompt: the whole text is in the prompt, as the CUDA server's CustomVoice requests are. The native server differs from the CUDA server in that:

- `response_format` is `wav` or `pcm`; `stream_format` must be `audio`.
- The prompt and `max_new_tokens` must fit the talker's 8192 positions, as on CUDA: a larger `max_new_tokens` returns 400, and the default 2048 shrinks to fit. `instructions` are at most 4096 characters, and `repetition_penalty` must be above 0.
- `seed` repeats a request's audio on the same chip and MLX version; it does not reproduce CUDA's audio.
- A streamed request decodes a first chunk of `initial_codec_chunk_frames` frames (default 5, at most 25; `0` starts at 25, as on CUDA), then 25-frame chunks, each with 25 frames of left context; a whole request decodes in the CUDA vocoder's 300-frame chunks. Streamed and whole audio can differ slightly near chunk edges. `stream_codec_output` and `suppress_bootstrap_silence` are checked and then ignored.
- Started with `--model-name`, it returns 404 for any other `model`; without that flag, `model` is ignored, as on CUDA.
- Errors are `{"detail": "..."}`.
- Requests run one at a time and `--max-queued-requests` (default 2) more wait; the rest get 503, and `/health` shows the queue.
- A request not finished `--request-timeout-s` (default 600) seconds after it arrives, time in the queue included, is cancelled: it returns 504, or its stream ends early. A client that disconnects cancels its request.
- A request body over 1 MiB returns 413.

Measured on an Apple M5 MacBook (24 GB), warm: 0.46 s from launch to a healthy `/health`, 0.22 s (p50) to the first streamed audio, real-time factor 0.50 over 17 requests, and a peak memory footprint of 5.6 GB for a 5 s utterance and 13.2 GB for a 20 s one. The peak is the speech decoder's float32 upsampling over a 300-frame chunk, as in mlx-audio. Transcribed back with Qwen3-ASR-0.6B, 10 English sentences give 0% WER and 5 Chinese sentences 0% CER. With the same seed and MLX build, its audio codes are bit-identical to mlx-audio 0.5.8's Qwen3-TTS on all 22 requests compared, in both the CUDA server's prompt layout and mlx-audio's own; so is its 16-bit PCM up to 72 frames (5.8 s), beyond which the decoder's 72-frame sliding window, which the CUDA vocoder applies and mlx-audio does not, changes the samples. Mac CI checks six seeded requests' codes (English, Chinese, Japanese, Korean, the Beijing dialect speaker, and a `max_new_tokens` cap) against `sglang_omni_mlx/native/ci/model_golden/qwen3-tts-12hz-0.6b-customvoice.json`.
