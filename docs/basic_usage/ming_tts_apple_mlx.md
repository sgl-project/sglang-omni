# Ming-TTS on Apple Silicon: native MLX implementation

This implementation targets `inclusionAI/Ming-omni-tts-16.8B-A3B`, not the
separate dense 0.5B model. Initial MLX Q4 HTTP audio generation has been manually
validated on an M4 Pro with 48 GiB memory. Torch/MPS support, broader numerical
and audio-quality qualification, and performance measurements remain pending.

## Backend and configuration

Use the existing Apple environment and select `SGLANG_USE_MLX=1`. The profile is
[`ming_omni_tts_apple_mlx.yaml`](../../examples/configs/ming_omni_tts_apple_mlx.yaml).
The four stages and speech API payloads are unchanged. Torch CPU tensors remain
at stage boundaries; model computation uses MLX, not Torch/MPS. CampPlus still
uses the existing CPU ONNX speaker encoder.

The profile allows one active generation request, TP=1, one decoder stream slot
and a 2048-token context. It disables radix/prefix reuse, chunked prefill, CUDA
graphs and overlap/lookahead execution. `torch_native` is only the CPU scheduler
bookkeeping backend, not the model's attention backend.

Ming uses a model-specific scheduler runner because the generic SGLang MLX
attention wrapper expects split Q/K/V projections and token-logit decoding.
The adapter preserves fused QKV, three-axis RoPE and continuous latent feedback,
uses SGLang's no-weight Torch stub and `ContiguousAttentionKVCache`, and releases
request-local state through normal completion, abort and failure callbacks.
It does not claim shared/paged MLX prefix-cache support.

## Weights and precision

Load official unquantized safetensors with the composite `config.json`, tokenizer
files and `campplus.onnx`. Loading is strict for each component. The AR stage
excludes AudioVAE, the unused LM head and known runtime rotary buffers; audio
stages load only the encoder or decoder they own. No checkpoint code is executed.

The profile selects `quantization: mlx_q4`: on-load, group-size-64 quantization
of backbone linear/expert layers only. `mlx_q8` is also accepted. Remove this
setting to retain checkpoint precision. Embeddings, routers, CFM/DiT, Aggregator,
stop/speaker heads and AudioVAE are not quantized. Prequantized community
checkpoints are explicitly rejected: their metadata/layout and acoustic
quantization have not been qualified.

On-load quantization still starts from the original weights. Short single-request
Q4 generation succeeded on a 48 GiB Mac, but peak load memory and sustained-load
behavior have not been measured. This configuration is not a guarantee that
memory, quality or throughput targets are met.

CFM retains FP32 solver state; backbone RoPE retains FP32 phase/rotation followed
by casting back to the input dtype. BF16 CUDA equivalence remains to be measured.
Matching random seeds across frameworks does not imply matching noise arrays;
numerical tests supply identical noise explicitly.

## Audio and lifecycle

The AudioVAE port reuses MLX-LM Qwen2 layers with explicit causal/sliding-window
masks, implements patch aggregation and posterior sampling for reference audio,
and preserves linear interpolation lookahead and ISTFT overlap/flush for decode.
It reuses the existing streaming vocoder scheduler, terminal latent patch and
CPU waveform payload format. Stream state is per decoder slot and cleared on
terminal/error/abort; waveform output is 44.1 kHz.

Reference audio loading uses TorchAudio/TorchCodec and requires compatible FFmpeg
shared libraries. The manually tested TorchCodec 0.15 environment used Homebrew
`ffmpeg@8`, with `DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@8)/lib"` set in the
server launch environment; FFmpeg 9 alone did not satisfy its library requirements.

## Validation

On a Metal-capable terminal, using the project's `.venv-apple`:

```bash
.venv-apple/bin/python -m pytest -q \
  tests/unit_test/ming_tts/test_mlx_config.py \
  tests/unit_test/ming_tts/test_mlx_loading.py \
  tests/unit_test/ming_tts/test_mlx_runtime.py \
  tests/unit_test/ming_tts/test_mlx_model_metal.py \
  tests/unit_test/ming_tts/test_mlx_audio_metal.py
```

Metal tests use tiny synthetic weights and CPU Torch references, not real-model
generation or HTTP. They skip when Metal is unavailable. Scheduler/import tests
also require the installed SGLang Apple runtime dependencies; a collection error
is not a passing test. Broader checkpoint coverage, memory, semantic/quality and
performance acceptance remain separate follow-up validation tasks.

Hardware validation reported by the developer (2026-09-22/23): the new runtime,
model and AudioVAE unit suites passed 85 tests; the existing Ming interface
regression suite passed 132 tests. These results precede the subsequent
cross-thread first-inference regression test and are not a final-head test count.
On M4 Pro / 48 GiB, official A3B weights with on-load `mlx_q4` completed text-only
non-streaming WAV, streaming PCM, and non-streaming reference-voice HTTP requests.
Saved audio was reported to sound normal, with similar reference/generated voice
identity. Real-time streaming continuity, reference streaming, cancellation
recovery, sustained memory, and BF16/CUDA numerical parity remain unqualified.
