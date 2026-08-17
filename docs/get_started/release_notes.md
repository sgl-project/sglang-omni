# Release Notes

## v0.1.2

Install: `uv pip install "sglang-omni==0.1.2"`.

Highlights since v0.1.1:

- **Models and hardware**: MiniMax Music 3 and experimental Intel XPU support for Qwen3-ASR, Qwen3-TTS, and Qwen3-Omni
- **ASR**: automatic Qwen3-ASR language detection, `/v1/audio/translations`, long-audio support, and performance improvements
- **TTS and omni**: improved Qwen3-TTS playback continuity and Qwen3-Omni prefill and Code2Wav performance
- **Serving and runtime**: `sglang serve --model-type omni`, realtime barge-in, and communication and runtime fixes

## v0.1.1

First PyPI release. Install: `uv pip install "sglang-omni==0.1.1"`.

Highlights since the early 0.1 line:

- **TTS**: dots.tts, ZONOS2; TTS architecture refactor (shared engine/ref-encode/vocoder paths); OpenAI speech + uploaded voices
- **ASR**: Fun-ASR, ARK-ASR; shared transcription serving; Qwen3-ASR pre-LM encoder and multilingual support
- **Router**: multiprocess CP/DP router; full TTS route set (`/v1/audio/speech`, batch, stream, voices)
- **Runtime**: CUDA-IPC weight sync / MPS-DP; speech API surface cleanup
- **Omni**: Qwen3-Omni talker/streaming fixes and perf work

See [Installation](installation.md). New model cookbooks are listed on the docs home page.
