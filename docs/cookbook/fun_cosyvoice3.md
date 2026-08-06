# Fun-CosyVoice3

[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) is a
lightweight text-to-speech model (0.5B parameters) from the FunAudioLLM team at Alibaba.
It uses a Qwen2.5-0.5B backbone with FSQ speech tokens (vocab = 6561 + 200 special tokens),
conditioned on a CAMPPlus speaker embedding and prompt speech tokens extracted via an ONNX
speech tokenizer. It supports zero-shot voice cloning, cross-lingual synthesis, and
instruction-based style control. The model produces 24 kHz speech at a
25 Hz token frame rate through the `preprocessing → tts_engine → vocoder` pipeline and the
OpenAI-compatible `/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

Fun-CosyVoice3 depends on the `cosyvoice` package (GitHub-only, no PyPI release) and several
audio-processing libraries:

```bash
apt-get update && apt-get install -y sox
uv pip install sox onnxruntime openai-whisper
```

Clone the CosyVoice repository with its Matcha-TTS submodule and add both
packages to `PYTHONPATH`:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice
export PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
```

If CosyVoice was cloned without submodules, initialize Matcha-TTS with
`git -C /opt/CosyVoice submodule update --init --recursive` before starting the server.

The checkpoint includes ONNX models for the speech tokenizer and speaker encoder, so
`onnxruntime` is required.

Download the checkpoint (public repository, no token required):

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`. First startup can take several
minutes while the `tts_engine` captures CUDA graphs.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --port 8000
```

## Synthesizing Speech

### Zero-shot Voice Cloning

CosyVoice3 clones a voice from a short reference audio clip. `ref_audio` can be a local
path, file URL, data URL, or HTTP URL. `ref_text` (the transcript of the reference clip)
is optional but recommended for better alignment.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "SGLang-Omni makes text-to-speech fast and easy to deploy.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Cross-lingual Synthesis

CosyVoice3 supports cross-lingual voice cloning where the reference speaker speaks a
different language than the synthesis text. Omit `ref_text` to enter cross-lingual mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "今天天气真好，我们一起出去散步吧。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
  }' \
  --output output.wav
```

### Instruction-based Style Control

Pass `instructions` to guide prosody, emotion, or speaking style:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Welcome to our annual developer conference.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "instructions": "Speak in a cheerful and energetic tone, as if addressing a large audience."
  }' \
  --output output.wav
```

### Speed Control

Adjust playback speed with `speed` (default `1.0`):

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "This is spoken at one point three times normal speed.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "speed": 1.3
  }' \
  --output output.wav
```

### Streaming (Planned)

Incremental Flow + HiFT decoding is planned but is not enabled in the current implementation.
The current vocoder buffers the generated speech tokens and returns one complete waveform.
Do not rely on `stream=true` for time-to-first-audio until the streaming decoder is wired.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "response_format": "wav"
  }' \
  --output output.wav
```

The request above uses the supported buffered response path. A future streaming implementation
will use `response_format="pcm"` and emit audio before speech-token generation completes.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `ref_audio` | `null` | Reference audio for voice cloning (path / URL / data URL) |
| `ref_text` | `null` | Transcript of the reference audio. Improves cloning quality; omit for cross-lingual mode |
| `instructions` | `null` | Instruction text for style/prosody/emotion guidance |
| `speed` | `1.0` | Playback speed multiplier |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.8` | Top-p sampling |
| `top_k` | `20` | Top-k sampling |
| `repetition_penalty` | `1.1` | Repetition penalty |
| `max_new_tokens` | `2048` | Maximum number of generated speech tokens |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Reserved for the planned incremental decoder; current decode is buffered |

## Model Architecture

| Component | Detail |
|---|---|
| LLM Backbone | Qwen2.5-0.5B (24 layers, hidden=896, 14 heads, 2 KV heads GQA) |
| Speech Tokenizer | FSQ codebook (vocab=6561) + 200 special tokens, 25 Hz frame rate |
| Speaker Encoder | CAMPPlus (192-dim embedding, ONNX) |
| Flow Model | CausalMaskedDiffWithDiT (DiT depth=22, dim=1024, heads=16) |
| Vocoder | CausalHiFTGenerator (24 kHz output) |
| Sample Rate | 24000 Hz |

## Known Limitations

- **Reference audio required.** CosyVoice3 requires a reference audio clip for voice
  cloning; it does not support text-only synthesis without a speaker reference.
- **30-second limit.** Reference audio must be 30 seconds or shorter for speech token
  extraction.
- **Speaker similarity.** Providing `ref_text` (the transcript) yields better voice
  similarity than omitting it (cross-lingual mode).
- **Reference shape.** The endpoint accepts either `ref_audio` plus optional `ref_text`,
  or one item in `references`; multiple references are rejected for this checkpoint.
- **Prompt modes.** Provide either `ref_text` or `instructions` for the reference prompt,
  not both. `instructions` selects CosyVoice3 `instruct2` conditioning.
- **Reference conditioning cache.** Local files, data URLs, and byte payloads are cached
  by audio content and encoder configuration. Mutable HTTP URLs are intentionally encoded
  on every request instead of being cached by URL alone.
- **Speed control.** Non-streaming mode only: speed changes interpolate the mel
  spectrogram and are not available during streaming inference.
- **Voice conversion.** Voice conversion is outside the current zero-shot TTS scope.
- **Streaming decode.** The current implementation buffers all speech tokens before Flow + HiFT
  decoding. Incremental PCM output is planned but is not yet available.
- **cosyvoice dependency.** The `cosyvoice` package has no PyPI release and must be
  installed from GitHub. Matcha-TTS is a required submodule and must also be importable;
  only the CosyVoice Flow and HiFT paths are used by the buffered decoder.
