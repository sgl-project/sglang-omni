# CosyVoice3

[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) is a
zero-shot voice-cloning text-to-speech model from the FunAudioLLM (CosyVoice) team. It clones a
speaker from a short reference clip and its transcript, reconstructs **24 kHz** speech, and is
tuned for Chinese with multilingual coverage. In SGLang-Omni it runs as a three-stage
`preprocessing → tts_engine → vocoder` pipeline served through the OpenAI-compatible
`/v1/audio/speech` endpoint.

| Component | Spec |
|---|---|
| Architecture | `CosyVoice3ForCausalLM` / `CosyVoice3LM` (registry key `CosyVoice3ForCausalLM`) |
| Pipeline | `preprocessing → tts_engine → vocoder` |
| `preprocessing` | CosyVoice frontend: tokenizes the input text (Qwen tokenizer) and builds the zero-shot prompt from a reference wav and its transcript |
| `tts_engine` | Qwen2-0.5B autoregressive speech-token LM on the SGLang engine (driven by `OmniScheduler`) |
| `vocoder` | DiT conditional-flow-matching (`CausalMaskedDiffWithDiT`) + HiFi-GAN (`CausalHiFTGenerator`), speech tokens → 24 kHz waveform |
| Output audio | 24 kHz |
| Languages | Chinese, multilingual |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then
download the model (public, no token required):

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

The CosyVoice frontend and vocoder code ship vendored with SGLang-Omni, so no extra TTS package
is needed.

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`. The model path is baked into
`examples/configs/cosyvoice3.yaml`, so no `--model-path` flag is required. Pass
`--allowed-local-media-path` pointing at the directory that holds your reference wavs so the
server is allowed to read them:

```bash
sgl-omni serve \
  --config examples/configs/cosyvoice3.yaml \
  --host 127.0.0.1 \
  --port 8390 \
  --allowed-local-media-path /data/refs
```

First startup can take several minutes while the `tts_engine` captures CUDA graphs.

### Two-GPU deployment (split vocoder)

By default all three stages colocate on GPU 0, so the AR engine caps itself at
`mem_fraction_static=0.45` to leave headroom for the vocoder. On a second GPU the
vocoder (flow + HiFi-GAN, ~1.6 GB) can run in its own process, freeing that headroom.
`preprocessing` must stay in the `tts_engine` process (and on its GPU): it builds the
LM prompt from the engine model's embedding tables. Override the stages in your config:

```yaml
config_cls: CosyVoice3PipelineConfig
model_path: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
relay_backend: shm
stages:
  - name: preprocessing
    process: pipeline
    factory: sglang_omni.models.cosyvoice3.stages.create_preprocessing_executor
    gpu: 0
    next: tts_engine
  - name: tts_engine
    process: pipeline
    factory: sglang_omni.models.cosyvoice3.stages.create_sglang_tts_engine_executor
    factory_args: {dtype: bfloat16}
    gpu: 0
    next: vocoder
  - name: vocoder
    process: vocoder
    factory: sglang_omni.models.cosyvoice3.stages.create_vocoder_executor
    factory_args: {dtype: float32}
    gpu: 1
    terminal: true
```

Output is identical to the colocated deployment (verified byte-identical for seeded
requests on 2×RTX 3090). Tensor parallelism (`tp_size > 1`) is not supported for the
speech LM and is rejected at startup.

## Synthesizing Speech

### Zero-shot Voice Cloning

CosyVoice3 is a cloning model: every request supplies a `references` entry with `audio_path` (a
reference wav under `--allowed-local-media-path`) and `text` (the transcript of that clip). The
transcript anchors the zero-shot prompt and materially improves speaker similarity.

```bash
curl -X POST http://127.0.0.1:8390/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "今天天气不错，就该出去晒晒太阳。",
    "references": [{
      "audio_path": "/data/refs/reference.wav",
      "text": "这是参考音频的文字转录。"
    }],
    "response_format": "wav"
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and
`references[0].text`.

#### Python

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8390/v1/audio/speech",
    json={
        "model": "cosyvoice3",
        "input": "今天天气不错，就该出去晒晒太阳。",
        "references": [{
            "audio_path": "/data/refs/reference.wav",
            "text": "这是参考音频的文字转录。",
        }],
        "response_format": "wav",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | (server default) | Served model identifier, e.g. `cosyvoice3` |
| `input` | (required) | Text to synthesize |
| `references` | `null` | Reference clip for cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `response_format` | `wav` | Output container (`wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`) |
| `temperature` / `top_k` / `top_p` / `repetition_penalty` | `null` | Supported per-request sampling parameters for the autoregressive `tts_engine`; fall back to model defaults (`1.0` / `25` / `0.8` / `1.5`) when unset. Note: `repetition_penalty` is applied by both the shared runner and the native sampler, so the effective factor is its square (model default `1.5` → `2.25`); tune with that in mind. |
| `max_new_tokens` | `null` | Caps generated speech tokens; the engine also bounds generation to `text_len * 20` within the ~4k-token context |
| `seed` | `null` | Per-request seed for reproducible sampling |
| `speed` | `1.0` | Playback-rate factor (`0.25`–`4.0`); applied by the serving layer on the encoded waveform (same for all TTS models), not by the CosyVoice3 pipeline itself |

## License and Attribution

SGLang-Omni is Apache-2.0. The CosyVoice3 package vendors model and inference code from two
upstream projects — CosyVoice (Apache-2.0, Alibaba Inc.) and Matcha-TTS (MIT, Copyright (c) 2023
Shivam Mehta). License headers are preserved in-file; see
`sglang_omni/models/cosyvoice3/NOTICE` for the full attribution.

## Known Limitations

- **Reference required.** CosyVoice3 clones from a reference clip; supply both the wav and its
  transcript for the best speaker similarity.
- **Local reference access.** Reading a local reference wav requires the server to be launched
  with `--allowed-local-media-path` pointing at the directory that contains it.
- **Language coverage.** The model is tuned for Chinese with multilingual support; quality
  varies by language.
- **No text normalization.** Unlike the reference CosyVoice3 pipeline (`inference_zero_shot`,
  which runs `frontend.text_normalize()` for number/date/punctuation expansion), this adapter
  synthesizes the request text **verbatim**. Numbers, dates, currency, and abbreviations
  (e.g. `3:45`, `$1,200`) are read as-is and may differ from the reference. Normalize
  upstream of the request if you need it.
- **Single-utterance / short text.** The adapter is single-pass and does not auto-chunk long
  input. Keep the request within the engine's ~4k-token context after the reference prompt (the
  `tts_engine` also budgets generation to `text_len * 20` tokens); multi-paragraph input is not
  supported and overlong requests are rejected.
- **Reference length.** Reference audio must be at most 30 seconds.
- **Single GPU only.** The speech LM is not tensor-parallel-safe, so launching with `tp_size > 1`
  is rejected at startup with a clear error rather than producing corrupted audio.
