# Qwen3-TTS Reference

[Back to the Qwen3-TTS quick start](../qwen3_tts.md).

## Generation Parameters

These defaults describe the SGLang-Omni Python server. The [native Apple Silicon runtime](platforms.md#apple-silicon-native-mlx) has separate defaults and limitations.

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `voice` | `default` | Voice identifier. For Base reference cloning, the reference clip provides the speaker conditioning |
| `references` | `null` | Reference clip for cloning. Each item has `audio_path` and `text`, or only `speaker_embedding`: the checkpoint's speaker-encoder x-vector of the clip, 1024 values for 0.6B Base and 2048 for 1.7B Base. See [Precomputed Speaker Embedding](#precomputed-speaker-embedding) |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `language` | `auto` | Target-language hint; see [supported languages](../qwen3_tts.md#common-options) |
| `temperature` | `0.9` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | `50` | Top-k sampling |
| `repetition_penalty` | `1.05` | Repetition penalty |
| `max_new_tokens` | Base: twice the slowest natural stop for the target text, at most `2048`. CustomVoice and VoiceDesign: the checkpoint's `generation_config.json` value, `8192` | Maximum number of generated codec tokens (80 ms each). The Base default is `ceil(2 × (1.38 s + 0.39 s × target-text tokens) × 12.5)`: about 5 s for a three-token word, 26 s for 30 tokens, 2048 for long text. Set it for speech deliberately slower than natural. The server lowers any limit to what the context and the KV pool can hold |
| `seed` | `null` | Random seed for reproducibility |
| `response_format` | `wav` | Explicit values override the default in every mode. HTTP streaming accepts `pcm` or `wav` |
| `stream` | `false` | Stream raw WAV audio chunks by default; set `response_format="pcm"` for raw PCM |
| `stream_format` | `audio` | Raw audio bytes by default; explicit `sse` enables SSE events even when `stream` is omitted |
| `initial_codec_chunk_frames` | ramp `1 -> 2 -> 4` when omitted | First streaming vocoder chunk size in codec frames. An explicit value replaces the ramp's first chunk only. Smaller values lower TTFA but underrun more easily; `0` uses the steady stride from the start |
| `stream_codec_output` | `true` | Forward codec frames to the vocoder as they are generated. Set `false` to restore whole-utterance decoding for CustomVoice/VoiceDesign |
| `suppress_bootstrap_silence` | `true` | Withhold the silent bootstrap codec frame's audio from streamed CustomVoice output (validated voice/language pairs only, guarded by a runtime silence check). Set `false` to keep the leading silence |

## Precomputed Speaker Embedding

A Base reference can be given as its speaker embedding, `references[0].speaker_embedding`, instead of its clip. The server then loads and encodes no audio, and every replica serves every voice without stored voice state.

The embedding is the x-vector the checkpoint's speaker encoder computes from a reference clip: 1024 values for 0.6B Base and 2048 for 1.7B Base. An embedding from one checkpoint size does not fit the other. Sending the embedding the server computes for a clip produces the same audio as sending that clip with `x_vector_only_mode: true`, including the [x-vector leading-silence mask](#leading-silence-in-x-vector-mode).

The `qwen-tts` package computes it. Install it in its own Python environment (`pip install qwen-tts`), since it pins an older transformers release than the server. On SeedTTS EN references its embeddings match the server's to bfloat16 rounding (cosine above 0.99999 on 100 references for each size).

```python
import json

import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0", dtype=torch.bfloat16
)
(prompt,) = model.create_voice_clone_prompt(
    ref_audio="reference.wav", x_vector_only_mode=True
)
with open("speaker_embedding.json", "w") as f:
    json.dump(prompt.ref_spk_embedding.float().cpu().tolist(), f)
```

Then send the stored embedding with the text:

```python
import json

import requests

with open("speaker_embedding.json") as f:
    speaker_embedding = json.load(f)
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "input": "Get the trust fund to the bank early.",
        "references": [{"speaker_embedding": speaker_embedding}],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

The request runs in x-vector mode. A transcript is not used, and `x_vector_only_mode` may be omitted or set to `true`. The embedding reference must be the only reference, with no audio, no `vq_codes` and no `ref_audio`, and every value must be finite in the model's dtype. CustomVoice and VoiceDesign checkpoints reject it.

## Leading silence in x-vector mode

Without a reference transcript the talker starts cold, and on most requests it opens with several frames of room-tone silence before speaking, which the listener hears as extra time to first audio. The engine therefore excludes silence codec ids from the first two frames of x-vector-only requests. The silence ids are derived at startup from the checkpoint's own codec (it encodes stationary noise up to -50 dBFS) and logged once. ICL, CustomVoice, and VoiceDesign requests are not affected.

On SeedTTS EN (1088 clips, no reference text, seed 0):

| Checkpoint | Frames masked | Median onset | Onsets > 160 ms | WER | Speaker similarity |
|---|---|---|---|---|---|
| 1.7B Base | 0 | 495 ms | 95% | 0.854% | 61.00 |
| 1.7B Base | 2 (default) | 155 ms | 47% | 0.904% | 60.87 |
| 0.6B Base | 0 | 480 ms | 86% | 1.532% | 58.41 |
| 0.6B Base | 2 (default) | 75 ms | 23% | 1.072% | 58.10 |

The mask changes which codec ids the first frames can sample, so outputs differ from the unmasked model. To turn it off:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.leading_silence_mask_frames 0 \
  --port 8000
```

## CustomVoice behavior

Both released checkpoints provide `Serena`, `Vivian`, `Uncle_Fu`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`, `Eric`, and `Dylan`. Speaker matching is case-insensitive; an omitted or `default` voice selects `Vivian`. `GET /v1/audio/voices` lists `default` and the served checkpoint's speakers. Unknown speakers or supplied cloning fields return HTTP 400; uploaded reference voices are not used for CustomVoice synthesis.

Both sizes support buffered speech, batch requests, incremental HTTP audio output, and WebSocket audio output. HTTP streaming uses `stream=true` and defaults to raw WAV; explicit `response_format="pcm"` selects raw PCM. Set `stream_format="sse"` for SSE events, which default to WAV. WebSocket sessions use `stream_audio=true` with `response_format="pcm"`.

**0.6B instruction compatibility:** SGLang-Omni continues to pass optional `instructions` into the 0.6B prompt, preserving existing behavior. The released 0.6B model does not provide reliable instruction control; omit this field or use 1.7B when style control is needed.

**Eric/Dylan language behavior:** For both sizes, `language: Auto` selects Eric's Sichuan dialect token or Dylan's Beijing dialect token. An explicit language takes precedence: `language: Chinese` keeps the Chinese language token. This preserves existing SGLang-Omni behavior and differs from the QwenLM/Qwen3-TTS Python wrapper (`qwen-tts` 0.1.1), which also selects dialect tokens for `Chinese`. This is a conditioning choice, not a guarantee that the speaker's accent disappears.

## Streaming behavior

All three modes support audio streaming over HTTP and WebSocket. By default, the talker passes codec frames to the vocoder as it generates them. Set `"stream_codec_output": false` in the request or launch with `--preprocessing.factory.stream_codec_output false` to decode the whole utterance at once.

Streamed CustomVoice output on validated voice/language pairs (currently Ryan/English with default sampling) withholds the model's silent bootstrap codec frame, removing about 80 ms of leading silence from the first chunk. The frame still feeds the vocoder, so every later sample is unchanged, and a runtime silence check emits the audio unmodified whenever the first frame is not actually silent. Opt out per request with `"suppress_bootstrap_silence": false` or per deployment with `--vocoder.factory.suppress_bootstrap_silence false`.

By default the [first-audio chunk ramp](tuning.md#first-audio-chunk-ramp) starts decoding after one generated codec frame. When an utterance ends before the first chunk fills, its audio arrives in a single final flush.

## Finish reasons

Non-streaming responses include `X-Finish-Reason: stop` after codec EOS or `X-Finish-Reason: length` when generation reaches `max_new_tokens`. A `length` response still contains decodable audio, but the utterance may be incomplete. Models that do not report how generation ended send `X-Finish-Reason: unknown`. Batch responses expose the same value as each item's `finish_reason`.

## Dependency troubleshooting

Keep `--no-deps` on both Python package installation commands in the [prerequisites](../qwen3_tts.md#prerequisites).

`qwen-tts` pins Transformers 4.57.3, which would replace the project's 5.19.0. And resolving `sox` normally pulls `numpy` past the ceiling `numba==0.65.1` imposes (numba requires `numpy<=2.4`); the upgraded `numpy` then breaks `librosa`, so `import qwen_tts` fails with `Numba needs NumPy 2.4 or less` before the server can start.

Do not add `onnxruntime` to those installation commands — it is already a SGLang-Omni dependency, and resolving it pulls `numpy` the same way.

> Do **not** install `qwen-tts` with dependencies here. Its declared dependency set can pull a different Transformers/Torch stack than the SGLang-Omni runtime.

Concretely, `qwen-tts` 0.1.1 pins Transformers 4.57.3, and its model code calls APIs that Transformers 5.12 has since renamed or removed — most visibly the mask factories (`create_causal_mask` and friends), which now spell `input_embeds` as `inputs_embeds` and no longer accept `cache_position`. SGLang-Omni patches these differences in `sglang_omni/models/qwen3_tts/compat.py`, which every Qwen3-TTS entry point applies before importing `qwen_tts`. The pinned Transformers 5.19 / SGLang 0.5.23 stack is therefore the supported configuration, not a workaround.

If you hit a `TypeError` raised from inside `qwen_tts`, do not resolve it by installing the package's own Transformers pin — that breaks the rest of the runtime. Report it instead, so the shim can cover it.

The Python `sox` package shells out to the system `sox` binary on some paths, so install both.
