# AURA Omni

[AURA](https://huggingface.co/aurateam/AURA) is a fine-tune of Qwen3-VL-8B-Instruct for live
video assistance: it watches a clip whose last frame is the present moment and either answers
or outputs `<|silent|>` when nothing calls for a reply. AURA Omni wraps it in speech on both
ends:

```text
speech ──► Qwen3-ASR ──► transcript ─┐
video ────────────────────────────────┼─► AURA (Qwen3-VL) ──► answer text ──► Qwen3-TTS ──► speech
typed text ──────────────────────────┘                     └─► <|silent|>: no speech
```

SGLang-Omni serves it as one pipeline behind `/v1/chat/completions`, one turn per request.

| Stage | Process | Default card | What it runs |
|---|---|---|---|
| `preprocessing` | `preprocessing` | CPU | splits the chat turn into speech and the AURA turn |
| `asr` | `asr` | 0 | Qwen3-ASR, transcribes the speech |
| `aura` | `aura` | 1 | AURA on SGLang's Qwen3-VL model and multimodal processor |
| `decode` | `aura` | CPU | text output, streamed or whole |
| `tts_preprocessing`, `tts_engine`, `vocoder` | `tts` | 2 | Qwen3-TTS talker and codec |

## Checkpoints

```bash
hf download aurateam/AURA                          # 17 GB, BF16
hf download Qwen/Qwen3-ASR-1.7B
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

`--model-path` names AURA; the other two are `asr_model_path` and `tts_model_path`, with the
repos above as defaults. AURA's `config.json` points `auto_map` at remote code (and at a
processor module the repository does not ship); the stage loads it with the stock
Transformers Qwen3-VL classes and never runs that code.

## Launch

```bash
sgl-omni serve --model-path aurateam/AURA --port 8000
```

On 24 GB Intel Arc Pro B60 cards use the checked-in placement, which also sizes AURA's KV pool
(its 16.5 GiB of weights leave SGLang's automatic budget no room for one on XPU):

```bash
sgl-omni serve --config examples/configs/aura_omni_xpu_b60.yaml --port 8000
```

Override any stage with dotted flags, for example `--aura.gpu 3`,
`--aura.engine.context_length 65536` or `--tts_model_path /path/to/Qwen3-TTS-12Hz-1.7B-Base`.

## Requests

Send the speech as `input_audio` (or `audio_url`) and the clip as `video_url`, and ask for
`"modalities": ["text", "audio"]` to get speech back:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aura",
    "modalities": ["text", "audio"],
    "audio": {"voice": "ryan", "format": "wav"},
    "messages": [{"role": "user", "content": [
      {"type": "video_url", "video_url": {"url": "file:///path/to/clip.mp4"}},
      {"type": "audio_url", "audio_url": {"url": "file:///path/to/question.wav"}}
    ]}]
  }' -o reply.json
python -c "import base64, json; m = json.load(open('reply.json'))['choices'][0]['message']; \
  print(m.get('content')); open('reply.wav', 'wb').write(base64.b64decode(m['audio']['data']))"
```

- Typed text in the same user message follows the transcript; a request may also carry no
  speech at all, and then AURA reads the text alone. Images work like videos.
- A `system` message replaces AURA's trained system prompt; earlier `user` and `assistant`
  messages render as text history. One audio input per request.
- `"stream": true` streams the answer text, then the speech in chunks.
- Unset sampling fields take AURA's `generation_config.json` (temperature 0.7, top-k 20,
  top-p 0.8); `max_tokens` defaults to 256.

**Silence.** When AURA decides no reply is needed, the response has empty `content`, no
`audio`, and `completion_tokens: 2` (`<|silent|>` and the end of turn); Qwen3-TTS never runs.

**Voices.** With the default CustomVoice checkpoint, `audio.voice` picks the speaker: `aiden`,
`dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu` or `vivian` (the default).
A Base checkpoint clones a reference voice instead, passed per request:

```json
"stage_params": {"tts_preprocessing": {
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "What the reference clip says.",
  "language": "English"
}}
```

`stage_params.asr` reaches Qwen3-ASR the same way (for example `{"language": "English"}`), and
`stage_params.tts_engine` carries Qwen3-TTS sampling overrides.

## Offline example

```bash
python examples/run_aura_omni.py \
  --config examples/configs/aura_omni_xpu_b60.yaml \
  --audio tests/data/query_to_draw.wav --video tests/data/draw.mp4 --out reply.wav
```

It starts the pipeline in-process, sends one turn and writes the spoken answer.

## Performance

Measured on Intel Arc Pro B60 with the XPU config above, one request at a time, for the
6.5 s 4K `tests/data/draw.mp4` clip (3,456 video tokens) and the 5 s `query_to_draw.wav`
question:

| Leg | Time |
|---|---|
| Qwen3-ASR transcript | 0.3 s |
| AURA video preprocessing (decode and resize on CPU) | 2.4 s |
| AURA prefill, vision encoder included | 3.4 s |
| AURA decode | 50 ms per token |
| Qwen3-TTS | about 0.5 s per second of speech |

AURA matches stock SGLang serving the same checkpoint on the same card, both with a cold
cache: 5.7 s to the first token against 6.0 s, identical greedy output, and the same 50 ms per
token (streaming adds about 3.5 ms per token for the hop to the `decode` stage). A clip the
radix cache has already seen skips the vision encoder and most of the prefill.

## Limitations

- One turn per request. The full-duplex `/v1/realtime` session mode, with its per-session
  history and sentence-by-sentence speech, is not supported.
- Speech starts after AURA has finished its answer.
