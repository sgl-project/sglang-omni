# MiniCPM-o

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is a Gemini 2.5 Flash level model for vision, speech, and full-duplex live streaming. SGLang-Omni serves it in two ways:

| Mode | Endpoint | Use it for |
|---|---|---|
| Chat | `/v1/chat/completions` | One request, one reply, in text or speech |
| Full duplex | `/v1/realtime` (WebSocket) | A live voice or video call where the model listens and speaks at the same time |

## Prerequisites

Follow [Installation](../get_started/installation.md), then run from the repository root.

Chat with speech output:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --port 8000
```

Chat with text output only:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --text-only --port 8000
```

## Normal Usage


```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reference = base64.b64encode(Path("docs/_static/audio/male-voice.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
message = response.choices[0].message
Path("reply.wav").write_bytes(base64.b64decode(message.audio.data))
print(message.audio.transcript or message.content)
```

## Video Chat Inputs

Chat accepts inline video_url content and top-level video or videos fields.
Top-level requests without use_audio_in_video retain the existing shared video
sampling and resizing policy. They place explicit images, video frames, and
independent audio before the final user message's text, in that order.
Predecoded video tensors remain supported, including mixed file/tensor inputs.

Inline videos and top-level requests with use_audio_in_video use MiniCPM timed
video sampling. Set use_audio on an inline video, or use_audio_in_video on a
top-level request, to include interleaved audio from file or URL sources.
Top-level interleaved videos follow explicit images and independent audio.

Timed video frames retain native resolution by default; the MiniCPM image processor
performs the final resize. video_max_frames limits the sampled frame count,
while video_max_pixels and video_total_pixels resize frames after decoding.
Those pixel options do not bound the initial native-resolution allocation.

## Native Full-Duplex Ability

MiniCPM-o 4.5 also serves live full-duplex voice and video calls on `/v1/realtime`. Launching, settings, the browser demo and the protocol are in [Full Duplex Interactive Model](../developer_reference/full_duplex_interactive_model.md#native-full-duplex-minicpm-o).

## Text to speech

`/v1/audio/speech` reads the given text aloud. MiniCPM-o prefills the text in one Thinker pass and conditions the Talker on each of its tokens, instead of generating the same words one token at a time. Set `language` to `Chinese` for Chinese text, and pass a base64 audio data URI as `ref_audio` to clone a voice.

```python
audio = client.audio.speech.create(
    model="MiniCPM-o-4_5",
    voice="default",
    input="Hello!",
    extra_body={"language": "English"},
)
```

The speech output is non-streaming. The sampling fields you set, such as `temperature`, `top_p` and `max_new_tokens`, apply to the Talker, which generates the speech; the rest keep the Talker's defaults.

## Speculative Decoding

The Thinker can verify tokens proposed by a DSpark draft trained on its own answers to text, audio, image and video prompts, [`Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm`](https://huggingface.co/Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm):

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --port 8000 \
  --thinker.engine.speculative_algorithm DSPARK \
  --thinker.engine.speculative_draft_model_path Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm \
  --thinker.engine.speculative_dspark_block_size 7
```

An EAGLE3 draft for Qwen3-8B, such as [`Tengyunw/qwen3_8b_eagle3`](https://huggingface.co/Tengyunw/qwen3_8b_eagle3), runs with `--thinker.engine.speculative_algorithm EAGLE3`, `speculative_num_steps 3`, `speculative_eagle_topk 1` and `speculative_num_draft_tokens 4`.

- **Speech output.** The Talker reads one Thinker hidden row per text token, and the draft's layer capture keeps the row of every verified token, so a speech server accepts drafts that verify one linear block per request: a chain EAGLE3 draft (`speculative_eagle_topk 1`), DFlash or DSpark. Qwen3-8B DFlash drafts mask with token 151669, which is `<image>` in MiniCPM-o, so they accept almost no tokens here.
- **Sampling.** `length_penalty` applies to every verified token. Requests with `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `min_p` or logprobs are rejected, because the draft workers cannot reproduce them.
- **Scheduling.** Async decode and mixed chunks are turned off.
