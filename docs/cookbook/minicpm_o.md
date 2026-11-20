# MiniCPM-o

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) accepts text, images, audio, and video, and can respond with text or speech. It also supports live full-duplex conversations:

| Mode | Endpoint | Use it for |
|---|---|---|
| Chat | `/v1/chat/completions` | One request, one reply, in text or speech |
| Text to speech | `/v1/audio/speech` | Reading a given text aloud |
| Full duplex | `/v1/realtime` (WebSocket) | A live voice or video call where the model listens and speaks at the same time |

## Quick start

### Installation requirements

Follow [Installation](../get_started/installation.md), then run the commands on this page from the repository root. For Intel GPUs, follow the [MiniCPM-o XPU recipe](../other_platforms/installation_xpu.md#minicpm-o-45-omni-chat-and-speech-single-xpu). For Ascend NPU, see [Installation — NPU](../other_platforms/installation_npu.md#minicpm-o) for the launch command, configuration, and validated versions.

### Start the server and check readiness

```bash
sgl-omni serve --model-path openbmb/MiniCPM-o-4_5 --port 8000
```

This serves chat with speech output. Add `--text-only` to serve text output only.

In another terminal, check readiness:

```bash
curl -fS http://localhost:8000/health
```

The server is ready when this returns HTTP 200 with `"status": "healthy"`. While it starts, the request returns HTTP 503 or the connection is refused.

### Send a chat request

```bash
curl -fS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Please say hello."}],
    "modalities": ["text"]
  }'
```

The answer is in `choices[0].message.content`.

For a spoken reply, set `modalities` and `audio` in the same request:

```json
{
  "messages": [{"role": "user", "content": "Please say hello."}],
  "modalities": ["text", "audio"],
  "audio": {"format": "wav"}
}
```

The reply contains base64-encoded WAV audio in `message.audio.data` and a transcript in `message.audio.transcript`. To clone a voice, add `audio.ref_audio` as a base64 audio data URI, using a clip such as `docs/_static/audio/male-voice.wav`.

For video input, use inline `video_url` content parts or the top-level `videos` field. To include the video's audio, set `use_audio` (inline) or `use_audio_in_video` (top-level). See [Video and Audio Input](../basic_usage/omni.md#video-and-audio-input) for an example and `video_max_frames` for frame limits.

## Text to speech

The speech server also reads a given text aloud on `/v1/audio/speech`; [Non-streaming Speech](../basic_usage/tts.md#non-streaming-speech) has the request format. Set `language` to `Chinese` for Chinese text, and pass a base64 audio data URI as `ref_audio` to clone a voice. The output is non-streaming, and sampling fields such as `temperature`, `top_p` and `max_new_tokens` apply to the Talker.

## Full duplex

MiniCPM-o 4.5 serves live voice and video calls on `/v1/realtime` with its `session` variant:

```bash
sgl-omni serve --model-path openbmb/MiniCPM-o-4_5 --variant session --enable-realtime --port 8000
```

The server is ready when `curl --fail http://localhost:8000/v1/realtime/capabilities` returns JSON containing `"native_full_duplex":true`. Then start the browser demo:

```bash
python playground/realtime/app.py --api-base http://127.0.0.1:8000 --port 8080
```

Once the terminal prints `Running on`, open <http://localhost:8080> and click **Start talking**; the camera can be turned on during the call. Open the page through `localhost`, or the browser will not grant microphone access. Settings, the protocol, and benchmarking are in the [Full-Duplex Guide](../developer_reference/full_duplex_guide.md#native-full-duplex-minicpm-o).

### Realtime-Venus checkpoints

[Realtime-Venus](https://huggingface.co/inclusionAI/Realtime-Venus) is a separate pair of checkpoints that inclusionAI fine-tuned from MiniCPM-o 4.5: Realtime-Venus-Omni for audio-visual full-duplex calls and Realtime-Venus-Audio for audio-only chat and calls. Both keep the MiniCPM-o weight layout and serve with the commands on this page.

Download them with the repository's script, which keeps the shared `config.yaml` beside the model directories. `--model audio` downloads Realtime-Venus-Audio instead.

```bash
hf download inclusionAI/Realtime-Venus download_models.py --local-dir .
python download_models.py --model omni --local-dir Realtime-Venus
```

Pass the model directory, not the repository id:

```bash
sgl-omni serve --model-path Realtime-Venus/Realtime-Venus-Omni --variant session --enable-realtime --port 8000
```

## Speculative decoding

To enable speculative decoding for the Thinker, use the multimodal [DSpark draft](https://huggingface.co/Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm):

```bash
sgl-omni serve --model-path openbmb/MiniCPM-o-4_5 --port 8000 \
  --thinker.engine.speculative_algorithm DSPARK \
  --thinker.engine.speculative_draft_model_path Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm
```

On one H100, the draft raised speech throughput 1.26–1.46x at 1 to 32 concurrent requests on Seed-TTS English voice cloning, and text throughput 1.13–1.21x at 1 to 4 concurrent requests on MMSU.

Add `"return_spec_tokens_details": true` to a chat request to get the acceptance statistics in `sglext.spec_tokens_details`.

## Known limitations

- `/v1/audio/speech` output is non-streaming.
- With a draft, requests that set `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `min_p` or logprobs are rejected.
- The limits shared by all omni models are in [Known Limitations](../basic_usage/omni.md#known-limitations).

## Next steps

For more request examples, see [Omni Model Usage](../basic_usage/omni.md). Its examples use `qwen3-omni` on port 8008; use port 8000 and omit the optional `model` field for MiniCPM-o.
