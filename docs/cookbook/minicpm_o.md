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

## Native Full-Duplex Ability

```bash
python -m sglang_omni.cli serve \
  --model-path openbmb/MiniCPM-o-4_5 \
  --variant session \
  --enable-realtime --port 8000
```

`--variant session` selects the full-duplex pipeline. The defaults below match the MiniCPM-o demo. Change any of them with a flag of the same name, for example `--max_sessions 4` or `--sampling.temperature 0.5`. To print the whole configuration:

```bash
sgl-omni config resolve --model-path openbmb/MiniCPM-o-4_5 --variant session
```

For repeatable output in regression and parity recordings, add `--sampling.greedy true --sampling.top_k 100 --thinker.engine.disable_cuda_graph true --talker.engine.disable_cuda_graph true`.

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 2 | Conversations at the same time. Further connections get HTTP 503. The thinker and talker reserve GPU memory for this many full-length conversations, so raise it only as far as the GPU has room. Startup warms up perception at each batch size up to this value |
| `reference_audio` | checkpoint default | Voice used when a session sends no reference |
| `speech_state_bytes_per_session` | 2 GiB | Memory the speech stage may hold per conversation. A conversation that needs more is closed and the others keep running |
| `speech.dtype` | `float32` | Precision of the voice decoder's flow model: `float32`, `float16` or `bfloat16`; the lower precisions change the voice slightly |
| `speech.enable_dit_torch_compile` | `false` | Compile the voice decoder's flow model with `torch.compile` while the server starts; CUDA only |
| `speech.n_timesteps` | 10 | Flow-matching steps per audio chunk; fewer steps decode faster at some cost in voice quality |
| `stages.thinker/talker.engine.enable_torch_compile` | `false` | Compiles every decode graph batch size; adds minutes to startup |
| `sampling` | printed by `config resolve` | Default sampling when a session does not set its own |
| `vision` | printed by `config resolve` | Camera-frame limits per unit (1 s of audio) |

While the server starts, the speech stage records the voice decoder as CUDA graphs for up to 8 conversations decoded together, or `max_sessions` if lower; the graphs hold GPU memory and add to startup time, and a voice whose reference audio is longer than the default voice's is decoded without them.

A session holds at most 8192 tokens of history, which is the model's trained context length. When that fills, the server sends `context_exhausted` and closes the session.

The full-duplex server is ready when this returns JSON containing `"native_full_duplex":true`:

```bash
curl --fail http://localhost:8000/v1/realtime/capabilities
```

The results should be as follows:

```json
{"model":"openbmb/MiniCPM-o-4_5","interaction":"native","native_full_duplex":true,"proactive_output":false,"turn_control":[null],"client_commit":false,"input_modalities":["audio","image"],"output_modalities":["audio","text"],"input_audio_format":{"type":"audio/pcm","rate":16000},"output_audio_format":{"type":"audio/pcm","rate":24000},"native_unit_ms":1000,"first_unit_ms":1000,"microturn_ms":"variable","tail_policy":"pad","supports_server_interrupt":false,"supports_truncate":false,"supports_resume":false,"partial_style":"append_only","pressure_policy":"reject","strict_order":true,"sampling_parameters":["greedy","temperature","top_k","top_p","repetition_penalty","listen_prob_scale","force_listen_count","max_new_tokens_per_unit","repetition_window_size","talker_temperature","talker_repetition_penalty"],"supports_reference_audio":true,"input_image_format":{"types":["image/jpeg","image/png"],"max_bytes":524288,"max_frames_per_unit":4,"max_slice_nums":9},"limits":{"max_input_bytes":1920000,"max_output_bytes":4194304,"max_output_events":256,"max_history_chars":65536,"cleanup_timeout_s":30}}#
```

The duplex server only accepts `/v1/realtime` requests; other chat completions or voice cloning against it fail.

### Browser demo

With the full-duplex server running, start the demo page in another terminal:

```bash
python playground/realtime/app.py --api-base http://127.0.0.1:8000 --port 8080
```

Thanks to the OpenMoss team for the demo page.

Once the terminal prints `Running on`, open <http://localhost:8080> and click **Start talking** to talk through your microphone; the camera can be turned on during the call. If the server is remote, first run `ssh -N -L 8080:127.0.0.1:8080 USER@SERVER` on your own machine. Open the page through `localhost`, or the browser will not grant microphone access.

The settings panel switches between English and Chinese, changes the voice and adjusts sampling. The download button in the top bar saves the conversation trace; attach it when reporting a problem.


### Full-duplex protocol

Clients connect to `/v1/realtime` over WebSocket. They send 16 kHz mono PCM16 audio in `input_audio_buffer.append`; the server returns 24 kHz audio in `response.output_audio.delta` and text in `response.output_audio_transcript.delta`. The model decides once per second of audio whether to keep listening or to speak.

1. After `session.created`, send `session.update` and wait for `session.updated`.
2. Send audio with `input_audio_buffer.append` at the pace it is captured; replies arrive while you send.
3. When the audio ends, send `sglang.input_audio.end`, wait for `sglang.input_audio.drained`, then send `session.close`.

For a Python client, see `warmup` in `playground/realtime/app.py`.

Session settings go in the `sglang` field of `session.update`, before the first audio packet, and stay fixed for the session:

| Setting | Field | Notes |
|---|---|---|
| Voice | `reference_audio` | `{"media_type": "audio/wav", "data": "<base64>"}`, a PCM16 WAV of at most 30 s and 1 MiB; `tts_reference_audio` changes only the output voice |
| Sampling | `sampling` | For example `temperature`, `top_p` and `listen_prob_scale`; unset fields keep the server's defaults |
| Image detail | `max_slice_nums` | Higher is sharper but accepts fewer frames per second |

Send camera frames with `sglang.input_image.append`: a base64 JPEG or PNG in `image`, and its position on the audio timeline in `sglang.t_ms`. By default up to 4 frames per second are accepted; `session.updated` reports the actual limit.

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
