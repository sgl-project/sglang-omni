# Examples

Pick a task below. These examples use the [CUDA installation](installation.md).

| I want to… | Example |
| --- | --- |
| See the words as they are transcribed | [Stream a transcript](#stream-a-transcript) |
| Turn a sentence into a WAV file | [Generate speech from text](#generate-speech-from-text) |
| Get a text answer about an audio clip | [Ask a question about audio](#ask-a-question-about-audio) |

## Stream a transcript

**What you need:** The Qwen3-ASR-1.7B server and `query_to_cars.wav` from the
[Quickstart](quickstart.md).

**Start or reuse the server:** Keep the Quickstart server running on port `8000`.

**Send a request:** From the terminal where you downloaded the clip, upload it
with `stream=true`. `curl -N` prints events as they arrive.

```bash
curl -fsS -N http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@query_to_cars.wav \
  -F response_format=json \
  -F stream=true
```

**Expected result:** The server emits `transcript.text.delta` events as it decodes,
then `transcript.text.done` with the full transcript, followed by `data: [DONE]`.
The complete file is uploaded before decoding starts. For live audio and
more options, see the [Qwen3-ASR Cookbook](../cookbook/qwen3_asr.md#stream-transcription).

## Generate speech from text

**What you need:** A CUDA GPU, the [Qwen3-TTS prerequisites](../cookbook/qwen3_tts.md#prerequisites),
and a source checkout for the CustomVoice config. You do not need a reference recording.

**Start or reuse the server:** Stop the Quickstart server before reusing port
`8000`. From the root of the checkout, start Qwen3-TTS CustomVoice:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

**Send a request:** Ask the built-in speaker Ryan to read a sentence:

```bash
curl -fsS http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from SGLang-Omni.",
    "voice": "Ryan",
    "language": "English",
    "response_format": "wav"
  }' \
  -o output.wav
```

**Expected result:** Play `output.wav` to hear the sentence. See
[TTS Model Usage](../basic_usage/tts.md) for more speech requests.

## Ask a question about audio

**What you need:** Qwen3-Omni-30B-A3B-Instruct and one CUDA GPU with enough memory
for the BF16 thinker, such as an H200. The server fetches the Quickstart clip
from GitHub. For other hardware and precision options, see the
[Qwen3-Omni Cookbook](../cookbook/qwen3_omni.md#server-configuration).

**Start or reuse the server:** Stop the previous server if you are reusing its
GPU. Start a separate Qwen3-Omni server on port `8008`:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --text-only \
  --allowed-media-domain raw.githubusercontent.com \
  --port 8008
```

If you are using the Quickstart Docker setup, run the server in that container.

**Send a request:** Ask the model to identify the spoken question:

```bash
curl -fsS http://localhost:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "What question is spoken in this clip?"}],
    "audios": ["https://raw.githubusercontent.com/sgl-project/sglang-omni/7fc1a31850efbc1138e1895b5a3bcbeecf3a8481/tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 128
  }'
```

**Expected result:** Check `choices[0].message.content` for a text answer that
identifies the question about the number of cars in the picture. See [Omni Model Usage](../basic_usage/qwen3_omni.md#request-parameters) for more request options.
