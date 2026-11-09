# MOSS-Speech

MOSS-Speech generates spoken responses to user text. It uses the supplied text as the user turn without adding a repetition instruction. A question such as `What is the capital of France?` asks the model for a spoken answer; this endpoint does not guarantee verbatim reading of its input. This model has a separate implementation from [MOSS-TTS](moss_tts.md).

## Installation and serving

Follow the [MOSS-Speech codec installation instructions](../get_started/installation.md#moss-speech-codec-dependencies), including the external codec source paths. Configure the local model and codec checkpoints in `moss-speech.yaml`:

```yaml
config_cls: MossSpeechPipelineConfig
model_path: checkpoints/MOSS-Speech
stages:
  audio_vocoder:
    factory:
      codec_path: checkpoints/MOSS-Speech-Codec
```

```bash
sgl-omni serve --config moss-speech.yaml --model-name moss-speech --port 8000
```

The default pipeline is `preprocessing → ar_engine → audio_vocoder`. It uses BF16 autoregressive inference with FlashInfer and native decode CUDA graphs, followed by offline FP32 codec decoding.

## Request behavior

Send a text question and a reference voice clip through `/v1/audio/speech`:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moss-speech",
    "input": "What is the capital of France?",
    "ref_audio": "voices/reference.wav",
    "response_format": "wav",
    "stream": false,
    "max_new_tokens": 512,
    "seed": 42
  }' \
  --output answer.wav
```

The fixed system prompt asks the model to answer the user's questions with spoken responses. The user text is passed through unchanged. Generated content depends on that text and the model's sampling parameters, so question-answering quality must be evaluated against the expected answer rather than against the question's transcript. Use an explicit instruction in `input` when requesting a particular response style or a reading task.

Each request requires one reference audio source through `ref_audio` or a single `references` entry. It conditions the output voice; it is not interpreted as a spoken user question. The response is a complete WAV or PCM payload, and this interface does not return token usage or a generation finish reason.

Reference audio accepts a server-local path, a `file://` URI, an HTTP(S) URL, or a base64 audio DataURI. Relative paths resolve from the server's working directory. Audio is downmixed to mono and must be finite, long enough for codec conditioning, and at most 30 seconds. The original sample rate is preserved until each conditioning path resamples it.

Voice conditioning follows the official HF codec's `torchaudio.transforms.Resample` calls: the token encoder resamples the original audio directly to 16 kHz on the encoder device, while mel conditioning resamples the original audio to 24 kHz on CPU and speaker conditioning resamples that 24 kHz waveform to 16 kHz on CPU. Reusing the speaker waveform for token extraction would introduce an extra resampling stage. Strict numerical comparisons also require matching Torch/CUDA and ONNX Runtime versions; aligning this frontend alone does not establish complete HF equivalence.

## Current interface coverage

The current adapter accepts one text user turn and produces non-streaming audio through `/v1/audio/speech`. It requires `stream=false`, `voice=default`, and `speed=1.0`. Speech input, text output, chat messages, custom system prompts, multi-turn history, and streaming responses are not exposed by this adapter.

Speech-question-answering suites that require S2T or S2S through `/v1/chat/completions` cannot qualify this interface. Evaluate its supported text-to-audio response path separately, retain generation-budget limits and failed requests, and score the generated answer after transcription. Removing an injected repetition instruction does not add the missing interfaces or establish numerical equivalence with the complete HF implementation.
