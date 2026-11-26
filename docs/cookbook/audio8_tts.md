# Audio8 TTS Realistic-Air

[Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) is Edge0's 0.6B text-to-speech model. It generates speech from text alone or clones a voice from one reference clip and its transcript. The OpenAI-compatible `/v1/audio/speech` endpoint returns the complete waveform with a native sample rate of 44.1 kHz.

## Start the server

Follow the [CUDA installation guide](../get_started/installation.md) on a machine with one NVIDIA GPU. The example configuration uses the published Realistic-Air checkpoint.

```bash
sgl-omni serve --config examples/configs/audio8_tts.yaml --port 8000
```

To use a downloaded checkpoint, add `--model-path /path/to/Audio8-TTS-Realistic-Air` and point it at the directory containing the model weights, tokenizer, and `codec.pth`.

## Synthesize speech

Send the text in `input` and use `stream: false`. Omit `voice` or use `"default"`.

```bash
curl -fS http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天天气很好，非常适合去公园散步。",
    "response_format": "wav",
    "stream": false,
    "seed": 42
  }' \
  --output audio8.wav
```

The WAV response uses 44.1 kHz by default. Set the request's `sample_rate` in Hz to resample the output when the response format supports that rate. The response's `X-Sample-Rate` header reports the delivered rate. See [TTS request parameters](../basic_usage/tts.md#request-parameters).

## Clone a voice

Add one reference clip and its transcript to the same request. Use `ref_audio` and `ref_text`, or use one entry in `references` with `audio_path` and `text`. A local path must be readable by the server.

```json
{
  "input": "This sentence uses the reference voice.",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "The transcript of the reference recording.",
  "response_format": "wav"
}
```

See the [shared voice cloning examples](../basic_usage/tts.md#voice-cloning) for supported reference formats and media access rules.

## Sampling

Audio8 accepts `temperature`, `top_k`, `top_p`, `seed`, and `max_new_tokens`. Set `temperature` to `0` for greedy sampling or `top_k` to `-1` to remove the top-k limit. The `max_new_tokens` field limits generated tokens within the model's remaining context.
