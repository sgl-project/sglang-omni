# Cascade voices

`vivian.wav` is the voice of the `qwen3-asr-qwen3-tts-vivian` and `qwen3-asr-qwen3.8-27b-vivian` profiles: 5.36 s of 24 kHz mono PCM16, the Qwen3-TTS-12Hz-1.7B-CustomVoice speaker Vivian reading "Hi there, it is nice to meet you, and I am happy to help with whatever you need today." with seed 1234. A CustomVoice server on port 8005 regenerates the same samples with:

```bash
curl -s http://127.0.0.1:8005/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "voice": "Vivian", "language": "English", "seed": 1234,
       "response_format": "pcm", "stream": true,
       "input": "Hi there, it is nice to meet you, and I am happy to help with whatever you need today."}' \
  | ffmpeg -f s16le -ar 24000 -ac 1 -i - vivian.wav
```

[`openai_speech`](../../../docs/developer_reference/full_duplex_guide.md#openai_speech) says why the profiles clone one recording instead of naming a CustomVoice speaker.
