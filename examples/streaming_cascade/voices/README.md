# Cascade voices

`vivian.wav` is the voice of the `qwen3-asr-qwen3-tts-vivian` and `qwen3-asr-qwen3.8-27b-vivian` profiles and of the `neutral` style in `qwen3-asr-qwen3.8-27b-vivian-styles`: 5.36 s of 24 kHz mono PCM16, the Qwen3-TTS-12Hz-1.7B-CustomVoice speaker Vivian reading "Hi there, it is nice to meet you, and I am happy to help with whatever you need today." with seed 1234. A CustomVoice server on port 8005 regenerates the same samples with:

```bash
curl -s http://127.0.0.1:8005/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "voice": "Vivian", "language": "English", "seed": 1234,
       "response_format": "pcm", "stream": true,
       "input": "Hi there, it is nice to meet you, and I am happy to help with whatever you need today."}' \
  | ffmpeg -f s16le -ar 24000 -ac 1 -i - vivian.wav
```

[`openai_speech`](../../../docs/developer_reference/full_duplex_guide.md#openai_speech) says why the profiles clone one recording instead of naming a CustomVoice speaker.

`qwen3-asr-qwen3.8-27b-vivian-styles` clones one more recording per [speaking style](../../../docs/developer_reference/full_duplex_guide.md#speaking-styles), each 24 kHz mono PCM16 of the same speaker reading a sentence of that style with an instruction. Each is the take, among seeds 1 to 8, that sounded most like Vivian while carrying the style when listened to side by side; for `calm` the other takes were too slow.

| File | Seconds | Seed | Instruction | Sentence |
| --- | --- | --- | --- | --- |
| `vivian_cheerful.wav` | 5.52 | 3 | Speak in a cheerful, upbeat tone. | That is wonderful news, congratulations! I am so glad it all worked out for you. |
| `vivian_sympathetic.wav` | 6.64 | 8 | Speak gently and warmly, with sympathy and a touch of sadness. | I am so sorry you are going through this. It is completely okay to feel sad right now. |
| `vivian_excited.wav` | 5.76 | 6 | Speak with excited, delighted surprise. | Wow, I can't believe it, you actually did it! This is absolutely amazing! |
| `vivian_calm.wav` | 9.04 | 7 | Speak calmly and slowly, in a soft, soothing voice. | Let's take a slow, deep breath together. There is no rush, and everything is going to be fine. |

A CustomVoice server started with `--config examples/configs/qwen3_tts_1_7b_customvoice.yaml` made each with the request above plus the instruction, its seed and its sentence, for example:

```bash
curl -s http://127.0.0.1:8005/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "voice": "Vivian", "language": "English", "seed": 3,
       "instructions": "Speak in a cheerful, upbeat tone.", "response_format": "pcm", "stream": true,
       "input": "That is wonderful news, congratulations! I am so glad it all worked out for you."}' \
  | ffmpeg -f s16le -ar 24000 -ac 1 -i - vivian_cheerful.wav
```
