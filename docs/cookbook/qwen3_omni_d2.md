# Qwen3-Omni D2

Qwen3-Omni D2 listens and speaks on the same audio timeline. SGLang-Omni serves a single conversation over `/v1/realtime`. The D2 worker owns the causal audio encoder, Thinker and Talker caches, Code2Wav and delayed self-audio feedback. [Paper](https://arxiv.org/abs/2610.17124) · [Official implementation](https://github.com/penguinfish1688/D2-dialogue-to-duplex/tree/13ac442f013665a2b9cfd882c5966ccdeda3594a/qwen3-omni-D2)

## Install and launch

Install SGLang-Omni in its [CUDA environment](../get_started/installation.md). Create a second Python 3.12 environment for the D2 worker. Its release pins PyTorch 2.11 and Transformers 5.13. Inference uses BF16 and compiled CUDA graphs. The authors recommend an RTX PRO 6000 Blackwell with 96 GB of GPU memory, 96 GB of system RAM, a C++ compiler and Python development headers.

```bash
git clone https://github.com/penguinfish1688/D2-dialogue-to-duplex.git
git -C D2-dialogue-to-duplex checkout 13ac442f013665a2b9cfd882c5966ccdeda3594a
uv venv --python 3.12 .venv-d2
uv pip install --python .venv-d2/bin/python \
  -c D2-dialogue-to-duplex/qwen3-omni-D2/configs/requirements.txt \
  './D2-dialogue-to-duplex[qwen]'
```

Run the server from the Omni environment. The pinned 80 ms release downloads the D2 adapter, distilled encoder and the Qwen3-Omni-30B-A3B-Instruct revision named in its manifest. Set `HF_HOME` to a cache with room for at least 73 GB. The first launch also compiles GPU kernels.

```bash
D2_REV=0237f21b29d2a762750ddee08e37ec4fe8e00473
sgl-omni serve \
  --model-path "penguinfish1688/dialogue-to-duplex@${D2_REV}" \
  --duplex.factory.runtime_python "$PWD/.venv-d2/bin/python" \
  --enable-realtime --port 8000
curl --fail http://localhost:8000/v1/realtime/capabilities
```

This command starts D2 with Mandarin spoken replies. To select Mandarin explicitly, pass `--language=zh`. Use `--language=auto` to let the original checkpoint choose its response language. A pipeline YAML accepts the same top-level field (`language: zh` or `language: auto`). The setting applies to every conversation on that server.

In Mandarin mode, D2 uses a Mandarin system prompt. If the Thinker would start a reply with a Latin word, the worker selects its highest-scoring allowed token beginning with two Chinese characters. The Talker synthesizes that choice directly. The model controls listening, interruptions and the rest of the reply.

A local D2 release directory containing `d2.json`, `encoder.safetensors` and `d2.safetensors` also works as `--model-path`. The manifest selects the input interval. The 80 ms release consumes 1,280 input samples per interval. Other Qwen D2 revisions use 160, 320, 640 or 1040 ms intervals. The internal playback frame is always 80 ms.

## Talk to the model

Run the realtime playground from the Omni checkout and open `http://localhost:8080`. Use headphones and keep the microphone running during the reply so D2 hears overlapping speech.

```bash
python playground/realtime/app.py --api-base http://localhost:8000 --port 8080
```

A WebSocket client sends mono 16 kHz little-endian PCM16 in `input_audio_buffer.append`. D2 returns mono 24 kHz PCM16 in `response.output_audio.delta`, including silence, and generated words in `response.output_audio_transcript.delta` for an audio grant. If D2 emits its learned interrupt token, the previous response ends with status `cancelled` and reason `interrupted`. Discard unplayed audio for that response and continue reading the stream. To end the session, send `sglang.input_audio.end`, wait for `sglang.input_audio.drained` and send `session.close`. [Native full-duplex protocol](../developer_reference/full_duplex_guide.md#native-full-duplex-protocol)

For a repeatable 80 ms session, use a 16 kHz mono PCM16 WAV containing a spoken question followed by several seconds of silence. The model advances only as input samples arrive. The recorder saves the request, event trace and protocol report to a new directory.

```bash
python -m benchmarks.eval.benchmark_duplex \
  --url ws://localhost:8000/v1/realtime --profile qwen3-omni-d2-80ms \
  --audio question-with-silence.wav --output d2-replay \
  --model penguinfish1688/dialogue-to-duplex --model-revision "$D2_REV" \
  --server-revision "$(git rev-parse HEAD)" --timeout 220
```

When testing another D2 release, use the matching replay profile, for example `qwen3-omni-d2-320ms` with the `qwen-320ms` checkpoint.

## Reference comparison

To reproduce the comparisons below, launch Omni with `--language=auto` so it uses the same prompt and decoding policy as the reference runtime.

On one RTX PRO 6000 Blackwell, a question asking for the sum of two and two produced the transcript “4” in the 80 ms release with seed 1337. The 160,166 input samples included trailing silence. Omni streamed 125 of 126 frames before input EOS, and all 241,920 output samples matched the official runtime byte for byte. A second session on the same server replayed the user channel of the [public interruption example](https://d2-duplex-demo.github.io/assets/audio/qwen-interruption.wav). User speech resumed during the smartphone answer, D2 emitted an interrupt at 9.92 s on the media timeline, and its next reply discussed smartwatches.

Each public checkpoint also answered the arithmetic question over WebSocket. Output length reflects padding the last input interval. The 80 ms and 320 ms runs included a separate official inference run with the same seed and input.

| Checkpoint | Input intervals | Received 80 ms frames | Output samples at 24 kHz | Reference PCM |
| --- | ---: | ---: | ---: | --- |
| [80 ms](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/0237f21b29d2a762750ddee08e37ec4fe8e00473) | 126 | 126 | 241,920 | Exact match |
| [160 ms](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/1f772b9aab3acac649bf6b972bef3ec71b635ef7) | 63 | 126 | 241,920 | Unmeasured |
| [320 ms](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/93d6ec8ba5068671d9199889bcbfa07b0d115606) | 32 | 128 | 245,760 | Exact match |
| [640 ms](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/c15b678d53073a2787ba5e3292f88793363c315f) | 16 | 128 | 245,760 | Unmeasured |
| [1040 ms](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/cbddf68f3b601593ea1d0cbda2163a53cffed151) | 10 | 130 | 249,600 | Unmeasured |

To compare your own recording, stop the Omni server to release its GPU, then run the reference with the same input, revision, seed and trailing silence. It writes the waveform and a JSON file of per-frame events next to it.

```bash
.venv-d2/bin/d2-qwen infer --model penguinfish1688/dialogue-to-duplex \
  --revision "$D2_REV" --seed 1337 --kv-budget 2048 --tail-seconds 0 \
  --input question-with-silence.wav --output d2-reference.wav
```

## Session limits

The stage owns one model and one fixed KV allocation. Closing or disconnecting releases the conversation for the next client. `--duplex.factory.kv_budget` defaults to 2,048 Thinker tokens, including the system prompt. A full cache ends the session with `context_exhausted`. `--duplex.factory.seed` defaults to 1337 and resets when each session opens. The server selects the prompt at startup and the checkpoint supplies the voice. Its interruption decision is driven by incoming audio.
