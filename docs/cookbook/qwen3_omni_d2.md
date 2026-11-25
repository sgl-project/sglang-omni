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

On one RTX PRO 6000 Blackwell, a question asking for the sum of two and two produced the transcript “4” in the 80 ms release with seed 1337. The 160,166 input samples included trailing silence. Omni streamed 125 of 126 frames before input EOS, and all 241,920 output samples matched the official runtime byte for byte. A second session on the same server replayed the user channel of the [public interruption example](https://d2-duplex-demo.github.io/assets/audio/qwen-interruption.wav). User speech resumed during the smartphone answer, D2 emitted an interrupt at 9.92 s on the media timeline, and its next reply discussed smartwatches.

To compare your own recording, stop the Omni server to release its GPU, then run the reference with the same input, revision, seed and trailing silence. It writes the waveform and a JSON file of per-frame events next to it.

```bash
.venv-d2/bin/d2-qwen infer --model penguinfish1688/dialogue-to-duplex \
  --revision "$D2_REV" --seed 1337 --kv-budget 2048 --tail-seconds 0 \
  --input question-with-silence.wav --output d2-reference.wav
```

## Session limits

The stage owns one model and one fixed KV allocation. Closing or disconnecting releases the conversation for the next client. `--duplex.factory.kv_budget` defaults to 2,048 Thinker tokens, including the system prompt. A full cache ends the session with `context_exhausted`. `--duplex.factory.seed` defaults to 1337 and resets when each session opens. The D2 checkpoint fixes the system prompt and voice. Its interruption decision is driven by incoming audio.
