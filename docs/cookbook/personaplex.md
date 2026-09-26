# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): every 80 ms it reads one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next frame. A voice prompt and a `<system>` role prompt set the persona, and the model decides for itself when to speak; there is no VAD.

SGLang-Omni serves it two ways: an **offline** pipeline (one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same length out), and a **realtime** variant that holds a live full-duplex call over `/v1/realtime`, one 80 ms frame at a time ([#1909](https://github.com/sgl-project/sglang-omni/issues/1909)).

## Prerequisites

The checkpoint is gated; accept the license on the model page and log in (`hf auth login`), then:

```bash
hf download nvidia/personaplex-7b-v1
```

It ships the 7B weights, the Mimi codec, the SentencePiece text model and `voices.tgz`, which is unpacked into `voices/` next to the checkpoint on first use, or into the temp directory when that folder cannot be written. Recorded voice prompts (`--voice some.wav`) also need `pip install pyloudnorm`; the packaged `.pt` voices do not.

Everything runs on one GPU. On an H200 the LM engine reserves `mem_fraction_static=0.3` and the two Mimi instances stay under 1 GB each; lower `--lm.engine.mem_fraction_static` on smaller cards.

## Running the offline example

```bash
python examples/run_personaplex.py \
  --model-path nvidia/personaplex-7b-v1 \
  --audio /path/to/caller.wav \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --out reply.wav
```

Five stages run under `MultiProcessPipelineRunner` (preprocessing, Mimi encode, the LM engine, text decode, streaming code2wav). Pick the GPU with `CUDA_VISIBLE_DEVICES`; stage settings take the same dotted flags as `serve`, such as `--lm.engine.mem_fraction_static 0.25`. Input conventions:

- The recording is resampled to 24 kHz and **channel 0 is used**.
- The reply is exactly as long as the input, offset by one frame: the model answers while it listens, so leave silence after the caller's last words if you want a full answer.
- Prompt plus reply must fit the LM context, 8192 positions by default (about 10.8 minutes); a longer recording is rejected with the limit in the message and needs `--lm.engine.context_length`.
- The reply text is the model's inner monologue with the frame markers (`PAD`, `EPAD`, `BOS`, `EOS`) removed.

## Serving over HTTP

```bash
python -m sglang_omni.cli serve --model-path nvidia/personaplex-7b-v1 --port 8000
```

Send the recording as the `prompt` of `POST /generate` (a path, URL or `data:` URI, with `"return_logprob": false`), or as the single entry of `audios` in `POST /v1/chat/completions` with `"modalities": ["text", "audio"]`; the reply audio comes back base64-encoded as WAV. Options with no request field of their own go under `stage_params`: `voice` and `text_prompt` under `preprocessing`, `audio_temperature`, `audio_top_k` and `seed` under `lm`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/personaplex-7b-v1",
  "messages": [{"role": "user", "content": ""}],
  "audios": ["/path/to/caller.wav"],
  "modalities": ["text", "audio"],
  "stage_params": {"preprocessing": {"voice": "NATM1"}, "lm": {"audio_temperature": 0.8}}
}'
```

## Full-duplex calls over `/v1/realtime`

The realtime variant runs the same model as a live call: the client streams the caller's microphone, and the server streams the agent's voice back continuously, speaking, listening and interrupting as the model decides.

```bash
python -m sglang_omni.cli serve --config examples/configs/personaplex_realtime.yaml --enable-realtime --port 8000
```

Each 80 ms unit of caller audio (24 kHz mono PCM16, 3840 bytes) travels preprocessing → Mimi encode → LM → Mimi decode as one step. The LM holds one SGLang streaming session per call: the first unit prefills the voice and role prompt and steps once, and every later unit extends the retained KV cache by exactly one position, so a call produces the same frames as the offline pipeline given the same audio.

Open `ws://localhost:8000/v1/realtime`, wait for `session.created`, then send `session.update`. `instructions` sets the role prompt; the voice is the default (`NATF2`). The server grants `native_unit_ms: 80`, one output modality (`audio` by default, with the spoken words as `response.output_audio_transcript.delta`; or `text`), and pads a trailing partial unit (`tail_policy: pad`). Stream audio with `input_audio_buffer.append` (`sglang.seq` counting up from 0) and finish with `sglang.input_audio.end`.

```json
{"event_id": "e0", "type": "session.update", "session": {"instructions": "You are a patient support agent."}}
{"event_id": "e1", "type": "input_audio_buffer.append", "audio": "<base64 PCM16>", "sglang": {"seq": 0}}
{"event_id": "e9", "type": "sglang.input_audio.end"}
```

The whole call is one response: `response.created` arrives with the first unit's output, `response.output_audio.delta` carries 80 ms of agent audio per unit (160 ms for the first), and `response.done` follows the end of the caller's input. The realtime variant serves only `/v1/realtime`; use the default variant for `/generate` and chat completions.

## Request parameters

| Parameter | Effect |
|---|---|
| `voice` | A packaged voice name (`NATF0..3`, `NATM0..3`, `VARF0..4`, `VARM0..4`), a `.pt` file, or a recording. Default `NATF2`; an empty string runs without a voice prompt. |
| `text_prompt` (or `instructions`) | The role prompt, wrapped in `<system>` tags. Default: the reference assistant prompt. |
| `temperature`, `top_k` | Text sampling; defaults 0.7 / 25 unless the request sets them. `temperature=0` is greedy. |
| `audio_temperature`, `audio_top_k` | Code sampling in the depformer; defaults 0.8 / 250. |
| `seed` | Makes both draws reproducible (a child seed each for text and audio). |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length. |

## Known limitations

- Offline, one request at a time (`max_running_requests=1`); realtime, one call at a time for the same reason. Raise `--lm.engine.max_running_requests` for more.
- A realtime unit's output is released once the whole route has finished it, so each 80 ms frame must clear Mimi encode, the LM, the depformer and Mimi decode within 80 ms, or the call falls behind.
- A realtime call is bounded by the LM context like an offline request (about 10.8 minutes at 8192 positions); the unit that would pass it fails with the limit in the message.
- CUDA graphs are off; a 7B decode step plus 8 depformer steps runs close to the 80 ms frame budget rather than well inside it.
- Past the 3000-position attention window (about four minutes) a sliding window stands in for the reference's ring cache. Behaviour matches the reference there, but long-input quality was judged by listening, not measured.

## Tests

`tests/unit_test/personaplex/` runs on CPU without weights: the delayed timeline, chunked Mimi against whole-sequence Mimi, the depformer, checkpoint weight routing, the model-runner hooks, the streaming codec stage, the checkpoint shim, preprocessing and voice unpacking, and request lowering. `test_session.py` drives a realtime call unit by unit through the LM session adapter and the model runner and checks every forward's input rows and every output frame against one offline request; `test_realtime.py` runs a call over the `/v1/realtime` WebSocket against a scripted pipeline.

`tests/test_model/test_personaplex_parity.py` (marker `accelerator`, one GPU) recreates the greedy parity numbers: it runs the port and the reference's `moshi.offline --greedy` on the reference checkout's `assets/test` recordings (`input_assistant.wav` with `NATF2`, `input_service.wav` with `NATM1` and the service prompt) and compares the reply audio frame by frame. Because the reference is not deterministic past its first near-tie, each case asserts at least 100 leading identical frames and matching text up to the first divergence; two more tests check that the port is identical across reruns and that `seed` makes sampling reproducible. The measured frame counts and texts are printed (`-s`). The reference pins an older torch, so it runs from its own interpreter:

```bash
PERSONAPLEX_REFERENCE_SOURCE=~/personaplex \
PERSONAPLEX_REFERENCE_PYTHON=~/personaplex/.venv/bin/python \
PERSONAPLEX_REFERENCE_DIR=~/.cache/personaplex-parity \
pytest tests/test_model/test_personaplex_parity.py -s
```

`PERSONAPLEX_REFERENCE_SOURCE` is the [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) checkout and `PERSONAPLEX_REFERENCE_PYTHON` an interpreter with its `moshi/` package installed. `PERSONAPLEX_REFERENCE_DIR` is optional and caches the reference outputs (`<case>/output.wav`, `output.json`) across runs; with the cache filled, the reference interpreter is not needed. `PERSONAPLEX_PARITY_CHECKPOINT` and `PERSONAPLEX_PARITY_STAGE_ARGS` (for example `--lm.engine.mem_fraction_static 0.5` on a 48 GB card) configure the port, `PERSONAPLEX_REFERENCE_REPO` the reference, and `PERSONAPLEX_PARITY_ATOL` the per-sample tolerance for an identical frame (default 1e-4).

`tests/test_model/test_personaplex_components.py` (marker `accelerator`, one GPU) checks the components on the public Moshi base, [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16), which shares Mimi and every dimension: Mimi encode and decode (whole and chunked, against the reference's streaming path, which is what it serves with), the summed input embeddings, and the depformer's teacher-forced logits for one frame in float32 and bf16, with and without the reference's ring-cache behaviour at the last step. TF32 is off on both sides, as in the reference's own tests. The reference side is `tests/test_model/personaplex_reference_dump.py`, which runs under the reference interpreter and writes one safetensors file; the test creates it when `PERSONAPLEX_REFERENCE_DUMP` does not exist yet:

```bash
PERSONAPLEX_REFERENCE_SOURCE=~/personaplex \
PERSONAPLEX_REFERENCE_PYTHON=~/personaplex/.venv/bin/python \
PERSONAPLEX_REFERENCE_DUMP=~/.cache/personaplex-parity/moshi_base_reference.safetensors \
pytest tests/test_model/test_personaplex_components.py -s
```

`PERSONAPLEX_MOSHI_BASE` overrides the base checkpoint (a directory or repo id).
