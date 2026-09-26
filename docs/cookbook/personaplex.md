# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): every 80 ms it reads one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next frame. A voice prompt and a `<system>` role prompt set the persona, and the model decides for itself when to speak; there is no VAD.

SGLang-Omni serves it as an **offline** pipeline: one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same length out. Live duplex sessions over `/v1/realtime` are tracked in [#1909](https://github.com/sgl-project/sglang-omni/issues/1909).

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

## Request parameters

| Parameter | Effect |
|---|---|
| `voice` | A packaged voice name (`NATF0..3`, `NATM0..3`, `VARF0..4`, `VARM0..4`), a `.pt` file, or a recording. Default `NATF2`; an empty string runs without a voice prompt. |
| `text_prompt` (or `instructions`) | The role prompt, wrapped in `<system>` tags. Default: the reference assistant prompt. |
| `temperature`, `top_k` | Text sampling; defaults 0.7 / 25, which the client's filler values (1.0 / -1) do not override unless set explicitly. `temperature=0` is greedy. |
| `audio_temperature`, `audio_top_k` | Code sampling in the depformer; defaults 0.8 / 250. |
| `seed` | Makes both draws reproducible (a child seed each for text and audio). |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length. |

Use `stage_params.lm` for exact text overrides, including `temperature: 1.0` and `top_k: -1`. The shared client still fills these values into `stage_sampling`, where PersonaPlex treats them as defaults rather than overrides. A non-null `stage_sampling.lm.seed` takes precedence over `stage_params.lm.seed` and the global seed, and seeds both text and audio draws.

## Known limitations

- Offline, one request at a time (`max_running_requests=1`).
- CUDA graphs are off; a 7B decode step plus 8 depformer steps runs close to the 80 ms frame budget rather than well inside it.
- The temporal attention window follows the streaming ring, including the masked oldest slot once its 3000-position cache fills. Boundary tests check this rule; they do not measure long-input audio quality.

## Tests

`tests/unit_test/personaplex/` runs on CPU without weights: the delayed timeline, chunked Mimi against whole-sequence Mimi, the depformer, checkpoint weight routing, the model-runner hooks, the streaming codec stage, the checkpoint shim, preprocessing and voice unpacking, and request lowering.

`tests/test_model/test_personaplex_parity.py` (marker `accelerator`, one GPU) recreates the greedy parity numbers: it runs the port and the reference's `moshi.offline --greedy` on the reference checkout's `assets/test` recordings (`input_assistant.wav` with `NATF2`, `input_service.wav` with `NATM1` and the service prompt) and compares the reply audio frame by frame. Because the reference is not deterministic past its first near-tie, each case first requires both replies to have exactly the input sample count, including the partial final frame, then asserts at least 100 leading identical frames and matching text up to the first divergence; two more tests check that the port is identical across reruns and that `seed` makes sampling reproducible. The measured frame counts and texts are printed (`-s`). The reference pins an older torch, so it runs from its own interpreter:

```bash
export PERSONAPLEX_REFERENCE_SOURCE=~/personaplex
export PERSONAPLEX_REFERENCE_REVISION=3428dfd95309a7f3c84fd93259ded0f810d1ff91
export PERSONAPLEX_REFERENCE_PYTHON=~/personaplex/.venv/bin/python
# Use an immutable snapshot, replacing the placeholder with its full commit SHA.
export PERSONAPLEX_PARITY_CHECKPOINT="nvidia/personaplex-7b-v1@<full-checkpoint-commit>"
export PERSONAPLEX_REFERENCE_DIR=~/.cache/personaplex-parity
pytest tests/test_model/test_personaplex_parity.py -s --junitxml=personaplex-parity.xml
```

`PERSONAPLEX_REFERENCE_SOURCE` is a clean [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) checkout at `PERSONAPLEX_REFERENCE_REVISION`. The reference subprocess imports `moshi/` from that checkout using its own interpreter. The revision above is the reference used for the boundary checks, not a claim about the version used for earlier reported measurements.

`PERSONAPLEX_PARITY_CHECKPOINT` accepts `repo@<full-commit-SHA>` or a local checkpoint directory. Both implementations receive the same local weights, tokenizer and voice file. `PERSONAPLEX_PARITY_STAGE_ARGS` configures the port (for example `--lm.engine.mem_fraction_static 0.5` on a 48 GB card); `PERSONAPLEX_PARITY_ATOL` is the per-sample tolerance (default 1e-4). `PERSONAPLEX_REFERENCE_REPO` only supplies the reference CLI's otherwise-unused config download; the weight paths are explicit.

Each reference cache contains `output.wav`, `output.json`, `manifest.json` and a log. The manifest records the reference revision, checkpoint/input/voice hashes, prompt, seed, command, Python, torch, CUDA and GPU. Reuse requires matching inputs and output hashes; stale or unversioned outputs are regenerated when the reference interpreter is available, otherwise the test fails with regeneration instructions. A valid cache can be used without the reference interpreter. To remeasure reference nondeterminism or a different runtime/GPU, use a fresh cache directory.

The short parity cases measure a matching prefix, not full-recording numerical equality or semantic quality. Complete audio agreement also requires complete text agreement. Keep the manifest, pytest output and JUnit report with any reported numbers.

`tests/test_model/test_personaplex_components.py` (marker `accelerator`, one GPU) checks the components on the public Moshi base, [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16), which shares Mimi and every dimension: Mimi encode and decode (whole and chunked, against the reference's streaming path, which is what it serves with), the summed input embeddings, and the depformer's teacher-forced logits for one frame in float32 and bf16, with and without the reference's ring-cache behaviour at the last step. TF32 is off on both sides, as in the reference's own tests. The reference side is `tests/test_model/personaplex_reference_dump.py`, which runs under the reference interpreter and writes one safetensors file; the test creates it when `PERSONAPLEX_REFERENCE_DUMP` does not exist yet:

```bash
# Reuse the reference checkout, revision and interpreter set above.
export PERSONAPLEX_MOSHI_BASE="kyutai/moshiko-pytorch-bf16@<full-checkpoint-commit>"
export PERSONAPLEX_REFERENCE_DUMP=~/.cache/personaplex-parity/moshi_base_reference.safetensors
pytest tests/test_model/test_personaplex_components.py -s --junitxml=personaplex-components.xml
```

The component dump uses the same cache checks, including the dump script hash, frame count, batch size and seed, in a sibling `.manifest.json` file. A manually produced dump without that manifest must be regenerated through the fixture.

### Coverage of reported results

| Result | Reproduction coverage |
|---|---|
| Assistant/service greedy prefixes | `test_greedy_matches_reference`; also checks full reply length. |
| Same-seed and greedy reruns | `test_seed_reproducibility`, `test_port_is_deterministic`. Cross-GPU equality still requires separate runs and saved outputs. |
| Mimi codes/decode, embeddings, depformer logits | `test_personaplex_components.py`; covers the base checkpoint and the documented step-7 ring difference. |
| Window fill/wrap and text/audio alignment | CPU tests in `test_sglang_model.py`, `test_request_builders.py`, and `test_model_runner.py`; these do not establish long-audio quality. |
| Five-minute per-round parity counts | Not covered by the short parity cases; needs a separate reference comparison with the exact repeated input and saved per-round results. |
| 704-second completion | Manual length check below; listening-based coherence is not an automated assertion. |
| HTTP versus offline audio | The HTTP example above exercises serving; numerical equality still needs decoded PCM and text compared against the same greedy offline input and options. |
| Recorded voice, base-model end-to-end, packaged voice versus WAV | Separate measurements are still needed with the exact voice assets and hashes; the component tests do not reproduce these claims. |

The following manual smoke test crosses the window boundary and the default request-length limit. It verifies completion length, not the historical per-round parity counts:

```bash
python - "$PERSONAPLEX_REFERENCE_SOURCE/assets/test/input_assistant.wav" <<'PYTHON'
import sys
import numpy as np
import soundfile as sf
waveform, rate = sf.read(sys.argv[1], always_2d=True)
for seconds in (300, 704):
    sf.write(f"caller-{seconds}.wav", np.resize(waveform[:, 0], seconds * rate), rate)
PYTHON
# CHECKPOINT_DIR is the resolved local snapshot used by the parity tests.
for seconds in 300 704; do
  python examples/run_personaplex.py --model-path "$CHECKPOINT_DIR" \
    --audio "caller-$seconds.wav" --voice NATF2 --greedy \
    --lm.engine.context_length 16384 \
    --out "reply-$seconds.wav" --out-text "reply-$seconds.txt"
done
python - <<'PYTHON'
import soundfile as sf
for seconds in (300, 704):
    reply = sf.info(f"reply-{seconds}.wav")
    assert reply.samplerate == 24000
    assert reply.frames == seconds * reply.samplerate
PYTHON
```

Record the Omni commit, checkpoint revision, runtime/GPU, full command, input hashes and output artifacts when publishing new measurements. Do not carry forward older numerical results after a window or alignment change without rerunning them.
