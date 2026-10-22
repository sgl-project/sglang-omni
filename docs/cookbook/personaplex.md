# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): every 80 ms it reads one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next frame. A voice prompt and a `<system>` role prompt set the persona, and the model decides for itself when to speak; there is no VAD.

SGLang-Omni serves it as an **offline** pipeline: one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same length out. Live duplex sessions over `/v1/realtime` are tracked in [#1909](https://github.com/sgl-project/sglang-omni/issues/1909).

## Prerequisites

The checkpoint is gated; accept the license on the model page and log in (`hf auth login`), then:

```bash
hf download nvidia/personaplex-7b-v1
```

It ships the 7B weights, the Mimi codec, the SentencePiece text model and `voices.tgz`, which is unpacked into `voices/` next to the checkpoint on first use, or into the temp directory when that folder cannot be written. Recorded voice prompts (`--voice some.wav`) also need `pip install pyloudnorm`; the packaged `.pt` voices do not.

Everything runs on one GPU. On an H200 the LM engine reserves `mem_fraction_static=0.3` and the two Mimi instances stay under 1 GB each. This value is a fraction of total device memory, so a smaller-memory device can need a higher fraction to fit the model weights.

The Mimi encoder stage loads only its encoding path, and Code2Wav loads only its
decoding path. Both retain the codebooks and quantizer projections their path
needs, reducing the combined FP32 codec weights and buffers by about 302 MiB.

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

For Intel GPUs, follow the [PersonaPlex XPU recipe](../get_started/installation_xpu.md#personaplex-speech-to-speech-single-xpu).

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
| `temperature`, `top_k` | Text sampling, defaults 0.7 / 25. `temperature=0` is greedy. |
| `audio_temperature`, `audio_top_k` | Code sampling in the depformer; defaults 0.8 / 250. |
| `seed` | Makes both draws reproducible (a child seed each for text and audio). |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length. |

### Stage sampling

Fields left out of `stage_sampling.lm` keep the PersonaPlex text defaults.
For example, `{"seed": 7}` keeps temperature 0.7 and top-k 25, while
`{"temperature": 1.0, "top_k": -1}` samples at temperature 1.0 with no top-k
filtering. The same holds for `SamplingParams` objects passed to the Python
client: `SamplingParams(seed=7)` in `stage_sampling.lm` sets only the seed.

Text sampling uses `stage_sampling.lm`, then `stage_params.lm`, then top-level
request values. Stage sampling also takes precedence for `seed`.
`top_p`, `min_p`, and `repetition_penalty` apply to the text sampler only;
`audio_temperature` and `audio_top_k` continue to control the audio sampler.
The fixed caller-frame budget still determines the number of generated frames.


## Known limitations

- Offline, one request at a time by default (`max_running_requests=1`). If you raise `--lm.engine.max_running_requests`, the requests in a batch share one depformer pass per frame, and each seeded request still draws from its own generator. A seeded reply repeats exactly only with one request in flight, because batched kernels can round differently.
- On CUDA the backbone's decode step replays a CUDA graph (`--lm.engine.disable_cuda_graph true` turns it off); the 8 depformer steps still run eagerly, so a frame runs close to the 80 ms budget rather than well inside it.
- Decode feedback stays on device. On CUDA, `--lm.factory.enable_async_decode true` launches the next decode before collecting the previous result. The eager Depformer can delay frame delivery by one decode interval, so synchronous decode remains the default. Other platforms, requests needing host token history for sampling, and requests returning log probabilities use the synchronous path. When any request reaches its caller-frame budget, the whole batch uses synchronous decode for that step.
- Streaming Mimi decode follows `--code2wav.factory.chunk_ramp_frames` (default `[1, 4]`: the first frame alone, then four at a time; a finished stream flushes its shorter remainder). Ready requests whose chunks share a width decode in one batched step of up to `max_batch_size` (default 8), each at its own decode position. `num_decode_slots` resident decode states default to the LM's effective `max_running_requests`, and a smaller explicit value is rejected; a stream that finds every slot taken while another stream's tail drains waits for the next free slot, so sustained overload is not bounded by this queue. On CUDA, graphs cover every batch size up to `min(max_batch_size, num_decode_slots)` and every chunk width up to `max_graph_frames` (the widest ramp entry unless set; `0` turns the graphs off), 8 × 4 = 32 graphs at LM concurrency 16; other shapes and platforms decode eagerly with the same states.
- On CUDA, streaming Mimi decode uses a dedicated stream. The stage accepts CPU codes only; transfers and decode-slot reset/replay stay on that stream, so waveform collection does not drain queued LM forwards. Other devices retain their normal execution context.
- The temporal attention window follows the streaming ring, including the masked oldest slot once its 3000-position cache fills. Boundary tests check this rule; they do not measure long-input audio quality.
- On a CUDA platform with resolved `device=cuda`, `page_size=1` and radix caching disabled (the default), window KV releases expired positions for reuse. Steady decode holds about 3.1k positions (1.5 GiB of BF16 KV) per long request with the default eviction interval; prefill and retract replay can temporarily hold more. CPU stages, other platforms, page sizes and radix-enabled configurations use the full KV pool. Context and request-length limits remain unchanged, and returning slots does not shrink the preallocated pool. Set `--lm.engine.disable_hybrid_swa_memory true` to use the full pool explicitly.

## Tests

`tests/unit_test/personaplex/` runs on CPU without weights: the delayed timeline, chunked Mimi against whole-sequence Mimi, the depformer, checkpoint weight routing, the model-runner hooks, the streaming codec stage, the checkpoint shim, preprocessing and voice unpacking, and request lowering.

```bash
pytest tests/unit_test/personaplex/ -q
```

### Reference parity

Two opt-in tests compare against an [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) checkout. Its `moshi` package pins `torch < 2.5`, so it needs its own environment, separate from the serving one:

```bash
git clone https://github.com/NVIDIA/personaplex.git /path/to/personaplex
uv venv /path/to/personaplex/.venv -p 3.12
uv pip install --python /path/to/personaplex/.venv/bin/python /path/to/personaplex/moshi/
```

Then run the tests from the serving environment. They need CUDA, skip unless both variables are set, and run one file per command:

```bash
export PERSONAPLEX_REFERENCE_SOURCE=/path/to/personaplex
export PERSONAPLEX_REFERENCE_PYTHON=/path/to/personaplex/.venv/bin/python
python -m pytest tests/test_model/test_personaplex_parity.py -v -s
python -m pytest tests/test_model/test_personaplex_components.py -v -s
```

- `test_personaplex_parity.py`: greedy replies to the reference's two test recordings must have the input's length, match it for at least the first 100 frames, and agree in text over that prefix; greedy and seeded replies must also repeat. **This is not full-output parity.**
- `test_personaplex_components.py`: Mimi, the input embeddings and the depformer logits on the public `kyutai/moshiko-pytorch-bf16` base. The last depformer step matches only with the reference's ring behavior emulated, which diagnoses a known difference rather than proving a match.

`PERSONAPLEX_PARITY_CHECKPOINT` and `PERSONAPLEX_MOSHI_CHECKPOINT` override the checkpoints, `PERSONAPLEX_STAGE_ARGS` forwards pipeline flags such as `'--lm.engine.mem_fraction_static 0.5'`, and `PERSONAPLEX_REFERENCE_REPO` sets the reference CLI's config lookup. Reference outputs are regenerated on every run under pytest's temporary directory.
