# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): every 80 ms it reads one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next frame. A voice prompt and a `<system>` role prompt set the persona, and the model decides for itself when to speak; there is no VAD.

SGLang-Omni serves it as an **offline** pipeline: one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same length out. Live duplex sessions over `/v1/realtime` are tracked in [#1909](https://github.com/sgl-project/sglang-omni/issues/1909).

## Prerequisites

The checkpoint is gated; accept the license on the model page and log in (`hf auth login`), then:

```bash
hf download nvidia/personaplex-7b-v1
```

It ships the 7B weights, the Mimi codec, the SentencePiece text model and `voices.tgz`, which is unpacked into `voices/` next to the checkpoint on first use, or into the temp directory when that folder cannot be written. Recorded voice prompts (`--voice some.wav`) also need `pip install pyloudnorm`; the packaged `.pt` voices do not.

Everything runs on one accelerator. On an H200 the LM engine reserves `mem_fraction_static=0.3` and the two Mimi instances stay under 1 GB each. This value is a fraction of total device memory, so a smaller-memory device can need a higher fraction to fit the model weights.

## Running the offline example

Five stages run under `MultiProcessPipelineRunner` (preprocessing, Mimi encode, the LM engine, text decode, streaming code2wav). Pick the CUDA GPU with `CUDA_VISIBLE_DEVICES`, or the Intel XPU with `ZE_AFFINITY_MASK`; stage settings take the same dotted flags as `serve`. Input conventions:

- The recording is resampled to 24 kHz and **channel 0 is used**.
- The reply is exactly as long as the input, offset by one frame: the model answers while it listens, so leave silence after the caller's last words if you want a full answer.
- Prompt plus reply must fit the LM context, 8192 positions by default (about 10.8 minutes); a longer recording is rejected with the limit in the message and needs `--lm.engine.context_length`.
- The reply text is the model's inner monologue with the frame markers (`PAD`, `EPAD`, `BOS`, `EOS`) removed.

### CUDA GPU

```bash
python examples/run_personaplex.py \
  --model-path nvidia/personaplex-7b-v1 \
  --audio /path/to/caller.wav \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --out reply.wav
```

### Intel GPU

PersonaPlex was validated on one 24 GB Intel Arc Pro B60 with `--lm.engine.mem_fraction_static 0.70`:

```bash
python examples/run_personaplex.py \
  --model-path nvidia/personaplex-7b-v1 \
  --audio /path/to/caller.wav \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --out reply.wav \
  --lm.engine.mem_fraction_static 0.70
```

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

### Explicit stage sampling

HTTP chat and rollout requests preserve which fields were provided in
`stage_sampling.lm`. For example, `{"seed": 7}` keeps the PersonaPlex text
defaults (temperature 0.7, top-k 25), while
`{"temperature": 1.0, "top_k": -1}` explicitly selects temperature 1.0 and
disables top-k filtering. In-process stage `SamplingParams` objects are complete configurations:
all serialized fields are explicit. Thus `SamplingParams(seed=7)` in
`stage_sampling.lm` selects temperature 1.0 and top-k -1 as well as seed 7.
To keep PersonaPlex text defaults, omit `stage_sampling` and set the seed in
the top-level `sampling=SamplingParams(seed=7)` instead. Top-level client
filler handling is unchanged.

Text sampling uses `stage_sampling.lm`, then `stage_params.lm`, then top-level
request values. Stage sampling also takes precedence for `seed`.
`top_p`, `min_p`, and `repetition_penalty` apply to the text sampler only;
`audio_temperature` and `audio_top_k` continue to control the audio sampler.
The fixed caller-frame budget still determines the number of generated frames.


## Known limitations

- Offline, one request at a time (`max_running_requests=1`).
- CUDA graphs are off; a 7B decode step plus 8 depformer steps runs close to the 80 ms frame budget rather than well inside it.
- The temporal attention window follows the streaming ring, including the masked oldest slot once its 3000-position cache fills. Boundary tests check this rule; they do not measure long-input audio quality.

## Tests

`tests/unit_test/personaplex/` runs on CPU without weights: the delayed timeline, chunked Mimi against whole-sequence Mimi, the depformer, checkpoint weight routing, the model-runner hooks, the streaming codec stage, the checkpoint shim, preprocessing and voice unpacking, and request lowering.

```bash
pytest tests/unit_test/personaplex/ -q
```

Reference comparisons and reproducibility checks are evaluation scripts under
`benchmarks/eval/`, separate from the unit suite. See the
[PersonaPlex evaluation guide](../../benchmarks/eval/personaplex.md) for checkpoint,
reference-environment and output setup. The greedy comparison checks a matching
prefix, not full-output equality; the component comparison includes a diagnostic
for the reference depformer ring behavior.
