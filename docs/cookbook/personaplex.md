# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): a Helium temporal transformer reads, every 80 ms, one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next text token, while a 6-layer depth transformer spells out its next 8 codes. A voice prompt and a `<system>` role prompt set the persona. The model decides for itself when to speak; there is no VAD.

SGLang-Omni currently serves it as an **offline** pipeline: one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same duration out. Live duplex sessions over `/v1/realtime` are tracked in [#1909](https://github.com/sgl-project/sglang-omni/issues/1909).

## Prerequisites

The checkpoint is gated; accept the license on the model page and log in (`hf auth login`). It ships the 7B weights, the Mimi codec, the SentencePiece text model and the packaged voices:

```bash
hf download nvidia/personaplex-7b-v1
```

`voices.tgz` is unpacked into `voices/` next to the checkpoint on first use, or into the temp directory when the checkpoint folder cannot be written (permissions, a read-only mount). Recorded voice prompts (`--voice some.wav`) additionally need `pip install pyloudnorm` for the reference's loudness normalisation; the packaged `.pt` voices do not.

Everything runs on one GPU. Measured on an H200 with the offline example, the LM engine reserves `mem_fraction_static=0.3` (bf16 weights are 15.4 GB; the KV pool takes the rest) and the two Mimi instances need well under 1 GB each. Lower `--lm.engine.mem_fraction_static` on smaller cards.

## Running the offline example

```bash
python examples/run_personaplex.py \
  --model-path nvidia/personaplex-7b-v1 \
  --audio /path/to/caller.wav \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --out reply.wav
```

The example builds `PersonaPlexPipelineConfig`, starts five stages with `MultiProcessPipelineRunner` (preprocessing, Mimi encode, the LM engine, text decode, streaming code2wav), sends one request and writes the reply as 16-bit PCM at 24 kHz. Pick the GPU with `CUDA_VISIBLE_DEVICES`.

Input conventions:

- The recording is resampled to 24 kHz and **channel 0 is used**; it is zero-padded to a multiple of 1920 samples (80 ms).
- The reply is as long as the input, offset by one frame: the model answers while it listens, so leave silence after the caller's last words if you want a full answer.
- Prompt plus reply must fit the LM context, 8192 positions by default (about 10.8 minutes). A longer recording is rejected with an error naming the limit; raise it with `--lm.engine.context_length`. At the default `mem_fraction_static=0.3` the KV pool holds about 55k positions (73 minutes), and Mimi encodes the whole recording at once, which peaks at about 1.7 GiB of GPU memory per minute of audio.
- The reply text is the model's inner monologue with the frame markers (`PAD`, `EPAD`, `BOS`, `EOS`) removed.

## Serving over HTTP

```bash
python -m sglang_omni.cli serve --model-path nvidia/personaplex-7b-v1 --port 8000
```

Send the caller recording as the `prompt` of `POST /generate` (a path, URL or `data:` URI; set `"return_logprob": false`), or as the single entry of `audios` in `POST /v1/chat/completions` with `"modalities": ["text", "audio"]`. The reply audio comes back base64-encoded as WAV. PersonaPlex options that have no field of their own go under `stage_params`: `voice` and `text_prompt` under `preprocessing`, `audio_temperature`, `audio_top_k` and `seed` under `lm`.

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
| `text_prompt` (or `instructions`) | The role prompt, wrapped in `<system>` tags for the model. Default: the reference assistant prompt. |
| `temperature`, `top_k` | Text sampling; reference defaults 0.7 / 25, which the client's own filler values (1.0 / -1) do not override unless the caller sets them explicitly. `temperature=0` is greedy. |
| `audio_temperature`, `audio_top_k` | Code sampling in the depformer; reference defaults 0.8 / 250. |
| `seed` | Makes both draws reproducible (a child seed each for text and audio). |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length. |

## How it maps onto SGLang

The temporal transformer is Llama-shaped, so it runs as SGLang's `LlamaForCausalLM` with paged KV: interleaved RoPE (`rope_is_neox_style=False`), RMSNorm with the checkpoint's `1e-8` epsilon, SwiGLU with hidden size 11264, and a 3000-position sliding window in place of the reference ring cache. A per-request *timeline* lays the delayed streams out as a table: the whole prompt (voice, half a second of silence, the role text one token per frame, silence again) is one prefill of pre-fused embeddings, and each caller frame is one decode step. The text token is sampled before the post hook, the depformer spells the frame's codes from it in the hook, and those codes are both the next row's input and, one position later, a finished frame streamed to the Mimi decoder.

## Known limitations

- Offline, one request at a time (`max_running_requests=1`).
- CUDA graphs are off; a 7B decode step plus 8 depformer steps runs close to the 80 ms frame budget rather than well inside it.
- Inputs longer than the model's 3000-position context (about four minutes including the prompt) attend through the sliding window exactly as the reference does, but were not evaluated.
- With a base Moshi checkpoint (`kyutai/moshiko-pytorch-bf16`, which this pipeline also loads), the reference's ring cache drops its oldest slot when exactly full, so its last depformer step ignores step 0; this port attends to all steps, as training did. PersonaPlex runs 16 depformer steps, so its 8 agent codebooks never hit that case.

## Tests

`tests/unit_test/personaplex/` runs on CPU without weights: the delayed timeline, chunked Mimi against whole-sequence Mimi, depformer weight slicing and teacher forcing, checkpoint weight routing, the model-runner hooks, the streaming codec stage, the checkpoint shim, preprocessing and voice unpacking, and request lowering.
