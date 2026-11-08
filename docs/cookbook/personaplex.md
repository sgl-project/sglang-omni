# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): every 80 ms it reads one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next frame. A voice prompt and a `<system>` role prompt set the persona, and the model decides for itself when to speak; there is no VAD.

SGLang-Omni serves it two ways: an **offline** pipeline (one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same length out) and a **session** variant that holds live full-duplex sessions over `/v1/realtime` ([#1909](https://github.com/sgl-project/sglang-omni/issues/1909)).

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

For Intel GPUs, follow the [PersonaPlex XPU recipe](../other_platforms/installation_xpu.md#personaplex-speech-to-speech-single-xpu).

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

## Native Full-Duplex Ability

PersonaPlex also serves live full-duplex voice sessions on `/v1/realtime` with its `session` variant. Launching, settings and the protocol are in [Full Duplex Interactive Model](../developer_reference/full_duplex_guide.md#native-full-duplex-personaplex).

### Duplex serving benchmark

The shared recorder supports `--profile personaplex`: 80 ms input units, mono 24 kHz PCM16 input and output, and continuous output. Dataset audio is resampled before sessions start; `input.pcm`, `input.wav` and the input manifest describe the actual transmitted audio. Replay checks one extra 80 ms output frame at the start of each nonempty call, including calls whose final input unit needs padding. The empty end-of-input unit contributes no audio.

For a single-call protocol check, start the session server above and pass a mono 16 kHz PCM16 WAV with trailing silence as `--audio`:

```bash
python -m benchmarks.eval.benchmark_duplex \
  --url ws://localhost:8000/v1/realtime \
  --profile personaplex --audio caller.wav --output personaplex-smoke \
  --server-revision "$SERVER_REVISION" --model nvidia/personaplex-7b-v1
python -m benchmarks.duplex.artifacts personaplex-smoke
```

Set `SERVER_REVISION` to the server checkout's full commit SHA. For an FDB v1.5 serving run, set `FDB_ROOT` to the dataset directory, set `FDB_REVISION` to its release or archive digest, and keep the requested concurrency within the server's `--max_sessions`:

```bash
python -m benchmarks.eval.benchmark_duplex_serving sweep \
  --dataset v1.5 --concurrency 1 --output personaplex-serving -- \
  --url ws://localhost:8000/v1/realtime --profile personaplex \
  --dataset-root "$FDB_ROOT" --dataset-revision "$FDB_REVISION" \
  --server-revision "$SERVER_REVISION" \
  --model nvidia/personaplex-7b-v1
```

The serving report counts input units at 24 kHz, measures lag from input-send completion, and excludes empty end-of-input units. Inspect `serving.json` and `SERVING.md` for the 80 ms serving deadline verdict. FDB reference export validates the transmitted 24 kHz PCM and packet times, then resamples the scoring audio to mono 16 kHz PCM16. The scoring window keeps the original input duration.

### Quality baseline

The [single-call quality baseline](../developer_reference/personaplex_quality_baseline.md) records the H100 experiment configuration, subset coverage and greedy comparison results.

Record FDB v1.0 and v1.5 at concurrency 1 with the same checkpoint, voice, role prompt and sampling configuration. Set `FDB10_ROOT` and `FDB15_ROOT` to the extracted datasets, and `FDB10_REVISION` and `FDB15_REVISION` to their archive digests. The following selection records three samples from each subset; the run manifests retain their IDs and source hashes.

```bash
python -m benchmarks.eval.benchmark_duplex_v15 record \
  --url ws://localhost:8000/v1/realtime --profile personaplex \
  --dataset-root "$FDB15_ROOT" --dataset-revision "$FDB15_REVISION" \
  --server-revision "$SERVER_REVISION" --model nvidia/personaplex-7b-v1 \
  --model-revision "$MODEL_REVISION" --concurrency 1 --max-per-subset 3 \
  --timeout 230 --output quality-v15
python -m benchmarks.eval.benchmark_duplex_v10 record \
  --url ws://localhost:8000/v1/realtime --profile personaplex \
  --dataset-root "$FDB10_ROOT" --dataset-revision "$FDB10_REVISION" \
  --server-revision "$SERVER_REVISION" --model nvidia/personaplex-7b-v1 \
  --model-revision "$MODEL_REVISION" --concurrency 1 --max-per-subset 3 \
  --timeout 230 --output quality-v10
FDB_SELECTION=()
for subset in user_interruption user_backchannel talking_to_other background_speech; do
  for sample in 1 2 3; do
    FDB_SELECTION+=(--only "$subset/$sample")
  done
done
python -m benchmarks.eval.benchmark_duplex_reference export "${FDB_SELECTION[@]}" \
  --run quality-v15 --dataset-root "$FDB15_ROOT" \
  --engine personaplex --out quality-v15-reference
python -m benchmarks.eval.benchmark_duplex_v10 reference-export \
  --run quality-v10 --dataset-root "$FDB10_ROOT" --out quality-v10-reference
```

Use the [FDB scoring commands](../developer_reference/full_duplex_guide.md) on these reference trees. Record the ASR checkpoint, evaluator revision, judge model and revision with the scores. Audio reconstruction uses receipt-time FIFO playout over the input duration, including initial silence and cropping audio at the observation window's end.

### Greedy session and offline comparison

For complete-reply equality, launch both servers with `--lm.engine.enable_deterministic_inference true`. Use greedy text and audio sampling on the session server, record a single call with the recorder above, then launch the offline server with `--session_reference true` from the same revision and checkpoint. Save the session's role prompt in `instructions.txt` and use the same voice on both routes:

```bash
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" --variant session --enable-realtime --port 8000 \
  --sampling.temperature 0 --sampling.audio_temperature 0 \
  --voice NATF2 --instructions "$(cat instructions.txt)" \
  --lm.engine.enable_deterministic_inference true

# Record the call, stop the session server, then launch the offline reference.
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" --port 8001 --session_reference true \
  --lm.engine.enable_deterministic_inference true
```

The offline reference processes the same caller PCM through its own request: Mimi encodes one 80 ms frame at a time, the LM generates one extra initial frame, and Mimi decodes two frames first followed by one frame per chunk. The returned waveform includes all generated frames and final caller-frame padding. This mode selects the session's codec chunk schedule, including when `code2wav.factory.chunk_ramp_frames` is set. The deterministic LM setting fixes attention reduction order while retaining CUDA graphs. Run the comparison at concurrency 1:

```bash
python -m benchmarks.eval.benchmark_personaplex_parity \
  --session-run personaplex-smoke --offline-url http://localhost:8001 \
  --offline-mode session-reference \
  --voice NATF2 --instructions instructions.txt --output greedy-parity \
  --server-revision "$SERVER_REVISION" --model nvidia/personaplex-7b-v1 \
  --model-revision "$MODEL_REVISION"
```

The command sends the recorded 24 kHz PCM to the offline endpoint as a WAV data URI. It saves the request, response, both audio outputs and `parity.json`, including the first differing sample and SHA-256 of each PCM stream. It also records exact text equality and the elapsed HTTP request time in seconds. Its exit status is zero when the complete PCM replies are byte-identical and the text replies match. `--offline-mode` records the server mode in the report; the serving command selects that mode. The report also compares the prefix covering the caller's duration.

Default offline serving generates one frame per caller frame and trims output to the caller's duration. To measure that contract, omit `--session_reference true` at launch and use `--offline-mode default` in the comparison. The [quality baseline report](../developer_reference/personaplex_quality_baseline.md) retains those measurements separately from the session reference experiment.

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


## Opt-in FP8 inference

BF16 is the default. Two LM engine flags switch the backbone to FP8, separately or together; the text head, embeddings, Depformer and Mimi keep their precision.

| Flags | Backbone weights | KV positions in 8 GiB | Audio s/s, C1 / C8 | First audio, C1 / C8 |
|---|---:|---:|---:|---:|
| None (BF16) | 13.15 GB | 16,384 | 6.95 / 44.40 | 58.2 / 172.7 ms |
| `--lm.engine.quantization fp8` | 6.58 GB | 16,384 | 8.23 / 50.66 | 66.5 / 179.1 ms |
| `--lm.engine.kv_cache_dtype fp8_e4m3` | 13.15 GB | 32,768 | 6.82 / 44.68 | 59.8 / 171.8 ms |
| Both | 6.58 GB | 32,768 | 7.89 / 50.14 | 69.3 / 176.6 ms |

*Offline requests on one H100 with FA3, SGLang 0.5.22, `--lm.engine.max_running_requests 8`.*

FP8 weights are the speed option: they halve the weight memory and raise throughput by 14–18%, at 6–10 ms more first-audio latency. On A100 they run as weight-only Marlin and raise C1 throughput by 27%. FP8 KV is the capacity option: a KV budget holds twice the positions, with no speed gain. On 416 English prompts at C8, every FP8 option stayed within 0.10 UTMOS and 1 WER point of BF16.

These results cover offline requests, not full-duplex sessions. FP8 KV was tested only on H100 with FA3, and its scales are fixed at 1.0.

## Known limitations

### All platforms

- Offline, one request at a time by default (`max_running_requests=1`). If you raise `--lm.engine.max_running_requests`, the requests in a batch share one depformer pass per frame, and each seeded request still draws from its own generator. A seeded reply repeats exactly only with one request in flight, because batched kernels can round differently.
- Streaming Mimi decode follows `--code2wav.factory.chunk_ramp_frames` (default `[1, 4]`: the first frame alone, then four at a time; a finished stream flushes its shorter remainder). Ready requests whose chunks share a width decode in one batched step of up to `max_batch_size` (default 8), each at its own decode position. `num_decode_slots` resident decode states default to the LM's effective `max_running_requests`, and a smaller explicit value is rejected; a stream that finds every slot taken while another stream's tail drains waits for the next free slot, so sustained overload is not bounded by this queue.
- The temporal attention window follows the streaming ring, including the masked oldest slot once its 3000-position cache fills. Boundary tests check this rule; they do not measure long-input audio quality.
- Radix caching stays off: the stage refuses `--lm.engine.disable_radix_cache false` at startup. The prompt ids do not identify the voice, so a radix hit could reuse another voice's prompt KV.

### CUDA

- On CUDA, the Depformer's RMSNorm and SiLU-gate run as fused Triton kernels whose outputs match the eager operations bit for bit; other devices use the eager operations.
- On CUDA the backbone's decode step and all 8 Depformer steps replay CUDA graphs. At startup, offline serving captures the default audio-sampling layout for every configured batch bucket through the smallest bucket that covers the effective `max_running_requests`; session serving captures the same buckets with the deployment's resolved audio sampling. Other sampling layouts and batch shapes run eagerly. `--lm.engine.disable_cuda_graph true` disables both backbone and Depformer graphs, while `--lm.factory.depformer_cuda_graph_batch_sizes '[]'` disables only Depformer graphs. Seeded requests preserve their per-request RNG draws.
- Decode feedback stays on device. On CUDA, `--lm.factory.enable_async_decode true` launches the next decode before collecting the previous result and can add 16–21% throughput with identical audio; synchronous decode remains the default. Other platforms, requests needing host token history for sampling, and requests returning log probabilities use the synchronous path. When any request reaches its caller-frame budget, the whole batch uses synchronous decode for that step.
- On CUDA, Mimi decode graphs cover every batch size up to `min(max_batch_size, num_decode_slots)` and every chunk width up to `max_graph_frames` (the widest ramp entry unless set; `0` turns the graphs off), 8 × 4 = 32 graphs at LM concurrency 16; other shapes and platforms decode eagerly with the same states.
- On CUDA, streaming Mimi decode uses a dedicated stream. The stage accepts CPU codes only; transfers and decode-slot reset/replay stay on that stream, so waveform collection does not drain queued LM forwards. Other devices retain their normal execution context.
- On a CUDA platform with resolved `device=cuda` and `page_size=1`, window KV releases expired positions for reuse. Steady decode holds about 3.1k positions (1.5 GiB of BF16 KV) per long request with the default eviction interval; prefill and retract replay can temporarily hold more. CPU stages, other platforms and other page sizes use the full KV pool. Context and request-length limits remain unchanged, and returning slots does not shrink the preallocated pool. Set `--lm.engine.disable_hybrid_swa_memory true` to use the full pool explicitly.

### Ascend NPU

- On Ascend NPU the backbone graph is off by default; pass `--lm.engine.disable_cuda_graph false` to replay the backbone's NPU graph.
- On Ascend NPU, the platform enables the same Mimi graph ladder; the capture temporarily disables internal formats so Mimi's convolutions run through [aclnn](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/aolapi/context/ops-math/op_api_list.md).

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
