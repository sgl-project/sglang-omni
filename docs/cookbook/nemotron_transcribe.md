# Nemotron Transcribe

[Nemotron Transcribe 30B-A4B](https://huggingface.co/nvidia/Nemotron-3.5-Transcribe-30B-A4B) serves whole-file transcription through `/v1/audio/transcriptions`. A native PyTorch frontend, speech transformer, and Sortformer speaker-fusion encoder produce audio embeddings for SGLang's NemotronH decoder. The serving path does not require NeMo.

This integration accepts consolidated NeMo SpeechLM checkpoints with the `NeMoSpeechLMForConditionalGeneration` architecture, a NemotronH `llm_config`, and `perception` and `pe_encoder_config` sections. The checkpoint directory must contain its tokenizer files and model weights. The loader creates a small configuration shim and reads the original weights without copying them.

## Serve a checkpoint

Install SGLang-Omni following [Installation](../get_started/installation.md), then download the checkpoint:

```bash
hf download nvidia/Nemotron-3.5-Transcribe-30B-A4B
```

The BF16 30B-A4B checkpoint was validated on one H100 80 GB. Leave memory for audio encoding and graph workspaces in addition to the weights and KV cache:

```bash
sgl-omni serve \
  --model-path nvidia/Nemotron-3.5-Transcribe-30B-A4B \
  --model-name nemotron-transcribe \
  --mem-fraction-static 0.85 \
  --port 8000
```

On a fresh install, the first launch also builds FlashInfer kernels, which can take longer than the default 600 second startup timeout. Set `SGLANG_OMNI_STARTUP_TIMEOUT=1800` for that launch.

The pipeline runs one ASR stage. It accepts uploads up to one hour, resamples to mono 16 kHz, and keeps each upload as one decoder request. Measured transcripts are the short clips in [Short-clip validation](#short-clip-validation): 4.352 s and 5.855 s. A one-hour recording can repeat sentences until the 16384-token response budget. The HTTP status stays 200. `json` and `verbose_json` then include `"finish_reason": "length"`. `text` sets `X-Finish-Reason: length`. A stream puts the same reason on `transcript.text.done`. The encoder processes the recording in checkpoint-defined windows with left and right context; the default checkpoint uses 40-second core windows.

| Setting | Default | Purpose |
| --- | --- | --- |
| `--asr.engine.max_running_requests` | `64` | Maximum concurrent decoder requests |
| `--asr.factory.encoder_max_batch_size` | `8` | Maximum waveforms per encoder invocation |
| `--asr.factory.context_length` | `73728` | Decoder context budget in tokens |
| `--asr.factory.max_new_tokens` | `16384` | Response token budget; `length` means the transcript stopped there |
| `--asr.factory.chunked_prefill_size` | `4096` | Decoder prefill chunk in tokens |
| `--asr.factory.mm_embedding_cache_size_bytes` | `0` | Audio embedding cache capacity |
| `--asr.factory.compile_perception` | `false` | Compile encoder transformer layers |
| `--asr.factory.perception_cuda_graphs` | `false` | Capture encoder graphs; requires compilation |

Decoder CUDA graphs are enabled. Decoder compilation is unsupported in this integration; use encoder compilation to reduce launch overhead. Encoder batch size bounds activation memory separately from decoder concurrency. For longer audio or a smaller GPU, reduce the encoder batch size and choose a memory fraction that still fits the model weights.

## Transcribe an upload

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=nemotron-transcribe \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

The default instruction asks for a verbatim transcript in the spoken language. Supply `prompt` to replace that instruction:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=nemotron-transcribe \
  -F file=@recording.wav \
  -F 'prompt=Produce a verbatim transcript of the audio.' \
  -F response_format=text
```

Set `stream=true` to receive text deltas as Server-Sent Events. The server encodes the uploaded audio before generating text; this is not live PCM streaming. Word timestamps, speaker-attributed output, translation, speculative decoding, and `/v1/realtime` are outside this integration.

## Compile the encoder

Enable compilation and encoder CUDA graphs independently of decoder graphs:

```bash
SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1 \
SGLANG_FLASHINFER_AUTOTUNE_CACHE=0 \
sgl-omni serve \
  --model-path nvidia/Nemotron-3.5-Transcribe-30B-A4B \
  --model-name nemotron-transcribe \
  --mem-fraction-static 0.85 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --asr.factory.encoder_max_batch_size 32 \
  --asr.factory.request_build_max_workers 16 \
  --asr.factory.request_build_max_pending 256 \
  --asr.factory.compile_perception true \
  --asr.factory.perception_compile_mode default \
  --asr.factory.perception_cuda_graphs true
```

The first requests compile kernels and capture encoder shapes. Exclude this warmup from throughput measurements. Single-request serving uses static encoder compilation; concurrent serving uses dynamic batch shapes. Explicit encoder graphs use private pools and cache at most four shapes per encoder. Additional shapes run without capturing more graphs. `default` is the pipeline default and skips Inductor's kernel search. `max-autotune-no-cudagraphs` times candidates on the serving GPU and keeps graph ownership outside Inductor.

For c=1, set both decoder limits to `1`. For long recordings, start with the default encoder batch cap rather than this profile's cap of `32`.

## Validate the integration

The unit suite covers pipeline registration, checkpoint conversion, tensor mapping, placeholder counts, request validation, text streaming, mel features, rotary positions, encoder padding, and graph replay across changed inputs.

```bash
pytest -q tests/unit_test/nemotron_transcribe
```

CPU tests run without checkpoint downloads. CUDA tests skip when CUDA is unavailable.

RTFx is successful audio seconds divided by elapsed wall time, excluding warmup. Short-clip throughput and normalized transcript agreement do not establish WER on a representative evaluation corpus.

## Benchmark on a corpus

Use the shared ASR benchmark against the server name from the launch command:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path nemotron-transcribe \
  --concurrencies 1,16 --repeats 3 --warmup \
  --disable-resource-monitor
```

This reports corpus WER, RTFx, and request latency. The resource monitor is disabled in this example to keep GPU sampling out of timed runs. Corpus WER results for this checkpoint are not included in the short-clip validation.

## Short-clip validation

On 2026-10-27, the default launch above ran on one H100 80 GB with BF16 and SGLang 0.5.22, keeping 64 decoder slots with encoder compilation and encoder graphs disabled.

| Client concurrency | Median RTFx | RTFx range | Median p50 | Median p90 |
| --- | --- | --- | --- | --- |
| 1 | 23.40 | 23.31 to 23.41 | 0.220 s | 0.229 s |
| 16 | 102.17 | 100.22 to 104.91 | 0.716 s | 0.999 s |

Each concurrency had three rounds of 256 measured requests on two English clips lasting 4.352 s and 5.855 s, with warmup requests excluded. All 1,536 measured requests succeeded with matching transcripts. These measurements cover short clips and do not establish long-audio throughput or corpus WER.
