# Fun-ASR-Nano

[Fun-ASR-Nano](https://arxiv.org/abs/2509.12508) transcribes audio through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It runs one ASR stage on one accelerator and accepts clips up to 30 seconds.

## Prerequisites

Follow the [installation guide](../get_started/installation.md) or the [Apple Silicon installation guide](../other_platforms/installation_apple_silicon.md). Use the Hugging Face variant of the checkpoint shown below.

```bash
hf download FunAudioLLM/Fun-ASR-Nano-2512-hf
```

## Server Configuration

Start the server on port 8000.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --port 8000
```

### Ascend NPU

Prepare the runtime with the [NPU installation guide](../other_platforms/installation_npu.md), then start the server with the NPU configuration. Add `--model-path` to use a local checkpoint instead of the configured one, and `--asr.gpu 1` to select the second visible NPU.

```bash
sgl-omni serve \
  --config examples/configs/fun_asr_npu.yaml \
  --port 8000
```

## Transcribe Audio

Run this request from the repository root to transcribe the included English sample. If you installed from a package, [download the sample WAV](../get_started/quickstart.md#send-a-request) and replace `tests/data/query_to_cars.wav` with its path. The JSON response contains the transcript in `text`.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=FunAudioLLM/Fun-ASR-Nano-2512-hf \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

For the [shared Python example](../basic_usage/asr.md#basic-transcription), set `model` to `FunAudioLLM/Fun-ASR-Nano-2512-hf` and add `"language": "en"` to the form data for this sample.

## Stream Transcription

Use the [shared SSE example](../basic_usage/asr.md#stream-transcription) and set `model` to `FunAudioLLM/Fun-ASR-Nano-2512-hf`. Keep `language=en` for the sample. Once the full clip is uploaded, the server streams transcript text over Server-Sent Events.

## Request Parameters

See [ASR Model Usage](../basic_usage/asr.md#request-parameters) for the shared fields. Fun-ASR-Nano handles these fields as follows.

| Parameter | Fun-ASR-Nano behavior |
|---|---|
| `language` | `en`, `english`, and `英文` request English output. `zh`, `cn`, `chinese`, and `中文` use the same generic transcription prompt as an omitted value. Other values are passed through to the model's language prompt. |
| `prompt` | Supply a comma-separated list of names or terms to bias transcription toward those words. |
| `response_format` | Use `json`, `text`, or `verbose_json` for a complete response. `verbose_json` puts a non-empty transcript in one segment spanning the clip. Subtitle formats require model-derived timestamps and return HTTP 400 for Fun-ASR-Nano. |
| `temperature` | Defaults to `0.0` (greedy decoding). Torch/MPS requires this value. |
| `max_new_tokens` | Defaults to a budget scaled to the clip duration. With the stock configuration, explicit values range from 1 to 200. |

## Apple Silicon

Follow the [Apple Silicon installation guide](../other_platforms/installation_apple_silicon.md) for environment activation and FFmpeg library setup.

### Torch/MPS

Use the [server command above](#server-configuration) with the official BF16 checkpoint. This path processes one request at a time and queues additional requests. Use `temperature=0` with JSON or SSE transcription.

## Speculative Decoding

On one CUDA GPU, Fun-ASR-Nano serves with its DFlash draft, [`wilsonzheng0327/Fun-ASR-Nano-dflash`](https://huggingface.co/wilsonzheng0327/Fun-ASR-Nano-dflash), by default, at the revision `sglang_omni/speculative/draft_registry.py` pins. This holds when the model path is `FunAudioLLM/Fun-ASR-Nano-2512-hf`, with or without an `@<revision>` pin, or a Hugging Face cache snapshot of it. Apple Silicon, Ascend NPUs, a stage with `tp_size` above 1, a copy outside the Hugging Face cache and the [two-replica H100 MPS profile](../basic_usage/mps_dp.md) serve without it. To serve without the draft:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --asr.engine.disable_speculative_decoding true \
  --port 8000
```

With the draft, `temperature` and `repetition_penalty` work, torch compile and async decode are off, and the server starts in about 30 seconds. Send `return_spec_tokens_details=true` to get the acceptance length in `sglext.spec_tokens_details`. See the [DFlash evaluation](https://github.com/sgl-project/sglang-omni/pull/3453) for accuracy, throughput, and time to first text at different concurrency levels.

## Benchmarking

Follow the [ASR benchmark runbook](../developer_reference/asr_bench.md) for the pinned model and dataset revisions. Use the Fun-ASR-Nano checkpoint for `--model-path` in the [shared SeedTTS example](../basic_usage/asr.md#benchmarking).

## Benchmark Results

See the [recorded H100 80 GB results](https://github.com/sgl-project/sglang-omni/blob/eeaf0390f5bcd4cd335217622db253df03c7fc7a/docs/cookbook/fun_asr.md#benchmark-results) for the SeedTTS English and Chinese sweeps and the queue settings used in that run.

## Known Limitations

- Split recordings longer than 30 seconds before uploading. The server resamples audio to 16 kHz before feature extraction.
- Use the transcription route. Fun-ASR-Nano returns HTTP 400 on [the speech translation route](../basic_usage/audio_translations.md).
- Use BF16 weights. FP16 can overflow in the audio adaptor, and Torch/MPS requires the unquantized checkpoint.
