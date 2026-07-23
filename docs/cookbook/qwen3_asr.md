# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH=$(hf download Qwen/Qwen3-ASR-1.7B --revision "${MODEL_REVISION}")
```

Source installations also need the system NUMA runtime used by `sglang-kernel`:

```bash
apt-get update
apt-get install -y libnuma1
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

### RTX 4090 (24 GB)

The validated consumer profile uses BF16, at most 16 running requests, CUDA
Graph batches through 16, `mem_fraction_static=0.65`, FlashInfer language-model
attention, Triton multimodal attention, and no `torch.compile`.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

The equivalent settings can be overridden explicitly:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000 \
  --mem-fraction-static 0.65 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "language": "en",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `en` | Language hint; `zh`/`cn` select Chinese, other values use English prompting |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |
| `max_new_tokens` | integer | stage default | Maximum generated transcription tokens |
| `stream` | boolean | `false` | Return transcript events over SSE |

`verbose_json` is accepted, but currently returns the same minimal JSON shape as `json`:
`{"text": "..."}`.

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. It defaults to
`--model-path Qwen/Qwen3-ASR-1.7B`; the shared request and metric logic lives in
`benchmarks.tasks.asr` and also supports Fun-ASR through `--model-path`.

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies 1,2,4,8,16,32 \
  --repeats 3 --warmup \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --mm-attention-backend triton_attn \
  --cuda-graph --no-torch-compile \
  --max-running-requests 16 \
  --mem-fraction-static 0.65
```

The result JSON includes the fixed revisions and normalization, repository and
dependency fingerprints, the launch configuration, complete sample counts,
latency/RTF/throughput, CPU use, power, and peak/steady GPU memory.

For functional paths and a mixed 30-minute soak:

```bash
python -m benchmarks.eval.benchmark_asr_stability \
  --port 8000 \
  --duration-s 1800 \
  --concurrencies 1,4,8,16 \
  --output asr_stability_results.json
```

The ASR CI gate runs Fun-ASR-Nano on this same benchmark entry point
(`tests/test_model/test_asr_ci_fun_asr.py`). Qwen3-ASR remains the
transcriber for the TTS and talker WER stages.

## RTX 4090 Validation Results

Validated on Linux with one RTX 4090 (24,564 MiB, SM89), driver 580.159.04,
CUDA 13.0, PyTorch 2.11.0, SGLang 0.5.12.post1, Transformers 5.6.0, and a
450 W power limit. The validation worktree was based on commit
`aa6f92c30e5279ed831776d0d6649344df3a94cb` with this change set applied.
SeedTTS used revision
`27f4c1adee83b5b29b7c4b375f6b976324bda308`; the model used revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`.

Full SeedTTS EN (1088 clips, three measured repeats after one warmup):

- Concurrency 1: 10.527 samples/s, 0.095 s mean latency, 0.119 s p95,
  0.0205 mean RTF, worst corpus WER 0.0123.
- Concurrency 4: 28.258 samples/s, 0.141 s mean latency, 0.181 s p95,
  0.0306 mean RTF, worst corpus WER 0.0122.
- Concurrency 8: 45.102 samples/s, 0.177 s mean latency, 0.233 s p95,
  0.0382 mean RTF, worst corpus WER 0.0123.
- Concurrency 16: 65.428 samples/s, 0.243 s mean latency, 0.321 s p95,
  0.0526 mean RTF, worst corpus WER 0.0123.
- Concurrency 32 (queueing stress above the 16-request admission limit):
  65.747 samples/s, 0.482 s mean latency, 0.570 s p95, 0.1051 mean RTF,
  worst corpus WER 0.0123.

All EN runs evaluated 1088/1088 clips with no skips. The WER range
(`0.0120`–`0.0123`) matches the published H100 BF16 reference (`0.0122`).

Full SeedTTS ZH (2020 clips, three measured repeats after one warmup):

- Concurrency 1: 11.671 samples/s, 0.086 s mean latency, 0.113 s p95,
  0.0185 mean RTF, corpus CER 0.0062.
- Concurrency 4: 34.622 samples/s, 0.115 s mean latency, 0.152 s p95,
  0.0249 mean RTF, corpus CER 0.0062.
- Concurrency 8: 55.214 samples/s, 0.145 s mean latency, 0.196 s p95,
  0.0312 mean RTF, corpus CER 0.0062.
- Concurrency 16: 79.532 samples/s, 0.201 s mean latency, 0.290 s p95,
  0.0434 mean RTF, corpus CER 0.0062.
- Concurrency 32 (queueing stress): 75.583 samples/s, 0.421 s mean latency,
  0.518 s p95, 0.0910 mean RTF, corpus CER 0.0062.

All ZH runs evaluated 2020/2020 clips with no skips. A matching Qwen3-ASR H100
ZH reference was not available in the repository, so this is an absolute
consumer result rather than a cross-hardware parity claim.

Memory checkpoints for the 24 GB profile:

- Before model load: 0.38 GiB process memory.
- After weights and KV-cache allocation: 15.97 GiB process memory.
- After CUDA Graph capture: 16.12 GiB process memory.
- During the 30-minute mixed workload: 17,116 MiB at each stage boundary,
  17,990 MiB sampled peak, and at least 6,574 MiB free.
- The default FP16/graph-32 configuration used 18,458 MiB before the full
  validation workload; the BF16/graph-16 profile preserves more headroom.

Cold/warm behavior on this network-mounted source environment:

- The initial default process took about 401 seconds from launch to `/health`;
  the final profile took about 308 seconds with caches populated. Roughly
  54 seconds of the final launch covered model/static allocation and graph
  capture; Python imports dominated the remaining startup time.
- The first uncached BF16 transcription took 35.8 seconds while Triton kernels
  compiled. After warmup, the 20-clip concurrency-1 mean was 0.155 seconds.

The 30-minute soak completed 78,418 successful transcriptions at concurrency
1/4/8/16 with zero unexpected errors. It also passed 28 malformed-input checks
and 28 stream-cancel/reconnect checks. Memory was 17,116 MiB before and after
the soak and after cooldown, `/health` remained available, and SIGTERM stopped
the worker and released GPU memory.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- Audio duration is bounded by the configured context and requested
  `max_new_tokens`, rather than a fixed 30-second window. Split audio or reduce
  `max_new_tokens` if the request exceeds that token budget.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
- SSE transport, cancellation, and reconnect are validated, but this model
  emitted only the terminal transcript event in the recorded run; no
  low-latency partial-text claim is made.
- `torch.compile` is disabled in the validated profile because CUDA Graphs
  already provided higher throughput with only about 0.14 GiB capture memory.
- `flash-attn-4` is still installed as a core dependency even though the SM89
  validation selected FlashInfer and Triton. Making architecture-specific
  kernels optional remains a repository-wide consumer-installation task.
- Source installs can log optional Mooncake/NIXL relay warnings when RDMA
  libraries are absent; those relays are not used by this single-GPU pipeline.
- Consumer-GPU validation is manual and reproducible; there is no blocking
  RTX 4090 CI gate.
