# MOSS-Transcribe-Diarize

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) is a multi-speaker ASR and diarization model from the OpenMOSS team.

![Model Architecture](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/main/Model_Architecture.png)

It transcribes speech, assigns speakers, and predicts timestamps in a single generation pass. With 128K context, it supports up to ~90-minute audio, handles meetings, interruptions, long conversations, and overlapping speech, and adds hotword boosting for names, companies, product terms, and domain vocabulary. MOSS-Transcribe-Diarize is served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint.

| Component | Spec |
|---|---|
| Architecture | `MossTranscribeDiarizeForConditionalGeneration` |
| Audio encoder | Whisper encoder (24 L, d_model=1024) |
| Text decoder | Qwen3 (28 L, hidden=1024, GQA 16/8) |
| Output | Speaker-labelled transcript with start/end timestamps |
| Endpoint | `/v1/audio/transcriptions` |

## Architecture & Optimization Principles

### Inference Pipeline

MOSS-TD follows the **Audio LLM** pattern: an audio encoder produces continuous embeddings that are injected into a decoder-only LLM, which then autoregressively generates the transcript.

The inference pipeline has three stages:

1. **Encoder** — The Whisper encoder converts the input waveform into a log-mel spectrogram (80 frequency bins), passes it through 24 Transformer layers with bidirectional attention, and applies a 4× Time Merge (concatenating every 4 consecutive frames) to shorten the sequence. A VQAdaptor (MLP projection: Linear → SiLU → Linear → LayerNorm, 4096→1024) maps the merged features into the LLM's embedding space. Despite the "VQ" name, no vector quantization is involved — the representations remain continuous float vectors throughout.
2. **LLM Prefill** — The projected audio embeddings replace `<|audio_pad|>` placeholder tokens in the prompt. Qwen3 processes the full prompt in one parallel forward pass to build the KV cache.
3. **AR Decode** — Qwen3 autoregressively generates text tokens (transcript with speaker labels and timestamps) one at a time until EOS.

### Chunked Prefill

For long audio (up to ~90 min), the encoded token sequence can reach tens of thousands of tokens. Prefilling all at once would spike memory usage and block other requests. Chunked Prefill splits the sequence into fixed-size chunks (4096 tokens) and processes one chunk per scheduling step, interleaving decode steps for other requests between chunks. This prevents OOM on long inputs and keeps GPU utilization high under concurrent load.

Stream output is suppressed during chunked prefill to avoid emitting intermediate states as transcript text.

### Where Time Is Spent

Profiling on a single H100 (CUDA Graph, bf16) reveals a clear pattern: **AR Decode dominates latency, but its share shrinks under concurrency as batching amortizes per-token cost.**

| Audio length | Concurrency | Encoder | LLM Prefill | AR Decode | Decode per token | Throughput (× real-time) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 1 | 8.9% | 14.7% | **76.4%** | 1.44 ms | 46× |
| 5 s | 16 | **38.2%** | 29.7% | **32.1%** | 0.13 ms | 267× |
| 60 s | 1 | 4.0% | 2.1% | **94.0%** | 1.49 ms | 78× |
| 60 s | 16 | 13.7% | 9.5% | **76.8%** | 0.14 ms | 713× |
| 20 min | 1 | 4.7% | 0.8% | **94.5%** | 2.17 ms | 78× |
| 20 min | 16 | 11.6% | 2.6% | **85.7%** | 0.86 ms | 210× |

Key observations:

- At **low concurrency**, decode accounts for 76–95% of wall time. Optimizations targeting the decode loop (CUDA Graph, Async Decode) deliver the most impact.
- At **high concurrency**, decode is amortized by batching. Encoder + Prefill can account for 40–68% (especially on short audio), making encoder-side optimizations (Torch Compile, caching) worthwhile.
- **Longer audio has higher per-token decode cost** (0.13 ms vs 0.86 ms at c=16) due to growing KV cache attention overhead.

### Optimization Strategies

**CUDA Graph (decode only)** — The encoder cannot use CUDA Graph because each request has a different audio length, producing variable tensor shapes. The decode step, however, pads its batch size to predefined buckets (1, 2, 4, 8, …) and replays a captured CUDA graph, eliminating kernel launch overhead on every token.

**Async Decode** — Uses one-step lookahead: launch the current decode step's GPU forward pass, then resolve the *previous* step's host-side work (D2H copy, finish detection, result dispatch) in parallel. A minimum batch size of 2 is required — at bs=1, the host work is too small to overlap, so it falls back to synchronous mode. Two alternating pinned host buffers ("ping-pong") prevent read/write races.

**LRU Encoder Cache** — An `StageOutputCache` (max 64 entries, 4 GB, CPU-resident) keyed by audio fingerprint hash. When the same audio is re-submitted (retries, A/B tests, debugging), the encoder forward is skipped entirely. The cache has limited impact in production where most inputs are unique, but it is a low-cost safety net.

**Stream Output** — Emits transcript text incrementally during decode via SSE (`transcript.text.delta` events) instead of waiting for the full sequence. A configurable rate limit (default 50 ms) prevents excessive small messages. Incomplete UTF-8 sequences are buffered until complete, and output is suppressed during chunked prefill.

### Long Audio Considerations

- **Memory allocation**: Long audio (≥ 10 min) requires reserving enough GPU memory for the encoder. The Whisper encoder processes audio in 30-second mel windows — a 20-minute clip produces ~40 windows in a single forward pass. If `mem-fraction-static` is too high (most memory given to the KV cache pool), the encoder forward will OOM. Reduce it (e.g. `0.40`–`0.60`) for long-audio workloads.
- **Max tokens**: Set `max_new_tokens=65536` for long audio to avoid truncated transcripts.

## Model Usage

### Launching Commands

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize
```

Serve the model:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --port 8000 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80
```

### Sending Requests

Use `response_format=verbose_json` when you need parsed speaker segments. `json` returns the raw transcript text only.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=verbose_json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "response_format": "verbose_json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
payload = resp.json()
print(payload["text"])
for segment in payload.get("segments", []):
    print(
        f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}"
    )
```

For longer multi-speaker audio, raise `max_new_tokens` so the decoder can finish the full diarized transcript. The example below uses a repo-local clip with two speakers:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@docs/_static/audio/gaokao-listening.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=65536
```

### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional language hint |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | model default (`0.0`) | Sampling temperature |
| `max_new_tokens` | int | `5120` | Max generated tokens; raise for long audio (e.g. `65536`) |
| `prompt` | string | unset | Optional instruction override; omit to use the built-in transcribe+diarize prompt |

`verbose_json` parses the model markup into OpenAI-style `segments` with
`start`, `end`, and speaker-prefixed `text` (for example `[S01]...`).
`json` / `text` return the full transcript string without segment parsing.

## Benchmarking

Thanks to the Moss team for providing the benchmark datasets, we prepare movies800times and aishell4_long as benchmark datasets for multi-speaker ASR. movies800times is a short-sequence dataset with 800 dialog clips, and aishell4_long is a long-sequence dataset with 20 long-form meeting audio. These two datasets are right now under private license, and you can contact the Moss team for access.


```bash
# Short-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --concurrency 16 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80 \
  --output-dir results/moss_transcribe_diarize_movies800times

# Long-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset aishell4_long \
  --concurrency 16 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80 \
  --max-new-tokens 65536 \
  --request-timeout-s 1800 \
  --output-dir results/moss_transcribe_diarize_aishell4_long
```

## Benchmark Results

Here we provide the benchmark results of movies800times and aishell4_long on a single H100 80GB GPU. Each row is the **mean of 3 runs** against a server with `max_running_requests=16`, `cuda_graph_max_bs=16`, and `mem_fraction_static=0.80`.

### movies800times

| Concurrency | Throughput (req/s) | Mean latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 2.57 | 0.388 | 0.0612 | 29.76 |
| 2 | 4.89 | 0.409 | 0.0659 | 56.55 |
| 4 | 6.62 | 0.513 | 0.0790 | 76.64 |
| 8 | 6.80 | 0.533 | 0.0810 | 78.70 |
| 16 | 7.08 | 0.659 | 0.0922 | 81.98 |

### aishell4_long

| Concurrency | Throughput (req/s) | Mean latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 0.022 | 45.2 | 0.0197 | 50.64 |
| 2 | 0.032 | 60.7 | 0.0265 | 74.25 |
| 4 | 0.036 | 105.6 | 0.0461 | 81.64 |
| 8 | 0.040 | 172.6 | 0.0754 | 90.62 |
| 16 | 0.043 | 282.8 | 0.1237 | 98.83 |


- **Concurrency** — Maximum number of in-flight client requests (`--concurrency`).
- **Throughput (req/s)** — Completed requests divided by total benchmark wall-clock time.
- **Mean latency** — Average end-to-end time per request (send to full response received).
- **RTF mean** — Average ratio of processing time to input audio duration per request. `<1` is faster than real time.
- **audio_s/s** — Total seconds of input audio processed divided by total benchmark wall-clock time.

To reproduce the results, follow the commands above or the entry point in [`benchmark_asr_transcribe_diarize.py`](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_asr_transcribe_diarize.py).

## Acknowledgments

Thanks for the joint effort of the OpenMOSS team and SGLang Omni team.

MOSS Team: Donghua Yu, Zhengyuan Lin, Hanfu Chen, Yiyang Zhang, Yang Gao, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu

SGLang Omni Team: Yijiang Tian, Xinli Jin, Xiangrui Ke, Zhihao Guo, Ruoqi Zhang, Lifan Shen, Jintao Qu, Xuxiang Tian, Kaige Li, Ratish P, Haoguang Cai, Zijie Xia, Chenchen Hong, Xuesong Ye, Jingwen Gu,  Jiaxin Deng, Jiaxuan Luo, Xinyu Lu, Hao Jin, Chenyang Zhao, Yichi Zhang
