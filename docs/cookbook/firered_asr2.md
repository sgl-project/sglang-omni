# FireRedASR2-LLM

FireRedASR2-LLM uses a Kaldi fbank/CMVN frontend, Conformer encoder, frame
adapter, and SGLang's Qwen2 decoder in one ASR stage. No FireRed-specific
patch to SGLang or vLLM dependency is required.

## Installation

Follow the [installation guide](../get_started/installation.md), or the
[Ascend guide](../get_started/installation_npu.md), then install the frontend:

```bash
pip install 'kaldi-native-fbank>=1.20.0'
hf download allendou/FireRedASR2-LLM-vllm --local-dir /models/firered-asr2
```

The `firered-asr2` extra also installs the frontend. Use the merged checkpoint
referenced by the [official serving guide](https://github.com/FireRedTeam/FireRedASR2S):
`config.json` with `audio_encoder_conf`, CMVN statistics in
`preprocessor_config.json`, a tokenizer, and safetensors with `encoder.*`,
`encoder_projector.*`, and `llm.*` weights.

The tested Ascend environment uses SGLang 0.5.21, PyTorch/torch-npu 2.10.0,
CANN 9.0.0, and Transformers 5.12.1. This Omni base's NPU manifest still names
SGLang 0.5.20, which lacks the required `SpawnRanks` API. In a separate
virtual environment inheriting the prepared Ascend stack, install the matching
runtime with `python -m pip install --no-deps sglang==0.5.21`.

## Serve

```bash
sgl-omni serve \
  --config examples/configs/firered_asr2.yaml \
  --model-path /models/firered-asr2 \
  --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=json

curl -N http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=json -F stream=true
```

For Ascend, select a physical device with `ASCEND_RT_VISIBLE_DEVICES` and add
`--asr.factory.device npu`. The stage uses logical device 0. Defaults are a
BF16 decoder, eight concurrent requests, and greedy decoding with up to 256
output tokens. The acoustic encoder and adapter retain FP32 weights and
compute; reducing their precision changes recognition results. Encoder, prefill,
and decode graphs are enabled by default on NPU. Encoder graphs also capture
the adapter and retain FP32 compute. Decode uses SGLang's graph runner; CUDA
prefill uses its breakable graph backend.

NPU prefill captures the Qwen2 transformer inside the model, with attention
metadata and the logits processor outside the graph. Query lengths are padded
to power-of-two buckets; cached-prefix lengths remain exact. Each replay updates
the embeddings, positions, KV write locations, and block tables. Captures are
lazy, and the prefill cache retains at most 32 shapes. The supported NPU graph
configuration is TP=1, PP=1, and the default `ascend` attention kernels
(`ASCEND_USE_FIA` and `ASCEND_USE_FA` unset). The scheduler's generic prefill
graph flag describes its outer runner; model-owned captures and replays appear
in the FireRed graph logs (replay counters at debug level).

Use `--asr.engine.disable_cuda_graph true` for eager execution throughout.
To disable individual phases, use
`--asr.factory.enable_encoder_cuda_graph false` or
`--asr.engine.disable_prefill_cuda_graph true`.

The example sets `mem_fraction_static=0.70`; lower it or concurrency to leave
more encoder activation headroom. The audio embedding cache defaults to
256 MiB. For uncached evaluation, set
`--asr.factory.mm_embedding_cache_size_bytes 0` and
`--asr.engine.disable_radix_cache true`.

## Supported requests

The endpoint accepts `json`, `text`, and `verbose_json`. SSE streams text
from a complete uploaded clip; it does not accept incremental audio input.
Audio is converted to mono 16 kHz. Non-streaming uploads longer than 30 seconds
use shared audio chunking; each direct model request is limited to 40 seconds.

The prompt is fixed to `请转写音频为文字`; `language` and `prompt` do not change
model conditioning. Beam search, translation, word timestamps, and speaker
diarization are not implemented. CUDA and tensor parallelism are unvalidated.

Run `pytest -q tests/unit_test/firered_asr2` for model tests. Use the shared
`benchmarks.eval.benchmark_asr_seedtts` entry point for corpus WER/CER and
latency measurements against a running server.
