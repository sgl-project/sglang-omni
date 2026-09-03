# Fun-CosyVoice3

[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) is a lightweight text-to-speech model (0.5B parameters) from the FunAudioLLM team at Alibaba.
It uses a Qwen2.5-0.5B backbone with FSQ speech tokens (vocab = 6561 + 200 special tokens), conditioned on a CAMPPlus speaker embedding and prompt speech tokens extracted via an ONNX
speech tokenizer. It supports zero-shot voice cloning, cross-lingual synthesis, and instruction-based style control. The model produces 24 kHz speech at a rate of 25 Hz token frame rate through the `preprocessing → tts_engine → vocoder` pipeline and the OpenAI-compatible `/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` from source as in [Installation](../get_started/installation.md).

Fun-CosyVoice3 needs `sox` and a few extra Python packages. From the repository root, install the extra against **this checkout**:

```bash
apt-get update && apt-get install -y sox
uv pip install -e ".[fun-cosyvoice3]"
```

Clone the CosyVoice repository with its Matcha-TTS submodule and add both to `PYTHONPATH`:

```bash
COSYVOICE_PATH=/path/to/CosyVoice
COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
MATCHA_TTS_COMMIT=dd9105b34bf2be2230f4aa1e4769fb586a3c824e

git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ${COSYVOICE_PATH}
git -C ${COSYVOICE_PATH} checkout ${COSYVOICE_COMMIT}
git -C ${COSYVOICE_PATH} submodule update --init --recursive
git -C ${COSYVOICE_PATH}/third_party/Matcha-TTS checkout ${MATCHA_TTS_COMMIT}
export PYTHONPATH="${COSYVOICE_PATH}:${COSYVOICE_PATH}/third_party/Matcha-TTS:$PYTHONPATH"
```

**Do not** run `pip install -r requirements.txt` from the CosyVoice checkout. That file pins `torch`, `torchaudio`, `transformers`, and `diffusers` versions that conflict with the `sglang-omni` core pins. Only the `fun-cosyvoice3` extra above and the two `PYTHONPATH` entries are needed; the CosyVoice Flow and HiFT modules import fine against the `sglang-omni` versions of those shared packages.

The checkpoint includes ONNX models for the speech tokenizer and speaker encoder, which use the `onnxruntime` already pinned in `sglang-omni`'s core dependencies.

Download the checkpoint:

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

The pipeline is `preprocessing → tts_engine → vocoder`. First startup can take several minutes while the `tts_engine` captures CUDA graphs.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000
```


## Synthesizing Speech

### Zero-shot Voice Cloning

CosyVoice3 clones a voice from a short reference audio clip. `ref_audio` can be a local path, file URL, data URL, or HTTP URL. `ref_text` (the transcript of the reference clip) is optional but recommended for better alignment.

1. Using CURL:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "SGLang-Omni makes text-to-speech fast and easy to deploy.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

2. Using Python:

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Cross-lingual Synthesis

CosyVoice3 supports cross-lingual voice cloning where the reference speaker speaks a different language than the synthesis text. Omit `ref_text` to enter cross-lingual mode.

1. Using CURL:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "今天天气真好，我们一起出去散步吧。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
  }' \
  --output output.wav
```

2. Using Python:

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "今天天气真好，我们一起出去散步吧。",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Instruction-based Style Control

Pass `instructions` to guide prosody, emotion, or speaking style:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Welcome to our annual developer conference.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "instructions": "Speak in a cheerful and energetic tone, as if addressing a large audience."
  }' \
  --output output.wav
```

### Speed Control

Adjust playback speed with `speed` (default `1.0`):

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "This is spoken at one point three times normal speed.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "speed": 1.3
  }' \
  --output output.wav
```

### Streaming

Incremental Flow + HiFT decoding supports streaming. Set `stream: true` and `response_format: "pcm"` to emit audio before AR generation completes. The vocoder uses CosyVoice3's causal chunk loop: 25 speech tokens per hop (~1 s of audio). The first PCM chunk is emitted once AR produces `28 + prompt_pad` tokens (where `prompt_pad` rounds the Flow prompt length to a 25-token multiple). Later hops grow from 25 to 50 to 100 tokens in vocoder to increase batching efficiency without affecting audio continuity or real-time factor, only the decoding granularity changes, not the total synthesis time. The scheduler processes at most one hop per request per step to prevent backlogged streams from monopolizing the GPU. Non-streaming requests decode the entire utterance in one pass.

Optional serving knobs (vocoder factory):

| Factory arg | Default | Notes |
|---|---|---|
| `token_hop_len` | `25` | Base hop size; must match training chunk size |
| `token_max_hop_len` | `100` | Cap for `25 → 50 → 100` growth |
| `disable_hop_growth` | `false` | Leave off; growth ON performs better at high concurrency |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `ref_audio` | `null` | Reference audio for voice cloning (path / URL / data URL) |
| `ref_text` | `null` | Transcript of the reference audio. Improves cloning quality; omit for cross-lingual mode |
| `instructions` | `null` | Instruction text for style/prosody/emotion guidance |
| `speed` | `1.0` | Playback speed multiplier |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.8` | Top-p sampling |
| `top_k` | `20` | Top-k sampling |
| `repetition_penalty` | `1.1` | Repetition penalty |
| `max_new_tokens` | `min(2048, 20x target text tokens)` | Maximum number of generated speech tokens. If omitted, derived from the target text length (capped at 2048); stop tokens are also suppressed until at least `2x` that length has been generated |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Incremental causal Flow + HiFT; first PCM chunk after `28 + prompt_pad` speech tokens (`prompt_pad` rounds the prompt length to a multiple of 25) |

## Serving Optimization

### Flow Decoder Batching

The buffered vocoder batches requests through a two-stage pipeline. The scheduler collects up to 16 requests over at most 30 ms, then groups them by mel length in 50-frame buckets. Each bucket, including single-request buckets, invokes `Flow.inference()` once, where padding, masking, and CFM decoding are handled internally. To prevent long requests from blocking later ones, a bucket-rounded admission budget (`flow_batch_admission_frames`, default 8000 frames) controls batch assembly; requests exceeding this budget run as independent `B=1` batches.

HiFT vocoding follows the same batching pattern. Mels from a Flow bucket are right-zero-padded into a single tensor, decoded in one HiFT call, and sliced back to each request's true length. The padding budget (`hift_max_padding_waste`, default 1.5) limits wasted computation; since right-padding matches HiFT's single-request behavior, batched output is equivalent to individual inference except on padded frames. This design trades off throughput against latency while maintaining output correctness.

Change the mel-frame bucket size, for example to 100 frames:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.flow_batch_bucket_frames 100
```

This would increase the throughput at the cost of latency and higher peak GPU memory. Increase the normal Flow batching budget only after measuring the target GPU.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.flow_batch_admission_frames 4000
```

On the other hand, decrease the admission budget to reduce latency and lower peak GPU memory.

### Vocoder Configuration

Vocoder configuration controls batching, precision, and acceleration. The scheduler accepts `max_batch_size` (16) and `max_batch_wait_ms` (30) to tune batch assembly. Flow uses `dtype` (bfloat16) for autocast, while HiFT uses `hift_dtype` (float32), independent of Flow; bfloat16 shows no speedup on H200 and reduces fidelity. Two mutually exclusive accelerators are available: `enable_dit_torch_compile` and `enable_flow_estimator_trt`.

The TTS engine stage accepts `onnx_intra_op_threads` (16) for the speech tokenizer and speaker encoder ONNX sessions. Preprocessing takes `max_concurrency` (8) to limit concurrent reference conditioning requests.

### torch.compile for the DiT backbone

`torch.compile` is off by default. Enable it when you want the lowest DiT kernel-launch overhead. The first startup with an empty Inductor cache takes about 100 s and builds one symbolic (`dynamic=True`) graph for every utterance length; later starts reuse that cache. Keep the cache so you do not pay the compile cost again (`~/.cache/torch/inductor`, or `TORCHINDUCTOR_CACHE_DIR`).

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --vocoder.factory.enable_dit_torch_compile true \
  --port 8000
```

Do not enable it together with TensorRT.

### TensorRT for the DiT backbone

TensorRT accelerates the DiT by building a cached `.plan` engine from the bundled ONNX. The CFG batch is frozen at 2 with dynamic mel dimensions; larger request batches are handled by chunking cond/uncond pairs. TensorRT and torch.compile are mutually exclusive.

```bash
uv pip install tensorrt
```
See NVIDIA's [pip install guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html) for more details.

Enable the flag:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --vocoder.factory.enable_flow_estimator_trt true \
  --port 8000
```

On one H200, Flow latency and vocoder RTF comparisons:

| Backend | Batch size | Flow latency | Vocoder RTF |
|---|---|---|---|
| eager | 1 | 1131 ms | 0.359 |
| torch.compile | 1 | 1070 ms | 0.340 |
| TensorRT (CFG batch=2 engine) | 1 | 20 ms | 0.012 |
| eager | 4 | 267 ms | 0.026 |
| torch.compile | 4 | 220 ms | 0.022 |
| TensorRT (chunked 4× CFG pairs) | 4 | 77 ms | 0.011 |

Full SeedTTS EN set (1088 samples) on buffered `/v1/audio/speech`, one H200:

| Backend | Concurrency | Latency mean | RTF mean | Throughput |
|---|---|---|---|---|
| eager | 1 | 1.091 s | 0.241 | 0.916 req/s |
| torch.compile | 1 | 1.016 s | 0.221 | 0.984 req/s |
| TensorRT | 1 | 0.871 s | 0.189 | 1.147 req/s |
| eager | 16 | 5.690 s | 1.300 | 2.800 req/s |
| torch.compile | 16 | 3.151 s | 0.706 | 5.059 req/s |
| TensorRT | 16 | 2.549 s | 0.570 | 6.243 req/s |

This result is only demonstrative, since we have further optimization after the evluation of TensorRT is done.

### Running the flow vocoder in its own process

The three stages share one process by default. The flow vocoder holds most
in-flight requests while the ONNX reference encoders are the busiest thread in
that same interpreter, so on a saturated replica the two loops slow each other
down. `FunCosyVoice3IsolatedVocoderPipelineConfig` puts the vocoder in its own
process on the same GPU and declares the per-stage memory budgets that a shared
GPU requires:

```yaml
config_cls: FunCosyVoice3IsolatedVocoderPipelineConfig
model_path: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

```bash
sgl-omni serve --config fun_cosyvoice3_isolated_vocoder.yaml --port 8000
```

Setting `--vocoder.process vocoder` on the default config is not enough: two
process groups on one GPU must each declare a footprint, and the default config
declares none, so topology compilation refuses the launch. Measured on one H200,
this topology is worth about +24% throughput at admission cap 16 and +26% at cap
32, with bit-identical output; it costs one more CUDA context, and the AR engine
runs on a declared 0.80 budget instead of the whole-card default.

### Zero-shot Voice Cloning

CosyVoice3 clones a voice from a short reference audio clip. `ref_audio` can be a local
path, file URL, data URL, or HTTP URL. `ref_text` (the transcript of the reference clip)
is optional but recommended for better alignment.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "SGLang-Omni makes text-to-speech fast and easy to deploy.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Cross-lingual Synthesis

CosyVoice3 supports cross-lingual voice cloning where the reference speaker speaks a
different language than the synthesis text. Omit `ref_text` to enter cross-lingual mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "今天天气真好，我们一起出去散步吧。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
  }' \
  --output output.wav
```

### Instruction-based Style Control

Pass `instructions` to guide prosody, emotion, or speaking style:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Welcome to our annual developer conference.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "instructions": "Speak in a cheerful and energetic tone, as if addressing a large audience."
  }' \
  --output output.wav
```

### Speed Control

Adjust playback speed with `speed` (default `1.0`):

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "This is spoken at one point three times normal speed.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "speed": 1.3
  }' \
  --output output.wav
```

### Streaming

Incremental Flow + HiFT decoding is enabled. Set `stream: true` and use
`response_format: "pcm"` so the server can emit audio before speech-token
generation finishes.

The vocoder follows CosyVoice3's causal chunk loop: 25 speech tokens per hop
(`pre_lookahead_len=3`, 25 Hz → about 1 s of audio per chunk). The first PCM
chunk is emitted once the AR stage has produced `28 + prompt_pad` tokens,
where `prompt_pad` (0–24) rounds the Flow prompt-token length up to a
multiple of 25. Later hops grow 25 → 50 → 100 tokens like the upstream
`CosyVoice3Model` (default; keep growth on). Each scheduler step runs at most
one hop per request so a backlogged stream cannot monopolize the GPU.
Non-streaming requests still decode the whole utterance in one Flow + HiFT
pass.

Optional serving knobs (vocoder factory; keep `tts_engine.factory.token_hop_len`
in sync if you change the hop):

| Factory arg | Default | Notes |
|---|---|---|
| `token_hop_len` | `25` | Base hop / AR follow-up flush; must match training chunk size |
| `token_max_hop_len` | `100` | Cap for `25 → 50 → 100` growth |
| `disable_hop_growth` | `false` | Leave off unless A/B'ing; growth ON was better at c=16 |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `ref_audio` | `null` | Reference audio for voice cloning (path / URL / data URL) |
| `ref_text` | `null` | Transcript of the reference audio. Improves cloning quality; omit for cross-lingual mode |
| `instructions` | `null` | Instruction text for style/prosody/emotion guidance |
| `speed` | `1.0` | Playback speed multiplier |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.8` | Top-p sampling |
| `top_k` | `20` | Top-k sampling |
| `repetition_penalty` | `1.21` | Repetition penalty |
| `max_new_tokens` | `min(2048, 20x target text tokens)` | Maximum number of generated speech tokens. If omitted, derived from the target text length (capped at 2048); stop tokens are also suppressed until at least `2x` that length has been generated |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Incremental causal Flow + HiFT; first PCM chunk after `28 + prompt_pad` speech tokens (`prompt_pad` rounds the prompt length to a multiple of 25) |

## Benchmarking

Measure causal streaming with Seed-TTS-Eval against a running server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 --lang en --max-concurrency 16 \
  --use-existing-server --generate-only --stream \
  --output-dir results/fun_cosyvoice3_en
```

Use `--lang zh --no-ref-text` for the Chinese cross-lingual split. See
`benchmarks/README.md` for the full workflow.

## Model Architecture

| Component | Detail |
|---|---|
| LLM Backbone | Qwen2.5-0.5B (24 layers, hidden=896, 14 heads, 2 KV heads GQA) |
| Speech Tokenizer | FSQ codebook (vocab=6561) + 200 special tokens, 25 Hz frame rate |
| Speaker Encoder | CAMPPlus (192-dim embedding, ONNX) |
| Flow Model | CausalMaskedDiffWithDiT (DiT depth=22, dim=1024, heads=16) |
| Vocoder | CausalHiFTGenerator (24 kHz output) |
| Sample Rate | 24000 Hz |

## Known Limitations

- **Reference audio required.** CosyVoice3 requires a reference audio clip for voice cloning; it does not support text-only synthesis without a speaker reference.
- **30-second limit.** Reference audio must be 30 seconds or shorter for speech token extraction.
- **Speaker similarity.** Providing `ref_text` (the transcript) yields better voice similarity than omitting it (cross-lingual mode).
- **Reference shape.** The endpoint accepts either `ref_audio` plus optional `ref_text`, or one item in `references`; multiple references are rejected for this checkpoint.
- **Prompt modes.** Provide either `ref_text` or `instructions` for the reference prompt, not both. `instructions` selects CosyVoice3 `instruct2` conditioning.
- **Reference conditioning cache.** Local files, data URLs, and byte payloads are cached by audio content and encoder configuration. Mutable HTTP URLs are intentionally encoded on every request instead of being cached by URL alone.
- **Speed control.** Applied once, on the decoded waveform, by the shared `/v1/audio/speech` response-encoding path.
- **Voice conversion.** Voice conversion is outside the current zero-shot TTS scope.
- **Streaming decode.** Causal Flow + HiFT emit PCM after each hop (hop grows 25 → 50 → 100 by default). Quality can differ slightly from the buffered whole-utterance path. Opt-in TensorRT (`enable_flow_estimator_trt`) also accelerates streaming hops; do not enable it together with `enable_dit_torch_compile`. TRT freezes DiT attention, so streaming+TRT is not bit-exact with PyTorch streaming. Keep the Module TRT wrapper when streaming: CosyVoice's raw TRT enqueue is incompatible with packed hop-batch CFG shapes.
- **Flow batch scope.** Flow batching supports the CosyVoice PyTorch estimator and the opt-in TensorRT estimator. HiFT batches only the mels produced by one Flow bucket while padding waste stays within `hift_max_padding_waste`. Streaming coalesces first/follow-up hops across requests (`_can_batch_stream_chunks`, short peer wait) so TTFP stays low under load.
- **cosyvoice dependency.** The `cosyvoice` package has no PyPI release and must be installed from GitHub. Matcha-TTS is a required submodule and must also be importable; only the CosyVoice Flow and HiFT paths are used by the vocoder.
