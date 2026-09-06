# Fun-CosyVoice3

[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) is a
lightweight text-to-speech model (0.5B parameters) from the FunAudioLLM team at Alibaba.
It uses a Qwen2.5-0.5B backbone with FSQ speech tokens (vocab = 6561 + 200 special tokens),
conditioned on a CAMPPlus speaker embedding and prompt speech tokens extracted via an ONNX
speech tokenizer. It supports zero-shot voice cloning, cross-lingual synthesis, and
instruction-based style control. The model produces 24 kHz speech at a
25 Hz token frame rate through the `preprocessing → tts_engine → vocoder` pipeline and the
OpenAI-compatible `/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

Fun-CosyVoice3 depends on the `cosyvoice` package:

```bash
apt-get update && apt-get install -y sox
uv pip install "sglang-omni[fun-cosyvoice3]"
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

**Do not** run `pip install -r requirements.txt` from the CosyVoice checkout. That file pins
`torch`, `torchaudio`, `transformers`, and `diffusers` versions that conflict with the
`sglang-omni` core pins — only the `fun-cosyvoice3` extra above and the two `PYTHONPATH`
entries are needed; the CosyVoice Flow and HiFT modules import fine against the
`sglang-omni` versions of those shared packages.

The checkpoint includes ONNX models for the speech tokenizer and speaker encoder, which use
the `onnxruntime` already pinned in `sglang-omni`'s core dependencies.

Download the checkpoint:

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`. First startup can take several
minutes while the `tts_engine` captures CUDA graphs.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --port 8000
```

### Flow Decoder Batching

The buffered vocoder uses the batch-capable `FunCosyVoice3Flow.inference` API for every
request. `SimpleScheduler` collects up to 16 requests for at most 30 ms, then the vocoder
groups requests by total mel length in 50-frame buckets. Every bucket, including a
single-request bucket, calls the same built-in Flow inference method; packing, padding,
masking, CFM Euler/CFG, and output unpadding are handled inside the Flow implementation.
The 50-frame default matches the current DiT estimator's static chunk size; a larger bucket
can combine more requests at the cost of additional padding, compute, and peak GPU memory.

The scheduler uses a bucket-rounded Flow admission budget, configured by
`flow_batch_admission_frames` (8,000 by default). It controls whether a later request joins the
current Flow batch; it is not a maximum supported request length. A request whose total
prompt-plus-output mel length exceeds that budget runs as a B=1 Flow batch through the same
adapter, and later requests wait for the next scheduler batch. This preserves valid long
generations while preventing them from being combined with more work.

HiFT is batched the same way: the mels from one Flow bucket are right-zero-padded into a single
tensor, decoded in one HiFT call, and sliced back to each request's true length, under the
padding budget `hift_max_padding_waste` (1.5 by default; `1.0` only groups requests that need no
padding at all). HiFT is prepared for this at load time by folding away its `weight_norm`
parametrizations. Right-zero-padding matches the zero padding HiFT applies in single-request
inference, so batched output is identical except in the final mel frame of padded requests.

The built-in Flow implementation supports the pinned CosyVoice PyTorch estimator, an opt-in
TensorRT estimator (see below), and buffered `streaming=False, finalize=True` inference only.

Change the mel-frame bucket size, for example to 100 frames:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --port 8000 \
  --vocoder.factory.flow_batch_bucket_frames 100
```

The same setting can be written in the pipeline config under the vocoder's
`factory` group:

```yaml
stages:
  vocoder:
    factory:
      flow_batch_bucket_frames: 100
```

Increase the normal Flow batching budget only after measuring the target GPU. This changes the
maximum aggregate padded work admitted into one scheduler batch; it does not reject a longer
single request.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --port 8000 \
  --vocoder.factory.flow_batch_admission_frames 4000
```

The YAML equivalent is:

```yaml
stages:
  vocoder:
    factory:
      flow_batch_admission_frames: 4000
```

The remaining vocoder `factory` options are `max_batch_size` (16) and `max_batch_wait_ms` (30)
for the scheduler batch, `dtype` (`bfloat16`) for the Flow autocast, `hift_dtype` (`float32`,
independent of `dtype`; `bfloat16` measured no faster for HiFT on H200 and lowers output fidelity)
for the HiFT autocast, `enable_dit_torch_compile`, and `enable_flow_estimator_trt`
(see below; mutually exclusive DiT accelerators). The
`tts_engine` stage takes `onnx_intra_op_threads` (16) for the speech tokenizer and speaker
encoder ONNX sessions, and `preprocessing` takes `max_concurrency` (8) for concurrent reference
conditioning.

The built-in Flow implementation is tied to the Flow/CFM structure in the documented CosyVoice commit
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`. It does not patch the CosyVoice source on disk; an
incompatible Flow structure fails directly instead of using a fallback implementation.

### torch.compile for the DiT backbone

The flow decoder's DiT backbone (`flow.decoder.estimator`, a 22-layer DiT invoked
once per Euler step) can be compiled with `torch.compile` to reduce per-step
kernel-launch overhead. It is opt-in because it pays a one-time compile cost at
startup and is most beneficial under sustained throughput load. The compile uses
`dynamic=True`, so one symbolic-sequence-length graph serves every utterance
length.

Enable it by overriding the vocoder stage's `factory` args (the `stages.`
prefix is implied in the CLI dotted path):

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --vocoder.factory.enable_dit_torch_compile true \
  --port 8000
```

Do not enable this together with TensorRT for the same DiT (see below).

### TensorRT for the DiT backbone

The same `flow.decoder.estimator` can be replaced with a TensorRT engine built
from the checkpoint's bundled ONNX (`flow.decoder.estimator.fp32.onnx` is
preferred). This is opt-in: TensorRT is not a `sglang-omni` extra, first
startup builds and caches a `.plan` under `COSYVOICE3_TRT_CACHE` or
`~/.cache/sglang-omni/cosyvoice3_trt`, and the official CosyVoice ONNX freezes
CFG batch at 2 (t and spks are static; only the mel time dim is dynamic).
Packed Flow (CFG batch = `2 * request_batch`) still works by chunking
request-wise cond/uncond pairs into that CFG=2 engine.

TensorRT and `torch.compile` both replace the same DiT, so they are mutually
exclusive. Enable only one:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --vocoder.factory.enable_flow_estimator_trt true \
  --port 8000
```

On one H200, packed Flow + HiFT (80 target tokens, 25 prompt tokens,
6 timed iters after 2 warmups) was:

| Backend | B | Flow latency | Vocoder RTF |
|---|---|---|---|
| eager | 1 | 1131 ms | 0.359 |
| torch.compile | 1 | 1070 ms | 0.340 |
| TensorRT (CFG batch=2 engine) | 1 | 20 ms | 0.012 |
| eager | 4 | 267 ms | 0.026 |
| torch.compile | 4 | 220 ms | 0.022 |
| TensorRT (chunked 4× CFG pairs) | 4 | 77 ms | 0.011 |

TensorRT here is not bit-exact with eager PyTorch (FP16 TensorRT tactics on
the fp32 ONNX; cosine similarity about 0.995 on one packed mel).

The same three vocoder backends on the full SeedTTS EN set (1088 samples),
buffered `/v1/audio/speech` (non-streaming, generate-only), one H200, 0
failures. ASR WER was not remeasured; vocoder-only mel cosine vs eager is
about 0.995.

| Backend | Concurrency | Latency mean | RTF mean | Throughput |
|---|---|---|---|---|
| eager | 1 | 1.091 s | 0.241 | 0.916 req/s |
| torch.compile | 1 | 1.016 s | 0.221 | 0.984 req/s |
| TensorRT | 1 | 0.871 s | 0.189 | 1.147 req/s |
| eager | 16 | 5.690 s | 1.300 | 2.800 req/s |
| torch.compile | 16 | 3.151 s | 0.706 | 5.059 req/s |
| TensorRT | 16 | 2.549 s | 0.570 | 6.243 req/s |

At concurrency 1 the pipeline is still mostly preprocessing + AR, so TensorRT
is about 1.17× `torch.compile`. At concurrency 16 the vocoder is the bottleneck
and TensorRT is about 1.23× compile throughput (2.23× eager). It remains
opt-in because TensorRT is a separate install and the first engine build takes
about a minute.

## Synthesizing Speech

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

### Streaming (Planned)

Incremental Flow + HiFT decoding is planned but is not enabled in the current implementation.
The current vocoder buffers the generated speech tokens and returns one complete waveform.
Do not rely on `stream=true` for time-to-first-audio until the streaming decoder is wired.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "response_format": "wav"
  }' \
  --output output.wav
```

The request above uses the supported buffered response path. A future streaming implementation
will use `response_format="pcm"` and emit audio before speech-token generation completes.

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
| `stream` | `false` | Reserved for the planned incremental decoder; current decode is buffered |

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

- **Reference audio required.** CosyVoice3 requires a reference audio clip for voice
  cloning; it does not support text-only synthesis without a speaker reference.
- **30-second limit.** Reference audio must be 30 seconds or shorter for speech token
  extraction.
- **Speaker similarity.** Providing `ref_text` (the transcript) yields better voice
  similarity than omitting it (cross-lingual mode).
- **Reference shape.** The endpoint accepts either `ref_audio` plus optional `ref_text`,
  or one item in `references`; multiple references are rejected for this checkpoint.
- **Prompt modes.** Provide either `ref_text` or `instructions` for the reference prompt,
  not both. `instructions` selects CosyVoice3 `instruct2` conditioning.
- **Reference conditioning cache.** Local files, data URLs, and byte payloads are cached
  by audio content and encoder configuration. Mutable HTTP URLs are intentionally encoded
  on every request instead of being cached by URL alone.
- **Speed control.** Applied once, on the decoded waveform, by the shared
  `/v1/audio/speech` response-encoding path.
- **Voice conversion.** Voice conversion is outside the current zero-shot TTS scope.
- **Streaming decode.** The current implementation buffers all speech tokens before Flow + HiFT
  decoding. Incremental PCM output is planned but is not yet available.
- **Flow batch scope.** Flow batching supports the CosyVoice PyTorch estimator
  and the opt-in TensorRT estimator (`enable_flow_estimator_trt`). Do not enable
  TensorRT together with `enable_dit_torch_compile`. HiFT batches only the mels
  produced by one Flow bucket while padding waste stays within
  `hift_max_padding_waste`. Streaming Flow/HiFT batching is outside the current
  buffered decoder.
- **cosyvoice dependency.** The `cosyvoice` package has no PyPI release and must be
  installed from GitHub. Matcha-TTS is a required submodule and must also be importable;
  only the CosyVoice Flow and HiFT paths are used by the buffered decoder.
