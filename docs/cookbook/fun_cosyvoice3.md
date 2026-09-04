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

Fun-CosyVoice3 depends on the `cosyvoice` package. On Linux:

```bash
apt-get update && apt-get install -y sox
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "sglang-omni[fun-cosyvoice3]"
```

On Apple Silicon, use the repository installer so the same environment also
gets pinned SGLang `v0.5.18` from source with its Metal dependencies:

```bash
brew install sox
SGLANG_OMNI_EXTRAS=fun-cosyvoice3 ./install.sh
source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

The installer uses Homebrew for `ffmpeg@7` and `uv`; `sox` is the additional
Fun-CosyVoice3 system dependency. Keep `DYLD_LIBRARY_PATH` set in the shell that
starts `sgl-omni`, so TorchCodec can find the keg-only FFmpeg libraries.

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

The pipeline is `preprocessing → tts_engine → vocoder`. On CUDA, first startup can take
several minutes while the `tts_engine` captures CUDA graphs.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --port 8000
```

### Apple Silicon MLX

Set `SGLANG_USE_MLX=1` to run the speech-token Qwen2 model, Flow/DiT, and HiFT
with native MLX. Reference encoding continues to use the official checkpoint's
ONNX assets. Keep that checkpoint as `--model-path`, and provide the converted
MLX artifact separately; the converted artifact contains the Qwen2, Flow, and
HiFT weights but not the preprocessing assets. `mlx-audio` is not a runtime
dependency.

```bash
SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --tts-engine.factory.mlx_model_path \
    mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
  --port 8000
```

The pipeline passes `tts_engine.factory.mlx_model_path` to the MLX vocoder by
default, so the common launch only needs one artifact override. Use
`vocoder.factory.mlx_model_path` only when the Flow/HiFT weights live in a
different converted artifact. Both paths also accept a corresponding
`mlx_model_revision` override.

The native vocoder currently accepts the single-file `config.json` plus
`model.safetensors` layout shown above, with `flow.*` and either public
`hifigan.*` or canonical sanitized `hift.*` weights. Raw `flow.pt`/`hift.pt`
and unsanitized converted weights are not loaded by this path. Its compute
dtype comes from the converted artifact (the example artifact is FP16); an
explicit `vocoder.factory.dtype` is only accepted when it matches that dtype.
The MLX vocoder also requires the stage device to remain MPS.
The validated public artifact revision is
`55a6713d54751f1ec2645aa11294676f2067202a`; set it with
`--tts-engine.factory.mlx_model_revision` when a reproducible model snapshot is
required.

### Apple Silicon Torch/MPS

Without the MLX opt-in, the same pipeline runs the speech-token model and
vocoder through PyTorch MPS. This path is useful as a correctness baseline and
does not need a separate checkpoint:

```bash
unset SGLANG_USE_MLX
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  --port 8000
```

### Flow Decoder Batching

The Torch vocoder uses the batch-capable `FunCosyVoice3Flow.inference` API for every
request. `SimpleScheduler` collects up to 8 requests for at most 2 ms, then the vocoder
groups requests by total mel length in 50-frame buckets. Every bucket, including a
single-request bucket, calls the same built-in Flow inference method; packing, padding,
masking, CFM Euler/CFG, and output unpadding are handled inside the Flow implementation.
The 50-frame default matches the current DiT estimator's static chunk size; a larger bucket
can combine more requests at the cost of additional padding, compute, and peak GPU memory.
The native MLX Flow currently supports batch size 1 and rejects a larger
`vocoder.factory.max_batch_size` at startup.

The scheduler uses a bucket-rounded Flow admission budget, configured by
`flow_batch_admission_frames` (2,000 by default). It controls whether a later request joins the
current Flow batch; it is not a maximum supported request length. A request whose total
prompt-plus-output mel length exceeds that budget runs as a B=1 Flow batch through the same
adapter, and later requests wait for the next scheduler batch. This preserves valid long
generations while preventing them from being combined with more work.

HiFT still runs once per request. The built-in Flow implementation supports the pinned
CosyVoice PyTorch estimator and buffered `streaming=False, finalize=True` inference only.
TensorRT Flow is not supported by this integration and fails during vocoder initialization
rather than falling back to another inference path.

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

The built-in Flow implementation is tied to the Flow/CFM structure in the documented CosyVoice commit
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`. It does not modify or monkey-patch the
CosyVoice checkout; an incompatible Flow structure fails directly instead of using a
fallback implementation.

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
| `top_k` | `25` | Top-k sampling (matches CosyVoice3's RAS sampler) |
| `repetition_penalty` | `1.0` | Optional additional repetition penalty |
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
- **Flow batch scope.** Flow batching currently supports only the PyTorch estimator. HiFT
  remains serial, and streaming Flow/HiFT batching is outside the current buffered decoder.
- **cosyvoice dependency.** The `cosyvoice` package has no PyPI release and must be
  installed from GitHub. Matcha-TTS is a required submodule and must also be importable;
  only the CosyVoice Flow and HiFT paths are used by the buffered decoder.
