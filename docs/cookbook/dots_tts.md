# dots.tts

[dots.tts](https://huggingface.co/dots-studio/dots.tts-mf) is a text-to-speech model from rednote-hilab. It outputs 48 kHz speech and clones a speaker from a short reference clip plus its transcript.

dots.tts is a continuous-latent model, not a codec model. The backbone emits no audio tokens. Each AR step gives a hidden state; a MeanFlow DiT uses it to sample one latent patch (4 frames × 128 dims); a semantic encoder turns the patch back into the next backbone input; an AudioVAE decodes latents to waveform. No codebook, no token sampler. So `temperature` and `top_k` do nothing here — use the solver knobs (`num_steps`, `guidance_scale`) instead.

| Component | Spec |
|---|---|
| Backbone | Qwen2 1.5B decoder (28 L, hidden=1536, GQA 12/2) |
| Acoustic tail | MeanFlow DiT (18 L, hidden=1024, 16 heads) + VAE semantic encoder |
| Latent patch | 4 frames × 128 dims (one patch ≈ 160 ms of audio) |
| Context length | 2,048 tokens |
| Sample rate | 48 kHz |
| Solver | MeanFlow + Euler, engine-wide `num_steps=4` (SOAR: flow matching + CFG, `num_steps=10`) |

## Supported checkpoints

| Checkpoint | Status |
|---|---|
| [`dots-studio/dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf) | MeanFlow. Continuous batching, `num_steps=4`. `examples/configs/dots_tts.yaml` |
| [`dots-studio/dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar) | Flow matching. Single request at a time (`max_running_requests=1`) with CFG, `num_steps=10`. `examples/configs/dots_tts_soar.yaml` |
| [`dots-studio/dots.tts-base`](https://huggingface.co/dots-studio/dots.tts-base) | Flow matching, same as SOAR. Serve it with `examples/configs/dots_tts_soar.yaml` and `--model-path dots-studio/dots.tts-base` |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download and launch the server:

```bash
hf download dots-studio/dots.tts-mf

sgl-omni serve \
  --model-path dots-studio/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

To serve SOAR instead, swap both the checkpoint and the config:

```bash
hf download dots-studio/dots.tts-soar

sgl-omni serve \
  --model-path dots-studio/dots.tts-soar \
  --config examples/configs/dots_tts_soar.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

SOAR is a flow-matching checkpoint. It runs the single-request solver with classifier-free guidance, so its config pins `max_running_requests: 1` and `num_steps: 10`; continuous batching is MeanFlow-only. Every request example below works on either checkpoint — only the `model` field changes.

`examples/configs/dots_tts.yaml` is the canonical MeanFlow deployment. It is already tuned; compiled acoustic tail and vocoder (`optimize: true`, on by default); continuous batching at `max_running_requests=16`; and the backbone decode CUDA graph. `--model-path` alone keeps the compiled tail and batching but leaves backbone decode eager, which is slower per request (see [Performance](#performance)). Use the config file.

If startup fails with `dots.tts acoustic-tail admission failed at startup`, the GPU cannot hold `max_running_requests × max_generate_length` full-length acoustic pools — lower those knobs yourself. The engine never silently shrinks them.

The examples below read local clips from `docs/_static/audio`. To fetch reference audio over HTTP instead, allow the domains you need, e.g. `--allowed-media-domain huggingface.co`.

## Memory and capacity

Continuous batching eagerly allocates acoustic-tail state for every slot at the full generate length:

```text
patch_capacity = max_generate_length + 1
dit_cache_tokens = patch_capacity × (hidden_patch_size + latent_patch_size)   # MF: ×5
```

Pool bytes scale roughly as `max_running_requests × patch_capacity` and include DiT KV (per NFE), semantic-encoder KV, scratch K/V, masks, window, and AdaLN mods. Startup logs the estimated breakdown and free CUDA memory, then refuses to allocate when free VRAM is below the estimate plus a 15% headroom for graphs and workspace.

`mem_fraction_static` (default `0.20` in `examples/configs/dots_tts.yaml`) only budgets the **SGLang backbone** KV cache. Acoustic-tail pools are separate and are **not** covered by that fraction.

| Knob | Effect on pool size |
|---|---|
| `max_running_requests` | Number of full-length slots |
| `max_generate_length` | `patch_capacity` (and therefore DiT / encoder sequence length) |
| `num_steps` | DiT KV is replicated per NFE step |

On a full GPU the default `16 × 500` layout is intended. On tighter cards, lower `max_running_requests` and/or `max_generate_length` explicitly rather than hoping for automatic downsizing. A live request that needs a free slot when all are busy fails with `dots.tts acoustic tail admission failed: ran out of slots` — lower client concurrency or raise `max_running_requests` (if memory allows).

Incremental / bucketed tail-KV allocation is not implemented yet; capacity is still the eager full-length pool.

## Synthesizing Speech

dots.tts needs a reference clip and its transcript. The speaker comes entirely from the reference (x-vector plus prompt latents), so there is no zero-shot `voice` preset. Under the default continuous-batching deployment, a request without `references` is rejected.

### Voice Cloning

The reference transcript matters. It is prefixed to your input text so the model can align prompt audio with prompt text. A wrong transcript hurts cloning quality.

1. Use curl:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "seed": 42
  }' \
  --output output.wav
```

2. Use Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "dots-studio/dots.tts-mf",
        "input": "Have a nice day and enjoy south california sunshine.",
        "references": [{
            "audio_path": "docs/_static/audio/male-voice.wav",
            "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
        }],
        "seed": 42,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

`ref_audio` / `ref_text` are accepted as a shorthand for `references[0].audio_path` / `references[0].text`.

### Streaming

Streaming lets you play audio while generation is still running, which cuts time-to-first-audio. dots.tts streams raw 48 kHz PCM: the AudioVAE decoder emits a waveform chunk every few latent patches instead of waiting for the whole utterance.

Set `"stream": true` and `"response_format": "pcm"`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "docs/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output output.pcm
```

The `-N` flag disables curl's output buffering so chunks are written as they arrive. The response is 16-bit mono PCM at 48 kHz with no in-band JSON framing; convert it with:

```bash
ffmpeg -f s16le -ar 48000 -ac 1 -i output.pcm output.wav
```

### Request parameters

Top-level fields:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served dots.tts model identifier |
| `input` | string | (required) | Text to synthesize |
| `references` | list | (required) | Reference audio for cloning. Each item has `audio_path` (local path, file URL, data URL, or HTTP URL) and `text` (transcript). Exactly one reference is accepted |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `response_format` | string | `"wav"` | Output audio format (`wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`) |
| `stream` | bool | `false` | Enable raw PCM streaming |
| `seed` | int | `null` | Seed for the flow sampler. Fixes the output for a given request |
| `language` | string | `null` | Language tag for the prompt text; `auto` or `auto_detect` detects it from the input |
| `instructions` | string | `null` | Style instructions; switches the prompt template to `instruction_tts` |

Solver knobs go under `stage_params.latent_engine`. They are **not** top-level fields:
`CreateSpeechRequest` drops unknown top-level keys, so sending `num_steps` next to
`input` changes nothing and reports no error.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `speaker_scale` | float | `1.5` | Scales the speaker x-vector before conditioning. Higher values push harder toward the reference timbre |
| `guidance_scale` | float | `1.2` | Classifier-free guidance strength for the flow sampler |
| `eos_threshold` | float | `0.8` | Probability above which the EOS head ends generation. Lower values cut utterances earlier |
| `num_steps` | int | `4` (MF), `10` (SOAR) | Flow solver steps. MeanFlow fixes this engine-wide: another value fails the request. SOAR and base run one request at a time and honour it |
| `ode_method` | string | `"euler"` | Flow solver. MeanFlow accepts only `euler` |
| `max_generate_length` | int | `500` | Cap on generated latent patches (≈ 160 ms each), bounded by the engine's `max_generate_length` |
| `normalize_text` | bool | `false` | Run the upstream text normalizer (numbers, symbols) before tokenizing |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "seed": 42,
    "stage_params": {"latent_engine": {"speaker_scale": 2.0, "eos_threshold": 0.6}}
  }' \
  --output output.wav
```

A rejected solver value comes back as HTTP 500 with the engine's message, for example `dots.tts num_steps is fixed for continuous batching`. It is a validation failure, not a server fault.

`temperature`, `top_p`, and `top_k` do not apply. The backbone token logits are unused; the acoustic tail is a deterministic flow solve once the seed is fixed.

### Performance

Seed-TTS EN benchmark at main commit `2b45073c`, seed 42, with 10 warmup requests. Throughput and latency were measured on **1x H100**. The server used `examples/configs/dots_tts.yaml`, which enables the optimized acoustic tail, vocoder, and backbone CUDA Graph. Every row uses all 1,088 samples.

| Concurrency | Throughput (req/s) | Mean latency | RTF (per-req) | audio_s/s | WER |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.935 | 1.070 s | 0.275 | 3.726 | 1.241% |
| 2 | 1.556 | 1.286 s | 0.314 | 6.493 | 1.256% |
| 4 | 2.493 | 1.603 s | 0.390 | 10.407 | 1.264% |
| 8 | 3.875 | 2.062 s | 0.502 | 16.173 | 1.323% |
| 16 | 4.760 | 3.344 s | 0.812 | 19.859 | 1.348% |
| 32 | 4.988 | 6.344 s | 1.596 | 20.818 | 1.331% |

All requests completed successfully.

- **Concurrency** — Maximum number of in-flight client requests (`--max-concurrency`).
- **Throughput (req/s)** — Completed requests divided by total benchmark wall-clock time.
- **Mean latency** — Average end-to-end time per request (send to full response received).
- **RTF (per-req)** — Average ratio of processing time to generated audio duration per request. `<1` is faster than real time.
- **audio_s/s** — Total seconds of audio produced divided by total benchmark wall-clock time.
- **WER** — Corpus word error rate of the generated speech, transcribed with `Qwen/Qwen3-ASR-1.7B`.

To reproduce, start the server as in [Prerequisites](#prerequisites) and run the benchmark against it:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf \
  --ref-format references \
  --base-url http://127.0.0.1:8000 --port 8000 \
  --lang en --max-concurrency 16 --max-samples 1088 --warmup 10 --seed 42 \
  --generate-only --use-existing-server \
  --output-dir results/dots-seedtts-en-c16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf \
  --ref-format references --lang en --seed 42 \
  --transcribe-only --port 8000 \
  --output-dir results/dots-seedtts-en-c16
```
