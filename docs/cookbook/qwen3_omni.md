# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) accepts text, images, audio, and video, and returns text or text with speech through the OpenAI-compatible `/v1/chat/completions` endpoint. SGLang-Omni serves it in [text-only](#text-only), [speech](#speech), and [realtime](#realtime) modes.

The thinker generates the answer from the multimodal input. In speech mode, the talker generates codec tokens alongside that answer, and Code2Wav decodes them into speech. Text-only mode skips the speech stages while keeping multimodal input.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then run the commands from the repository root. The following example uses Linux with one NVIDIA H200 GPU. For other hardware and precisions, see [platform deployment](qwen3_omni/platforms.md).

## Server Configuration

This example uses the BF16 Instruct checkpoint with the single-H200 colocated speech configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h200.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

The first startup includes compilation and warmup. Then, in another terminal, run the following command and wait until `/health` returns HTTP 200:

```bash
curl -i http://localhost:8008/health
```

For other GPU layouts, use the [launch command selector](qwen3_omni/platforms.md#server-configuration). Colocated deployments need the hardware-specific config to set per-stage memory budgets.

## Generating Responses

With the speech server running, request an answer in text and audio:

```bash
curl --fail-with-body http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say one friendly sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"},
    "max_tokens": 64
  }' \
  --output response.json
```

The response is JSON. The answer is in `choices[0].message.content`, and the WAV audio is base64-encoded in `choices[0].message.audio.data`. Use the [Omni speech client](../basic_usage/omni.md#speech-output) to decode it to a playable file. For Python and the OpenAI SDK, see the [shared client examples](../basic_usage/omni.md#text-output).

<span id="compatibility-matrix"></span>

## Choose a mode

| Goal | Mode | What to configure |
|---|---|---|
| Answer questions with text | [Text-only](#text-only) | Add `--text-only`; send `modalities: ["text"]` |
| Answer with text and speech | [Speech](#speech) | Launch the speech pipeline; send `modalities: ["text", "audio"]` |
| Build a turn-based voice assistant | [Realtime](#realtime) | Choose a `/v1/realtime` deployment and its session protocol |

Use text-only mode when audio output is unnecessary. See the [compatibility matrix](qwen3_omni/reference.md#compatibility-matrix) for supported topology, tensor parallelism, and precision combinations.

<span id="input-output-modalities"></span>

## Output modes

<span id="text-only-deployment"></span>
<span id="launch-a-text-only-server"></span>

### Text-only

Text-only mode accepts text, image, audio, and video input and returns text. Start the thinker pipeline with:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --model-name qwen3-omni \
  --port 8008
```

Use the [shared media request examples](../basic_usage/omni.md#general-usage) with `modalities: ["text"]`. For preprocessing concurrency, video decoding threads, and the fused MMSU text path, see [Text-Only Deployment](qwen3_omni/platforms.md#text-only-deployment).

<span id="speech-deployment-and-tuning"></span>
<span id="launch-a-speech-server"></span>

### Speech

The [server configuration](#server-configuration) starts the speech pipeline. All four input modalities remain available. Request `modalities: ["text", "audio"]` for text with speech, or `["text"]` for text alone. See [speech deployment](qwen3_omni/platforms.md#speech-deployment) for H20 and multi-GPU launch commands.

<span id="realtime-speech-with-server-side-turn-detection"></span>
<span id="qwen3-omni-persistent-session-stages"></span>

### Realtime

Qwen3-Omni supports turn-based voice sessions on `/v1/realtime`. The standard speech pipeline uses `--enable-realtime` for server-side turn detection and barge-in; the persistent session profile uses a VAD node to end each user turn. See [Realtime Sessions](qwen3_omni/platforms.md#realtime-sessions) for both deployment paths and their protocol references.

<span id="single-gpu-fp8-on-h100h20"></span>
<span id="single-gpu-autoround-int4-thinker-on-h100h20"></span>
<span id="single-gpu-fp8-on-h100-h20"></span>
<span id="single-gpu-autoround-int4-thinker-on-h100-h20"></span>

### Model variants

| Checkpoint | Precision | Deployment guide |
|---|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | BF16 | [Speech deployment](qwen3_omni/platforms.md#speech-deployment) |
| `marksverdhei/Qwen3-Omni-30B-A3B-FP8` | Native FP8 thinker and talker | [Single-GPU FP8](qwen3_omni/platforms.md#single-gpu-fp8-on-h100h20) |
| `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` | INT4 thinker; BF16 talker and Code2Wav | [Single-GPU AutoRound INT4](qwen3_omni/platforms.md#single-gpu-autoround-int4-thinker-on-h100h20) |

## Streaming

With the speech server running, send a streaming request:

```bash
curl --fail-with-body -N http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say one friendly sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"},
    "stream": true,
    "max_tokens": 64
  }'
```

This prints Server-Sent Events. Text arrives in `choices[0].delta.content` and audio in `choices[0].delta.audio.data`. For decoding audio chunks and handling errors, use the [shared streaming client](../basic_usage/omni.md#streaming). For first-audio latency settings, see [codec coalescing](qwen3_omni/tuning.md#codec-coalescing-and-first-audio-latency).

<span id="sampling-parameters"></span>

## Common options

With the speech server running, set the text and speech generation limits separately:

```bash
curl --fail-with-body http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Explain why the sky is blue in one sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"},
    "temperature": 0.0,
    "max_tokens": 128,
    "talker_temperature": 0.9,
    "talker_max_new_tokens": 1024
  }' \
  --output response.json
```

`max_tokens` limits the thinker's answer; `talker_max_new_tokens` limits the talker's codec generation. Allow enough tokens for the whole reply. See [Qwen3-Omni sampling parameters](qwen3_omni/reference.md#sampling-parameters) for defaults, and the [shared Omni API](../basic_usage/omni.md#request-parameters) for common fields.

### Known limitations

- **Speech requires the speech pipeline.** A text-only chat server accepts an audio modality request but returns text without an error.
- **Context limits cover the prompt and output.** Long inputs can leave too little room for the spoken reply. See [context limits](qwen3_omni/reference.md#known-limitations).
- **Colocated topology supports thinker TP=1.** Use disaggregated topology for `--thinker.tp_size 2`.
- **Video input has frame-rate and audio-track constraints.** See [input modalities](qwen3_omni/reference.md#input-and-output-modalities) and [known limitations](qwen3_omni/reference.md#known-limitations).

## Advanced Guides

<span id="memory-and-concurrency"></span>
<span id="codec-coalescing-and-first-audio-latency"></span>
<span id="code2wav-execution"></span>
<span id="speech-stage-placement"></span>
<span id="server-readiness"></span>
<span id="speculative-decoding"></span>

- [Tuning](qwen3_omni/tuning.md): readiness, concurrency, first-audio latency, Code2Wav execution, GPU placement, and speculative decoding.

- [Reference](qwen3_omni/reference.md): compatibility matrix, sampling defaults, modality behavior, and model-specific limitations.

- [Platform deployment](qwen3_omni/platforms.md): launch command selector, text and speech deployments, FP8 and INT4 recipes, and realtime profiles.

- [Benchmarks](qwen3_omni/benchmarks.md): speech stage placement measurements with their concurrency and single-stream trade-offs.

```{toctree}
---
hidden: true
maxdepth: 1
---
qwen3_omni/tuning
qwen3_omni/reference
qwen3_omni/platforms
qwen3_omni/benchmarks
```
