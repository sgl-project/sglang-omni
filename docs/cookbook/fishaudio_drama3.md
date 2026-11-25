# Fish Audio Drama 3

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) is Fish Audio's text-to-speech model for multi-character scenes and long-form content. It supports multi-speaker generation, voice cloning and natural-language control, and SGLang-Omni serves it through the OpenAI-compatible `/v1/audio/speech` endpoint.

![Fish Audio Drama 3 overview](../_static/image/fishaudio-drama3-overview.png)

- **35B parameters, 3B active per token**: a mixture-of-experts backbone with a codebook-MTP audio decoder; the [technical report](https://github.com/fishaudio/fish-speech/tree/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf) has the architecture
- **37.0 GB of engine weights** in the default FP8 checkpoint; the [int4 checkpoint](#int4-checkpoint) needs 20.8 GB
- **Speaker tags** `<|speaker:0|>` … `<|speaker:15|>` — one text can carry up to 16 characters
- **Long text** is chunked automatically and can be streamed per chunk
- **Shared request fields, streaming and clients** are in [TTS Model Usage](../basic_usage/tts.md); this page covers the CUDA server, and the native Apple Silicon server is under [Platform deployment](fishaudio_drama3/platforms.md)

Figure from the [Fish Audio Drama 3 Technical Report](https://github.com/fishaudio/fish-speech/tree/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf), reproduced from the report's vector drawing.

<span id="installation-and-startup"></span>

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md). Then, from the repository root, add the Fish Audio dependencies:

```bash
uv pip install --prerelease=allow -v -e ".[fishaudio]"
```

Download the checkpoint:

```bash
hf download fishaudio/drama-3
```

The default checkpoint holds 37.0 GB of engine weights, and the GPU must also fit the KV cache, the preprocessing stage and the vocoder. [Choose a model](#choose-a-model) lists the smaller int4 checkpoint and the multi-GPU option.

## Server Configuration

Drama 3 runs as three stages: preprocessing, which prepares the text and any reference audio; the engine, which generates the audio codec frames; and the vocoder, which turns those frames into audio. The stages can sit on different GPUs. This command places all three on GPU 0. A stage moves to another GPU only when its `--<stage>.gpu` flag says so; see [Deployment Performance](fishaudio_drama3/tuning.md#deployment-performance).

```bash
sgl-omni serve \
  --model-path fishaudio/drama-3 \
  --port 8000
```

Startup captures CUDA graphs, so it takes longer than loading the weights alone. In another terminal, wait until `/health` returns HTTP 200:

```bash
curl -i http://localhost:8000/health
```

<span id="attention-backend"></span>

Drama 3 runs its attention on the `triton` backend only. The server refuses to start with any other attention backend, for example `--tts_engine.engine.attention_backend fa3`.

<span id="synthesize-speech"></span>

## Synthesizing Speech

Plain TTS:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?"
  }' \
  --output output.wav
```

Open `output.wav` in an audio player. The response is a WAV file by default. For Python or the OpenAI SDK, use the [TTS client examples](../basic_usage/tts.md#python) with the request fields on this page.

## Choose a model

| Checkpoint | Engine weights | Use it when |
|---|---|---|
| [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) | 37.0 GB | The default: FP8 MoE experts, remaining weights in bf16. |
| [`fishaudio/drama-3-int4`](https://huggingface.co/fishaudio/drama-3-int4) | 20.8 GB | GPU memory is tight. 4-bit GPTQ weights on GPTQ-Marlin kernels; needs compute capability 8.0 or newer, such as A100. |
| `mlx-community/fish-audio-drama-3-4bit` and `-8bit` | 24 GB and 42 GB on disk | Apple Silicon, through the native MLX server. See [Platform deployment](fishaudio_drama3/platforms.md). |

The int4 checkpoint is about 8% slower per request and its WER stays at the FP8 level; [Benchmarks](fishaudio_drama3/benchmarks.md#fp8-and-int4-checkpoints) has the measurements. When the FP8 checkpoint does not fit one GPU, `--tts_engine.tp_size` splits the engine across 2 or 4 GPUs; see [Tensor Parallelism](fishaudio_drama3/tuning.md#tensor-parallelism).

### Int4 Checkpoint

[`fishaudio/drama-3-int4`](https://huggingface.co/fishaudio/drama-3-int4) is a 4-bit GPTQ quantization of Drama 3. Thanks so much to [AK](https://x.com/_akhaliq) for the quantization! The server reads its quantization from the checkpoint, so it starts with the same command:

```bash
hf download fishaudio/drama-3-int4

sgl-omni serve \
  --model-path fishaudio/drama-3-int4 \
  --port 8000
```

The engine's weights take 20.8 GB of GPU memory instead of 37.0 GB, which leaves more of the engine's share for the KV cache. GPTQ-Marlin needs compute capability 8.0 or newer, such as A100. [Int4 weight layout](fishaudio_drama3/reference.md#int4-weight-layout) lists which modules are quantized and which stay in bf16.

<span id="example-usage"></span>

## Voice modes

Drama 3 takes the shared speech request fields, documented in [Request Parameters](../basic_usage/tts.md#request-parameters). The modes below are what sets it apart. Every sample on this page clones the same reference voice, `en0`. It is a 3.9 s clip, `common_voice_en_10119832.wav`, the file the cloning example sends. Its transcript is `We asked over twenty different people, and they all said it was his.` Listen to it here:

<audio controls>
  <source src="https://github.com/user-attachments/files/33393249/en0.wav" type="audio/wav">
</audio>

### Voice cloning

Send a reference clip and its transcript in `references`:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output_voice_cloning.wav
```

Each reference may be at most 60 seconds long by default, and a request may carry any number of references within the prompt limit; see [Known limitations](#known-limitations). For the accepted audio sources and the `ref_audio` and `ref_text` shorthand, see [Voice Cloning](../basic_usage/tts.md#voice-cloning).

### Point tags

A point tag is a cue in square brackets, such as `[sighs]` or `[laughs]`, written into the text at the position where the event should land. Figure 1 shows the same idea as inline direction. This page documents the two tags in the sample; the [technical report](https://github.com/fishaudio/fish-speech/tree/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf) describes fine-grained direction.

The sample is a plain `/v1/tts` request rather than a `/v1/audio/speech` one, so the text goes in `text`. It sends one reference (`en0`) and no history:

```json
{
  "text": "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.",
  "references": [{"audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav", "text": "We asked over twenty different people, and they all said it was his."}]
}
```

<audio controls>
  <source src="https://github.com/user-attachments/files/33393205/A1_point_tags.wav" type="audio/wav">
</audio>

### Editing as continuation

Pass the text of earlier takes in `previous_takes`. The model reads them as context and generates the new text as a continuation, so an edited line stays consistent with the take it replaces.

```json
{
  "text": "The quarterly report is due on Thursday, so please finish it by noon.",
  "references": [{"audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav", "text": "We asked over twenty different people, and they all said it was his."}],
  "previous_takes": ["The quarterly report is due on Wednesday, so please finish it by noon."]
}
```

<audio controls>
  <source src="https://github.com/user-attachments/files/33393156/K1_edit_word_with_context.wav" type="audio/wav">
</audio>

A second edit can chain onto the first. Here `by noon` becomes `by five`, with both earlier takes in the context:

```json
{
  "text": "The quarterly report is due on Thursday, so please finish it by five.",
  "references": [{"audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav", "text": "We asked over twenty different people, and they all said it was his."}],
  "previous_takes": [
    "The quarterly report is due on Wednesday, so please finish it by noon.",
    "The quarterly report is due on Thursday, so please finish it by noon."
  ]
}
```

<audio controls>
  <source src="https://github.com/user-attachments/files/33393164/K5_second_edit_chained.wav" type="audio/wav">
</audio>

## Streaming

Add `"stream": true` and `"response_format": "pcm"` to either request above to receive audio while generation is still running. Long text is chunked automatically and streamed chunk by chunk. With the server running, use the [raw WAV and PCM examples](../basic_usage/tts.md#raw-wav-and-pcm) in TTS Model Usage; Drama 3 needs no `model` or `voice` field. The guide also includes the [Python streaming client](../basic_usage/tts.md#python-streaming-client) and [SSE](../basic_usage/tts.md#sse-event-framing) clients.

For many streams at once, see [High Concurrency](fishaudio_drama3/tuning.md#high-concurrency); under sustained load, the opt-in [incremental streaming codec](fishaudio_drama3/tuning.md#incremental-streaming-codec) reuses each stream's decoder state.

## Common options

Shared fields such as `input`, `response_format`, `stream` and the sampling controls are described in [Request Parameters](../basic_usage/tts.md#request-parameters). Drama 3 applies these defaults and model-specific fields:

| Field | Default | Drama 3 behavior |
|---|---|---|
| `temperature` | `1.0` | Shared field. |
| `top_p` | `0.9` | Shared field. |
| `top_k` | unset | Limits how many candidate tokens the sampler may pick from. Leave it unset: the main token is then drawn with no top-k cap, and a cap of 30 lets a stream fall silent after its first phrase. The codebook heads always cap at 30, whatever the request asks. |
| `max_new_tokens` | the remaining context, up to 2048 frames | A prompt and its `max_new_tokens` must fit the engine's context length. |
| `references` | unset | Any number of clips within the prompt limit; each item has `audio_path` and `text`. |
| `previous_takes` | unset | Earlier takes for [editing as continuation](#editing-as-continuation). |
| `epsilon_cutoff` | unset | Accepted only with reference sampling on; otherwise a non-zero value is refused. See [Defaults Parameters](fishaudio_drama3/tuning.md#defaults-parameters). |
| `seed` | not supported | The server refuses a request that sets it. |

Engine-side defaults, such as CUDA graphs, Triton sampling and vocoder placement, are server settings rather than request fields. [Defaults Parameters](fishaudio_drama3/tuning.md#defaults-parameters) lists each one with the flag that changes it.

<span id="serving-limits"></span>

### Known limitations

- **Attention backend.** Only `triton`; any other `--tts_engine.engine.attention_backend` stops startup.
- **Reference audio length.** Each reference given as audio may be at most 60 seconds long; a longer one fails the request with HTTP 400 before it is encoded. Raise the limit with `--preprocessing.factory.max_reference_audio_seconds` and the preprocessing reserve with it; see [Reference Audio Length](fishaudio_drama3/reference.md#reference-audio-length).
- **Prompt length.** A prompt may have at most 8192 rows, counting the text tokens and about 21.5 rows per second of reference audio. A longer one fails with HTTP 400. Change the cap with `--tts_engine.factory.max_prompt_rows`; see [Prompt Length](fishaudio_drama3/reference.md#prompt-length).
- **Hopper-only kernels.** The fastest sampling and decoder kernels run only on H100 and H200 class GPUs (compute capability 9.0). Other GPUs use the standard PyTorch sampler and decoder layers instead, and the startup log says which path is in use. See [Kernel fallbacks](fishaudio_drama3/tuning.md#kernel-fallbacks).
- **Request `seed`.** Not supported; the server refuses a request that sets it.
- **Native MLX server.** Reference audio only as base64 WAV, at most 4 references of 30 s together, one request at a time. See [Platform deployment](fishaudio_drama3/platforms.md).

## Advanced Guides

<span id="deployment"></span>
<span id="deployment-performance"></span>
<span id="tensor-parallelism"></span>
<span id="high-concurrency"></span>
<span id="production-serving"></span>
<span id="non-streaming-requests"></span>
<span id="defaults-parameters"></span>
<span id="apple-silicon-native-mlx"></span>
<span id="optional-native-expert-offload"></span>
<span id="reference-parity-and-performance"></span>
<span id="incremental-streaming-codec"></span>
<span id="reference-audio-length"></span>
<span id="prompt-length"></span>

- [Tuning](fishaudio_drama3/tuning.md): vocoder placement, tensor parallelism, high concurrency, non-streaming decode limits, engine defaults, kernel fallbacks, and the incremental streaming codec.

- [Reference](fishaudio_drama3/reference.md): the reference-length and prompt-length limits in detail, and the int4 weight layout.

- [Platform deployment](fishaudio_drama3/platforms.md): the native Apple Silicon MLX server and expert offload.

- [Benchmarks](fishaudio_drama3/benchmarks.md): streaming and non-streaming measurements on H200, the FP8 and int4 comparison, tensor-parallel layouts, and native MLX parity.
