# Cookbook reference cases

## Qwen3-TTS

Lead with the supported CustomVoice path from installation and readiness to speech. The shared cURL request runs against that checkpoint, so the model page can link straight to it and name the resulting file. Keep the choices that change the request in the model page. Base needs reference inputs or an embedding, CustomVoice uses built-in speakers, and VoiceDesign uses a voice description. A sourced two-track model figure makes those choices easier to explain. Put tuning, platform setup, parameters, and benchmarks behind links when they make the first-use path hard to scan. Keep the published `#quick-start` and `#generate-speech` fragments working when the page headings change. [Shared speech guide](https://github.com/sgl-project/sglang-omni/pull/3674) · [Model boundary](https://github.com/sgl-project/sglang-omni/pull/3675) · [First-use order](https://github.com/sgl-project/sglang-omni/pull/3737) · [Architecture](https://github.com/sgl-project/sglang-omni/pull/3974) · [Shared first request](https://github.com/sgl-project/sglang-omni/pull/3920)

The Qwen3-TTS review found that removing a checked-in `--config` changed concurrency, the GPU memory fraction, and vocoder execution. A missing shell continuation also turned `--port` into a separate command. Compare commands during prose edits and verify the effective setup when a command changes. [Command review](https://github.com/sgl-project/sglang-omni/pull/3031#issuecomment-6385566650)

## Qwen3-ASR

Lead with the default server and a choice between file upload, streamed output, and live audio input. Share the multipart request, common response formats, and SSE client with ASR Model Usage. Keep the model's vocabulary bias, language hints, long-audio limits, realtime session details, and DFlash behavior with the model. Consolidate related Ascend NPU setup in one platform section and link to the platform installation guide. When adapting this pattern to another ASR model, verify translation, timestamps, and realtime support per backend before sending readers to a shared example. [Shared ASR guide](https://github.com/sgl-project/sglang-omni/pull/3700) · [Cookbook order](https://github.com/sgl-project/sglang-omni/pull/3980)

## Qwen3-Omni

Treat a buffered multimodal request, SSE output, and a persistent realtime session as three reader paths. Put shared request formats and modality examples in Omni Model Usage. Keep the supported modality matrix, Thinker-only and speech deployment choices, placement, and model-specific constraints in the Cookbook. Qualify readiness, warmup, and realtime event schemas by pipeline. The shared guide can point readers to the protocol for their deployment. When moving shared memory or readiness guidance, verify that the model's selector and compatibility matrix still describe the same combinations. [Shared Omni guide](https://github.com/sgl-project/sglang-omni/pull/3669) · [Protocol outline](https://github.com/sgl-project/sglang-omni/pull/3669#issuecomment-6725917645) · [Memory and readiness follow-up](https://github.com/sgl-project/sglang-omni/pull/4214)

## Zonos3

Use a compact page when it already explains the model well. Zonos3 keeps a launch command and text-only request on the page. It links to shared cloning, streaming, and SDK examples while stating the fields to adapt. It keeps the optional reference audio, transcript handling, 44.1 kHz PCM output, and model-specific limitations together with the recipe. This is a practical pattern for a model with one main serving path. [Cookbook at release](https://github.com/sgl-project/sglang-omni/blob/65f23ccc1/docs/cookbook/zonos3.md) · [Zonos3 release PR](https://github.com/sgl-project/sglang-omni/pull/3738)
