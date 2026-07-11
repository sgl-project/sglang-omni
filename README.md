<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="logo" width="400"></img>

<p>
<a href="https://github.com/sgl-project/sglang-omni/stargazers"><img src="https://img.shields.io/github/stars/sgl-project/sglang-omni?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-raw/sgl-project/sglang-omni?style=for-the-badge&label=open%20issues" alt="open issues"></a>
<a href="https://deepwiki.com/sgl-project/sglang-omni"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
</p>

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="#quick-start"><b>Quick Start</b></a> |
<a href="./docs/cookbook/"><b>Cookbook</b></a> |
<a href="https://github.com/sgl-project/sglang"><b>SGLang</b></a> |
<a href="https://slack.sglang.io"><b>Join Slack</b></a>
</p>

<p align="center">
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">Star SGLang-Omni</a> to help more builders discover open infrastructure for multimodal and speech serving!</b>
</p>

## News

- [2026/06] 🔥 MOSS-TTS Local Transformer v1.5 runs on SGLang-Omni with native-streaming 48 kHz speech. \[[Blog](https://lmsys.org/blog/2026-06-17-moss-tts-local-v15/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)\]
- [2026/06] 🔥 Higgs Audio v3 TTS runs on SGLang-Omni for real-time, controllable speech for voice agents. \[[Blog](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)\]

## About

SGLang-Omni is a multi-stage serving runtime for omni, speech, and TTS models. Its design target is multi-stage decoding: generation split across heterogeneous stages with different compute patterns, dependency structures, and resource needs. SGLang-Omni owns the pipeline topology, stage lifecycle, inter-stage transport, model-family integration layer, and OpenAI-compatible serving surface, while composing with [SGLang](https://github.com/sgl-project/sglang) for high-performance autoregressive scheduling and model execution where applicable.

- **Multi-stage runtime**: SGLang-Omni models generation as coordinated stages: preprocessing, encoders, autoregressive engines, talkers, decoders, vocoders, and aggregators.
- **Stage-specialized scheduling**: Each stage runs behind a scheduler matched to its workload, from SGLang-backed autoregressive scheduling to lightweight preprocessing and streaming vocoder loops.
- **Transport-aware execution**: A control plane coordinates requests while the relay data plane moves tensor payloads across shared-memory, NCCL, NIXL, and Mooncake backends.
- **API surface**: OpenAI-compatible endpoints expose multimodal chat, speech generation, batch speech, streaming speech, uploaded voices, and transcription.

## What SGLang-Omni Serves

- **Omni chat and speech**: Run models such as [Qwen3-Omni](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_omni.html) and [Ming-Omni](https://sgl-project.github.io/sglang-omni/cookbook/ming_omni.html) with multimodal inputs, text/audio outputs, and thinker-talker generation pipelines.
- **Speech generation**: Serve [Higgs Audio v3](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html), [MOSS-TTS](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts.html), [MOSS-TTS Local](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html), [Fish Speech S2-Pro](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html), [Qwen3-TTS](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html), [Voxtral TTS](https://sgl-project.github.io/sglang-omni/cookbook/voxtral_tts.html), and related TTS systems through speech, batch speech, streaming speech, and uploaded-voice APIs.
- **Audio transcription and diarization**: Serve [Qwen3-ASR](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html) and [MOSS-Transcribe-Diarize](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html) through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. MOSS-TD supports speaker-labelled, timestamped transcript segments with `response_format=verbose_json`.
- **SGLang-Omni Router**: Serve multiple Omni servers behind one OpenAI-compatible endpoint, with health checks, readiness tracking, worker lifecycle control, and model-capability discovery across the worker pool. See the [Router guide](https://sgl-project.github.io/sglang-omni/basic_usage/omni_router.html).

Additional model guides, including experimental and research-oriented paths, are available in the [Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/).

## Quick Start

- [Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)
- [TTS usage](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html)
- [Qwen3-Omni usage](https://sgl-project.github.io/sglang-omni/basic_usage/qwen3_omni.html)
- [Qwen3-ASR cookbook](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html)
- [MOSS-Transcribe-Diarize cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html)
- [Omni router](https://sgl-project.github.io/sglang-omni/basic_usage/omni_router.html)
- [Developer reference](https://sgl-project.github.io/sglang-omni/developer_reference/main.html)

## Community & Support

SGLang-Omni welcomes contributors working on inference systems, kernels, scheduling, inter-stage communication, model runners and cache efficiency, model integration, benchmarking, production deployment. Join the [SGLang Slack](https://slack.sglang.io) or read the [developer reference](https://sgl-project.github.io/sglang-omni/developer_reference/main.html).

Organizations interested in supporting SGLang-Omni, TTS, or omni model serving can contact Chenyang Zhao at [zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org).

## Acknowledgments

SGLang-Omni builds on the SGLang ecosystem and on open model work from the TTS, speech, and omni-model communities. We thank the model teams, systems contributors, and partner organizations helping make open multimodal serving faster, more reliable, and easier to extend.
