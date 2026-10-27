<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="logo" width="400"></img>

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
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
<a href="https://sgl-project.github.io/sglang-omni/index.html"><b>Cookbook</b></a> |
<a href="https://github.com/sgl-project/sglang"><b>SGLang</b></a> |
<a href="https://slack.sglang.io"><b>Join Slack</b></a>
</p>

<p align="center">
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">Star SGLang-Omni</a> to help more builders discover open infrastructure for multimodal and speech serving!</b>
</p>

## News

- [2026/10] 🗺️ A unified runtime for interactive intelligence: native full-duplex models, turn-based multimodal models, and cascaded systems, coordinated through composable graphs and persistent sessions. \[[Blog](https://www.sglang.io/blog/omni-interactive-runtime)\]
- [2026/10] 🚀 SGLang-Omni **v0.2.0** is on [PyPI](https://pypi.org/project/sglang-omni/). Install with `uv pip install --prerelease=allow "sglang-omni==0.2.0"`. \[[Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)\]
- [2026/10] 🐟 Day-0 support for [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3): multi-speaker text-to-speech with voice cloning and natural-language control, streamed on `/v1/audio/speech`. \[[Blog](https://www.sglang.io/blog/spg-topology)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)\] \[[Technical Report](https://github.com/fishaudio/fish-speech/tree/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf)\]
- [2026/10] 🎙️ Introducing [Voxt](https://github.com/sgl-project/sglang-omni/tree/main/Voxt), an open-source voice input tool for Mac, now integrating SGLang-Omni for faster local speech recognition on Apple Silicon. Combine ASR with LLM-based text refinement and personal system prompts for accurate technical terminology and personalized voice input, with fully local or remote inference options. [[Download](https://github.com/sgl-project/sglang-omni/releases#release-voxt-v1.16.0)][[Demo]](https://x.com/GenAI_is_real/status/2109419016205017181)

<details>
<summary>Earlier news</summary>

- [2026/09] 🐧 Day-0 support for [AuK](https://huggingface.co/tencent/AuK) and [AuK-Flash](https://huggingface.co/tencent/AuK-Flash): text + voice instructions → 24 kHz speech on `/v1/audio/speech`, audio + editing instructions → edited speech on `/generate`. \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/auk.html)\]
- [2026/08] 🎵 Day-0 support for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3): lyrics + caption → 32 kHz stereo song on `/v1/audio/speech`. \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html)\]
- [2026/08] 🚀 TTS architecture refactor: shared pipeline state, engine construction, reference encoding, capability metadata, and vocoder scheduling. \[[Roadmap](https://github.com/sgl-project/sglang-omni/issues/985)\] \[[Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-refactor.md)\]
- [2026/06] 🔥 MOSS-TTS Local Transformer v1.5 on SGLang-Omni with native-streaming 48 kHz speech. \[[Blog](https://lmsys.org/blog/2026-06-17-moss-tts-local-v15/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)\]
- [2026/06] 🔥 Higgs Audio v3 TTS for real-time, controllable speech. \[[Blog](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)\]

</details>

## About

SGLang-Omni is a multi-stage serving runtime for omni, speech, and TTS models. Its design target is multi-stage decoding: generation split across heterogeneous stages with different compute patterns, dependency structures, and resource needs. SGLang-Omni owns the pipeline topology, stage lifecycle, inter-stage transport, model-family integration layer, and OpenAI-compatible serving surface, while composing with [SGLang](https://github.com/sgl-project/sglang) for high-performance autoregressive scheduling and model execution where applicable.

- **Multi-stage runtime**: SGLang-Omni models generation as coordinated stages: preprocessing, encoders, autoregressive engines, talkers, decoders, vocoders, and aggregators.
- **Stage-specialized scheduling**: Each stage runs behind a scheduler matched to its workload, from SGLang-backed autoregressive scheduling to lightweight preprocessing and streaming vocoder loops.
- **Transport-aware execution**: A control plane coordinates requests while the relay data plane moves tensor payloads across shared-memory, NCCL, NIXL, and Mooncake backends.
- **API surface**: OpenAI-compatible endpoints expose multimodal chat, speech generation, batch speech, streaming speech, uploaded voices, and transcription.

## What SGLang-Omni Serves

- **Omni chat and speech**: [Qwen3-Omni](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_omni.html), [MiniCPM-o 4.5](https://sgl-project.github.io/sglang-omni/cookbook/minicpm_o.html), [Ming-Omni](https://sgl-project.github.io/sglang-omni/cookbook/ming_omni.html), [Ming-3.0-audio-flash](https://sgl-project.github.io/sglang-omni/cookbook/ming3_audio.html) — multimodal understanding and text/audio replies.
- **Native full-duplex conversation**: [MiniCPM-o 4.5](https://sgl-project.github.io/sglang-omni/cookbook/minicpm_o.html), [PersonaPlex](https://sgl-project.github.io/sglang-omni/cookbook/personaplex.html), [NemotronLabs VoiceChat](https://sgl-project.github.io/sglang-omni/cookbook/nemotron_voicechat.html) — live listening and speaking over `/v1/realtime`, including camera input with MiniCPM-o. [Full-duplex guide](https://sgl-project.github.io/sglang-omni/developer_reference/full_duplex_guide.html).
- **Cascaded voice assistants**: Compose ASR → LLM → TTS services into streaming, turn-based conversations with voice activity detection, conversation history, turn taking, and interruption. [Streaming cascade guide](https://sgl-project.github.io/sglang-omni/developer_reference/full_duplex_guide.html#streaming-cascade).
- **Image and video generation**: [LLaDA2.0-Uni](https://sgl-project.github.io/sglang-omni/cookbook/llada2_uni.html) for text/image understanding, image generation, and editing; [Cosmos3 Nano](https://sgl-project.github.io/sglang-omni/cookbook/cosmos3_nano.html) for interleaved understanding and generation, with native image, video, and action routes.
- **Music generation**: [MiniMax Music 3](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html) — lyrics + caption → 32 kHz stereo song.
- **Speech generation**: [Higgs Audio v3](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html), [MOSS-TTS](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts.html), [MOSS-TTS Local](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html), [Fish Audio Drama 3](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html), [Fish Speech S2-Pro](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html), [Qwen3-TTS](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html), [Fun-CosyVoice3](https://sgl-project.github.io/sglang-omni/cookbook/fun_cosyvoice3.html), [Voxtral TTS](https://sgl-project.github.io/sglang-omni/cookbook/voxtral_tts.html), [Ming-Omni-TTS](https://sgl-project.github.io/sglang-omni/cookbook/ming_tts.html), [Ming-3.0-TTS](https://sgl-project.github.io/sglang-omni/cookbook/ming3_tts.html), [dots.tts](https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html), [ZONOS2](https://sgl-project.github.io/sglang-omni/cookbook/zonos2.html) — `/v1/audio/speech`, with batch, streaming, and voice features varying by model.
- **Instruction-driven speech generation and editing**: [AuK and AuK-Flash](https://sgl-project.github.io/sglang-omni/cookbook/auk.html) — text and voice instructions → speech; audio and editing instructions → edited speech.
- **Audio transcription**: [Qwen3-ASR](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html), [Fun-ASR](https://sgl-project.github.io/sglang-omni/cookbook/fun_asr.html), [ARK-ASR](https://sgl-project.github.io/sglang-omni/cookbook/arkasr.html), [Nemotron 3.5 ASR](https://sgl-project.github.io/sglang-omni/cookbook/nemotron3_5_asr.html), [Ming-3.0-ASR](https://sgl-project.github.io/sglang-omni/cookbook/ming3_asr.html), experimental [Whisper](https://sgl-project.github.io/sglang-omni/cookbook/whisper_asr.html), and [MOSS-Transcribe-Diarize](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html) via `/v1/audio/transcriptions`. MOSS-TD also returns speaker labels and timestamps (`response_format=verbose_json`).
- **Speaker diarization**: [Nemotron 3 Diarization](https://sgl-project.github.io/sglang-omni/cookbook/nemotron_diarization.html) — speaker intervals for uploaded recordings and live audio, including overlapping speech.
- **Speech translation**: [Hy-SimulMT](https://sgl-project.github.io/sglang-omni/cookbook/hy_simulmt.html) — Chinese speech → English text and English speech → Chinese text, with file and realtime translation. See the [audio translation guide](https://sgl-project.github.io/sglang-omni/basic_usage/audio_translations.html) for endpoint and model support.

## Hardware Support

| Backend | Status | Notes |
|---------|--------|-------|
| **NVIDIA CUDA** | Supported | Default backend with full model coverage. |
| **Apple Silicon** | Experimental | Qwen3-ASR runs through Torch MPS on macOS arm64. Install with [`install.sh`](./install.sh) and follow the [Qwen3-ASR guide](./docs/cookbook/qwen3_asr.md#apple-silicon). |
| **Intel GPU (XPU)** | Experimental | Intel Arc GPUs via PyTorch XPU. **Qwen3-ASR, Qwen3-TTS, ZONOS2, and Qwen3-Omni serve end-to-end** (ZONOS2 with decode graphs; Omni thinker via multi-XPU tensor parallelism). Install per [Intel XPU guide](./docs/get_started/installation_xpu.md); the backend is auto-detected. |

Additional model guides, including experimental and research-oriented paths, are available in the [Cookbook](https://sgl-project.github.io/sglang-omni/).

## Quick Start

- [Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)
- [TTS usage](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html)
- [Qwen3-Omni usage](https://sgl-project.github.io/sglang-omni/basic_usage/qwen3_omni.html)
- [Qwen3-ASR cookbook](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html)
- [Nemotron 3.5 ASR cookbook](https://sgl-project.github.io/sglang-omni/cookbook/nemotron3_5_asr.html)
- [MOSS-Transcribe-Diarize cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html)
- [Omni router](https://sgl-project.github.io/sglang-omni/basic_usage/omni_router.html)
- [Developer reference](https://sgl-project.github.io/sglang-omni/developer_reference/main.html)

## Community & Support

SGLang-Omni welcomes contributors working on inference systems, kernels, scheduling, inter-stage communication, model runners and cache efficiency, model integration, benchmarking, production deployment. Join the [SGLang Slack](https://slack.sglang.io) or read the [developer reference](https://sgl-project.github.io/sglang-omni/developer_reference/main.html).

Organizations interested in supporting SGLang-Omni, TTS, or omni model serving can contact Chenyang Zhao at [zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org).

## Citation

If you use SGLang-Omni for your research, please cite our technical blogs on the [unified interactive runtime](https://www.sglang.io/blog/omni-interactive-runtime) and the [Stage--Process--GPU topology](https://www.sglang.io/blog/spg-topology):

```bibtex
@misc{sglangomniinteractive2026,
  title={Unified Runtime for Full-Duplex Interactive Intelligence with {SGLang Omni}},
  author={{SGLang-Omni Team}},
  howpublished={\url{https://www.sglang.io/blog/omni-interactive-runtime}},
  year={2026},
  month={oct}
}

@misc{sglangomni2026,
  title={Fish Audio Drama 3 Meets {SGLang-Omni}: Day-0 Support and the Stage--Process--GPU Topology},
  author={{SGLang-Omni Team}},
  howpublished={\url{https://www.sglang.io/blog/spg-topology}},
  year={2026},
  month={oct}
}
```

## Acknowledgments

SGLang-Omni builds on the SGLang ecosystem and on open model work from the TTS, speech, and omni-model communities. We thank the model teams, systems contributors, and partner organizations helping make open multimodal serving faster, more reliable, and easier to extend.
