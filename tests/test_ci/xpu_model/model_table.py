# SPDX-License-Identifier: Apache-2.0
"""Every model family Intel XPU serves, shared by the contract tests, the serving lane and the baseline.

A row's serve_args are its recipe from docs/other_platforms/installation_xpu.md, with stage gpu ids relative to the ZE_AFFINITY_MASK the lane sets; device_args add what one card family needs on top, and serving says why the lane skips a row.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ROOTS_ENV = "MODEL_ROOTS"
ALLOW_DOWNLOAD_ENV = "OMNI_XPU_ALLOW_HF_DOWNLOAD"
# Names the device family outright, for a card whose name the lane does not know.
DEVICE_FAMILY_ENV = "OMNI_XPU_DEVICE_FAMILY"

KNOWN_FAMILIES = ("bmg", "d270")
# torch names a D270 card "Intel(R) Graphics [0x674c]"; the PCI id is the only stable part.
D270_PCI_IDS = ("0x674c",)


def device_family(device_name: str, override: str = "") -> str:
    """Which recipe family a card belongs to: d270, bmg, or unknown, which gets the base recipe."""
    if override:
        return override
    else:
        lowered = device_name.lower()
    if any(pci_id in lowered for pci_id in D270_PCI_IDS):
        return "d270"
    elif "arc(tm)" in lowered or "battlemage" in lowered:
        return "bmg"
    else:
        return "unknown"


@dataclass(frozen=True)
class XpuModel:
    """One XPU-served checkpoint: how to build its pipeline and how to ask it."""

    key: str
    package: str
    architecture: str
    variant: str | None
    stages: tuple[str, ...]
    repo_id: str
    cards: int
    kind: str
    serve_args: tuple[str, ...] = ()
    # Pairs rather than a dict so the row stays hashable and frozen.
    env: tuple[tuple[str, str], ...] = ()
    startup_seconds: int = 900
    # For a first request that does one-off work; the lane waits the longer of this and its default.
    request_seconds: int = 0
    # (family, extra serve args) pairs, applied after serve_args on that family.
    device_args: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Why the device lane cannot serve this one; empty when it can.
    serving: str = ""
    # The checkpoint revision the recipe was validated on; empty follows main.
    revision: str = ""
    # Further checkpoints the server loads by repo id, which must be cached before it starts.
    companions: tuple[str, ...] = ()

    def serve_args_on(self, family: str) -> tuple[str, ...]:
        """The full recipe for a card of the given family."""
        extra: tuple[str, ...] = ()
        for name, args in self.device_args:
            if name == family:
                extra += args
            else:
                continue
        return self.serve_args + extra

    @property
    def checkpoint_spec(self) -> str:
        """The checkpoint's identity: the repo id, with the pinned revision when there is one."""
        return f"{self.repo_id}@{self.revision}" if self.revision else self.repo_id

    @property
    def tp_size(self) -> int:
        """The largest tensor-parallel degree any stage of this row is served at."""
        sizes = [
            int(self.serve_args[index + 1])
            for index, arg in enumerate(self.serve_args[:-1])
            if arg.endswith(".tp_size")
        ]
        return max(sizes, default=1)

    @property
    def pipeline_key(self) -> str:
        """Identity of the *pipeline*, which several checkpoints can share."""
        return f"{self.package}:{self.variant or 'default'}"

    @property
    def mirror_dirname(self) -> str:
        return self.repo_id.split("/")[-1]

    @property
    def checkpoint_env(self) -> str:
        """Environment variable that points this row at a local copy of its checkpoint."""
        return f"OMNI_XPU_{self.key.upper().replace('-', '_')}_MODEL"


XPU_MODELS: tuple[XpuModel, ...] = (
    XpuModel(
        key="qwen3-asr",
        package="qwen3_asr",
        architecture="Qwen3ASRForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="Qwen/Qwen3-ASR-1.7B",
        cards=1,
        kind="transcription",
        env=(("SGLANG_ENABLE_METADATA_GLUE_GRAPH", "0"),),
    ),
    XpuModel(
        key="whisper",
        package="whisper_asr",
        architecture="WhisperForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="openai/whisper-large-v3",
        cards=1,
        kind="transcription",
    ),
    XpuModel(
        key="fun-asr",
        package="fun_asr",
        architecture="FunAsrNanoForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="FunAudioLLM/Fun-ASR-Nano-2512-hf",
        cards=1,
        kind="transcription",
    ),
    XpuModel(
        key="moss-transcribe-diarize",
        package="moss_transcribe_diarize",
        architecture="MossTranscribeDiarizeForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        cards=1,
        kind="transcription",
        serve_args=(
            "--asr.engine.max_running_requests",
            "16",
            "--asr.engine.cuda_graph_max_bs",
            "16",
            "--asr.engine.enable_torch_compile",
            "false",
            "--mem-fraction-static",
            "0.70",
        ),
    ),
    XpuModel(
        key="nemotron35-asr",
        package="nemotron3_5_asr",
        architecture="Nemotron3_5AsrForRNNT",
        variant=None,
        stages=("asr",),
        repo_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        cards=1,
        kind="transcription",
    ),
    XpuModel(
        key="nemotron-transcribe",
        package="nemotron_transcribe",
        architecture="Nemotron3_5TranscribeForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        # 62.9 GiB of bf16 weights: TP=4 is the smallest layout that fits 24 GB cards.
        cards=4,
        kind="transcription",
        serve_args=(
            "--asr.tp_size",
            "4",
            "--asr.gpu",
            "[0, 1, 2, 3]",
            "--mem-fraction-static",
            "0.75",
        ),
        startup_seconds=1800,
        # The first requests tune the decoder's mamba kernels: 108 to 232 s on an idle B60.
        request_seconds=600,
    ),
    XpuModel(
        key="arkasr",
        package="arkasr",
        architecture="ArkasrForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="AutoArk-AI/ARK-ASR-3B",
        cards=1,
        kind="transcription",
    ),
    XpuModel(
        key="ming3-asr-tiny",
        package="ming3_asr",
        architecture="MingV3ASRForConditionalGeneration",
        variant=None,
        stages=("preprocessing", "thinker"),
        repo_id="inclusionAI/Ming-3.0-asr-tiny",
        # About 22 GB of a 24 GB card, so it needs an otherwise idle one.
        cards=1,
        kind="transcription",
        # A cold cache tunes the linear-attention kernels on the first request: 279 s on a B60.
        request_seconds=900,
    ),
    XpuModel(
        key="hy-simulmt",
        package="hy_simulmt",
        architecture="HYV3SpeechForCausalLM",
        variant=None,
        stages=("asr",),
        repo_id="tencent/Hy-SimulMT-30B-A3B",
        # 57 GiB of bf16 weights: TP=4 is the smallest layout that fits 24 GB cards.
        cards=4,
        kind="translation",
        serve_args=(
            "--asr.tp_size",
            "4",
            "--asr.gpu",
            "[0, 1, 2, 3]",
            "--mem-fraction-static",
            "0.75",
            "--enable-realtime",
        ),
        startup_seconds=1800,
    ),
    XpuModel(
        key="qwen3-tts",
        package="qwen3_tts",
        architecture="Qwen3TTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "vocoder",
            "tts_engine",
        ),
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        cards=1,
        kind="speech_cloned",
        # Only ref_audio reaches the batcher whose private-stream capture loses a D270 device.
        device_args=(("d270", ("--tts_engine.engine.disable_cuda_graph", "true")),),
    ),
    XpuModel(
        key="qwen3-tts-cv",
        package="qwen3_tts",
        architecture="Qwen3TTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "vocoder",
            "tts_engine",
        ),
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        cards=1,
        kind="speech_builtin_voice",
        serve_args=(
            "--config",
            "examples/configs/qwen3_tts_1_7b_customvoice.yaml",
        ),
    ),
    XpuModel(
        key="qwen3-tts-vd",
        package="qwen3_tts",
        architecture="Qwen3TTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "vocoder",
            "tts_engine",
        ),
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        cards=1,
        kind="speech_voice_design",
        serve_args=(
            "--config",
            "examples/configs/qwen3_tts_1_7b_voicedesign.yaml",
        ),
    ),
    XpuModel(
        key="higgs",
        package="higgs_tts",
        architecture="HiggsMultimodalQwen3ForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "audio_encoder",
            "tts_engine",
            "vocoder",
        ),
        repo_id="bosonai/higgs-audio-v3-tts-4b",
        cards=1,
        kind="speech_default_voice",
        serve_args=(
            "--allowed-local-media-path",
            "docs/_static/audio",
        ),
        # A cold torch.compile cache makes this row's startup far longer than a warm one.
        startup_seconds=2400,
    ),
    XpuModel(
        key="ming-tts",
        package="ming_tts",
        architecture="BailingMMNativeForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "reference_encode",
            "tts_engine",
            "audio_decode",
        ),
        repo_id="inclusionAI/Ming-omni-tts-16.8B-A3B",
        cards=2,
        kind="speech_default_voice",
        serve_args=(
            "--config",
            "examples/configs/ming_omni_tts.yaml",
            "--tts_engine.tp_size",
            "2",
            "--tts_engine.gpu",
            "[0, 1]",
            "--reference_encode.gpu",
            "0",
            "--audio_decode.gpu",
            "0",
            "--tts_engine.gpu_memory_fraction",
            "0.78",
            "--tts_engine.engine.mem_fraction_static",
            "0.78",
        ),
        startup_seconds=1800,
    ),
    XpuModel(
        key="dots-tts",
        package="dots_tts",
        architecture="DotsTTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "reference_encode",
            "latent_engine",
            "vocoder",
        ),
        repo_id="dots-studio/dots.tts-soar",
        cards=1,
        kind="speech_references",
        serve_args=(
            "--config",
            "examples/configs/dots_tts_soar.yaml",
        ),
    ),
    XpuModel(
        key="dots-tts-base",
        package="dots_tts",
        architecture="DotsTTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "reference_encode",
            "latent_engine",
            "vocoder",
        ),
        repo_id="dots-studio/dots.tts-base",
        cards=1,
        kind="speech_references",
        serve_args=(
            "--config",
            "examples/configs/dots_tts_soar.yaml",
        ),
    ),
    XpuModel(
        key="dots-tts-mf",
        package="dots_tts",
        architecture="DotsTTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "reference_encode",
            "latent_engine",
            "vocoder",
        ),
        repo_id="dots-studio/dots.tts-mf",
        cards=1,
        kind="speech_references",
        serve_args=(
            "--config",
            "examples/configs/dots_tts.yaml",
            "--latent_engine.engine.max_running_requests",
            "4",
            "--latent_engine.engine.cuda_graph_max_bs",
            "4",
        ),
        revision="c28105adc8228143392b4e346994ff613ee48a06",
    ),
    XpuModel(
        key="auk",
        package="auk",
        architecture="AuKForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "conditioning",
            "auk_engine",
            "decode",
        ),
        repo_id="tencent/AuK",
        cards=1,
        kind="speech_instructions",
        startup_seconds=1800,
        # On D270 the DiT step graph fails capture with error 40, then the run dies with error 20.
        device_args=(
            ("d270", ("--auk_engine.factory.enable_dit_cuda_graph", "false")),
        ),
        companions=("Qwen/Qwen2.5-Omni-3B",),
    ),
    XpuModel(
        key="auk-flash",
        package="auk",
        architecture="AuKForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "conditioning",
            "auk_engine",
            "decode",
        ),
        repo_id="tencent/AuK-Flash",
        cards=1,
        kind="speech_instructions",
        startup_seconds=1800,
        # On D270 the DiT step graph fails capture with error 40, then the run dies with error 20.
        device_args=(
            ("d270", ("--auk_engine.factory.enable_dit_cuda_graph", "false")),
        ),
        companions=("Qwen/Qwen2.5-Omni-3B",),
    ),
    XpuModel(
        key="minimax-music3",
        package="minimax_music3",
        architecture="MiniMaxMusic3ForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "minimax_music3_ar",
            "dit_dav",
        ),
        repo_id="MiniMaxAI/MiniMax-Music3",
        cards=2,
        kind="music",
        serve_args=(
            "--mem-fraction-static",
            "0.7",
        ),
        startup_seconds=1800,
    ),
    XpuModel(
        key="yue2",
        package="yue2",
        architecture="YuE2ForCausalLM",
        variant=None,
        stages=("yue2_ar", "yue2_acoustic"),
        repo_id="m-a-p/YuE2-3B",
        # Both stages together hold about 21 GB, so it needs an otherwise idle card.
        cards=1,
        kind="music",
        companions=("m-a-p/YuE2-Vae",),
    ),
    XpuModel(
        key="omni-text",
        package="qwen3_omni",
        architecture="Qwen3OmniMoeForConditionalGeneration",
        variant="text",
        stages=(
            "preprocessing",
            "image_encoder",
            "audio_encoder",
            "mm_aggregate",
            "thinker",
            "decode",
        ),
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        cards=8,
        kind="chat",
        serve_args=(
            "--thinker.engine.mem_fraction_static",
            "0.435",
            "--skip-server-warmup",
            "--text-only",
            "--thinker.process",
            "thinker",
            "--thinker.tp_size",
            "8",
            "--thinker.gpu",
            "[0, 1, 2, 3, 4, 5, 6, 7]",
        ),
        startup_seconds=2400,
    ),
    XpuModel(
        key="omni-text-tp4",
        package="qwen3_omni",
        architecture="Qwen3OmniMoeForConditionalGeneration",
        variant="text",
        stages=(
            "preprocessing",
            "image_encoder",
            "audio_encoder",
            "mm_aggregate",
            "thinker",
            "decode",
        ),
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        cards=4,
        kind="chat",
        serve_args=(
            "--thinker.engine.mem_fraction_static",
            "0.80",
            "--skip-server-warmup",
            "--text-only",
            "--thinker.process",
            "thinker",
            "--thinker.tp_size",
            "4",
            "--thinker.gpu",
            "[0, 1, 2, 3]",
        ),
        startup_seconds=2400,
    ),
    XpuModel(
        key="omni-speech",
        package="qwen3_omni",
        architecture="Qwen3OmniMoeForConditionalGeneration",
        variant="speech",
        stages=(
            "preprocessing",
            "image_encoder",
            "audio_encoder",
            "thinker",
            "decode",
            "talker_ar",
            "code2wav",
        ),
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        cards=6,
        kind="chat_speech",
        serve_args=(
            # The talker's sampler reads the host inside its captured region: error 44 on XPU.
            "--talker_ar.engine.disable_cuda_graph",
            "true",
            "--skip-server-warmup",
            "--image_encoder.gpu",
            "0",
            "--audio_encoder.gpu",
            "0",
            "--thinker.tp_size",
            "4",
            "--thinker.gpu",
            "[0, 1, 2, 3]",
            # 14.3 GB of weights per 24 GB card at TP=4; 0.80 is the fraction this layout was measured at.
            "--thinker.engine.mem_fraction_static",
            "0.80",
            "--talker_ar.gpu",
            "4",
            "--talker_ar.engine.mem_fraction_static",
            "0.35",
            "--code2wav.gpu",
            "5",
            "--code2wav.gpu_memory_fraction",
            "0.05",
        ),
        startup_seconds=2400,
    ),
    XpuModel(
        key="minicpm-o",
        package="minicpm_o",
        architecture="MiniCPMO",
        variant="text",
        stages=(
            "preprocessing",
            "thinker",
            "image_encoder",
            "audio_encoder",
            "decode",
        ),
        repo_id="openbmb/MiniCPM-o-4_5",
        cards=1,
        kind="chat",
        serve_args=(
            "--text-only",
            "--thinker.engine.mem_fraction_static",
            "0.85",
        ),
        startup_seconds=1800,
    ),
    XpuModel(
        key="minicpm-o-tp2",
        package="minicpm_o",
        architecture="MiniCPMO",
        variant="text",
        stages=(
            "preprocessing",
            "thinker",
            "image_encoder",
            "audio_encoder",
            "decode",
        ),
        repo_id="openbmb/MiniCPM-o-4_5",
        cards=2,
        kind="chat",
        serve_args=(
            "--text-only",
            "--thinker.engine.mem_fraction_static",
            "0.85",
            "--thinker.process",
            "thinker",
            "--thinker.tp_size",
            "2",
            "--thinker.gpu",
            "[0, 1]",
            "--placement.require_memory_fraction_for_colocation",
            "false",
        ),
        startup_seconds=1800,
    ),
    XpuModel(
        key="ming-flash-omni",
        package="ming_omni",
        architecture="BailingMM2NativeForConditionalGeneration",
        variant="text",
        stages=(
            "preprocessing",
            "audio_encoder",
            "image_encoder",
            "mm_aggregate",
            "thinker",
            "decode",
        ),
        repo_id="inclusionAI/Ming-flash-omni-2.0",
        cards=1,
        kind="chat",
        serve_args=(
            "--model-name",
            "ming-omni",
            "--text-only",
            "--thinker.tp_size",
            "1",
            "--thinker.gpu",
            "0",
            "--audio_encoder.gpu",
            "0",
            "--image_encoder.gpu",
            "0",
            "--mem-fraction-static",
            "0.30",
            "--thinker.engine.cpu_offload_gb",
            "190",
            "--thinker.engine.cuda_graph_backend_decode",
            "disabled",
        ),
        startup_seconds=1800,
        serving="the checkpoint is a 222 GiB fetch, larger than any CI runner should pull",
    ),
    XpuModel(
        key="cosyvoice3",
        package="fun_cosyvoice3",
        architecture="FunCosyVoice3SGLangModel",
        variant=None,
        stages=(
            "preprocessing",
            "tts_engine",
            "vocoder",
        ),
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        cards=1,
        kind="speech_cloned",
        serving="needs the fun-cosyvoice3 extra; module 'cosyvoice' is absent from the XPU image",
    ),
    XpuModel(
        key="zonos2",
        package="zonos2",
        architecture="Zonos2ForCausalLM",
        variant="default",
        stages=(
            "preprocessing",
            "speaker_encode",
            "tts_engine",
            "vocoder",
        ),
        repo_id="Zyphra/zonos2",
        cards=1,
        kind="speech_default_voice",
        serving="needs the 'dac' codec module, which the XPU image does not install",
    ),
    XpuModel(
        key="personaplex",
        package="personaplex",
        architecture="PersonaPlexForCausalLM",
        variant=None,
        stages=(
            "preprocessing",
            "mimi_encode",
            "lm",
            "decode",
            "code2wav",
        ),
        repo_id="nvidia/personaplex-7b-v1",
        cards=1,
        kind="speech_default_voice",
        serving="runs through examples/run_personaplex.py, not an HTTP server",
    ),
    XpuModel(
        key="nemotron-voicechat",
        package="nemotron_voicechat",
        architecture="NemotronVoiceChatForCausalLM",
        variant=None,
        stages=(
            "preprocessing",
            "perception",
            "thinker",
            "decode",
            "talker",
            "code2wav",
        ),
        repo_id="nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        cards=2,
        kind="speech_default_voice",
        serving="runs through examples/run_nemotron_voicechat.py, not an HTTP server",
    ),
    XpuModel(
        key="audar-tts",
        package="audar_tts",
        architecture="AudarTTSForConditionalGeneration",
        variant=None,
        stages=(
            "preprocessing",
            "reference_encoder",
            "tts_engine",
            "vocoder",
        ),
        repo_id="AudarAI/Audar-TTS-V1-Turbo",
        cards=1,
        kind="speech_default_voice",
        serving="needs the audar-tts extra and a SYCL build of llama.cpp, not an SGLang engine",
    ),
)


def xpu_pipelines() -> tuple[XpuModel, ...]:
    """One row per distinct pipeline, since checkpoints of one family share a pipeline config."""
    seen: set[str] = set()
    unique: list[XpuModel] = []
    for model in XPU_MODELS:
        if model.pipeline_key in seen:
            continue
        else:
            seen.add(model.pipeline_key)
        unique.append(model)
    return tuple(unique)


def xpu_servable() -> tuple[XpuModel, ...]:
    """Rows the device lane can actually serve over HTTP."""
    return tuple(model for model in XPU_MODELS if not model.serving)


def model_named(key: str) -> XpuModel:
    for model in XPU_MODELS:
        if model.key == key:
            return model
    known = ", ".join(model.key for model in XPU_MODELS)
    raise KeyError(f"no XPU model named {key!r}; known keys: {known}")
