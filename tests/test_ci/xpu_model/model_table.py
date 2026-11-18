# SPDX-License-Identifier: Apache-2.0
"""Every model family Intel XPU serves, shared by both XPU test lanes.

tests/unit_test/xpu/test_model_support.py and tests/test_ci/xpu_model/
test_model_smoke.py parametrize over these entries, so a family cannot be
covered by one lane and forgotten by the other.

The package, architecture and stage tuple of every row were read out of the
pipeline config registry and then pinned here, which is the point: the stage
tuple is the contract, so deriving it at test time from the code under test
would assert nothing.

``serve_args`` is the recipe from that model's section of
docs/other_platforms/installation_xpu.md, and several rows do not start without it.
It lives on the row rather than in the test so that the placement a topology
needs and the flags a checkpoint needs are not split across two files. Stage
gpu ids are lane-relative, because the lane sets ZE_AFFINITY_MASK to the cards
it claimed and a stage's gpu id indexes into the mask.

``device_args`` holds what one device family needs on top of that recipe.
D270 is the only such family today: three captures that a B60 records lose the
device there, so those rows turn the graph off on D270 alone.
The recipe is keyed on the hardware the lane finds, not on the runner, so the
same table serves both.

A row carries ``serving`` when the device lane has to skip it -- a model not
reachable over HTTP at all, or a checkpoint no runner should pull. It keeps its
row either way, because the weight-free lane still has something to say about a
family this image cannot serve, and a skip the table explains is worth more
than one a reader has to reconstruct.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ROOTS_ENV = "MODEL_ROOTS"
ALLOW_DOWNLOAD_ENV = "OMNI_XPU_ALLOW_HF_DOWNLOAD"
# Names the device family outright, for a card whose name the lane does not know.
DEVICE_FAMILY_ENV = "OMNI_XPU_DEVICE_FAMILY"

KNOWN_FAMILIES = ("bmg", "d270")
# torch reports a D270 card as "Intel(R) Graphics [0x674c]"; the PCI id is the
# only stable part of that name.
D270_PCI_IDS = ("0x674c",)


def device_family(device_name: str, override: str = "") -> str:
    """Which recipe family a card belongs to: "d270", "bmg" or "unknown".

    An unknown card gets the base recipe, which is what a B60 runs, rather than
    a guess at another family's workarounds.
    """
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
    # For a model whose first request does one-off work: the lane waits the longer
    # of this and its own request timeout. 0 keeps the lane's default.
    request_seconds: int = 0
    # (family, extra serve args) pairs, applied after serve_args on that family.
    device_args: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Why the device lane cannot serve this one; empty when it can.
    serving: str = ""

    def serve_args_on(self, family: str) -> tuple[str, ...]:
        """The full recipe for a card of ``family``."""
        extra: tuple[str, ...] = ()
        for name, args in self.device_args:
            if name == family:
                extra += args
            else:
                continue
        return self.serve_args + extra

    @property
    def pipeline_key(self) -> str:
        """Identity of the *pipeline*, which several checkpoints can share."""
        return f"{self.package}:{self.variant or 'default'}"

    @property
    def mirror_dirname(self) -> str:
        return self.repo_id.split("/")[-1]

    @property
    def checkpoint_env(self) -> str:
        """Env var pointing this row at a local copy.

        Derived rather than spelled out per row: twenty-eight hand-written names
        are twenty-eight chances to paste the wrong one, and nothing outside this
        package names them literally.
        """
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
        # The first request compiles and tunes the backbone's linear-attention Triton
        # kernels for its audio length: 279 s on a B60 with a cold cache, which a fresh
        # CI container always has, against a 300 s default.
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
        # The reference-code batcher captures on a private stream, and on D270 that
        # capture loses the device (error 20 later). Only ref_audio reaches it, so
        # CustomVoice and VoiceDesign keep their graph.
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
        # The longest window in the table: a cold torch.compile cache makes this
        # row's startup far longer than a warm one.
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
        # On D270 the DiT step graph fails capture with error 40 and the run then
        # dies with error 20 instead of staying eager.
        device_args=(
            ("d270", ("--auk_engine.factory.enable_dit_cuda_graph", "false")),
        ),
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
        # On D270 the DiT step graph fails capture with error 40 and the run then
        # dies with error 20 instead of staying eager.
        device_args=(
            ("d270", ("--auk_engine.factory.enable_dit_cuda_graph", "false")),
        ),
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
            "--minimax_music3_ar.engine.attention_backend",
            "triton",
        ),
        startup_seconds=1800,
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
            # The talker's sampler reads the host inside its captured region, so on
            # XPU capture fails with Level-Zero error 44 and the stage exits.
            "--talker_ar.engine.disable_cuda_graph",
            "true",
            # Warmup reaches the vision encoder, whose RoPE head dim (72) the XPU
            # kernel does not build; drop once sgl-project/sglang-omni#3907 lands.
            "--skip-server-warmup",
            "--image_encoder.gpu",
            "0",
            "--audio_encoder.gpu",
            "0",
            "--thinker.tp_size",
            "4",
            "--thinker.gpu",
            "[0, 1, 2, 3]",
            # TP=4 puts 14.3 GB of weights on each 24 GB card, so the KV check needs
            # more than 0.60; 0.80 is the value this layout was measured at.
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
    """One row per distinct pipeline.

    The three Qwen3-TTS checkpoints share Qwen3TTSPipelineConfig, as do the
    three dots.tts ones and the two AuK ones, so the weight-free lane runs over
    each pipeline once instead of restating the same assertions per checkpoint.
    """
    seen: set[str] = set()
    unique: list[XpuModel] = []
    for model in XPU_MODELS:
        if model.pipeline_key in seen:
            continue
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
