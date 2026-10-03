# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.fun_cosyvoice3"

FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES: tuple[tuple[int, int], ...] = (
    (15, 624),
    (16, 576),
    (16, 560),
    (15, 592),
    (16, 544),
    (15, 576),
    (16, 512),
    (12, 512),
    (10, 608),
    (10, 592),
    (9, 592),
    (9, 576),
    (9, 560),
    (9, 544),
    (9, 528),
    (7, 608),
    (7, 576),
    (7, 560),
    (7, 544),
    (7, 528),
    (6, 608),
    (7, 496),
    (6, 544),
    (5, 640),
    (5, 624),
    (5, 592),
    (6, 480),
    (5, 576),
    (5, 560),
    (6, 464),
    (5, 544),
    (5, 528),
    (4, 640),
    (5, 496),
    (5, 480),
    (4, 512),
    (3, 592),
    (3, 576),
    (3, 560),
    (3, 544),
    (3, 528),
    (3, 512),
    (3, 496),
    (3, 480),
    (3, 464),
    (3, 448),
    (3, 432),
    (1, 608),
    (1, 576),
    (1, 560),
    (1, 544),
    (1, 496),
    (1, 464),
    (1, 432),
    (1, 416),
)

FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES: tuple[
    tuple[int, int, int, int, tuple[int, ...]], ...
] = (
    (1, 300, 300, 3072, (300,)),
    (1, 450, 450, 512, (450,)),
    (3, 400, 200, 1536, (200, 100, 100)),
    (4, 900, 400, 2048, (400, 200, 150, 150)),
    (2, 500, 350, 2048, (350, 150)),
    (3, 650, 350, 3072, (350, 150, 150)),
    (3, 1100, 400, 512, (400, 350, 350)),
    (1, 100, 100, 512, (100,)),
    (2, 200, 100, 512, (100, 100)),
    (3, 500, 300, 1024, (300, 100, 100)),
    (4, 1200, 400, 1024, (400, 300, 250, 250)),
    (3, 850, 350, 1024, (350, 250, 250)),
    (5, 1150, 350, 512, (350, 200, 200, 200, 200)),
    (8, 1700, 300, 1536, (300, 300, 250, 200, 200, 150, 150, 150)),
    (2, 600, 300, 512, (300, 300)),
    (4, 700, 300, 1024, (300, 150, 150, 100)),
    (6, 1450, 400, 2048, (400, 250, 200, 200, 200, 200)),
)

_DIT_ACCELERATOR_CONFLICT = (
    "enable_flow_estimator_trt and enable_dit_torch_compile both "
    "target flow.decoder.estimator; enable only one"
)


def reject_conflicting_dit_accelerators(
    *,
    enable_dit_torch_compile: bool,
    enable_flow_estimator_trt: bool,
) -> None:
    if enable_flow_estimator_trt and enable_dit_torch_compile:
        raise ValueError(_DIT_ACCELERATOR_CONFLICT)
    else:
        pass


class FunCosyVoice3EngineFactoryArgs(FactoryArgs):
    """Engine-only knobs, including the optional native MLX checkpoint."""

    mlx_model_path: str | None = Field(default=None)
    mlx_model_revision: str | None = Field(default=None)


class FunCosyVoice3EngineStageConfig(EngineStageConfig):
    factory: FunCosyVoice3EngineFactoryArgs = Field(
        default_factory=FunCosyVoice3EngineFactoryArgs
    )


class FunCosyVoice3VocoderFactoryArgs(FactoryArgs):
    """Vocoder knobs, including the converted native MLX artifact."""

    mlx_model_path: str | None = Field(default=None)
    mlx_model_revision: str | None = Field(default=None)
    flow_prefix_cuda_graph_max_slack_frames: int | None = Field(default=None, gt=0)


class FunCosyVoice3VocoderStageConfig(StageConfig):
    factory: FunCosyVoice3VocoderFactoryArgs = Field(
        default_factory=FunCosyVoice3VocoderFactoryArgs
    )


class FunCosyVoice3PipelineConfig(PipelineConfig):
    """3-stage Fun-CosyVoice3 pipeline: preprocessing -> tts_engine -> vocoder."""

    architecture: ClassVar[str] = "FunCosyVoice3SGLangModel"
    # note (PoTaTo): This checkpoint has no built-in speaker presets, so public requests need
    # one reference clip for speaker conditioning.
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_excludes_instructions: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": FunCosyVoice3EngineStageConfig,
        "vocoder": FunCosyVoice3VocoderStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next="tts_engine",
        ),
        # note(ratish): stages in one process are built in list order; the
        # vocoder comes before the engine so Flow, HiFT and their graphs are
        # resident when sglang sizes the KV pool from free memory.
        FunCosyVoice3VocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FunCosyVoice3VocoderFactoryArgs(
                dtype="bfloat16",
                flow_batch_admission_frames=8000,
                flow_merge_max_gap_frames=384,
                flow_merge_pad_budget_percent=25.0,
                # Note (chenyang): Adjacent length-sorted requests may share a Flow solve
                # when their mel-length gap and total added padding stay within these limits.
                max_batch_size=16,
                max_batch_wait_ms=30,
                enable_flow_cuda_graph=True,
                enable_flow_prefix_cuda_graph=False,
                flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
                flow_prefix_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES,
                # note (guozhihao-224, chenyang):
                # CUDA Graph and DiT torch.compile are on by default. TensorRT stays opt-in;
                # stage_factory_kwargs sets the compile default.
                enable_flow_estimator_trt=False,
                token_hop_len=25,
                token_max_hop_len=100,
                disable_hop_growth=False,
                flow_prefix_cache_gb=24.0,
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        FunCosyVoice3EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FunCosyVoice3EngineFactoryArgs(
                dtype="bfloat16",
                onnx_intra_op_threads=16,
                token_hop_len=25,
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
    ]

    def model_post_init(self, __context: object = None) -> None:
        # TODO (chenyang): Indeed, TRT and Torch compile conflicts are pretty
        # common in this repo, so we should make this into config level, not in each model.
        super().model_post_init(__context)
        if "OMP_NUM_THREADS" not in self.env_defaults:
            config_cls = type(self)
            for stage in self.stages:
                if config_cls.stage_config_cls(stage.name).engine_stage:
                    # note(chenye): SGLang pins Torch CPU threads to 1 for its GPU process.
                    # Apply the same policy at spawn to every stage colocated with the engine.
                    stage.env.setdefault("OMP_NUM_THREADS", "1")
                else:
                    pass
        else:
            pass
        vocoder = next(stage for stage in self.stages if stage.name == "vocoder")
        extras = vocoder.factory.model_extra
        # note(ratish): only explicit flags reach here;
        # stage_factory_kwargs sets the compile default.
        reject_conflicting_dit_accelerators(
            enable_dit_torch_compile=bool(extras.get("enable_dit_torch_compile")),
            enable_flow_estimator_trt=bool(extras.get("enable_flow_estimator_trt")),
        )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, bool | str]:
        if stage_name != "vocoder":
            return {}
        else:
            pass
        vocoder_factory = self.stage_named("vocoder").factory
        # note(ratish): the resolver reads a stage literal back as an explicit choice;
        # a set enable_dit_torch_compile overrides this default.
        kwargs: dict[str, bool | str] = {
            "enable_dit_torch_compile": not bool(
                vocoder_factory.model_extra.get("enable_flow_estimator_trt")
            )
        }
        if vocoder_factory.mlx_model_path is not None:
            # Note (yexiaodong): A separate vocoder artifact must keep its own
            # revision; both explicit fields therefore stay in typed config.
            return kwargs
        else:
            pass
        # Note (yexiaodong): The converted artifact contains the speech-token
        # LLM, Flow, and HiFT weights, so reuse it unless the vocoder overrides it.
        engine_factory = self.stage_named("tts_engine").factory
        if engine_factory.mlx_model_path is not None:
            kwargs["mlx_model_path"] = engine_factory.mlx_model_path
        else:
            pass
        if engine_factory.mlx_model_revision is not None:
            kwargs["mlx_model_revision"] = engine_factory.mlx_model_revision
        else:
            pass
        return kwargs


EntryClass = FunCosyVoice3PipelineConfig
