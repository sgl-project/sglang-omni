from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
    stage_process_name,
)

# The ckpt is all saved in float32.
MODEL_DTYPE = "float32"

MODEL_STAGES_PREFIX = "sglang_omni.models.nemotron_voicechat.stages"


def nemotron_voicechat_stages_factory() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_preprocessing_executor",
            next="perception",
        ),
        StageConfig(
            name="perception",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_perception_executor",
            factory=FactoryArgs(dtype=MODEL_DTYPE),
            gpu=0,
            # The talker builds its request by merging this payload
            # (wait_for/merge_fn); nothing fans out to a wait_for consumer on
            # its own, so both destinations are named here.
            next=["thinker", "talker"],
        ),
        EngineStageConfig(
            name="thinker",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_thinker_executor",
            # NemotronH runs under SGLang, whose rmsnorm kernel has no float32
            # path; the rest of the chain keeps the checkpoint's precision.
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next="decode",
            stream_to=["talker"],
        ),
        StageConfig(
            name="decode",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_decode_executor",
            terminal=True,
        ),
        EngineStageConfig(
            name="talker",
            process="talker",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_talker_executor",
            # Backbone only: FlashAttention takes fp16/bf16/fp8, and the
            # talker's own layers cast themselves back to float32.
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            engine=EngineArgs(mem_fraction_static=0.35),
            wait_for=["perception"],
            merge_fn="sglang_omni.models.nemotron_voicechat.request_builders.merge_for_talker",
            can_accept_stream_before_payload=True,
            next="code2wav",
            stream_to=["code2wav"],
        ),
        StageConfig(
            name="code2wav",
            process="talker",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_code2wav_executor",
            factory=FactoryArgs(dtype=MODEL_DTYPE),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class NemotronVoiceChatPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NemotronVoiceChatForCausalLM"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
        "talker": EngineStageConfig,
    }

    model_path: str
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=nemotron_voicechat_stages_factory)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, bool]:
        if stage_name not in ("talker", "code2wav"):
            return {}
        else:
            talker_stage = self.stage_named("talker")
            code2wav_stage = self.stage_named("code2wav")
            talker_gpu = (
                talker_stage.gpu[0]
                if isinstance(talker_stage.gpu, list)
                else talker_stage.gpu
            )
            code2wav_gpu = (
                code2wav_stage.gpu[0]
                if isinstance(code2wav_stage.gpu, list)
                else code2wav_stage.gpu
            )
            return {
                "can_use_local_code_handoff": (
                    stage_process_name(talker_stage)
                    == stage_process_name(code2wav_stage)
                    and talker_stage.tp_size == code2wav_stage.tp_size == 1
                    and talker_gpu is not None
                    and talker_gpu == code2wav_gpu
                )
            }


EntryClass = NemotronVoiceChatPipelineConfig
