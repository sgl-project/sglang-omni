# SPDX-License-Identifier: Apache-2.0
"""Every published draft a model stage serves, pinned to one Hugging Face revision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class PinnedDraft:
    """One draft for one model stage, and the engine keys that serve it.

    A cookbook command, an example config or a model's defaults that serve this
    draft give the keys from engine_overrides. They can leave out the revision,
    which EngineArgs.overrides fills from this entry.
    """

    model_path: str
    stage: str
    algorithm: Literal["DFLASH", "DSPARK", "EAGLE3", "STANDALONE"]
    draft_model_path: str
    draft_model_revision: str
    engine_settings: dict[str, int] = field(default_factory=dict)
    # note (Haoling Pu): None means the model card states no license, so the draft stays off by default.
    license: str | None
    is_default_on: bool = False

    def engine_overrides(self) -> dict[str, str | int]:
        return {
            "speculative_algorithm": self.algorithm,
            "speculative_draft_model_path": self.draft_model_path,
            "speculative_draft_model_revision": self.draft_model_revision,
            **self.engine_settings,
        }


DRAFT_REGISTRY: tuple[PinnedDraft, ...] = (
    PinnedDraft(
        model_path="Qwen/Qwen3-ASR-1.7B",
        stage="asr",
        algorithm="DFLASH",
        draft_model_path="wilsonzheng0327/Qwen3-ASR-1.7B-dflash",
        draft_model_revision="d62bd4051e68282e02f93de774036a35f22f1373",
        engine_settings={"speculative_num_draft_tokens": 8},
        license="apache-2.0",
    ),
    PinnedDraft(
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        stage="asr",
        algorithm="DFLASH",
        draft_model_path="wilsonzheng0327/MOSS-Transcribe-Diarize-dflash",
        draft_model_revision="2e3f00b0932112e4f92a4ef67a4cd60ee855056c",
        license="apache-2.0",
    ),
    PinnedDraft(
        model_path="openai/whisper-large-v3",
        stage="asr",
        algorithm="STANDALONE",
        draft_model_path="wilsonzheng0327/Whisper-large-v3-distil-draft",
        draft_model_revision="257104348ea24677cf6338967592a9d6078b383b",
        engine_settings={"speculative_num_steps": 7},
        license="apache-2.0",
    ),
    PinnedDraft(
        model_path="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        stage="tts_engine",
        algorithm="DSPARK",
        draft_model_path="yuekai/cosyvoice3_llm_dspark",
        draft_model_revision="0b1b28dbb751bac3b74eea5dcc11e0570c91563b",
        license=None,
    ),
    PinnedDraft(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        stage="thinker",
        algorithm="DFLASH",
        draft_model_path="yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8",
        draft_model_revision="47029d797b43f6fb8fba8948784ad221a55aafd6",
        engine_settings={"speculative_num_draft_tokens": 8},
        license=None,
    ),
    PinnedDraft(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        stage="thinker",
        algorithm="EAGLE3",
        draft_model_path="yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio",
        draft_model_revision="4c197a033a9ffec3f4fe693edc4e5e65a9d7fe89",
        engine_settings={
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
        },
        license=None,
    ),
    PinnedDraft(
        model_path="openbmb/MiniCPM-o-4_5",
        stage="thinker",
        algorithm="DSPARK",
        draft_model_path="Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm",
        draft_model_revision="c4c310aca2c768343165261cb22c21b90c1f79e0",
        license=None,
    ),
)
