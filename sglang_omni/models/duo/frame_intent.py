"""Frame-conditioned Intent planning with the training protocol."""

import numpy as np
from torchtitan.models.duo_realization.modules import intent_conditioning

from sglang_omni.models.duo.live_intent import (
    IntentVisual,
    LiveIntentPlanner,
    LiveIntentState,
)


class FrameIntentPlanner(LiveIntentPlanner):
    """Use training's exact ChatML and selected H50 rows with incremental KV."""

    def visual_patch(self, frames: np.ndarray) -> IntentVisual:
        return intent_conditioning.visual_patch(self.processor, frames)

    def system_prompt_fragments(self) -> tuple[str, str]:
        return (
            intent_conditioning.INTRO.replace(
                "<chunk_160>", self.tokenizer.duplex_chunk_token, 1
            ),
            intent_conditioning.INSTRUCTION + intent_conditioning.SPEECH_CUE,
        )

    def serialize(
        self, state: LiveIntentState, block_index: int, block_duration_ms: float
    ) -> dict[str, object]:
        prefix = intent_conditioning.serialize(
            state.record,
            self.tokenizer,
            state.visuals,
            stop_block=block_index,
            assistant_bodies=state.bodies,
            generation=True,
            start_block=block_index if self.incremental_inputs else 0,
            user_bodies=(
                {
                    block_index: state.user_token_ids
                    or intent_conditioning.intent.encode_protocol(
                        self.tokenizer, "<blank>"
                    )
                }
                if self.incremental_inputs
                else None
            ),
        )
        if not self.incremental_inputs or block_index == 0:
            state.description_tokens = sum(
                group["end"] - group["start"]
                for group in prefix["text_groups"]
                if group["section"] == "assistant_description"
            )
        else:
            pass
        return prefix
