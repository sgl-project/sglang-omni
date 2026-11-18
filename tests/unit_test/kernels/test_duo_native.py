"""Validate the optional DCP adapter against the reference causal transformer."""

import os
from argparse import Namespace

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoProcessor

pytest.importorskip("torchtitan.models.duo_realization.modules.model")

from threading import Event

from torchtitan.models.duo_realization.modules import intent_conditioning
from torchtitan.models.duo_realization.modules.intent import prepare_tokenizer

from sglang_omni.models.duo.intent_srt import SRTPlan
from sglang_omni.models.duo.live_intent_srt import (
    IntentPositions,
    IntentReply,
    SRTFrameIntentPlanner,
)


@pytest.mark.accelerator
def test_incremental_intent_preserves_protocol_and_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = os.environ.get("DUO_INTENT_MODEL")
    adapter_path = os.environ.get("DUO_INTENT_ADAPTER")
    if model_path is None or adapter_path is None:
        pytest.skip("Set DUO_INTENT_MODEL and DUO_INTENT_ADAPTER to local checkpoints")
    else:
        pass
    planner = SRTFrameIntentPlanner.__new__(SRTFrameIntentPlanner)
    arguments = Namespace(
        intent=model_path,
        intent_adapter=adapter_path,
        origin_intent=False,
        decode="duo_5_17",
    )
    planner.tokenizer, _ = prepare_tokenizer(arguments)
    planner.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    planner.processor.tokenizer = planner.tokenizer
    planner.processor.video_processor.size = {
        "shortest_edge": 1024,
        "longest_edge": 16384,
    }
    planner.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    planner.positions = IntentPositions(planner.config)
    planner.states = {}
    planner.usages = {}
    planner.connections = []
    planner.max_context_tokens = 32768
    planner.kv_window_tokens = 32768
    planner.max_new_tokens = 128
    planner.end_token_id = planner.tokenizer.convert_tokens_to_ids("<|im_end|>")
    planner.open(
        "protocol",
        {
            "appearance": "A person.",
            "environment": "A room.",
            "initial_pose": "Facing the camera.",
        },
    )
    generated_ids: list[int] = []

    def receive_plan(current_planner: SRTFrameIntentPlanner) -> IntentReply:
        state = current_planner.states["protocol"]
        block_index = len(state.record["blocks"]) - 1
        prefix = intent_conditioning.serialize(
            state.record,
            current_planner.tokenizer,
            state.visuals,
            stop_block=block_index,
            assistant_bodies=state.bodies,
            generation=True,
        )
        return IntentReply(
            plan=SRTPlan(
                (
                    torch.zeros(sum(prefix["keep"]), 5120)
                    if state.shared_h50 is None
                    else None
                ),
                torch.zeros(len(generated_ids) - 1, 5120),
                list(generated_ids),
                len(prefix["input_ids"]) + len(generated_ids),
                0,
                0,
                0 if state.input_ids is None else state.input_ids.numel(),
                0,
            )
        )

    monkeypatch.setattr(SRTFrameIntentPlanner, "receive", receive_plan)
    for block_index, (user_text, plan_text) in enumerate(
        [
            ("", "<wait>"),
            ("你好", "<speak>你好"),
            ("Hello", "<wait>"),
            ("", "<speak>Hello"),
        ]
    ):
        generated_ids = planner.tokenizer.encode(
            plan_text, add_special_tokens=False
        ) + [planner.end_token_id]
        frames = np.zeros((17, 112, 112 + 56 * (block_index % 2), 3), dtype=np.uint8)
        planner.append(
            "protocol",
            frames,
            user_text,
            block_index,
            Event(),
            frames_per_block=17,
            block_duration_ms=17000 / 24,
        )
        state = planner.states["protocol"]
        expected = intent_conditioning.serialize(
            state.record,
            planner.tokenizer,
            state.visuals,
            stop_block=block_index,
            assistant_bodies=state.bodies,
            generation=True,
        )
        expected_ids = torch.tensor(expected["input_ids"]).reshape(1, -1)
        positions, _ = planner.positions.get_rope_index(
            expected_ids,
            (expected_ids == planner.config.video_token_id).long() * 2,
            video_grid_thw=torch.tensor(expected["video_grid_thw"]).reshape(-1, 3),
            attention_mask=torch.ones_like(expected_ids),
        )
        next_position = int(positions.max()) + 1
        generated_positions = (
            torch.arange(next_position, next_position + len(generated_ids))
            .reshape(1, 1, -1)
            .expand(3, 1, -1)
        )
        torch.testing.assert_close(
            state.input_ids,
            torch.cat((expected_ids, torch.tensor([generated_ids])), dim=1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.position_ids,
            torch.cat((positions, generated_positions), dim=-1),
            rtol=0,
            atol=0,
        )
        assert state.input_keep == expected["keep"] + [True] * (
            len(generated_ids) - 1
        ) + [False]
