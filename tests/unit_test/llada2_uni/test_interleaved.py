# SPDX-License-Identifier: Apache-2.0
"""Interleaved transitions, asynchronous collection, and EOI stopping."""

import asyncio
from queue import Empty
from types import SimpleNamespace

import pytest

from sglang_omni.models.llada2_uni.components.preprocessor import (
    IMAGE_TOKEN_OFFSET,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.interleaved import (
    InterleavedCollectorScheduler,
    advance_interleaved_state,
    interleaved_next,
    project_interleaved_payload,
)
from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.protocol import normalize_interleaved_content


class Tokenizer:
    mask_token_id = 99
    eos_token_id = 2
    additional_stop_token_ids = [666]

    def convert_tokens_to_ids(self, token):
        return {"<|image|>": 10, "<|/image|>": 11, "<boi>": 12}[token]

    def convert_ids_to_tokens(self, token):
        return f"<|reserved_token_{token - 20}|>"

    def encode(self, text, add_special_tokens=False):
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token) for token in ids if token != self.eos_token_id)


CONFIG = {
    "mode": "interleaved",
    "max_frames": 3,
    "cfg_text_scale": 4.0,
    "cfg_image_scale": 1.0,
    "seed": 42,
    "decode_mode": "decoder-turbo",
    "decoder_steps": 8,
}


def initial_payload():
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor._tokenizer = Tokenizer()
    preprocessor._max_seq_len = 8192
    request = OmniRequest(
        inputs=[{"role": "user", "content": "Illustrate a three-frame story"}],
        params={"stream": False},
        metadata={"image_generation": CONFIG},
    )
    return asyncio.run(
        preprocessor(StagePayload(request_id="story", request=request, data={}))
    )


def advance(state, ids):
    state.thinker_out = {"output_ids": ids, "is_final": True}
    advance_interleaved_state(
        state, Tokenizer(), completed_phase=state.stream_state["interleaved"]["phase"]
    )


@pytest.mark.parametrize("final_first", [False, True])
def test_three_frames_continue_before_decoding_and_collect_in_order(final_first):
    payload = initial_payload()
    state = LLaDA2UniPipelineState.from_dict(payload.data)
    frames = []
    for index in range(1, 4):
        advance(state, [70 + index, 10, 22, 23, 12])
        assert state.stream_state["interleaved"]["cfg_plan"]["mode"] == (
            "simple" if index == 1 else "editing"
        )
        advance(state, [IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 1])
        advance(state, [IMAGE_TOKEN_OFFSET + i for i in range(2, 6)] + [11])
        payload.data = state.to_dict()
        assert interleaved_next("story", payload) == [
            "image_decode",
            "interleaved_collect" if index == 3 else "thinker",
        ]
        frames.append(project_interleaved_payload(payload))
        assert extract_image_vq_tokens(state) == (list(range(6)), 2, 3, CONFIG)
        if index < 3:
            state.stream_state["interleaved"].pop("emit_frame")
    assert [
        frame.data["stream_state"]["interleaved"]["frame_index"] for frame in frames
    ] == [1, 2, 3]

    collector = InterleavedCollectorScheduler()
    if final_first:
        collector.receive("story", project_interleaved_payload(payload))
    for frame in reversed(frames):
        index = frame.data["stream_state"]["interleaved"]["frame_index"]
        frame.data = {
            "kind": "interleaved_frame",
            "frame": {
                "index": index,
                "image": {
                    "id": f"image-{index}",
                    "data": "cG5n",
                    "format": "png",
                    "width": 64,
                    "height": 96,
                },
            },
        }
        collector.receive("story", frame)
    if not final_first:
        collector.receive("story", project_interleaved_payload(payload))
    result = collector.outbox.get_nowait().data.data
    assert [part.get("text", part.get("image_id")) for part in result["content"]] == [
        "71",
        "image-1",
        "72",
        "image-2",
        "73",
        "image-3",
    ]
    assert result["finish_reason"] == "max_frames"
    assert result["usage"]["completion_tokens"] == 36
    normalize_interleaved_content(result["content"], result["images"])
    collector.receive("story", frames[0])
    collector.abort("aborted")
    collector.receive("aborted", frames[0])
    assert not collector.frames and not collector.final_payloads
    with pytest.raises(Empty):
        collector.outbox.get_nowait()


def test_text_only_completion_and_duplicate_frame_cleanup():
    payload = initial_payload()
    state = LLaDA2UniPipelineState.from_dict(payload.data)
    advance(state, [75, 2])
    payload.data = state.to_dict()
    collector = InterleavedCollectorScheduler()
    collector.receive("story", payload)
    result = collector.outbox.get_nowait().data.data
    assert result["content"] == [{"type": "text", "text": "75"}]
    assert result["images"] == []
    frame = StagePayload(
        request_id="bad",
        request=payload.request,
        data={"kind": "interleaved_frame", "frame": {"index": 1, "image": {}}},
    )
    collector.receive("bad", frame)
    with pytest.raises(ValueError, match="duplicate"):
        collector.receive("bad", frame)
    collector.receive("bad", frame)
    assert not collector.frames and not collector.final_payloads


def test_completed_request_id_reuse_ignores_old_frames_and_preserves_abort() -> None:
    collector = InterleavedCollectorScheduler()
    previous: list[StagePayload] = []
    for generation in range(3):
        payload = initial_payload()
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        state.stream_state["interleaved"]["max_frames"] = 1
        advance(state, [71, 10, 22, 23, 12])
        advance(state, [IMAGE_TOKEN_OFFSET + i for i in range(6)] + [11])
        payload.data = state.to_dict()
        final = project_interleaved_payload(payload)
        frame = StagePayload(
            request_id="story",
            request=payload.request,
            data={
                "kind": "interleaved_frame",
                "frame": {"index": 1, "image": {"id": f"image-{generation}"}},
            },
        )
        frame = StagePayload.from_dict(frame.to_dict())
        collector.receive("story", frame)
        for old_payload in previous:
            collector.receive("story", project_interleaved_payload(old_payload))
        assert collector.outbox.empty()
        if generation == 2:
            collector.abort("story")
        collector.receive("story", project_interleaved_payload(final))
        collector.receive("story", frame)
        if generation == 2:
            assert collector.outbox.empty()
        else:
            result = collector.outbox.get_nowait()
            assert result.request_id == "story"
            assert result.data.data["images"] == [{"id": f"image-{generation}"}]
            assert collector.outbox.empty()
        assert not collector.frames and not collector.final_payloads
        previous.extend([frame, final])


def test_image_request_uses_current_cfg_and_eoi_only_stopping():
    pytest.importorskip("sglang")
    from sglang_omni.models.llada2_uni.request_builders import (
        build_dllm_thinker_request,
    )

    state = LLaDA2UniPipelineState.from_dict(initial_payload().data)
    advance(state, [71, 10, 22, 23, 12])
    req = build_dllm_thinker_request(
        state,
        params={"stop": ["stop"], "stop_token_ids": [666]},
        tokenizer=Tokenizer(),
        vocab_size=IMAGE_TOKEN_OFFSET + 16,
        dllm_config=SimpleNamespace(block_size=4, mask_id=99),
        request_id="story",
    ).req
    assert req.tokenizer is None and req.eos_token_ids == {11}
    assert req.sampling_params.stop_strs == []
    assert req.sampling_params.stop_token_ids == {11}
    assert req.sampling_params.max_new_tokens == 1500
    assert len(req.origin_input_ids) == len(req._uncond_input_ids)
    assert req._cfg_scale == 4.0 and req._allowed_stop_token_ids == (11,)
    assert req._dllm_steps == 32

    advance(state, [IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 1])
    req = build_dllm_thinker_request(
        state,
        params={},
        tokenizer=Tokenizer(),
        vocab_size=IMAGE_TOKEN_OFFSET + 16,
        dllm_config=SimpleNamespace(block_size=4, mask_id=99),
        request_id="story",
    ).req
    assert req.sampling_params.max_new_tokens == 1498


@pytest.mark.parametrize(
    "scale, mode", [(0.0, "simple"), (1.0, "none"), (4.0, "simple")]
)
def test_frame_cfg_preserves_unconditional_and_history_branches(scale, mode):
    state = LLaDA2UniPipelineState.from_dict(initial_payload().data)
    config = state.request_metadata["image_generation"]
    config.update(cfg_scale=0.0, cfg_text_scale=scale, cfg_image_scale=0.0)
    advance(state, [71, 10, 22, 23, 12])
    assert state.stream_state["interleaved"]["cfg_plan"]["mode"] == mode
    advance(state, [IMAGE_TOKEN_OFFSET + i for i in range(6)] + [11])
    advance(state, [72, 10, 22, 23, 12])
    plan = state.stream_state["interleaved"]["cfg_plan"]
    assert plan["mode"] == ("editing" if scale > 0 else "simple")
    assert len(plan["branches"]) == (2 if scale > 0 else 1)
