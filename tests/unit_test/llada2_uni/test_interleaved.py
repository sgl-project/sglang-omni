# SPDX-License-Identifier: Apache-2.0
"""Interleaved transitions, frame CFG, EOI stopping, and the UMM adapter."""

import asyncio
from types import SimpleNamespace

import pytest

from sglang_omni.models.llada2_uni.components.preprocessor import (
    IMAGE_TOKEN_OFFSET,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.config import LLaDA2UniInterleavedPipelineConfig
from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.umm import LLaDA2UniUMMAdapter
from sglang_omni.pipeline.umm import UMMContext, UMMOutput, UMMRequestError
from sglang_omni.proto import OmniRequest, StagePayload


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


STORY = [{"role": "user", "content": "Illustrate a three-frame story"}]


def preprocess(request, interleaved=True):
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor.tokenizer = Tokenizer()
    preprocessor.max_seq_len = 8192
    preprocessor.interleaved = interleaved
    return asyncio.run(
        preprocessor(StagePayload(request_id="story", request=request, data={}))
    )


def initial_payload():
    return preprocess(
        OmniRequest(inputs=STORY, params={}, metadata={"image_generation": CONFIG})
    )


def advance(state, ids):
    state.thinker_out = {"output_ids": ids, "is_final": True}
    advance_interleaved_state(
        state, Tokenizer(), completed_phase=state.stream_state["interleaved"]["phase"]
    )


def test_image_phase_generates_the_frame_grid_and_closes_it_with_eoi():
    pytest.importorskip("sglang")
    from sglang_omni.models.llada2_uni.request_builders import (
        build_dllm_thinker_request,
    )

    def image_request(state):
        return build_dllm_thinker_request(
            state,
            params={"stop": ["stop"], "stop_token_ids": [666]},
            tokenizer=Tokenizer(),
            vocab_size=IMAGE_TOKEN_OFFSET + 16,
            dllm_config=SimpleNamespace(
                block_size=4, mask_id=99, requires_separate_context_encoding=False
            ),
            request_id="story",
        ).req

    state = LLaDA2UniPipelineState.from_dict(initial_payload().data)
    advance(state, [71, 10, 22, 23, 12])
    req = image_request(state)
    # A text-to-image pass over the frame's 2x3 grid.
    assert req._task_kind == "t2i"  # noqa: leading-underscore  # DLLM protocol
    assert req.sampling_params.max_new_tokens == 6
    assert req.sampling_params.stop_strs == []
    assert not req.sampling_params.stop_token_ids
    assert len(req.origin_input_ids) == len(
        req._uncond_input_ids
    )  # noqa: leading-underscore  # DLLM protocol
    assert req._cfg_scale == 4.0  # noqa: leading-underscore  # DLLM protocol
    assert req._dllm_steps == 32  # noqa: leading-underscore  # DLLM protocol

    advance(state, [IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 1])
    assert image_request(state).sampling_params.max_new_tokens == 4
    advance(state, [IMAGE_TOKEN_OFFSET + i for i in range(2, 6)])
    interleaved = state.stream_state["interleaved"]
    assert interleaved["phase"] == "text" and interleaved["emit_frame"]
    assert state.prompt["input_ids"][0, -1].item() == 11


@pytest.mark.parametrize(
    "scale, mode", [(0.0, "simple"), (1.0, "none"), (4.0, "simple")]
)
def test_frame_cfg_preserves_unconditional_and_history_branches(scale, mode):
    state = LLaDA2UniPipelineState.from_dict(initial_payload().data)
    config = state.request_metadata["image_generation"]
    config.update(cfg_scale=0.0, cfg_text_scale=scale, cfg_image_scale=0.0)
    advance(state, [71, 10, 22, 23, 12])
    assert state.stream_state["interleaved"]["cfg_plan"]["mode"] == mode
    advance(state, [IMAGE_TOKEN_OFFSET + i for i in range(6)])
    advance(state, [72, 10, 22, 23, 12])
    plan = state.stream_state["interleaved"]["cfg_plan"]
    assert plan["mode"] == ("editing" if scale > 0 else "simple")
    assert len(plan["branches"]) == (2 if scale > 0 else 1)


def think(dispatch, ids):
    """One thinker pass, as the stage runs it on the dispatched state."""
    state = LLaDA2UniPipelineState.from_dict(dispatch.request.inputs)
    state.stream_state["interleaved"].pop("emit_frame", None)
    advance(state, ids)
    return state.to_dict()


def image(name):
    return {"kind": "image", "url": f"data:image/png;base64,{name}"}


def umm_context(max_frames=3):
    request = OmniRequest(
        inputs=STORY,
        params={"stage_params": {"orchestrator": {"max_frames": max_frames}}},
        metadata={},
    )
    return UMMContext("story", request, deadline=60.0, remaining_dispatches=32)


def test_frames_decode_while_the_thinker_writes_and_segments_keep_model_order():
    adapter = LLaDA2UniUMMAdapter("m")
    context = umm_context()
    start = adapter.start(context)
    assert start.phase == "prepare"
    step = adapter.advance(context, "prepare", preprocess(start.request).data)
    (text_phase,) = step.dispatches

    header, frame = [71, 10, 22, 23, 12], [IMAGE_TOKEN_OFFSET + i for i in range(6)]
    step = adapter.advance(context, "think", think(text_phase, header))
    assert step.outputs == (UMMOutput("text", "71"),)
    step = adapter.advance(context, "think", think(step.dispatches[0], frame))
    thinker, decode = step.dispatches
    assert (thinker.phase, decode.phase) == ("think", "decode")

    # The next text waits behind the first image, and the second frame waits
    # for the decoder, which runs one frame of a request at a time.
    step = adapter.advance(context, "think", think(thinker, [72, 10, 22, 23, 12]))
    assert step.outputs == ()
    step = adapter.advance(context, "think", think(step.dispatches[0], frame))
    assert [dispatch.phase for dispatch in step.dispatches] == ["think"]
    step = adapter.advance(context, "think", think(step.dispatches[0], [75, 2]))
    assert step.outputs == () and step.dispatches == ()
    # The held text and the waiting frame's image tokens count as retained.
    assert adapter.usage(context) >= len("7275") + len(frame)

    step = adapter.advance(context, "decode", image("first"))
    assert step.outputs == (
        UMMOutput("image", image("first")),
        UMMOutput("text", "72"),
    )
    (second,) = step.dispatches
    assert second.phase == "decode"
    for index, dispatch in enumerate((decode, second), start=1):
        state = LLaDA2UniPipelineState.from_dict(dispatch.request.inputs)
        assert extract_image_vq_tokens(state) == (
            list(range(6)),
            2,
            3,
            start.request.metadata["image_generation"],
        )
        assert state.stream_state["interleaved"]["segments"][-1]["frame_index"] == index
    step = adapter.advance(context, "decode", image("second"))
    assert step.outputs == (
        UMMOutput("image", image("second")),
        UMMOutput("text", "75"),
    )
    assert step.dispatches == () and adapter.usage(context) == 0
    adapter.close(context, "completed")
    assert not adapter.runs


def test_interleaved_variant_runs_the_adapter_and_refuses_what_it_cannot_serve(
    monkeypatch,
):
    from sglang_omni.config import manager
    from sglang_omni.config.runtime import resolve_stage_typed_kwargs
    from sglang_omni.models.llada2_uni.config import LLaDA2UniPipelineConfig
    from sglang_omni.pipeline.umm import create_umm_controller

    monkeypatch.setattr(
        manager, "resolve_config_cls_for_model_path", lambda _: LLaDA2UniPipelineConfig
    )
    config = manager.ConfigManager.from_model_path("m", "interleaved").config
    assert type(config) is LLaDA2UniInterleavedPipelineConfig
    args = {
        **resolve_stage_typed_kwargs(config.stage_named("orchestrator")),
        **config.stage_factory_kwargs("orchestrator"),
    }
    umm = create_umm_controller("m", **args)
    assert isinstance(umm.adapter, LLaDA2UniUMMAdapter)
    umm.stop()

    with pytest.raises(UMMRequestError, match="max_frames=11 needs 35 phases"):
        LLaDA2UniUMMAdapter("m").start(umm_context(max_frames=11))
    context = umm_context()
    context.request.inputs = {"messages": STORY, "images": ["cat.png"]}
    with pytest.raises(UMMRequestError, match="text-only input"):
        LLaDA2UniUMMAdapter("m").start(context)
    # Other variants refuse interleaved controls as a bad request, including
    # the stage_params.orchestrator a chat client sends.
    for request in (
        OmniRequest(STORY, {}, {"image_generation": CONFIG}),
        OmniRequest(STORY, {"stage_params": {"orchestrator": {"max_frames": 3}}}, {}),
    ):
        with pytest.raises(UMMRequestError, match="--variant interleaved"):
            preprocess(request, interleaved=False)
