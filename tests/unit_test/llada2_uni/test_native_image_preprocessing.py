# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import re

import pytest
import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components import common
from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    DUMMY_IMAGE_TOKEN_ID,
    EOI_TOKEN,
    SOI_TOKEN,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    merge_image_tokens_for_thinker,
    prepare_dllm_input_group,
    resolve_native_image_token_offset,
)
from sglang_omni.proto import OmniRequest, StagePayload


def test_image_token_offset_is_loaded_from_checkpoint_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"image_token_offset": 321000}),
        encoding="utf-8",
    )

    assert common.load_llada2_image_token_offset(str(tmp_path)) == 321000


def test_missing_image_token_offset_is_rejected(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="image_token_offset"):
        common.load_llada2_image_token_offset(str(tmp_path))


def test_pipeline_state_round_trips_native_image_fields() -> None:
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": [1, 2, 3]},
        task_kind="t2i",
        image_token_offset=321000,
        generation_state={
            "image_grid": {"height": 32, "width": 48},
            "cfg": {"scale": 4.0, "rescale": 0.7},
        },
        request_metadata={"image_generation": {"width": 1536, "height": 1024}},
    )

    restored = LLaDA2UniPipelineState.from_dict(state.to_dict())

    assert restored.task_kind == "t2i"
    assert restored.image_token_offset == 321000
    assert restored.generation_state["image_grid"] == {
        "height": 32,
        "width": 48,
    }
    assert restored.request_metadata["image_generation"]["width"] == 1536


def test_vq_tokens_use_checkpoint_offset_instead_of_module_constant() -> None:
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": [-200, -200]},
        encoder_outs={"image_encoder": {"image_token_ids": [[3, 9]]}},
        image_token_offset=321000,
    )

    merge_image_tokens_for_thinker(state)

    assert state.prompt["input_ids"].tolist() == [[321003, 321009]]


def test_edit_cfg_source_tokens_are_merged_before_final_group_assembly() -> None:
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": [1, DUMMY_IMAGE_TOKEN_ID, DUMMY_IMAGE_TOKEN_ID, 2]},
        encoder_outs={"image_encoder": {"image_token_ids": [[3, 9]]}},
        image_token_offset=321000,
        generation_state={
            "cfg": {
                "unconditional_input_ids": [
                    7,
                    DUMMY_IMAGE_TOKEN_ID,
                    DUMMY_IMAGE_TOKEN_ID,
                    8,
                ]
            }
        },
    )

    merge_image_tokens_for_thinker(state)

    assert state.generation_state["cfg"]["unconditional_input_ids"] == [
        7,
        321003,
        321009,
        8,
    ]


def test_final_request_assembly_aligns_cfg_branches_and_records_offset() -> None:
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": [10, 11]},
        task_kind="edit",
        image_token_offset=321000,
        generation_state={
            "cfg": {
                "unconditional_input_ids": [7],
                "no_image_input_ids": [6, 5, 4],
                "text_scale": 4.0,
                "image_scale": 1.5,
                "rescale": 0.7,
            }
        },
    )

    conditional, group = prepare_dllm_input_group(state, mask_token_id=99)

    assert conditional == (99, 10, 11)
    assert group is not None
    assert group.primary_left_pad_length == 1
    assert tuple(companion.role for companion in group.companions) == (
        "unconditional",
        "no_image",
    )
    assert group.algorithm_args == {
        "cfg_scale": 4.0,
        "cfg_image_scale": 1.5,
        "cfg_rescale": 0.7,
        "force_image_only": True,
        "image_token_offset": 321000,
    }


class _Tokenizer:
    mask_token_id = 99

    _special = {
        SOI_TOKEN: 10,
        EOI_TOKEN: 11,
        BOI_TOKEN: 12,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special.setdefault(token, 100 + len(self._special))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        parts = re.findall(r"<[^>]+>|[^<]+", text)
        return [
            (
                self.convert_tokens_to_ids(part)
                if part.startswith("<")
                else 1000 + len(part)
            )
            for part in parts
            if part
        ]


def _preprocessor() -> LLaDA2Preprocessor:
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor._max_seq_len = 8192
    preprocessor._tokenizer = _Tokenizer()
    preprocessor._image_token_offset = 321000
    preprocessor._soi_id = preprocessor._tokenizer.convert_tokens_to_ids(SOI_TOKEN)
    preprocessor._eoi_id = preprocessor._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
    preprocessor._boi_id = preprocessor._tokenizer.convert_tokens_to_ids(BOI_TOKEN)
    preprocessor._merge_size = 1
    preprocessor._factor = 16
    return preprocessor


def test_non_thinking_t2i_builds_raw_cfg_group_and_exact_output_grid() -> None:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "paint a red fox"}],
            metadata={
                "output_modalities": ["image"],
                "image_generation": {
                    "width": 1024,
                    "height": 768,
                    "cfg_scale": 4.0,
                    "cfg_rescale": 0.7,
                },
            },
        ),
        data=None,
    )

    result = asyncio.run(_preprocessor()(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)

    assert state.task_kind == "t2i"
    assert state.generation_state["image_grid"] == {"height": 24, "width": 32}
    assert state.encoder_inputs["image_encoder"]["_skip"] is True
    cfg = state.generation_state["cfg"]
    assert cfg["scale"] == 4.0
    assert "left_pad" not in " ".join(cfg)
    conditional, group = prepare_dllm_input_group(state, mask_token_id=99)
    assert group is not None
    assert len(conditional) == len(group.companions[0].input_ids)


def test_image_modality_without_params_builds_default_t2i() -> None:
    payload = StagePayload(
        request_id="request-modality-only",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "paint a red fox"}],
            metadata={"output_modalities": ["image"]},
        ),
        data=None,
    )

    result = asyncio.run(_preprocessor()(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)

    assert state.task_kind == "t2i"
    assert state.generation_state["output_size"] == {
        "height": 1024,
        "width": 1024,
    }
    assert state.request_metadata["image_generation"] == {}


def test_image_modality_with_source_image_builds_default_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = StagePayload(
        request_id="request-modality-edit",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "make this brighter"}],
                "images": [Image.new("RGB", (64, 64))],
            },
            metadata={"output_modalities": ["image"]},
        ),
        data=None,
    )

    preprocessor = _preprocessor()
    captured: dict[str, object] = {}

    def _build_edit_payload(
        payload,
        messages,
        images,
        request_metadata,
        image_generation,
    ):
        del payload, messages, images
        captured["params"] = image_generation
        return StagePayload(
            request_id="request-modality-edit",
            request=OmniRequest(inputs=[], metadata=request_metadata),
            data=LLaDA2UniPipelineState(
                task_kind="edit",
                request_metadata=request_metadata,
            ).to_dict(),
        )

    monkeypatch.setattr(preprocessor, "_build_edit_payload", _build_edit_payload)

    result = asyncio.run(preprocessor(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)

    assert state.task_kind == "edit"
    assert state.request_metadata["image_generation"] == {}
    assert captured["params"] == {}


@pytest.mark.parametrize(
    "target_size",
    (
        {"width": 1024},
        {"height": 1024},
        {"size": "1024x1024"},
    ),
)
def test_image_edit_rejects_explicit_target_dimensions(target_size) -> None:
    payload = StagePayload(
        request_id="request-edit-size",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "make this brighter"}],
                "images": [Image.new("RGB", (64, 64))],
            },
            metadata={
                "output_modalities": ["image"],
                "image_generation": target_size,
            },
        ),
        data=None,
    )

    with pytest.raises(ValueError, match="editing follows source dimensions"):
        asyncio.run(_preprocessor()(payload))


def test_image_edit_resolution_multiplier_scales_source_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.llada2_uni.components import preprocessor as module

    preprocessor = _preprocessor()
    preprocessor._image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=1,
        merge_size=1,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        rescale_factor=1.0,
    )
    monkeypatch.setattr(module, "preprocess_image_edit", lambda images, factor: images)
    monkeypatch.setattr(
        module,
        "edit_image_pixel_values",
        lambda *args, **kwargs: {
            "pixel_values": torch.zeros(6, 3),
            "image_grid_thw": torch.tensor([[1, 2, 3]]),
        },
    )
    payload = StagePayload(
        request_id="request-edit-multiplier",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "make this brighter"}],
                "images": [Image.new("RGB", (64, 64))],
            },
            metadata={
                "output_modalities": ["image"],
                "image_generation": {"resolution_multiplier": 3},
            },
        ),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)

    assert state.task_kind == "edit"
    assert state.generation_state["image_grid"] == {"height": 2, "width": 3}
    assert state.generation_state["output_size"] == {"height": 96, "width": 144}


def test_t2i_context_validation_uses_exact_image_grid_budget() -> None:
    payload = StagePayload(
        request_id="request-context",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "paint a red fox"}],
            metadata={
                "output_modalities": ["image"],
                "image_generation": {
                    "width": 32,
                    "height": 32,
                    "cfg_scale": 4.0,
                },
            },
        ),
        data=None,
    )

    probe = asyncio.run(_preprocessor()(payload))
    prompt_len = (
        LLaDA2UniPipelineState.from_dict(probe.data).prompt["input_ids"].numel()
    )
    constrained = _preprocessor()
    constrained._max_seq_len = prompt_len + 1

    result = asyncio.run(constrained(payload))

    assert LLaDA2UniPipelineState.from_dict(result.data).generation_state[
        "image_grid"
    ] == {"height": 1, "width": 1}


def test_t2i_cfg_scale_one_stays_single_row() -> None:
    payload = StagePayload(
        request_id="request-no-cfg",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "paint a red fox"}],
            metadata={
                "output_modalities": ["image"],
                "image_generation": {
                    "width": 32,
                    "height": 32,
                    "cfg_scale": 1.0,
                },
            },
        ),
        data=None,
    )

    result = asyncio.run(_preprocessor()(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)

    assert "cfg" not in state.generation_state
    _, group = prepare_dllm_input_group(state, mask_token_id=99)
    assert group is None


def test_unknown_image_generation_mode_is_rejected() -> None:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "paint a red fox"}],
            metadata={
                "output_modalities": ["image"],
                "image_generation": {"mode": "turbo"},
            },
        ),
        data=None,
    )

    with pytest.raises(ValueError, match="mode must be"):
        asyncio.run(_preprocessor()(payload))


def test_native_image_request_uses_exact_grid_token_budget() -> None:
    from sglang_omni.models.llada2_uni.request_builders import (
        resolve_thinker_max_new_tokens,
    )

    state = LLaDA2UniPipelineState(
        task_kind="t2i",
        generation_state={"image_grid": {"height": 24, "width": 32}},
    )

    assert resolve_thinker_max_new_tokens(state, {"max_new_tokens": 2048}) == 768


def test_native_image_request_uses_checkpoint_vocab_boundary_without_cfg() -> None:
    state = LLaDA2UniPipelineState(
        task_kind="edit",
        image_token_offset=321000,
    )

    assert resolve_native_image_token_offset(state, vocab_size=337384) == 321000
    assert (
        resolve_native_image_token_offset(
            LLaDA2UniPipelineState(task_kind="chat"), vocab_size=337384
        )
        is None
    )


def test_native_image_grid_requires_exact_decoder_pixel_alignment() -> None:
    from sglang_omni.models.llada2_uni.components.preprocessor import (
        resolve_native_image_grid,
    )

    with pytest.raises(ValueError, match="divisible by 32"):
        resolve_native_image_grid({"height": 1000, "width": 1024})


def test_native_image_grid_accepts_shared_size_string() -> None:
    from sglang_omni.models.llada2_uni.components.preprocessor import (
        resolve_native_image_grid,
    )

    assert resolve_native_image_grid(
        {
            "size": "1024x768",
            "width": 32,
            "height": 32,
        }
    ) == (24, 32, 768, 1024, 2)


def test_native_image_grid_rejects_malformed_shared_size() -> None:
    from sglang_omni.models.llada2_uni.components.preprocessor import (
        resolve_native_image_grid,
    )

    with pytest.raises(ValueError, match="size must use WIDTHxHEIGHT"):
        resolve_native_image_grid({"size": "1024"})


def test_native_image_grid_rejects_unsupported_output_format() -> None:
    from sglang_omni.models.llada2_uni.components.preprocessor import (
        resolve_native_image_grid,
    )

    with pytest.raises(ValueError, match="format must be 'png'"):
        resolve_native_image_grid({"format": "jpeg"})


def test_edit_without_any_cfg_guidance_stays_single_row() -> None:
    from sglang_omni.models.llada2_uni.components.preprocessor import (
        _resolve_edit_cfg_scales,
    )

    assert _resolve_edit_cfg_scales(
        {"cfg_text_scale": 0.0, "cfg_image_scale": 0.0}
    ) == (0.0, 0.0)


def test_legacy_edit_cfg_scale_one_stays_single_row() -> None:
    from sglang_omni.models.llada2_uni.components.preprocessor import (
        _resolve_edit_cfg_scales,
    )

    assert _resolve_edit_cfg_scales({"cfg_scale": 1.0}) == (0.0, 0.0)
