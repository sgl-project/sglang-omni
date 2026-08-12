# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from sglang_omni.models.llada2_uni.components.image_decoder import (
    LLaDA2ImageDecoder,
    _remap_zimage_checkpoint_keys,
    euler_sample,
)
from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def test_extract_image_vq_tokens_uses_checkpoint_offset_and_exact_grid() -> None:
    state = LLaDA2UniPipelineState(
        task_kind="t2i",
        image_token_offset=321000,
        thinker_out={"output_ids": [5, 321001, 321002, 321003, 321004]},
        generation_state={"image_grid": {"height": 2, "width": 2}},
        request_metadata={"image_generation": {"decoder_steps": 8, "seed": 11}},
    )

    tokens, height, width, params = extract_image_vq_tokens(state)

    assert tokens == [1, 2, 3, 4]
    assert (height, width) == (2, 2)
    assert params == {"decoder_steps": 8, "seed": 11}


def test_extract_image_vq_tokens_rejects_incomplete_native_image() -> None:
    state = LLaDA2UniPipelineState(
        task_kind="edit",
        image_token_offset=321000,
        thinker_out={"output_ids": [321001, 321002, 321003]},
        generation_state={"image_grid": {"height": 2, "width": 2}},
    )

    with pytest.raises(ValueError, match="expected 4 VQ tokens"):
        extract_image_vq_tokens(state)


def test_text_requests_skip_image_decoder() -> None:
    state = LLaDA2UniPipelineState(
        task_kind="chat",
        image_token_offset=321000,
        thinker_out={"output_ids": [321001]},
    )

    assert extract_image_vq_tokens(state) is None


def test_euler_sampler_matches_reference_num_steps_time_grid() -> None:
    calls: list[float] = []

    def _velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        calls.append(float(time[0]))
        return torch.ones_like(value)

    result = euler_sample(torch.zeros(1, 2), _velocity, num_steps=4)

    assert len(calls) == 3
    assert torch.allclose(result, torch.ones(1, 2))


def test_decoder_validates_exact_codebook_grid() -> None:
    with pytest.raises(ValueError, match=r"h \* w"):
        LLaDA2ImageDecoder._validate_decode_inputs(
            token_ids=[1, 2, 3], height=2, width=2, num_steps=8
        )

    with pytest.raises(ValueError, match="codebook"):
        LLaDA2ImageDecoder._validate_decode_inputs(
            token_ids=[1, 2, 3, 16384], height=2, width=2, num_steps=8
        )


def test_decoder_remaps_reference_semantic_embedder_for_diffusers() -> None:
    semantic_weight = torch.tensor([1.0])
    unrelated_weight = torch.tensor([2.0])

    remapped = _remap_zimage_checkpoint_keys(
        {
            "semantic_embedder.weight": semantic_weight,
            "transformer_blocks.0.weight": unrelated_weight,
        }
    )

    assert remapped == {
        "cap_embedder.weight": semantic_weight,
        "transformer_blocks.0.weight": unrelated_weight,
    }


def test_turbo_decoder_is_rejected(tmp_path) -> None:
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")

    with pytest.raises(NotImplementedError, match="not supported"):
        decoder.decode([1], 1, 1, decode_mode="decoder-turbo")


def test_decoder_resolves_remote_checkpoint_when_not_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from sglang_omni.models.llada2_uni.components import image_decoder

    observed = []

    def _resolve(model_path, *, local_files_only):
        observed.append((model_path, local_files_only))
        return tmp_path

    monkeypatch.setattr(image_decoder, "resolve_model_path", _resolve)

    decoder = LLaDA2ImageDecoder("inclusionAI/LLaDA2.0-Uni", device="cpu")

    assert decoder.model_path == tmp_path
    assert observed == [("inclusionAI/LLaDA2.0-Uni", False)]
