# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 runner: prefill, generation limits, sampling and graph replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.voxcpm2.components.projections import VoxCPM2Projections
from sglang_omni.models.voxcpm2.model_runner import VoxCPM2ModelRunner, recipe_groups
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.models.voxcpm2.sglang_model import VoxCPM2SGLangModel

_HIDDEN = 6
_FEAT_DIM = 8
_PATCH_SIZE = 4


class _GenerationModel:
    patch_size = 4
    feat_dim = 8

    def parameters(self):
        yield torch.zeros(1, dtype=torch.bfloat16)

    def decode_patch(self, cond, **kwargs):
        return torch.zeros(len(cond), 4, 8), torch.zeros(len(cond), 2)

    def stop_flags(self, rows):
        return torch.ones(len(rows), dtype=torch.bool)


def _generation_runner():
    instance = object.__new__(VoxCPM2ModelRunner)
    instance.model = _GenerationModel()
    return instance


def _generation_request(*, patches=0, min_len=2, max_len=8):
    return SimpleNamespace(
        data=SimpleNamespace(
            state=VoxCPM2State(min_len=min_len, max_len=max_len),
            cond=None,
            next_embed=None,
            decode_input_embeds=[],
            noise_generator=None,
            latent_patches=[torch.zeros(4, 8) for _ in range(patches)],
            finish_reason=None,
            req=SimpleNamespace(finished_reason=None),
        )
    )


def test_stop_uses_upstreams_zero_based_generated_patch_index():
    req = _generation_request(patches=2, min_len=2)
    instance = _generation_runner()
    instance.advance([req], rows=None, is_prefill=False)
    assert len(req.data.latent_patches) == 3
    assert req.data.finish_reason is None
    instance.advance([req], rows=None, is_prefill=False)
    assert len(req.data.latent_patches) == 4
    assert req.data.finish_reason == "stop"


def test_one_patch_limit_is_enforced_on_prefill():
    req = _generation_request(max_len=1)
    _generation_runner().advance([req], rows=torch.tensor([0]), is_prefill=True)
    assert len(req.data.latent_patches) == 1
    assert req.data.finish_reason == "length"
    assert req.data.req.finished_reason is None


def test_continuation_condition_is_cast_to_the_model_dtype():
    req = _generation_request()
    req.data.cond = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    condition = _generation_runner().batch_cond([req.data, _generation_request().data])
    assert condition.dtype == torch.bfloat16
    torch.testing.assert_close(condition[0].float(), req.data.cond[0])
    assert torch.count_nonzero(condition[1]) == 0


class _PrefillModel:
    def build_input_embeds(self, text_token, audio_feat, text_mask, audio_mask):
        del audio_feat, text_mask, audio_mask
        return torch.ones(int(text_token.shape[0]), _HIDDEN)


class _Prefill:
    def __init__(self, length: int):
        self.text_token = torch.zeros(length, dtype=torch.int32)
        self.audio_feat = torch.zeros(length, 4, 8)
        self.text_mask = torch.ones(length, dtype=torch.int32)
        self.audio_mask = torch.zeros(length, dtype=torch.int32)


class _PrefixReq:
    def __init__(self, prefix_indices):
        self.prefix_indices = prefix_indices


class _PrefillData:
    def __init__(self, length, prefix_indices):
        self.prefill = _Prefill(length)
        self.req = _PrefixReq(prefix_indices)
        self.req.extend_range = SimpleNamespace(length=length)
        self.decode_input_embeds = []


class _PrefillRequest:
    def __init__(self, length, prefix_indices):
        self.data = _PrefillData(length, prefix_indices)


class _PrefillBatch:
    input_embeds = None
    replace_embeds = None

    def __init__(self, length=0):
        self.input_ids = torch.zeros(length, dtype=torch.long)


def _prefill_runner():
    runner = object.__new__(VoxCPM2ModelRunner)
    runner.model = _PrefillModel()
    return runner


def test_the_batch_embedding_is_every_request_end_to_end():
    batch = _PrefillBatch(8)
    _prefill_runner().before_prefill(
        batch,
        None,
        [
            _PrefillRequest(3, torch.empty(0, dtype=torch.int64)),
            _PrefillRequest(5, torch.empty(0, dtype=torch.int64)),
        ],
    )
    assert get_omni_prefill_inputs(batch).input_embeds.shape == (8, _HIDDEN)


def test_an_empty_prefix_tensor_is_not_a_reuse():
    """prefix_indices is a tensor, so testing it for truth raises instead."""
    batch = _PrefillBatch(3)
    _prefill_runner().before_prefill(
        batch, None, [_PrefillRequest(3, torch.empty(0, dtype=torch.int64))]
    )
    assert get_omni_prefill_inputs(batch).input_embeds.shape == (3, _HIDDEN)


def test_an_empty_prefix_list_is_not_a_reuse():
    batch = _PrefillBatch(3)
    _prefill_runner().before_prefill(batch, None, [_PrefillRequest(3, [])])
    assert get_omni_prefill_inputs(batch).input_embeds.shape == (3, _HIDDEN)


def test_a_reused_prefix_is_refused_rather_than_mis_assembled():
    """The embedding is built from the full layout, so a short extend would
    silently pair the wrong rows with the wrong positions."""
    with pytest.raises(RuntimeError, match="prefix reuse"):
        _prefill_runner().before_prefill(
            _PrefillBatch(), None, [_PrefillRequest(5, torch.tensor([0, 1]))]
        )


def test_an_empty_batch_leaves_the_forward_batch_alone():
    batch = _PrefillBatch()
    _prefill_runner().before_prefill(batch, None, [])
    assert batch.input_embeds is None


def test_audio_masks_follow_the_packed_request_order():
    requests = [_PrefillRequest(3, []), _PrefillRequest(2, [])]
    requests[0].data.prefill.audio_mask = torch.tensor([0, 1, 1])
    requests[1].data.prefill.audio_mask = torch.tensor([1, 0])
    batch = _PrefillBatch(5)
    _prefill_runner().before_prefill(batch, None, requests)
    assert get_omni_prefill_inputs(batch).audio_mask.tolist() == [0, 1, 1, 1, 0]


class _NoiseModel:
    feat_dim = _FEAT_DIM
    patch_size = _PATCH_SIZE

    def parameters(self):
        yield torch.zeros(1)


class _NoiseData:
    def __init__(self, seed: int | None):
        self.noise_generator = (
            None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        )


def _noise_runner():
    runner = object.__new__(VoxCPM2ModelRunner)
    runner.model = _NoiseModel()
    return runner


def _draw_noise(seeds, steps=1):
    runner = _noise_runner()
    rows = [_NoiseData(seed) for seed in seeds]
    return [runner.batch_noise(rows) for _ in range(steps)]


def test_unseeded_batch_leaves_the_draw_to_the_sampler():
    assert _draw_noise([None, None]) == [None]


def test_same_seed_replays_the_same_noise_sequence():
    first = _draw_noise([7], steps=3)
    second = _draw_noise([7], steps=3)
    for one, two in zip(first, second, strict=True):
        assert torch.equal(one, two)


def test_a_step_does_not_repeat_the_previous_step_noise():
    first, second = _draw_noise([7], steps=2)
    assert not torch.equal(first, second)


def test_each_row_draws_from_its_own_seed():
    noise = _draw_noise([7, 7, 11])[0]
    assert noise.shape == (3, _FEAT_DIM, _PATCH_SIZE)
    assert torch.equal(noise[0], noise[1])
    assert not torch.equal(noise[0], noise[2])


def test_a_seeded_row_is_unaffected_by_an_unseeded_neighbour():
    torch.manual_seed(0)
    alone = _draw_noise([7])[0]
    torch.manual_seed(1234)
    shared = _draw_noise([7, None])[0]
    assert torch.equal(alone[0], shared[0])


class _RecipeData:
    def __init__(self, timesteps: int = 10, cfg: float = 2.0):
        self.state = VoxCPM2State(inference_timesteps=timesteps, cfg_value=cfg)


def test_one_recipe_stays_one_group():
    groups = recipe_groups([_RecipeData(), _RecipeData(), _RecipeData()])
    assert groups == [[0, 1, 2]]


def test_a_differing_step_count_splits_the_batch():
    groups = recipe_groups([_RecipeData(10), _RecipeData(12), _RecipeData(10)])
    assert groups == [[0, 2], [1]]


def test_a_differing_guidance_scale_splits_the_batch():
    groups = recipe_groups([_RecipeData(cfg=2.0), _RecipeData(cfg=1.5)])
    assert groups == [[0], [1]]


def test_every_request_lands_in_exactly_one_group():
    """A dropped index would silently leave a request without a patch."""
    rows = [_RecipeData(10), _RecipeData(12), _RecipeData(10), _RecipeData(8)]
    indices = sorted(index for group in recipe_groups(rows) for index in group)
    assert indices == list(range(len(rows)))


def test_decode_consumes_both_replay_outputs_after_prefill_and_batch_changes():
    model = VoxCPM2SGLangModel.__new__(VoxCPM2SGLangModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(lm_config=SimpleNamespace(hidden_size=2))
    instance = object.__new__(VoxCPM2ModelRunner)
    instance.model = model
    observed = []
    instance.advance = lambda requests, rows, is_prefill: observed.append(
        model.rows(None)
    )
    for step, size in enumerate((2, 1, 2)):
        # A previous prefill overwrites Python attributes, but replay does not.
        model.last_lm_hidden = torch.full((8, 2), -1.0)
        model.last_residual_hidden = torch.full((8, 2), -2.0)
        hidden = torch.arange(size * 4).reshape(size, 4).float() + step * 100
        result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden))
        instance.post_decode(result, None, None, [object()] * size)
        lm, residual = observed[-1]
        torch.testing.assert_close(lm, hidden[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(residual, hidden[:, 2:], rtol=0, atol=0)


def _mask_projections():
    model = VoxCPM2Projections(
        lm_hidden_size=2,
        encoder_hidden_size=2,
        dit_hidden_size=2,
        quantization_latent_dim=2,
        quantization_scale=9,
    )
    model.fusion_concat_proj = nn.Identity()
    model.fsq_layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.fsq_layer.weight.copy_(2 * torch.eye(2))
    return model


def _forward_with_mask(hidden, embed, audio_mask=None):
    model = SimpleNamespace(
        graph_feedback_buffer=None,
        projections=_mask_projections(),
        forward_base=lambda *args: hidden,
        forward_residual=lambda inputs, *args: inputs,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: audio_mask is None,
            is_extend=lambda: audio_mask is not None,
        )
    )
    VoxCPM2SGLangModel.forward(
        model,
        input_ids=torch.zeros(len(hidden), dtype=torch.long),
        positions=torch.arange(len(hidden)),
        forward_batch=batch,
        input_embeds=embed,
        audio_mask=audio_mask,
    )
    return model.last_lm_hidden, model.last_residual_hidden


def test_text_prefill_is_not_quantized_and_does_not_condition_residual_audio():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    embed = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    actual, fused = _forward_with_mask(hidden, embed, torch.zeros(2))
    torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
    torch.testing.assert_close(fused[:, :2], hidden, rtol=0, atol=0)
    assert torch.count_nonzero(fused[:, 2:]) == 0


def test_mixed_prefill_quantizes_only_audio_positions():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    embed = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    actual, fused = _forward_with_mask(hidden, embed, torch.tensor([0, 1, 0]))
    torch.testing.assert_close(
        actual, torch.tensor([[1.0, 2.0], [6.0, 8.0], [5.0, 6.0]])
    )
    torch.testing.assert_close(
        fused[:, 2:], torch.tensor([[0.0, 0.0], [9.0, 10.0], [0.0, 0.0]])
    )


def test_decode_quantizes_every_row_and_keeps_the_previous_patch_embedding():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    embed = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    actual, fused = _forward_with_mask(hidden, embed)
    torch.testing.assert_close(actual, 2 * hidden, rtol=0, atol=0)
    torch.testing.assert_close(fused[:, 2:], embed, rtol=0, atol=0)


def test_graph_feedback_keeps_buffer_address_when_inputs_change():
    model = VoxCPM2SGLangModel.__new__(VoxCPM2SGLangModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(lm_config=SimpleNamespace(hidden_size=3))
    model.register_parameter("weight", nn.Parameter(torch.zeros(1)))
    model.enable_graph_feedback(2)
    buffer = model.graph_feedback_buffer
    feedback = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    model.write_feedback(feedback)
    assert model.graph_feedback_buffer is buffer
    torch.testing.assert_close(buffer, feedback)
