# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    FunCosyVoice3Flow,
    _pack_flow_inputs,
)


class _RecordingEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, torch.Tensor | bool]] = []

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        streaming: bool,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "x": x.detach().clone(),
                "mask": mask.detach().clone(),
                "mu": mu.detach().clone(),
                "t": t.detach().clone(),
                "spks": spks.detach().clone(),
                "cond": cond.detach().clone(),
                "streaming": streaming,
            }
        )
        return (0.1 * x + mu + spks.unsqueeze(-1) + cond) * mask


class _FakeDecoder:
    def __init__(
        self,
        channels: int,
        *,
        max_frames: int = 64,
        estimator: object | None = None,
    ) -> None:
        self.rand_noise = (
            torch.arange(channels * max_frames, dtype=torch.float32).reshape(
                1, channels, max_frames
            )
            / 100
        )
        self.t_scheduler = "cosine"
        self.inference_cfg_rate = 0.7
        self.estimator = estimator or _RecordingEstimator()

    def forward_estimator(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        streaming: bool,
    ) -> torch.Tensor:
        return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)


class _FakeFlow(torch.nn.Module):
    def __init__(
        self,
        *,
        channels: int = 4,
        token_mel_ratio: int = 2,
        max_frames: int = 64,
        estimator: object | None = None,
    ) -> None:
        super().__init__()
        self.output_size = channels
        self.token_mel_ratio = token_mel_ratio
        self.input_embedding = torch.nn.Embedding(32, channels)
        self.spk_embed_affine_layer = torch.nn.Linear(3, channels, bias=False)
        self.pre_lookahead_layer = torch.nn.Identity()
        self.decoder = _FakeDecoder(
            channels, max_frames=max_frames, estimator=estimator
        )
        with torch.no_grad():
            self.input_embedding.weight.copy_(
                torch.arange(32 * channels, dtype=torch.float32).reshape(32, channels)
                / 50
            )
            self.spk_embed_affine_layer.weight.copy_(
                torch.arange(channels * 3, dtype=torch.float32).reshape(channels, 3)
                / 20
            )


def _input(
    token: list[int],
    *,
    prompt_token: list[int] | None = None,
    prompt_value: float = 0.0,
    embedding: tuple[float, float, float] = (1.0, 2.0, 3.0),
    channels: int = 4,
) -> FlowBatchInput:
    prompt_token = prompt_token or []
    return FlowBatchInput(
        token=torch.tensor([token], dtype=torch.int32),
        prompt_token=torch.tensor([prompt_token], dtype=torch.int32),
        prompt_feat=torch.full((1, len(prompt_token) * 2, channels), prompt_value),
        embedding=torch.tensor([embedding]),
    )


def _infer_flow(flow: _FakeFlow, inputs: list[FlowBatchInput]) -> list[torch.Tensor]:
    return FunCosyVoice3Flow(flow).inference(inputs)


def test_pack_flow_inputs_keeps_prompt_and_target_contiguous() -> None:
    flow = _FakeFlow()
    items = [
        _input([0, 8, 0], prompt_token=[4]),
        _input([0], prompt_token=[5, 6, 7]),
    ]

    packed = _pack_flow_inputs(flow, items)

    assert packed.token.dtype == torch.int32
    assert packed.token.tolist() == [[4, 0, 8, 0], [5, 6, 7, 0]]
    assert packed.combined_token_lengths == (4, 4)
    assert packed.prompt_token_lengths == (1, 3)
    assert packed.target_token_lengths == (3, 1)
    assert packed.combined_token_lengths_tensor.tolist() == [4, 4]
    assert packed.token_mask.squeeze(-1).tolist() == [
        [True, True, True, True],
        [True, True, True, True],
    ]


def test_pack_flow_inputs_builds_variable_length_token_masks() -> None:
    packed = _pack_flow_inputs(
        _FakeFlow(),
        [
            _input([0], prompt_token=[]),
            _input([3, 0], prompt_token=[4, 0]),
        ],
    )

    assert packed.token.tolist() == [[0, 0, 0, 0], [4, 0, 3, 0]]
    assert packed.token_mask.squeeze(-1).tolist() == [
        [True, False, False, False],
        [True, True, True, True],
    ]
    assert packed.prompt_mel_lengths == (0, 4)
    assert packed.total_mel_lengths == (2, 8)
    assert packed.total_mel_lengths_tensor.tolist() == [2, 8]


def test_flow_batch_builds_variable_length_mel_masks_and_conditions() -> None:
    flow = _FakeFlow()
    items = [
        _input([1], prompt_token=[], prompt_value=3.0),
        _input([2, 3], prompt_token=[4, 5], prompt_value=7.0),
    ]

    _infer_flow(flow, items)

    first = flow.decoder.estimator.calls[0]
    mask = first["mask"][:2]
    cond = first["cond"][:2]
    assert mask[:, 0].bool().tolist() == [
        [True, True, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ]
    assert torch.count_nonzero(cond[0]) == 0
    torch.testing.assert_close(cond[1, :, :4], torch.full((4, 4), 7.0))
    assert torch.count_nonzero(cond[1, :, 4:]) == 0


def test_flow_batch_cfg_uses_two_times_request_batch() -> None:
    flow = _FakeFlow()

    _infer_flow(
        flow,
        [_input([1]), _input([2]), _input([3])],
    )

    for call in flow.decoder.estimator.calls:
        assert call["x"].shape[0] == 6
        assert call["mask"].shape[0] == 6
        assert call["mu"].shape[0] == 6
        assert call["t"].shape[0] == 6
        assert call["spks"].shape[0] == 6
        assert call["cond"].shape[0] == 6
        assert call["streaming"] is False


def test_flow_batch_cfg_builds_conditional_and_unconditional_halves() -> None:
    flow = _FakeFlow()
    items = [
        _input([1], prompt_token=[2], prompt_value=2.0),
        _input([3], prompt_token=[4], prompt_value=4.0),
    ]

    _infer_flow(flow, items)

    first = flow.decoder.estimator.calls[0]
    torch.testing.assert_close(first["x"][:2], first["x"][2:])
    torch.testing.assert_close(first["mask"][:2], first["mask"][2:])
    assert torch.count_nonzero(first["mu"][:2]) > 0
    assert torch.count_nonzero(first["spks"][:2]) > 0
    assert torch.count_nonzero(first["cond"][:2]) > 0
    assert torch.count_nonzero(first["mu"][2:]) == 0
    assert torch.count_nonzero(first["spks"][2:]) == 0
    assert torch.count_nonzero(first["cond"][2:]) == 0


def test_flow_batch_reuses_same_noise_prefix_per_request() -> None:
    pair_flow = _FakeFlow()
    _infer_flow(pair_flow, [_input([1, 2]), _input([3, 4])])
    pair_noise = pair_flow.decoder.estimator.calls[0]["x"][:2]

    single_flow = _FakeFlow()
    _infer_flow(single_flow, [_input([1, 2])])
    single_noise = single_flow.decoder.estimator.calls[0]["x"][:1]

    torch.testing.assert_close(pair_noise[0], pair_noise[1])
    torch.testing.assert_close(pair_noise[:1], single_noise)


def test_flow_batch_matches_serial_reference_for_mixed_lengths() -> None:
    items = [
        _input([1, 0], prompt_token=[2], prompt_value=0.5),
        _input([3], prompt_token=[4, 5], prompt_value=1.5),
        _input([6, 7, 8], prompt_token=[], prompt_value=2.5),
    ]
    serial = [_infer_flow(_FakeFlow(), [item])[0] for item in items]
    batched = _infer_flow(_FakeFlow(), items)

    assert [mel.shape for mel in batched] == [(1, 4, 4), (1, 4, 2), (1, 4, 6)]
    for actual, expected in zip(batched, serial, strict=True):
        torch.testing.assert_close(actual, expected)


def test_flow_batch_crops_each_prompt_independently() -> None:
    items = [
        _input([1, 2], prompt_token=[3], prompt_value=0.5),
        _input([4, 5], prompt_token=[6, 7, 8], prompt_value=1.5),
    ]
    serial = [_infer_flow(_FakeFlow(), [item])[0] for item in items]

    batched = _infer_flow(_FakeFlow(), items)

    for actual, expected in zip(batched, serial, strict=True):
        assert actual.shape[-1] == 4
        torch.testing.assert_close(actual, expected)


def test_flow_batch_rejects_noise_buffer_overflow() -> None:
    flow = _FakeFlow(max_frames=4)

    with pytest.raises(ValueError, match="rand_noise.*4.*6"):
        _infer_flow(flow, [_input([1, 2, 3])])


def test_flow_batch_rejects_prompt_alignment_mismatch() -> None:
    item = _input([1], prompt_token=[2])
    item = FlowBatchInput(
        token=item.token,
        prompt_token=item.prompt_token,
        prompt_feat=torch.zeros(1, 1, 4),
        embedding=item.embedding,
    )

    with pytest.raises(ValueError, match="prompt feature length"):
        _infer_flow(_FakeFlow(), [item])
