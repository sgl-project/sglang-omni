# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from sglang_omni.models.ming_omni.talker.talker_module.aggregator import Aggregator
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT
from sglang_omni.models.ming_omni.talker.talker_module.execution import (
    TalkerExecutionConfig,
)
from sglang_omni.models.ming_omni.talker.talker_module.rotary import (
    CachedRotaryEmbedding,
    RotaryInputs,
    apply_rotary_embedding,
)


def test_native_rotary_keeps_partial_head_semantics() -> None:
    torch.manual_seed(1)
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 5, 8)
    freqs, scale = RotaryEmbedding(8).forward_from_seq_len(5)
    expected_query = query.clone()
    expected_key = key.clone()
    expected_query[:, :2] = apply_rotary_pos_emb(expected_query[:, :2], freqs, 1.0)
    expected_key[:, :2] = apply_rotary_pos_emb(expected_key[:, :2], freqs, 1.0)

    actual_query, actual_key = apply_rotary_embedding(
        query, key, (freqs, scale), pe_attn_head=2
    )

    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)


def test_cached_rotary_owns_fp32_coefficients_and_bounded_positions() -> None:
    seq_len, max_batch_size, dim = 5, 2, 8
    kernel = Mock()
    rotary = CachedRotaryEmbedding(
        dim,
        kernel=kernel,
        seq_len=seq_len,
        max_batch_size=max_batch_size,
    )
    freqs, _ = rotary.forward_from_seq_len(seq_len)
    phase = freqs.reshape(seq_len, dim)[:, 0::2]
    expected_cache = torch.cat((phase.cos(), phase.sin()), dim=-1).contiguous()

    assert rotary.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(rotary.cos_sin_cache, expected_cache, rtol=0, atol=0)
    assert "cos_sin_cache" not in rotary.state_dict()
    assert "positions" not in rotary.state_dict()
    resident_pointer = rotary.positions.data_ptr()
    for batch_size in (1, max_batch_size, 1):
        inputs = rotary.for_batch(batch_size)
        assert inputs.kernel is kernel
        assert inputs.cos_sin_cache is rotary.cos_sin_cache
        assert inputs.positions.tolist() == list(range(seq_len)) * batch_size
        assert inputs.positions.data_ptr() == resident_pointer
    with pytest.raises(RuntimeError, match="exceeds dimension size"):
        rotary.for_batch(max_batch_size + 1)
    assert rotary.positions.data_ptr() == resident_pointer
    assert set(rotary.state_dict()) == set(RotaryEmbedding(dim).state_dict())
    kernel.assert_not_called()


@pytest.mark.parametrize(
    ("component_type", "options"),
    [
        (Aggregator, {"qk_norm": "rms_norm"}),
        (Aggregator, {"pe_attn_head": 1}),
        (DiT, {"qk_norm": "rms_norm"}),
        (DiT, {"pe_attn_head": 1}),
        (DiT, {"grad_checkpointing": True}),
    ],
)
def test_joint_rotary_rejects_unsupported_component_options(
    component_type: type[Aggregator] | type[DiT],
    options: dict[str, object],
) -> None:
    kernel = Mock()
    with pytest.raises(ValueError, match="Joint RoPE requires"):
        component_type(
            in_channels=4,
            hidden_size=8,
            depth=1,
            num_heads=2,
            execution_config=TalkerExecutionConfig(
                rope_kernel=kernel, rope_seq_len=3, rope_max_batch_size=2
            ),
            **options,
        )
    kernel.assert_not_called()


def test_joint_rotary_hands_the_kernel_token_major_aliases() -> None:
    batch, seq_len, heads, head_dim = 2, 5, 3, 8
    query_storage = torch.randn(batch, seq_len, heads, head_dim)
    key_storage = torch.randn(batch, seq_len, heads, head_dim)
    query = query_storage.transpose(1, 2)
    key = key_storage.transpose(1, 2)
    cache = torch.randn(seq_len, head_dim)
    positions = torch.arange(seq_len).repeat(batch)
    expected_cache, expected_positions = cache.clone(), positions.clone()
    seen: dict[str, object] = {}

    def fake_apply_rope_inplace(
        query_tokens,
        key_tokens,
        cos_sin_cache,
        token_positions,
        *,
        is_neox,
    ) -> None:
        seen.update(
            query=query_tokens,
            key=key_tokens,
            cache=cos_sin_cache,
            positions=token_positions,
            is_neox=is_neox,
        )
        query_tokens.fill_(3.0)
        key_tokens.fill_(4.0)

    actual_query, actual_key = apply_rotary_embedding(
        query,
        key,
        RotaryInputs(cache, positions, fake_apply_rope_inplace),
    )

    assert seen["query"].shape == (batch * seq_len, heads, head_dim)
    assert seen["key"].shape == (batch * seq_len, heads, head_dim)
    assert seen["query"].data_ptr() == query_storage.data_ptr()
    assert seen["key"].data_ptr() == key_storage.data_ptr()
    assert seen["cache"] is cache
    assert seen["positions"] is positions
    assert seen["is_neox"] is False
    assert actual_query is query
    assert actual_key is key
    assert torch.all(query_storage == 3.0)
    assert torch.all(key_storage == 4.0)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(positions, expected_positions, rtol=0, atol=0)


def test_joint_rotary_propagates_kernel_failure_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.ming_omni.talker.talker_module import rotary

    native = Mock(side_effect=AssertionError("Native fallback must not run"))
    monkeypatch.setattr(rotary, "apply_rotary_pos_emb", native)
    error = RuntimeError("provider failure")
    kernel = Mock(side_effect=error)
    query = torch.randn(2, 3, 2, 4).transpose(1, 2)
    rope = RotaryInputs(torch.randn(3, 4), torch.arange(3).repeat(2), kernel)

    with pytest.raises(RuntimeError, match="provider failure") as raised:
        apply_rotary_embedding(query, query.clone(), rope)

    assert raised.value is error
    kernel.assert_called_once()
    native.assert_not_called()


def test_no_rotary_returns_inputs_unchanged() -> None:
    query, key = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)

    actual_query, actual_key = apply_rotary_embedding(query, key, None)

    assert actual_query is query
    assert actual_key is key


@pytest.mark.parametrize(
    ("component_type", "condition_key"),
    [(Aggregator, "llm_input_dim"), (DiT, "llm_cond_dim")],
)
def test_shared_acoustic_components_keep_native_default(
    monkeypatch: pytest.MonkeyPatch,
    component_type: type[Aggregator] | type[DiT],
    condition_key: str,
) -> None:
    from sglang_omni.platforms import current_platform

    provider = Mock(side_effect=AssertionError("Native must not query joint RoPE"))
    monkeypatch.setattr(current_platform, "get_joint_rope_inplace_kernel", provider)
    component = component_type(
        in_channels=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        attn_backend="torch",
        pe_attn_head=1,
        **{condition_key: 6},
    )

    assert type(component.rotary_embed) is RotaryEmbedding
    component.eval()
    with torch.no_grad():
        if component_type is Aggregator:
            output = component(torch.randn(2, 2, 4))
            assert output.shape == (2, 1, 6)
        else:
            output = component(
                torch.randn(2, 2, 4),
                torch.rand(2),
                torch.randn(2, 1, 6),
                torch.randn(2, 3, 4),
            )
            assert output.shape == (2, 6, 4)
    provider.assert_not_called()


def test_execution_config_reaches_both_acoustic_components() -> None:
    aggregator_kernel, dit_kernel = Mock(), Mock()
    aggregator_config = TalkerExecutionConfig(
        attn_backend="torch",
        rope_kernel=aggregator_kernel,
        rope_seq_len=3,
        rope_max_batch_size=4,
    )
    dit_config = TalkerExecutionConfig(
        attn_backend="torch",
        rope_kernel=dit_kernel,
        rope_seq_len=5,
        rope_max_batch_size=8,
    )

    aggregator = Aggregator(
        in_channels=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        llm_input_dim=6,
        execution_config=aggregator_config,
    )
    dit = DiT(
        in_channels=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        llm_cond_dim=6,
        execution_config=dit_config,
    )

    assert isinstance(aggregator.rotary_embed, CachedRotaryEmbedding)
    assert aggregator.rotary_embed.cos_sin_cache.shape == (3, 4)
    assert aggregator.rotary_embed.positions.numel() == 12
    assert aggregator.rotary_embed.kernel is aggregator_kernel
    assert aggregator.blocks[0].attn.attn_backend == "torch"
    assert isinstance(dit.rotary_embed, CachedRotaryEmbedding)
    assert dit.rotary_embed.cos_sin_cache.shape == (5, 4)
    assert dit.rotary_embed.positions.numel() == 40
    assert dit.rotary_embed.kernel is dit_kernel
    assert dit.blocks[0].attn.attn_backend == "torch"
    aggregator_kernel.assert_not_called()
    dit_kernel.assert_not_called()


@pytest.mark.parametrize("pe_attn_head", [None, 2])
def test_acoustic_component_forwards_reach_joint_rotary(
    monkeypatch: pytest.MonkeyPatch,
    pe_attn_head: int | None,
) -> None:
    calls: list[tuple[torch.Size, torch.Size, list[int], bool]] = []
    attention_calls: list[torch.Size] = []
    attention = torch.nn.functional.scaled_dot_product_attention

    def attention_after_rotary(query, key, value, *args, **kwargs):
        assert torch.all(query == 3.0)
        assert torch.all(key == 4.0)
        attention_calls.append(query.shape)
        return attention(query, key, value, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", attention_after_rotary
    )

    def fake_apply_rope_inplace(
        query_tokens,
        key_tokens,
        cos_sin_cache,
        token_positions,
        *,
        is_neox,
    ) -> None:
        assert query_tokens.device.type == "cpu"
        assert key_tokens.device.type == "cpu"
        assert query_tokens.shape == key_tokens.shape
        assert cos_sin_cache.dtype == torch.float32
        calls.append(
            (
                query_tokens.shape,
                cos_sin_cache.shape,
                token_positions.tolist(),
                is_neox,
            )
        )
        query_tokens.fill_(3.0)
        key_tokens.fill_(4.0)

    aggregator = Aggregator(
        in_channels=4,
        hidden_size=8,
        depth=2,
        num_heads=2,
        llm_input_dim=6,
        pe_attn_head=pe_attn_head,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch",
            rope_kernel=fake_apply_rope_inplace,
            rope_seq_len=3,
            rope_max_batch_size=4,
        ),
    )
    dit = DiT(
        in_channels=4,
        hidden_size=8,
        depth=2,
        num_heads=2,
        llm_cond_dim=6,
        cfg_dropout_prob=0,
        pe_attn_head=pe_attn_head,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch",
            rope_kernel=fake_apply_rope_inplace,
            rope_seq_len=6,
            rope_max_batch_size=4,
        ),
    )

    with torch.no_grad():
        aggregator_output = aggregator(torch.randn(2, 2, 4))
        reference_output = aggregator(torch.randn(4, 2, 4))
        dit_output = dit(
            torch.randn(2, 2, 4),
            torch.rand(2),
            torch.randn(2, 1, 6),
            torch.randn(2, 3, 4),
        )
        cfg_output = dit.forward_with_cfg(
            torch.randn(2, 2, 4),
            torch.tensor(0.5),
            torch.randn(2, 1, 6),
            torch.randn(2, 3, 4),
        )

    assert aggregator_output.shape == (2, 1, 6)
    assert reference_output.shape == (4, 1, 6)
    assert dit_output.shape == (2, 6, 4)
    assert cfg_output.shape == (4, 2, 4)
    assert calls == [
        ((batch * seq_len, 2, 4), (seq_len, 4), list(range(seq_len)) * batch, False)
        for batch, seq_len in ((2, 3), (4, 3), (2, 6), (4, 6))
        for _layer in range(2)
    ]
    assert attention_calls == [
        (batch, 2, seq_len, 4)
        for batch, seq_len in ((2, 3), (4, 3), (2, 6), (4, 6))
        for _layer in range(2)
    ]
