# SPDX-License-Identifier: Apache-2.0
"""Numerical parity tests for ARK-ASR audio projection fusion."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from transformers import WhisperConfig

import sglang_omni.models.arkasr.audio_tower as audio_tower
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.sglang_model import ArkasrForConditionalGeneration


def _tiny_config(*, use_rope: bool) -> ArkasrConfig:
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=64,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=4,
        use_rope=use_rope,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


class _UnfusedWhisperRoPESdpaAttention(nn.Module):
    """Reference implementation matching the checkpoint's separate projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.is_causal = False

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return (
            states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        **_: object,
    ):
        query = self._shape(self.q_proj(hidden_states))
        key = self._shape(self.k_proj(hidden_states))
        value = self._shape(self.v_proj(hidden_states))
        if rotary_pos_emb is not None:
            query = audio_tower.apply_rotary_pos_emb(query, rotary_pos_emb)
            key = audio_tower.apply_rotary_pos_emb(key, rotary_pos_emb)
        target_dtype = self.q_proj.weight.dtype
        output = scaled_dot_product_attention(
            query.to(target_dtype),
            key.to(target_dtype),
            value.to(target_dtype),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .reshape(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.embed_dim,
            )
        )
        return self.out_proj(output), None, None


def _build_reference_and_candidate(
    monkeypatch,
    *,
    use_rope: bool,
    load_packed_attention: bool = True,
) -> tuple[nn.Module, nn.Module]:
    config = _tiny_config(use_rope=use_rope)
    with monkeypatch.context() as context:
        context.setattr(
            audio_tower,
            "WhisperRoPESdpaAttention",
            _UnfusedWhisperRoPESdpaAttention,
        )
        torch.manual_seed(7)
        reference = audio_tower.ArkAudioMLPAdapter(config).eval()

    torch.manual_seed(11)
    candidate = audio_tower.ArkAudioMLPAdapter(config).eval()

    candidate_state = candidate.state_dict()
    shared_state = {
        name: tensor
        for name, tensor in reference.state_dict().items()
        if name in candidate_state and candidate_state[name].shape == tensor.shape
    }
    candidate.load_state_dict(shared_state, strict=False)

    if not load_packed_attention:
        return reference, candidate

    for reference_layer, candidate_layer in zip(
        reference.whisper.layers,
        candidate.whisper.layers,
    ):
        reference_attention = reference_layer.self_attn
        candidate_attention = candidate_layer.self_attn
        packed_weight = candidate_attention.qkv_proj.weight
        packed_weight.weight_loader(
            packed_weight,
            reference_attention.q_proj.weight,
            "q",
        )
        packed_weight.weight_loader(
            packed_weight,
            reference_attention.k_proj.weight,
            "k",
        )
        packed_weight.weight_loader(
            packed_weight,
            reference_attention.v_proj.weight,
            "v",
        )

        packed_bias = candidate_attention.qkv_proj.bias
        packed_bias.weight_loader(
            packed_bias,
            reference_attention.q_proj.bias,
            "q",
        )
        packed_bias.weight_loader(
            packed_bias,
            torch.zeros_like(reference_attention.q_proj.bias),
            "k",
        )
        packed_bias.weight_loader(
            packed_bias,
            reference_attention.v_proj.bias,
            "v",
        )

    return reference, candidate


def test_audio_adapter_exposes_one_packed_qkv_parameter(monkeypatch) -> None:
    _, candidate = _build_reference_and_candidate(
        monkeypatch,
        use_rope=True,
    )
    parameter_names = set(dict(candidate.named_parameters()))

    for layer_index in range(2):
        prefix = f"whisper.layers.{layer_index}.self_attn"
        assert f"{prefix}.qkv_proj.weight" in parameter_names
        assert f"{prefix}.q_proj.weight" not in parameter_names
        assert f"{prefix}.k_proj.weight" not in parameter_names
        assert f"{prefix}.v_proj.weight" not in parameter_names


@torch.no_grad()
@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 3e-2, 3e-2),
        (torch.float16, 3e-2, 3e-2),
    ],
)
def test_fused_audio_adapter_matches_unfused_reference(
    monkeypatch,
    use_rope: bool,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    reference, candidate = _build_reference_and_candidate(
        monkeypatch,
        use_rope=use_rope,
    )
    reference.to(dtype)
    candidate.to(dtype)
    torch.manual_seed(23)
    mel = torch.randn(2, 8, 40, dtype=dtype)
    attention_mask = torch.tensor(
        [
            [1] * 33 + [0] * 7,
            [1] * 40,
        ],
        dtype=torch.bool,
    )

    expected = reference(mel, attention_mask=attention_mask)
    actual = candidate(mel, attention_mask=attention_mask)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_separate_audio_checkpoint_weights_load_into_fused_adapter(
    monkeypatch,
) -> None:
    reference, candidate = _build_reference_and_candidate(
        monkeypatch,
        use_rope=True,
        load_packed_attention=False,
    )
    model = ArkasrForConditionalGeneration.__new__(ArkasrForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = _tiny_config(use_rope=True)
    model.audio_encoder = candidate

    model.load_weights(
        (
            f"audio_encoder.{name}",
            parameter.detach().clone(),
        )
        for name, parameter in reference.named_parameters()
    )

    packed_bias = candidate.whisper.layers[0].self_attn.qkv_proj.bias
    embed_dim = model.config.whisper_config.d_model
    torch.testing.assert_close(
        packed_bias[embed_dim : 2 * embed_dim],
        torch.zeros(embed_dim),
        atol=0,
        rtol=0,
    )

    torch.manual_seed(31)
    mel = torch.randn(2, 8, 40)
    attention_mask = torch.tensor(
        [
            [1] * 29 + [0] * 11,
            [1] * 40,
        ],
        dtype=torch.bool,
    )
    expected = reference(mel, attention_mask=attention_mask)
    actual = candidate(mel, attention_mask=attention_mask)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
