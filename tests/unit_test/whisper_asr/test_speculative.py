# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr.sglang_model import WhisperEncoder

speculative = pytest.importorskip("sglang_omni.models.whisper_asr.speculative")


def tiny_config(**overrides) -> WhisperConfig:
    return WhisperConfig(
        d_model=8,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=32,
        max_target_positions=16,
        max_source_positions=8,
        pad_token_id=0,
        **overrides,
    )


def test_materialized_draft_names_the_decoder_only_class(tmp_path) -> None:
    source = tmp_path / "draft"
    source.mkdir()
    tiny_config().to_json_file(source / "config.json")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    (source / "preprocessor_config.json").write_text("{}")

    checkpoint = speculative.materialize_draft_checkpoint(str(source), None)
    with open(os.path.join(checkpoint.name, "config.json")) as config_file:
        config = json.load(config_file)

    # The second name keeps SGLang's encoder-decoder handling for the draft.
    assert config["architectures"] == [
        speculative.DRAFT_ARCHITECTURE,
        "WhisperForConditionalGeneration",
    ]
    assert config["encoder_layers"] == 0
    assert sorted(os.listdir(checkpoint.name)) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    checkpoint.cleanup()


def test_materialize_rejects_a_draft_that_is_not_whisper(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ValueError, match="not a Whisper checkpoint"):
        speculative.materialize_draft_checkpoint(str(tmp_path), None)


def test_draft_loads_decoder_weights_and_skips_the_encoder(monkeypatch) -> None:
    monkeypatch.setattr(speculative, "LogitsProcessor", lambda config: None)
    config = tiny_config()
    draft = speculative.WhisperDraftForCausalLM(config)
    assert not hasattr(draft.model, "encoder")
    weight = torch.full((config.d_model, config.d_model), 0.5)
    draft.load_weights(
        [
            ("model.encoder.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
            ("model.decoder.layers.0.self_attn.k_proj.weight", weight),
            ("model.decoder.layers.0.encoder_attn.v_proj.weight", weight),
        ]
    )
    layer = draft.model.decoder.layers[0]
    d_model = config.d_model
    assert torch.equal(layer.self_attn.qkv_proj.weight[d_model : 2 * d_model], weight)
    assert torch.equal(layer.encoder_attn.kv_proj.weight[d_model:], weight)
    with pytest.raises(RuntimeError, match="no encoder of its own"):
        draft.run_encoder(torch.zeros(1))


class RecordingTranslator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def pack_read_stream(self, plan, **kwargs) -> None:
        del plan
        self.calls.append(kwargs)


def chain_of_three() -> speculative.WhisperVerifyInput:
    """Two requests with five encoder slots each, verifying three tokens each."""
    return speculative.WhisperVerifyInput(
        draft_token=torch.zeros(6, dtype=torch.long),
        positions=torch.zeros(6, dtype=torch.long),
        draft_token_num=3,
        custom_mask=None,
        encoder_token_count=10,
    )


def test_verify_cross_attention_reads_only_the_encoder_slots() -> None:
    encoder_lens = torch.tensor([5, 5])
    translator = RecordingTranslator()

    kv_indices, kv_indptr, qo_indptr, mask = chain_of_three().generate_attn_arg_prefill(
        req_pool_indices=torch.tensor([0, 1]),
        paged_kernel_lens=encoder_lens,
        paged_kernel_lens_sum=10,
        translator=translator,
        plan=None,
        kv_start_idx=torch.zeros_like(encoder_lens),
    )

    assert kv_indptr.tolist() == [0, 5, 10]
    assert qo_indptr.tolist() == [0, 3, 6]
    assert kv_indices.numel() == 10 and mask is None
    assert translator.calls[0]["seq_lens"] is encoder_lens


def test_verify_self_attention_widens_by_the_chain() -> None:
    translator = RecordingTranslator()

    _, kv_indptr, _, _ = chain_of_three().generate_attn_arg_prefill(
        req_pool_indices=torch.tensor([0, 1]),
        paged_kernel_lens=torch.tensor([4, 5]),
        paged_kernel_lens_sum=9,
        translator=translator,
        plan=None,
        kv_start_idx=torch.tensor([5, 5]),
    )

    assert kv_indptr.tolist() == [0, 7, 15]
    assert translator.calls[0]["kv_start_idx"].tolist() == [5, 5]


def tiny_target() -> nn.Module:
    target = nn.Module()
    target.model = nn.Module()
    target.model.encoder = WhisperEncoder(tiny_config())
    return target


def test_shared_encoder_accepts_the_same_weights_in_another_half_precision(
    tmp_path,
) -> None:
    target = tiny_target()
    save_file(
        {
            name: dict(target.named_parameters())[name].detach().to(torch.bfloat16)
            for name in speculative.SHARED_ENCODER_TENSOR_NAMES
        },
        str(tmp_path / "model.safetensors"),
    )

    speculative.check_shared_encoder(target, str(tmp_path))


@pytest.mark.parametrize(
    ("draft_names", "shift", "message"),
    [
        (speculative.SHARED_ENCODER_TENSOR_NAMES, 0.5, "differs from the target's"),
        (speculative.SHARED_ENCODER_TENSOR_NAMES[:1], 0.0, "cannot be checked"),
    ],
)
def test_shared_encoder_refuses_a_different_or_unknown_encoder(
    tmp_path, draft_names: tuple[str, ...], shift: float, message: str
) -> None:
    """A draft paired with another encoder reads states it was never trained on."""
    target = tiny_target()
    save_file(
        {
            name: dict(target.named_parameters())[name].detach() + shift
            for name in draft_names
        },
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(ValueError, match=message):
        speculative.check_shared_encoder(target, str(tmp_path))
