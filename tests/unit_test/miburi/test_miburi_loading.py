# SPDX-License-Identifier: Apache-2.0
"""Release-layout checkpoints map onto the native modules: every tensor loaded or named unused."""

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.miburi.architecture import GestureLMSpec, MotionCodecSpec
from sglang_omni.models.miburi.components.gesture_model import GestureModel
from sglang_omni.models.miburi.components.motion_codec import MotionCodec
from sglang_omni.models.miburi.loading import (
    gesture_lm_state,
    motion_codec_state,
    read_checkpoint,
)

LM_SPEC = GestureLMSpec(
    codebooks_per_part=(2, 2, 1),
    codebook_size=16,
    dim=8,
    codebook_dim=4,
    condition_dim=8,
    text_vocab=8,
    audio_vocab=8,
    ffn_dim=8,
    temporal_layers=1,
    depth_layers=1,
    num_speakers=2,
)
CODEC_SPEC = MotionCodecSpec(
    features=6,
    codebooks=2,
    num_layers=1,
    num_heads=2,
    dim=8,
    codebook_dim=4,
    codebook_size=16,
    ffn_dim=16,
    n_filters=4,
    residual_blocks=1,
)


def save_release(tmp_path, tensors: dict[str, torch.Tensor]):
    path = tmp_path / "last.safetensors"
    save_file(
        {f"module.{key}": tensor.contiguous() for key, tensor in tensors.items()},
        str(path),
    )
    return path


# GestureModel's key fragments → the release checkpoint's.
RELEASE_LM_KEYS = (
    ("text_embedding.", "text_procemb."),
    ("audio_embeddings.", "audio_procemb."),
    ("temporal_condition_projs.", "temp_condproj."),
    ("depth_condition_projs.", "dep_condproj."),
    ("temporal_speaker_embedding.", "spk_tempemb."),
    ("depth_speaker_embeddings.", "spk_depemb."),
    ("temporal_code_embeddings.", "temporal_gemb."),
    ("depth_code_embeddings.", "depformer_gemb."),
    ("temporal_code_projs.", "temporal_gproj."),
    ("depth_code_projs.", "depformer_gproj."),
    ("depth_hidden_projs.", "depformer_in."),
    ("temporal_out_norm.", "out_norm."),
    ("depth_classifiers.", "depformer_classifier."),
    (".query_projs.", ".in_projs."),
    (".value_projs.", ".val_projs."),
    (".cross_attn_norms.", ".norm3."),
    (".memory_fusion.", ".memory_projs.0."),
    (".cross_attn_feed_forward.", ".ca_gating."),
    (".feed_forward.", ".gating."),
)


def release_lm_key(key: str) -> str:
    for native, release in RELEASE_LM_KEYS:
        key = key.replace(native, release)
    return key


def release_lm_checkpoint(source: GestureModel) -> dict[str, torch.Tensor]:
    """source's tensors in bfloat16 under the release keys, beside the tensors serving never reads and the dead temporal tables of the tied codebooks."""
    checkpoint = {
        release_lm_key(key): tensor.to(torch.bfloat16)
        for key, tensor in source.state_dict().items()
    }
    unused = [
        "vad_predictor.0.weight",
        "gesture_codec_layers.0._codebook.embedding_sum",
    ]
    unused += [f"depth_transformer.layers.0.norm3.{memory}.weight" for memory in (0, 1)]
    unused += [
        f"depth_transformer.layers.0.cross_attns.{memory}.{name}.{step}.weight"
        for memory in (0, 1)
        for name in ("in_projs", "key_projs")
        for step in range(LM_SPEC.depth_steps)
    ]
    for key in unused:
        checkpoint[key] = torch.randn(3).to(torch.bfloat16)
    for codebook in range(LM_SPEC.depth_steps):
        checkpoint[f"temporal_gemb.{codebook}.weight"] = torch.randn(
            LM_SPEC.codebook_size, LM_SPEC.codebook_dim
        ).to(torch.bfloat16)
    return checkpoint


def test_the_temporal_path_reads_the_depth_tables_and_conditions_stay_bfloat16(
    tmp_path,
):
    torch.manual_seed(0)
    checkpoint = release_lm_checkpoint(GestureModel(LM_SPEC))
    model = GestureModel(LM_SPEC)

    model.load_state_dict(
        gesture_lm_state(read_checkpoint(save_release(tmp_path, checkpoint)), LM_SPEC)
    )

    for codebook in range(LM_SPEC.depth_steps):
        table = model.depth_code_embeddings[codebook].weight
        assert model.temporal_code_embeddings[codebook].weight is table
        assert torch.equal(
            table, checkpoint[f"depformer_gemb.{codebook}.weight"].float()
        )
    last = LM_SPEC.depth_steps
    assert torch.equal(
        model.temporal_code_embeddings[last].weight,
        checkpoint[f"temporal_gemb.{last}.weight"].float(),
    )
    conditions = [
        model.text_embedding.weight,
        *(table.weight for table in model.audio_embeddings),
    ]
    assert {table.dtype for table in conditions} == {torch.bfloat16}
    assert {
        tensor.dtype
        for tensor in model.parameters()
        if all(tensor is not table for table in conditions)
    } == {torch.float32}


def test_a_tensor_the_loader_does_not_know_is_refused(tmp_path):
    checkpoint = release_lm_checkpoint(GestureModel(LM_SPEC))
    checkpoint["vad_head.weight"] = torch.zeros(1)

    with pytest.raises(RuntimeError, match="vad_head.weight"):
        GestureModel(LM_SPEC).load_state_dict(
            gesture_lm_state(
                read_checkpoint(save_release(tmp_path, checkpoint)), LM_SPEC
            )
        )


def release_codec_key(key: str) -> str:
    """MotionCodec's key → the release checkpoint's."""
    if key.startswith("transformer."):
        return f"transformer_decoder.{key}".replace(
            ".in_proj.", ".in_projs.0."
        ).replace(".out_proj.", ".out_projs.0.")
    elif key.startswith("decoder."):
        return f"seanet_decoder.model.{key.removeprefix('decoder.')}".replace(
            ".conv.", ".conv.conv."
        ).replace(".convtr.", ".convtr.convtr.")
    else:
        return key.replace(".codebook.", "._codebook.")


def test_codec_weight_norms_fold_over_all_but_the_first_axis_and_the_encoder_is_dropped(
    tmp_path,
):
    torch.manual_seed(0)
    source = MotionCodec(CODEC_SPEC)
    checkpoint = {
        "seanet_encoder.model.0.conv.conv.weight_v": torch.randn(4, 6, 7),
        "transformer_encoder.transformer.layers.0.linear1.weight": torch.randn(2, 2),
        "quantizer.input_proj.weight": torch.randn(4, 8, 1),
    }
    folded = {}
    for key, tensor in source.state_dict().items():
        release = release_codec_key(key)
        if release.startswith("seanet_decoder.") and key.endswith(".weight"):
            stem = release.removesuffix(".weight")
            direction = torch.randn_like(tensor)
            magnitude = torch.rand(tensor.shape[0], *[1] * (tensor.dim() - 1)) + 0.5
            checkpoint[f"{stem}.weight_v"] = direction
            checkpoint[f"{stem}.weight_g"] = magnitude
            folded[key] = (
                magnitude
                * direction
                / direction.norm(dim=tuple(range(1, tensor.dim())), keepdim=True)
            )
        else:
            checkpoint[release] = torch.randn_like(tensor)
    codec = MotionCodec(CODEC_SPEC)

    codec.load_state_dict(
        motion_codec_state(read_checkpoint(save_release(tmp_path, checkpoint)))
    )

    state = codec.state_dict()
    assert sorted(folded) == [
        "decoder.0.conv.weight",
        "decoder.2.convtr.weight",
        "decoder.3.block.1.conv.weight",
        "decoder.3.block.3.conv.weight",
        "decoder.5.conv.weight",
    ]
    for key, expected in folded.items():
        torch.testing.assert_close(state[key], expected)
    for key, tensor in state.items():
        if key not in folded:
            assert torch.equal(tensor, checkpoint[release_codec_key(key)])
        else:
            pass
