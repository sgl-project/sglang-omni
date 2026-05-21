# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

_THINKER = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "ming_omni"
    / "thinker.py"
)


def _source() -> str:
    return _THINKER.read_text(encoding="utf-8")


def _section(src: str, start: str, end: str) -> str:
    start_idx = src.index(start)
    return src[start_idx : src.index(end, start_idx)]


def test_ming_attention_and_layer_boundary_tp_pattern():
    src = _source()
    attention_src = _section(
        src, "class BailingMoeV2Attention", "class BailingMoeV2MLP"
    )
    decoder_src = _section(
        src,
        "class BailingMoeV2DecoderLayer",
        "class BailingMoeV2TextModel",
    )

    assert 'prefix="layers"' in src
    assert "get_attention_tp_rank" in src
    assert "get_attention_tp_size" in src
    assert "validate_attention_tp_config" in src
    assert "attn_tp_rank = get_attention_tp_rank()" in src
    assert "attn_tp_size = get_attention_tp_size()" in src
    assert "tp_rank=attn_tp_rank" in src
    assert "tp_size=attn_tp_size" in src
    assert "reduce_results=False" in attention_src

    assert "LayerCommunicator" in src
    assert "LayerScatterModes" in src
    assert "prepare_attn_and_capture_last_layer_outputs" in decoder_src
    assert "prepare_mlp" in decoder_src
    assert "should_fuse_mlp_allreduce_with_next_layer" in decoder_src
    assert "should_use_reduce_scatter" in decoder_src
    assert "postprocess_layer" in decoder_src
    assert "_sglang_needs_allreduce_fusion" in decoder_src
    assert "allow_reduce_scatter=True" in decoder_src
    assert "should_allreduce_fusion = False" not in decoder_src
    assert "use_reduce_scatter = False" not in decoder_src


def test_ming_mlp_and_weight_loader_tp_pattern():
    src = _source()
    mlp_src = _section(src, "class BailingMoeV2MLP", "class BailingMoeV2SparseMoeBlock")
    load_src = _section(src, "    def load_weights", "# ForCausalLM Wrapper")

    assert "MergedColumnParallelLinear" in mlp_src
    assert "RowParallelLinear" in mlp_src
    assert "SiluAndMul" in mlp_src
    assert "ReplicatedLinear(" not in mlp_src
    assert "reduce_results=reduce_results" in mlp_src
    assert "forward_batch: Optional[ForwardBatch] = None" in mlp_src
    assert "should_allreduce_fusion: bool = False" in mlp_src
    assert "use_reduce_scatter: bool = False" in mlp_src
    assert "skip_all_reduce=should_allreduce_fusion or use_reduce_scatter" in mlp_src

    assert '(".gate_proj.", 0)' in load_src
    assert '(".up_proj.", 1)' in load_src
    assert "param.weight_loader(param, loaded_weight, shard_id)" in load_src
    assert "default_weight_loader(param, fused)" not in load_src
    assert "torch.cat([buf" not in load_src
    assert "_unmatched_weight_names" in src
    assert "_loaded_weight_count" in src
    assert "_skipped_weight_count" in src
    assert "Incomplete Ming gate/up fused weights" in src
    assert "logger.info(" in src


def test_ming_moe_unified_reduction_pattern():
    src = _source()
    moe_src = _section(
        src,
        "class BailingMoeV2SparseMoeBlock",
        "class BailingMoeV2DecoderLayer",
    )

    assert "self.tp_size = get_tensor_model_parallel_world_size()" in moe_src
    assert "reduce_results=False" in moe_src
    assert "final_hidden_states = routed_output + shared_output" in moe_src
    assert "tensor_model_parallel_all_reduce(final_hidden_states)" in moe_src
    assert "should_allreduce_fusion" in moe_src
    assert "use_reduce_scatter" in moe_src
    assert "if should_use_flashinfer_cutlass_moe_fp4_allgather():" in moe_src
    assert "shared_tp_rank, shared_tp_size = 0, 1" in moe_src
    assert "shared_tp_rank, shared_tp_size = None, None" in moe_src
    assert "tp_rank=shared_tp_rank" in moe_src
    assert "tp_size=shared_tp_size" in moe_src


def test_ming_dense_fully_dp_pattern():
    src = _source()
    mlp_src = _section(src, "class BailingMoeV2MLP", "class BailingMoeV2SparseMoeBlock")
    decoder_src = _section(
        src,
        "class BailingMoeV2DecoderLayer",
        "class BailingMoeV2TextModel",
    )

    assert "enable_moe_dense_fully_dp" in src
    assert "if enable_moe_dense_fully_dp():" in decoder_src
    assert "mlp_tp_rank, mlp_tp_size = 0, 1" in decoder_src
    assert "mlp_tp_rank, mlp_tp_size = None, None" in decoder_src
    assert "tp_rank=mlp_tp_rank" in decoder_src
    assert "tp_size=mlp_tp_size" in decoder_src
    assert "self.tp_size = tp_size" in mlp_src
    assert "if (self.tp_size == 1) and hidden_states.shape[0] == 0:" in mlp_src
    assert "return hidden_states" in mlp_src
