"""W8A8 FP8 lm_head for Qwen3-ASR on SM90 through torch._scaled_mm (cuBLASLt).

note (jvzibro): the SM120 path keeps its CuTe kernels; SM90 has no block-scaled
FP8 MMA, but Hopper runs rowwise-scaled FP8 GEMMs natively in cuBLASLt. The
vocab projection is bandwidth-bound and beats bf16 at every measured batch
size on both SM90 cards tried (H20: 1.5-2.0x at M=1..256; H200: 1.1-1.5x),
while the decoder linears are machine-dependent (gate_up_proj won at prefill
sizes on the H20 but lost at every size on the H200), so only lm_head converts.

Activations quantize per token (rowwise), matching the SM120 W8A8 semantics:
under CUDA graph replay the padded rows of a bucket hold garbage, and a
per-tensor amax would let one padded row poison every real row's scale.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FP8_MAX = 448.0
# cuBLASLt FP8 wants 16-byte-aligned operands: K for the contraction, N for the column-major B.
SCALED_MM_ALIGNMENT = 16
# note (jvzibro): quantize in row chunks so the fp32 cast of a [151936, 2048]
# vocab table never materializes (its full-size cast is a 1.2 GiB transient).
QUANTIZE_CHUNK_ROWS = 8192


def is_scaled_mm_supported(weight: torch.Tensor) -> bool:
    """True when _scaled_mm can serve this weight: bf16, and K, N tile-align."""
    N, K = weight.shape
    if weight.dtype != torch.bfloat16:
        return False
    elif K % SCALED_MM_ALIGNMENT != 0 or N % SCALED_MM_ALIGNMENT != 0:
        return False
    else:
        pass
    return True


def quantize_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """bf16 [N, K] -> fp8 [N, K], fp32 [1, N] column scale and the fp32 scalar."""
    N, K = weight.shape
    amax = 0.0
    for start in range(0, N, QUANTIZE_CHUNK_ROWS):
        chunk = weight[start : start + QUANTIZE_CHUNK_ROWS]
        amax = max(amax, chunk.float().abs().max().item())
    scale = max(amax, 1e-12) / FP8_MAX
    weight_fp8 = torch.empty(N, K, device=weight.device, dtype=torch.float8_e4m3fn)
    for start in range(0, N, QUANTIZE_CHUNK_ROWS):
        chunk = weight[start : start + QUANTIZE_CHUNK_ROWS]
        weight_fp8[start : start + QUANTIZE_CHUNK_ROWS] = (chunk.float() / scale).to(
            torch.float8_e4m3fn
        )
    weight_scale_col = torch.full(
        (1, N), scale, device=weight.device, dtype=torch.float32
    )
    return weight_fp8, weight_scale_col, scale


def quantize_activation_rows(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> fp8 [M, K] and fp32 [M, 1] per-token scales."""
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


def scaled_mm_w8a8(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_col: torch.Tensor,
) -> torch.Tensor:
    """bf16 X [M, K] @ fp8 W [N, K]^T with rowwise scales -> bf16 Y [M, N]."""
    M = x.shape[0]
    if M == 0:
        N = weight_fp8.shape[0]
        return torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    else:
        pass
    x_fp8, x_scale = quantize_activation_rows(x)
    # weight_fp8 is [N, K] row-major, so .t() is the column-major [K, N] B operand.
    return torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        scale_a=x_scale,
        scale_b=weight_scale_col,
        out_dtype=torch.bfloat16,
    )


class ScaledMmFp8LinearMethod:
    """quant_method for a module holding weight (fp8) and weight_scale_col."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # note (jvzibro): the loader calls this on every quant_method, and the
        # weights were already quantized when the module was swapped.
        pass

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        y = scaled_mm_w8a8(x_2d, layer.weight, layer.weight_scale_col)
        if bias is not None:
            y = y + bias
        else:
            pass
        return y.reshape(*leading_shape, y.shape[-1])


def set_scaled_mm_fp8_weight(module: nn.Module, weight: torch.Tensor) -> None:
    assert is_scaled_mm_supported(
        weight
    ), f"no _scaled_mm FP8 kernel for {weight.dtype} {tuple(weight.shape)}"
    weight_fp8, weight_scale_col, scale = quantize_weight(weight)
    module.register_buffer("weight", weight_fp8, persistent=False)
    module.register_buffer("weight_scale_col", weight_scale_col, persistent=False)
    module.weight_scale = scale
    module.quant_method = ScaledMmFp8LinearMethod()


class ScaledMmFp8LMHead(nn.Module):
    """FP8 copy of the tied lm_head. The embedding table itself stays bf16."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        set_scaled_mm_fp8_weight(self, weight.detach())


def install_fp8_scaled_mm_lm_head(language_model: nn.Module) -> bool:
    """Swap the lm_head to _scaled_mm FP8. Returns whether it converted."""
    # note (jvzibro): only the vocab projection converts. It wins at every
    # measured batch size on both SM90 cards tried (H20 and H200), while the
    # decoder linears are machine-dependent: gate_up_proj won at prefill sizes
    # on the H20 but lost at every size on the H200, so they stay bf16.
    if not is_scaled_mm_supported(language_model.lm_head.weight):
        return False
    else:
        pass
    language_model.lm_head = ScaledMmFp8LMHead(language_model.lm_head.weight)
    repack_on_device(language_model)
    return True


def repack_on_device(module: nn.Module) -> None:
    """Round-trip every CUDA tensor of module through the host into fresh device memory.

    note (jvzibro): the FP8 weights land in the allocator segments of the freed
    bf16 weights, so empty_cache can not return those segments. After the round
    trip the KV cache gets that memory. Same trick as the SM120 module, which
    can not be imported here without pulling in the CuTe DSL.
    """
    tensors = []
    seen_ids = set()
    candidates = list(module.parameters()) + list(module.buffers())
    for tensor in candidates:
        if not tensor.is_cuda or id(tensor) in seen_ids:
            continue
        else:
            pass
        seen_ids.add(id(tensor))
        tensors.append(tensor)

    host_copies = []
    for tensor in tensors:
        host_copies.append(tensor.data.to("cpu"))
        tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    torch.cuda.empty_cache()

    for tensor, host_copy in zip(tensors, host_copies):
        tensor.data = host_copy.to(tensor.device)
