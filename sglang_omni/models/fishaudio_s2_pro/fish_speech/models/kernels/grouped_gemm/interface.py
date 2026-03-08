import torch

from .backward_dw import grouped_gemm_backward_dw
from .forward import grouped_gemm_forward
from .forward_transposed import grouped_gemm_forward_transposed


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m_sizes):
        ctx.save_for_backward(x, w, m_sizes)
        return grouped_gemm_forward(x, w, m_sizes)

    @staticmethod
    def backward(ctx, dy):
        x, w, m_sizes = ctx.saved_tensors

        if x.requires_grad:
            dx = grouped_gemm_forward_transposed(dy, w, m_sizes, x.dtype)
        else:
            dx = None

        if w.requires_grad:
            dw = grouped_gemm_backward_dw(x, dy, m_sizes, w.dtype)
        else:
            dw = None

        return dx, dw, None


grouped_gemm = GroupedGemm.apply
