import torch

from .grouped_gemm.interface import grouped_gemm


def triton_grouped_gemm(
    input: torch.Tensor, weight: torch.Tensor, m_sizes: torch.Tensor
) -> torch.Tensor:
    """
    Computes a MoE linear operation using grouped GEMM (Triton kernel).

    The operation is defined as:
    `output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]`

    Args:
        input (`torch.FloatTensor`): input tensor of shape `(batch_size, in_features)`.
        weight (`torch.FloatTensor`): weight tensor of shape `(num_experts, out_features, in_features)`.
        m_sizes (`torch.LongTensor`): counts of selected experts in shape `(num_experts)`. Should sum to `batch_size`.

    Returns:
        output (`torch.FloatTensor`): output tensor of shape `(batch_size, out_features)`.
    """
    return grouped_gemm(input, weight, m_sizes)
