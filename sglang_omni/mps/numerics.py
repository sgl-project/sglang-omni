# SPDX-License-Identifier: Apache-2.0
"""Pre-model numerical gate for CUDA MPS static partitions."""

from __future__ import annotations

import math


def validate_sm_partition(sm_cap: int) -> None:
    """Fail startup on a wrong partition or unsafe BF16 GEMM kernel selection.

    The caller runs this once per capped OS process, before any model factory.
    Reference inputs are rounded to BF16 before the CPU FP32 multiplication;
    the gate measures relative Frobenius error, not elementwise relative error.
    """
    import torch

    torch.cuda.init()
    actual_sm = torch.cuda.get_device_properties(0).multi_processor_count
    if actual_sm != sm_cap:
        raise RuntimeError(
            f"Static partition requested {sm_cap} SM but CUDA reported {actual_sm} SM"
        )

    with torch.inference_mode():
        generator = torch.Generator(device="cpu").manual_seed(0)
        a = torch.randn(
            512, 1024, generator=generator, device="cpu", dtype=torch.float32
        ).bfloat16()
        b = torch.randn(
            1024, 1024, generator=generator, device="cpu", dtype=torch.float32
        ).bfloat16()
        # Note (Jiaxin Deng): a CPU reference avoids the same cuBLAS kernel
        # selection defect or TF32 policy contaminating both sides of the gate.
        reference = a.float() @ b.float()
        result = torch.mm(a.to("cuda"), b.to("cuda")).float().cpu()
        relative_error = ((result - reference).norm() / reference.norm()).item()
    if not math.isfinite(relative_error) or relative_error > 1e-2:
        raise RuntimeError(
            f"Static partition with {actual_sm} SM is unsafe for this cuBLAS/PyTorch "
            f"build: BF16 GEMM relative error {relative_error:.6g} exceeds 0.01. "
            "Choose a different sm_cap value; kernel selection depends on the reported SM count"
        )
