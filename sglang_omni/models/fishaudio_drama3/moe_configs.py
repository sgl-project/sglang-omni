# SPDX-License-Identifier: Apache-2.0
"""Tuned fused-MoE Triton configs for the Drama 3 experts on H100.

sglang has no config for this expert shape, so its kernel runs one 64-row tile
that is mostly padding at Drama 3 token counts. The configs choose tiling only.
"""

from __future__ import annotations

import logging

import triton
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

logger = logging.getLogger(__name__)

DEVICE_NAME = "NVIDIA H100 80GB HBM3"
TRITON_VERSION = "3.8.0"
# note (Wenyao Gao): (E, N, dtype, block_n, block_k) as sglang's get_moe_configs
# receives them.
EXPERT_SHAPE = (256, 512, "fp8_w8a8", 128, 128)

# note (Wenyao Gao): sglang's get_default_config for this shape, which does not
# depend on M; token counts that were not tuned keep it.
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}
DEFAULT_TOKEN_COUNTS = (24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 4096)


def tuned_tile(block_size_m: int, group_size_m: int) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": 4,
        "num_stages": 3,
    }


# note (Wenyao Gao): tuned with sglang's fused_moe_triton tuning script on one H100
# 80GB HBM3 under Triton 3.8.0; keys are token counts M. Decode runs one token per
# running request (up to 64 at concurrency 64); a prefill runs every prompt token of
# its batch (about 200 at concurrency 16, 500 to 1,000 at concurrency 64).
# BLOCK_SIZE_K stays 128 at every count, as in the default tile.
TUNED_CONFIGS: dict[int, dict[str, int]] = {
    8: tuned_tile(16, 64),
    16: tuned_tile(16, 64),
    24: tuned_tile(16, 16),
    32: tuned_tile(16, 16),
    40: tuned_tile(16, 16),
    48: tuned_tile(16, 32),
    56: tuned_tile(16, 16),
    64: tuned_tile(16, 32),
    128: tuned_tile(16, 1),
    256: tuned_tile(16, 1),
    512: tuned_tile(32, 16),
    1024: tuned_tile(64, 32),
}

# note (Wenyao Gao): sglang serves the key nearest to M and breaks a tie by key
# order, so the default counts come first even where a tuned tile replaces them.
CONFIGS = {
    **{token_count: DEFAULT_CONFIG for token_count in DEFAULT_TOKEN_COUNTS},
    **TUNED_CONFIGS,
}


SGLANG_GET_MOE_CONFIGS = fused_moe_triton_config.get_moe_configs


def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = 0,
    block_k: int | None = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> dict[int, dict[str, int]] | None:
    """sglang's get_moe_configs, with CONFIGS for the Drama 3 expert shape."""
    if (
        (E, N, dtype, block_n, block_k) == EXPERT_SHAPE
        and not per_channel_quant
        and not down_moe
        and not fused_moe_triton_config.get_exec().deterministic.enable_deterministic_inference
    ):
        return CONFIGS
    else:
        return SGLANG_GET_MOE_CONFIGS(
            E,
            N,
            dtype,
            block_n,
            block_k,
            per_channel_quant=per_channel_quant,
            down_moe=down_moe,
        )


def install_tuned_moe_configs() -> None:
    """Serve CONFIGS for the Drama 3 expert shape from sglang's config lookup; other
    shapes, devices and Triton versions keep sglang's lookup."""
    device_name = fused_moe_triton_config.get_device_name()
    if device_name != DEVICE_NAME or triton.__version__ != TRITON_VERSION:
        logger.info(
            f"Drama 3 fused MoE: no tuned configs for {device_name} "
            f"under Triton {triton.__version__}"
        )
    else:
        fused_moe_triton_config.get_moe_configs = get_moe_configs
        logger.info(
            f"Drama 3 fused MoE: tuned configs for token counts {sorted(TUNED_CONFIGS)}"
        )
