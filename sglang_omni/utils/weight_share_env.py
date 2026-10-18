# SPDX-License-Identifier: Apache-2.0
"""Environment variable names of the weight share handoff.

The pipeline runner sets these when it spawns stage processes, and
``sglang_omni.utils.ipc_weights`` reads them in the stage. They live here so
the runner can name them without importing torch.
"""

ENV_WEIGHT_SHARE = "SGLANG_OMNI_WEIGHT_SHARE"
ENV_WEIGHT_SHARE_TIMEOUT_S = "SGLANG_OMNI_WEIGHT_SHARE_TIMEOUT_S"
ENV_WEIGHT_SHARE_RUN_ID = "SGLANG_OMNI_WEIGHT_SHARE_RUN_ID"
ENV_WEIGHT_SHARE_COMPAT = "SGLANG_OMNI_WEIGHT_SHARE_COMPAT"
