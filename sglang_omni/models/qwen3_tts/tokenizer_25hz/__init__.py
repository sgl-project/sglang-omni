# SPDX-License-Identifier: Apache-2.0
"""25Hz speech tokenizer (V1) package."""

from sglang_omni.models.qwen3_tts.tokenizer_25hz.configuration import *  # noqa: F401, F403

# Modeling module is imported lazily by from_pretrained / AutoModel
# to avoid pulling in heavy VQ deps at package-level.
