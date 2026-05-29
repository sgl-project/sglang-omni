# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni model integration for sglang-omni."""

from . import config  # noqa: F401
from .registration import register_ming_hf_config

register_ming_hf_config()
