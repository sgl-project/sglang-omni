# SPDX-License-Identifier: Apache-2.0
"""SGLang model registry entry for MiniCPM-V LLM with paged attention.

This top-level module is needed because SGLang's model registry only scans
non-package modules (*.py files) directly under the registered package path.
"""

from sglang_omni.models.minicpm_v.sglang_llm import MiniCPMVSGLangLLM

EntryClass = MiniCPMVSGLangLLM
