# SPDX-License-Identifier: Apache-2.0
"""Shim module to register Qwen3-Omni Talker in SGLang's model registry.

SGLang's model registry only scans non-package modules at the top level
of the external model package. This module re-exports the Talker model
class so it can be discovered via EntryClass.
"""
from sglang_omni.models.qwen3_omni.talker import Qwen3OmniTalker

EntryClass = Qwen3OmniTalker
