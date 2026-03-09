# SPDX-License-Identifier: Apache-2.0
"""SGLang model registry entry for S2-Pro paged-attention text model.

This top-level module is needed because SGLang's model registry only scans
non-package modules (*.py files) directly under the registered package path.
"""

from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel

EntryClass = S2ProSGLangTextModel
