# SPDX-License-Identifier: Apache-2.0
"""Expose Ming 3 audio's text backbone to the hybrid SGLang runtime."""

from __future__ import annotations

from typing import ClassVar

from pydantic import JsonValue
from sglang.srt.configs.bailing_hybrid import BailingHybridConfig


class Ming3AudioConfig(BailingHybridConfig):
    model_type: ClassVar[str] = "bailing_moe_v3_audio"

    def __init__(
        self,
        text_config: dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        fields = dict(text_config or {})
        fields.update(kwargs)
        fields.pop("model_type", None)
        fields.pop("auto_map", None)
        super().__init__(**fields)
