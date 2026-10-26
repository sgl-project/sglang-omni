# SPDX-License-Identifier: Apache-2.0
"""Inline image validation at UMM boundaries."""

import base64
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InlineImage(BaseModel):
    """Immutable inline bytes with validated metadata, without pixel decoding."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: Literal["image"] = "image"
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str

    @classmethod
    def from_png(cls, encoded_image: str) -> "InlineImage":
        image_bytes = base64.b64decode(encoded_image, validate=True)
        return cls(
            mime_type="image/png",
            size_bytes=len(image_bytes),
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            url=f"data:image/png;base64,{encoded_image}",
        )

    @model_validator(mode="after")
    def validate_content(self) -> "InlineImage":
        prefix = f"data:{self.mime_type};base64,"
        if not self.url.startswith(prefix):
            raise ValueError(
                "UMM image must be an inline data URL matching its MIME type"
            )
        else:
            encoded_image = self.url[len(prefix) :]
        image_bytes = base64.b64decode(encoded_image, validate=True)
        if (
            len(image_bytes) != self.size_bytes
            or hashlib.sha256(image_bytes).hexdigest() != self.sha256
        ):
            raise ValueError("UMM image size or SHA-256 does not match its content")
        else:
            return self

    def encoded_image(self) -> str:
        return self.url.split(",", 1)[1]
