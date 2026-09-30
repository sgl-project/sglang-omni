# SPDX-License-Identifier: Apache-2.0
"""Ordered text and inline image segments shared by SDK and HTTP output."""

from typing import Literal

from typing_extensions import TypedDict


class InlineImage(TypedDict):
    kind: Literal["image"]
    mime_type: Literal["image/png"]
    url: str
    sha256: str
    size_bytes: int


class UMMSegment(TypedDict):
    type: Literal["segment"]
    session_id: str
    segment_index: int
    kind: Literal["text", "image"]
    data: str | InlineImage
