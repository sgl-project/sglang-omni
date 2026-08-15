# SPDX-License-Identifier: Apache-2.0
"""Production request limits for SenseNova U1 image generation."""

from __future__ import annotations

from typing import Any

U1_IMAGE_SIZE_DIVISOR = 32
U1_GENERATED_IMAGE_TOKEN_OVERHEAD = 2
U1_MAX_IMAGE_DIMENSION = 2048
U1_MAX_IMAGE_PIXELS = 1024 * 1024
U1_MAX_DIFFUSION_STEPS = 64
U1_MAX_GENERATED_IMAGES = 4
U1_MAX_INPUT_IMAGES = 4
U1_MAX_NEW_TOKENS = 2048
U1_MAX_TOTAL_TOKENS = 4096


def parse_int_param(value: Any, *, name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required.")
        return int(default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def validate_image_size(width: Any, height: Any) -> tuple[int, int]:
    parsed_width = parse_int_param(width, name="width")
    parsed_height = parse_int_param(height, name="height")
    for name, value in (("width", parsed_width), ("height", parsed_height)):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        if value > U1_MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"{name} exceeds the deployment maximum "
                f"{U1_MAX_IMAGE_DIMENSION}: {value}."
            )
        if value % U1_IMAGE_SIZE_DIVISOR != 0:
            raise ValueError(
                f"{name} must be divisible by {U1_IMAGE_SIZE_DIVISOR}: {value}."
            )
    pixels = parsed_width * parsed_height
    if pixels > U1_MAX_IMAGE_PIXELS:
        raise ValueError(
            "image pixel count exceeds the deployment maximum "
            f"{U1_MAX_IMAGE_PIXELS}: {pixels}."
        )
    return parsed_width, parsed_height


def validate_num_steps(value: Any) -> int:
    num_steps = parse_int_param(value, name="num_steps")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if num_steps > U1_MAX_DIFFUSION_STEPS:
        raise ValueError(
            "num_steps exceeds the deployment maximum "
            f"{U1_MAX_DIFFUSION_STEPS}: {num_steps}."
        )
    return num_steps


def validate_image_count(
    value: Any,
    *,
    name: str,
    maximum: int = U1_MAX_GENERATED_IMAGES,
) -> int:
    image_count = parse_int_param(value, name=name)
    if image_count <= 0:
        raise ValueError(f"{name} must be positive.")
    if image_count > maximum:
        raise ValueError(
            f"{name} exceeds the deployment maximum {maximum}: {image_count}."
        )
    return image_count


def validate_input_image_count(images: list[Any]) -> None:
    if len(images) > U1_MAX_INPUT_IMAGES:
        raise ValueError(
            "input image count exceeds the deployment maximum "
            f"{U1_MAX_INPUT_IMAGES}: {len(images)}."
        )


def validate_max_new_tokens(value: Any) -> int:
    max_new_tokens = parse_int_param(value, name="max_new_tokens")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if max_new_tokens > U1_MAX_NEW_TOKENS:
        raise ValueError(
            "max_new_tokens exceeds the deployment maximum "
            f"{U1_MAX_NEW_TOKENS}: {max_new_tokens}."
        )
    return max_new_tokens


def generated_image_token_count(image_size: tuple[int, int]) -> int:
    width, height = validate_image_size(*image_size)
    return (width // U1_IMAGE_SIZE_DIVISOR) * (
        height // U1_IMAGE_SIZE_DIVISOR
    )


def generated_image_span_token_count(image_size: tuple[int, int]) -> int:
    return (
        generated_image_token_count(image_size)
        + U1_GENERATED_IMAGE_TOKEN_OVERHEAD
    )


def validate_total_token_budget(
    *,
    image_size: tuple[int, int],
    image_count: int,
    max_new_tokens: int = 0,
    prefix_tokens: int = 0,
    max_total_tokens: int = U1_MAX_TOTAL_TOKENS,
) -> int:
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive.")
    image_tokens = generated_image_span_token_count(image_size)
    return validate_token_budget_components(
        prefix_tokens=prefix_tokens,
        text_tokens=max_new_tokens,
        image_tokens=int(image_count) * image_tokens,
        max_total_tokens=max_total_tokens,
        detail=f"images={image_count}x{image_tokens}",
    )


def validate_token_budget_components(
    *,
    prefix_tokens: int,
    text_tokens: int,
    image_tokens: int,
    max_total_tokens: int = U1_MAX_TOTAL_TOKENS,
    detail: str | None = None,
) -> int:
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative.")
    if text_tokens < 0:
        raise ValueError("text_tokens must be non-negative.")
    if image_tokens < 0:
        raise ValueError("image_tokens must be non-negative.")
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive.")
    total_tokens = (
        int(prefix_tokens)
        + int(text_tokens)
        + int(image_tokens)
    )
    if total_tokens > max_total_tokens:
        detail_text = f", {detail}" if detail else ""
        raise ValueError(
            "request token budget exceeds the deployment maximum "
            f"{max_total_tokens}: prefix={prefix_tokens}, "
            f"text={text_tokens}, image_tokens={image_tokens}{detail_text}, "
            f"total={total_tokens}."
        )
    return total_tokens


__all__ = [
    "U1_IMAGE_SIZE_DIVISOR",
    "U1_GENERATED_IMAGE_TOKEN_OVERHEAD",
    "U1_MAX_DIFFUSION_STEPS",
    "U1_MAX_GENERATED_IMAGES",
    "U1_MAX_IMAGE_DIMENSION",
    "U1_MAX_IMAGE_PIXELS",
    "U1_MAX_INPUT_IMAGES",
    "U1_MAX_NEW_TOKENS",
    "U1_MAX_TOTAL_TOKENS",
    "generated_image_token_count",
    "generated_image_span_token_count",
    "parse_int_param",
    "validate_image_count",
    "validate_image_size",
    "validate_input_image_count",
    "validate_max_new_tokens",
    "validate_num_steps",
    "validate_token_budget_components",
    "validate_total_token_budget",
]
