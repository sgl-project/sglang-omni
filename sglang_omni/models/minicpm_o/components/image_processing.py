# SPDX-License-Identifier: Apache-2.0

"""Pillow-compatible CUDA image resizing and patch packing."""

import math
from functools import lru_cache
from typing import Protocol, TypedDict

import numpy as np
import torch
import triton
import triton.language as tl
from numpy.typing import NDArray
from PIL import Image

RESAMPLE_PRECISION_BITS = 22
RESAMPLE_BLOCK_SIZE = 256


class SliceGeometry(Protocol):
    max_slice_nums: int
    scale_resolution: int
    patch_size: int
    slice_mode: bool
    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    def get_sliced_grid(
        self, image_size: tuple[int, int], max_slice_nums: int
    ) -> list[int] | None: ...

    def find_best_resize(
        self,
        original_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]: ...

    def get_refine_size(
        self,
        original_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]: ...


class ImageFeatures(TypedDict):
    pixel_values: list[list[torch.Tensor]]
    image_sizes: list[list[tuple[int, int]]]
    tgt_sizes: list[torch.Tensor]


@triton.jit
def resize_pass(
    pixels: tl.tensor,
    coefficients: tl.tensor,
    starts: tl.tensor,
    output: tl.tensor,
    input_width: tl.constexpr,
    input_height: tl.constexpr,
    output_width: tl.constexpr,
    output_height: tl.constexpr,
    taps: tl.constexpr,
    horizontal: tl.constexpr,
    block_size: tl.constexpr,
    precision_bits: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < 3 * output_width * output_height
    columns = offsets % output_width
    rows = offsets // output_width % output_height
    channels = offsets // (output_width * output_height)
    coordinate = columns if horizontal else rows
    start = tl.load(starts + coordinate, valid, other=0)
    total = tl.full((block_size,), 1 << (precision_bits - 1), tl.int32)
    for tap in range(taps):
        source_columns = start + tap if horizontal else columns
        source_rows = rows if horizontal else start + tap
        weight = tl.load(coefficients + coordinate * taps + tap, valid, other=0)
        source = (channels * input_height + source_rows) * input_width + source_columns
        value = tl.load(
            pixels + source,
            valid & (source_columns < input_width) & (source_rows < input_height),
            other=0,
        ).to(tl.int32)
        total += value * weight
    tl.store(
        output + offsets,
        tl.minimum(tl.maximum(total >> precision_bits, 0), 255),
        valid,
    )


class CUDAImageProcessor:
    """Reuse device constants and fixed-point bicubic filters across requests."""

    def __init__(self, geometry: SliceGeometry, device: torch.device) -> None:
        self.geometry = geometry
        self.device = device
        values = np.arange(256, dtype=np.float32)[None, :] / np.float32(255)
        normalized = (
            values - np.asarray(geometry.mean, dtype=np.float32)[:, None]
        ) / np.asarray(geometry.std, dtype=np.float32)[:, None]
        self.normalized_pixels = torch.from_numpy(normalized).to(device)

    @lru_cache(maxsize=32)
    def coefficients(
        self, input_length: int, output_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = input_length / output_length
        filter_scale = max(scale, 1.0)
        support = 2.0 * filter_scale
        taps = math.ceil(support) * 2 + 1
        coefficients = np.zeros((output_length, taps), dtype=np.int32)
        starts = np.zeros(output_length, dtype=np.int32)
        for i in range(output_length):
            center = (i + 0.5) * scale
            start = max(int(center - support + 0.5), 0)
            end = min(int(center + support + 0.5), input_length)
            weights: list[float] = []
            for j in range(start, end):
                distance = abs((j - center + 0.5) * (1.0 / filter_scale))
                if distance < 1.0:
                    weight = (1.5 * distance - 2.5) * distance * distance + 1.0
                elif distance < 2.0:
                    weight = (
                        ((distance - 5.0) * distance + 8.0) * distance - 4.0
                    ) * -0.5
                else:
                    weight = 0.0
                weights.append(weight)
            total = sum(weights)
            starts[i] = start
            for j, weight in enumerate(weights):
                scaled_weight = weight / total * (1 << RESAMPLE_PRECISION_BITS)
                coefficients[i, j] = int(
                    scaled_weight + (0.5 if scaled_weight >= 0 else -0.5)
                )
        return torch.from_numpy(coefficients).to(self.device), torch.from_numpy(
            starts
        ).to(self.device)

    def resize(self, pixels: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        output_width, output_height = size
        for horizontal, output_length in ((True, output_width), (False, output_height)):
            input_height, input_width = pixels.shape[-2:]
            input_length = input_width if horizontal else input_height
            if input_length == output_length:
                continue
            else:
                pass
            coefficients, starts = self.coefficients(input_length, output_length)
            shape = (
                (3, input_height, output_length)
                if horizontal
                else (3, output_length, input_width)
            )
            resized = torch.empty(shape, dtype=torch.uint8, device=self.device)
            with torch.cuda.device(self.device):
                resize_pass[(triton.cdiv(resized.numel(), RESAMPLE_BLOCK_SIZE),)](
                    pixels,
                    coefficients,
                    starts,
                    resized,
                    input_width,
                    input_height,
                    shape[2],
                    shape[1],
                    coefficients.shape[1],
                    horizontal,
                    RESAMPLE_BLOCK_SIZE,
                    RESAMPLE_PRECISION_BITS,
                )
            pixels = resized
        return pixels

    @torch.inference_mode()
    def __call__(
        self,
        images: Image.Image | list[Image.Image] | list[list[Image.Image]] | None = None,
        do_pad: bool = True,
        max_slice_nums: int | None = 1,
        return_tensors: str = "pt",
    ) -> ImageFeatures:
        """Pack slices; do_pad is retained for the checkpoint processor contract."""
        geometry = self.geometry
        if return_tensors != "pt":
            raise ValueError("CUDA image preprocessing requires return_tensors='pt'")
        else:
            pass
        pixel_batches: list[list[torch.Tensor]] = []
        size_batches: list[list[tuple[int, int]]] = []
        target_batches: list[torch.Tensor] = []
        device = self.device
        slice_limit = (
            geometry.max_slice_nums if max_slice_nums is None else max_slice_nums
        )
        if isinstance(images, Image.Image):
            image_batches = [[images]]
        elif images and isinstance(images[0], Image.Image):
            image_batches = [images]
        else:
            image_batches = images or [[]]
        for image_batch in image_batches:
            packed_slices: list[torch.Tensor] = []
            target_sizes: list[tuple[int, int]] = []
            original_sizes: list[tuple[int, int]] = []
            for image in image_batch:
                original_sizes.append(image.size)
                pixels = torch.from_numpy(np.array(image.convert("RGB"))).to(device)
                pixels = pixels.permute(2, 0, 1).contiguous()
                if geometry.slice_mode:
                    grid = geometry.get_sliced_grid(image.size, slice_limit)
                    overview_size = geometry.find_best_resize(
                        image.size,
                        geometry.scale_resolution,
                        geometry.patch_size,
                        allow_upscale=grid is None,
                    )
                    output_sizes = [overview_size]
                    if grid is not None:
                        output_sizes.append(
                            geometry.get_refine_size(
                                image.size,
                                grid,
                                geometry.scale_resolution,
                                geometry.patch_size,
                                allow_upscale=True,
                            )
                        )
                    else:
                        pass
                else:
                    grid = None
                    output_sizes = [image.size]
                for output_index, (width, height) in enumerate(output_sizes):
                    resized = self.resize(pixels, (width, height))
                    normalized = self.normalized_pixels[
                        torch.arange(3, device=device)[:, None, None], resized.long()
                    ]
                    if output_index == 0:
                        slices = [normalized]
                    else:
                        assert grid is not None
                        tile_width, tile_height = width // grid[0], height // grid[1]
                        slices = [
                            normalized[:, y : y + tile_height, x : x + tile_width]
                            for y in range(0, height, tile_height)
                            for x in range(0, width, tile_width)
                        ]
                    for image_slice in slices:
                        channels, slice_height, slice_width = image_slice.shape
                        patch_size = geometry.patch_size
                        patch_rows = slice_height // patch_size
                        patch_columns = slice_width // patch_size
                        packed = (
                            image_slice.reshape(
                                channels,
                                patch_rows,
                                patch_size,
                                patch_columns,
                                patch_size,
                            )
                            .permute(0, 2, 1, 3, 4)
                            .reshape(channels, patch_size, -1)
                        )
                        packed_slices.append(packed)
                        target_sizes.append((patch_rows, patch_columns))
            pixel_batches.append(packed_slices)
            size_batches.append(original_sizes)
            target_batches.append(
                torch.tensor(target_sizes, dtype=torch.int64).reshape(-1, 2)
            )
        return {
            "pixel_values": pixel_batches,
            "image_sizes": size_batches,
            "tgt_sizes": target_batches,
        }
