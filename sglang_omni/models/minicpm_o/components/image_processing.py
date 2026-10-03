# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypedDict

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray
from PIL import Image


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


@torch.inference_mode()
def process_images(
    geometry: SliceGeometry,
    images: list[list[Image.Image]] | None,
    max_slice_nums: int | None = None,
    return_tensors: str = "pt",
) -> ImageFeatures:
    """Resize, normalize, and pack image slices"""
    if return_tensors != "pt":
        raise ValueError("CUDA image preprocessing requires return_tensors='pt'")
    else:
        pass
    pixel_batches: list[list[torch.Tensor]] = []
    size_batches: list[list[tuple[int, int]]] = []
    target_batches: list[torch.Tensor] = []
    device = torch.device("cuda", torch.cuda.current_device())
    mean = torch.as_tensor(geometry.mean, dtype=torch.float32, device=device).view(
        3, 1, 1
    )
    standard_deviation = torch.as_tensor(
        geometry.std, dtype=torch.float32, device=device
    ).view(3, 1, 1)
    slice_limit = geometry.max_slice_nums if max_slice_nums is None else max_slice_nums
    for image_batch in images or [[]]:
        packed_slices: list[torch.Tensor] = []
        target_sizes: list[tuple[int, int]] = []
        original_sizes: list[tuple[int, int]] = []
        for image in image_batch:
            original_sizes.append(image.size)
            pixels = torch.from_numpy(np.array(image.convert("RGB"))).to(device)
            pixels = pixels.permute(2, 0, 1).unsqueeze(0).float()
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
                resized = (
                    functional.interpolate(
                        pixels,
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )[0]
                    .round()
                    .clamp_(0, 255)
                )
                normalized = resized.div_(255).sub_(mean).div_(standard_deviation)
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
                            channels, patch_rows, patch_size, patch_columns, patch_size
                        )
                        .permute(0, 2, 1, 3, 4)
                        .reshape(channels, patch_size, -1)
                    )
                    packed_slices.append(packed)
                    target_sizes.append((patch_rows, patch_columns))
        pixel_batches.append(packed_slices)
        size_batches.append(original_sizes)
        target_batches.append(
            torch.tensor(target_sizes, dtype=torch.int32).reshape(-1, 2)
        )
    return {
        "pixel_values": pixel_batches,
        "image_sizes": size_batches,
        "tgt_sizes": target_batches,
    }
