"""Image loading and native-resolution preprocessing."""

from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image

# Modified for SGLang; see this directory's README.md for upstream source.


SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_pixel_values(
    pixel_values: torch.Tensor, patch_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]], device=pixel_values.device)
    return flatten_pixel_values, grid_hw


def load_image_native(
    image: Image.Image,
    patch_size: int,
    downsample_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and pack a reference already resized by image_processing."""
    image_factor = int(patch_size / downsample_ratio)
    assert image.width % image_factor == 0 and image.height % image_factor == 0
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return preprocess_pixel_values(
        transform(image).to(torch.float32), patch_size=patch_size
    )
