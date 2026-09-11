# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 checkpoint constants and released-recipe defaults."""

ARCHITECTURE = "voxcpm2"

SAMPLE_RATE: int = 16000
OUT_SAMPLE_RATE: int = 48000

PATCH_SIZE: int = 4
FEAT_DIM: int = 64

DEFAULT_INFERENCE_TIMESTEPS: int = 10
DEFAULT_CFG_VALUE: float = 2.0
DEFAULT_MIN_LEN: int = 2
DEFAULT_MAX_LEN: int = 2000
DEFAULT_STREAMING_PREFIX_LEN: int = 4

AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_PROMPT_START_TOKEN = "<|audio_prompt_start|>"
AUDIO_PROMPT_END_TOKEN = "<|audio_prompt_end|>"

CONFIG_FILE = "config.json"
WEIGHT_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin")
AUDIO_VAE_FILE_CANDIDATES = ("audiovae.pth",)
