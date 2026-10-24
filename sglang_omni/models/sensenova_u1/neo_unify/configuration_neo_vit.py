"""SenseNova vision configuration."""

import logging
import os
from collections.abc import Sequence
from typing import overload

from pydantic import JsonValue
from transformers.configuration_utils import PretrainedConfig

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


@overload
def as_singleton_tuple(
    value: int | Sequence[int],
) -> tuple[int, ...]:
    pass


@overload
def as_singleton_tuple(
    value: float | Sequence[float],
) -> tuple[float, ...]:
    pass


def as_singleton_tuple(
    value: int | float | Sequence[int | float],
) -> tuple[int | float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 1:
            return as_singleton_tuple(value[0])
        else:
            pass
        return tuple(value)
    else:
        pass
    return (value,)


class NEOVisionConfig(PretrainedConfig):
    model_type: str = "neo_vision"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 1024,
        llm_hidden_size: int | Sequence[int] = 2048,
        downsample_ratio: float | Sequence[float] = 0.5,
        rope_theta_vision: float = 10000.0,
        max_position_embeddings_vision: int = 10000,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
        **kwargs: JsonValue,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size: int = hidden_size
        self.llm_hidden_size: tuple[int, ...] = as_singleton_tuple(llm_hidden_size)
        self.downsample_ratio: tuple[float, ...] = as_singleton_tuple(downsample_ratio)
        self.rope_theta_vision: float = rope_theta_vision
        self.max_position_embeddings_vision: int = max_position_embeddings_vision
        self.num_channels: int = num_channels
        self.patch_size: int = patch_size
        self.min_pixels: int = min_pixels
        self.max_pixels: int = max_pixels

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: JsonValue
    ) -> PretrainedConfig:
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]
        else:
            pass

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        else:
            pass

        return cls.from_dict(config_dict, **kwargs)
