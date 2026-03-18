from copy import deepcopy
from typing import Any

import yaml
from transformers import AutoConfig

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


class ConfigManager:
    """
    The ConfigManager is responsible for managing the configuration based on the user CLI arguments, configuration file
    given by the user, and the default configuration for the model. As the omni models have various architectures, setting a uniform
    list of arguments is not feasible. Thus, we take reference from the TorchTitan's configuration management system to allow users to
    dynamically configure their runtime settings.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config

    def parse_extra_args(self, args: list[str]) -> dict[str, Any]:
        """
        Parse the CLI arguments and return the configuration.
        """
        # we expect the arguments to be key-values pairs
        extra_args = {}
        cur_key, cur_value = None, None
        for arg in args:
            if "=" in arg and cur_key is None and cur_value is None:
                cur_key, cur_value = arg.split("=", 1)
            elif cur_key is None and cur_value is None:
                cur_key = arg
            elif cur_key is not None and cur_value is None:
                # record the key value pair
                cur_value = arg
            else:
                raise ValueError(f"Invalid argument: {arg}")

            if cur_key is not None and cur_value is not None:
                # remove the -- in front of the key
                formatted_key = cur_key.lstrip("-").replace("-", "_")
                extra_args[formatted_key] = cur_value
                cur_key, cur_value = None, None
        return extra_args

    def _convert_types(self, extra_args: dict[str, str]) -> dict[str, Any]:
        """
        Convert the configuration to the inferred data types.
        """
        for key, value in extra_args.items():
            if value.lower() == "true":
                extra_args[key] = True
            elif value.lower() == "false":
                extra_args[key] = False
            elif value.lower() == "none":
                extra_args[key] = None
            elif value.isnumeric():
                extra_args[key] = float(value) if "." in value else int(value)
            else:
                extra_args[key] = value
        return extra_args

    def merge_config(self, extra_args: dict[str, Any]) -> PipelineConfig:
        """
        Merge the configuration and the extra arguments.
        """
        extra_args = self._convert_types(extra_args)

        # we then update the configuration
        # note that the key of the extra argumeents is in the chained format, e.g. "stages.0.executor.args.dtype"
        # we need to update the configuration in place
        config_data = self.config.model_dump()
        config_cls = type(self.config)

        cfg_copy = deepcopy(config_data)
        for key, value in extra_args.items():
            current = cfg_copy
            keys = key.split(".")
            for k in keys[:-1]:
                # if k is an digit, treat it as an index
                if k.isdigit():
                    k = int(k)
                current = current[k]

            # update the value
            current[keys[-1]] = value

        # validate the configuration
        merged_config = config_cls(**cfg_copy)
        return merged_config

    @staticmethod
    def from_model_path(model_path: str) -> "ConfigManager":
        """
        Load the configuration from the model path.
        """
        hf_config = AutoConfig.from_pretrained(model_path)
        config_cls = PIPELINE_CONFIG_REGISTRY.get_config(hf_config.architectures[0])
        config = config_cls(model_path=model_path)
        return ConfigManager(config)

    @staticmethod
    def from_file(file_path: str) -> "ConfigManager":
        """
        Load the configuration from the file path.
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        config_cls_str = data["config_cls"]
        config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(config_cls_str)
        config = config_cls(**data)
        return ConfigManager(config)
