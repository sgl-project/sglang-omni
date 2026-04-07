import json
import os
from copy import deepcopy
from typing import Any

import yaml
from transformers import AutoConfig

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

_CONFIG_MODEL_TYPE_TO_ARCH = {
    "voxtral_tts": "VoxtralTTSForConditionalGeneration",
}


def _architecture_from_hf_config(hf_config: Any) -> str | None:
    """Prefer HF ``architectures``; fall back to ``model_type`` when needed."""
    archs = getattr(hf_config, "architectures", None)
    if archs:
        for a in archs:
            if a:
                return a
    mt = getattr(hf_config, "model_type", None)
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]
    return None


def _load_mistral_params_json(model_path: str) -> dict | None:
    """Load Mistral-format ``params.json`` from a local dir or Hugging Face hub id.

    Official Voxtral TTS checkpoints ship without ``config.json``; architecture is
    only indicated by ``model_type`` inside ``params.json`` (see Hub repo files).
    """
    params_path = os.path.join(model_path, "params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            return json.load(f)
    # Local directory without params — do not treat as hub repo id
    if os.path.isdir(model_path):
        return None
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=model_path, filename="params.json")
        with open(cached) as f:
            return json.load(f)
    except Exception:
        return None


def _try_resolve_arch_from_mistral_config(model_path: str) -> str | None:
    """Resolve architecture from Mistral-format params.json (local or Hub)."""
    params = _load_mistral_params_json(model_path)
    if params is None:
        return None
    model_type = params.get("model_type", "")
    return _CONFIG_MODEL_TYPE_TO_ARCH.get(model_type)


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
    def from_model_path(model_path: str, variant: str | None = None) -> "ConfigManager":
        """Load config from model path, optionally selecting a variant."""
        import importlib

        arch = None

        # 1) Hugging Face: local snapshot or hub id (hub ids have no local config.json)
        try:
            hf_config = AutoConfig.from_pretrained(model_path)
            arch = _architecture_from_hf_config(hf_config)
        except Exception:
            pass

        # 2) Mistral-format params.json (e.g. Voxtral TTS) when HF config is absent
        if arch is None:
            arch = _try_resolve_arch_from_mistral_config(model_path)

        if arch is None:
            raise ValueError(
                f"Could not resolve model architecture for {model_path!r}. "
                "Use a Hugging Face model id or a local directory with config.json "
                "(architectures) or Mistral params.json (model_type, e.g. voxtral_tts)."
            )

        config_cls = PIPELINE_CONFIG_REGISTRY.get_config(arch)

        if variant:
            module = importlib.import_module(config_cls.__module__)
            variants = getattr(module, "Variants", None)
            if variants and variant in variants:
                config_cls = variants[variant]
            else:
                raise ValueError(
                    f"Unknown variant '{variant}' for {config_cls.__name__}"
                )

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
