from copy import deepcopy
from typing import Any

import yaml
from transformers import AutoConfig

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.utils import (
    architecture_from_hf_config,
    try_resolve_arch_from_mistral_config,
    try_resolve_arch_from_raw_config,
)


def resolve_config_cls_for_model_path(model_path: str):
    """Resolve a PipelineConfig class from HF config metadata."""
    hf_config = None
    try:
        hf_config = AutoConfig.from_pretrained(model_path)
    except (OSError, ValueError, KeyError):
        hf_config = None

    arch = architecture_from_hf_config(hf_config) if hf_config is not None else None
    if arch is None:
        arch = try_resolve_arch_from_raw_config(model_path)
    if arch is None:
        arch = try_resolve_arch_from_mistral_config(model_path)
    if arch is None:
        raise ValueError(f"Could not resolve model architecture for {model_path!r}")
    return PIPELINE_CONFIG_REGISTRY.get_config(arch)


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
        if cur_key is not None and cur_value is None:
            raise ValueError(f"Missing value for argument: {cur_key}")
        return extra_args

    def _convert_types(self, extra_args: dict[str, Any]) -> dict[str, Any]:
        """
        Convert the configuration to the inferred data types.
        """
        return {key: _convert_scalar(value) for key, value in extra_args.items()}

    def merge_config(self, extra_args: dict[str, Any]) -> PipelineConfig:
        """
        Merge the configuration and the extra arguments.
        """
        extra_args = self._convert_types(extra_args)
        config_data = self.config.model_dump()
        config_cls = type(self.config)

        cfg_copy = deepcopy(config_data)
        for key, value in extra_args.items():
            current = cfg_copy
            keys = key.split(".")
            for k in keys[:-1]:
                current = _resolve_child(current, k)

            # update the value
            last_key = keys[-1]
            if isinstance(current, list):
                current[_resolve_list_index(current, last_key)] = value
            else:
                current[last_key] = value

        _sync_stage_parallelism_aliases(cfg_copy, set(extra_args))

        # validate the configuration
        merged_config = config_cls(**cfg_copy)
        return merged_config

    @staticmethod
    def from_model_path(model_path: str, variant: str | None = None) -> "ConfigManager":
        """Load config from model path, optionally selecting a variant."""
        import importlib

        config_cls = resolve_config_cls_for_model_path(model_path)

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
        if not isinstance(data, dict):
            raise ValueError(f"Config file {file_path!r} must contain a mapping")

        data = dict(data)
        has_stage_overrides = "stage_overrides" in data
        stage_overrides = data.pop("stage_overrides", {})
        config_cls_str = data["config_cls"]
        config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(config_cls_str)
        config = config_cls(**data)
        if has_stage_overrides:
            config = _apply_stage_overrides(config, stage_overrides)
        return ConfigManager(config)


def _apply_stage_overrides(
    config: PipelineConfig,
    stage_overrides: dict[str, Any],
) -> PipelineConfig:
    """Apply compact file-level runtime overrides by stage name."""

    if not isinstance(stage_overrides, dict):
        raise ValueError(
            "stage_overrides must be a mapping from stage name to overrides"
        )

    config_data = config.model_dump()
    stages = config_data["stages"]
    stage_by_name = {stage["name"]: stage for stage in stages}

    for stage_name, override in stage_overrides.items():
        if stage_name not in stage_by_name:
            raise ValueError(f"stage_overrides references unknown stage {stage_name!r}")
        if not isinstance(override, dict):
            raise ValueError(f"stage_overrides.{stage_name} must be a mapping")

        unsupported = sorted(set(override) - {"runtime"})
        if unsupported:
            raise ValueError(
                f"stage_overrides.{stage_name} supports only runtime overrides; "
                f"got unsupported keys {unsupported}"
            )

        if "runtime" not in override:
            continue
        runtime_override = override["runtime"]
        if not isinstance(runtime_override, dict):
            raise ValueError(f"stage_overrides.{stage_name}.runtime must be a mapping")
        stage = stage_by_name[stage_name]
        stage["runtime"] = _deep_merge_dict(
            stage.get("runtime", {}),
            runtime_override,
        )

    return type(config)(**config_data)


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sync_stage_parallelism_aliases(
    config_data: dict[str, Any],
    override_keys: set[str],
) -> None:
    """Keep StageConfig.tp_size and parallelism.tp coherent for dotted CLI args."""
    stages = config_data.get("stages")
    if not isinstance(stages, list):
        return

    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        stage_keys = (str(index), stage["name"])
        has_tp_size_override = any(
            f"stages.{key}.tp_size" in override_keys for key in stage_keys
        )
        has_parallelism_override = any(
            f"stages.{key}.parallelism.tp" in override_keys for key in stage_keys
        )
        if has_tp_size_override == has_parallelism_override:
            continue

        if has_tp_size_override:
            parallelism = dict(stage.get("parallelism") or {})
            parallelism["tp"] = stage["tp_size"]
            stage["parallelism"] = parallelism
        else:
            stage["tp_size"] = stage["parallelism"]["tp"]


def _resolve_list_index(items: list[Any], key: str) -> int:
    """Resolve a dotted-path segment to a list index.

    Supports numeric indices (e.g. ``stages.4.tp_size``) as well as stage names
    (e.g. ``stages.thinker.runtime``) when the list contains mappings with a
    ``name`` field.
    """
    if key.isdigit():
        return int(key)

    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("name") == key:
            return index

    available = [
        item["name"] for item in items if isinstance(item, dict) and "name" in item
    ]
    raise KeyError(
        f"Could not resolve list element {key!r}; expected an integer index "
        f"or one of the named entries {available}"
    )


def _resolve_child(current: Any, key: str) -> Any:
    """Traverse one segment of a dotted config path."""
    if isinstance(current, list):
        return current[_resolve_list_index(current, key)]
    return current[key]


def _convert_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
