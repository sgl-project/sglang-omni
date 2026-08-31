from collections.abc import Iterable, Mapping
from typing import Any

from transformers import AutoConfig

from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import patches_from_dotted_cli, sources_from_config_file
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

    def parse_extra_args(self, args: list[str]) -> list[tuple[str, str]]:
        """Parse the CLI arguments into ordered ``(key, value)`` pairs.

        Pairs, not a mapping: a flag written twice must survive parsing so the
        resolver can rule on it -- two equal values are harmless, two
        different ones are refused as a conflict. A dict here would keep
        whichever came last and silently invent an argument-order rule.
        """
        # we expect the arguments to be key-values pairs
        extra_args: list[tuple[str, str]] = []
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
                extra_args.append((_normalize_flag_key(cur_key), cur_value))
                cur_key, cur_value = None, None
        if cur_key is not None and cur_value is None:
            raise ValueError(f"Missing value for argument: {cur_key}")
        return extra_args

    def merge_config(
        self,
        extra_args: Mapping[str, Any] | Iterable[tuple[str, Any]],
        *,
        extra_patches: ConfigPatchSet | None = None,
    ) -> PipelineConfig:
        """Merge the configuration and the extra arguments.

        The dotted keys are translated into canonical patches and applied by
        :class:`~sglang_omni.config.resolver.ConfigResolver`, which is the only
        code that writes into a configuration.

        ``extra_patches`` carries patches a caller has already translated --
        the ``--model-path`` flag in ``sgl-omni serve``, for instance.
        Everything is resolved together, in one patch set, so that writing the
        same path two ways is refused (or settled by declared specificity)
        rather than by the order the translations happen to run in.
        """
        patches = patches_from_dotted_cli(extra_args, self.config)
        if extra_patches is not None:
            patches = patches.merge(extra_patches)
        resolved = ConfigResolver(self.config).resolve(patches)
        _validate_dotted_gpu_override_conflicts(
            resolved.config, {patch.key for patch in patches.ordered()}
        )
        return resolved.config

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

        The file's ``stages:`` mapping entries are folded into the
        configuration that comes back, so callers holding a ``ConfigManager``
        see one settled config rather than a config plus a pile of pending
        overrides. ``sgl-omni config explain`` wants the opposite and calls
        ``sources_from_config_file`` directly.
        """
        config, patches = sources_from_config_file(file_path)
        if not patches:
            return ConfigManager(config)
        resolved = ConfigResolver(config).resolve(patches)
        return ConfigManager(resolved.config)


def _validate_dotted_gpu_override_conflicts(
    config: PipelineConfig,
    override_keys: set[str],
) -> None:
    """Reject stage GPU overrides shadowed by process replica placement."""
    stage_by_name = {stage.name: stage for stage in config.stages}
    for key in sorted(override_keys):
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "stages" or parts[2] != "gpu":
            continue

        stage = stage_by_name.get(parts[1])
        if stage is None:
            continue
        process_name = stage.process or stage.name
        process_config = config.processes.get(process_name)
        if process_config is None or process_config.replica_devices is None:
            continue

        raise ValueError(
            f"{key} cannot override GPU placement for stage {stage.name!r} "
            f"because process {process_name!r} declares replica_devices; update "
            f"processes.{process_name}.replica_devices instead"
        )


def _normalize_flag_key(key: str) -> str:
    """Strip the leading dashes and normalize the flag's first segment.

    Only the first dotted segment gets its dashes rewritten to underscores:
    later segments can be document keys -- ``--stages.thinker.env.MY-FLAG``
    names an env var whose spelling must survive verbatim.
    """
    key = key.lstrip("-")
    head, separator, rest = key.partition(".")
    return head.replace("-", "_") + separator + rest
