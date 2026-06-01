import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, Type

from sglang_omni_v1.config import PipelineConfig

logger = logging.getLogger(__name__)


@lru_cache()
def import_pipeline_configs(
    package_name: str, config_path: str, strict: bool = False
) -> Dict[str, Type[PipelineConfig]]:
    # Import the package first so pkgutil can enumerate its model subpackages.
    package = importlib.import_module(package_name)
    model_arch_to_config_cls = {}

    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            if strict:
                raise
            logger.warning(f"Ignore import error when loading {name}: {exc}")
            continue
        config_module_name = f"{name}.{config_path}"
        try:
            config_module = importlib.import_module(config_module_name)
        except ModuleNotFoundError as exc:
            if exc.name == config_module_name:
                if strict:
                    raise
                logger.debug(f"Skipping {name}: no submodule {config_path}")
                continue
            if strict:
                raise
            logger.warning(
                f"Ignore import error when loading {config_module_name}: {exc}"
            )
            continue
        except ImportError as exc:
            if strict:
                raise
            logger.warning(
                f"Ignore import error when loading {config_module_name}: {exc}"
            )
            continue
        if not hasattr(config_module, "EntryClass"):
            raise AssertionError(
                f"Config module {name}.{config_path} must have an EntryClass"
            )
        config_cls = config_module.EntryClass
        model_arch_to_config_cls[config_cls.architecture] = config_cls
    return model_arch_to_config_cls


@dataclass
class _PipelineConfigRegistry:
    configs: Dict[str, Type[PipelineConfig]] = field(default_factory=dict)

    def register_config(
        self,
        package_name: str,
        config_path: str = "config",
        overwrite: bool = False,
        strict: bool = False,
    ) -> None:
        pipeline_configs = import_pipeline_configs(package_name, config_path, strict)

        if overwrite:
            self.configs.update(pipeline_configs)
        else:
            for arch, cfg_cls in pipeline_configs.items():
                if arch in self.configs:
                    raise ValueError(
                        f"Config for {arch} already registered in the pipeline config registry"
                    )
                else:
                    self.configs[arch] = cfg_cls

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.configs.keys()

    def get_config(self, arch: str) -> Type[PipelineConfig]:
        if arch not in self.configs:
            raise ValueError(
                f"Config for {arch} not found in the pipeline config registry"
            )
        return self.configs[arch]

    def get_config_cls_by_name(self, name: str) -> Type[PipelineConfig]:
        for config_cls in self.configs.values():
            if config_cls.__name__ == name:
                return config_cls
        raise ValueError(
            f"Config class {name} not found in the pipeline config registry"
        )


PIPELINE_CONFIG_REGISTRY = _PipelineConfigRegistry()
PIPELINE_CONFIG_REGISTRY.register_config("sglang_omni_v1.models", "config")
