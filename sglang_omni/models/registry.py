import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, Type

from sglang_omni.config import PipelineConfig

logger = logging.getLogger(__name__)


@lru_cache()
def import_pipeline_configs(
    package_name: str, config_path: str, strict: bool = False
) -> Dict[str, Type[PipelineConfig]]:
    # import the package
    package = importlib.import_module(package_name)
    model_arch_to_config_cls = {}

    # copy from sglang for on-the-fly model discovery
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        # if this is a package, we import it
        if ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, config_path):
                config_module = getattr(module, config_path)
                assert hasattr(
                    config_module, "EntryClass"
                ), f"Config module {module.__name__} must have an EntryClass in its submodule {config_path}"
                config_cls = getattr(config_module, "EntryClass")
                model_arch_to_config_cls[config_cls.architecture] = config_cls
            else:
                logger.debug(f"Skipping {module.__name__}: no submodule {config_path}")
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
        # we register the model
        pipeline_configs = import_pipeline_configs(package_name, config_path, strict)
        print(pipeline_configs)

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
PIPELINE_CONFIG_REGISTRY.register_config("sglang_omni.models", "config")
