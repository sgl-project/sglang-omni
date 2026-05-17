from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "find_available_port": ("sglang_omni.utils.connection", "find_available_port"),
    "load_hf_config": ("sglang_omni.utils.hf", "load_hf_config"),
    "instantiate_module": ("sglang_omni.utils.hf", "instantiate_module"),
    "architecture_from_hf_config": (
        "sglang_omni.utils.hf",
        "architecture_from_hf_config",
    ),
    "load_mistral_params_json": (
        "sglang_omni.utils.hf",
        "load_mistral_params_json",
    ),
    "try_resolve_arch_from_mistral_config": (
        "sglang_omni.utils.hf",
        "try_resolve_arch_from_mistral_config",
    ),
    "import_string": ("sglang_omni.utils.imports", "import_string"),
    "get_layer_id": ("sglang_omni.utils.misc", "get_layer_id"),
    "add_prefix": ("sglang_omni.utils.misc", "add_prefix"),
    "set_random_seed": ("sglang_omni.utils.misc", "set_random_seed"),
    "broadcast_pyobj": ("sglang_omni.utils.misc", "broadcast_pyobj"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
