# SPDX-License-Identifier: Apache-2.0
"""GPU-specific runtime compatibility helpers."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shutil
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

from sglang_omni.utils.gpu_memory import (
    get_device_handle,
    parse_cuda_visible_devices,
    resolve_visible_device_id,
    shutdown_nvml,
    try_import_pynvml,
)

logger = logging.getLogger(__name__)

_FLASHINFER_USE_CUDA_NORM = "FLASHINFER_USE_CUDA_NORM"
_FLASHINFER_CUDA_ARCH_LIST = "FLASHINFER_CUDA_ARCH_LIST"

PIP_CUDA_BUCKET_PREFIX = "cu"
SYSTEM_CUDA_HOME_DIRECTORY = Path("/usr/local/cuda")
DRIVER_LIBRARY_PATHS = (
    "/usr/lib/wsl/lib/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/aarch64-linux-gnu/libcuda.so",
    "/usr/local/cuda/lib64/stubs/libcuda.so",
)

_GPU_ARCHITECTURES = {
    89: "ada",
    90: "hopper",
    100: "blackwell",
    103: "blackwell",
    120: "blackwell",
}


def gpu_architecture_for_sm(sm_version: int | None) -> str:
    """Return the CUDA architecture family for a detected SM version."""
    if sm_version is None:
        return "unknown"
    else:
        pass
    return _GPU_ARCHITECTURES.get(sm_version, f"sm{sm_version}")


def get_compute_capability(
    logical_gpu_id: int,
    env: Mapping[str, str] | None = None,
) -> tuple[int, int] | None:
    source_env = os.environ if env is None else env
    visible_devices = parse_cuda_visible_devices(
        source_env.get("CUDA_VISIBLE_DEVICES") or ""
    )
    try:
        device_id = resolve_visible_device_id(logical_gpu_id, visible_devices)
    except Exception:
        return None

    pynvml = try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            if visible_devices:
                handle = get_device_handle(pynvml, device_id)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(logical_gpu_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return int(major), int(minor)
        except Exception as exc:
            logger.debug(
                "NVML compute capability query failed for gpu_id=%s: %s",
                logical_gpu_id,
                exc,
            )
        finally:
            shutdown_nvml(pynvml)
    else:
        pass

    if source_env.get("CUDA_VISIBLE_DEVICES") != os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    else:
        pass

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(logical_gpu_id)
            return int(properties.major), int(properties.minor)
        else:
            pass
    except Exception as exc:
        logger.debug(
            "PyTorch compute capability query failed for gpu_id=%s: %s",
            logical_gpu_id,
            exc,
        )
    return None


def get_cuda_device_count() -> int | None:
    pynvml = try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            return int(pynvml.nvmlDeviceGetCount())
        except Exception as exc:
            logger.debug("NVML device count query failed: %s", exc)
        finally:
            shutdown_nvml(pynvml)
    else:
        pass

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
        else:
            pass
    except Exception as exc:
        logger.debug("PyTorch CUDA device count query failed: %s", exc)
    return None


def visible_gpu_ids(env: Mapping[str, str] | None = None) -> list[int]:
    source_env = os.environ if env is None else env
    visible_devices = parse_cuda_visible_devices(
        source_env.get("CUDA_VISIBLE_DEVICES") or ""
    )
    if visible_devices:
        return list(range(len(visible_devices)))
    else:
        pass
    device_count = get_cuda_device_count()
    if device_count is not None:
        return list(range(device_count))
    else:
        pass
    return [0]


def get_visible_gpu_sm_version(
    logical_gpu_id: int,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Return the CUDA SM version for a logical visible GPU."""
    source_env = os.environ if env is None else env
    capability = get_compute_capability(logical_gpu_id, source_env)
    if capability is None:
        return None
    else:
        pass
    major, minor = capability
    return major * 10 + minor


def visible_gpus_need_flashinfer_cuda_norm(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether any visible CUDA device needs the FlashInfer CUDA norm workaround."""
    source_env = os.environ if env is None else env
    for gpu_id in visible_gpu_ids(source_env):
        sm_version = get_visible_gpu_sm_version(gpu_id, source_env)
        if sm_version is not None and sm_version >= 100:
            return True
        else:
            pass
    return False


def get_gpu_compat_env_defaults(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return env overrides needed for the current visible GPU topology."""
    source_env = os.environ if env is None else env
    defaults: dict[str, str] = {}
    if source_env.get(
        _FLASHINFER_USE_CUDA_NORM
    ) is None and visible_gpus_need_flashinfer_cuda_norm(source_env):
        defaults[_FLASHINFER_USE_CUDA_NORM] = "1"
    else:
        pass
    if source_env.get(_FLASHINFER_CUDA_ARCH_LIST) is None:
        # note (Junnan Li): on SM120 the model runner targets 12.0a and FlashInfer alone 12.0f in one
        # JIT cache, so stages with and without a model runner rebuild each other's ops.
        capability = get_compute_capability(0, source_env)
        if capability is not None:
            major, minor = capability
            defaults[_FLASHINFER_CUDA_ARCH_LIST] = (
                f"{major}.{minor}{'a' if major >= 9 else ''}"
            )
        else:
            pass
    else:
        pass
    return defaults


def apply_torch_compile_cache_env(
    env: MutableMapping[str, str] | None = None,
) -> str:
    """Pin TORCHINDUCTOR_CACHE_DIR so the first compile is reused on later starts."""
    target_env = os.environ if env is None else env
    if "TORCHINDUCTOR_CACHE_DIR" not in target_env:
        cache_directory = str(Path.home() / ".cache" / "sglang-omni" / "torchinductor")
        target_env["TORCHINDUCTOR_CACHE_DIR"] = cache_directory
        logger.info(f"Torch compile cache directory: {cache_directory}")
    else:
        pass
    return target_env["TORCHINDUCTOR_CACHE_DIR"]


def find_pip_cuda_toolkit(search_roots: Sequence[Path] | None = None) -> Path | None:
    """Return the CUDA toolkit that the nvidia pip wheels install.

    The pinned stack installs one bucket such as site-packages/nvidia/cu13 that
    carries the compiler, the headers, and the libraries.
    """
    if search_roots is not None:
        package_roots = list(search_roots)
    else:
        specification = importlib.util.find_spec("nvidia")
        if specification is None or specification.submodule_search_locations is None:
            return None
        else:
            pass
        package_roots = [
            Path(location).parent
            for location in specification.submodule_search_locations
        ]
    buckets = sorted(
        (
            bucket
            for package_root in package_roots
            for bucket in (package_root / "nvidia").glob(f"{PIP_CUDA_BUCKET_PREFIX}*")
            if (bucket / "bin" / "nvcc").is_file()
        ),
        key=lambda bucket: bucket.name,
        reverse=True,
    )
    if buckets:
        return buckets[0]
    else:
        pass
    return None


def find_driver_library(
    search_directories: Sequence[Path] | None = None,
) -> Path | None:
    """Return a libcuda library the linker can resolve, if this host has one."""
    if search_directories is not None:
        candidates = list(search_directories)
    else:
        candidates = [Path(path) for path in DRIVER_LIBRARY_PATHS]
        for directory in sorted(Path("/usr/lib").glob("*")):
            candidates.extend(sorted(directory.glob("libcuda.so*")))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def link_entry(source: Path, link: Path) -> None:
    """Point link at source, replacing a previous entry."""
    if link.is_symlink() or link.exists():
        link.unlink()
    else:
        pass
    link.symlink_to(source)


def build_pip_cuda_home(toolkit_directory: Path, home_directory: Path) -> Path | None:
    """Mirror a pip toolkit as a system-toolkit-shaped CUDA root.

    FlashInfer and the sglang JIT both link against cuda_home/lib64 and
    cuda_home/lib64/stubs, while the wheels install lib without a driver stub.
    """
    driver_library = find_driver_library()
    if driver_library is None:
        logger.warning(
            f"Cannot link against {toolkit_directory}: no libcuda library found"
        )
        return None
    else:
        pass
    home_directory.mkdir(parents=True, exist_ok=True)
    for entry in ("bin", "include", "nvvm", "cccl"):
        entry_directory = toolkit_directory / entry
        if entry_directory.is_dir():
            link_entry(entry_directory, home_directory / entry)
        else:
            pass
    library_directory = home_directory / "lib64"
    stub_directory = library_directory / "stubs"
    stub_directory.mkdir(parents=True, exist_ok=True)
    for library in sorted((toolkit_directory / "lib").iterdir()):
        link_entry(library, library_directory / library.name)
        if ".so." in library.name:
            unversioned_name = f"{library.name.split('.so.')[0]}.so"
            link_entry(library, library_directory / unversioned_name)
        else:
            pass
    link_entry(driver_library, stub_directory / "libcuda.so")
    link_entry(driver_library, library_directory / "libcuda.so")
    return home_directory


def derive_pip_cuda_home(env: Mapping[str, str]) -> Path | None:
    """Derive a CUDA root from the nvidia pip wheels when no toolkit is installed.

    JIT builds resolve nvcc from CUDA_HOME, then PATH, then /usr/local/cuda, so
    a wheel-provided toolkit is invisible to them without this override.
    """
    if env.get("CUDA_HOME") is not None or env.get("CUDA_PATH") is not None:
        return None
    else:
        pass
    if shutil.which("nvcc") is not None or SYSTEM_CUDA_HOME_DIRECTORY.exists():
        return None
    else:
        pass
    toolkit_directory = find_pip_cuda_toolkit()
    if toolkit_directory is None:
        return None
    else:
        pass
    home_directory = (
        Path.home() / ".cache" / "sglang-omni" / f"cuda-home-{toolkit_directory.name}"
    )
    return build_pip_cuda_home(toolkit_directory, home_directory)


def apply_gpu_compat_env_defaults(
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply GPU compatibility env overrides to the current process."""
    target_env = os.environ if env is None else env
    overrides = get_gpu_compat_env_defaults(target_env)
    pip_cuda_home = derive_pip_cuda_home(target_env)
    if pip_cuda_home is not None:
        overrides["CUDA_HOME"] = str(pip_cuda_home)
    else:
        pass
    for key, value in overrides.items():
        target_env[key] = value
        logger.info(f"Applied GPU compatibility env override: {key}={value}")
    apply_torch_compile_cache_env(target_env)
    return overrides


def gpu_ids_support_p2p_mesh(
    logical_gpu_ids: Sequence[int],
    env: Mapping[str, str] | None = None,
) -> bool | None:
    """Return whether the given logical GPUs form a full peer-to-peer mesh.

    Custom (P2P/NVLink) all-reduce only works when every tensor-parallel rank can
    directly access every other rank's memory. This queries NVML's pairwise P2P
    status (so it does not create a CUDA context in the caller) for all ordered
    pairs of the given logical GPU ids (resolved through ``CUDA_VISIBLE_DEVICES``).

    Returns ``True`` only when every pair reports P2P-capable, ``False`` when any
    pair is not, and ``None`` when the topology cannot be determined (pynvml
    missing, NVML query error, or fewer than two distinct GPUs).
    """
    ids = list(dict.fromkeys(int(g) for g in logical_gpu_ids))
    if len(ids) < 2:
        return None
    else:
        pass

    pynvml = try_import_pynvml()
    if pynvml is None:
        return None
    else:
        pass

    get_status = getattr(pynvml, "nvmlDeviceGetP2PStatus", None)
    if get_status is None:
        return None
    else:
        pass
    status_ok = getattr(pynvml, "NVML_P2P_STATUS_OK", 0)
    read_index = getattr(pynvml, "NVML_P2P_CAPS_INDEX_READ", 0)
    # note (luojiaxuan): nvidia-ml-py 13.595.45 ships a stray trailing comma
    # (`= 0,`), making this constant a 1-tuple;
    # nvmlDeviceGetP2PStatus needs a plain int.
    if isinstance(read_index, tuple):
        read_index = read_index[0]
    else:
        pass

    source_env = os.environ if env is None else env
    visible_devices = parse_cuda_visible_devices(
        source_env.get("CUDA_VISIBLE_DEVICES") or ""
    )

    try:
        pynvml.nvmlInit()
        handles = []
        for logical_id in ids:
            device_id = resolve_visible_device_id(logical_id, visible_devices)
            handles.append(get_device_handle(pynvml, device_id))
        for i, handle_i in enumerate(handles):
            for j, handle_j in enumerate(handles):
                if i == j:
                    continue
                else:
                    pass
                if get_status(handle_i, handle_j, read_index) != status_ok:
                    return False
                else:
                    pass
        return True
    except Exception as exc:
        logger.warning(
            f"NVML P2P mesh query failed for gpus={ids}: {exc}; keeping custom all-reduce disabled",
        )
        return None
    finally:
        shutdown_nvml(pynvml)


def should_disable_custom_all_reduce_for_gpus(
    logical_gpu_ids: Sequence[int] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether to disable SGLang custom all-reduce for a TP thinker.

    Custom all-reduce requires a direct P2P mesh between the tensor-parallel GPUs;
    on topologies without it (or that can't be confirmed) it must fall back to
    NCCL. This returns ``True`` (disable) unless NVML confirms a full P2P mesh,
    so the safe default is preserved and custom all-reduce is only enabled on
    capable topologies (e.g. NVLink).
    """
    if not logical_gpu_ids:
        return True
    else:
        pass
    return gpu_ids_support_p2p_mesh(logical_gpu_ids, env) is not True
