# SPDX-License-Identifier: Apache-2.0
"""Tests for deriving CUDA_HOME from the nvidia pip wheels."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import sglang_omni.utils.gpu_compat as gpu_compat

posix_only = pytest.mark.skipif(
    os.name != "posix", reason="CUDA toolkit layout and symlinks are POSIX-only"
)


def make_toolkit(directory: Path, *, with_nvcc: bool) -> Path:
    (directory / "bin").mkdir(parents=True)
    if with_nvcc:
        (directory / "bin" / "nvcc").write_text("")
    else:
        pass
    (directory / "include").mkdir()
    (directory / "lib").mkdir()
    (directory / "lib" / "libcudart.so.13").write_text("")
    return directory


def make_driver_library(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libcuda.so"
    library.write_text("")
    return library


def test_find_pip_cuda_toolkit_selects_bucket_with_nvcc(tmp_path) -> None:
    make_toolkit(tmp_path / "nvidia" / "cu12", with_nvcc=False)
    expected = make_toolkit(tmp_path / "nvidia" / "cu13", with_nvcc=True)
    assert gpu_compat.find_pip_cuda_toolkit([tmp_path]) == expected


def test_find_pip_cuda_toolkit_returns_none_without_bucket(tmp_path) -> None:
    (tmp_path / "nvidia" / "cudnn").mkdir(parents=True)
    assert gpu_compat.find_pip_cuda_toolkit([tmp_path]) is None


def test_find_driver_library_uses_first_existing(tmp_path) -> None:
    expected = make_driver_library(tmp_path / "wsl")
    assert gpu_compat.find_driver_library([tmp_path / "absent", expected]) == expected


def test_find_driver_library_returns_none_without_library(tmp_path) -> None:
    assert gpu_compat.find_driver_library([tmp_path / "absent"]) is None


@posix_only
def test_link_entry_is_idempotent(tmp_path) -> None:
    source = tmp_path / "libcudart.so.13"
    source.write_text("")
    link = tmp_path / "libcudart.so"

    gpu_compat.link_entry(source, link)
    gpu_compat.link_entry(source, link)

    assert link.is_symlink()
    assert link.readlink() == source


@posix_only
def test_link_entry_replaces_a_stale_target(tmp_path) -> None:
    previous = tmp_path / "previous" / "libcudart.so.12"
    previous.parent.mkdir()
    previous.write_text("")
    source = tmp_path / "libcudart.so.13"
    source.write_text("")
    link = tmp_path / "libcudart.so"
    link.symlink_to(previous)

    gpu_compat.link_entry(source, link)

    assert link.readlink() == source


@posix_only
def test_build_pip_cuda_home_creates_system_layout(tmp_path, monkeypatch) -> None:
    toolkit_directory = make_toolkit(
        tmp_path / "site-packages" / "nvidia" / "cu13", with_nvcc=True
    )
    driver_library = make_driver_library(tmp_path / "driver")
    monkeypatch.setattr(gpu_compat, "find_driver_library", lambda: driver_library)

    home_directory = gpu_compat.build_pip_cuda_home(
        toolkit_directory, tmp_path / "cuda-home"
    )

    assert home_directory == tmp_path / "cuda-home"
    assert (home_directory / "bin" / "nvcc").resolve() == (
        toolkit_directory / "bin" / "nvcc"
    ).resolve()
    assert (home_directory / "lib64" / "libcudart.so").readlink() == (
        toolkit_directory / "lib" / "libcudart.so.13"
    )
    assert (home_directory / "lib64" / "libcudart.so.13").readlink() == (
        toolkit_directory / "lib" / "libcudart.so.13"
    )
    assert (
        home_directory / "lib64" / "stubs" / "libcuda.so"
    ).readlink() == driver_library
    assert (home_directory / "lib64" / "libcuda.so").readlink() == driver_library


@posix_only
def test_build_pip_cuda_home_requires_driver_library(tmp_path, monkeypatch) -> None:
    toolkit_directory = make_toolkit(tmp_path / "nvidia" / "cu13", with_nvcc=True)
    monkeypatch.setattr(gpu_compat, "find_driver_library", lambda: None)

    home_directory = gpu_compat.build_pip_cuda_home(
        toolkit_directory, tmp_path / "cuda-home"
    )

    assert home_directory is None
    assert not (tmp_path / "cuda-home").exists()


def test_derive_pip_cuda_home_respects_configured_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_compat, "find_pip_cuda_toolkit", lambda: tmp_path / "nvidia" / "cu13"
    )

    assert gpu_compat.derive_pip_cuda_home({"CUDA_HOME": "/opt/cuda"}) is None
    assert gpu_compat.derive_pip_cuda_home({"CUDA_PATH": "/opt/cuda"}) is None


def test_derive_pip_cuda_home_skips_system_toolkit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gpu_compat.shutil, "which", lambda _name: "/usr/bin/nvcc")
    monkeypatch.setattr(
        gpu_compat, "find_pip_cuda_toolkit", lambda: tmp_path / "nvidia" / "cu13"
    )

    assert gpu_compat.derive_pip_cuda_home({}) is None


def test_derive_pip_cuda_home_skips_usr_local_toolkit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gpu_compat.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu_compat, "SYSTEM_CUDA_HOME_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        gpu_compat, "find_pip_cuda_toolkit", lambda: tmp_path / "nvidia" / "cu13"
    )

    assert gpu_compat.derive_pip_cuda_home({}) is None


def test_derive_pip_cuda_home_skips_when_wheels_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gpu_compat.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu_compat, "SYSTEM_CUDA_HOME_DIRECTORY", tmp_path / "absent")
    monkeypatch.setattr(gpu_compat, "find_pip_cuda_toolkit", lambda: None)

    assert gpu_compat.derive_pip_cuda_home({}) is None


def test_derive_pip_cuda_home_builds_from_wheels(tmp_path, monkeypatch) -> None:
    toolkit_directory = tmp_path / "nvidia" / "cu13"
    toolkit_directory.mkdir(parents=True)
    monkeypatch.setattr(gpu_compat.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu_compat, "SYSTEM_CUDA_HOME_DIRECTORY", tmp_path / "absent")
    monkeypatch.setattr(gpu_compat, "find_pip_cuda_toolkit", lambda: toolkit_directory)
    monkeypatch.setattr(
        gpu_compat,
        "build_pip_cuda_home",
        lambda toolkit, home: home if toolkit == toolkit_directory else None,
    )

    derived = gpu_compat.derive_pip_cuda_home({})

    assert derived is not None
    assert derived.name == "cuda-home-cu13"


def test_apply_gpu_compat_env_defaults_sets_cuda_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_compat, "derive_pip_cuda_home", lambda _env: tmp_path / "cuda-home"
    )

    environment: dict[str, str] = {}
    overrides = gpu_compat.apply_gpu_compat_env_defaults(environment)

    assert environment["CUDA_HOME"] == str(tmp_path / "cuda-home")
    assert overrides["CUDA_HOME"] == str(tmp_path / "cuda-home")


def test_apply_gpu_compat_env_defaults_keeps_configured_home(monkeypatch) -> None:
    monkeypatch.setattr(gpu_compat, "derive_pip_cuda_home", lambda _env: None)

    environment: dict[str, str] = {"CUDA_HOME": "/opt/cuda"}
    gpu_compat.apply_gpu_compat_env_defaults(environment)

    assert environment["CUDA_HOME"] == "/opt/cuda"
