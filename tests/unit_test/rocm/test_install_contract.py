# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the pinned ROCm images.
    import tomli as tomllib


def test_rocm_manifest_has_no_cuda_only_dependencies() -> None:
    manifest = tomllib.loads(Path("pyproject_rocm.toml").read_text())
    dependencies = "\n".join(manifest["project"]["dependencies"]).lower()

    for forbidden in (
        "flashinfer",
        "flash-attn",
        "nixl-cu",
        "mooncake-transfer-engine-cuda",
        "nvidia-",
    ):
        assert forbidden not in dependencies


def test_rocm_manifest_keeps_platform_stack_external() -> None:
    manifest = tomllib.loads(Path("pyproject_rocm.toml").read_text())
    names = {
        dependency.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].lower()
        for dependency in manifest["project"]["dependencies"]
    }

    assert not {"torch", "torchvision", "torchaudio", "sglang", "kernels"} & names


def test_rocm_image_pins_ucx_and_nixl_commits() -> None:
    dockerfile = Path("docker/rocm.Dockerfile").read_text()

    assert "UCX_REF=8a6b06fb880accbb933a79cda893883872c68d9d" in dockerfile
    assert "NIXL_REF=c0a1102b94d173049a5478c23e765ba37681e2ca" in dockerfile
    assert "--with-rocm=/opt/rocm" in dockerfile
    assert "-Denable_plugins=UCX" in dockerfile
    assert "-Drocm_path" not in dockerfile
    assert "pip uninstall -y nixl nixl-cu12 nixl-cu13" in dockerfile
    assert 'rm -rf "${nixl_python_site}/nixl"' in dockerfile
    assert "ln -sT" in dockerfile
    assert "Path(_api.__file__).resolve().is_relative_to" in dockerfile


def test_rocm_base_image_and_gpu_arch_are_required_build_arguments() -> None:
    dockerfile = Path("docker/rocm.Dockerfile").read_text()

    assert "ARG SGLANG_IMAGE\n" in dockerfile
    assert "ARG GPU_ARCH\n" in dockerfile
    assert "ARG SGLANG_IMAGE=" not in dockerfile
    assert "ARG GPU_ARCH=" not in dockerfile


def test_rocm_install_uses_manifest_dependency_helper() -> None:
    dockerfile = Path("docker/rocm.Dockerfile").read_text()
    installer = Path("scripts/rocm/install_rocm.sh").read_text()

    command = "scripts/rocm/install_dependencies.py"
    assert command in dockerfile
    assert command in installer
    assert "COPY pyproject_rocm.toml ./pyproject.toml" in dockerfile
    assert "COPY pyproject_rocm.toml pyproject.toml README.md" not in dockerfile
    assert "-r pyproject_rocm.toml" not in dockerfile
    assert 'uv pip install --no-deps -r "${manifest}"' not in installer
    assert "qwen-tts==0.1.1" in dockerfile
    assert "qwen-tts==0.1.1" in installer


def test_rocm_media_dependencies_avoid_incompatible_torchcodec_wheels() -> None:
    manifest = tomllib.loads(Path("pyproject_rocm.toml").read_text())
    dependencies = set(manifest["project"]["dependencies"])
    dockerfile = Path("docker/rocm.Dockerfile").read_text()

    assert not any(dependency.startswith("torchcodec") for dependency in dependencies)
    assert "pip uninstall -y torchcodec" in dockerfile
    assert "ffmpeg" in dockerfile
    assert "sox" in dockerfile


def test_audar_option_preserves_the_base_image_torchao_build() -> None:
    dockerfile = Path("docker/rocm.Dockerfile").read_text()

    assert "neucodec==0.0.6 torchtune==0.6.1 torchdata==0.11.0" in dockerfile
    assert "python3 -m pip install --no-deps" in dockerfile
    assert "from llama_cpp import Llama; from neucodec import NeuCodec" in dockerfile


def test_rocm_ci_uses_image_environment_without_reinstalling_metadata() -> None:
    workflow = Path(".github/workflows/rocm-ci.yaml").read_text()

    assert "uv pip install --system" not in workflow
    assert "run_rocm_unit_smoke.sh" in workflow
    assert "run_rocm_model_e2e.py" in workflow
    assert "SGLANG_OMNI_HF_CACHE" in workflow
    assert "/data/cache" not in workflow
