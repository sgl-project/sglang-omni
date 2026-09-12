# SPDX-License-Identifier: Apache-2.0
"""Lazy registration and compatibility with standard/custom MLX runners."""
from types import SimpleNamespace

import pytest

from sglang_omni.model_runner import mlx_model_worker as worker


@pytest.mark.parametrize(
    "architecture,module,attribute",
    [
        (
            "Qwen3ASRForConditionalGeneration",
            "sglang_omni.models.qwen3_asr.mlx.runner",
            "make_qwen3_asr_mlx_runner_class",
        ),
        (
            "FishS2ProMlxModel",
            "sglang_omni.models.fishaudio_s2_pro.mlx.runner",
            "make_fish_mlx_runner_class",
        ),
    ],
)
def test_resolution_imports_only_selected_model(
    monkeypatch, architecture, module, attribute
):
    calls = []
    sentinel = object()

    def import_selected(name):
        calls.append(name)
        return SimpleNamespace(**{attribute: sentinel})

    monkeypatch.setattr(worker, "import_module", import_selected)
    assert worker.resolve_mlx_runner_factory(architecture) is sentinel
    assert calls == [module]


@pytest.mark.parametrize("architecture", [None, "UnknownModel"])
def test_unknown_architecture_fails_before_import(monkeypatch, architecture):
    def unexpected_import(name):
        pytest.fail(f"Unexpected import: {name}")

    monkeypatch.setattr(worker, "import_module", unexpected_import)
    with pytest.raises(NotImplementedError, match="supported architectures:"):
        worker.resolve_mlx_runner_factory(architecture)


def test_external_registration(monkeypatch):
    monkeypatch.setattr(worker, "_MLX_RUNNER_FACTORIES", {})
    worker.register_mlx_runner_factory("ExternalModel", "external.runner:factory")
    factory = lambda: object
    monkeypatch.setattr(
        worker, "import_module", lambda name: SimpleNamespace(factory=factory)
    )
    assert worker.resolve_mlx_runner_factory("ExternalModel") is factory


@pytest.mark.parametrize("pool_size", [None, 4096])
def test_standard_runner_preserves_sglang_constructor(pool_size):
    from sglang.srt.runtime_context import get_context

    class Runner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    with get_context().override_server_args(
        model_path="model",
        trust_remote_code=False,
        disable_radix_cache=True,
        mem_fraction_static=0.5,
        quantization=None,
        revision="revision",
        mlx_enable_sampling=False,
        random_seed=42,
        enable_deterministic_inference=True,
        max_total_tokens=pool_size,
    ):
        runner = worker._create_registered_runner(Runner)
    expected = dict(
        model_path="model",
        trust_remote_code=False,
        disable_radix_cache=True,
        mem_fraction_static=0.5,
        quantization=None,
        revision="revision",
        enable_sampling=False,
        sampling_rng_seed=42,
        deterministic_seeding=True,
    )
    if pool_size is not None:
        expected["pool_size"] = pool_size
    assert runner.kwargs == expected


def test_custom_runner_owns_initialization():
    sentinel = object()

    class Runner:
        @classmethod
        def from_runtime_config(cls):
            return sentinel

        def __init__(self, **kwargs):
            pytest.fail("Standard constructor must not run for custom adapter")

    assert worker._create_registered_runner(Runner) is sentinel
