# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni import apple_runtime
from sglang_omni.models.qwen3_omni import request_builders as qwen_request_builders
from sglang_omni.models.qwen3_omni import stages as qwen_stages
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import build_sglang_thinker_request
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer, make_qwen_state


@pytest.mark.parametrize("enabled", [False, True])
def test_apple_backend_selection_uses_current_sglang_mlx_runtime(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime

    monkeypatch.setattr(runtime, "use_mlx", lambda: enabled)

    assert apple_runtime.qwen3_omni_uses_mlx_backend() is enabled


def _set_mps(monkeypatch: pytest.MonkeyPatch, enabled: bool = True) -> None:
    monkeypatch.setattr(
        apple_runtime.current_platform,
        "is_mps",
        lambda: enabled,
    )


def test_apple_checkpoint_requires_native_mlx_before_loading(monkeypatch):
    _set_mps(monkeypatch)
    with pytest.raises(ValueError, match="SGLANG_USE_MLX=1"):
        apple_runtime.validate_qwen3_omni_apple_checkpoint(
            "nonexistent-checkpoint", speech_enabled=True, use_mlx=False
        )


def _write_checkpoint_index(
    root: Path,
    keys: list[str],
    *,
    architecture: str = "Qwen3OmniMoeForConditionalGeneration",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": [architecture]}),
        encoding="utf-8",
    )
    weight_map = {key: "model-00001-of-00001.safetensors" for key in keys}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    return root


def _write_indexed_safetensors(
    directory: Path,
    tensors: dict[str, torch.Tensor],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, directory / shard_name)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard_name for key in tensors}}),
        encoding="utf-8",
    )


@pytest.fixture
def torch_speech_checkpoint_with_split_code2wav(tmp_path: Path) -> Path:
    root = tmp_path / "torch-speech-split-code2wav"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    _write_indexed_safetensors(
        root,
        {
            "thinker.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "talker.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight": torch.ones(
                1
            ),
        },
    )
    _write_indexed_safetensors(
        root / "code2wav",
        {"pre_transformer.layers.0.input_layernorm.weight": torch.ones(1)},
    )
    return root


def test_apple_thinker_profile_forces_single_request_eager_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps(monkeypatch)

    actual = apple_runtime.apply_qwen3_omni_apple_profile(
        {"max_running_requests": 64, "disable_cuda_graph": False, "tp_size": 1},
        explicit_overrides={},
        stage_name="thinker",
    )

    assert actual["max_running_requests"] == 1
    assert actual["disable_cuda_graph"] is True
    assert actual["disable_decode_cuda_graph"] is True
    assert actual["disable_overlap_schedule"] is True
    assert actual["disable_radix_cache"] is True
    assert actual["enable_torch_compile"] is False
    assert actual["enable_mixed_chunk"] is False
    assert actual["chunked_prefill_size"] == -1
    assert actual["sampling_backend"] == "pytorch"


def test_apple_profile_rejects_explicit_unsupported_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps(monkeypatch)

    with pytest.raises(ValueError, match="tp_size=1"):
        apple_runtime.apply_qwen3_omni_apple_profile(
            {"tp_size": 2},
            explicit_overrides={},
            stage_name="talker_ar",
        )
    with pytest.raises(ValueError, match="does not support disable_cuda_graph=False"):
        apple_runtime.apply_qwen3_omni_apple_profile(
            {"tp_size": 1},
            explicit_overrides={"disable_cuda_graph": False},
            stage_name="thinker",
        )
    with pytest.raises(ValueError, match="does not support max_running_requests=2"):
        apple_runtime.apply_qwen3_omni_apple_profile(
            {"tp_size": 1},
            explicit_overrides={"max_running_requests": 2},
            stage_name="thinker",
        )


def test_non_mps_profile_is_a_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_mps(monkeypatch, False)
    overrides = {"tp_size": 2, "disable_cuda_graph": False}

    assert (
        apple_runtime.apply_qwen3_omni_apple_profile(
            overrides,
            explicit_overrides={"disable_cuda_graph": False},
            stage_name="thinker",
        )
        == overrides
    )


def test_apple_thinker_factory_applies_profile_after_tp_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps(monkeypatch)
    captured: dict[str, object] = {}

    class StopAfterOverrides(RuntimeError):
        pass

    def capture_server_args(model_path: str, **kwargs: object) -> object:
        del model_path
        captured.update(kwargs)
        raise StopAfterOverrides

    monkeypatch.setattr(qwen_stages, "build_sglang_server_args", capture_server_args)

    with pytest.raises(StopAfterOverrides):
        qwen_stages.create_sglang_thinker_executor_from_config(
            "unused",
            tp_size=1,
            server_args_overrides={},
        )

    assert captured["tp_size"] == 1
    assert captured["max_running_requests"] == 1
    assert captured["disable_cuda_graph"] is True
    assert captured["enable_mixed_chunk"] is False


def test_fresh_apple_import_initializes_inductor_before_sglang_stubs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sglang_omni.platforms;"
                "import sglang_omni.models.qwen3_omni.request_builders"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_importing_sglang_first_still_lets_sglang_omni_platforms_import() -> None:
    """SGLang installs a stub ``triton`` package on Apple, after which importing"""

    completed = subprocess.run(
        [sys.executable, "-c", "import sglang; import sglang_omni.platforms"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_scheduling_backend_import_never_fails_inside_the_apple_prewarm() -> None:
    """The scheduling backend package runs the same prewarm at import time."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sglang; import sglang_omni.scheduling.sglang_backend",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert "apple_torch_compat" not in completed.stderr, completed.stderr


def test_scheduling_backend_import_prewarms_inductor_before_sglang_stubs() -> None:
    """The supported order -- ``sglang_omni`` first -- must keep working, so the
    prewarm still wins the race for every real entry point."""

    completed = subprocess.run(
        [sys.executable, "-c", "import sglang_omni.scheduling.sglang_backend"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_inductor_prewarm_swallows_only_its_own_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must cover exactly the prewarm import, and must be"""

    from sglang_omni.utils import apple_torch_compat

    calls: list[str] = []

    def _boom(name: str):
        calls.append(name)
        raise TypeError("unsupported operand type(s) for |: 'module' and 'type'")

    monkeypatch.setattr(apple_torch_compat, "platform", _DarwinArm64Platform())
    monkeypatch.setattr(
        apple_torch_compat.torch.backends.mps, "is_available", lambda: True
    )
    monkeypatch.setattr(apple_torch_compat, "import_module", _boom)

    apple_torch_compat.prepare_torch_inductor_for_sglang()

    assert calls == ["torch._inductor.runtime.triton_heuristics"]
    with pytest.raises(TypeError):
        apple_torch_compat.import_module("torch._inductor.runtime.triton_heuristics")


def test_inductor_prewarm_is_skipped_entirely_off_apple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive pairing for the guard above: the prewarm must stay Apple-only,
    so a non-Darwin host never imports Torch Inductor at all."""

    from sglang_omni.utils import apple_torch_compat

    calls: list[str] = []
    monkeypatch.setattr(apple_torch_compat, "platform", _LinuxX86Platform())
    monkeypatch.setattr(
        apple_torch_compat, "import_module", lambda name: calls.append(name)
    )

    apple_torch_compat.prepare_torch_inductor_for_sglang()

    assert calls == []


class _DarwinArm64Platform:
    @staticmethod
    def system() -> str:
        return "Darwin"

    @staticmethod
    def machine() -> str:
        return "arm64"


class _LinuxX86Platform:
    @staticmethod
    def system() -> str:
        return "Linux"

    @staticmethod
    def machine() -> str:
        return "x86_64"


@pytest.mark.parametrize(
    ("stage_name", "params", "expected"),
    [
        ("thinker", {"temperature": 0.1}, "temperature"),
        ("thinker", {"return_logprob": True}, "return_logprob"),
        ("talker_ar", {"talker_top_k": 50}, "talker_top_k"),
        (
            "talker_ar",
            {"talker_repetition_penalty": 1.05},
            "talker_repetition_penalty",
        ),
    ],
)
def test_apple_request_validation_rejects_non_greedy_options(
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
    params: dict[str, object],
    expected: str,
) -> None:
    _set_mps(monkeypatch)

    with pytest.raises(ValueError, match=expected):
        apple_runtime.validate_qwen3_omni_apple_request(
            params,
            stage_name=stage_name,
        )


def test_apple_request_validation_allows_omitted_greedy_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps(monkeypatch)

    apple_runtime.validate_qwen3_omni_apple_request({}, stage_name="thinker")
    apple_runtime.validate_qwen3_omni_apple_request({}, stage_name="talker_ar")


def test_talker_adapter_uses_greedy_apple_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps(monkeypatch)
    captured: dict[str, object] = {}
    sentinel = object()
    prefill_builder = SimpleNamespace(
        append_text_chunk=lambda *args, **inner_kwargs: None,
        mark_thinker_done=lambda *args, **inner_kwargs: None,
    )

    def fake_build(payload: StagePayload, **kwargs: object) -> object:
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return sentinel

    monkeypatch.setattr(
        qwen_request_builders,
        "_build_talker_request_data",
        fake_build,
    )
    request_builder, *_ = qwen_request_builders.make_talker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(),
        codec_vocab_size=4096,
        prefill_builder=prefill_builder,
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=24,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="talker-defaults",
        request=OmniRequest(inputs="hello", params={}),
        data={},
    )

    assert request_builder(payload) is sentinel
    assert captured["temperature"] == 0.0
    assert captured["top_k"] == -1
    assert captured["top_p"] == 1.0
    assert captured["repetition_penalty"] == 1.0


def test_checkpoint_validation_accepts_official_text_and_speech_layouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    text = _write_checkpoint_index(
        tmp_path / "text",
        ["thinker.model.layers.0.self_attn.q_proj.weight"],
    )
    speech = _write_checkpoint_index(
        tmp_path / "speech",
        [
            "thinker.model.layers.0.self_attn.q_proj.weight",
            "talker.model.layers.0.self_attn.q_proj.weight",
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight",
            "code2wav.pre_transformer.layers.0.weight",
        ],
    )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(text),
        speech_enabled=False,
        use_mlx=True,
    )
    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(speech),
        speech_enabled=True,
        use_mlx=True,
    )


@pytest.mark.parametrize("metadata_key", ["quantization", "quantization_config"])
@pytest.mark.parametrize(
    "pipeline_cls", [Qwen3OmniPipelineConfig, Qwen3OmniSpeechPipelineConfig]
)
def test_apple_pipeline_requires_mlx_before_stage_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata_key: str,
    pipeline_cls,
) -> None:
    _set_mps(monkeypatch)
    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_mlx_backend", lambda: False)
    checkpoint = _write_checkpoint_index(
        tmp_path / "converted",
        [
            "thinker.model.layers.0.weight",
            "talker.model.layers.0.weight",
            "talker.code_predictor.model.layers.0.weight",
            "code2wav.pre_transformer.layers.0.weight",
        ],
    )
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config[metadata_key] = {"bits": 4, "group_size": 64, "mode": "affine"}
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="SGLANG_USE_MLX=1") as exc:
        pipeline_cls(model_path=str(checkpoint))
    assert "SGLANG_USE_MLX=1" in str(exc.value)

    # The same checkpoint remains valid for native MLX.
    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_mlx_backend", lambda: True)
    assert pipeline_cls(model_path=str(checkpoint)).model_path == str(checkpoint)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("suffix", ["scales", "biases"])
def test_mlx_checkpoint_accepts_packed_weights_without_quantization_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    indexed: bool,
    suffix: str,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = _write_checkpoint_index(
        tmp_path / "packed", ["thinker.model.layers.0.weight"]
    )
    tensors = {
        "thinker.model.layers.0.weight": torch.ones(1),
        f"thinker.model.layers.0.{suffix}": torch.ones(1),
    }
    if indexed:
        _write_indexed_safetensors(checkpoint, tensors)
    else:
        (checkpoint / "model.safetensors.index.json").unlink()
        save_file(tensors, checkpoint / "model.safetensors")

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint), speech_enabled=False, use_mlx=True
    )


def test_checkpoint_validation_accepts_official_root_with_indexed_split_code2wav(
    monkeypatch: pytest.MonkeyPatch,
    torch_speech_checkpoint_with_split_code2wav: Path,
) -> None:
    from sglang_omni.models.weight_loader import load_weights_by_prefix

    _set_mps(monkeypatch)

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(torch_speech_checkpoint_with_split_code2wav),
        speech_enabled=True,
        use_mlx=True,
    )

    weights = load_weights_by_prefix(
        str(torch_speech_checkpoint_with_split_code2wav / "code2wav"),
        prefix="",
    )
    assert set(weights) == {"pre_transformer.layers.0.input_layernorm.weight"}


def test_mlx_speech_pipeline_initializes_with_official_root_and_indexed_split_code2wav(
    monkeypatch: pytest.MonkeyPatch,
    torch_speech_checkpoint_with_split_code2wav: Path,
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime as tensor_bridge

    _set_mps(monkeypatch)
    monkeypatch.setenv("SGLANG_USE_MLX", "1")
    tensor_bridge.use_mlx.cache_clear()
    try:
        assert apple_runtime.qwen3_omni_uses_mlx_backend() is True
        config = Qwen3OmniSpeechPipelineConfig(
            model_path=str(torch_speech_checkpoint_with_split_code2wav)
        )
    finally:
        tensor_bridge.use_mlx.cache_clear()

    assert config.model_path == str(torch_speech_checkpoint_with_split_code2wav)
    assert config.code2wav_stage() == "code2wav"


@pytest.mark.parametrize(
    "weight_map",
    [
        {},
        {"metadata.version": "model-00001-of-00001.safetensors"},
    ],
)
def test_checkpoint_validation_rejects_empty_or_unowned_split_code2wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    weight_map: dict[str, str],
) -> None:
    _set_mps(monkeypatch)
    checkpoint = _write_checkpoint_index(
        tmp_path / "invalid-split",
        [
            "thinker.model.layers.0.self_attn.q_proj.weight",
            "talker.model.layers.0.self_attn.q_proj.weight",
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight",
        ],
    )
    code2wav = checkpoint / "code2wav"
    code2wav.mkdir()
    (code2wav / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="code2wav"):
        apple_runtime.validate_qwen3_omni_apple_checkpoint(
            str(checkpoint),
            speech_enabled=True,
            use_mlx=True,
        )


def test_checkpoint_validation_accepts_language_model_thinker_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = _write_checkpoint_index(
        tmp_path / "mlx-community",
        ["thinker.language_model.model.layers.0.self_attn.q_proj.weight"],
    )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint),
        speech_enabled=False,
        use_mlx=True,
    )


def test_checkpoint_validation_accepts_quantized_root_code2wav_for_mlx_speech(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = _write_checkpoint_index(
        tmp_path / "mlx-community-speech",
        [
            "thinker.language_model.model.layers.0.self_attn.q_proj.weight",
            "talker.model.layers.0.self_attn.q_proj.weight",
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight",
            "code2wav.pre_transformer.layers.0.mlp.down_proj.weight",
            "code2wav.pre_transformer.layers.0.mlp.down_proj.scales",
            "code2wav.pre_transformer.layers.0.mlp.down_proj.biases",
        ],
    )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint),
        speech_enabled=False,
        use_mlx=True,
    )
    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint),
        speech_enabled=True,
        use_mlx=True,
    )


def test_checkpoint_validation_ignores_unindexed_stale_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = _write_checkpoint_index(
        tmp_path / "indexed",
        ["thinker.language_model.model.layers.0.weight"],
    )
    save_file(
        {"talker.model.layers.0.weight": torch.ones(1)},
        checkpoint / "stale.safetensors",
    )

    with pytest.raises(ValueError, match="talker"):
        apple_runtime.validate_qwen3_omni_apple_checkpoint(
            str(checkpoint),
            speech_enabled=True,
            use_mlx=True,
        )


def test_checkpoint_validation_accepts_mixed_component_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = tmp_path / "mixed"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    thinker = checkpoint / "thinker"
    thinker.mkdir()
    (thinker / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": ("model-00001-of-00001.safetensors")
                }
            }
        ),
        encoding="utf-8",
    )
    talker = checkpoint / "talker"
    talker.mkdir()
    save_file(
        {
            "model.layers.0.weight": torch.ones(1),
            "code_predictor.model.layers.0.weight": torch.ones(1),
        },
        talker / "model.safetensors",
    )
    code2wav = checkpoint / "code2wav"
    code2wav.mkdir()
    save_file(
        {"pre_transformer.layers.0.weight": torch.ones(1)},
        code2wav / "model.safetensors",
    )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint),
        speech_enabled=True,
        use_mlx=True,
    )


def test_checkpoint_validation_ignores_directories_named_safetensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = _write_checkpoint_index(
        tmp_path / "indexed",
        ["thinker.language_model.model.layers.0.weight"],
    )
    (checkpoint / ".downloads" / "partial.safetensors").mkdir(parents=True)

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint),
        speech_enabled=False,
        use_mlx=True,
    )


def test_checkpoint_validation_accepts_quantized_single_file_code2wav_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    checkpoint = tmp_path / "single"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    save_file(
        {
            "thinker.language_model.model.layers.0.weight": torch.ones(1),
            "talker.model.layers.0.weight": torch.ones(1),
            "talker.code_predictor.model.layers.0.weight": torch.ones(1),
            "code2wav.linear.weight": torch.ones(1),
            "code2wav.linear.scales": torch.ones(1),
            "code2wav.linear.biases": torch.ones(1),
        },
        checkpoint / "model.safetensors",
    )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(checkpoint),
        speech_enabled=True,
        use_mlx=True,
    )


def test_checkpoint_validation_accepts_component_local_mlx_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    root = tmp_path / "mlx"
    (root / "config.json").parent.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    component_keys = {
        "thinker": ["model.layers.0.weight"],
        "talker": [
            "model.layers.0.weight",
            "code_predictor.model.layers.0.weight",
        ],
        "code2wav": ["pre_transformer.layers.0.weight"],
    }
    for component, keys in component_keys.items():
        component_dir = root / component
        component_dir.mkdir()
        (component_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "weights.safetensors" for key in keys}}),
            encoding="utf-8",
        )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        str(root),
        speech_enabled=True,
        use_mlx=True,
    )


def test_checkpoint_validation_rejects_unrelated_component_local_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    root = tmp_path / "mlx-invalid"
    (root / "config.json").parent.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    thinker_dir = root / "thinker"
    thinker_dir.mkdir()
    (thinker_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"metadata.version": "weights.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="thinker"):
        apple_runtime.validate_qwen3_omni_apple_checkpoint(
            str(root),
            speech_enabled=False,
            use_mlx=True,
        )


def test_checkpoint_validation_discovers_remote_component_indexes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    root = tmp_path / "remote-cache"
    (root / "config.json").parent.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3OmniMoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    files = ["config.json"]
    for component, keys in {
        "thinker": ["model.layers.0.weight"],
        "talker": [
            "model.layers.0.weight",
            "code_predictor.model.layers.0.weight",
        ],
        "code2wav": ["pre_transformer.layers.0.weight"],
    }.items():
        relative = f"{component}/model.safetensors.index.json"
        files.append(relative)
        path = root / relative
        path.parent.mkdir()
        path.write_text(
            json.dumps({"weight_map": {key: "weights.safetensors" for key in keys}}),
            encoding="utf-8",
        )

    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo_id: files)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename: str(root / filename),
    )

    apple_runtime.validate_qwen3_omni_apple_checkpoint(
        "org/qwen3-omni-mlx",
        speech_enabled=True,
        use_mlx=True,
    )


@pytest.mark.parametrize(
    ("architecture", "keys", "speech_enabled", "expected"),
    [
        ("WrongArchitecture", ["thinker.model.weight"], False, "architecture"),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            ["talker.model.weight"],
            False,
            "thinker",
        ),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            ["thinker.model.weight"],
            True,
            "talker",
        ),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            ["thinker.model.weight", "talker.model.weight"],
            True,
            "code2wav",
        ),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            [
                "thinker.model.weight",
                "talker.model.weight",
                "code2wav.model.weight",
            ],
            True,
            "predictor",
        ),
    ],
)
def test_checkpoint_validation_rejects_incompatible_layouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    keys: list[str],
    speech_enabled: bool,
    expected: str,
) -> None:
    _set_mps(monkeypatch)
    path = _write_checkpoint_index(
        tmp_path / expected,
        keys,
        architecture=architecture,
    )

    with pytest.raises(ValueError, match=expected):
        apple_runtime.validate_qwen3_omni_apple_checkpoint(
            str(path),
            speech_enabled=speech_enabled,
            use_mlx=True,
        )


def test_pipeline_construction_validates_real_model_paths_on_mps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_mps(monkeypatch)
    observed: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(
        apple_runtime,
        "validate_qwen3_omni_apple_checkpoint",
        lambda model_path, *, speech_enabled, use_mlx: observed.append(
            (model_path, speech_enabled, use_mlx)
        ),
    )
    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_mlx_backend", lambda: True)

    Qwen3OmniPipelineConfig(model_path=str(tmp_path))
    Qwen3OmniSpeechPipelineConfig(model_path=str(tmp_path))

    assert observed == [
        (str(tmp_path), False, True),
        (str(tmp_path), True, True),
    ]


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_accepts_several_runs_of_one_modality(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    """Two images (or two audio clips, or two videos) put two *separate* runs of"""

    data = _build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([token_id, token_id, 10, 11, token_id, token_id]),
        model_inputs={model_input_key: torch.ones((4, 4))},
        request_id="rid-multi",
    )

    positions = data.req._omni_mm_positions[modality]
    assert positions.tolist() == [0, 1, 4, 5]


def test_thinker_request_accepts_two_images_and_two_audio_clips_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleaved modalities, each with more than one run, stay separable."""

    data = _build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([55, 10, 77, 11, 55, 12, 77]),
        model_inputs={
            "image_embeds": torch.ones((2, 4)),
            "audio_embeds": torch.ones((2, 4)),
        },
        request_id="rid-mixed",
    )

    positions = data.req._omni_mm_positions
    assert positions["image"].tolist() == [0, 4]
    assert positions["audio"].tolist() == [2, 6]
    assert positions["video"].tolist() == []


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_rejects_a_modality_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    """Placement is by position, so a row/placeholder mismatch would silently
    drop or misplace encoder rows. It has to fail loudly instead."""

    with pytest.raises(
        ValueError,
        match=rf"{modality}.*rid-mismatch.*placeholders=2 encoder rows=3",
    ):
        _build_apple_thinker_request(
            monkeypatch,
            input_ids=torch.tensor([token_id, 10, token_id]),
            model_inputs={model_input_key: torch.ones((3, 4))},
            request_id="rid-mismatch",
        )


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_rejects_placeholders_without_encoder_rows(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    """Without encoder rows the merge leaves the placeholder's *text* embedding
    in place and the model silently generates from noise."""

    del model_input_key
    with pytest.raises(
        ValueError,
        match=rf"{modality}.*rid-missing.*placeholders=1 encoder rows=0",
    ):
        _build_apple_thinker_request(
            monkeypatch,
            input_ids=torch.tensor([token_id, 10]),
            model_inputs={},
            request_id="rid-missing",
        )


@pytest.mark.parametrize(
    ("modality", "token_id", "model_input_key"),
    [
        ("image", 55, "image_embeds"),
        ("video", 66, "video_embeds"),
        ("audio", 77, "audio_embeds"),
    ],
)
def test_thinker_request_rejects_encoder_rows_without_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    token_id: int,
    model_input_key: str,
) -> None:
    del token_id
    with pytest.raises(
        ValueError,
        match=rf"{modality}.*rid-orphan.*placeholders=0 encoder rows=2",
    ):
        _build_apple_thinker_request(
            monkeypatch,
            input_ids=torch.tensor([10, 11]),
            model_inputs={model_input_key: torch.ones((2, 4))},
            request_id="rid-orphan",
        )


def test_thinker_request_counts_chunked_encoder_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encoder merges may concatenate list-valued rows, so the count
    check has to sum the chunks rather than look at the first one."""

    data = _build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([55, 10, 55, 55]),
        model_inputs={"image_embeds": [torch.ones((1, 4)), torch.ones((2, 4))]},
        request_id="rid-chunked",
    )

    assert data.req._omni_mm_positions["image"].tolist() == [0, 2, 3]


def test_non_apple_thinker_request_skips_modality_count_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive pairing for the guards above: off Apple the CUDA thinker owns
    this contract, so the request builder must stay a pass-through."""

    data = _build_apple_thinker_request(
        monkeypatch,
        input_ids=torch.tensor([55, 10, 55]),
        model_inputs={"image_embeds": torch.ones((3, 4))},
        request_id="rid-cuda",
        apple=False,
    )

    assert data.req._omni_mm_positions["image"].tolist() == [0, 2]


def _build_apple_thinker_request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_ids: torch.Tensor,
    model_inputs: dict[str, object],
    request_id: str,
    apple: bool = True,
):
    _set_mps(monkeypatch, apple)
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.request_builders._compute_mrope_positions",
        lambda input_ids, model_inputs, thinker_config: (
            torch.zeros((3, input_ids.numel()), dtype=torch.long),
            torch.tensor(0),
        ),
    )
    state: Qwen3OmniPipelineState = make_qwen_state(
        prompt={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        thinker_inputs={"model_inputs": model_inputs},
    )
    return build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 1},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        request_id=request_id,
        thinker_config=SimpleNamespace(
            image_token_id=55,
            video_token_id=66,
            audio_token_id=77,
        ),
    )
