# SPDX-License-Identifier: Apache-2.0
"""Claim-to-execution checks for the mps_dp weight-share support registry.

Every model advertised as weight-share supported must resolve through the real
launcher preflight, every unsupported topology must fail before any resource
is created, and the docs table must agree with the code registries.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from sglang_omni.utils import ipc_weights

REPO_ROOT = Path(__file__).resolve().parents[3]
MPS_DP_DIR = REPO_ROOT / "examples" / "mps_dp"
DOCS_PAGE = REPO_ROOT / "docs" / "basic_usage" / "mps_dp.md"

_spec = importlib.util.spec_from_file_location(
    "mps_dp_config", MPS_DP_DIR / "config.py"
)
mps_dp_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mps_dp_config)

VALIDATED_CONFIG_CLASSES = {
    "HiggsTtsPipelineConfig": HiggsTtsPipelineConfig,
    "MossTTSLocalPipelineConfig": MossTTSLocalPipelineConfig,
    "MossTTSPipelineConfig": MossTTSPipelineConfig,
    "MossTranscribeDiarizePipelineConfig": MossTranscribeDiarizePipelineConfig,
    "Qwen3ASRPipelineConfig": Qwen3ASRPipelineConfig,
    "WhisperASRPipelineConfig": WhisperASRPipelineConfig,
    "FunASRPipelineConfig": FunASRPipelineConfig,
}


def _write_yaml(tmp_path: Path, config_cls: str, name: str = "probe") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"config_cls: {config_cls}\nname: {name}\nmodel_path: dummy/none\n",
        encoding="utf-8",
    )
    return path


def test_registry_matches_weight_share_policies():
    # Note (Jiaxin Deng): implications, not an identity: every validated
    # config's architecture must be gate-enabled and never audit-only, and
    # (under today's binary support rule) every gate-enabled architecture must
    # have at least one validated config. One architecture may back several
    # configs.
    registry = mps_dp_config.WEIGHT_SHARE_VALIDATED_CONFIGS
    assert set(registry) == set(VALIDATED_CONFIG_CLASSES)
    assert set(registry.values()) <= set(ipc_weights.WEIGHT_SHARE_POLICIES)
    assert set(ipc_weights.WEIGHT_SHARE_POLICIES) <= set(registry.values())
    assert not set(registry.values()) & set(
        ipc_weights.AUDIT_ONLY_WEIGHT_SHARE_POLICIES
    )


def test_every_validated_config_class_has_a_drivable_topology():
    for name, cls in VALIDATED_CONFIG_CLASSES.items():
        config = cls(model_path="dummy")
        engine_stages = [
            stage.name
            for stage in config.stages
            if cls.stage_config_cls(stage.name).engine_stage
        ]
        assert (
            len(engine_stages) == 1
        ), f"{name} is not a single-SGLang-engine pipeline: {engine_stages}"


def test_every_shipped_example_config_resolves_with_sharing(tmp_path):
    yamls = sorted((MPS_DP_DIR / "configs").glob("*.yaml"))
    assert yamls, "no example configs shipped"
    for yaml_path in yamls:
        stage_name, value = mps_dp_config.resolve_max_total_tokens(
            yaml_path, require_single_sglang_engine=True, weight_share=True
        )
        assert stage_name
        assert (
            isinstance(value, int) and value > 0
        ), f"{yaml_path.name} does not pin a positive max_total_tokens"


@pytest.mark.parametrize(
    "config_cls", ["LLaDA2UniPipelineConfig", "Qwen3OmniPipelineConfig"]
)
def test_single_engine_pipelines_resolve_but_pin_nothing(tmp_path, config_cls):
    """These pipelines drive one SGLang engine (the thinker), which is the
    launcher's structural requirement; with no max_total_tokens pinned they
    resolve to (stage, None), and launch.sh refuses an unpinned KV budget for
    N > 1 before creating any state."""
    yaml_path = _write_yaml(tmp_path, config_cls)
    stage_name, value = mps_dp_config.resolve_max_total_tokens(yaml_path)
    assert stage_name == "thinker"
    assert value is None


def test_multi_engine_pipeline_fails_the_singleton_check(tmp_path):
    yaml_path = _write_yaml(tmp_path, "Qwen3OmniSpeechPipelineConfig")
    with pytest.raises(ValueError, match="one SGLang engine stage"):
        mps_dp_config.resolve_max_total_tokens(
            yaml_path, require_single_sglang_engine=True
        )


@pytest.mark.parametrize(
    "config_cls",
    ["VoxtralTTSPipelineConfig", "Qwen3TTSPipelineConfig", "MingTTSPipelineConfig"],
)
def test_weight_share_rejected_for_unvalidated_configs(tmp_path, config_cls):
    yaml_path = _write_yaml(tmp_path, config_cls)
    with pytest.raises(ValueError, match="not passed end-to-end validation"):
        mps_dp_config.resolve_max_total_tokens(yaml_path, weight_share=True)


def test_docs_table_matches_the_code_registries():
    text = DOCS_PAGE.read_text(encoding="utf-8")
    supported_rows = [
        line
        for line in text.splitlines()
        if line.startswith("|") and "| Supported" in line
    ]
    for arch in ipc_weights.WEIGHT_SHARE_POLICIES:
        assert any(
            arch in row for row in supported_rows
        ), f"{arch} is gate-supported but has no Supported row in {DOCS_PAGE.name}"
    for arch in ipc_weights.AUDIT_ONLY_WEIGHT_SHARE_POLICIES:
        assert not any(
            arch in row for row in supported_rows
        ), f"{arch} is audit-only but {DOCS_PAGE.name} lists it as Supported"


@pytest.mark.skipif(os.name != "posix", reason="launch.sh needs a POSIX shell")
class TestLaunchFailsClosedBeforeResources:
    def _run(self, tmp_path, yaml_path, **env_extra):
        state_root = tmp_path / "state"
        env = os.environ.copy()
        env.update(
            {
                "STATE_ROOT": str(state_root),
                "CONFIG": str(yaml_path),
                "N": "2",
                "CORE_BLOCKS": "0 1",
                "PYTHON_BIN": sys.executable,
            }
        )
        env.update(env_extra)
        proc = subprocess.run(
            ["bash", str(MPS_DP_DIR / "launch.sh"), "up"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc, state_root

    def test_unpinned_kv_budget_leaves_no_state(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, "LLaDA2UniPipelineConfig")
        proc, state_root = self._run(tmp_path, yaml_path)
        assert proc.returncode != 0
        assert "MAX_TOTAL_TOKENS is required" in proc.stdout + proc.stderr
        assert not state_root.exists()

    def test_unvalidated_weight_share_leaves_no_state(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, "VoxtralTTSPipelineConfig")
        proc, state_root = self._run(tmp_path, yaml_path, WEIGHT_SHARE="1")
        assert proc.returncode != 0
        assert "not passed end-to-end validation" in proc.stdout + proc.stderr
        assert not state_root.exists()

    def test_run_id_traversal_is_rejected(self, tmp_path):
        yaml_path = (
            REPO_ROOT / "examples" / "mps_dp" / "configs" / "higgs_h100_dp3.yaml"
        )
        proc, state_root = self._run(
            tmp_path,
            yaml_path,
            RUN_ID="../gpu-1/run-x",
            MAX_TOTAL_TOKENS="1000",
            BASE_PORT="29411",
        )
        assert proc.returncode != 0
        assert not state_root.exists()
        # Note (Jiaxin Deng): the RUN_ID check sits after the GPU probes, so
        # only assert its message where nvidia-smi exists; without a GPU the
        # launch still fails closed before any state is created.
        if shutil.which("nvidia-smi"):
            assert "RUN_ID must be a single" in proc.stdout + proc.stderr
