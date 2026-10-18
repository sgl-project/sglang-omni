# SPDX-License-Identifier: Apache-2.0
"""GPU and port isolation and the model server command for Full-Duplex-Bench jobs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from benchmarks.duplex.fdb.common import (
    DEFAULT_JUDGE_PORT,
    DEFAULT_SERVER_PORT,
    JOB_PORT_STRIDE,
    NCCL_PORT_END,
    load_settings,
)
from benchmarks.duplex.fdb.generate import generate
from benchmarks.duplex.fdb.selection import SampleSelection
from benchmarks.duplex.fdb.servers import model_server_command
from sglang_omni.cli import app
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_MAX_SESSIONS,
    MiniCPMODuplexPipelineConfig,
)


@pytest.fixture
def clean_job_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CUDA_VISIBLE_DEVICES", "GPU", "SERVER_PORT", "JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)


def test_default_job_uses_gpu_zero(clean_job_env: None) -> None:
    settings = load_settings("isolation")
    assert settings.gpu == "0"
    assert settings.server_port == DEFAULT_SERVER_PORT
    assert settings.judge_port == DEFAULT_JUDGE_PORT
    assert settings.judge_nccl_port == DEFAULT_JUDGE_PORT + 1
    assert settings.judge_dir.name == f"port-{DEFAULT_JUDGE_PORT}"


def test_visible_device_derives_a_disjoint_port_block(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    settings = load_settings("isolation")
    assert settings.gpu == "1"
    assert settings.server_port == DEFAULT_SERVER_PORT + JOB_PORT_STRIDE
    assert settings.judge_port == DEFAULT_JUDGE_PORT + JOB_PORT_STRIDE
    assert settings.judge_dir.name == f"port-{settings.judge_port}"


def test_explicit_ports_override_the_gpu_offset(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("SERVER_PORT", "8097")
    monkeypatch.setenv("JUDGE_PORT", "30000")
    settings = load_settings("isolation")
    assert settings.server_port == DEFAULT_SERVER_PORT
    assert settings.judge_port == DEFAULT_JUDGE_PORT


def test_the_first_visible_device_is_the_job_gpu(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    settings = load_settings("isolation")
    assert settings.gpu == "1"
    assert settings.server_port == DEFAULT_SERVER_PORT + JOB_PORT_STRIDE
    for visible in ("0,0", "GPU-973ab323", "0,"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        with pytest.raises(SystemExit, match="distinct numeric GPU indices"):
            load_settings("isolation")


def test_gpu_and_visible_device_must_agree(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GPU", "0")
    with pytest.raises(SystemExit, match="disagree"):
        load_settings("isolation")


def test_gpu_without_visible_devices_still_offsets_ports(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU", "1")
    settings = load_settings("isolation")
    assert settings.gpu == "1"
    assert settings.server_port == DEFAULT_SERVER_PORT + JOB_PORT_STRIDE


def test_generate_refuses_more_shards_than_the_server_admits(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FDB_WORK", str(tmp_path / "fdb"))
    settings = load_settings("isolation")
    # note (luojiaxuan): every refusal comes before a directory or server exists.
    for num_shards, gpus, message in (
        (
            DEFAULT_MAX_SESSIONS + 1,
            ("0",),
            "over 1 GPU\\(s\\) exceeds the model server's max_sessions",
        ),
        (
            2 * DEFAULT_MAX_SESSIONS + 1,
            ("0", "1"),
            "over 2 GPU\\(s\\) exceeds the model server's max_sessions",
        ),
        (1, ("0", "1"), "leaves model servers idle"),
    ):
        with pytest.raises(SystemExit, match=f"--num-shards {num_shards} {message}"):
            generate(
                settings,
                1,
                SampleSelection(per_subset=1),
                SampleSelection(per_subset=0),
                num_shards,
                gpus,
            )
    monkeypatch.setenv("SERVER_PORT", "9000")
    with pytest.raises(SystemExit, match="SERVER_PORT gives every model server"):
        generate(
            load_settings("isolation"),
            1,
            SampleSelection(per_subset=1),
            SampleSelection(per_subset=0),
            2,
            ("0", "1"),
        )
    assert not (tmp_path / "fdb").exists()


def test_each_gpu_gets_its_own_port_and_caches(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FDB_WORK", str(tmp_path / "fdb"))
    settings = load_settings("isolation")
    replica = settings.on_gpu("3")
    assert (replica.gpu, replica.server_port) == (
        "3",
        DEFAULT_SERVER_PORT + 3 * JOB_PORT_STRIDE,
    )
    assert replica.judge_port == settings.judge_port
    assert replica.cache_env()["TRITON_CACHE_DIR"] == str(
        tmp_path / "fdb" / "cache" / "gpu-3" / "triton"
    )
    assert settings.on_gpu(settings.gpu) == settings


def test_judge_ports_stay_above_the_nccl_claim_range() -> None:
    for gpu_index in range(8):
        judge_port = DEFAULT_JUDGE_PORT + gpu_index * JOB_PORT_STRIDE
        assert judge_port >= NCCL_PORT_END
        assert judge_port + 1 >= NCCL_PORT_END


def test_port_inside_nccl_range_is_rejected(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_PORT", "29600")
    with pytest.raises(SystemExit, match="NCCL"):
        load_settings("isolation")


def test_model_server_command_launches_the_duplex_pipeline(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sglang_omni.config.manager.resolve_config_cls_for_model_path",
        lambda model_path: MiniCPMOSpeechPipelineConfig,
    )
    launch_server = Mock()
    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", launch_server)
    server_command = model_server_command(load_settings("isolation"))

    result = CliRunner().invoke(app, server_command[server_command.index("serve") :])

    assert result.exit_code == 0, result.output
    assert type(launch_server.call_args.args[0]) is MiniCPMODuplexPipelineConfig
