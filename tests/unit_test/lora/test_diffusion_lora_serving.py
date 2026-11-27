# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disk LoRA loading, step-distillation ODE overrides, and diffusion stages."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from sglang_omni.client.client import Client
from sglang_omni.lora.capability import MergedLoRASpec
from sglang_omni.lora.ipc_wire import encode_inline_tensor
from sglang_omni.lora.merged_manager import MergedLoRAManager
from sglang_omni.lora.solver_override import (
    ODESolverFieldMap,
    build_attr_solver_controller,
)
from sglang_omni.lora.weight_transfer import (
    MergedLoRAWorkerSession,
    build_merged_lora_admin_handler,
)
from sglang_omni.model_runner import model_worker as model_worker_mod
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.stages import build_auk_lora_admin_handler
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.admin import (
    ADMIN_FINISH_LORA_UPDATE,
    ADMIN_LOAD_LORA_ADAPTER,
    ADMIN_RESTORE_BASE_WEIGHTS,
    ADMIN_START_LORA_UPDATE,
    ADMIN_UPDATE_LORA_WEIGHTS,
    AdminResponse,
    is_update_action,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.serve.openai_api import create_app


class TinyPackedFlowDIT(nn.Module):
    """Lightweight CPU stand-in mirroring MiniMaxMusic3DIT's packed QKV structure."""

    def __init__(self) -> None:
        super().__init__()
        self.diffusion_transformer = nn.Module()
        self.diffusion_transformer.transformer = nn.Module()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.to_qkv = nn.Linear(16, 48, bias=False)
        layer.attn.to_out = nn.Linear(16, 16, bias=False)
        self.diffusion_transformer.transformer.layers = nn.ModuleList([layer])

    def merged_lora_spec(self) -> MergedLoRASpec:
        return MergedLoRASpec(
            target_module_root="diffusion_transformer",
            packed_modules_mapping={"to_qkv": ("to_q", "to_k", "to_v")},
            strip_prefixes=(
                "dit.diffusion_transformer.",
                "diffusion_transformer.",
                "dit.",
                "model.",
            ),
            allow_plain_linear=True,
        )


def test_load_lora_adapter_action_registered() -> None:
    assert is_update_action(ADMIN_LOAD_LORA_ADAPTER) is True


def test_auk_step_distillation_lora_load_switch_and_restore(tmp_path: Path) -> None:
    torch.manual_seed(11)
    dit = AuKDit(
        dim=16,
        num_layers=1,
        num_single_layers=1,
        heads=2,
        dim_head=8,
        text_hidden_dim=8,
        latent_dim=4,
    )
    base_qkv = dit.transformer_blocks[0].attn.to_qkv.weight.detach().clone()
    base_ptr = dit.transformer_blocks[0].attn.to_qkv.weight.data_ptr()

    sampling: dict[str, object] = {
        "steps": 32,
        "cfg_strength": 4.0,
        "sway_sampling_coef": -1.0,
        "t_grid": None,
    }
    handler = build_auk_lora_admin_handler(dit, sampling)
    scheduler = SimpleScheduler(lambda x: x, admin_handler=handler)

    # 1. Write a Flash-style 4-step distilled LoRA checkpoint with sidecar solver config.
    flash_dir = tmp_path / "auk_flash_lora"
    flash_dir.mkdir()
    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)
    save_file(
        {
            "dit.transformer_blocks.0.attn.to_q.lora_A.weight": lora_a,
            "dit.transformer_blocks.0.attn.to_q.lora_B.weight": lora_b,
        },
        str(flash_dir / "adapter_model.safetensors"),
    )
    (flash_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4.0, "target_modules": ["to_q"]}),
        encoding="utf-8",
    )
    flash_grid = [0.0, 0.35, 0.7, 0.9, 1.0]
    (flash_dir / "ode_solver_config.json").write_text(
        json.dumps(
            {
                "steps": 4,
                "cfg_strength": 1.0,
                "sway_sampling_coef": None,
                "t_grid": flash_grid,
            }
        ),
        encoding="utf-8",
    )

    resp_flash = scheduler.admin(
        ADMIN_LOAD_LORA_ADAPTER,
        {"lora_path": str(flash_dir)},
    )
    assert resp_flash["success"] is True
    assert resp_flash["data"]["adapter_id"] == "auk_flash_lora"
    assert resp_flash["data"]["updated_params"] == 1
    assert dit.transformer_blocks[0].attn.to_qkv.weight.data_ptr() == base_ptr
    expected_flash_q = base_qkv[:16] + 2.0 * (lora_b @ lora_a)
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[:16],
        expected_flash_q,
        atol=1e-5,
    )
    assert sampling["steps"] == 4
    assert sampling["cfg_strength"] == 1.0
    assert sampling["sway_sampling_coef"] is None
    assert sampling["t_grid"] == tuple(flash_grid)

    # 2. Failed disk load with unsupported solver field rolls back atomically.
    bad_dir = tmp_path / "auk_bad_lora"
    bad_dir.mkdir()
    save_file(
        {
            "dit.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 16),
            "dit.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(16, 2),
        },
        str(bad_dir / "adapter_model.safetensors"),
    )
    resp_bad = scheduler.admin(
        ADMIN_LOAD_LORA_ADAPTER,
        {
            "lora_path": str(bad_dir),
            "solver_override": {"solver_method": "unsupported_rk4"},
        },
    )
    assert resp_bad["success"] is False
    # Live weights and sampling stay unchanged on failure.
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[:16],
        expected_flash_q,
        atol=1e-5,
    )
    assert sampling["steps"] == 4

    # 3. Switching to a non-distilled LoRA resets sampling back to pristine base.
    style_dir = tmp_path / "auk_style_lora"
    style_dir.mkdir()
    style_a = torch.randn(2, 16)
    style_b = torch.randn(16, 2)
    save_file(
        {
            "dit.transformer_blocks.0.attn.to_v.lora_A.weight": style_a,
            "dit.transformer_blocks.0.attn.to_v.lora_B.weight": style_b,
        },
        str(style_dir / "adapter_model.safetensors"),
    )
    (style_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2.0, "target_modules": ["to_v"]}),
        encoding="utf-8",
    )
    resp_style = scheduler.admin(
        ADMIN_LOAD_LORA_ADAPTER,
        {"lora_path": str(style_dir)},
    )
    assert resp_style["success"] is True
    # to_q restored to base; to_v merged; sampling reset to 32-step base config.
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[:16],
        base_qkv[:16],
        atol=1e-5,
    )
    assert torch.allclose(
        dit.transformer_blocks[0].attn.to_qkv.weight[32:],
        base_qkv[32:] + (style_b @ style_a),
        atol=1e-5,
    )
    assert sampling["steps"] == 32
    assert sampling["cfg_strength"] == 4.0
    assert sampling["sway_sampling_coef"] == -1.0
    assert sampling["t_grid"] is None

    # 4. Restore base weights restores both model weights and sampling config.
    resp_restore = scheduler.admin(ADMIN_RESTORE_BASE_WEIGHTS, {})
    assert resp_restore["success"] is True
    assert torch.equal(dit.transformer_blocks[0].attn.to_qkv.weight, base_qkv)
    assert sampling["steps"] == 32
    assert sampling["cfg_strength"] == 4.0


def test_streaming_scheduler_packed_dit_lora_and_attr_solver_override(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    dit = TinyPackedFlowDIT()
    base_qkv = (
        dit.diffusion_transformer.transformer.layers[0]
        .attn.to_qkv.weight.detach()
        .clone()
    )
    decoder = SimpleNamespace(dit=dit, dit_steps=50, dit_cfg_scale=7.0)
    manager = MergedLoRAManager(
        dit,
        architecture="MiniMaxMusic3ForConditionalGeneration",
        prefer_pinned=False,
    )
    solver_ctrl = build_attr_solver_controller(
        decoder,
        ODESolverFieldMap(num_steps="dit_steps", cfg_scale="dit_cfg_scale"),
    )
    session = MergedLoRAWorkerSession(manager, solver_controller=solver_ctrl)
    scheduler = StreamingSimpleScheduler(
        compute_fn=None,
        admin_handler=build_merged_lora_admin_handler(session),
    )

    lora_dir = tmp_path / "music3_fast"
    lora_dir.mkdir()
    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)
    save_file(
        {
            "diffusion_transformer.transformer.layers.0.attn.to_k.lora_A.weight": lora_a,
            "diffusion_transformer.transformer.layers.0.attn.to_k.lora_B.weight": lora_b,
        },
        str(lora_dir / "adapter_model.safetensors"),
    )
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 2.0,
                "target_modules": ["to_k"],
                "ode_solver": {"dit_steps": 8, "dit_cfg_scale": 1.5},
            }
        ),
        encoding="utf-8",
    )

    resp = scheduler.admin(
        ADMIN_LOAD_LORA_ADAPTER,
        {"lora_path": str(lora_dir)},
    )
    assert resp["success"] is True
    assert decoder.dit_steps == 8
    assert decoder.dit_cfg_scale == 1.5
    merged_qkv = dit.diffusion_transformer.transformer.layers[0].attn.to_qkv.weight
    assert torch.allclose(
        merged_qkv[16:32], base_qkv[16:32] + (lora_b @ lora_a), atol=1e-5
    )

    # Restore brings both weights and decoder attributes back to defaults.
    restored = scheduler.admin(ADMIN_RESTORE_BASE_WEIGHTS, {})
    assert restored["success"] is True
    assert decoder.dit_steps == 50
    assert decoder.dit_cfg_scale == 7.0
    assert torch.equal(merged_qkv, base_qkv)

    # StreamingSimpleScheduler without admin_handler returns skipped/unsupported.
    plain_streaming = StreamingSimpleScheduler(compute_fn=None)
    skipped = plain_streaming.admin(ADMIN_LOAD_LORA_ADAPTER, {"lora_path": "x"})
    assert skipped["success"] is True
    assert skipped["data"]["unsupported"] is True


def test_minimax_music3_acoustic_scheduler_lora_and_solver_override(
    tmp_path: Path,
) -> None:
    try:
        from sglang_omni.models.minimax_music3.acoustic import (
            MiniMaxMusic3AcousticScheduler,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional minimax_music3 dependency missing: {exc}")

    torch.manual_seed(9)
    dit = TinyPackedFlowDIT()
    base_qkv = (
        dit.diffusion_transformer.transformer.layers[0]
        .attn.to_qkv.weight.detach()
        .clone()
    )
    decoder = SimpleNamespace(
        dit=dit,
        dit_steps=50,
        dit_cfg_scale=7.0,
        serial_offload=False,
    )
    scheduler = MiniMaxMusic3AcousticScheduler(decoder)  # type: ignore[arg-type]

    lora_dir = tmp_path / "music3_acoustic_fast"
    lora_dir.mkdir()
    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)
    save_file(
        {
            "diffusion_transformer.transformer.layers.0.attn.to_k.lora_A.weight": lora_a,
            "diffusion_transformer.transformer.layers.0.attn.to_k.lora_B.weight": lora_b,
        },
        str(lora_dir / "adapter_model.safetensors"),
    )
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 2.0,
                "target_modules": ["to_k"],
                "ode_solver": {"dit_steps": 6, "dit_cfg_scale": 1.25},
            }
        ),
        encoding="utf-8",
    )

    resp = scheduler.admin(ADMIN_LOAD_LORA_ADAPTER, {"lora_path": str(lora_dir)})
    assert resp["success"] is True
    assert decoder.dit_steps == 6
    assert decoder.dit_cfg_scale == 1.25
    merged_qkv = dit.diffusion_transformer.transformer.layers[0].attn.to_qkv.weight
    assert torch.allclose(
        merged_qkv[16:32], base_qkv[16:32] + (lora_b @ lora_a), atol=1e-5
    )

    restored = scheduler.admin(ADMIN_RESTORE_BASE_WEIGHTS, {})
    assert restored["success"] is True
    assert decoder.dit_steps == 50
    assert decoder.dit_cfg_scale == 7.0
    assert torch.equal(merged_qkv, base_qkv)


def test_staged_update_with_inline_solver_override_commit_and_abort() -> None:
    torch.manual_seed(19)
    dit = AuKDit(
        dim=16,
        num_layers=1,
        num_single_layers=1,
        heads=2,
        dim_head=8,
        text_hidden_dim=8,
        latent_dim=4,
    )
    sampling: dict[str, object] = {
        "steps": 32,
        "cfg_strength": 4.0,
        "sway_sampling_coef": -1.0,
        "t_grid": None,
    }
    handler = build_auk_lora_admin_handler(dit, sampling)

    lora_a = torch.randn(2, 16)
    lora_b = torch.randn(16, 2)
    assert handler(ADMIN_START_LORA_UPDATE, {"update_id": "u1"})["success"] is True
    staged = handler(
        ADMIN_UPDATE_LORA_WEIGHTS,
        {
            "update_id": "u1",
            "adapter_id": "distill_ipc",
            "target_modules": ["to_q"],
            "tensors": {
                "dit.transformer_blocks.0.attn.to_q.lora_A.weight": encode_inline_tensor(
                    lora_a
                ),
                "dit.transformer_blocks.0.attn.to_q.lora_B.weight": encode_inline_tensor(
                    lora_b
                ),
            },
            "solver_override": {"steps": 6, "cfg_strength": 1.25},
        },
    )
    assert staged["success"] is True
    # Live sampling not mutated before finish_lora_update.
    assert sampling["steps"] == 32

    finished = handler(ADMIN_FINISH_LORA_UPDATE, {"update_id": "u1"})
    assert finished["success"] is True
    assert sampling["steps"] == 6
    assert sampling["cfg_strength"] == 1.25


def test_model_runner_omni_scheduler_and_http_load_lora_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Shared-weight runner blocks load_lora_adapter.
    shared_runner = SGLModelRunner.__new__(SGLModelRunner)
    shared_runner.weight_share_config = SimpleNamespace(role="follower")
    shared_runner.merged_lora_session = None
    ok, msg, _ = shared_runner.load_lora_adapter({"lora_path": "/tmp/a"})
    assert ok is False
    assert "weight sharing" in msg

    # 2. OmniScheduler + ModelWorker dispatch load_lora_adapter and flush cache.
    epochs: list[int] = []
    flushed: list[int] = []
    version_overrides: list[tuple[str, dict[str, object]]] = []
    recorded_payloads: list[dict[str, Any]] = []

    def fake_override(server_args: object, source: str, **fields: object) -> None:
        version_overrides.append((source, dict(fields)))
        for key, value in fields.items():
            setattr(server_args, key, value)

    monkeypatch.setattr(model_worker_mod, "override_server_args", fake_override)

    class FakeRunner:
        def load_lora_adapter(
            self, payload: dict[str, Any]
        ) -> tuple[bool, str, dict[str, object]]:
            recorded_payloads.append(dict(payload))
            return True, "loaded", {"adapter_id": "disk_lora", "updated_params": 2}

    worker = ModelWorker.__new__(ModelWorker)
    worker.model_runner = FakeRunner()
    worker.server_args = SimpleNamespace(weight_version="0")

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.admin_lock = threading.Lock()
    scheduler.model_worker = worker
    scheduler.enable_overlap = False
    scheduler.async_pending = None
    scheduler._engine_paused = False  # noqa: leading-underscore
    scheduler.last_pause_mode = None
    scheduler.active_request_ids = lambda: []
    scheduler.abort_all_requests = lambda: []
    scheduler.advance_prompt_cache_epoch = lambda: epochs.append(1)
    scheduler.flush_cache_after_update = lambda: flushed.append(1) or True
    scheduler.empty_torch_cache = lambda: None

    res = scheduler.run_admin_action(
        ADMIN_LOAD_LORA_ADAPTER,
        {
            "lora_path": "/models/flash_lora",
            "weight_version": "flash-v1",
            "solver_override": {"num_steps": 4},
        },
    )
    assert res["success"] is True
    assert worker.server_args.weight_version == "flash-v1"
    assert version_overrides == [
        ("sglang-omni-lora-load", {"weight_version": "flash-v1"})
    ]
    assert len(epochs) == 1
    assert len(flushed) == 1
    assert recorded_payloads[0]["lora_path"] == "/models/flash_lora"


@pytest.mark.asyncio
async def test_http_load_lora_adapter_route_and_client() -> None:
    recorded: list[tuple[str, dict[str, Any]]] = []
    coord = Coordinator.__new__(Coordinator)

    async def fake_admin(
        action: str,
        payload: dict[str, object] | None = None,
        *,
        stages: Any = None,
        timeout_s: float = 30.0,
    ) -> AdminResponse:
        recorded.append((action, dict(payload or {})))
        return AdminResponse(
            operation_id="op_load",
            action=action,
            success=True,
            message="ok",
            results={},
        )

    coord.admin = fake_admin
    client = Client.__new__(Client)
    client.coordinator = coord

    await client.load_lora_adapter({"lora_path": "/tmp/lora_a"})
    assert recorded == [(ADMIN_LOAD_LORA_ADAPTER, {"lora_path": "/tmp/lora_a"})]

    recorded.clear()
    app = create_app(client=client, model_name="test-model")
    http = TestClient(app)

    resp = http.post(
        "/load_lora_adapter",
        json={
            "lora_path": "/tmp/flash_lora",
            "adapter_id": "flash",
            "scale": 0.75,
            "solver_override": {"num_steps": 4, "cfg_scale": 1.0},
            "weight_version": "v2",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert recorded[0][0] == ADMIN_LOAD_LORA_ADAPTER
    assert recorded[0][1]["lora_path"] == "/tmp/flash_lora"
    assert recorded[0][1]["adapter_id"] == "flash"
    assert recorded[0][1]["scale"] == 0.75
    assert recorded[0][1]["solver_override"] == {"num_steps": 4, "cfg_scale": 1.0}
    assert recorded[0][1]["weight_version"] == "v2"
