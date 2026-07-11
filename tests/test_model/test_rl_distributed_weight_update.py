# SPDX-License-Identifier: Apache-2.0
"""End-to-end distributed weight-update (refit) test for the RL admin control plane.

Boots a Higgs TTS-4B *instruct* omni server and refits it to the *base* checkpoint
over the admin control plane (``/init_weights_update_group`` +
``/update_weights_from_distributed``), with a rank-0 trainer broadcasting the base
``body.*`` weights over NCCL. Asserts that every refit-changed parameter bit-matches
(SHA256) a server loaded directly from base, and that the server still serves audio.

Parity is checked inference-vs-inference (a base-loaded reference server vs the
refitted server) rather than against the raw checkpoint, because the model renames
and fuses params during ``load_weights`` (397 checkpoint ``body.*`` tensors collapse
to ~226 fused parameters) -- the checkpoint and the served model use different keys.

Requires 2 GPUs + the Higgs 4B base and instruct checkpoints in the HF cache; skipped
otherwise. Run: ``pytest tests/test_model/test_rl_distributed_weight_update.py -s``
"""

from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import requests

from tests.unit_test._util.process import _wait_for, _wait_for_process_line

HF_TTS_MODEL = "bosonai/higgs-audio-v3-tts-4b"
BASE_CACHE_DIRNAME = "models--boson-sglang--higgs-audio-v3-generation-4B-base"
INSTRUCT_CACHE_DIRNAME = "models--bosonai--higgs-audio-v3-tts-4b"
SERVER_PORT = int(os.environ.get("OMNI_E2E_PORT", "8200"))
BASE_REF_PORT = int(os.environ.get("OMNI_E2E_BASE_PORT", "8201"))
MASTER_PORT = int(os.environ.get("OMNI_E2E_MASTER_PORT", "29570"))
GROUP_NAME = "rl_e2e_weight_update_group"


def _hf_hub_root() -> str:
    return os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )


def _resolve_snapshot(cache_dirname: str) -> str | None:
    for snap in glob.glob(
        os.path.join(_hf_hub_root(), cache_dirname, "snapshots", "*")
    ):
        if glob.glob(os.path.join(snap, "*.safetensors")):
            return snap
    return None


def _resolve_base_dir() -> str | None:
    return _resolve_snapshot(BASE_CACHE_DIRNAME)


def _two_gpus() -> bool:
    try:
        import torch

        return torch.cuda.device_count() >= 2
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (
        _two_gpus()
        and _resolve_base_dir()
        and _resolve_snapshot(INSTRUCT_CACHE_DIRNAME)
    ),
    reason="requires 2 GPUs + Higgs 4B-base and 4B-instruct checkpoints in the HF cache",
)


def _run_trainer() -> None:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    import torch
    from safetensors.torch import load_file

    try:
        from sglang.srt.utils import init_custom_process_group
    except Exception:
        from sglang.srt.utils.common import init_custom_process_group

    _, _, port, group, base_dir, prefix, limit, manifest_path = sys.argv
    port, limit = int(port), int(limit)

    torch.cuda.set_device(0)
    state: dict[str, "torch.Tensor"] = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        state.update(load_file(f, device="cuda:0"))
    names = sorted(n for n in state if n.startswith(prefix))
    if limit > 0:
        names = names[:limit]
    manifest = {
        "names": names,
        "dtypes": [str(state[n].dtype).replace("torch.", "") for n in names],
        "shapes": [list(state[n].shape) for n in names],
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh)
    print(f"[trainer] MANIFEST_WRITTEN n={len(names)}", flush=True)

    print("[trainer] RENDEZVOUS_START", flush=True)
    group_handle = init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        world_size=2,
        rank=0,
        group_name=group,
    )
    torch.cuda.synchronize()
    print("[trainer] RENDEZVOUS_OK", flush=True)
    for n in names:
        torch.distributed.broadcast(state[n], src=0, group=group_handle)
    torch.cuda.synchronize()
    print("[trainer] BROADCAST_DONE", flush=True)
    try:
        torch.distributed.destroy_process_group(group_handle)
    except Exception:
        pass


def _boot_server(model_path: str, port: int, gpu: int, timeout: int = 600):
    # Own process group: proc.kill() leaves the per-stage children leaking GPU
    # memory; start_new_session + killpg (see _kill_server) reaps the whole tree.
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        NCCL_CUMEM_ENABLE="0",
        NCCL_NVLS_ENABLE="0",
        HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"),
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            model_path,
            "--port",
            str(port),
        ],
        env=env,
        start_new_session=True,
    )
    url = f"http://localhost:{port}"

    def _ready() -> bool:
        try:
            return requests.get(f"{url}/model_info", timeout=5).status_code == 200
        except Exception:
            return False

    if not _wait_for(_ready, timeout=timeout):
        _kill_server(proc)
        pytest.fail(
            f"omni server ({model_path}) did not become ready within {timeout}s"
        )
    return proc, url


def _kill_server(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def _tts_checksums(url: str) -> dict:
    r = requests.post(
        f"{url}/weights_checker", json={"action": "checksum"}, timeout=600
    )
    r.raise_for_status()
    for res in r.json().get("results", []):
        if res.get("stage") == "tts_engine":
            return res.get("data", {}).get("checksums", {})
    return {}


def test_distributed_refit_matches_base_and_keeps_serving(tmp_path):
    base_dir = _resolve_base_dir()

    proc_b, url_b = _boot_server(base_dir, BASE_REF_PORT, gpu=1)
    try:
        base_ck = _tts_checksums(url_b)
    finally:
        _kill_server(proc_b)
    assert base_ck, "base reference server returned no tts_engine checksums"

    proc_i, url_i = _boot_server(HF_TTS_MODEL, SERVER_PORT, gpu=1)
    trainer = None
    try:
        instruct_ck = _tts_checksums(url_i)
        target = [n for n in base_ck if base_ck.get(n) != instruct_ck.get(n)]
        assert (
            len(target) > 50
        ), f"base and instruct barely differ ({len(target)} params); wrong checkpoints?"

        manifest = str(tmp_path / "manifest.json")
        trainer = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "trainer",
                str(MASTER_PORT),
                GROUP_NAME,
                base_dir,
                "body.",
                "0",
                manifest,
            ],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _wait_for_process_line(trainer, "MANIFEST_WRITTEN", timeout=180)
        _wait_for_process_line(trainer, "RENDEZVOUS_START", timeout=30)
        with open(manifest) as fh:
            spec = json.load(fh)

        r = requests.post(
            f"{url_i}/init_weights_update_group",
            json={
                "master_address": "localhost",
                "master_port": MASTER_PORT,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": GROUP_NAME,
                "backend": "nccl",
                "stages": ["tts_engine"],
            },
            timeout=180,
        )
        assert r.status_code == 200 and r.json()["success"], r.text
        _wait_for_process_line(trainer, "RENDEZVOUS_OK", timeout=180)

        t0 = time.perf_counter()
        r = requests.post(
            f"{url_i}/update_weights_from_distributed",
            json={
                "names": spec["names"],
                "dtypes": spec["dtypes"],
                "shapes": spec["shapes"],
                "group_name": GROUP_NAME,
                "stages": ["tts_engine"],
            },
            timeout=600,
        )
        latency = time.perf_counter() - t0
        assert r.status_code == 200 and r.json()["success"], r.text
        print(
            f"\n[refit] {len(spec['names'])} params updated in {latency:.3f}s",
            flush=True,
        )

        refit_ck = _tts_checksums(url_i)
        changed = [n for n in refit_ck if refit_ck.get(n) != instruct_ck.get(n)]
        assert len(changed) >= len(target) - 32, (
            f"refit changed only {len(changed)} params but base differs from instruct on "
            f"{len(target)}; refit looks incomplete"
        )
        mismatched = [n for n in changed if refit_ck[n] != base_ck.get(n)]
        assert not mismatched, (
            f"{len(mismatched)}/{len(changed)} refit params do not bit-match the base server "
            f"(SHA256 parity failed): {mismatched[:5]}"
        )

        audio = requests.post(
            f"{url_i}/v1/audio/speech",
            json={"input": "After distributed refit."},
            timeout=180,
        )
        assert (
            audio.status_code == 200 and len(audio.content) > 1000
        ), "server stopped serving audio after refit"
        r = requests.post(
            f"{url_i}/destroy_weights_update_group",
            json={"group_name": GROUP_NAME, "stages": ["tts_engine"]},
            timeout=180,
        )
        assert r.status_code == 200 and r.json()["success"], r.text
    finally:
        if trainer is not None:
            try:
                trainer.wait(timeout=120)
            except Exception:
                trainer.kill()
        _kill_server(proc_i)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trainer":
        _run_trainer()
    else:
        raise SystemExit("run via pytest; 'trainer' subcommand is internal")
