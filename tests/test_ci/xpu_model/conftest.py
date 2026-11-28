# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the Intel XPU serving lane: checkpoints, cards, and servers.

Kept out of tests/test_model/, whose conftest pulls in jiwer and aiohttp, which the XPU install does not carry.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Protocol

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from benchmarks.benchmarker.utils import (
    server_log_file,
    start_server_from_cmd,
    stop_server,
)
from tests.test_ci.xpu_model.model_table import (
    ALLOW_DOWNLOAD_ENV,
    DEVICE_FAMILY_ENV,
    MODEL_ROOTS_ENV,
    XpuModel,
    device_family,
)

logger = logging.getLogger(__name__)

SINGLE_CARD_WAIT = int(os.environ.get("OMNI_XPU_SINGLE_CARD_WAIT", "600"))
OMNI_WAIT = int(os.environ.get("OMNI_XPU_OMNI_WAIT", "2400"))

REQUEST_TIMEOUT = int(os.environ.get("OMNI_XPU_REQUEST_TIMEOUT", "300"))
SPEECH_REQUEST_TIMEOUT = int(os.environ.get("OMNI_XPU_SPEECH_REQUEST_TIMEOUT", "600"))

REQUIRE_FREE_GPU = os.environ.get("REQUIRE_FREE_GPU", "1") == "1"

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
# Every convention a row in the table uses: AuK ships config.yaml, ZONOS2 params.json.
CONFIG_NAMES = ("config.json", "config.yaml", "config.yml", "params.json")

STARTUP_HINTS = (
    (
        "No XPU devices are available",
        "The spawned stage process could not initialize XPU although the parent can. Put a "
        "sitecustomize.py that calls torch.xpu.init() on PYTHONPATH and rerun.",
    ),
    (
        "exit code -8",
        "The stage process took SIGFPE: oneCCL loaded no plugin because CCL_ROOT is unset. "
        "Activate the conda environment rather than invoking its python by path.",
    ),
)


class ClaimCards(Protocol):
    def __call__(self, count: int) -> list[int]: ...


class ServeModel(Protocol):
    def __call__(
        self,
        model: XpuModel,
        checkpoint: str,
        extra_args: list[str],
        env: dict[str, str],
    ) -> contextlib.AbstractContextManager[str]: ...


@pytest.fixture(scope="session")
def xpu_host() -> None:
    """Skip unless a live XPU is visible, so nothing passes on a CPU fallback."""
    if not torch.xpu.is_available():
        pytest.skip("no Intel XPU visible; this lane serves models on the device")
    else:
        pass


@pytest.fixture(scope="session")
def xpu_device_family(xpu_host: None) -> str:
    """The recipe family of the cards this run serves on, read from the first visible card."""
    name = torch.xpu.get_device_name(0)
    family = device_family(name, os.environ.get(DEVICE_FAMILY_ENV, "").strip())
    logger.info(f"serving on {name!r}, using the {family} recipe")
    return family


def mirror_roots() -> list[Path]:
    spec = os.environ.get(MODEL_ROOTS_ENV, "").strip()
    return [Path(root) for root in spec.split(":") if root]


def hub_repo_dir(repo_id: str) -> Path:
    return Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


def main_snapshot_holds_a_checkpoint(repo_id: str) -> bool:
    """An offline load by bare repo id reads the snapshot refs/main names, and only that one."""
    main = hub_repo_dir(repo_id) / "refs" / "main"
    if not main.is_file():
        return False
    else:
        snapshot = hub_repo_dir(repo_id) / "snapshots" / main.read_text().strip()
    return holds_a_checkpoint(snapshot)


def hub_cache_path(repo_id: str, revision: str = "") -> str | None:
    """A cached snapshot of repo_id that holds a servable checkpoint, or None."""
    try:
        listed = snapshot_download(
            repo_id=repo_id, revision=revision or None, local_files_only=True
        )
    except Exception as exc:
        logger.debug(f"{repo_id} has no complete snapshot in the Hub cache: {exc}")
    else:
        # Offline, the library returns the folder main names without reading it.
        if holds_a_checkpoint(Path(listed)):
            return listed
        else:
            pass

    # local_files_only wants every file a repo publishes, and Whisper ships its weights three ways.
    repo_dir = hub_repo_dir(repo_id)
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    elif revision:
        ref = repo_dir / "refs" / revision
        wanted = ref.read_text().strip() if ref.is_file() else revision
        candidates = [path for path in snapshots.iterdir() if path.name == wanted]
    else:
        main = repo_dir / "refs" / "main"
        pinned = main.read_text().strip() if main.is_file() else ""
        candidates = sorted(
            snapshots.iterdir(),
            key=lambda path: (path.name == pinned, path.stat().st_mtime),
            reverse=True,
        )
    for path in candidates:
        if holds_a_checkpoint(path):
            logger.info(f"{repo_id}: serving the cached snapshot at {path}")
            return str(path)
        else:
            continue
    return None


def download(repo_id: str, revision: str = "") -> str:
    return snapshot_download(repo_id=repo_id, revision=revision or None)


def holds_a_checkpoint(path: Path) -> bool:
    """Whether path has a config, at least one weight file and every shard its index names."""
    if not any((path / name).is_file() for name in CONFIG_NAMES):
        return False
    else:
        # Below the top level too: MiniMax-Music3 keeps 56 of its 58 weight files in subdirectories.
        weights = [
            item
            for item in path.rglob("*")
            if item.name.endswith(WEIGHT_SUFFIXES) and item.is_file()
        ]
    for index in path.rglob("*.index.json"):
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        if not all((index.parent / shard).is_file() for shard in shards):
            return False
        else:
            continue
    return bool(weights)


def require_companions(model: XpuModel) -> None:
    """Skip a row whose server also loads a checkpoint by repo id that is not cached."""
    for companion in model.companions:
        if main_snapshot_holds_a_checkpoint(companion):
            continue
        elif os.environ.get(ALLOW_DOWNLOAD_ENV) == "1":
            download(companion)
        else:
            pytest.skip(
                f"{model.key} also loads {companion} by repo id, and the snapshot its refs/main "
                f"names is not a complete checkpoint in the Hub cache. Warm the cache or set "
                f"{ALLOW_DOWNLOAD_ENV}=1."
            )


def resolve_checkpoint(model: XpuModel) -> str:
    """The row's checkpoint from its env override, a mirror root or the Hub cache, in that order."""
    require_companions(model)
    override = os.environ.get(model.checkpoint_env, "").strip()
    if override:
        if not Path(override).exists():
            pytest.fail(f"{model.checkpoint_env}={override!r} does not exist")
        elif not holds_a_checkpoint(Path(override)):
            pytest.fail(
                f"{model.checkpoint_env}={override!r} is not a complete checkpoint: "
                "it lacks a config, a weight file, or a shard its index names"
            )
        else:
            return override
    else:
        pass

    for root in mirror_roots():
        candidate = root / model.mirror_dirname
        if not candidate.exists():
            continue
        elif model.revision:
            logger.warning(
                f"{candidate} is a plain copy with no revision, and {model.key} is pinned "
                f"to {model.revision}; resolving it from the Hub cache instead"
            )
            continue
        elif holds_a_checkpoint(candidate):
            return str(candidate)
        else:
            logger.warning(
                f"{candidate} is not a complete checkpoint (no config, no weights, "
                "or a missing shard); looking further"
            )

    cached = hub_cache_path(model.repo_id, model.revision)
    if cached is not None:
        return cached
    elif os.environ.get(ALLOW_DOWNLOAD_ENV) == "1":
        return download(model.repo_id, model.revision)
    else:
        roots = ", ".join(str(root) for root in mirror_roots()) or "none set"
        pytest.skip(
            f"{model.checkpoint_spec} is in neither the Hub cache nor a mirror ({roots}). "
            f"Warm the cache, point {model.checkpoint_env} at a local copy, set "
            f"{MODEL_ROOTS_ENV}, or set {ALLOW_DOWNLOAD_ENV}=1 to let this test download it."
        )


def render_nodes() -> list[int]:
    """Host card ids this run may claim: every render node, narrowed by an inherited ZE_AFFINITY_MASK."""
    cards = []
    for node in Path("/dev/dri").glob("renderD*"):
        try:
            cards.append(int(node.name[len("renderD") :]) - 128)
        except ValueError:
            continue
    cards = sorted(card for card in cards if card >= 0)

    mask = (os.environ.get("ZE_AFFINITY_MASK") or "").strip()
    if not mask:
        return cards
    else:
        granted = {int(item) for item in mask.split(",") if item.strip().isdigit()}
        return [card for card in cards if card in granted]


def card_busy(card: int) -> bool:
    node = Path(f"/dev/dri/renderD{128 + card}")
    if not node.exists():
        return True
    else:
        pass
    try:
        probe = subprocess.run(
            ["fuser", str(node)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    # Reading the card's name opens every render node, so this process does not count.
    holders = {int(pid) for pid in probe.stdout.split() if pid.isdigit()}
    return bool(holders - {os.getpid()})


def lock_card(card: int) -> IO[bytes] | None:
    """An exclusive advisory flock on the card's render node, or None when another lane holds it.

    It arbitrates between lanes that share one /dev, such as processes on one host; each container gets its own device node, so lanes in separate containers need disjoint ZE_AFFINITY_MASKs.
    """
    node = Path(f"/dev/dri/renderD{128 + card}")
    try:
        handle = node.open("rb")
    except OSError as exc:
        logger.debug(f"cannot open {node} to claim it: {exc}")
        return None
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def require_card_count(count: int) -> list[int]:
    """The cards this run may claim, skipping when there are fewer than the topology needs; takes no lock."""
    cards = render_nodes()
    if len(cards) < count:
        pytest.skip(f"host has {len(cards)} XPU cards, this topology needs {count}")
    else:
        return cards


def claim_free_cards(count: int, held: list[IO[bytes]]) -> list[int]:
    cards = require_card_count(count)

    claimed: list[int] = []
    contended: list[int] = []
    for card in cards:
        if len(claimed) == count:
            break
        elif REQUIRE_FREE_GPU and card_busy(card):
            contended.append(card)
            continue
        else:
            handle = lock_card(card)
        if handle is None:
            contended.append(card)
            continue
        else:
            held.append(handle)
            claimed.append(card)

    if len(claimed) < count:
        for handle in held:
            handle.close()
        held.clear()
        pytest.skip(
            f"only {len(claimed)} of {len(cards)} cards could be claimed "
            f"({claimed}); {contended} are busy or held by another lane, and "
            f"this topology needs {count}. Set REQUIRE_FREE_GPU=0 to ignore "
            "processes already on a card."
        )
    else:
        pass
    return claimed


@pytest.fixture
def claim_cards() -> Iterator[ClaimCards]:
    """Claim cards for one test and release them at teardown; a row that cannot claim them skips."""
    held: list[IO[bytes]] = []
    try:
        yield lambda count: claim_free_cards(count, held)
    finally:
        for handle in held:
            handle.close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def explained(exc: RuntimeError | TimeoutError) -> RuntimeError | TimeoutError:
    message = str(exc)
    for needle, hint in STARTUP_HINTS:
        if needle in message:
            return type(exc)(f"{message}\n\nLikely cause: {hint}")
        else:
            continue
    return exc


@pytest.fixture
def serve_xpu_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServeModel]:
    """Launch one XPU server per call and stop it at teardown, also when the test fails."""
    started: list[subprocess.Popen[bytes]] = []

    @contextlib.contextmanager
    def serve(
        model: XpuModel, checkpoint: str, extra_args: list[str], env: dict[str, str]
    ) -> Iterator[str]:
        port = free_port()
        log_file = server_log_file(tmp_path_factory, prefix=f"xpu_{model.key}")
        command = [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            checkpoint,
            *extra_args,
        ]
        # Card count does not predict load time, so the row's own window wins over the default.
        default_wait = SINGLE_CARD_WAIT if model.cards == 1 else OMNI_WAIT
        timeout = max(model.startup_seconds, default_wait)
        logger.info(f"serving {model.key} on port {port}: {' '.join(command)}")
        try:
            proc = start_server_from_cmd(
                command,
                log_file,
                port,
                timeout=timeout,
                env=env,
                strip_proxy=True,
                health_path="/v1/models",
                health_body_contains=None,
            )
        except (RuntimeError, TimeoutError) as exc:
            raise explained(exc) from exc
        started.append(proc)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            stop_server(proc)
            started.remove(proc)

    yield serve

    for proc in started:
        stop_server(proc)
