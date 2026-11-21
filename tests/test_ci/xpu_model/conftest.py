# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the Intel XPU serving lane: checkpoints, cards, and servers.

Not under tests/test_model/: that conftest declares ``pytest_plugins =
["tests.utils"]``, which needs jiwer and aiohttp, neither in
pyproject_xpu.toml's core scope, and a subdirectory cannot un-declare it.
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
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pytest

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

if TYPE_CHECKING:
    from subprocess import Popen

logger = logging.getLogger(__name__)

_DEFAULT_MIRROR_ROOTS = ("/data/xpu_models", "/data/ss/xpu_models")

_SINGLE_CARD_WAIT = int(os.environ.get("OMNI_XPU_SINGLE_CARD_WAIT", "600"))
_OMNI_WAIT = int(os.environ.get("OMNI_XPU_OMNI_WAIT", "2400"))

REQUEST_TIMEOUT = int(os.environ.get("OMNI_XPU_REQUEST_TIMEOUT", "300"))
SPEECH_REQUEST_TIMEOUT = int(os.environ.get("OMNI_XPU_SPEECH_REQUEST_TIMEOUT", "600"))

_REQUIRE_FREE_GPU = os.environ.get("REQUIRE_FREE_GPU", "1") == "1"


@pytest.fixture(scope="session")
def xpu_host() -> None:
    """Skip unless a live XPU is visible, so nothing passes on a CPU fallback."""
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip("no Intel XPU visible; this lane serves models on the device")


@pytest.fixture(scope="session")
def xpu_device_family(xpu_host: None) -> str:
    """The recipe family of the cards this run serves on.

    Read from the first visible card: a runner's cards are one model, and the
    lane already refuses to start on a host it cannot see.
    """
    import torch

    name = torch.xpu.get_device_name(0)
    family = device_family(name, os.environ.get(DEVICE_FAMILY_ENV, "").strip())
    logger.info(f"serving on {name!r}, using the {family} recipe")
    return family


def mirror_roots() -> list[Path]:
    spec = os.environ.get(MODEL_ROOTS_ENV, "").strip()
    roots = spec.split(":") if spec else list(_DEFAULT_MIRROR_ROOTS)
    return [Path(root) for root in roots if root]


def hub_cache_path(repo_id: str) -> str | None:
    """A cached snapshot of ``repo_id``, or None.

    A complete snapshot first, then a usable one. ``local_files_only`` wants
    every file the repo publishes, which is stricter than serving: Whisper
    ships its weights as safetensors, bin and ONNX, the loader takes one set,
    and so a cache that serves it fine still reads as absent. That cost a model
    its coverage in this lane while the same checkpoint was being served by
    hand.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - huggingface_hub is a core dep
        return None
    try:
        listed = snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception as exc:
        logger.debug(f"{repo_id} has no complete snapshot in the Hub cache: {exc}")
    else:
        # Offline, the library returns the folder main names without reading it.
        if holds_a_checkpoint(Path(listed)):
            return listed
        else:
            pass

    from huggingface_hub.constants import HF_HUB_CACHE

    # The library's own resolution, because HUGGINGFACE_HUB_CACHE outranks
    # HF_HOME and some hosts set the two to different volumes.
    repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    else:
        main = repo_dir / "refs" / "main"
        pinned = main.read_text().strip() if main.is_file() else ""
    # The revision main points at first, then the newest that holds a
    # checkpoint: snapshot names are commit hashes, so their order means nothing.
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


def download(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id)


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
# Every convention a row in the table uses: AuK ships config.yaml, ZONOS2 params.json.
CONFIG_NAMES = ("config.json", "config.yaml", "config.yml", "params.json")


def holds_a_checkpoint(path: Path) -> bool:
    """Whether ``path`` holds a servable checkpoint rather than just a directory.

    Three near misses, each seen on one host. A mirror collects empty
    placeholder directories, which the server died reading. A Hugging Face
    snapshot can hold a config.json and nothing else, when only the config of a
    newer revision was fetched. And an interrupted download leaves some shards
    linked and others absent. So: a config, every shard an index names,
    and at least one weight file -- searched below the top level, since
    MiniMax-Music3 keeps 56 of its 58 weight files in subdirectories.
    """
    if not any((path / name).is_file() for name in CONFIG_NAMES):
        return False
    else:
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


def resolve_checkpoint(model: XpuModel) -> str:
    """Env override, then a mirror root, then the Hub cache. Never downloads
    unless opted into, since an Omni pull is 66 GB."""
    override = os.environ.get(model.checkpoint_env, "").strip()
    if override:
        if not Path(override).exists():
            pytest.fail(f"{model.checkpoint_env}={override!r} does not exist")
        elif not holds_a_checkpoint(Path(override)):
            # Named explicitly, so this is a mistake to report rather than
            # something to look past.
            pytest.fail(
                f"{model.checkpoint_env}={override!r} is not a complete checkpoint: "
                "it lacks a config, a weight file, or a shard its index names"
            )
        else:
            return override

    for root in mirror_roots():
        candidate = root / model.mirror_dirname
        if not candidate.exists():
            continue
        elif holds_a_checkpoint(candidate):
            return str(candidate)
        else:
            logger.warning(
                f"{candidate} is not a complete checkpoint (no config, no weights, "
                "or a missing shard); looking further"
            )

    cached = hub_cache_path(model.repo_id)
    if cached is not None:
        return cached

    if os.environ.get(ALLOW_DOWNLOAD_ENV) == "1":
        return download(model.repo_id)

    roots = ", ".join(str(root) for root in mirror_roots())
    pytest.skip(
        f"{model.repo_id} is in neither the Hub cache nor a mirror ({roots}). "
        f"Warm the cache, point {model.checkpoint_env} at a local copy, or set "
        f"{ALLOW_DOWNLOAD_ENV}=1 to let this test download it."
    )


def render_nodes() -> list[int]:
    """Host card ids this run may claim.

    /dev/dri is whole even when the runner was given a slice of the host, so an
    inherited ZE_AFFINITY_MASK narrows the set: omni-xpu-ci.yaml gives each runner
    a disjoint card pair that way, and claiming outside it would collide with a
    sibling runner that is doing everything else right.
    """
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
    granted = {int(item) for item in mask.split(",") if item.strip().isdigit()}
    return [card for card in cards if card in granted]


def card_busy(card: int) -> bool:
    node = Path(f"/dev/dri/renderD{128 + card}")
    if not node.exists():
        return True
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
    # Not counting this process: reading the card's name initialises Level Zero,
    # which opens every render node, and the lane then refused all eight cards on
    # account of itself.
    holders = {int(pid) for pid in probe.stdout.split() if pid.isdigit()}
    return bool(holders - {os.getpid()})


def lock_card(card: int) -> IO[bytes] | None:
    """An exclusive flock on the card's render node, or None when a lane holds it.

    fuser answers who is on a card *now*, which is a sample, not a reservation:
    two workers polling together both see the same card idle. The lock sits on the
    device node rather than a lock file because that is the one name every claimant
    shares -- a containerized lane has its own /tmp and its own PID namespace, so a
    lock file is invisible to its siblings and so are their processes. It is
    advisory, and it arbitrates between lanes that take it, not against arbitrary
    GPU users.
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


def claim_free_cards(count: int, held: list[IO[bytes]]) -> list[int]:
    cards = render_nodes()
    if len(cards) < count:
        pytest.skip(f"host has {len(cards)} XPU cards, this topology needs {count}")

    claimed: list[int] = []
    contended: list[int] = []
    for card in cards:
        if len(claimed) == count:
            break
        if _REQUIRE_FREE_GPU and card_busy(card):
            contended.append(card)
            continue
        handle = lock_card(card)
        if handle is None:
            contended.append(card)
            continue
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
    return claimed


@pytest.fixture
def claim_cards() -> Iterator[Callable[[int], list[int]]]:
    """Claim cards for one test, releasing them at teardown. A model that cannot
    claim its cards is skipped rather than run into an OOM that reads as a bug."""
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


_STARTUP_HINTS = (
    (
        "No XPU devices are available",
        "The spawned stage process could not initialize XPU although the parent "
        "can. Touch torch.xpu at interpreter start in the child -- put a "
        "sitecustomize.py doing `import torch; torch.xpu.device_count()` on "
        "PYTHONPATH -- then rerun. This is a host/runtime interaction rather "
        "than a pipeline defect: the parent process enumerates every card.",
    ),
    (
        "exit code -8",
        "The stage process took SIGFPE, which on this stack means oneCCL loaded "
        "no plugin because CCL_ROOT is unset. Activate the conda environment "
        "rather than invoking its python by path.",
    ),
)


def explained(exc: BaseException) -> BaseException:
    message = str(exc)
    for needle, hint in _STARTUP_HINTS:
        if needle in message:
            return type(exc)(f"{message}\n\nLikely cause: {hint}")
    return exc


@pytest.fixture
def serve_xpu_model(request: pytest.FixtureRequest, tmp_path_factory) -> Iterator:
    """Launch one XPU server per call, torn down when the test ends.

    Teardown must run even on assertion failure: a killed stage worker leaves the
    xe/GuC exec queue registered and the next test hits a GT reset.
    """
    started: list[Popen] = []

    @contextlib.contextmanager
    def _serve(
        model: XpuModel, checkpoint: str, extra_args: list[str], env: dict[str, str]
    ) -> Iterator[str]:
        port = free_port()
        log_file = server_log_file(tmp_path_factory, prefix=f"xpu_{model.key}")
        cmd = [
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
        # The row's own window, floored by the operator's: card count does not
        # predict load time. Higgs and MiniCPM-o each want one card and well
        # over the single-card default, so deciding on cards alone timed them
        # out before they had finished loading.
        floor = _SINGLE_CARD_WAIT if model.cards == 1 else _OMNI_WAIT
        timeout = max(model.startup_seconds, floor)
        logger.info(f"serving {model.key} on port {port}: {' '.join(cmd)}")
        try:
            proc = start_server_from_cmd(
                cmd,
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

    yield _serve

    for proc in started:
        stop_server(proc)
