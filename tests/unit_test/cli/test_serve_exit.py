# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# note (Richard Wang): run_launcher stands in for a server that stopped. The
# object in the script's globals reports interpreter teardown.
SERVE_THAT_STOPS = """
import atexit, sys, threading, time
from sglang_omni.cli import serve
from sglang_omni.serve import launcher

marker = sys.argv[1]

def note(text):
    with open(marker, "a") as fh:
        fh.write(text)

class Teardown:
    # Module globals are cleared before this runs at teardown.
    def __del__(self, open=open, marker=marker):
        with open(marker, "a") as fh:
            fh.write("teardown,")

def fail():
    raise RuntimeError("thread exit callback failed")

def run_launcher(*args, **kwargs):
    threading.Thread(target=lambda: (time.sleep(0.2), note("thread,"))).start()
    atexit.register(note, "atexit,")
    if sys.argv[2:] == ["failing"]:
        threading._register_atexit(fail)
    sys.stdout.write("unflushed")

teardown = Teardown()
launcher.launch_server = run_launcher
serve.launch_server(None)
"""


@pytest.mark.parametrize("callbacks", ["passing", "failing"])
def test_serve_exits_without_interpreter_teardown(
    tmp_path: Path, callbacks: str
) -> None:
    marker = tmp_path / "marker.txt"
    command = [sys.executable, "-c", SERVE_THAT_STOPS, str(marker), callbacks]

    done = subprocess.run(command, capture_output=True, text=True, timeout=300)

    # A failing thread exit callback is reported and keeps the exit code, as
    # at a normal exit.
    assert done.returncode == 0, done.stderr
    assert "unflushed" in done.stdout
    if callbacks == "failing":
        assert "thread exit callback failed" in done.stderr
    else:
        # Threads finish before the atexit callbacks, as at a normal exit.
        assert marker.read_text() == "thread,atexit,"


# A process that sent a CUDA tensor to another one keeps a reference counter
# file in shared memory, named after its pid.
SEND_CUDA_TENSOR = """
import inspect, json, os
from multiprocessing import reduction
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import reductions
from sglang_omni.utils.process import exit_without_teardown

reference_files = []

def reduce_and_record(tensor):
    rebuild, arguments = reductions.reduce_tensor(tensor)
    reference_counter_handle = (
        inspect.signature(rebuild).bind(*arguments).arguments["ref_counter_handle"]
    )
    reference_files.append(
        "/dev/shm/" + os.fsdecode(reference_counter_handle).lstrip("/")
    )
    return rebuild, arguments

def receive(queue, done):
    tensor = queue.get()
    tensor.sum().item()
    del tensor
    done.set()

if __name__ == "__main__":
    reduction.register(torch.Tensor, reduce_and_record)
    ctx = mp.get_context("spawn")
    queue, done = ctx.Queue(), ctx.Event()
    child = ctx.Process(target=receive, args=(queue, done))
    child.start()
    queue.put(torch.ones(4, device="cuda"))
    done.wait(120)
    child.join(120)
    print(json.dumps({path: os.path.exists(path) for path in reference_files}))
    exit_without_teardown(0)
"""


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_exit_removes_the_cuda_ipc_reference_file(tmp_path: Path) -> None:
    """A fast exit removes the reference counter file that the sent tensor's transfer names; other files in the shared /dev/shm with the same pid do not count."""
    script = tmp_path / "send_cuda_tensor.py"
    script.write_text(SEND_CUDA_TENSOR)

    sender = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )

    assert sender.returncode == 0, sender.stderr
    reference_files = json.loads(sender.stdout)
    assert list(reference_files.values()) == [True]
    assert [path for path in reference_files if Path(path).exists()] == []
