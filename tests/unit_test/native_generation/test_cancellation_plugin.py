# SPDX-License-Identifier: Apache-2.0
"""The cancellation plugin's hooks on the installed native multimodal_gen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sglang_omni.native_generation import cancellation

# Runs in a fresh interpreter, because applied hooks patch native classes for
# the rest of the process.
NATIVE_WORKER = r"""
import json, os, sys, tempfile
from importlib.metadata import entry_points
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.platforms import initialize_current_platform

initialize_current_platform()
from sglang.multimodal_gen.runtime.platforms import plugins
from sglang_omni.native_generation import cancellation


def discovered():
    return [entry.name for entry in entry_points(group=plugins.GENERAL_PLUGINS_GROUP)]


inactive = discovered()
directory = tempfile.mkdtemp(dir=sys.argv[1])
# A native worker spawned inside the activation inherits its import path and
# environment, then discovers and loads plugins at bootstrap as below.
with cancellation.activation_environment(directory):
    plugins.load_plugins()
    plugins.apply_plugin_hooks()
    cancellation.announce_worker(
        local_rank=0, rank=0, server_args=SimpleNamespace(num_gpus=1), pipe_writer=None
    )
after = discovered()

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

class LoopStage(PipelineStage):
    def __init__(self, iterate):
        self.server_args = SimpleNamespace(
            enable_layerwise_nvtx_marker=False, comfyui_mode=False
        )
        self.iterate = iterate

    def forward(self, batch, server_args):
        cancel = os.path.join(batch.output_path, cancellation.CANCELLED_MARKER)
        if self.iterate:
            for step in self.progress_bar(range(6), total=6, batch=batch):
                batch.steps.append(step)
                if step == 2:
                    open(cancel, "w").close()
        else:
            with self.progress_bar(total=6, batch=batch) as bar:
                for step in range(6):
                    batch.steps.append(step)
                    if step == 2:
                        open(cancel, "w").close()
                    bar.update()
        return batch

outcomes = {}
for iterate in (True, False):
    for armed in (True, False):
        output = tempfile.mkdtemp(dir=sys.argv[1])
        if armed:
            open(os.path.join(output, cancellation.ARMED_MARKER), "w").close()
        batch = SimpleNamespace(
            output_path=output, is_warmup=False, metrics=None,
            perf_dump_path=None, steps=[],
        )
        try:
            LoopStage(iterate)(batch, None)
        except cancellation.GenerationCancelled:
            batch.steps.append("cancelled")
        outcomes[f"{iterate}-{armed}"] = batch.steps
print(json.dumps({
    "inactive": inactive,
    "after": after,
    "handshake": os.path.exists(cancellation.handshake_path(directory, 0)),
    "outcomes": outcomes,
}))
"""


def test_hooks_stop_only_armed_cancelled_requests(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(repository), os.environ.get("PYTHONPATH", "")]
        ),
        "CUDA_VISIBLE_DEVICES": "",
    }
    env.pop("SGLANG_OMNI_NATIVE_CANCELLATION_DIR", None)
    completed = subprocess.run(
        [sys.executable, "-c", NATIVE_WORKER, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    # Only processes started inside the activation find the plugin.
    assert cancellation.PLUGIN_NAME not in report["inactive"] + report["after"]
    assert report["handshake"]
    for loop in ("True", "False"):
        assert report["outcomes"][f"{loop}-True"] == [0, 1, 2, "cancelled"]
        assert report["outcomes"][f"{loop}-False"] == [0, 1, 2, 3, 4, 5]
