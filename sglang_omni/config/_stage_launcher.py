# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point for pipeline stages.

No torch/CUDA imports here — CUDA_VISIBLE_DEVICES must be set before
any package that pulls in torch gets imported by the spawn child.
"""
import os


def stage_process_entry(config_dict, ready_event):
    gpu_visible = config_dict.get("cuda_visible_devices")
    if gpu_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible

    from sglang_omni.config.mp_runner import _stage_process_entry

    _stage_process_entry(config_dict, ready_event)
