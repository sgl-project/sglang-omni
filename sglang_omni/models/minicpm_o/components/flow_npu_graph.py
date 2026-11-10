# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-stream five-step Flow replay with owned outputs."""

import dataclasses
import importlib
import json
import logging
import os
import threading

import torch

from sglang_omni.platforms.device_graph import NpuDeviceGraphBackend

logger = logging.getLogger(__name__)
SUPPORTED_WINDOWS = {
    (20, 322),
    (10, 312),
    (50, 372),
    (50, 422),
    (50, 362),
    (50, 412),
    (50, 452),
    (6, 408),
}


def map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    else:
        pass
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: map_tensors(getattr(value, field.name), function)
                for field in dataclasses.fields(value)
            },
        )
    else:
        pass
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    else:
        pass
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    else:
        pass
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    else:
        pass
    return value


def tensor_list(value):
    result = []
    map_tensors(value, lambda tensor: result.append(tensor))
    return result


def install_flow_npu_graph(model):
    if os.environ.get("SPEECH_FLOW_NPU_GRAPH") != "1":
        return
    else:
        pass
    if model.device.type != "npu":
        raise ValueError("Flow NPU graph requires an NPU model")
    else:
        pass
    if os.environ.get("SPEECH_NPU_DISABLE_INTERNAL_FORMAT") != "1":
        raise ValueError(
            "Set SPEECH_NPU_DISABLE_INTERNAL_FORMAT=1 before model loading"
        )
    else:
        pass
    decoder = model.flow.decoder
    npu_extension = importlib.import_module("torch_npu")
    actual_format = npu_extension._C._npu_getOption(  # noqa: leading-underscore  # torch-npu 2.10 has no public getter.
        "ALLOW_INTERNAL_FORMAT"
    ).decode()
    if actual_format != "disable":
        raise ValueError(
            f"Flow graph requires disabled internal format, actual={actual_format}"
        )
    else:
        pass
    original = decoder.forward_chunk
    backend = NpuDeviceGraphBackend()
    lock = threading.Lock()
    entries = {}
    graph_stream = None
    max_graphs = len(SUPPORTED_WINDOWS)
    disabled = False
    verify = os.environ.get("SPEECH_FLOW_NPU_GRAPH_VERIFY", "1") == "1"
    stats = {"calls": 0, "capture": 0, "replay": 0, "verify_pass": 0, "fallback": 0}
    reasons = {}

    def fallback(reason, values):
        stats["fallback"] += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        if reasons[reason] <= 2 or reasons[reason] % 100 == 0:
            logger.warning(
                f"SPEECH_FLOW_GRAPH_FALLBACK reason={reason} stats={json.dumps(stats)}"
            )
        else:
            pass
        return original(*values)

    def forward_chunk(
        mu,
        speaker_embeddings,
        mel_conditioning,
        n_timesteps=10,
        temperature=1.0,
        convolution_cache=None,
        attention_window=None,
        layout=None,
    ):
        nonlocal disabled, graph_stream
        values = (
            mu,
            speaker_embeddings,
            mel_conditioning,
            n_timesteps,
            temperature,
            convolution_cache,
            attention_window,
            layout,
        )
        with lock:
            stats["calls"] += 1
            if disabled:
                return fallback("disabled", values)
            else:
                pass
            if (
                mu.device.type != "npu"
                or mu.shape[0] != 1
                or n_timesteps != 5
                or layout is not None
                or convolution_cache is None
                or attention_window is None
                or torch.is_grad_enabled()
                or decoder.training
                or torch.is_autocast_enabled("npu")
                or mu.dtype != torch.float32
            ):
                return fallback("unsupported_configuration", values)
            else:
                pass
            shape_key = (mu.shape[2], attention_window.shape[4])
            if shape_key not in SUPPORTED_WINDOWS:
                return fallback("unsupported_shape", values)
            else:
                pass
            sources = tensor_list(values)
            storages = [tensor.untyped_storage().data_ptr() for tensor in sources]
            if len(storages) != len(set(storages)):
                return fallback("shared_input_storage", values)
            else:
                pass
            current_stream = torch.npu.current_stream(mu.device)
            if graph_stream is not None and current_stream != graph_stream:
                return fallback("different_stream", values)
            else:
                pass
            signature = (
                tuple(
                    (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                    for t in sources
                ),
                n_timesteps,
                temperature,
            )
            try:
                if signature not in entries:
                    if len(entries) >= max_graphs:
                        return fallback("graph_capacity", values)
                    else:
                        pass
                    static = map_tensors(values, lambda tensor: tensor.clone())
                    destinations = tensor_list(static)
                    side = torch.npu.Stream()
                    side.wait_stream(current_stream)
                    with torch.npu.stream(side):
                        for _ in range(3):
                            for destination, source in zip(destinations, sources):
                                destination.copy_(source)
                            original(*static)
                    current_stream.wait_stream(side)
                    torch.npu.synchronize()
                    for destination, source in zip(destinations, sources):
                        destination.copy_(source)
                    with backend.capture(pool=backend.graph_pool_handle()) as graph:
                        output = original(*static)
                    entries[signature] = (static, graph, output)
                    graph_stream = current_stream
                    stats["capture"] += 1
                    logger.warning(
                        f"SPEECH_FLOW_GRAPH_CAPTURE frames={shape_key[0]} "
                        f"window={shape_key[1]} graphs={len(entries)}"
                    )
                else:
                    pass
                static, graph, output = entries[signature]
                for destination, source in zip(tensor_list(static), sources):
                    destination.copy_(source)
                graph.replay()
                owned = map_tensors(output, lambda tensor: tensor.clone())
                stats["replay"] += 1
                if verify:
                    reference_inputs = map_tensors(
                        values, lambda tensor: tensor.clone()
                    )
                    reference = original(*reference_inputs)
                    try:
                        for actual, expected in zip(
                            tensor_list(owned), tensor_list(reference), strict=True
                        ):
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-4, atol=1e-4, equal_nan=False
                            )
                        for actual, expected in zip(
                            tensor_list(static),
                            tensor_list(reference_inputs),
                            strict=True,
                        ):
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-4, atol=1e-4, equal_nan=False
                            )
                    except AssertionError:
                        disabled = True
                        attention_window.copy_(reference_inputs[6])
                        logger.exception(
                            "SPEECH_FLOW_GRAPH_VERIFY_FAIL disabled=True fallback=eager_reference"
                        )
                        return reference
                    stats["verify_pass"] += 1
                else:
                    pass
                attention_window.copy_(static[6])
                if stats["replay"] <= 3 or stats["replay"] % 25 == 0:
                    logger.warning(
                        f"SPEECH_FLOW_GRAPH_STATS {json.dumps({**stats, 'reasons': reasons})}"
                    )
                else:
                    pass
                return owned
            except RuntimeError:
                disabled = True
                logger.exception(
                    "SPEECH_FLOW_GRAPH_RUNTIME_ERROR restart_required=True"
                )
                raise

    decoder.forward_chunk = forward_chunk
    logger.warning(
        f"SPEECH_FLOW_GRAPH_ENABLED verify={verify} max_graphs={max_graphs} diagnostics=False target_tails=False"
    )
