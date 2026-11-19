# SPDX-License-Identifier: Apache-2.0
"""Load the packaged SenseNova model and tokenizer."""
from __future__ import annotations

import json
import logging
import math

import torch
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import cached_file

logger = logging.getLogger(__name__)

CHECKPOINT_INDEX_FILE_NAME = "model.safetensors.index.json"
# Denoising 2048x2048 with guidance peaked 1.4 GiB above the weights on an Arc
# Pro B60, so an eighth of a device covers activations at the supported sizes.
DEVICE_ACTIVATION_RESERVE_FRACTION = 0.125


def build_spanning_device_map(
    model_path: str,
    num_hidden_layers: int,
    first_device: torch.device,
    revision: str | None,
) -> dict[str, torch.device] | None:
    """Place the backbone across consecutive devices, or return None if one holds it.

    SenseNova-U1.5-8B-MoT carries a second expert branch for image tokens, so its
    bf16 checkpoint needs 32.7 GiB and does not fit a 24 GiB device.
    """
    if first_device.index is None:
        return None
    else:
        pass

    index_path = cached_file(model_path, CHECKPOINT_INDEX_FILE_NAME, revision=revision)
    with open(index_path) as index_stream:
        checkpoint_bytes = int(json.load(index_stream)["metadata"]["total_size"])

    device_count = torch.get_device_module(first_device).device_count()
    spanned_indexes: list[int] = []
    spanned_bytes = 0
    for device_index in range(first_device.index, device_count):
        spanned_indexes.append(device_index)
        spanned_bytes += int(
            current_platform.get_device_total_memory(device_index)
            * (1.0 - DEVICE_ACTIVATION_RESERVE_FRACTION)
        )
        if spanned_bytes >= checkpoint_bytes:
            break
        else:
            pass
    if len(spanned_indexes) == 1:
        return None
    elif spanned_bytes < checkpoint_bytes:
        raise RuntimeError(
            f"SenseNova-U1 weights need {checkpoint_bytes / 2**30:.1f} GiB but "
            f"{first_device.type} devices {spanned_indexes} offer "
            f"{spanned_bytes / 2**30:.1f} GiB."
        )
    else:
        pass

    first_spanned_device = torch.device(first_device.type, spanned_indexes[0])
    last_spanned_device = torch.device(first_device.type, spanned_indexes[-1])
    device_map = {
        "vision_model": first_spanned_device,
        "language_model.model.embed_tokens": first_spanned_device,
        "language_model.model.norm": last_spanned_device,
        "language_model.model.norm_mot_gen": last_spanned_device,
        "language_model.lm_head": last_spanned_device,
        # Denoising subtracts the noise latent, which is created next to the
        # prefix hidden states, from the flow-matching head output.
        "fm_modules": last_spanned_device,
    }
    layers_per_device = math.ceil(num_hidden_layers / len(spanned_indexes))
    for layer_index in range(num_hidden_layers):
        device_map[f"language_model.model.layers.{layer_index}"] = torch.device(
            first_device.type, spanned_indexes[layer_index // layers_per_device]
        )
    return device_map


def load_model_and_tokenizer(
    model_path: str,
    server_args: ServerArgs,
) -> dict[str, torch.nn.Module | PreTrainedTokenizerBase]:
    model_dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.model_precision, torch.bfloat16
    )
    load_options: dict[str, str | bool] = {}
    if server_args.trust_remote_code:
        load_options["trust_remote_code"] = True
    else:
        pass
    if server_args.revision is not None:
        load_options["revision"] = server_args.revision
    else:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_path, **load_options)
    model_device = get_local_torch_device()
    current_platform.set_device(model_device)
    checkpoint_config = AutoConfig.from_pretrained(model_path, **load_options)
    device_map = build_spanning_device_map(
        model_path,
        checkpoint_config.llm_config.num_hidden_layers,
        model_device,
        server_args.revision,
    )
    if device_map is None:
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=model_dtype, **load_options
        ).eval()
        model = model.to(model_device)
    else:
        logger.info(
            f"Spanning SenseNova-U1 over {model_device.type} devices "
            f"{sorted({device.index for device in device_map.values()})}"
        )
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=model_dtype, device_map=device_map, **load_options
        ).eval()
    return {"model": model, "tokenizer": tokenizer}
