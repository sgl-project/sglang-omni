# SPDX-License-Identifier: Apache-2.0
"""Load the packaged SenseNova model and tokenizer."""
from __future__ import annotations

import json
import logging

import torch
from safetensors import safe_open
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)
from transformers.utils import cached_file

from sglang_omni.vendor.sensenova_u1.configuration_neo_chat import NEOMoELLMConfig

logger = logging.getLogger(__name__)

CHECKPOINT_INDEX_FILE_NAME = "model.safetensors.index.json"
# Denoising 2048x2048 with guidance peaked 1.4 GiB above the weights on an Arc
# Pro B60, so an eighth of a device covers activations at the supported sizes.
DEVICE_ACTIVATION_RESERVE_FRACTION = 0.125


def read_shard_dtype(
    model_path: str, weight_map: dict[str, str], revision: str | None
) -> torch.dtype:
    """Read the storage dtype of the first shard, for checkpoints whose config omits it."""
    shard_path = cached_file(
        model_path, next(iter(weight_map.values())), revision=revision
    )
    with safe_open(shard_path, framework="pt") as shard:
        # Slicing to zero elements reads the header's dtype without any weights.
        return shard.get_slice(next(iter(shard.keys())))[0:0].dtype


def build_spanning_device_map(
    model_path: str,
    llm_config: PretrainedConfig,
    checkpoint_dtype: torch.dtype | None,
    model_dtype: torch.dtype,
    first_device: torch.device,
    revision: str | None,
) -> dict[str, torch.device] | None:
    """Place the backbone across the devices it needs, or return None if one holds it.

    SenseNova-U1.5-8B-MoT carries a second expert branch for image tokens, so its
    bf16 checkpoint needs 32.7 GiB and does not fit a 24 GiB device.
    """
    if first_device.index is None:
        return None
    elif current_platform.device_shares_host_memory():
        # Free device memory there omits page cache the kernel can reclaim, so a
        # warm cache would refuse a placement the single-device load still fits.
        return None
    else:
        pass

    index_path = cached_file(
        model_path,
        CHECKPOINT_INDEX_FILE_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if index_path is None:
        return None
    else:
        pass
    with open(index_path) as index_stream:
        checkpoint_index = json.load(index_stream)
    checkpoint_bytes = int(checkpoint_index["metadata"]["total_size"])
    if checkpoint_dtype is None:
        checkpoint_dtype = read_shard_dtype(
            model_path, checkpoint_index["weight_map"], revision
        )
    else:
        pass
    resident_bytes = (
        checkpoint_bytes * model_dtype.itemsize // checkpoint_dtype.itemsize
    )

    device_module = torch.get_device_module(first_device)
    free_bytes, _ = device_module.mem_get_info(first_device.index)
    if free_bytes >= resident_bytes:
        return None
    else:
        pass
    if isinstance(llm_config, NEOMoELLMConfig):
        raise RuntimeError(
            "SenseNova-U1 sizes a span from the dense backbone's uniform layers, "
            "and the MoE backbone mixes dense and sparse ones. Serve it on a "
            "device that holds the whole checkpoint."
        )
    else:
        pass
    num_hidden_layers = llm_config.num_hidden_layers
    # The embeddings sit on the first device and the head on the last, so hold that
    # much back on every candidate rather than averaging it into each layer. What
    # remains divides evenly because the dense backbone's layers are identical, and
    # the share it leaves over covers the smaller modules that ride along.
    endpoint_bytes = (
        llm_config.vocab_size * llm_config.hidden_size * model_dtype.itemsize
    )
    layer_bytes = -(-(resident_bytes - 2 * endpoint_bytes) // num_hidden_layers)
    layer_capacity_by_device_index: dict[int, int] = {}
    for device_index in range(first_device.index, device_module.device_count()):
        # Free memory, not capacity, so a card another stage already holds adds nothing.
        free_bytes, _ = device_module.mem_get_info(device_index)
        budget_bytes = (
            int(free_bytes * (1.0 - DEVICE_ACTIVATION_RESERVE_FRACTION))
            - endpoint_bytes
        )
        if budget_bytes >= layer_bytes:
            layer_capacity_by_device_index[device_index] = budget_bytes // layer_bytes
        else:
            pass
        if sum(layer_capacity_by_device_index.values()) >= num_hidden_layers:
            break
        else:
            pass
    spanned_layer_capacity = sum(layer_capacity_by_device_index.values())
    if spanned_layer_capacity < num_hidden_layers:
        raise RuntimeError(
            f"SenseNova-U1 weights need {resident_bytes / 2**30:.1f} GiB but "
            f"{first_device.type} devices {sorted(layer_capacity_by_device_index)} hold only "
            f"{spanned_layer_capacity} of {num_hidden_layers} layers."
        )
    elif list(layer_capacity_by_device_index) == [first_device.index]:
        return None
    else:
        pass

    # Even shares within those capacities, so the first device does not carry the
    # whole span while a later one sits half empty.
    layers_by_index: dict[int, int] = {}
    placed_layers = 0
    for position, (device_index, layer_capacity) in enumerate(
        layer_capacity_by_device_index.items()
    ):
        remaining_devices = len(layer_capacity_by_device_index) - position
        even_layer_share = -(-(num_hidden_layers - placed_layers) // remaining_devices)
        layers_by_index[device_index] = min(even_layer_share, layer_capacity)
        placed_layers += layers_by_index[device_index]
    for device_index, layer_capacity in layer_capacity_by_device_index.items():
        additional_layer_count = min(
            layer_capacity - layers_by_index[device_index],
            num_hidden_layers - placed_layers,
        )
        layers_by_index[device_index] += additional_layer_count
        placed_layers += additional_layer_count

    spanned_indexes = list(layers_by_index)
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
    layer_index = 0
    for device_index, layer_count in layers_by_index.items():
        for _ in range(layer_count):
            device_map[f"language_model.model.layers.{layer_index}"] = torch.device(
                first_device.type, device_index
            )
            layer_index += 1
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
        checkpoint_config.llm_config,
        checkpoint_config.dtype or checkpoint_config.llm_config.dtype,
        model_dtype,
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
