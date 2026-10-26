"""Flash cache."""

import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    npu_fia_available,
)

# Modified for SGLang; see this directory's README.md for upstream source.


def copy_right_aligned_prefix_bnsd(
    destination: torch.Tensor, source: torch.Tensor, lengths: list[int]
) -> None:
    prefix_width = source.shape[2]
    for batch_index, length in enumerate(lengths):
        destination[batch_index, :, prefix_width - length : prefix_width].copy_(
            source[batch_index, :, :length]
        )


def prepare_flash_kv_cache(
    past_key_values: DynamicCache | None,
    current_len: int,
    batch_size: int,
    prefix_lengths: torch.Tensor | None = None,
) -> None:
    """Allocate prefix and image-token KV buffers in the backend layout."""
    if past_key_values is None:
        return
    else:
        pass

    lengths = None
    if prefix_lengths is not None:
        lengths = [int(length) for length in prefix_lengths.reshape(-1).tolist()]
        if len(lengths) == 1 and batch_size > 1:
            lengths *= batch_size
        else:
            pass
        if len(lengths) != batch_size:
            raise ValueError(
                f"Expected {batch_size} prefix lengths, got {len(lengths)}"
            )
        else:
            pass
    else:
        pass

    key_padding_mask: torch.Tensor | None = None
    for layer in past_key_values.layers:
        past_k = layer.keys
        past_v = layer.values

        if past_k is None or past_v is None:
            layer.flash_prefix_len = 0
            layer.flash_total_len = current_len
            layer.flash_k_cache = None
            layer.flash_v_cache = None
            layer.flash_actual_seq_lengths_kv = None
            layer.flash_kv_padding_size = None
            layer.flash_key_padding_mask = None
            layer.flash_cache_layout = None
            continue
        else:
            pass

        prefix_len = past_k.shape[2]
        total_len = prefix_len + current_len
        if lengths is not None and any(
            length < 0 or length > prefix_len for length in lengths
        ):
            raise ValueError(
                f"Prefix lengths must be between 0 and {prefix_len}, got {lengths}"
            )
        else:
            pass

        use_npu_fia = (
            past_k.device.type == "npu" and lengths is not None and npu_fia_available()
        )
        if use_npu_fia:
            k_cache = torch.empty(
                (batch_size, past_k.shape[1], total_len, past_k.shape[3]),
                device=past_k.device,
                dtype=past_k.dtype,
            )
            v_cache = torch.empty(
                (batch_size, past_v.shape[1], total_len, past_v.shape[3]),
                device=past_v.device,
                dtype=past_v.dtype,
            )
            copy_right_aligned_prefix_bnsd(k_cache, past_k, lengths)
            copy_right_aligned_prefix_bnsd(v_cache, past_v, lengths)
            layer.flash_actual_seq_lengths_kv = [
                length + current_len for length in lengths
            ]
            layer.flash_kv_padding_size = torch.zeros(
                1, dtype=torch.int64, device=past_k.device
            )
            layer.flash_key_padding_mask = None
            layer.flash_cache_layout = "BNSD"
        else:
            past_k_flash = past_k.transpose(1, 2).contiguous()
            past_v_flash = past_v.transpose(1, 2).contiguous()
            k_cache = torch.empty(
                (batch_size, total_len, past_k_flash.shape[2], past_k_flash.shape[3]),
                device=past_k_flash.device,
                dtype=past_k_flash.dtype,
            )
            v_cache = torch.empty(
                (batch_size, total_len, past_v_flash.shape[2], past_v_flash.shape[3]),
                device=past_v_flash.device,
                dtype=past_v_flash.dtype,
            )
            k_cache[:, :prefix_len].copy_(past_k_flash)
            v_cache[:, :prefix_len].copy_(past_v_flash)
            layer.flash_actual_seq_lengths_kv = None
            layer.flash_kv_padding_size = None
            layer.flash_cache_layout = "BSND"
            # note (Wes): One key mask shared by all layers lets padded rows attend in a single call; NPU without FIA gets an explicit mask from denoise_images instead.
            if (
                lengths is None
                or past_k.device.type == "npu"
                or min(lengths) == prefix_len
            ):
                layer.flash_key_padding_mask = None
            else:
                if key_padding_mask is None:
                    key_positions = torch.arange(total_len, device=past_k.device)
                    valid_prefix_lengths = torch.tensor(lengths, device=past_k.device)
                    key_padding_mask = (
                        (key_positions < valid_prefix_lengths[:, None])
                        | (key_positions >= prefix_len)
                    )[:, None, None, :]
                else:
                    pass
                layer.flash_key_padding_mask = key_padding_mask

        layer.flash_prefix_len = prefix_len
        layer.flash_total_len = total_len
        layer.flash_k_cache = k_cache
        layer.flash_v_cache = v_cache


def clear_flash_kv_cache(past_key_values: DynamicCache | None) -> None:
    if past_key_values is None:
        return
    else:
        pass
    for layer in past_key_values.layers:
        if hasattr(layer, "flash_prefix_len"):
            delattr(layer, "flash_prefix_len")
        else:
            pass
        if hasattr(layer, "flash_total_len"):
            delattr(layer, "flash_total_len")
        else:
            pass
        if hasattr(layer, "flash_k_cache"):
            delattr(layer, "flash_k_cache")
        else:
            pass
        if hasattr(layer, "flash_v_cache"):
            delattr(layer, "flash_v_cache")
        else:
            pass
        if hasattr(layer, "flash_actual_seq_lengths_kv"):
            delattr(layer, "flash_actual_seq_lengths_kv")
        else:
            pass
        if hasattr(layer, "flash_kv_padding_size"):
            delattr(layer, "flash_kv_padding_size")
        else:
            pass
        if hasattr(layer, "flash_key_padding_mask"):
            delattr(layer, "flash_key_padding_mask")
        else:
            pass
        if hasattr(layer, "flash_cache_layout"):
            delattr(layer, "flash_cache_layout")
        else:
            pass
