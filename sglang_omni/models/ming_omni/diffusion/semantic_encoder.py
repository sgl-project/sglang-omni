# SPDX-License-Identifier: Apache-2.0
"""Semantic text encoder using Ming's LLM + connector.

Loads the BailingMoeV2 LLM, connector (Qwen2), and projection layers
independently from the Ming model directory to produce 2560-dim semantic
embeddings for ZImage diffusion conditioning.

Architecture:
    text -> tokenizer -> input_ids
    -> append 256 query token placeholders (16x16 scale)
    -> replace placeholders with learnable query token embeddings
    -> LLM forward pass (hidden_size=4096)
    -> extract 256 query token hidden states
    -> proj_in: Linear(4096, 1536)
    -> connector: Qwen2 non-causal transformer (28 layers, hidden=1536)
    -> proj_out: Linear(1536, 2560)
    -> L2 normalize
    -> [B, 256, 2560] semantic embeddings
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _find_first_index_of_consecutive_ones(lst: list[int]) -> list[int]:
    """Return the index of the first 1 in each consecutive-ones segment."""
    result = []
    i = 0
    n = len(lst)
    while i < n:
        if lst[i] == 1:
            result.append(i)
            while i < n and lst[i] == 1:
                i += 1
        else:
            i += 1
    return result


def _merge_consecutive_ones(lst: list[int], n: int) -> list[int]:
    """Merge every *n* consecutive 1s into a single 1."""
    result = []
    i = 0
    while i < len(lst):
        if lst[i] == 0:
            result.append(0)
            i += 1
        else:
            count = 0
            while i < len(lst) and lst[i] == 1:
                count += 1
                i += 1
            result.extend([1] * (count // n))
    return result


# Chat template constants — must match Ming's processing_bailingmm2.py
_SYSTEM_PROMPT = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
_USER_PREFIX = "<role>HUMAN</role>"
_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"


class MingSemanticEncoder:
    """Standalone semantic encoder: LLM + connector + projections.

    Loads components independently from the Ming model directory without
    depending on the full BailingMM2 class.

    Usage::

        encoder = MingSemanticEncoder()
        encoder.load(model_path, device)
        pos, neg = encoder.encode("A cat on a windowsill")
        # pos: list of [256, 2560] tensors
        # neg: list of [256, 2560] tensors (zeros)
    """

    def __init__(self) -> None:
        self._llm = None  # BailingMoeV2ForCausalLM
        self._tokenizer = None
        self._connector = None  # Qwen2ForCausalLM (non-causal)
        self._proj_in = None  # Linear(4096, 1536)
        self._proj_out = None  # Linear(1536, 2560)
        self._query_tokens = None  # Parameter(num_tokens, 4096)
        self._device: torch.device | None = None
        self._aux_device: torch.device | None = None  # for connector/proj
        self._dtype: torch.dtype = torch.bfloat16

        # Config values set during load
        self._image_patch_token: int = 0
        self._image_start_token: int = 0
        self._image_end_token: int = 0
        self._img_gen_scales: list[int] = []
        self._scale_indices: list[int] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Load all components from the Ming model directory.

        Args:
            model_path: Path to the Ming model directory containing
                config.json, model-*.safetensors, connector/, mlp/.
            device: Primary device (used for connector/projections).
                The LLM uses device_map="auto" for multi-GPU distribution.
            dtype: Model dtype (default bf16).
        """
        self._device = device
        self._dtype = dtype

        # 1. Read configs
        with open(os.path.join(model_path, "config.json")) as f:
            full_config = json.load(f)
        llm_config_dict = full_config["llm_config"]

        self._image_patch_token = llm_config_dict.get("image_patch_token", 157157)
        self._image_start_token = llm_config_dict.get(
            "image_start_token", self._image_patch_token + 1
        )
        self._image_end_token = self._image_start_token + 1

        with open(os.path.join(model_path, "mlp", "config.json")) as f:
            mlp_config = json.load(f)
        self._img_gen_scales = mlp_config.get("img_gen_scales", [16])

        # Pre-compute cumulative scale indices
        self._scale_indices = []
        current_idx = 0
        for scale in self._img_gen_scales:
            current_idx += scale * scale
            self._scale_indices.append(current_idx)

        # 2. Load LLM (BailingMoeV2ForCausalLM)
        self._load_llm(model_path, llm_config_dict, dtype)

        # Pick auxiliary device for connector/projections.  Avoid the GPU
        # hosting word_embeddings (usually GPU 0) — it needs maximum
        # headroom for the MoE forward pass activations.
        embed_gpu = self._llm_device_map.get("model.word_embeddings")
        best_gpu = device
        best_free = 0
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.mem_get_info(i)[0]
            if i == embed_gpu:
                continue  # skip the main LLM GPU
            if free > best_free:
                best_free = free
                best_gpu = torch.device(f"cuda:{i}")
        # Fallback: if all GPUs were skipped, use the embedding GPU
        if best_free == 0:
            best_free = torch.cuda.mem_get_info(embed_gpu)[0]
            best_gpu = torch.device(f"cuda:{embed_gpu}")
        aux_device = best_gpu
        self._aux_device = aux_device
        logger.info(
            "[SemanticEncoder] Auxiliary device: %s (%.1f GiB free)",
            aux_device,
            best_free / (1 << 30),
        )

        # 3. Load connector (Qwen2ForCausalLM with non-causal attention)
        self._load_connector(model_path, aux_device, dtype)

        # 4. Load projections and query tokens from mlp/model.safetensors
        self._load_projections(model_path, aux_device, dtype)

        logger.info("[SemanticEncoder] All components loaded successfully")

    def _load_llm(
        self,
        model_path: str,
        llm_config_dict: dict,
        dtype: torch.dtype,
    ) -> None:
        """Load BailingMoeV2ForCausalLM from main model safetensors."""
        from sglang_omni.models.ming_omni.diffusion.bailing_moe_config import (
            BailingMoeV2Config,
        )
        from sglang_omni.models.ming_omni.diffusion.bailing_moe_model import (
            BailingMoeV2ForCausalLM,
        )

        logger.info("[SemanticEncoder] Loading BailingMoeV2 LLM...")

        # Build config
        config = BailingMoeV2Config(**llm_config_dict)
        # Use flash_attention_2 (available on remote H200s) or fallback to eager
        try:
            from flash_attn import flash_attn_func  # noqa: F401

            config._attn_implementation = "flash_attention_2"
        except ImportError:
            config._attn_implementation = "eager"

        # Read weight index
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        # Build LLM key mapping: strip "model." prefix from BailingMM2 keys
        weight_map = index["weight_map"]
        llm_key_to_shard = {}
        shards_needed = set()
        for key, shard in weight_map.items():
            if key.startswith("model."):
                new_key = key[len("model.") :]
                llm_key_to_shard[new_key] = (shard, key)
                shards_needed.add(shard)

        logger.info(
            "[SemanticEncoder] LLM: %d keys across %d shards",
            len(llm_key_to_shard),
            len(shards_needed),
        )

        # Instantiate model on meta device (no memory allocation)
        with torch.device("meta"):
            self._llm = BailingMoeV2ForCausalLM(config)

        # Build proportional device map: distribute layers across GPUs
        # proportional to their available free memory.  Even distribution
        # fails on shared machines where some GPUs have less free memory.
        # Each MoE layer is ~6.5 GiB; we reserve headroom for activations.
        from accelerate.utils import set_module_tensor_to_device

        LAYER_SIZE_GIB = 6.5  # MoE layer size in bf16 (256 experts × 3 linear)
        HEADROOM_GIB = 15.0  # reserved for activations + framework overhead
        MIN_FREE_GIB = 30.0  # minimum free memory to consider a GPU

        gpu_count = torch.cuda.device_count()
        gpu_free_gib: dict[int, float] = {}
        for i in range(gpu_count):
            free_mem = torch.cuda.mem_get_info(i)[0]
            free_gib = free_mem / (1 << 30)
            if free_gib > MIN_FREE_GIB:
                gpu_free_gib[i] = free_gib
                logger.info(
                    "[SemanticEncoder] GPU %d: %.1f GiB free",
                    i,
                    free_gib,
                )

        if not gpu_free_gib:
            raise RuntimeError(
                f"No GPUs with >{MIN_FREE_GIB} GiB free memory available"
            )

        n_layers = config.num_hidden_layers

        # Calculate max layers each GPU can hold after reserving headroom
        gpu_max_layers: dict[int, int] = {}
        for gpu_id, free_gib in gpu_free_gib.items():
            usable = free_gib - HEADROOM_GIB
            max_layers = max(1, int(usable / LAYER_SIZE_GIB))
            gpu_max_layers[gpu_id] = max_layers

        total_capacity = sum(gpu_max_layers.values())
        if total_capacity < n_layers:
            raise RuntimeError(
                f"Not enough GPU capacity: {total_capacity} layers across "
                f"{len(gpu_max_layers)} GPUs, need {n_layers}. "
                f"Per-GPU: {gpu_max_layers}"
            )

        # Distribute layers proportionally to each GPU's capacity
        available_gpus = sorted(gpu_max_layers.keys())
        layers_assigned: dict[int, int] = {}
        remaining = n_layers
        for gpu_id in available_gpus:
            share = round(n_layers * gpu_max_layers[gpu_id] / total_capacity)
            share = min(share, remaining)
            layers_assigned[gpu_id] = share
            remaining -= share
        # Distribute any remainder to GPUs with the most headroom
        while remaining > 0:
            for gpu_id in sorted(
                available_gpus,
                key=lambda g: gpu_max_layers[g] - layers_assigned[g],
                reverse=True,
            ):
                if remaining <= 0:
                    break
                if layers_assigned[gpu_id] < gpu_max_layers[gpu_id]:
                    layers_assigned[gpu_id] += 1
                    remaining -= 1

        device_map = {"model.word_embeddings": available_gpus[0]}
        layer_idx = 0
        for gpu_id in available_gpus:
            for _ in range(layers_assigned[gpu_id]):
                device_map[f"model.layers.{layer_idx}"] = gpu_id
                layer_idx += 1
        device_map["model.norm"] = available_gpus[-1]
        device_map["lm_head"] = available_gpus[-1]

        # Save for aux device selection later
        self._llm_device_map = device_map

        # Log distribution
        from collections import Counter

        gpu_layer_counts = Counter(device_map.values())
        logger.info(
            "[SemanticEncoder] Proportional device map: %d modules across "
            "%d GPUs: %s (free_gib=%s)",
            len(device_map),
            len(available_gpus),
            dict(gpu_layer_counts),
            {g: f"{f:.0f}" for g, f in gpu_free_gib.items()},
        )

        # Load weights shard by shard, placing each tensor on its mapped device
        loaded = 0
        for shard_idx, shard_file in enumerate(sorted(shards_needed)):
            shard_path = os.path.join(model_path, shard_file)
            logger.info(
                "[SemanticEncoder] Loading shard %d/%d: %s",
                shard_idx + 1,
                len(shards_needed),
                shard_file,
            )

            from safetensors import safe_open

            with safe_open(shard_path, framework="pt") as f:
                for new_key, (s, orig_key) in llm_key_to_shard.items():
                    if s != shard_file:
                        continue
                    tensor = f.get_tensor(orig_key)

                    # Find the device via longest-prefix match in the device map
                    target_device = self._device
                    best_len = -1
                    for module_name, dev in device_map.items():
                        if (
                            new_key.startswith(module_name)
                            and len(module_name) > best_len
                        ):
                            target_device = dev
                            best_len = len(module_name)

                    set_module_tensor_to_device(
                        self._llm,
                        new_key,
                        target_device,
                        value=tensor,
                        dtype=dtype,
                    )
                    loaded += 1

        # Materialize non-persistent buffers (inv_freq) that aren't in
        # safetensors.  RotaryEmbedding computes inv_freq in __init__, but
        # since the model was created on meta device those buffers are still
        # meta tensors.  Recompute them on CPU before dispatch.
        from sglang_omni.models.ming_omni.diffusion.bailing_moe_model import (
            BailingMoeV2RotaryEmbedding,
            BailingMoeV2RotaryEmbedding3D,
        )

        for name, module in self._llm.named_modules():
            if isinstance(
                module, (BailingMoeV2RotaryEmbedding, BailingMoeV2RotaryEmbedding3D)
            ):
                inv_freq, attn_scaling = module.rope_init_fn(module.config, "cpu")
                # Replace the meta buffer with a real CPU tensor
                module.inv_freq = inv_freq
                module.original_inv_freq = inv_freq
                module.attention_scaling = attn_scaling

        # Set up model hooks for automatic input/output device transfer
        from accelerate import dispatch_model

        self._llm = dispatch_model(self._llm, device_map=device_map)
        self._llm.eval()
        logger.info("[SemanticEncoder] LLM loaded: %d parameters set", loaded)

        # Load tokenizer (use base fast tokenizer — we only need basic
        # text tokenization, not Ming's full BailingTokenizer features)
        from transformers import PreTrainedTokenizerFast

        self._tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        logger.info(
            "[SemanticEncoder] Tokenizer loaded (vocab_size=%d)",
            len(self._tokenizer),
        )

    def _load_connector(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load Qwen2ForCausalLM connector with non-causal attention."""
        from transformers import AutoModelForCausalLM

        logger.info("[SemanticEncoder] Loading connector from %s/connector", model_path)
        self._connector = AutoModelForCausalLM.from_pretrained(
            model_path,
            subfolder="connector",
            torch_dtype=dtype,
        )
        # Disable causal masking (Ming uses bidirectional attention)
        for layer in self._connector.model.layers:
            layer.self_attn.is_causal = False
        self._connector.to(device)
        self._connector.eval()
        logger.info("[SemanticEncoder] Connector loaded on %s", device)

    def _load_projections(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load proj_in, proj_out, and query_tokens from mlp/model.safetensors."""
        from safetensors.torch import load_file

        mlp_path = os.path.join(model_path, "mlp", "model.safetensors")
        logger.info("[SemanticEncoder] Loading projections from %s", mlp_path)
        state = load_file(mlp_path)

        # proj_in: Linear(llm_hidden=4096, connector_hidden=1536)
        self._proj_in = nn.Linear(
            state["proj_in.weight"].shape[1],
            state["proj_in.weight"].shape[0],
        )
        self._proj_in.load_state_dict(
            {
                "weight": state["proj_in.weight"],
                "bias": state["proj_in.bias"],
            }
        )
        self._proj_in.to(device=device, dtype=dtype)

        # proj_out: Linear(connector_hidden=1536, cap_feat_dim=2560)
        self._proj_out = nn.Linear(
            state["proj_out.weight"].shape[1],
            state["proj_out.weight"].shape[0],
        )
        self._proj_out.load_state_dict(
            {
                "weight": state["proj_out.weight"],
                "bias": state["proj_out.bias"],
            }
        )
        self._proj_out.to(device=device, dtype=dtype)

        # Query tokens (learnable, one per scale)
        # For Ming-flash-omni-2.0: only 16x16 scale -> 256 tokens of dim 4096
        query_tokens_list = []
        for scale in self._img_gen_scales:
            key = f"query_tokens_dict.{scale}x{scale}"
            query_tokens_list.append(state[key])
        self._query_tokens = torch.cat(query_tokens_list, dim=0).to(
            device=device, dtype=dtype
        )

        total_tokens = sum(s * s for s in self._img_gen_scales)
        logger.info(
            "[SemanticEncoder] Projections loaded: proj_in=%s, proj_out=%s, "
            "query_tokens=%s (%d tokens)",
            list(self._proj_in.weight.shape),
            list(self._proj_out.weight.shape),
            list(self._query_tokens.shape),
            total_tokens,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        text: str | list[str],
        max_length: int = 512,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode text into semantic condition embeddings.

        Args:
            text: Text prompt(s) to encode.
            max_length: Max token length for text.

        Returns:
            (condition_embeds, negative_condition_embeds):
                Each is a list of tensors with shape [num_tokens, 2560].
        """
        if self._llm is None or self._tokenizer is None:
            raise RuntimeError("Semantic encoder not loaded. Call load() first.")

        if isinstance(text, str):
            text = [text]

        # 0. Wrap in chat template (BailingMoeV2 expects this format)
        eos = self._tokenizer.eos_token
        text = [
            f"{_SYSTEM_PROMPT}{eos}{_USER_PREFIX}{t}{eos}{_ASSISTANT_PREFIX}"
            for t in text
        ]

        # 1. Tokenize
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device)

        # 2. Append multiscale query token placeholders
        input_ids, attention_mask, gen_mask = self._append_query_token_placeholders(
            input_ids, attention_mask
        )

        # 3. Build query token embeddings for vision patching
        image_grid_thw, query_embeds = self._build_query_embeds(input_ids, gen_mask)

        # 4. Embed tokens and patch in query token embeddings
        words_embeddings = self._llm.get_input_embeddings()(
            input_ids.clamp(0, self._llm.get_input_embeddings().weight.shape[0] - 1)
        )
        words_embeddings = self._patch_vision_embeddings(
            input_ids, words_embeddings, query_embeds, self._image_patch_token
        )
        image_mask = (
            (input_ids == self._image_patch_token).unsqueeze(-1).to(input_ids.device)
        )

        # 5. LLM forward pass
        with torch.cuda.amp.autocast(dtype=self._dtype):
            outputs = self._llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=words_embeddings,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                image_mask=image_mask,
                audio_mask=None,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        # 6. Extract generation-relevant hidden states
        condition_embeds = self._extract_and_project(hidden_states, gen_mask)

        # 7. Build negative (zero) embeddings
        negative_embeds = condition_embeds * 0.0

        pos_list = list(condition_embeds.unbind(dim=0))
        neg_list = list(negative_embeds.unbind(dim=0))

        return pos_list, neg_list

    def _append_query_token_placeholders(
        self,
        text_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append [start, patch*N, end] tokens for each scale after text."""
        start_token_id = self._image_start_token
        end_token_id = self._image_end_token
        patch_token_id = self._image_patch_token

        # Build token sequence for all scales
        default_tokens = []
        default_attn = []
        default_gen = []
        for scale in self._img_gen_scales:
            n = scale * scale
            default_tokens.append(start_token_id)
            default_tokens.extend([patch_token_id] * n)
            default_tokens.append(end_token_id)
            default_attn.extend([1] * (n + 2))
            default_gen.append(0)  # start token: not for generation
            default_gen.extend([1] * n)  # patch tokens: for generation
            default_gen.append(0)  # end token: not for generation

        text_ids_list = text_ids.cpu().tolist()
        attn_list = attention_mask.cpu().tolist()

        new_ids = []
        new_attn = []
        new_gen = []

        for tid, am in zip(text_ids_list, attn_list):
            # Find where padding starts
            pad_start = sum(1 for v in am if v != 0)

            new_ids.append(tid[:pad_start] + deepcopy(default_tokens) + tid[pad_start:])
            new_attn.append(am[:pad_start] + deepcopy(default_attn) + am[pad_start:])
            new_gen.append(
                [0] * pad_start + deepcopy(default_gen) + [0] * (len(tid) - pad_start)
            )

        device = text_ids.device
        new_ids_t = torch.tensor(new_ids, dtype=text_ids.dtype, device=device)
        new_attn_t = torch.tensor(new_attn, dtype=attention_mask.dtype, device=device)
        gen_mask_t = torch.tensor(new_gen, dtype=attention_mask.dtype, device=device)

        return new_ids_t, new_attn_t, gen_mask_t

    def _build_query_embeds(
        self,
        text_ids: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build image_grid_thw and query token embeddings."""
        text_ids_list = text_ids.cpu().view(-1).tolist()
        gen_mask_list = gen_mask.cpu().view(-1).tolist()

        is_patch = [1 if t == self._image_patch_token else 0 for t in text_ids_list]
        idxes = _find_first_index_of_consecutive_ones(is_patch)
        isgen = _merge_consecutive_ones(
            [1 if gen_mask_list[i] else 0 for i in idxes],
            len(self._img_gen_scales),
        )

        image_grid_thw = []
        image_embeds_list = []
        for is_gen in isgen:
            if is_gen:
                for scale in self._img_gen_scales:
                    image_grid_thw.append([1, 2, scale * scale * 2])
                image_embeds_list.append(self._query_tokens)

        grid_thw = torch.tensor(
            image_grid_thw, dtype=text_ids.dtype, device=text_ids.device
        )
        embeds = torch.cat(image_embeds_list, dim=0).to(text_ids.device)

        return grid_thw, embeds

    @staticmethod
    def _patch_vision_embeddings(
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
        vision_token_id: int,
    ) -> torch.Tensor:
        """Replace vision patch token embeddings with actual vision embeddings."""
        if vision_embeds.ndim == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        vision_mask = (
            (input_ids == vision_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(vision_mask, vision_embeds)
        return inputs_embeds

    def _extract_and_project(
        self,
        hidden_states: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract gen-marked hidden states, run through connector + proj."""
        with torch.cuda.amp.autocast(dtype=self._dtype):
            # Expand gen_mask to hidden_states dimensions
            mask = (
                gen_mask.unsqueeze(-1)
                .expand(gen_mask.shape[0], gen_mask.shape[1], hidden_states.shape[-1])
                .to(hidden_states.device)
                .bool()
            )

            # Extract generation-relevant hidden states
            hidden_gen = torch.masked_select(hidden_states, mask).view(
                hidden_states.shape[0], -1, hidden_states.shape[-1]
            )

            # Select highest resolution scale
            scale_starts = [0] + self._scale_indices[:-1]
            scale_ends = self._scale_indices
            _, start, end = list(zip(self._img_gen_scales, scale_starts, scale_ends))[
                -1
            ]

            scale_hidden = hidden_gen[:, start:end, :]

            # Move to auxiliary device (connector/proj may be on a
            # different GPU than the last LLM layer)
            scale_hidden = scale_hidden.to(self._aux_device)

            # Project to connector input dim
            scale_embeds = self._proj_in(scale_hidden)
            seq_shape = scale_embeds.shape

            # Run through connector (non-causal transformer)
            connector_out = self._connector(
                inputs_embeds=scale_embeds,
                attention_mask=torch.ones(
                    seq_shape[0],
                    1,
                    seq_shape[1],
                    seq_shape[1],
                    device=scale_embeds.device,
                ),
                output_hidden_states=True,
            )
            scale_embeds = connector_out.hidden_states[-1]

            # Project to diffusion input dim and L2 normalize
            scale_embeds = self._proj_out(scale_embeds)
            scale_embeds = torch.nn.functional.normalize(scale_embeds, dim=-1)

        return scale_embeds

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._connector is not None:
            del self._connector
            self._connector = None
        if self._proj_in is not None:
            del self._proj_in
            self._proj_in = None
        if self._proj_out is not None:
            del self._proj_out
            self._proj_out = None
        self._query_tokens = None
        self._tokenizer = None
        torch.cuda.empty_cache()
