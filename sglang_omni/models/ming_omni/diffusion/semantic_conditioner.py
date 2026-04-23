# SPDX-License-Identifier: Apache-2.0
"""Lightweight semantic conditioner for ZImage diffusion.

Takes pre-computed hidden states from the thinker LLM and projects them
into condition embeddings for ZImage.  This module loads ONLY the
connector (Qwen2, 28 layers), proj_in, proj_out, and query_tokens from
the Ming model directory — it does NOT load the BailingMoeV2 LLM (~200GB).

Architecture:
    hidden_states [B, 256, 4096]  (from thinker, gen_mask already applied)
    -> proj_in:  Linear(4096, 1536)
    -> connector: Qwen2 non-causal transformer (28 layers, hidden=1536)
    -> proj_out: Linear(1536, 2560)
    -> L2 normalize
    -> condition_embeds [B, 256, 2560]

Total footprint: ~2 GB (vs ~200 GB for the full MingSemanticEncoder).
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SemanticConditioner:
    """Project thinker hidden states into ZImage condition embeddings.

    This is a lightweight replacement for :class:`MingSemanticEncoder` when
    the BailingMoeV2 LLM forward pass is already performed by the SGLang
    thinker engine.  It loads only the connector, projections, and query
    tokens (~2 GB).

    Usage::

        conditioner = SemanticConditioner()
        conditioner.load(model_path, device)

        # hidden_states: already extracted by gen_mask in the thinker
        cond = conditioner.project(hidden_states)
        # cond: [B, 256, 2560], L2-normalized
    """

    def __init__(self) -> None:
        self._connector = None  # Qwen2ForCausalLM (non-causal)
        self._proj_in = None  # Linear(4096, 1536)
        self._proj_out = None  # Linear(1536, 2560)
        self._query_tokens = None  # Tensor(num_tokens, 4096)
        self._device: torch.device | None = None
        self._dtype: torch.dtype = torch.bfloat16

        # Config values set during load
        self._image_patch_token: int = 0
        self._image_start_token: int = 0
        self._image_end_token: int = 0
        self._img_gen_scales: list[int] = []
        self._scale_indices: list[int] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def query_tokens(self) -> torch.Tensor:
        """Learnable query token embeddings [num_tokens, 4096].

        These are needed by the thinker's preprocessor to build the
        input embeddings before the LLM forward pass.
        """
        if self._query_tokens is None:
            raise RuntimeError("SemanticConditioner not loaded. Call load() first.")
        return self._query_tokens

    @property
    def image_patch_token(self) -> int:
        """Token ID for image patch placeholder."""
        return self._image_patch_token

    @property
    def image_start_token(self) -> int:
        """Token ID for image generation start marker."""
        return self._image_start_token

    @property
    def image_end_token(self) -> int:
        """Token ID for image generation end marker."""
        return self._image_end_token

    @property
    def img_gen_scales(self) -> list[int]:
        """Image generation scales (e.g. [16] for 16x16)."""
        return self._img_gen_scales

    @property
    def device(self) -> torch.device | None:
        """Device where connector/projections are loaded."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype used by connector/projections."""
        return self._dtype

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Load connector, projections, and query tokens from the Ming model.

        This loads ONLY the lightweight components (~2 GB), NOT the
        BailingMoeV2 LLM.

        Args:
            model_path: Path to the Ming model directory containing
                config.json, connector/, and mlp/.
            device: Device for connector and projection layers.
            dtype: Model dtype (default bf16).
        """
        self._device = device
        self._dtype = dtype

        # 1. Read configs
        config_path = os.path.join(model_path, "config.json")
        logger.info("[SemanticConditioner] Reading config from %s", config_path)
        with open(config_path) as f:
            full_config = json.load(f)
        llm_config_dict = full_config["llm_config"]

        self._image_patch_token = llm_config_dict.get("image_patch_token", 157157)
        self._image_start_token = llm_config_dict.get(
            "image_start_token", self._image_patch_token + 1
        )
        self._image_end_token = self._image_start_token + 1

        mlp_config_path = os.path.join(model_path, "mlp", "config.json")
        with open(mlp_config_path) as f:
            mlp_config = json.load(f)
        self._img_gen_scales = mlp_config.get("img_gen_scales", [16])

        # Pre-compute cumulative scale indices
        self._scale_indices = []
        current_idx = 0
        for scale in self._img_gen_scales:
            current_idx += scale * scale
            self._scale_indices.append(current_idx)

        logger.info(
            "[SemanticConditioner] Scales: %s, total tokens: %d",
            self._img_gen_scales,
            current_idx,
        )

        # 2. Load connector (Qwen2ForCausalLM with non-causal attention)
        self._load_connector(model_path, device, dtype)

        # 3. Load projections and query tokens from mlp/model.safetensors
        self._load_projections(model_path, device, dtype)

        logger.info(
            "[SemanticConditioner] All components loaded on %s (~%.1f GB)",
            device,
            self._estimate_memory_gb(),
        )

    def _load_connector(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load Qwen2ForCausalLM connector with non-causal attention."""
        from transformers import AutoModelForCausalLM

        logger.info(
            "[SemanticConditioner] Loading connector from %s/connector",
            model_path,
        )
        self._connector = AutoModelForCausalLM.from_pretrained(
            model_path,
            subfolder="connector",
            torch_dtype=dtype,
        )
        # Disable causal masking (Ming uses bidirectional attention in the
        # connector for image generation conditioning)
        for layer in self._connector.model.layers:
            layer.self_attn.is_causal = False
        self._connector.to(device)
        self._connector.eval()
        logger.info("[SemanticConditioner] Connector loaded on %s", device)

    def _load_projections(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load proj_in, proj_out, and query_tokens from mlp/model.safetensors."""
        from safetensors.torch import load_file

        mlp_path = os.path.join(model_path, "mlp", "model.safetensors")
        logger.info("[SemanticConditioner] Loading projections from %s", mlp_path)
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

        # Query tokens (learnable, one set per scale)
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
            "[SemanticConditioner] Projections loaded: proj_in=%s, "
            "proj_out=%s, query_tokens=%s (%d tokens)",
            list(self._proj_in.weight.shape),
            list(self._proj_out.weight.shape),
            list(self._query_tokens.shape),
            total_tokens,
        )

    def _estimate_memory_gb(self) -> float:
        """Estimate total GPU memory used by loaded components."""
        total_params = 0
        if self._connector is not None:
            total_params += sum(p.numel() for p in self._connector.parameters())
        if self._proj_in is not None:
            total_params += sum(p.numel() for p in self._proj_in.parameters())
        if self._proj_out is not None:
            total_params += sum(p.numel() for p in self._proj_out.parameters())
        if self._query_tokens is not None:
            total_params += self._query_tokens.numel()

        # bf16 = 2 bytes per parameter
        bytes_per_param = 2 if self._dtype == torch.bfloat16 else 4
        return (total_params * bytes_per_param) / (1 << 30)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def project(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Project thinker hidden states into ZImage condition embeddings.

        Takes already-extracted hidden states (gen_mask applied upstream
        by the image generation executor) and runs them through:
        proj_in -> connector -> proj_out -> L2 normalize.

        Args:
            hidden_states: Tensor of shape [B, N, 4096] where N is the
                number of query tokens (256 for 16x16 scale).  These are
                the hidden states from the thinker LLM at the gen_mask
                positions.

        Returns:
            Condition embeddings of shape [B, N, 2560], L2-normalized
            along the last dimension.
        """
        if self._connector is None:
            raise RuntimeError("SemanticConditioner not loaded. Call load() first.")

        with torch.cuda.amp.autocast(dtype=self._dtype):
            # Move to conditioner device (may differ from thinker device)
            h = hidden_states.to(self._device)

            # For multi-scale support: select the highest resolution scale.
            # Currently with single 16x16 scale, this is a no-op slice.
            scale_starts = [0] + self._scale_indices[:-1]
            scale_ends = self._scale_indices
            _, start, end = list(zip(self._img_gen_scales, scale_starts, scale_ends))[
                -1
            ]
            h = h[:, start:end, :]

            # Project to connector input dimension
            h = self._proj_in(h)
            seq_shape = h.shape

            # Run through connector (non-causal Qwen2 transformer)
            connector_out = self._connector(
                inputs_embeds=h,
                attention_mask=torch.ones(
                    seq_shape[0],
                    1,
                    seq_shape[1],
                    seq_shape[1],
                    device=h.device,
                ),
                output_hidden_states=True,
            )
            h = connector_out.hidden_states[-1]

            # Project to diffusion input dimension and L2 normalize
            h = self._proj_out(h)
            h = torch.nn.functional.normalize(h, dim=-1)

        return h

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory held by connector, projections, and query tokens."""
        logger.info("[SemanticConditioner] Unloading components")
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
        torch.cuda.empty_cache()
        logger.info("[SemanticConditioner] Unloaded, GPU cache cleared")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._connector is not None
        return (
            f"SemanticConditioner(loaded={loaded}, "
            f"scales={self._img_gen_scales}, "
            f"device={self._device}, dtype={self._dtype})"
        )
