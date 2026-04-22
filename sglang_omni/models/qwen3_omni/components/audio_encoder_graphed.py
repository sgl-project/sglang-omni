# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-capturable wrapper for ``Qwen3OmniMoeAudioEncoder``.

Why
---

sglang main's ``Qwen3OmniMoeAudioEncoder.forward`` contains three classes of
ops that :func:`torch.cuda.graph` cannot capture:

  1. ``torch.tensor([python_list], device=...)`` — host-to-device alloc
     driven by a python list built at runtime
  2. ``.item()`` / ``.tolist()`` — implicit device-to-host sync
  3. ``padded_embed[bool_mask]`` — boolean masked indexing, whose output
     shape requires a device-to-host sync of ``mask.sum()``

For a fixed ``(batch, seq_len)`` the chunking layout is entirely
deterministic, so we precompute:

- ``chunk_lengths`` (as a python list; constant split sizes)
- ``flat_mask_indices`` (replaces boolean mask indexing with
  ``index_select`` on a pre-materialized long tensor)
- ``cu_seqlens`` + ``max_seqlen`` (passed to attention in the format
  ``VisionFlash3Attention`` / ``VisionTritonAttention`` /
  ``VisionAscendAttention`` accept when ``SGLANG_VIT_ENABLE_CUDA_GRAPH`` is
  set)
- the sliced + dtype-cast positional embedding

Caveats
-------

- Single fixed ``(batch, seq_len)`` per instance. For multiple shapes,
  build one wrapper per bucket and dispatch (Phase 2 — shape bucketing).
- ``feature_attention_mask`` / ``audio_feature_lengths`` are rejected at
  ``forward`` if they disagree with the configured ``(batch, seq_len)``.
- Reaches into base encoder's ``conv2d1/2/3``, ``layers``,
  ``positional_embedding``, ``ln_post``, ``proj1``, ``proj2`` directly.
  Upstream refactors to those internals will require re-syncing this file.
  A proper long-term fix is to have the base encoder expose a
  ``static_forward(input_features, layout)`` hook; this wrapper is the
  POC demonstrating why such a hook is worth adding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.models.qwen3_omni_moe import _get_feat_extract_output_lengths


class GraphedAudioEncoder(nn.Module):
    """Fixed-shape wrapper around ``Qwen3OmniMoeAudioEncoder`` suitable for
    :func:`torch.cuda.graph` capture.

    Parameters
    ----------
    base_encoder
        A loaded ``sglang.srt.models.qwen3_omni_moe.Qwen3OmniMoeAudioEncoder``
        (typically ``Qwen3OmniAudioEncoderNative(...).audio_tower``).
    batch, seq_len
        Shape contract for this instance. ``forward`` asserts any passed-in
        lengths match.
    device
        Target device for precomputed layout tensors. Base encoder params are
        already on their own device/dtype; inputs are cast to match at forward.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        *,
        batch: int,
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.base = base_encoder
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self._device = torch.device(device)

        cfg = self.base.config
        self.n_window = int(cfg.n_window)
        self.n_window_infer = int(cfg.n_window_infer)
        self.conv_chunksize = int(cfg.conv_chunksize)

        base_params = next(self.base.parameters())
        self._param_dtype = base_params.dtype
        self._param_device = base_params.device

        self._precompute()

    def _precompute(self) -> None:
        """Run base.forward's chunk-layout math once; stash results as
        python ints / fixed tensors.

        ``.item()`` / ``.tolist()`` are OK here — called once at init,
        outside any graph capture.
        """
        n_window = self.n_window
        n_window_infer = self.n_window_infer
        device = self._device

        feature_lens = torch.tensor(
            [self.seq_len] * self.batch, device=device, dtype=torch.long
        )
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)

        chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
        total_chunks = int(chunk_num.sum().item())
        chunk_lengths = torch.tensor(
            [n_window * 2] * total_chunks, dtype=torch.long, device=device
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
        chunk_lengths[chunk_lengths == 0] = n_window * 2

        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        max_len_after_cnn = int(feature_lens_after_cnn.max().item())

        idx = torch.arange(max_len_after_cnn, device=device)
        padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            n_window_infer // (n_window * 2)
        )
        for cnn_len in aftercnn_lens.tolist():
            num_full_chunks = cnn_len // window_aftercnn
            remainder = cnn_len % window_aftercnn
            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
            if remainder:
                cu_chunk_lens.append(remainder)
        cu_seqlens = torch.tensor(cu_chunk_lens, device=device).cumsum(
            -1, dtype=torch.int32
        )

        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(seq_lens.max().item())

        # Boolean mask indexing in forward would sync CPU to compute output
        # size. Pre-resolve to a long index tensor.
        flat_indices = (
            padded_mask_after_cnn.flatten()
            .nonzero(as_tuple=False)
            .squeeze(-1)
            .to(torch.long)
        )

        # Sliced + dtype-cast positional embedding (shape derived from
        # post-conv seq length, which is constant for fixed (batch, seq_len)).
        pe_len = padded_mask_after_cnn.shape[-1]
        pe = (
            self.base.positional_embedding.positional_embedding[:pe_len, :]
            .unsqueeze(0)
            .to(dtype=self._param_dtype)
            .contiguous()
        )

        # Per-sample output lengths (after CNN + whatever sglang's helper
        # computes). Constant for the fixed (batch, seq_len) contract, so we
        # precompute and reuse for the drop-in-compatible forward dict.
        audio_output_lengths = _get_feat_extract_output_lengths(feature_lens)

        self.chunk_lengths_list: list[int] = chunk_lengths.tolist()
        self.register_buffer("flat_mask_indices", flat_indices, persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        self.register_buffer("pe", pe, persistent=False)
        self.register_buffer(
            "_audio_feature_lengths", feature_lens, persistent=False
        )
        self.register_buffer(
            "_audio_output_lengths", audio_output_lengths, persistent=False
        )
        self._max_seqlen = max_seqlen

        # Resolve cu_seqlens arg format once at init. sglang's graph-capable
        # attention backends (fa3/triton/ascend) consume a (cu_tensor,
        # max_seqlen) tuple when SGLANG_VIT_ENABLE_CUDA_GRAPH is set, else a
        # plain tensor (non-graph branch). Picking at init avoids an env
        # lookup per forward inside the captured region.
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            self._cu_arg: object = (cu_seqlens, max_seqlen)
        else:
            self._cu_arg = cu_seqlens

    def _conv_stack(self, padded_feature: torch.Tensor) -> torch.Tensor:
        """Run ``conv2d1 -> conv2d2 -> conv2d3`` with chunk splitting when
        the batch exceeds ``conv_chunksize`` (matches base.forward)."""
        base = self.base
        if padded_feature.size(0) <= self.conv_chunksize:
            x = F.gelu(base.conv2d1(padded_feature))
            x = F.gelu(base.conv2d2(x))
            x = F.gelu(base.conv2d3(x))
            return x
        pieces: list[torch.Tensor] = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            x = F.gelu(base.conv2d1(chunk))
            x = F.gelu(base.conv2d2(x))
            x = F.gelu(base.conv2d3(x))
            pieces.append(x)
        return torch.cat(pieces, dim=0)

    def _static_forward(self, input_features: torch.Tensor) -> torch.Tensor:
        base = self.base
        # Cast input to base encoder's device/dtype. ``.to(...)`` is a no-op
        # when already matching, so this is safe inside CUDA graph capture.
        input_features = input_features.to(
            device=self._param_device, dtype=self._param_dtype
        )
        chunk_list = input_features.T.split(self.chunk_lengths_list, dim=0)
        padded_feature = (
            nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
            .transpose(1, 2)
            .unsqueeze(1)
        )

        padded_embed = self._conv_stack(padded_feature)

        b, c, f, t = padded_embed.size()
        padded_embed = base.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )
        padded_embed = padded_embed + self.pe

        # Graph-safe equivalent of base's ``padded_embed[padded_mask_after_cnn]``:
        # flatten to [B*T, D] then gather pre-computed positions.
        D = padded_embed.shape[-1]
        hidden_states = padded_embed.reshape(-1, D).index_select(
            0, self.flat_mask_indices
        )

        for encoder_layer in base.layers:
            hidden_states = encoder_layer(hidden_states, self._cu_arg)[0]

        hidden_states = base.ln_post(hidden_states)
        hidden_states = base.proj1(hidden_states)
        hidden_states = base.act(hidden_states)
        hidden_states = base.proj2(hidden_states)
        return hidden_states

    def _check_shape_contract(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
        audio_feature_lengths: torch.Tensor | None,
    ) -> None:
        expected_tokens = self.batch * self.seq_len
        if input_features.dim() != 2 or input_features.shape[1] != expected_tokens:
            raise ValueError(
                f"input_features shape {tuple(input_features.shape)} does not "
                f"match configured (batch={self.batch}, seq_len={self.seq_len}) "
                f"— expected second dim {expected_tokens}."
            )
        if feature_attention_mask is not None:
            if tuple(feature_attention_mask.shape) != (self.batch, self.seq_len):
                raise ValueError(
                    f"feature_attention_mask shape "
                    f"{tuple(feature_attention_mask.shape)} != "
                    f"({self.batch}, {self.seq_len})"
                )
            if not bool(feature_attention_mask.all().item()):
                raise ValueError(
                    "GraphedAudioEncoder does not accept variable-length input. "
                    "feature_attention_mask must be all-true."
                )
        if audio_feature_lengths is not None:
            if audio_feature_lengths.numel() != self.batch:
                raise ValueError(
                    f"audio_feature_lengths has {audio_feature_lengths.numel()} "
                    f"entries, expected batch={self.batch}."
                )
            if not bool((audio_feature_lengths == self.seq_len).all().item()):
                raise ValueError(
                    "GraphedAudioEncoder does not accept variable-length "
                    "input; all entries of audio_feature_lengths must equal "
                    f"seq_len={self.seq_len}."
                )

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        skip_shape_check: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run encoder.

        ``skip_shape_check=True`` is provided for the *replay* step of a
        captured CUDA graph, where host-side asserts would break capture/
        replay. The benchmark harness sets it; normal callers should not.
        """
        if not skip_shape_check:
            self._check_shape_contract(
                input_features, feature_attention_mask, audio_feature_lengths
            )
        out = self._static_forward(input_features)
        # Match Qwen3OmniAudioEncoder / Qwen3OmniAudioEncoderNative return
        # shape (length tensors are constant for our fixed-shape contract,
        # precomputed at init).
        return {
            "audio_embeds": out,
            "audio_feature_lengths": self._audio_feature_lengths,
            "audio_output_lengths": self._audio_output_lengths,
        }
