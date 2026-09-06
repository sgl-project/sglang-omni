# SPDX-License-Identifier: Apache-2.0
"""SGLang-native ZONOS2 backbone: custom MoE transformer on RadixAttention.

The verified forward math ported onto SGLang primitives so the
AR decode runs under the scheduler with a paged radix KV cache, continuous
batching, FusedMoE and CUDA-graph capture. The multi-codebook head + per-frame
sampling/feedback live in the model runner (model_runner.py); this module emits
backbone hidden states and exposes the head via :meth:`compute_logits`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams
from sglang_omni.models.zonos2.hf_config import Zonos2Config
from sglang_omni.models.zonos2.radix_hash import poly_row_hash
from sglang_omni.models.zonos2.sampler import sample_tts
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    RadixAttention,
    StandardTopKOutput,
    VocabParallelEmbedding,
    get_moe_impl_class,
    get_rope,
)

_QK_NORM_EPS = 1e-6


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(x / cap)


class Zonos2Attention(nn.Module):
    """GQA with QK-norm, learnable per-head temp, interleaved RoPE, headwise gate."""

    def __init__(self, cfg: Zonos2Config, layer_id: int):
        super().__init__()
        self.nq, self.nkv, self.hd, self.h = (
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.dim,
        )
        self.wq = nn.Parameter(torch.empty(self.nq * self.hd, self.h))
        self.wkv = nn.Parameter(torch.empty(2, self.nkv * self.hd, self.h))
        self.wo = nn.Parameter(torch.empty(self.h, self.nq * self.hd))
        self.gater = nn.Parameter(torch.empty(self.nq, self.h))
        self.temp = nn.Parameter(torch.empty(1, self.nq, 1))
        self.rotary = get_rope(
            self.hd, self.hd, cfg.max_seqlen, int(cfg.rope_theta), is_neox_style=False
        )
        self.attn = RadixAttention(
            self.nq, self.hd, self.hd**-0.5, self.nkv, layer_id=layer_id
        )

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        t = x.shape[0]
        gate = torch.sigmoid(F.linear(x, self.gater))
        q = F.linear(x, self.wq).view(t, self.nq, self.hd)
        k = F.linear(x, self.wkv[0]).view(t, self.nkv, self.hd)
        v = F.linear(x, self.wkv[1]).view(t, self.nkv, self.hd)
        q = F.rms_norm(q, (self.hd,), None, _QK_NORM_EPS) * self.temp.abs().to(q.dtype)
        k = F.rms_norm(k, (self.hd,), None, _QK_NORM_EPS)
        q = q.reshape(t, self.nq * self.hd)
        k = k.reshape(t, self.nkv * self.hd)
        v = v.reshape(t, self.nkv * self.hd)
        q, k = self.rotary(positions, q, k)
        o = self.attn(q, k, v, forward_batch)
        o = (o.view(t, self.nq, self.hd) * gate.unsqueeze(-1)).reshape(
            t, self.nq * self.hd
        )
        return F.linear(o, self.wo)


class Zonos2DenseFFN(nn.Module):
    def __init__(self, cfg: Zonos2Config):
        super().__init__()
        self.w_in = nn.Parameter(torch.empty(2, cfg.intermediate_size, cfg.dim))
        self.w_out = nn.Parameter(torch.empty(cfg.dim, cfg.intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.linear(x, self.w_in[0])
        gate = F.linear(x, self.w_in[1])
        return F.linear(up * F.silu(gate), self.w_out)


class Zonos2SonicRouter(nn.Module):
    """EDA router → StandardTopKOutput; carries router_states to the next MoE layer."""

    def __init__(self, cfg: Zonos2Config, layer_id: int):
        super().__init__()
        rd, e = cfg.moe_router_dim, cfg.moe_n_experts
        self.top_k = cfg.topk_for_layer(layer_id)
        self.use_eda = cfg.uses_eda(layer_id)
        self.eps = cfg.norm_eps
        strat = (
            (cfg.moe_balancing_strategy or "legacy").strip().lower().replace("-", "_")
        )
        self.bias_sign = -1.0 if strat in ("current", "quantile", "qbalancing") else 1.0
        self.down_proj = nn.Linear(cfg.dim, rd, bias=True)
        self.router_mlp_0 = nn.Linear(rd, rd, bias=True)
        self.router_mlp_2 = nn.Linear(rd, rd, bias=True)
        self.router_mlp_4 = nn.Linear(rd, e, bias=False)
        self.rmsnorm_eda = nn.Parameter(torch.empty(rd))
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.empty(rd))
        self.register_buffer("balancing_biases", torch.zeros(e), persistent=True)

    def forward(
        self, h: torch.Tensor, router_states: Optional[torch.Tensor]
    ) -> Tuple[StandardTopKOutput, torch.Tensor]:
        h = self.down_proj(h)
        if self.use_eda and router_states is not None:
            h = h + router_states * self.router_states_scale
        rs_next = h.clone()
        h = F.rms_norm(h, (h.shape[-1],), self.rmsnorm_eda, self.eps)
        logits = self.router_mlp_4(
            F.gelu(self.router_mlp_2(F.gelu(self.router_mlp_0(h))))
        )
        probs = torch.softmax(logits.float(), dim=-1)
        scores = probs + self.bias_sign * self.balancing_biases.float()
        _, ids = torch.topk(scores, self.top_k, dim=-1)
        weights = torch.gather(probs, -1, ids)
        topk = StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32), router_logits=logits
        )
        return topk, rs_next


class Zonos2MoEBlock(nn.Module):
    def __init__(
        self, cfg: Zonos2Config, layer_id: int, quant_config: Optional[Any] = None
    ):
        super().__init__()
        self.router = Zonos2SonicRouter(cfg, layer_id)
        self.experts = get_moe_impl_class(None)(
            num_experts=cfg.moe_n_experts,
            top_k=cfg.topk_for_layer(layer_id),
            hidden_size=cfg.dim,
            intermediate_size=cfg.intermediate_size,
            layer_id=layer_id,
            reduce_results=False,
            quant_config=quant_config,
        )

    def forward(
        self, x: torch.Tensor, router_states: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk, rs_next = self.router(x, router_states)
        return self.experts(x, topk), rs_next


class Zonos2DecoderLayer(nn.Module):
    def __init__(
        self, cfg: Zonos2Config, layer_id: int, quant_config: Optional[Any] = None
    ):
        super().__init__()
        self.eps = cfg.norm_eps
        self.attention = Zonos2Attention(cfg, layer_id)
        self.attention_norm = nn.Parameter(torch.empty(cfg.dim))
        self.ffn_norm = nn.Parameter(torch.empty(cfg.dim))
        self.is_moe = cfg.is_moe_layer(layer_id)
        self.feed_forward = (
            Zonos2MoEBlock(cfg, layer_id, quant_config)
            if self.is_moe
            else Zonos2DenseFFN(cfg)
        )

    def forward(self, x, residual, router_states, positions, forward_batch):
        if residual is None:
            residual = x
            h = F.rms_norm(x, (x.shape[-1],), self.attention_norm, self.eps)
        else:
            residual = x + residual
            h = F.rms_norm(residual, (x.shape[-1],), self.attention_norm, self.eps)
        h = self.attention(h, positions, forward_batch)
        residual = h + residual
        h = F.rms_norm(residual, (h.shape[-1],), self.ffn_norm, self.eps)
        if self.is_moe:
            h, router_states = self.feed_forward(h, router_states)
        else:
            h = self.feed_forward(h)
            router_states = None
        return h, residual, router_states


class Zonos2TransformerBody(nn.Module):
    """Transformer body captured by SGLang's breakable prefill graph.

    The body deliberately starts at the already-composed ``input_embeds``
    boundary.  Speaker/reference conditioning and decode feedback are owned by
    the outer wrapper, while this registered submodule contains only the
    graph-capturable transformer stack and final hidden-state normalization.
    """

    def __init__(self, cfg: Zonos2Config, quant_config: Optional[Any] = None) -> None:
        super().__init__()
        self.emb_norm_eps = cfg.norm_eps
        self.layers = nn.ModuleList(
            [Zonos2DecoderLayer(cfg, i, quant_config) for i in range(cfg.n_layers)]
        )
        self.out_norm = nn.Parameter(torch.empty(cfg.dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer stack and return hidden states as a Tensor."""
        del input_ids
        if input_embeds is None:
            raise RuntimeError(
                "ZONOS2 transformer body requires pre-composed input_embeds"
            )

        x = F.rms_norm(
            input_embeds,
            (input_embeds.shape[-1],),
            None,
            self.emb_norm_eps,
        )
        residual = None
        router_states = None
        for layer in self.layers:
            x, residual, router_states = layer(
                x, residual, router_states, positions, forward_batch
            )
        return F.rms_norm(
            x + residual, (x.shape[-1],), self.out_norm, self.emb_norm_eps
        )


class Zonos2SGLangModel(nn.Module):
    """ZONOS2 backbone for the SGLang scheduler.

    Prefill embeds 2D ``(T, 10)`` frame rows (9 audio codebooks + text) summed; the
    decode step reads per-request feedback embeddings staged by the runner into
    ``_decode_input_embedding`` (row indices arrive as ``input_ids``).
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[Any] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cfg = (
            config
            if isinstance(config, Zonos2Config)
            else Zonos2Config(**config.to_dict())
        )
        self.config = cfg
        self.n_codebooks = cfg.n_codebooks
        self.audio_vocab = cfg.audio_vocab
        self.frame_width = cfg.n_codebooks + 1

        self.embedders = nn.ModuleList(
            [
                VocabParallelEmbedding(cfg.codebook_size + 2, cfg.dim)
                for _ in range(cfg.n_codebooks)
            ]
            + [VocabParallelEmbedding(cfg.text_vocab + 1, cfg.dim)]
        )
        self.speaker_lda_projection = nn.Linear(
            cfg.speaker_embedding_dim, cfg.speaker_lda_dim, bias=True
        )
        self.speaker_projection = nn.Linear(cfg.speaker_lda_dim, cfg.dim, bias=True)
        self.model = Zonos2TransformerBody(cfg, quant_config)
        self.multi_output = nn.Parameter(
            torch.empty(self.audio_vocab * self.n_codebooks, cfg.dim)
        )

        try:
            from sglang.srt.server_args import get_global_server_args

            max_bs = int(get_global_server_args().max_running_requests or 1)
        except Exception:
            max_bs = 256
        w = self.embedders[0].weight
        self._decode_input_embedding = nn.Embedding(
            max_bs, cfg.dim, device=w.device, dtype=w.dtype
        )
        self._decode_input_embedding.weight.requires_grad_(False)
        # note (Yue Yin): on-device per-request decode state (feedback + EOS +
        # rep-history ring) for the eager-tail RTF work. Allocated here so its
        # tensors get stable addresses before CUDA-graph capture; the runner
        # gathers/scatters it each frame instead of per-request Python deques.
        from sglang_omni.models.zonos2.state_pool import Zonos2DecodeStatePool

        self._decode_state_pool = Zonos2DecodeStatePool(self)
        self.requires_grad_(
            False
        )  # inference-only; sglang in-place ops reject grad tensors

        # Opt-in tail CUDA graph (ZONOS2_FRAME_GRAPH): captures the launch-heavy
        # per-frame tail (head GEMM + per-request sample + embed + radix hash) into
        # one replay per decode bucket, cutting host dispatch in the host-bound
        # decode loop. Built by capture_tail_graphs; empty -> eager runner path.
        self._tail_buckets: list[int] = []
        self._tail_graphs: dict[int, Any] = {}
        self._tail_params: Optional[TTSSamplingParams] = None
        self._tail_top_k_max: int = 0
        self._tail_any_top_p: bool = False
        self._tail_any_min_p: bool = False
        self._cg: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return self.embedders[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedders[0].weight.dtype

    def reset_request(self, rid: str) -> None:
        """Release decode state for a finished or aborted request."""
        self._decode_state_pool.release_row(rid)

    def embed_frames(self, rows: torch.Tensor) -> torch.Tensor:
        """Sum the per-column embeddings of ``(T, frame_width)`` int rows."""
        out = self.embedders[0](rows[:, 0].contiguous())
        for i in range(1, rows.shape[1]):
            out = out + self.embedders[i](rows[:, i].contiguous())
        return out

    def _warmup_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Fallback embed for engine warmup (flat dummy ids; runner stages real
        input_embeds for every served step)."""
        ids = input_ids.view(-1)
        rows = torch.full(
            (ids.shape[0], self.frame_width),
            self.config.audio_pad_id,
            dtype=torch.long,
            device=ids.device,
        )
        rows[:, self.n_codebooks] = self.config.text_vocab
        return self.embed_frames(rows)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        # Prefill: the runner stages the summed (speaker-injected) embedding on
        # forward_batch. Decode: input_ids are row indices into the fixed feedback
        # buffer the runner wrote in-place, so the graph replays a stable input.
        if input_embeds is None:
            input_embeds = getattr(forward_batch, "input_embeds", None)
        if input_embeds is None:
            fm = getattr(forward_batch, "forward_mode", None)
            if fm is not None and fm.is_decode():
                input_embeds = self._decode_input_embedding(input_ids)
            else:
                input_embeds = self._warmup_embed(input_ids)
        hidden = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )
        return LogitsProcessorOutput(
            next_token_logits=hidden.new_empty((hidden.shape[0], 1)),
            hidden_states=hidden,
        )

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Backbone hidden → per-codebook soft-capped logits ``(..., 9, 1026)``."""
        logits = F.linear(hidden, self.multi_output)
        logits = logits.view(*logits.shape[:-1], self.n_codebooks, self.audio_vocab)
        return softcap(logits, self.config.loss_softcap)

    # ---- opt-in tail CUDA graph (ZONOS2_FRAME_GRAPH) ----

    @torch.no_grad()
    def _tail_compute(self, bs: int) -> None:
        """Graph-capturable per-frame tail over the first ``bs`` static rows: head
        GEMM -> break-mask -> per-request sample (torch.multinomial, capturable)
        -> embed -> radix hash. Reads/writes the ``_cg`` buffers in place."""
        cg = self._cg
        logits = self.compute_logits(cg["hidden"][:bs]).float()
        logits[:, 0].add_(cg["break_mask"][:bs])
        codes = sample_tts(
            logits,
            temperature=cg["temperature"][:bs],
            top_k=cg["top_k"][:bs],
            top_p=cg["top_p"][:bs],
            min_p=cg["min_p"][:bs],
            repetition_penalty=cg["rep_pen"][:bs],
            top_k_max=self._tail_top_k_max,
            rep_token_ids=cg["rep_ids"][:bs],
            any_top_p=self._tail_any_top_p,
            any_min_p=self._tail_any_min_p,
        )
        text_col = torch.full(
            (bs, 1), self.config.text_vocab, device=codes.device, dtype=torch.long
        )
        rows = torch.cat([codes, text_col], dim=1)
        cg["codes"][:bs].copy_(codes)
        cg["feedback"][:bs].copy_(self.embed_frames(rows))
        cg["keys"][:bs].copy_(poly_row_hash(rows))

    @torch.no_grad()
    def capture_tail_graphs(
        self, buckets: list[int], params: TTSSamplingParams
    ) -> None:
        """Allocate static buffers + bake the structural sampler flags, then
        capture one tail graph per batch bucket."""
        n, dim, V = self.n_codebooks, self.config.dim, self.audio_vocab
        W = int(params.repetition_window)
        mb = max(buckets)
        dev, dt = self.device, self.dtype
        f32, i64 = torch.float32, torch.long
        self._cg = {
            "hidden": torch.zeros(mb, dim, device=dev, dtype=dt),
            "temperature": torch.full((mb,), params.temperature, device=dev, dtype=f32),
            "top_k": torch.full((mb,), params.top_k, device=dev, dtype=i64),
            "top_p": torch.full((mb,), params.top_p, device=dev, dtype=f32),
            "min_p": torch.full((mb,), params.min_p, device=dev, dtype=f32),
            "rep_pen": torch.full(
                (mb,), params.repetition_penalty, device=dev, dtype=f32
            ),
            "rep_ids": torch.full((mb, n, W), -1, device=dev, dtype=i64),
            "break_mask": torch.zeros(mb, V, device=dev, dtype=f32),
            "codes": torch.zeros(mb, n, device=dev, dtype=i64),
            "keys": torch.zeros(mb, device=dev, dtype=i64),
            "feedback": torch.zeros(mb, dim, device=dev, dtype=dt),
        }
        self._tail_params = params
        self._tail_top_k_max = params.top_k if 0 < params.top_k < V else 0
        self._tail_any_top_p = 0.0 < params.top_p < 1.0
        self._tail_any_min_p = params.min_p > 0.0
        self._tail_buckets = sorted(buckets)
        self._tail_graphs = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for bs in self._tail_buckets:
                for _ in range(3):
                    self._tail_compute(bs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for bs in self._tail_buckets:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._tail_compute(bs)
            self._tail_graphs[bs] = g
        torch.cuda.synchronize()

    @torch.no_grad()
    def run_tail_graph(
        self, hidden, temperature, top_k, top_p, min_p, rep_pen, rep_ids, break_mask
    ):
        """Stage inputs into the static buffers, replay the bs-bucket graph (padded
        up to the next bucket; extra rows compute on stale data and are ignored),
        and return views of (codes [bs,n], keys [bs], feedback [bs,dim])."""
        bs = hidden.shape[0]
        bucket = next(g for g in self._tail_buckets if g >= bs)
        cg = self._cg
        cg["hidden"][:bs].copy_(hidden)
        cg["temperature"][:bs].copy_(temperature)
        cg["top_k"][:bs].copy_(top_k)
        cg["top_p"][:bs].copy_(top_p)
        cg["min_p"][:bs].copy_(min_p)
        cg["rep_pen"][:bs].copy_(rep_pen)
        cg["rep_ids"][:bs].copy_(rep_ids)
        cg["break_mask"][:bs].copy_(break_mask)
        self._tail_graphs[bucket].replay()
        return cg["codes"][:bs], cg["keys"][:bs], cg["feedback"][:bs]

    @torch.no_grad()
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        sd = {k: v for k, v in weights}
        sd = (
            sd.get("model", sd)
            if "model" in sd and isinstance(sd.get("model"), dict)
            else sd
        )
        fixed: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if ".router.ent_denom" in k or ".router.normalized_entropy" in k:
                continue
            if ".parametrizations." in k and ".original" in k:
                k = k.replace(".parametrizations.", ".").replace(".original", "")
            fixed[k] = v
        used: set[str] = set()

        def copy(name: str, dst: torch.Tensor, src: torch.Tensor):
            used.add(name)
            src = src.to(device=dst.device, dtype=dst.dtype)
            if dst.shape == src.shape:
                dst.data.copy_(src)
            elif (
                dst.dim() == src.dim()
                and dst.shape[0] >= src.shape[0]
                and dst.shape[1:] == src.shape[1:]
            ):
                # VocabParallelEmbedding pads the vocab dim; fill the real rows.
                dst.data[: src.shape[0]].copy_(src)
            else:
                raise RuntimeError(
                    f"shape mismatch for {name}: dst {tuple(dst.shape)} src {tuple(src.shape)}"
                )

        for i in range(self.n_codebooks + 1):
            copy(
                f"multi_embedder.embedders.{i}.weight",
                self.embedders[i].weight,
                fixed[f"multi_embedder.embedders.{i}.weight"],
            )
        copy("out_norm.weight", self.model.out_norm, fixed["out_norm.weight"])
        copy("multi_output.weight", self.multi_output, fixed["multi_output.weight"])
        copy(
            "speaker_lda_projection.weight",
            self.speaker_lda_projection.weight,
            fixed["speaker_lda_projection.weight"],
        )
        copy(
            "speaker_projection.weight",
            self.speaker_projection.weight,
            fixed["speaker_projection.weight"],
        )
        copy(
            "speaker_projection.bias",
            self.speaker_projection.bias,
            fixed["speaker_projection.bias"],
        )
        if "speaker_lda_projection.bias" in fixed:
            copy(
                "speaker_lda_projection.bias",
                self.speaker_lda_projection.bias,
                fixed["speaker_lda_projection.bias"],
            )
        else:
            self.speaker_lda_projection.bias.data.zero_()

        for i, layer in enumerate(self.model.layers):
            p = f"layers.{i}."
            a = layer.attention
            copy(p + "attention.wq.weight", a.wq, fixed[p + "attention.wq.weight"])
            copy(p + "attention.wkv.weight", a.wkv, fixed[p + "attention.wkv.weight"])
            copy(p + "attention.wo.weight", a.wo, fixed[p + "attention.wo.weight"])
            copy(
                p + "attention.gater.weight",
                a.gater,
                fixed[p + "attention.gater.weight"],
            )
            copy(p + "attention.temp", a.temp, fixed[p + "attention.temp"])
            copy(
                p + "attention_norm.weight",
                layer.attention_norm,
                fixed[p + "attention_norm.weight"],
            )
            copy(p + "ffn_norm.weight", layer.ffn_norm, fixed[p + "ffn_norm.weight"])
            ff = layer.feed_forward
            if layer.is_moe:
                w13 = fixed[p + "feed_forward.experts.w13"]
                used.add(p + "feed_forward.experts.w13")
                gate_up = torch.cat([w13[:, 0::2, :], w13[:, 1::2, :]], dim=1)
                ff.experts.w13_weight.data.copy_(
                    gate_up.to(
                        ff.experts.w13_weight.device, ff.experts.w13_weight.dtype
                    )
                )
                copy(
                    p + "feed_forward.experts.w2",
                    ff.experts.w2_weight,
                    fixed[p + "feed_forward.experts.w2"],
                )
                r = ff.router
                copy(
                    p + "feed_forward.router.down_proj.weight",
                    r.down_proj.weight,
                    fixed[p + "feed_forward.router.down_proj.weight"],
                )
                copy(
                    p + "feed_forward.router.down_proj.bias",
                    r.down_proj.bias,
                    fixed[p + "feed_forward.router.down_proj.bias"],
                )
                copy(
                    p + "feed_forward.router.router_mlp.0.weight",
                    r.router_mlp_0.weight,
                    fixed[p + "feed_forward.router.router_mlp.0.weight"],
                )
                copy(
                    p + "feed_forward.router.router_mlp.0.bias",
                    r.router_mlp_0.bias,
                    fixed[p + "feed_forward.router.router_mlp.0.bias"],
                )
                copy(
                    p + "feed_forward.router.router_mlp.2.weight",
                    r.router_mlp_2.weight,
                    fixed[p + "feed_forward.router.router_mlp.2.weight"],
                )
                copy(
                    p + "feed_forward.router.router_mlp.2.bias",
                    r.router_mlp_2.bias,
                    fixed[p + "feed_forward.router.router_mlp.2.bias"],
                )
                copy(
                    p + "feed_forward.router.router_mlp.4.weight",
                    r.router_mlp_4.weight,
                    fixed[p + "feed_forward.router.router_mlp.4.weight"],
                )
                copy(
                    p + "feed_forward.router.rmsnorm_eda.weight",
                    r.rmsnorm_eda,
                    fixed[p + "feed_forward.router.rmsnorm_eda.weight"],
                )
                if r.use_eda:
                    copy(
                        p + "feed_forward.router.router_states_scale",
                        r.router_states_scale,
                        fixed[p + "feed_forward.router.router_states_scale"],
                    )
                r.balancing_biases.data.copy_(
                    fixed[p + "feed_forward.router.balancing_biases"].float()
                )
                used.add(p + "feed_forward.router.balancing_biases")
            else:
                copy(
                    p + "feed_forward.w_in.weight",
                    ff.w_in,
                    fixed[p + "feed_forward.w_in.weight"],
                )
                copy(
                    p + "feed_forward.w_out.weight",
                    ff.w_out,
                    fixed[p + "feed_forward.w_out.weight"],
                )

        leftover = set(fixed) - used
        if leftover:
            raise RuntimeError(f"Unconsumed checkpoint keys: {sorted(leftover)[:12]}")


EntryClass = Zonos2SGLangModel
