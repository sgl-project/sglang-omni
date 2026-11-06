# SPDX-License-Identifier: Apache-2.0
"""YuE2 weight loading and complete song generation."""

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from sglang_omni.models.yue2.acoustic import synthesize_latents
from sglang_omni.models.yue2.batch_generation import TokenRequest, generate_batch_tokens
from sglang_omni.models.yue2.cuda_graph import DecodeGraph
from sglang_omni.models.yue2.generation import TokenResult, generate_tokens
from sglang_omni.models.yue2.layers import Yue2ModelConfig
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    CODEC_OFFSET,
    CancellationCheck,
    Yue2Request,
    Yue2Tokenizer,
    check_cancelled,
)
from sglang_omni.models.yue2.vae import VaeConfig, Yue2Vae
from sglang_omni.utils.checkpoint import resolve_checkpoint


def load_weights(module: nn.Module, directory: Path, *, decoder_only: bool) -> None:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(mapping.values()))
    else:
        filenames = ["model.safetensors"]
    weights: dict[str, torch.Tensor] = {}
    for filename in filenames:
        with safe_open(directory / filename, framework="pt", device="cpu") as archive:
            for name in archive.keys():
                if decoder_only and name.startswith("encoder."):
                    continue
                elif name in weights:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                else:
                    weights[name] = archive.get_tensor(name)
    module.load_state_dict(weights, strict=True, assign=True)


@dataclass(kw_only=True)
class SongResult:
    audio: torch.Tensor
    sample_rate: int
    abc_tokens: list[int]
    codec_tokens: list[int]
    abc_truncated: bool
    semantic_truncated: bool

    @property
    def finish_reason(self) -> str:
        return "length" if self.abc_truncated or self.semantic_truncated else "stop"


class Yue2Runtime:
    def __init__(
        self,
        model: Yue2Model,
        vae: Yue2Vae,
        tokenizer: Yue2Tokenizer,
        *,
        vae_core_frames: int,
        vae_halo_frames: int,
        ar_cuda_graph: bool = False,
        nar_cuda_graph: bool = False,
        sampling_cuda_graph: bool = False,
        fuse_projections: bool = False,
    ) -> None:
        self.model = model
        self.decode_graphs: list[DecodeGraph] | None = [] if ar_cuda_graph else None
        self.batch_graphs: OrderedDict[tuple[int, int], DecodeGraph] = OrderedDict()
        self.nar_cuda_graph = nar_cuda_graph
        self.sampling_cuda_graph = sampling_cuda_graph
        if fuse_projections:
            for layer in model.model.layers:
                layer.self_attn.fuse_projections()
                layer.nar_self_attn.fuse_projections()
                layer.mlp.fuse_projections()
                layer.nar_mlp.fuse_projections()
        else:
            pass
        self.vae = vae
        self.tokenizer = tokenizer
        self.vae_core_frames = vae_core_frames
        self.vae_halo_frames = vae_halo_frames
        if vae_core_frames < 1 or vae_halo_frames < vae.required_halo(vae_core_frames):
            raise ValueError(
                "YuE2 VAE tile configuration does not cover the decoder receptive field"
            )
        else:
            pass

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: str,
        device: torch.device,
        *,
        vae_core_frames: int,
        vae_halo_frames: int,
        ar_cuda_graph: bool = False,
        nar_cuda_graph: bool = False,
        sampling_cuda_graph: bool = False,
        fuse_projections: bool = False,
    ) -> "Yue2Runtime":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        model_directory = Path(resolve_checkpoint(model_path))
        vae_directory = Path(resolve_checkpoint(vae_path))
        model_config = Yue2ModelConfig.model_validate_json(
            (model_directory / "config.json").read_text()
        )
        vae_config = VaeConfig.model_validate_json(
            (vae_directory / "config.json").read_text()
        )
        with torch.device("meta"):
            model = Yue2Model(model_config)
            vae = Yue2Vae(vae_config)
        load_weights(model, model_directory, decoder_only=False)
        load_weights(vae, vae_directory, decoder_only=True)
        model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        tokenizer = Yue2Tokenizer(model_directory / "qwen.tiktoken")
        return cls(
            model,
            vae,
            tokenizer,
            vae_core_frames=vae_core_frames,
            vae_halo_frames=vae_halo_frames,
            ar_cuda_graph=ar_cuda_graph,
            nar_cuda_graph=nar_cuda_graph,
            sampling_cuda_graph=sampling_cuda_graph,
            fuse_projections=fuse_projections,
        )

    @torch.inference_mode()
    def generate(
        self, request: Yue2Request, cancelled: CancellationCheck
    ) -> SongResult:
        check_cancelled(cancelled)
        options = request.options
        abc_truncated = False
        if options.cot == "off":
            abc_tokens = []
        elif options.abc is not None:
            abc_tokens = self.tokenizer.encode(options.abc)
        else:
            plan = generate_tokens(
                self.model,
                self.tokenizer.prefix(request, None),
                request.planning_sampling,
                request.seed,
                "abc",
                cancelled,
                decode_graphs=self.decode_graphs,
                sampling_cuda_graph=self.sampling_cuda_graph,
            )
            abc_tokens, abc_truncated = plan.tokens, plan.truncated
        prefix = self.tokenizer.prefix(request, abc_tokens)
        negative = (
            self.tokenizer.negative_prefix(request, abc_tokens)
            if request.guidance != 1
            else None
        )
        semantic = generate_tokens(
            self.model,
            prefix,
            request.semantic_sampling,
            request.seed,
            "semantic",
            cancelled,
            negative=negative,
            cfg_scale=request.guidance,
            legacy_off=options.cot == "off",
            decode_graphs=self.decode_graphs,
            sampling_cuda_graph=self.sampling_cuda_graph,
        )
        return self.render(
            request,
            prefix,
            TokenResult(tokens=abc_tokens, truncated=abc_truncated),
            semantic,
            cancelled,
        )

    @torch.inference_mode()
    def render(
        self,
        request: Yue2Request,
        prefix: list[int],
        planning: TokenResult,
        semantic: TokenResult,
        cancelled: CancellationCheck,
    ) -> SongResult:
        check_cancelled(cancelled)
        codec_tokens = [token - CODEC_OFFSET for token in semantic.tokens]
        latents = synthesize_latents(
            self.model,
            prefix,
            codec_tokens,
            request.seed,
            request.options.ode_steps,
            cancelled,
            cuda_graph=self.nar_cuda_graph,
        )
        audio = self.vae.decode_tiled(
            latents.T[None],
            self.vae_core_frames,
            self.vae_halo_frames,
            cancelled,
        )[0]
        check_cancelled(cancelled)
        return SongResult(
            audio=audio,
            sample_rate=self.vae.config.sample_rate,
            abc_tokens=planning.tokens,
            codec_tokens=codec_tokens,
            abc_truncated=planning.truncated,
            semantic_truncated=semantic.truncated,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        requests: list[Yue2Request],
        cancelled: list[CancellationCheck],
        graph_cache_size: int,
    ) -> list[SongResult | Exception]:
        assert len(requests) == len(cancelled) and graph_cache_size > 0
        results: dict[int, SongResult | Exception] = {}
        candidates: list[int] = []
        planning_inputs: list[TokenRequest] = []
        planning_indices: list[int] = []
        plans: dict[int, TokenResult] = {}
        for index, request in enumerate(requests):
            if cancelled[index]():
                results[index] = InterruptedError("YuE2 request cancelled")
            elif (
                len(requests) == 1
                or self.decode_graphs is None
                or request.guidance != 1
            ):
                try:
                    results[index] = self.generate(request, cancelled[index])
                except (ValueError, InterruptedError, FloatingPointError) as error:
                    results[index] = error
            else:
                candidates.append(index)
                if request.options.cot == "off":
                    plans[index] = TokenResult(tokens=[], truncated=False)
                elif request.options.abc is not None:
                    plans[index] = TokenResult(
                        tokens=self.tokenizer.encode(request.options.abc),
                        truncated=False,
                    )
                else:
                    planning_indices.append(index)
                    planning_inputs.append(
                        TokenRequest(
                            prefix=self.tokenizer.prefix(request, None),
                            sampling=request.planning_sampling,
                            seed=request.seed,
                            cancelled=cancelled[index],
                            legacy_off=False,
                        )
                    )
        planning_outputs = generate_batch_tokens(
            self.model, planning_inputs, "abc", self.batch_graphs, graph_cache_size
        )
        for index, planning in zip(planning_indices, planning_outputs):
            if isinstance(planning, Exception):
                results[index] = planning
            else:
                plans[index] = planning
        semantic_inputs: list[TokenRequest] = []
        semantic_indices: list[int] = []
        for index in candidates:
            if index in results:
                continue
            else:
                request = requests[index]
                semantic_indices.append(index)
                semantic_inputs.append(
                    TokenRequest(
                        prefix=self.tokenizer.prefix(request, plans[index].tokens),
                        sampling=request.semantic_sampling,
                        seed=request.seed,
                        cancelled=cancelled[index],
                        legacy_off=request.options.cot == "off",
                    )
                )
        semantic_outputs = generate_batch_tokens(
            self.model, semantic_inputs, "semantic", self.batch_graphs, graph_cache_size
        )
        for index, token_request, semantic in zip(
            semantic_indices, semantic_inputs, semantic_outputs
        ):
            if isinstance(semantic, Exception):
                results[index] = semantic
            else:
                try:
                    results[index] = self.render(
                        requests[index],
                        token_request.prefix,
                        plans[index],
                        semantic,
                        cancelled[index],
                    )
                except (ValueError, InterruptedError, FloatingPointError) as error:
                    results[index] = error
        return [results[index] for index in range(len(requests))]
