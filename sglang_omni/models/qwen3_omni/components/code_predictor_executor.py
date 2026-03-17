# SPDX-License-Identifier: Apache-2.0
"""Code Predictor executor factory — streaming RVQ + feedback generator.

Consumes chunks one-by-one from Talker (via chunk mailbox), runs code predictor
forward for each chunk to produce 16-layer RVQ codes, and streams each result
to both Code2Wav and Talker feedback.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch
from torch import nn

from sglang_omni.executors.interface import Executor
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    CODE2WAV_STAGE,
    TALKER_AR_STAGE,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


class _CodePredictorWrapper(nn.Module):
    """Wrap the HF talker to generate RVQ codes via the reference AR loop.

    Processes a single chunk: one hidden state + one layer-0 codec code.
    This mirrors HF's talker.prepare_inputs_for_generation() path so feedback
    embeddings match the reference implementation.
    """

    def __init__(self, talker_model: nn.Module):
        super().__init__()
        self._talker = talker_model

    def forward(
        self, talker_hidden: torch.Tensor, layer0_code: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the HF code predictor on a single talker hidden state.

        Args:
            talker_hidden: [hidden_size] single position
            layer0_code: scalar layer-0 codec code

        Returns:
            {"codes": [num_code_groups], "summed_embeddings": [hidden_size]}
        """
        hidden = talker_hidden.unsqueeze(0).unsqueeze(0)
        codes_input = layer0_code.reshape(1, 1)
        layer0_embed = self._talker.get_input_embeddings()(codes_input)

        predictor_result = self._talker.code_predictor.generate(
            inputs_embeds=torch.cat((hidden, layer0_embed), dim=1),
            max_new_tokens=self._talker.config.num_code_groups - 1,
            do_sample=False,
            temperature=0.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        result_codes = torch.cat(
            (codes_input, predictor_result.sequences.to(codes_input.device)),
            dim=-1,
        )
        mid_residual_hiddens = [
            hid[0].to(layer0_embed.device) for hid in predictor_result.hidden_states[1:]
        ]
        last_residual_hidden = self._talker.code_predictor.get_input_embeddings()[-1](
            predictor_result.sequences[..., -1:]
        ).to(layer0_embed.device)
        codec_hiddens = torch.cat(
            [layer0_embed] + mid_residual_hiddens + [last_residual_hidden], dim=1
        )
        summed_embeddings = codec_hiddens.sum(1)

        return {
            "codes": result_codes[0],  # [num_code_groups]
            "summed_embeddings": summed_embeddings[0],  # [hidden_size]
        }


class _LightweightTalkerShell(nn.Module):
    """Minimal talker shell that exposes only what _CodePredictorWrapper needs.

    Provides the same interface as the full HF
    ``Qwen3OmniMoeTalkerForConditionalGeneration`` but contains only:
    - ``model.codec_embedding`` — single nn.Embedding (talker layer-0)
    - ``code_predictor`` — full HF CodePredictor (with ``.generate()``)
    - ``config`` — talker config (for ``num_code_groups``)

    This avoids loading the full ~6 GB talker model when only the code
    predictor sub-module (~0.4 GB) is needed.
    """

    def __init__(self, config, code_predictor: nn.Module, codec_embedding: nn.Module):
        super().__init__()
        self.config = config
        self.code_predictor = code_predictor
        # Wrap codec_embedding in a sub-module that mirrors talker.model structure
        self.model = nn.Module()
        self.model.codec_embedding = codec_embedding

    def get_input_embeddings(self):
        return self.model.codec_embedding


def _load_talker_model(model_path: str, gpu_id: int = 0):
    """Load only the code predictor + layer-0 codec embedding from the checkpoint.

    Instead of loading the full ~6 GB HF talker model, this creates a lightweight
    shell containing only the components that ``_CodePredictorWrapper`` needs:
    - ``talker.model.codec_embedding`` — single nn.Embedding (~0.004 GB)
    - ``talker.code_predictor.*`` — 5 dense layers + 15 embeddings + 15 heads (~0.4 GB)
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe import (
        modeling_qwen3_omni_moe as hf_modeling,
    )

    from sglang_omni.models.weight_loader import load_module, load_weights_by_prefix

    device = f"cuda:{gpu_id}"
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 1. Create lightweight HF code predictor (with .generate() support)
    code_predictor = hf_modeling.Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
        cfg.talker_config.code_predictor_config
    )
    code_predictor = load_module(
        code_predictor,
        model_path,
        prefix="talker.code_predictor.",
        dtype=torch.bfloat16,
        device=device,
        strict=False,
    )

    # 2. Create and load the talker's layer-0 codec embedding
    text_cfg = cfg.talker_config.text_config
    codec_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size)
    emb_weights = load_weights_by_prefix(
        model_path, prefix="talker.model.codec_embedding."
    )
    codec_embedding.load_state_dict(emb_weights, strict=True, assign=True)
    codec_embedding = codec_embedding.to(device=device, dtype=torch.bfloat16)

    # 3. Assemble the lightweight shell
    model = _LightweightTalkerShell(
        config=cfg.talker_config,
        code_predictor=code_predictor,
        codec_embedding=codec_embedding,
    )
    model.eval()
    return model


class _CodePredictorStreamingExecutor(Executor):
    """Stream talker hidden states through the RVQ predictor.

    Sends:
    - `codes` to `code2wav`
    - `summed_embeddings` to `talker_ar` as feedback
    """

    def __init__(self, model: nn.Module, device: str | torch.device):
        self._model = model
        self._device = torch.device(device)
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()
        self._stream_queue: Any | None = None  # Set by compiler
        self._stream_fn: Any | None = None  # Set by compiler

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        if self._stream_queue is None:
            raise RuntimeError("Code predictor requires a stream queue")

        loop = asyncio.get_running_loop()
        chunk_count = 0

        while True:
            if request_id in self._aborted:
                break

            item = await self._stream_queue.get(request_id)
            if item is None:
                break

            model_inputs = {
                "talker_hidden": item.data.to(device=self._device),
                "layer0_code": torch.tensor(
                    item.metadata["codec_code"],
                    dtype=torch.long,
                    device=self._device,
                ),
            }
            output = await loop.run_in_executor(None, self._run_model, model_inputs)
            self._dispatch_outputs(request_id, output)
            chunk_count += 1

        # EOS is signaled by the Worker after executor completes

        payload.data = {"chunk_count": chunk_count}
        await self._results.put(payload)

    def _dispatch_outputs(
        self, request_id: str, output: dict[str, torch.Tensor]
    ) -> None:
        if self._stream_fn is None:
            return
        codes = output["codes"]
        summed_embeddings = output["summed_embeddings"]
        self._stream_fn(request_id, summed_embeddings, target_stage=TALKER_AR_STAGE)
        self._stream_fn(request_id, codes, target_stage=CODE2WAV_STAGE)

    @torch.no_grad()
    def _run_model(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        return self._model(**inputs)

    def set_stream_fn(self, fn) -> None:
        """Set the streaming output callback. Sync, non-blocking."""
        self._stream_fn = fn

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        # Unblock any pending get() on the stream queue
        if self._stream_queue is not None:
            try:
                self._stream_queue.close(request_id)
            except Exception:
                pass


def create_code_predictor_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
) -> Executor:
    """Code Predictor executor — streaming mode.

    Consumes chunks one-by-one from Talker's chunk mailbox.
    Each chunk contains one hidden state + one codec_code.
    Forwards through lm_heads to produce [16] RVQ codes,
    then enqueues the codes as a chunk to Code2Wav.
    """
    device = f"cuda:{gpu_id}"
    model = _load_talker_model(model_path, gpu_id=gpu_id)
    wrapper = _CodePredictorWrapper(model)
    return _CodePredictorStreamingExecutor(model=wrapper, device=device)


def create_code_predictor_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    code_predictor_max_seq_len: int = 256,
    server_args_overrides: dict[str, Any] | None = None,
) -> Executor:
    """Create Code Predictor executor from config args."""
    return create_code_predictor_executor(
        model_path=model_path,
        gpu_id=gpu_id,
    )
