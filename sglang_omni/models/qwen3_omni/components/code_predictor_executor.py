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
            do_sample=True,
            top_k=50,
            top_p=0.8,
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


def _load_talker_model(model_path: str, gpu_id: int = 0):
    """Load the HF talker model for exact code predictor parity."""
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe import (
        modeling_qwen3_omni_moe as hf_modeling,
    )

    from sglang_omni.models.weight_loader import load_module

    device = f"cuda:{gpu_id}"
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = hf_modeling.Qwen3OmniMoeTalkerForConditionalGeneration._from_config(
        cfg.talker_config
    )
    model = load_module(
        model,
        model_path,
        prefix="talker.",
        dtype=torch.bfloat16,
        device=device,
        strict=False,
    )
    torch.manual_seed(123)
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
        self._chunk_mailbox: Any | None = None
        self._chunk_senders: list[Any] = []

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        if self._chunk_mailbox is None:
            raise RuntimeError("Code predictor requires a chunk mailbox")

        loop = asyncio.get_running_loop()
        chunk_count = 0

        while True:
            if request_id in self._aborted:
                break

            item = await self._chunk_mailbox.get(request_id)
            if item is None:
                break

            model_inputs = {
                "talker_hidden": item.tensor.to(device=self._device),
                "layer0_code": torch.tensor(
                    item.metadata["codec_code"],
                    dtype=torch.long,
                    device=self._device,
                ),
            }
            output = await loop.run_in_executor(None, self._run_model, model_inputs)
            self._dispatch_outputs(request_id, output)
            chunk_count += 1

        for sender in self._chunk_senders:
            if hasattr(sender, "enqueue_chunks_done"):
                sender.enqueue_chunks_done(request_id)

        payload.data = {"chunk_count": chunk_count}
        await self._results.put(payload)

    def _dispatch_outputs(
        self, request_id: str, output: dict[str, torch.Tensor]
    ) -> None:
        codes = output["codes"]
        summed_embeddings = output["summed_embeddings"]

        for sender in self._chunk_senders:
            to_stage = getattr(sender, "_to_stage", None)
            if to_stage == TALKER_AR_STAGE:
                sender.enqueue(request_id, summed_embeddings)
            elif to_stage == CODE2WAV_STAGE or to_stage is None:
                sender.enqueue(request_id, codes)
            else:
                sender.enqueue(request_id, codes)

    @torch.no_grad()
    def _run_model(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        return self._model(**inputs)

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)


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
