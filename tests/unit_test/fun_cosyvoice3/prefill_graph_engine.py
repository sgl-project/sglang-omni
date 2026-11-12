# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3's TTS engine on a two layer synthetic checkpoint, run in its own process
by test_model_runner.py: every prompt prefill is recorded beside the eager forward over
each request's enqueued prompt rows in its extend window."""

from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import Qwen2Config, Qwen2ForCausalLM

import sglang_omni.models.fun_cosyvoice3.engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3NullTokenizer,
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    CONTROL_TOKEN_IDS,
    TOTAL_VOCAB_SIZE,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

HIDDEN_SIZE = 128
TEXT_VOCAB_SIZE = 1000
GENERATED_TOKEN_COUNT = 2
FIRST_PROMPT_TOKEN_COUNT = 37
SECOND_PROMPT_TOKEN_COUNT = 23
# note (ratish): the prompt keys lie outside both embedding tables, as a composed
# prompt's radix keys do, so only the composed rows can reach the transformer.
FIRST_PROMPT_KEY = 10_000_000
NULL_TOKENIZER = CosyVoice3NullTokenizer()


@dataclass(kw_only=True)
class PrefillRecord:
    extend_token_counts: list[int]
    prefix_token_counts: list[int]
    is_graph_replay: bool
    graph_hidden_states: torch.Tensor
    eager_hidden_states: torch.Tensor


class UnusedPreprocessingModel:
    """The text tokenizer and the ONNX sessions, which serve the preprocessing stage
    this engine is not given."""

    def __init__(
        self, model_path: str, *, device: str = "cpu", intra_op_threads: int = 1
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.intra_op_threads = intra_op_threads

    def extract_speech_token(self, audio: np.ndarray) -> torch.Tensor:
        return torch.zeros((1, 0), dtype=torch.int32)


def write_synthetic_checkpoint(checkpoint_directory: Path) -> None:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2 * HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=FunCosyVoice3EngineBuilder.context_length,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(config).save_pretrained(checkpoint_directory / "CosyVoice-BlankEN")
    torch.save(
        {
            "speech_embedding.weight": torch.randn(TOTAL_VOCAB_SIZE, HIDDEN_SIZE),
            "llm_decoder.weight": torch.randn(TOTAL_VOCAB_SIZE, HIDDEN_SIZE)
            * HIDDEN_SIZE**-0.5,
        },
        checkpoint_directory / "llm.pt",
    )


def record_prefills_beside_eager(
    model_runner: FunCosyVoice3ModelRunner,
    prompt_rows_by_request: dict[str, torch.Tensor],
    records: list[PrefillRecord],
) -> None:
    forward_batch_generation = model_runner.tp_worker.forward_batch_generation

    def forward_beside_eager(forward_batch: ForwardBatch) -> GenerationBatchResult:
        if not forward_batch.forward_mode.is_extend():
            return forward_batch_generation(forward_batch)
        else:
            pass
        # note (ratish): the eager forward reads each request's enqueued rows, not the
        # rows the prefill hands on, and runs first so the served forward's KV is kept.
        eager_input_embeds = torch.cat(
            [
                prompt_rows_by_request[request_id][
                    prefix_token_count : prefix_token_count + extend_token_count
                ]
                for request_id, prefix_token_count, extend_token_count in zip(
                    forward_batch.rids,
                    forward_batch.extend_prefix_lens_cpu,
                    forward_batch.extend_seq_lens_cpu,
                )
            ]
        )
        eager_result = model_runner.forward_with_input_embeds(
            copy.copy(forward_batch), eager_input_embeds
        )
        served_result = forward_batch_generation(forward_batch)
        records.append(
            PrefillRecord(
                extend_token_counts=list(forward_batch.extend_seq_lens_cpu),
                prefix_token_counts=list(forward_batch.extend_prefix_lens_cpu),
                is_graph_replay=bool(served_result.can_run_cuda_graph),
                graph_hidden_states=served_result.logits_output.hidden_states.cpu(),
                eager_hidden_states=eager_result.logits_output.hidden_states.cpu(),
            )
        )
        return served_result

    model_runner.tp_worker.forward_batch_generation = forward_beside_eager


def enqueue_prompt(
    scheduler: OmniScheduler[CosyVoice3SGLangRequestData],
    request_id: str,
    prompt_keys: list[int],
    prompt_input_embeds: torch.Tensor,
) -> None:
    sampling_params = SamplingParams(
        max_new_tokens=GENERATED_TOKEN_COUNT,
        min_new_tokens=GENERATED_TOKEN_COUNT,
        temperature=0.0,
        stop_token_ids=list(CONTROL_TOKEN_IDS),
    )
    sampling_params.normalize(NULL_TOKENIZER)
    sampling_params.verify(TOTAL_VOCAB_SIZE)
    request = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=prompt_keys,
        sampling_params=sampling_params,
        vocab_size=TOTAL_VOCAB_SIZE,
    )
    request.tokenizer = NULL_TOKENIZER
    payload = StagePayload(
        request_id=request_id, request=OmniRequest(inputs=""), data={}
    )
    scheduler.enqueue_built_request(
        payload,
        False,
        CosyVoice3SGLangRequestData(
            req=request,
            input_ids=torch.tensor(prompt_keys),
            prompt_input_embeds=prompt_input_embeds,
            output_ids=request.output_ids,
            max_new_tokens=GENERATED_TOKEN_COUNT,
            stage_payload=payload,
        ),
    )


def run_until_idle(scheduler: OmniScheduler[CosyVoice3SGLangRequestData]) -> None:
    batch = scheduler.get_next_batch_to_run()
    while batch is not None:
        batch_result = scheduler.run_batch(batch)
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            if message.type == "error":
                raise RuntimeError(f"request {message.request_id} failed") from (
                    message.data
                )
            else:
                pass
        scheduler.process_batch_result(batch, batch_result)
        scheduler.last_batch = batch
        batch = scheduler.get_next_batch_to_run()
    scheduler.last_batch = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_synthetic_checkpoint(arguments.checkpoint_directory)
    engine_builder_module.CosyVoice3Tokenizer = UnusedPreprocessingModel
    engine_builder_module.SpeechTokenizerV3 = UnusedPreprocessingModel
    engine_builder_module.SpeakerEncoder = UnusedPreprocessingModel
    scheduler = FunCosyVoice3EngineBuilder().build(
        str(arguments.checkpoint_directory), device="cuda", gpu_id=0
    )
    model_runner = scheduler.model_runner
    prompt_token_count = FIRST_PROMPT_TOKEN_COUNT + SECOND_PROMPT_TOKEN_COUNT
    prompt_keys = list(range(FIRST_PROMPT_KEY, FIRST_PROMPT_KEY + prompt_token_count))
    text_ids = torch.randint(
        TEXT_VOCAB_SIZE,
        (prompt_token_count,),
        generator=torch.Generator().manual_seed(1),
    )
    with torch.no_grad():
        prompt_rows = model_runner.model.text_embed_tokens(text_ids.cuda())
    first_keys = prompt_keys[:FIRST_PROMPT_TOKEN_COUNT]
    first_rows = prompt_rows[:FIRST_PROMPT_TOKEN_COUNT]
    second_rows = prompt_rows[FIRST_PROMPT_TOKEN_COUNT:]
    records: list[PrefillRecord] = []
    record_prefills_beside_eager(
        model_runner,
        {
            "first": first_rows,
            "second": second_rows,
            "first-cached": first_rows,
            "first-extended": prompt_rows,
        },
        records,
    )
    try:
        enqueue_prompt(scheduler, "first", first_keys, first_rows)
        enqueue_prompt(
            scheduler, "second", prompt_keys[FIRST_PROMPT_TOKEN_COUNT:], second_rows
        )
        run_until_idle(scheduler)
        enqueue_prompt(scheduler, "first-cached", first_keys, first_rows)
        run_until_idle(scheduler)
        enqueue_prompt(scheduler, "first-extended", prompt_keys, prompt_rows)
        run_until_idle(scheduler)
    finally:
        scheduler.shutdown_request_build_executor()
        scheduler.shutdown_resources()
    torch.save([dataclasses.asdict(record) for record in records], arguments.output)


if __name__ == "__main__":
    main()
