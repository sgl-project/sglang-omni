# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Ming 3 ASR."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import torch
from pydantic import JsonValue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import AutoConfig, PreTrainedTokenizerFast

from sglang_omni.models.ming3_asr.encoder_service import Ming3ASRPreLMEncoderService
from sglang_omni.models.ming3_asr.preprocessing import (
    Ming3ASRPreprocessor,
    TranscriptionState,
    extract_transcript,
)
from sglang_omni.models.ming3_asr.text_streaming import (
    Ming3ASRRequestData,
    Ming3ASRStreamOutputBuilder,
)
from sglang_omni.modules.ming_audio.audio_encoder import (
    AudioEncoderConfig,
    Ming3AudioEncoder,
)
from sglang_omni.modules.ming_audio.hf_config import Ming3AudioConfig
from sglang_omni.modules.ming_audio.thinker import (
    Ming3AudioModelRunner,
    Ming3AudioThinker,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.speculative.dspark_checkpoint import materialize_draft_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

DRAFT_TOKENS_PER_STEP = 8
VERIFY_TOKENS_PER_STEP = DRAFT_TOKENS_PER_STEP + 1


def prepare_dspark_checkpoint(
    overrides: dict[str, JsonValue], device: torch.device
) -> TemporaryDirectory[str] | None:
    """Select the trained eight-token draft before allocating model resources."""
    algorithm = overrides.get("speculative_algorithm")
    draft_model_path = overrides.get("speculative_draft_model_path")
    draft_revision = overrides.get("speculative_draft_model_revision")
    if algorithm is None:
        return None
    elif algorithm != "DSPARK":
        raise ValueError("Ming3-ASR speculative decoding supports DSPARK only")
    elif device.type != "cuda":
        raise ValueError("Ming3-ASR DSpark requires one CUDA GPU")
    elif not isinstance(draft_model_path, str) or not draft_model_path:
        raise ValueError("Ming3-ASR DSpark requires speculative_draft_model_path")
    elif draft_revision is not None and not isinstance(draft_revision, str):
        raise ValueError("speculative_draft_model_revision must be a string")
    elif (
        overrides.get("speculative_dspark_block_size", DRAFT_TOKENS_PER_STEP)
        != DRAFT_TOKENS_PER_STEP
        or overrides.get("speculative_num_draft_tokens", VERIFY_TOKENS_PER_STEP)
        != VERIFY_TOKENS_PER_STEP
    ):
        raise ValueError(
            "Ming3-ASR DSpark requires eight proposals and nine verify tokens"
        )
    else:
        draft_checkpoint = materialize_draft_checkpoint(
            draft_model_path, draft_revision
        )
        overrides.update(
            speculative_draft_model_path=draft_checkpoint.name,
            speculative_draft_model_revision=None,
            speculative_dspark_block_size=DRAFT_TOKENS_PER_STEP,
            speculative_num_draft_tokens=VERIFY_TOKENS_PER_STEP,
            disable_overlap_schedule=True,
            enable_mixed_chunk=False,
        )
        return draft_checkpoint


class Ming3ASRConfig(Ming3AudioConfig):
    model_type: ClassVar[str] = "ming_v3_asr"

    # note (Codex): transformers replaces the constructor of a config subclass
    # that does not define one, which would skip flattening text_config.
    def __init__(
        self,
        text_config: dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        fields = dict(text_config or {})
        # note (wirybeaver): ASR normalizes selected expert probabilities even when the checkpoint omits this flag.
        fields.setdefault("norm_topk_prob", True)
        super().__init__(fields, **kwargs)
        self.architectures = ["BailingMoeV3ForCausalLM"]


def load_checkpoint_config(model_path: str) -> dict[str, JsonValue]:
    return json.loads((Path(model_path) / "config.json").read_text())


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    configuration = load_checkpoint_config(model_path)
    # note (Codex): the checkpoint's tokenizer class only adds chat helpers
    # to the fast tokenizer, so the fast tokenizer encodes identically.
    preprocessor = Ming3ASRPreprocessor(
        PreTrainedTokenizerFast.from_pretrained(model_path),
        AudioEncoderConfig.model_validate(configuration["audio_config"]),
        configuration["text_config"]["hidden_size"],
    )
    return SimpleScheduler(preprocessor)


def create_thinker_executor(
    model_path: str,
    *,
    max_seq_len: int,
    max_new_tokens: int,
    stream_emit_interval_seconds: float,
    pre_lm_max_batch_size: int,
    pre_lm_max_batch_wait_ms: int,
    request_build_max_workers: int,
    request_build_max_pending: int,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    server_args_overrides: dict[str, JsonValue] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler[Ming3ASRRequestData]:
    if tp_size != 1:
        raise ValueError(f"Ming ASR thinker supports tp_size 1, got {tp_size}")
    else:
        pass
    if pre_lm_max_batch_size < 1:
        raise ValueError(
            f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
        )
    else:
        pass
    if pre_lm_max_batch_wait_ms < 0:
        raise ValueError(
            f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
        )
    else:
        pass
    overrides = dict(server_args_overrides or {})
    resolved_device = resolve_concrete_device(device, gpu_id)
    with ExitStack() as resources:
        draft_checkpoint = prepare_dspark_checkpoint(overrides, resolved_device)
        if draft_checkpoint is not None:
            resources.enter_context(draft_checkpoint)
        else:
            pass
        configuration = load_checkpoint_config(model_path)
        encoder = Ming3AudioEncoder(
            AudioEncoderConfig.model_validate(configuration["audio_config"]),
            configuration["text_config"]["hidden_size"],
            should_upcast_layer_norm=True,
            should_batch_chunks=False,
        )
        encoder.load_checkpoint(model_path)
        encoder = encoder.eval().to(resolved_device, dtype=torch.bfloat16)
        AutoConfig.register(Ming3ASRConfig.model_type, Ming3ASRConfig, exist_ok=True)
        ModelRegistry.models["BailingMoeV3ForCausalLM"] = Ming3AudioThinker
        overrides.update(
            tp_size=tp_size,
            trust_remote_code=False,
            skip_tokenizer_init=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
        )
        server_args = build_sglang_server_args(
            model_path, context_length=max_seq_len, **overrides
        )
        worker, tree_cache, request_pool, token_pool, model_config = (
            create_sglang_infrastructure(
                server_args,
                resolved_device.index or 0,
                tp_rank=tp_rank,
                nccl_port=nccl_port,
                total_gpu_memory_fraction=total_gpu_memory_fraction,
                supports_speculative_decoding=draft_checkpoint is not None,
            )
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        end_token_id = configuration["text_config"]["eos_token_id"]
        encoder_service = Ming3ASRPreLMEncoderService(
            encoder,
            max_batch_size=pre_lm_max_batch_size,
            max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        )

        resources.callback(encoder_service.close)

        def build_request(payload: StagePayload) -> Ming3ASRRequestData:
            state: TranscriptionState = payload.data
            if state["features"]:
                state["audio_embeddings"] = encoder_service.encode(state["features"])
                state["features"] = []
            else:
                pass
            parameters = payload.request.params
            temperature = parameters.get("temperature", 0.0)
            if draft_checkpoint is not None and temperature != 0.0:
                raise ValueError("Ming3-ASR DSpark currently requires temperature=0")
            else:
                pass
            sampling = SamplingParams(
                temperature=temperature,
                max_new_tokens=parameters.get("max_new_tokens", max_new_tokens),
                top_p=parameters.get("top_p", 1.0),
                top_k=parameters.get("top_k", -1),
                repetition_penalty=parameters.get("repetition_penalty", 1.0),
                sampling_seed=parameters.get("seed"),
            )
            sampling.normalize(tokenizer)
            sampling.verify(model_config.vocab_size)
            request = Req(
                rid=payload.request_id,
                origin_input_text="",
                origin_input_ids=state["input_ids"].tolist(),
                sampling_params=sampling,
                vocab_size=model_config.vocab_size,
                eos_token_ids={end_token_id},
            )
            return Ming3ASRRequestData(
                input_ids=state["input_ids"],
                req=request,
                model_inputs={
                    "audio_positions": state["audio_positions"],
                    "audio_embeddings": state["audio_embeddings"],
                },
                max_new_tokens=sampling.max_new_tokens,
                temperature=sampling.temperature,
                output_ids=request.output_ids,
                stage_payload=payload,
                language=parameters.get("language"),
            )

        def build_result(request: Ming3ASRRequestData) -> StagePayload:
            payload = request.stage_payload
            generated_text = tokenizer.decode(
                request.output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            payload.data = {
                "text": extract_transcript(generated_text, language=request.language),
                "finish_reason": request.finish_reason,
                "spec_tokens_details": request.spec_tokens_details,
                "usage": {
                    "prompt_tokens": len(request.input_ids),
                    "completion_tokens": len(request.output_ids),
                },
            }
            return payload

        scheduler = OmniScheduler(
            tp_worker=worker,
            tree_cache=tree_cache,
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=token_pool,
            server_args=server_args,
            model_config=model_config,
            model_runner=Ming3AudioModelRunner(worker, SGLangOutputProcessor()),
            request_builder=build_request,
            result_adapter=build_result,
            request_build_max_workers=request_build_max_workers,
            request_build_max_pending=request_build_max_pending,
            stream_output_builder=Ming3ASRStreamOutputBuilder(
                tokenizer, interval_seconds=stream_emit_interval_seconds
            ),
        )
        scheduler.shutdown_callback = resources.pop_all().close
        return scheduler
