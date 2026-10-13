# SPDX-License-Identifier: Apache-2.0
"""Drama 3 as an SGLang model: the Qwen3.5 MoE backbone plus the codebook tower.

The tower keeps its key/value cache in the backbone's paged pool, under the
layer ids that follow the backbone's. One forward runs the backbone, fills the
tower cache of the extend rows, and samples the next frame of every request.
A decode forward reads fixed-address buffers, so a CUDA graph can replay it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.first_frame_graphs import (
    Drama3FirstFrameGraphs,
    FirstFrameInputs,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.model_inputs import (
    Drama3DecodeBuffers,
    Drama3StepInputs,
    StepExchange,
    next_frame_labels,
)
from sglang_omni.models.fishaudio_drama3.tower_prefill import (
    Drama3TowerPrefill,
    PagedTowerAttention,
)
from sglang_omni.models.fishaudio_drama3.weights import (
    checkpoint_part,
    load_audio_decoder_weights,
)

BACKBONE_CHECKPOINT_PREFIX = "model.language_model."
LM_HEAD_CHECKPOINT_NAME = "lm_head.weight"


__all__ = [
    "Drama3ForCausalLM",
    "Drama3Backbone",
    "Drama3DecodeBuffers",
    "Drama3StepInputs",
    "StepExchange",
    "next_frame_labels",
    "PagedTowerAttention",
    "Drama3TowerPrefill",
]


class Drama3Backbone(Qwen3_5MoeForCausalLM):
    tower_prefill: Drama3TowerPrefill | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = super().forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
            input_deepstack_embeds,
        )
        assert isinstance(hidden_states, torch.Tensor)
        tower_prefill = self.tower_prefill
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return hidden_states
        elif tower_prefill is None:
            raise RuntimeError("the engine allocated no Drama 3 tower prefill buffers")
        else:
            tower_prefill.run(hidden_states, positions, forward_batch)
            return hidden_states


class Drama3ForCausalLM(nn.Module):
    """Backbone, lm_head and codebook tower; one forward samples one frame per request."""

    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        text_config = config.text_config
        decoder_config = config.audio_decoder_config
        tower_layer_count = (
            decoder_config.shared_enrichment_layers + decoder_config.num_codebooks - 1
        )
        tower_geometry = (
            decoder_config.n_head,
            decoder_config.n_local_heads,
            decoder_config.head_dim,
        )
        backbone_geometry = (
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            text_config.head_dim,
        )
        if text_config.tie_word_embeddings:
            raise NotImplementedError("Drama 3 checkpoints keep a separate lm_head")
        elif tower_geometry != backbone_geometry:
            # The tower shares the backbone's KV pool and attention buffers.
            raise ValueError(
                f"decoder (n_head, n_local_heads, head_dim)={tower_geometry} must "
                f"equal the backbone's {backbone_geometry}"
            )
        elif text_config.tower_layer_count != tower_layer_count:
            raise ValueError(
                f"text_config.tower_layer_count={text_config.tower_layer_count} but "
                f"the decoder has {tower_layer_count} tower layers"
            )
        elif get_parallel().pp_size != 1 or get_parallel().tp_size != 1:
            raise NotImplementedError("Drama 3 runs with one tensor and pipeline rank")
        else:
            pass
        self.config = config
        self.model = Drama3Backbone(
            text_config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(text_config)
        self.audio_decoder = AudioDecoder(config)
        self.tower_attention = nn.ModuleList(
            RadixAttention(
                num_heads=decoder_config.n_head,
                head_dim=decoder_config.head_dim,
                scaling=decoder_config.head_dim**-0.5,
                num_kv_heads=decoder_config.n_local_heads,
                layer_id=text_config.num_hidden_layers + tower_index,
                prefix=add_prefix(f"tower_attention.{tower_index}", prefix),
            )
            for tower_index in range(tower_layer_count)
        )
        self.step_exchange = StepExchange()
        # The engine builder allocates these once the request pool is sized.
        self.decode_buffers: Drama3DecodeBuffers | None = None
        self.first_frame_graphs: Drama3FirstFrameGraphs | None = None

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        """One past the last tower layer id, so the KV pool covers the tower."""
        return self.model.end_layer + len(self.tower_attention)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        forward_mode = forward_batch.forward_mode
        decode_buffers = self.decode_buffers
        if forward_mode == ForwardMode.EXTEND:
            return self.forward_extend(input_ids, positions, forward_batch)
        elif forward_mode != ForwardMode.DECODE:
            raise NotImplementedError(f"forward mode {forward_mode} is not supported")
        elif decode_buffers is None:
            raise RuntimeError("the engine allocated no Drama 3 decode buffers")
        else:
            # A CUDA graph replays this branch with the batch padded to its size.
            row_count = input_ids.shape[0]
            frames = decode_buffers.frames[:row_count]
            hidden_states, logits_output = self.backbone_logits(
                input_ids, positions, forward_batch, frames
            )
            sampled_frames = self.audio_decoder.decode_step(
                logits_output.next_token_logits,
                hidden_states,
                positions,
                frames,
                decode_buffers.ras_windows[:row_count],
                decode_buffers.temperature[:row_count],
                decode_buffers.top_k[:row_count],
                decode_buffers.top_p[:row_count],
                PagedTowerAttention(self.tower_attention, forward_batch),
            )
            decode_buffers.sampled_frames[:row_count].copy_(sampled_frames)
            return logits_output

    def forward_extend(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Fill the tower cache of every extend row, then sample each first frame."""
        row_count = forward_batch.extend_num_tokens
        step_inputs = self.step_exchange.take_inputs(forward_batch.rids, row_count)
        frames = step_inputs.frames
        last_rows = forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
        tower_prefill = self.model.tower_prefill
        if tower_prefill is None:
            raise RuntimeError("the engine allocated no Drama 3 tower prefill buffers")
        else:
            tower_prefill.stage(
                frames, next_frame_labels(frames, last_rows, self.config.eos_token_id)
            )
        hidden_states, logits_output = self.backbone_logits(
            input_ids, positions, forward_batch, frames
        )
        hidden_states = hidden_states[:row_count]
        positions = positions[:row_count]
        first_frame_graphs = self.first_frame_graphs
        request_count = forward_batch.batch_size
        if (
            first_frame_graphs is not None
            and request_count <= first_frame_graphs.max_request_count
        ):
            self.step_exchange.publish(
                first_frame_graphs.replay(
                    FirstFrameInputs(
                        semantic_logits=logits_output.next_token_logits,
                        hidden_states=hidden_states[last_rows],
                        positions=positions[last_rows],
                        input_frames=frames[last_rows],
                        previous_main_tokens=step_inputs.ras_windows,
                        temperature=step_inputs.temperature,
                        top_k=step_inputs.top_k,
                        top_p=step_inputs.top_p,
                        request_pool_indices=forward_batch.req_pool_indices,
                        sequence_lengths=forward_batch.seq_lens,
                        cache_slots=forward_batch.out_cache_loc[last_rows],
                    )
                )
            )
            return logits_output
        else:
            pass
        # The backbone is done with the extend metadata, so the full-attention
        # backend is re-planned for one newest row per request.
        step_batch = dataclasses.replace(
            forward_batch,
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=forward_batch.out_cache_loc[last_rows],
            positions=positions[last_rows],
        )
        attention_backend = get_attn_backend()
        assert isinstance(attention_backend, HybridLinearAttnBackend)
        attention_backend.full_attn_backend.init_forward_metadata(step_batch)
        sampled_frames = self.sample_first_frames(
            logits_output.next_token_logits,
            hidden_states[last_rows],
            step_batch.positions,
            frames[last_rows],
            step_inputs.ras_windows,
            step_inputs.temperature,
            step_inputs.top_k,
            step_inputs.top_p,
            step_batch,
        )
        self.step_exchange.publish(sampled_frames)
        return logits_output

    def sample_first_frames(
        self,
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return self.audio_decoder.decode_step(
            semantic_logits,
            hidden_states,
            positions,
            input_frames,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            PagedTowerAttention(self.tower_attention, forward_batch),
        )

    def backbone_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, LogitsProcessorOutput]:
        """Final-norm hidden states of every row, and the logits SGLang keeps."""
        input_embeds = self.audio_decoder.embed_backbone(
            frames, self.model.embed_tokens(frames[:, 0])
        )
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        return hidden_states, self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        decoder_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_weights: list[torch.Tensor] = []

        # Streams the backbone tensors so the checkpoint is never held whole.
        def backbone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for checkpoint_name, tensor in weights:
                if checkpoint_part(checkpoint_name) == "audio_decoder":
                    decoder_weights.append((checkpoint_name, tensor))
                elif checkpoint_name == LM_HEAD_CHECKPOINT_NAME:
                    lm_head_weights.append(tensor)
                elif checkpoint_name.startswith(BACKBONE_CHECKPOINT_PREFIX):
                    backbone_name = checkpoint_name.removeprefix(
                        BACKBONE_CHECKPOINT_PREFIX
                    )
                    yield backbone_name, tensor
                else:
                    raise ValueError(
                        f"checkpoint key {checkpoint_name!r} is not a backbone weight"
                    )

        self.model.load_weights(backbone_weights())
        if len(lm_head_weights) != 1:
            raise ValueError(
                f"the checkpoint holds {len(lm_head_weights)} {LM_HEAD_CHECKPOINT_NAME} "
                f"tensors; expected one"
            )
        else:
            pass
        self.lm_head.weight_loader(self.lm_head.weight, lm_head_weights[0])
        load_audio_decoder_weights(self.audio_decoder, decoder_weights)
