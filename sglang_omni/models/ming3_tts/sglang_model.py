# SPDX-License-Identifier: Apache-2.0
"""Ling V3 speech backbone with the Ming 3 continuous-latent speech heads."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

from sglang_omni.models.ming3_tts.acoustic import (
    FLOW_STEPS,
    AcousticGraph,
    AcousticInputs,
    AcousticStep,
    FlowConstants,
    PackedProjection,
    build_flow_constants,
    can_fuse_flow_head,
    compute_acoustic_step,
    pack_projections,
)
from sglang_omni.models.ming3_tts.model_config import (
    Ming3DenseTTSConfig,
    Ming3TTSConfig,
    install_speech_heads,
)
from sglang_omni.models.ming3_tts.native_acoustic import NativeAcousticCapture
from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss

logger = logging.getLogger(__name__)

BACKBONE_WEIGHT_PREFIX = "model."
VOCODER_WEIGHT_PREFIX = "audio."
AGGREGATOR_WEIGHT_PREFIX = "linear_proj_audio."
SPEECH_HEAD_PREFIXES = (
    "flowloss.",
    AGGREGATOR_WEIGHT_PREFIX,
    "stop_head.",
    "spk_head.",
)
GUIDANCE_SCALE = 2.0
SOLVER_NOISE_SCALE = 0.25
ACOUSTIC_GRAPH_WARMUP_STEPS = 2
# Acoustic graphs are captured for every batch size up to this one.
EXACT_ACOUSTIC_BUCKETS = 16


class SpeechHeadModel:
    """Flow, stop, and feedback heads shared by the MoE and dense backbones."""

    enable_fused_flow_head: bool
    flow_head_path_logged: bool
    weights_loaded: bool
    flowloss: FlowLoss
    linear_proj_audio: Aggregator
    stop_head: nn.Linear
    spk_head: nn.Linear
    flow_timesteps: torch.Tensor
    acoustic_graphs: dict[int, AcousticGraph]
    flow_head_projections: tuple[list[PackedProjection], list[PackedProjection]] | None
    native_capture: NativeAcousticCapture | None = None
    decode_feedback: nn.Embedding
    decode_feedback_rows: torch.Tensor
    prefill_inputs: nn.Embedding
    prefill_input_rows: torch.Tensor
    config: Ming3TTSConfig | Ming3DenseTTSConfig

    def stage_decode_feedback(self, feedback_embeddings: torch.Tensor) -> torch.Tensor:
        """Write one feedback row per request and return the row ids decode reads."""
        batch_size = feedback_embeddings.shape[0]
        self.decode_feedback.weight[:batch_size].copy_(feedback_embeddings)
        return self.decode_feedback_rows[:batch_size]

    def compute_acoustic_step(
        self,
        inputs: AcousticInputs,
        sampling_scales: torch.Tensor,
        constants: FlowConstants | None,
    ) -> AcousticStep:
        return compute_acoustic_step(
            self.flowloss,
            self.linear_proj_audio,
            self.stop_head,
            self.flow_timesteps,
            inputs,
            sampling_scales,
            constants,
        )

    def build_flow_constants(self, batch_size: int) -> FlowConstants | None:
        """Return None when the flow head must run its own module forwards."""
        use_fused = self.enable_fused_flow_head and can_fuse_flow_head(
            self.flowloss, self.linear_proj_audio
        )
        if not self.flow_head_path_logged:
            logger.info(
                f"Ming 3 TTS acoustic flow head: {'fused' if use_fused else 'module'}"
            )
            self.flow_head_path_logged = True
        else:
            pass
        if not use_fused:
            return None
        elif self.flow_head_projections is None:
            # note (rsi): packed after load_weights filled the q, k, v projections.
            self.flow_head_projections = (
                pack_projections(self.flowloss.cfm.model),
                pack_projections(self.linear_proj_audio),
            )
        else:
            pass
        flow_projections, aggregator_projections = self.flow_head_projections
        return build_flow_constants(
            self.flowloss,
            self.linear_proj_audio,
            self.flow_timesteps,
            batch_size,
            flow_projections,
            aggregator_projections,
        )

    def build_sampling_scales(self, batch_size: int) -> torch.Tensor:
        return torch.tensor(
            [[GUIDANCE_SCALE, SOLVER_NOISE_SCALE]] * batch_size,
            device=self.flow_timesteps.device,
        )

    @torch.no_grad()
    def capture_acoustic_graphs(self, max_batch_size: int) -> None:
        """Capture batch buckets up to max_batch_size in one memory pool."""
        device = self.stop_head.weight.device
        latent_dim = int(self.config.tts_config["latent_dim"])
        memory_pool = torch.cuda.graph_pool_handle()
        # note (rsi): a replay pays for every padded row of its bucket, so
        # small batches, where decode spends most steps, get exact buckets.
        bucket_batch_sizes = {
            size
            for size in [*range(1, EXACT_ACOUSTIC_BUCKETS + 1), *range(24, 1025, 8)]
            if size < max_batch_size
        }
        for batch_size in sorted(bucket_batch_sizes | {max_batch_size}, reverse=True):
            inputs = AcousticInputs(
                hidden_states=torch.zeros(
                    (batch_size, self.config.hidden_size),
                    device=device,
                    dtype=self.stop_head.weight.dtype,
                ),
                latent_history=torch.zeros((batch_size, 1, latent_dim), device=device),
                initial_noise=torch.zeros((batch_size, latent_dim, 1), device=device),
                solver_noise=torch.zeros(
                    (FLOW_STEPS - 1, batch_size, 1, latent_dim), device=device
                ),
                temperatures=torch.zeros(batch_size, device=device),
            )
            sampling_scales = self.build_sampling_scales(batch_size)
            constants = self.build_flow_constants(batch_size)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(ACOUSTIC_GRAPH_WARMUP_STEPS):
                    self.compute_acoustic_step(inputs, sampling_scales, constants)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=memory_pool, stream=stream):
                outputs = self.compute_acoustic_step(inputs, sampling_scales, constants)
            self.acoustic_graphs[batch_size] = AcousticGraph(
                graph=graph,
                inputs=inputs,
                sampling_scales=sampling_scales,
                constants=constants,
                outputs=outputs,
                native_state=self.native_capture,
            )

    @torch.no_grad()
    def sample_acoustic_step(self, inputs: AcousticInputs) -> AcousticStep:
        batch_size = inputs.hidden_states.shape[0]
        if not self.acoustic_graphs or batch_size > max(self.acoustic_graphs):
            return self.compute_acoustic_step(
                inputs,
                self.build_sampling_scales(batch_size),
                self.build_flow_constants(batch_size),
            )
        else:
            pass
        bucket_batch_size = min(
            size for size in self.acoustic_graphs if size >= batch_size
        )
        captured = self.acoustic_graphs[bucket_batch_size]
        captured.inputs.hidden_states[:batch_size].copy_(inputs.hidden_states)
        captured.inputs.latent_history[:batch_size].copy_(inputs.latent_history)
        captured.inputs.initial_noise[:batch_size].copy_(inputs.initial_noise)
        captured.inputs.solver_noise[:, :batch_size].copy_(inputs.solver_noise)
        captured.inputs.temperatures[:batch_size].copy_(inputs.temperatures)
        captured.graph.replay()
        outputs = captured.outputs
        return AcousticStep(
            latents=outputs.latents[:batch_size].clone(),
            feedback_embeddings=outputs.feedback_embeddings[:batch_size].clone(),
            stop_probabilities=outputs.stop_probabilities[:batch_size].clone(),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.weights_loaded:
            raise RuntimeError(
                "Ming 3 TTS live weight updates are unsupported; restart the pipeline to refresh reference caches and acoustic graphs"
            )
        else:
            pass
        head_tensors = {
            name: tensor
            for name, tensor in self.state_dict(keep_vars=True).items()
            if name.startswith(SPEECH_HEAD_PREFIXES)
        }
        loaded_head_names: set[str] = set()

        def backbone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith(SPEECH_HEAD_PREFIXES):
                    head_tensors[name].data.copy_(weight)
                    loaded_head_names.add(name)
                elif name.startswith(BACKBONE_WEIGHT_PREFIX):
                    yield name.removeprefix(BACKBONE_WEIGHT_PREFIX), weight
                elif name.startswith(VOCODER_WEIGHT_PREFIX):
                    continue
                else:
                    raise ValueError(f"Unexpected Ming 3 TTS weight {name}")

        loaded_names = super().load_weights(backbone_weights())
        if loaded_names is None:
            loaded_names = set()
        else:
            pass
        missing_head_names = sorted(set(head_tensors) - loaded_head_names)
        if missing_head_names:
            raise ValueError(f"Missing Ming 3 TTS head weights: {missing_head_names}")
        else:
            pass
        self.weights_loaded = True
        return loaded_names | loaded_head_names


class TinySpeechHeadModel(SpeechHeadModel):
    def build_flow_constants(self, batch_size: int) -> None:
        if self.native_capture is not None and self.native_capture.active:
            self.native_capture.prepare_batch(self.flow_timesteps, batch_size)
        else:
            pass
        return None

    def compute_acoustic_step(
        self,
        inputs: AcousticInputs,
        sampling_scales: torch.Tensor,
        constants: FlowConstants | None,
    ) -> AcousticStep:
        """Preserve Tiny's native flow arithmetic rather than the dense fused head."""
        assert constants is None
        hidden_states = inputs.hidden_states
        conditioning = hidden_states.unsqueeze(1)
        with torch.autocast(
            device_type=hidden_states.device.type, dtype=torch.bfloat16
        ):
            latents = self.flowloss.sample(
                z=conditioning,
                latent_history=inputs.latent_history,
                noise=inputs.initial_noise,
                timesteps=self.flow_timesteps,
                sde_random=inputs.solver_noise,
                cfg=sampling_scales[:, 0],
                sigma=sampling_scales[:, 1],
                temperature=inputs.temperatures,
            )
            feedback_embeddings = self.linear_proj_audio(latents)
            stop_logits = self.stop_head(conditioning)
        return AcousticStep(
            latents=latents,
            feedback_embeddings=feedback_embeddings.reshape(hidden_states.shape),
            stop_probabilities=stop_logits.softmax(dim=-1)[:, 0, 1],
        )

    @torch.no_grad()
    def capture_acoustic_graphs(self, max_batch_size: int) -> None:
        self.native_capture = NativeAcousticCapture(
            self.flowloss, self.linear_proj_audio
        )
        try:
            self.native_capture.install()
            super().capture_acoustic_graphs(max_batch_size)
        finally:
            self.native_capture.restore()


class Ming3TTSModel(TinySpeechHeadModel, BailingMoeV3ForCausalLM):
    def __init__(
        self,
        *,
        config: Ming3TTSConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        install_speech_heads(self, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput:
        """Return each request's last hidden state; omni_prefill_rids belongs to Omni's interface."""
        if forward_batch.forward_mode.is_decode():
            input_embeds = self.decode_feedback(input_ids)
        else:
            pass
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return hidden_state_output(hidden_states, forward_batch)


class Ming3DenseTTSModel(SpeechHeadModel, Qwen2ForCausalLM):
    def __init__(
        self,
        config: Ming3DenseTTSConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        install_speech_heads(self, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput:
        """Return each request's last hidden state; omni_prefill_rids belongs to Omni's interface."""
        if forward_batch.forward_mode.is_decode():
            input_embeds = self.decode_feedback(input_ids)
        else:
            pass
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return hidden_state_output(hidden_states, forward_batch)


def hidden_state_output(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> LogitsProcessorOutput:
    """Keep the last token of a prefill and an empty logit column."""
    if forward_batch.forward_mode.is_extend():
        hidden_states = hidden_states[
            torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        ]
    else:
        pass
    return LogitsProcessorOutput(
        next_token_logits=hidden_states.new_empty((hidden_states.shape[0], 1)),
        hidden_states=hidden_states,
    )
