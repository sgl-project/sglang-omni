# SPDX-License-Identifier: Apache-2.0
"""Ling V3 speech backbone with the Ming 3 continuous-latent speech heads."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import torch
from pydantic import JsonValue
from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.runtime_context import get_exec, get_schedule
from torch import nn
from transformers import Qwen2Config

from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.cfm import get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss

BACKBONE_WEIGHT_PREFIX = "model."
VOCODER_WEIGHT_PREFIX = "audio."
SPEECH_HEAD_PREFIXES = ("flowloss.", "linear_proj_audio.", "stop_head.", "spk_head.")
FLOW_STEPS = 10
GUIDANCE_SCALE = 2.0
SOLVER_NOISE_SCALE = 0.25
ACOUSTIC_GRAPH_WARMUP_STEPS = 2


class Ming3TTSConfig(BailingHybridConfig):
    model_type: ClassVar[str] = "lingv3_tts"

    def __init__(
        self,
        llm_config: dict[str, JsonValue] | None = None,
        tts_config: dict[str, JsonValue] | None = None,
        flow_config: dict[str, JsonValue] | None = None,
        aggregator_config: dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        backbone_fields = dict(llm_config or {})
        backbone_fields.update(kwargs)
        backbone_fields.pop("model_type", None)
        backbone_fields.pop("auto_map", None)
        super().__init__(**backbone_fields)
        self.tts_config: dict[str, JsonValue] = dict(tts_config or {})
        self.flow_config: dict[str, JsonValue] = dict(flow_config or {})
        self.aggregator_config: dict[str, JsonValue] = dict(aggregator_config or {})


class Ming3DenseTTSConfig(Qwen2Config):
    """Qwen2 backbone config for the dense Ming 3 TTS checkpoint."""

    model_type: ClassVar[str] = "dense"

    def __init__(
        self,
        llm_config: dict[str, JsonValue] | None = None,
        ditar_config: dict[str, JsonValue] | None = None,
        aggregator_config: dict[str, JsonValue] | None = None,
        audio_tokenizer_config: dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        backbone_fields = dict(llm_config or {})
        for unused_name in (
            "model_type",
            "architectures",
            "auto_map",
            "torch_dtype",
            "transformers_version",
            "mlp_depth",
        ):
            kwargs.pop(unused_name, None)
            backbone_fields.pop(unused_name, None)
        super().__init__(**backbone_fields)
        # note (chenyang): AutoConfig matches the class attribute "dense",
        # while the Qwen2 runtime reads the instance model type.
        self.model_type = "qwen2"
        encoder_config = dict(audio_tokenizer_config or {}).get("enc_kwargs") or {}
        self.tts_config: dict[str, JsonValue] = {
            "latent_dim": int(encoder_config.get("latent_dim", 64)),
            "speaker_embedding_dim": 192,
        }
        self.flow_config: dict[str, JsonValue] = dict(ditar_config or {})
        self.aggregator_config: dict[str, JsonValue] = dict(aggregator_config or {})
        self.audio_tokenizer_config: dict[str, JsonValue] = dict(
            audio_tokenizer_config or {}
        )


def install_speech_heads(
    module: nn.Module, config: Ming3TTSConfig | Ming3DenseTTSConfig
) -> None:
    """Attach the flow, stop, and speaker heads shared by both backbones."""
    latent_dim = int(config.tts_config["latent_dim"])
    flow_config = dict(config.flow_config)
    flow_config.pop("z_channels", None)
    aggregator_config = dict(config.aggregator_config)
    aggregator_config.setdefault("in_channels", latent_dim)
    # The 0.5B checkpoint omits this key. Tiny checkpoints set it explicitly.
    aggregator_config.setdefault("attn_backend", "torch")
    module.flowloss = FlowLoss(
        z_channels=latent_dim, llm_cond_dim=config.hidden_size, **flow_config
    )
    module.linear_proj_audio = Aggregator(
        llm_input_dim=config.hidden_size, **aggregator_config
    )
    module.stop_head = nn.Linear(config.hidden_size, 2)
    module.spk_head = nn.Linear(
        int(config.tts_config["speaker_embedding_dim"]), config.hidden_size
    )
    module.flow_timesteps = get_epss_timesteps(
        FLOW_STEPS, module.stop_head.weight.device, torch.float32
    )
    module.acoustic_graphs = {}
    # note (chenyang): decode reads feedback rows from a fixed table by
    # row id, so the backbone CUDA graph replays them through input_ids.
    graph_config = get_exec().graph
    decode_batch_capacity = get_schedule().max_running_requests
    if not graph_config.disable_cuda_graph:
        decode_batch_capacity = max(
            decode_batch_capacity, graph_config.cuda_graph_config.decode.max_bs
        )
    else:
        pass
    module.decode_feedback = nn.Embedding(decode_batch_capacity, config.hidden_size)
    module.decode_feedback.weight.requires_grad_(False)
    module.decode_feedback_rows = torch.arange(
        decode_batch_capacity, device=module.decode_feedback.weight.device
    )


@dataclass(kw_only=True)
class AcousticInputs:
    """Batch-major step inputs; solver_noise is [flow steps - 1, batch, 1, latent]."""

    hidden_states: torch.Tensor
    latent_history: torch.Tensor
    initial_noise: torch.Tensor
    solver_noise: torch.Tensor
    temperatures: torch.Tensor


@dataclass(kw_only=True)
class AcousticStep:
    latents: torch.Tensor
    feedback_embeddings: torch.Tensor
    stop_probabilities: torch.Tensor


@dataclass(kw_only=True)
class AcousticGraph:
    graph: torch.cuda.CUDAGraph
    inputs: AcousticInputs
    outputs: AcousticStep


class SpeechHeadModel:
    """Flow, stop, and feedback heads shared by the MoE and dense backbones."""

    flowloss: FlowLoss
    linear_proj_audio: Aggregator
    stop_head: nn.Linear
    spk_head: nn.Linear
    flow_timesteps: torch.Tensor
    acoustic_graphs: dict[int, AcousticGraph]
    decode_feedback: nn.Embedding
    decode_feedback_rows: torch.Tensor
    config: Ming3TTSConfig | Ming3DenseTTSConfig

    def stage_decode_feedback(self, feedback_embeddings: torch.Tensor) -> torch.Tensor:
        """Write one feedback row per request and return the row ids decode reads."""
        batch_size = feedback_embeddings.shape[0]
        self.decode_feedback.weight[:batch_size].copy_(feedback_embeddings)
        return self.decode_feedback_rows[:batch_size]

    def compute_acoustic_step(
        self, inputs: AcousticInputs, sampling_scales: torch.Tensor
    ) -> AcousticStep:
        """Sample one latent per row; sampling_scales holds [guidance, noise] per row."""
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

    def build_sampling_scales(self, batch_size: int) -> torch.Tensor:
        return torch.tensor(
            [[GUIDANCE_SCALE, SOLVER_NOISE_SCALE]] * batch_size,
            device=self.flow_timesteps.device,
        )

    @torch.no_grad()
    def capture_acoustic_graphs(self, max_batch_size: int) -> None:
        """Capture power-of-two batch buckets up to max_batch_size in one memory pool."""
        device = self.stop_head.weight.device
        latent_dim = int(self.config.tts_config["latent_dim"])
        memory_pool = torch.cuda.graph_pool_handle()
        powers_of_two = [
            1 << exponent for exponent in range(max_batch_size.bit_length())
        ]
        bucket_batch_sizes = {size for size in powers_of_two if size < max_batch_size}
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
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(ACOUSTIC_GRAPH_WARMUP_STEPS):
                    self.compute_acoustic_step(inputs, sampling_scales)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=memory_pool, stream=stream):
                outputs = self.compute_acoustic_step(inputs, sampling_scales)
            self.acoustic_graphs[batch_size] = AcousticGraph(
                graph=graph, inputs=inputs, outputs=outputs
            )

    @torch.no_grad()
    def sample_acoustic_step(self, inputs: AcousticInputs) -> AcousticStep:
        batch_size = inputs.hidden_states.shape[0]
        if not self.acoustic_graphs:
            return self.compute_acoustic_step(
                inputs, self.build_sampling_scales(batch_size)
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
        return loaded_names | loaded_head_names


class Ming3TTSModel(SpeechHeadModel, BailingMoeV3ForCausalLM):
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
