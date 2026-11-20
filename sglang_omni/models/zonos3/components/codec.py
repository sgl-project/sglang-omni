# SPDX-License-Identifier: Apache-2.0
"""Zonos3 audio codec layout and decoder: 10 x 4096 codes at 15 frames per second to 44.1 kHz audio."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.zonos3.components.codec_decoder_kernels import (
    convnext_stack,
    fused_transformer_forward,
    polyphase_resample,
)
from sglang_omni.models.zonos3.components.codec_layers import (
    ConvNeXtBlock,
    FactorizedVectorQuantizer,
    fold_weight_norm,
    patch_up,
    rename_state_dict,
)
from sglang_omni.models.zonos3.components.codec_transformer import (
    TRANSFORMER_INFIX_RENAMES,
    ProjectedTransformer,
    TransformerStageSpec,
)
from sglang_omni.models.zonos3.components.istft_head import ISTFT_HOP_SAMPLES, IstftHead
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

SAMPLE_RATE_HZ = 44100
CODEBOOK_COUNT = 10
CODEBOOK_SIZE = 4096
# note (zhaochenyang20): One codec frame covers the iSTFT half-window, FIR mixer and resampler lookahead; only the final decode may emit it.
BOUNDARY_HOLDBACK_FRAMES = 1
# note (zhaochenyang20): Decoder lengths up to 256 frames (17 s) are captured as CUDA graphs; the 32 single-request lengths captured before serving took about 2.5 s and 0.25 GB on one H100, and the cap leaves 96 graphs for batched shapes.
MAX_DECODER_GRAPH_FRAMES = 256
MAX_DECODER_GRAPHS = 128
# note (zhaochenyang20): The trunk input is right-padded to a multiple of this many frames so 32 graphs per batch size cover every length; the trunk is causal, so padding never changes the real frames.
DECODER_GRAPH_FRAME_MULTIPLE = 8
# note (zhaochenyang20): On CUDA the decoder's transformer stages and iSTFT-head ConvNeXt blocks hold bfloat16 weights while quantizers, waveform convolutions, iSTFT and resamplers stay float32; 5 s of audio decodes in 16 ms instead of 30 ms on SeedTTS EN, with unchanged reconstruction error and WER. Against a float32 decoder with cuDNN TF32 off (35 dB waveform SNR apart), three SeedTTS EN runs per arm on one H100 scored WER below 50 1.21% vs 1.21% and SIM 50.8 vs 50.4, and the float32 decoder served 30% fewer requests per second.
CUDA_COMPUTE_DTYPE = torch.bfloat16
# note (zhaochenyang20): the sm90 FA3 build runs the decoder's causal local window directly, without the 64-frame block mask and its zero-padded key copies; with the fused norms the trunk of a 5 s decode replays in 4.9 ms instead of 8.6 ms on one H100, and SeedTTS EN served 8% more requests per second non-streaming and 12% more streaming (5 ABBA pairs each) at unchanged WER and SIM.
FLASH_ATTENTION_DEVICE_MAJOR = 9

# The codec runs at 44160 Hz internally; boundary resamplers convert by 736 / 735.
EXTERNAL_RESAMPLE_RATIO = 735
INTERNAL_RESAMPLE_RATIO = 736
TO_EXTERNAL_KERNEL_TAPS = 770

GENERATOR_CHECKPOINT = "autoencoder/generator.pt"
ISTFT_HEAD_CHECKPOINT = "autoencoder/heads/full_mr/step_0050000.pt"
CODEC_REGISTRY = "registries/codec_registry.json"
SILENCE_FRAME_FILE = "registries/silence_frame_{codec}.json"
# Samples per 15 Hz frame at the external 44.1 kHz rate and at the internal 44.16 kHz rate.
EXTERNAL_SAMPLES_PER_FRAME = 2940
INTERNAL_SAMPLES_PER_FRAME = 2944

LATENT_DIMENSION = 1024
SEMANTIC_DIMENSION = 768
CONVNEXT_HIDDEN_DIMENSION = 2048
CONVNEXT_DEPTH = 4
RESIDUAL_CODEBOOK_COUNT = CODEBOOK_COUNT - 1

STAGE_PATCH_SIZE = 2
# The iSTFT head runs at 240 Hz, 16 head frames per codec frame.
HEAD_FRAMES_PER_CODE_FRAME = INTERNAL_SAMPLES_PER_FRAME // ISTFT_HOP_SAMPLES

SMALL_STAGE = TransformerStageSpec(
    input_dimension=768,
    model_dimension=768,
    output_dimension=384,
    head_count=12,
    layer_count=12,
    feedforward_dimension=3072,
    context_frames=480,
)
LARGE_STAGE = TransformerStageSpec(
    input_dimension=1280,
    model_dimension=1280,
    output_dimension=1024,
    head_count=20,
    layer_count=32,
    feedforward_dimension=5120,
    context_frames=150,
)
DECODER_TRANSFORMER_STAGES = (
    replace(LARGE_STAGE, input_dimension=1024, output_dimension=1280),
    replace(SMALL_STAGE, input_dimension=640, output_dimension=768, context_frames=300),
    replace(SMALL_STAGE, input_dimension=384, output_dimension=768),
    replace(SMALL_STAGE, input_dimension=384, output_dimension=768),
    replace(SMALL_STAGE, input_dimension=384, output_dimension=None),
)
DECODER_TRANSFORMER_CHECKPOINT_INDICES = (0, 2, 4, 6, 8)

QUANTIZER_PREFIX_RENAMES: list[tuple[str, str]] = [
    ("quantizer.semantic_quantizer.quantizers.0.", "quantizer.semantic_quantizer."),
    ("quantizer.quantizer.quantizers.", "quantizer.residual_quantizers."),
    ("semantic_head.", "quantizer.semantic_head."),
]


def load_checkpoint_state(
    checkpoint_path: Path, section: str, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Load one tensor section of a codec checkpoint and fold its weight norms on device."""
    # Folding on the inference device reproduces the GPU weight-norm numerics bit for bit.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return fold_weight_norm(
        {name: tensor.float() for name, tensor in checkpoint[section].items()}
    )


class SemanticResidualQuantizer(nn.Module):
    """One semantic codebook decoded through a ConvNeXt head plus nine residual codebooks."""

    def __init__(self) -> None:
        super().__init__()
        self.semantic_quantizer = FactorizedVectorQuantizer(
            SEMANTIC_DIMENSION, CODEBOOK_SIZE
        )
        self.residual_quantizers = nn.ModuleList(
            FactorizedVectorQuantizer(LATENT_DIMENSION, CODEBOOK_SIZE)
            for _ in range(RESIDUAL_CODEBOOK_COUNT)
        )
        self.semantic_head = nn.Sequential(
            *(
                ConvNeXtBlock(SEMANTIC_DIMENSION, CONVNEXT_HIDDEN_DIMENSION)
                for _ in range(CONVNEXT_DEPTH)
            ),
            nn.Conv1d(SEMANTIC_DIMENSION, LATENT_DIMENSION, 1),
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Map (B, 10, T) codes to the (B, 1024, T) acoustic latent."""
        latent = self.semantic_head[-1](
            convnext_stack(
                self.semantic_head[:-1], self.semantic_quantizer.decode(codes[:, 0])
            ).transpose(1, 2)
        )
        for codebook_index, residual_quantizer in enumerate(self.residual_quantizers):
            latent = latent + residual_quantizer.decode(codes[:, codebook_index + 1])
        return latent

    def encode_residual(
        self, latent: torch.Tensor, semantic_prediction: torch.Tensor
    ) -> torch.Tensor:
        """Quantize what the semantic prediction leaves unexplained into (B, 9, T) residual codes, in float32."""
        residual = latent.float() - semantic_prediction.float()
        residual_codes: list[torch.Tensor] = []
        for residual_quantizer in self.residual_quantizers:
            codes, quantized = residual_quantizer.encode(residual)
            residual_codes.append(codes)
            residual = residual - quantized
        return torch.stack(residual_codes, dim=1)


@dataclass(kw_only=True)
class CodecDecodeGraph:
    """A captured decoder trunk: replay reads codes and writes hidden_states in place."""

    graph: ReplayableGraph
    codes: torch.Tensor
    hidden_states: torch.Tensor


class Zonos3CodecDecoder(nn.Module):
    """Codes to 44.1 kHz waveform through the quantizer, transformer stages and iSTFT head."""

    def __init__(self) -> None:
        super().__init__()
        self.quantizer = SemanticResidualQuantizer()
        self.transformer_stages = nn.ModuleList(
            ProjectedTransformer(spec) for spec in DECODER_TRANSFORMER_STAGES
        )
        self.head = IstftHead(CONVNEXT_HIDDEN_DIMENSION)
        self.register_buffer(
            "to_external_kernel",
            torch.empty(EXTERNAL_RESAMPLE_RATIO, 1, TO_EXTERNAL_KERNEL_TAPS),
            persistent=True,
        )
        # Graphs and once-eager shapes are keyed by (batch size, trunk frame count).
        self.decode_graphs: dict[tuple[int, int], CodecDecodeGraph] = {}
        self.eager_shapes: set[tuple[int, int]] = set()
        self.graph_pool: DeviceGraphPool | None = None
        self.graph_stream: torch.Stream | None = None
        self.uses_fused_trunk: bool = False
        # The backend that records this decoder's trunk graphs, or None for a
        # device whose platform records none; set when the weights land.
        self.graph_backend: DeviceGraphBackend | None = None

    @classmethod
    def from_model_directory(
        cls, model_directory: str | Path, device: torch.device | str
    ) -> Zonos3CodecDecoder:
        """Load the quantizer from generator.pt and stages 0-8 plus the iSTFT head from the head checkpoint."""
        model_directory = Path(model_directory)
        generator_state = rename_state_dict(
            load_checkpoint_state(
                model_directory / GENERATOR_CHECKPOINT, "generator", device
            ),
            QUANTIZER_PREFIX_RENAMES + [("to_external.kernel", "to_external_kernel")],
            [],
        )
        head_state = rename_state_dict(
            load_checkpoint_state(
                model_directory / ISTFT_HEAD_CHECKPOINT, "trainable", device
            ),
            [
                (f"stages.{checkpoint_index}.", f"transformer_stages.{stage_index}.")
                for stage_index, checkpoint_index in enumerate(
                    DECODER_TRANSFORMER_CHECKPOINT_INDICES
                )
            ]
            + [("head.", "head.")],
            TRANSFORMER_INFIX_RENAMES,
        )
        with torch.device("meta"):
            decoder = cls()
        decoder.load_state_dict(generator_state | head_state, strict=True, assign=True)
        decoder = decoder.to(device).eval()
        if decoder.to_external_kernel.is_cuda:
            decoder.transformer_stages.to(CUDA_COMPUTE_DTYPE)
            decoder.head.blocks.to(CUDA_COMPUTE_DTYPE)
            compute_capability_major, _ = torch.cuda.get_device_capability(
                decoder.to_external_kernel.device
            )
            # A ROCm device also reports type "cuda" and may report major 9.
            decoder.uses_fused_trunk = (
                current_platform.is_cuda()
                and compute_capability_major == FLASH_ATTENTION_DEVICE_MAJOR
            )
        else:
            pass
        decoder.graph_backend = current_platform.get_device_graph_backend(
            decoder.to_external_kernel.device
        )
        return decoder

    def decode_trunk(self, codes: torch.Tensor) -> torch.Tensor:
        """Run the causal quantizer and transformer stages at the input's exact shape; the states come back float32."""
        hidden_states = self.quantizer.decode(codes).to(
            self.transformer_stages[0].input_proj.weight.dtype
        )
        for stage_index, stage in enumerate(self.transformer_stages):
            if stage_index == 0:
                pass
            else:
                hidden_states = patch_up(hidden_states, STAGE_PATCH_SIZE)
            if self.uses_fused_trunk:
                hidden_states = fused_transformer_forward(stage, hidden_states)
            else:
                hidden_states = stage(hidden_states)
        return hidden_states.float()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Map (batch, 10, frames) codes to a (batch, frames * 2940) float waveform."""
        return self.synthesize(self.decode_hidden_states(codes))

    @torch.inference_mode()
    def synthesize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map (batch, head width, frames * 16) trunk states to a (batch, frames * 2940) float waveform."""
        return polyphase_resample(
            self.head(hidden_states),
            self.to_external_kernel,
            INTERNAL_RESAMPLE_RATIO,
            EXTERNAL_RESAMPLE_RATIO,
        )

    def capture_trunk_graph(self, codes: torch.Tensor) -> CodecDecodeGraph:
        """Capture the trunk at the padded (batch, 10, trunk frames) codes' shape on the private capture stream and keep it."""
        assert self.graph_backend is not None
        device_module = torch.get_device_module(codes.device)
        current_stream = device_module.current_stream(codes.device)
        if self.graph_stream is None:
            self.graph_stream = device_module.Stream(device=codes.device)
            self.graph_pool = self.graph_backend.graph_pool_handle()
            # note (zhaochenyang20): Warm the private capture stream once; the trunk's kernels need no per-shape eager pass before capture, and graph replays matched eager decodes bit for bit at every batch size 1-4 and length 8-256 on one H100.
            self.graph_stream.wait_stream(current_stream)
            with device_module.stream(self.graph_stream):
                self.decode_trunk(codes)
            current_stream.wait_stream(self.graph_stream)
        else:
            pass
        static_codes = codes.clone()
        self.graph_stream.wait_stream(current_stream)
        # note (zhaochenyang20): torch.cuda.graph empties the device and pinned host caches on entry; capturing while serving made the engine re-pin its batch uploads through cudaHostAlloc, which waits 13-30 ms behind other threads' blocking copies.
        with self.graph_backend.capture(
            pool=self.graph_pool,
            stream=self.graph_stream,
            thread_local_errors=True,
            reclaim_memory=False,
        ) as graph:
            hidden_states = self.decode_trunk(static_codes)
        decode_graph = CodecDecodeGraph(
            graph=graph, codes=static_codes, hidden_states=hidden_states
        )
        self.decode_graphs[(codes.shape[0], codes.shape[-1])] = decode_graph
        return decode_graph

    @torch.inference_mode()
    def capture_decode_graphs(self) -> None:
        """Before serving, capture the trunk of every single-request graph length and synthesize every single-request length once.

        Trunk lengths go longest first so the shared pool grows once. One synthesis per frame count leaves its FFT and convolution plans cached, so a non-streaming decode never builds them while serving.
        """
        device = self.to_external_kernel.device
        for trunk_frame_count in range(
            MAX_DECODER_GRAPH_FRAMES, 0, -DECODER_GRAPH_FRAME_MULTIPLE
        ):
            self.capture_trunk_graph(
                torch.zeros(
                    (1, CODEBOOK_COUNT, trunk_frame_count),
                    dtype=torch.long,
                    device=device,
                )
            )
        hidden_states = torch.zeros(
            (
                1,
                self.head.blocks[0].dwconv.in_channels,
                MAX_DECODER_GRAPH_FRAMES * HEAD_FRAMES_PER_CODE_FRAME,
            ),
            device=device,
        )
        for frame_count in range(1, MAX_DECODER_GRAPH_FRAMES + 1):
            self.synthesize(
                hidden_states[..., : frame_count * HEAD_FRAMES_PER_CODE_FRAME]
            )
        torch.get_device_module(device).synchronize(device)

    @torch.inference_mode()
    def decode_hidden_states(self, codes: torch.Tensor) -> torch.Tensor:
        """Map (batch, 10, frames) codes to (batch, head width, frames * 16) trunk states.

        A trunk shape not captured by capture_decode_graphs is captured as a graph on its second decode, so one-off shapes stay eager; replayed states stay valid until the next decode.
        """
        assert codes.dim() == 3 and codes.shape[1] == CODEBOOK_COUNT, codes.shape
        codes = codes.to(self.to_external_kernel.device)
        batch_size, _, frame_count = codes.shape
        trunk_frame_count = (
            math.ceil(frame_count / DECODER_GRAPH_FRAME_MULTIPLE)
            * DECODER_GRAPH_FRAME_MULTIPLE
        )
        codes = F.pad(codes, (0, trunk_frame_count - frame_count))
        trunk_shape = (batch_size, trunk_frame_count)
        is_graph_shape = (
            self.graph_backend is not None
            and 0 < trunk_frame_count <= MAX_DECODER_GRAPH_FRAMES
        )
        decode_graph = self.decode_graphs.get(trunk_shape) if is_graph_shape else None
        if (
            is_graph_shape
            and decode_graph is None
            and trunk_shape in self.eager_shapes
            and len(self.decode_graphs) < MAX_DECODER_GRAPHS
        ):
            decode_graph = self.capture_trunk_graph(codes)
        else:
            pass
        if decode_graph is not None:
            decode_graph.codes.copy_(codes)
            decode_graph.graph.replay()
            hidden_states = decode_graph.hidden_states
        elif is_graph_shape:
            self.eager_shapes.add(trunk_shape)
            hidden_states = self.decode_trunk(codes)
        else:
            hidden_states = self.decode_trunk(codes)
        if trunk_frame_count == frame_count:
            pass
        else:
            # The causal trunk upsamples every frame alike, so cutting its padded tail leaves the real frames exact.
            hidden_states = hidden_states[
                ..., : hidden_states.shape[-1] // trunk_frame_count * frame_count
            ]
        return hidden_states


@dataclass(frozen=True, kw_only=True)
class CodecRegistryEntry:
    """The registry's single codec: the decoder's output headroom in dB and the codec's silence frame (10,)."""

    headroom_db: float
    silence_frame: torch.Tensor


def codec_registry_entry(model_directory: str | Path) -> CodecRegistryEntry:
    """Read the codec registry and the codec's silence frame file under model_directory."""
    registry = json.loads((Path(model_directory) / CODEC_REGISTRY).read_text())
    assert len(registry) == 1, sorted(registry)
    entry = next(iter(registry.values()))
    codec_name = str(entry["codec"])
    silence = json.loads(
        (
            Path(model_directory) / SILENCE_FRAME_FILE.format(codec=codec_name)
        ).read_text()
    )
    assert (
        silence["codec"] == codec_name and len(silence["frame"]) == CODEBOOK_COUNT
    ), silence
    return CodecRegistryEntry(
        headroom_db=float(entry["decoder"]["headroom_db"]),
        silence_frame=torch.tensor(
            [int(code) for code in silence["frame"]], dtype=torch.int64
        ),
    )
