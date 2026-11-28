# SPDX-License-Identifier: Apache-2.0
"""Zonos3 codec encoder: 44.1 kHz reference audio to 10 x 4096 codes, with w2v-BERT semantic features."""

from __future__ import annotations

import logging
import math
import threading
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.zonos3.components.codec import (
    CODEBOOK_COUNT,
    CONVNEXT_DEPTH,
    CONVNEXT_HIDDEN_DIMENSION,
    EXTERNAL_RESAMPLE_RATIO,
    EXTERNAL_SAMPLES_PER_FRAME,
    GENERATOR_CHECKPOINT,
    INTERNAL_RESAMPLE_RATIO,
    INTERNAL_SAMPLES_PER_FRAME,
    LARGE_STAGE,
    QUANTIZER_PREFIX_RENAMES,
    SAMPLE_RATE_HZ,
    SEMANTIC_DIMENSION,
    SMALL_STAGE,
    STAGE_PATCH_SIZE,
    SemanticResidualQuantizer,
    load_checkpoint_state,
)
from sglang_omni.models.zonos3.components.codec_encoder_kernels import biased_patch_down
from sglang_omni.models.zonos3.components.codec_encoder_layers import (
    RESIDUAL_UNIT_DILATIONS,
    EncoderBlock,
    encoder_transformer_forward,
)
from sglang_omni.models.zonos3.components.codec_layers import (
    CausalConv1d,
    ConvNeXtBlock,
    patch_down,
    rename_state_dict,
    sinc_resample,
)
from sglang_omni.models.zonos3.components.codec_transformer import (
    TRANSFORMER_INFIX_RENAMES,
    ProjectedTransformer,
    rotary_tables,
)
from sglang_omni.models.zonos3.components.semantic_features import (
    TEACHER_SAMPLE_RATE_HZ,
    W2vBertSemanticFeatures,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.utils.device import device_guard, has_device_streams

logger = logging.getLogger(__name__)

TEACHER_DIRECTORY = "autoencoder/teacher"
TEACHER_HIDDEN_DIMENSION = 1024
# The encoder runs on zero-padded 20 s lanes; a clip longer than one lane continues on further lanes that each carry 10 s of real left context whose codes are discarded.
LANE_FRAMES = 300
LANE_CONTEXT_FRAMES = 150
# note (zhaochenyang20): Only the teacher sees the whole zero-padded lane; the acoustic encoder and the semantic modules are causal, so they run on the lane's real frames padded to a multiple of this many, one CUDA graph per length; on one H100 a 7.7 s reference encodes in 43 ms instead of the whole lane's 65 ms.
ENCODE_GRAPH_BUCKET_FRAMES = 8
TEACHER_LANE_SAMPLES = math.ceil(
    LANE_FRAMES * EXTERNAL_SAMPLES_PER_FRAME * TEACHER_SAMPLE_RATE_HZ / SAMPLE_RATE_HZ
)

ENCODER_INPUT_CHANNELS = 64
ENCODER_INPUT_KERNEL_SIZE = 7
# (input channels, output channels, stride) per waveform downsampling block.
ENCODER_BLOCK_SHAPES = ((64, 128, 2), (128, 256, 4))
WAVEFORM_PATCH_SIZE = 23
ENCODER_TRANSFORMER_STAGES = (
    replace(SMALL_STAGE, input_dimension=5888),
    SMALL_STAGE,
    SMALL_STAGE,
    replace(SMALL_STAGE, output_dimension=640, context_frames=300),
    LARGE_STAGE,
)
ENCODER_TRANSFORMER_CHECKPOINT_INDICES = (4, 6, 8, 10, 12)
# Every encoder stage attends with 64-dimensional heads; the first runs at 16 frames per codec frame.
ENCODER_HEAD_DIMENSION = 64
FIRST_STAGE_FRAMES_PER_CODE_FRAME = 16
# The fused convolution stack reads bfloat16 convolutions in the channels-last memory cuDNN returns from sm80 on.
FUSED_ENCODER_MIN_DEVICE_MAJOR = 8
RESIDUAL_UNIT_INFIX_RENAMES: list[tuple[str, str]] = [
    (
        f".block.{unit}.module.block.{checkpoint_index}.{checkpoint_suffix}",
        f".residual_units.{unit}.{module_name}.",
    )
    for unit in range(len(RESIDUAL_UNIT_DILATIONS))
    for checkpoint_index, checkpoint_suffix, module_name in (
        (0, "", "input_activation"),
        (1, "conv.", "dilated_conv"),
        (2, "", "output_activation"),
        (3, "conv.", "pointwise_conv"),
    )
] + [(".block.3.", ".activation."), (".block.4.conv.", ".downsample_conv.")]


@dataclass(frozen=True, kw_only=True)
class ReferenceLane:
    """One lane on the CPU: its internal-rate waveform zero padded to its length bucket, its whole zero-padded 16 kHz teacher waveform, and the frames it contributes.

    Lane frame f is clip frame source_frame + f; the lane contributes clip frames [output_start_frame, output_end_frame).
    """

    internal_waveform: torch.Tensor
    teacher_waveform: torch.Tensor
    source_frame: int
    output_start_frame: int
    output_end_frame: int


@dataclass(kw_only=True)
class ReferenceEncodeGraph:
    """A captured lane length bucket: replay reads the lane's inputs and writes its codes in place.

    teacher_waveform feeds the devices whose graph runs the filterbank itself; where the filterbank runs on the host instead, input_features carries its features into the graph.
    """

    graph: ReplayableGraph
    internal_waveform: torch.Tensor
    teacher_waveform: torch.Tensor | None
    input_features: torch.Tensor | None
    codes: torch.Tensor


class Zonos3CodecEncoder(nn.Module):
    """Reference waveform to codes: acoustic encoder, w2v-BERT semantic codebook, residual RVQ.

    The acoustic encoder, semantic encoder and semantic head run under bf16 autocast, the quantizers in float32.
    """

    def __init__(
        self,
        semantic_features: W2vBertSemanticFeatures,
        to_internal_kernel: torch.Tensor,
    ) -> None:
        super().__init__()
        # A plain CPU attribute, not a buffer: the internal-rate waveform is resampled on the CPU, as the codec's training data pipeline did.
        self.to_internal_kernel: torch.Tensor = to_internal_kernel
        self.input_conv = CausalConv1d(
            1, ENCODER_INPUT_CHANNELS, ENCODER_INPUT_KERNEL_SIZE
        )
        self.downsample_blocks = nn.ModuleList(
            EncoderBlock(input_channels, output_channels, stride)
            for input_channels, output_channels, stride in ENCODER_BLOCK_SHAPES
        )
        self.transformer_stages = nn.ModuleList(
            ProjectedTransformer(spec) for spec in ENCODER_TRANSFORMER_STAGES
        )
        self.semantic_encoder = nn.Sequential(
            nn.Conv1d(TEACHER_HIDDEN_DIMENSION, SEMANTIC_DIMENSION, 1),
            *(
                ConvNeXtBlock(SEMANTIC_DIMENSION, CONVNEXT_HIDDEN_DIMENSION)
                for _ in range(CONVNEXT_DEPTH)
            ),
        )
        self.quantizer = SemanticResidualQuantizer()
        self.semantic_features = semantic_features
        self.encode_graphs: dict[int, ReferenceEncodeGraph] = {}
        # note (zhaochenyang20): preprocessing may run several workers; a replay holds one bucket's static inputs, so its copy, replay and read must not interleave with another worker's.
        self.encode_lock = threading.Lock()
        # Plain attributes, set on the encoder's device after loading: RoPE tables of the longest lane's first stage, which every stage slices.
        self.rotary_cos: torch.Tensor | None = None
        self.rotary_signed_sin: torch.Tensor | None = None
        self.uses_fused_encoder: bool = False

    @classmethod
    def from_model_directory(
        cls, model_directory: str | Path, device: torch.device | str
    ) -> Zonos3CodecEncoder:
        """Load the acoustic encoder and quantizer from generator.pt and the w2v-BERT teacher."""
        model_directory = Path(model_directory)
        semantic_features = W2vBertSemanticFeatures(
            model_directory / TEACHER_DIRECTORY, SAMPLE_RATE_HZ
        )
        prefix_renames = (
            [
                ("to_internal.kernel", "to_internal_kernel"),
                ("encoder.0.conv.", "input_conv."),
                ("encoder.1.module.", "downsample_blocks.0."),
                ("encoder.2.module.", "downsample_blocks.1."),
                ("semantic_encoder.", "semantic_encoder."),
            ]
            + [
                (f"encoder.{checkpoint_index}.", f"transformer_stages.{stage_index}.")
                for stage_index, checkpoint_index in enumerate(
                    ENCODER_TRANSFORMER_CHECKPOINT_INDICES
                )
            ]
            + QUANTIZER_PREFIX_RENAMES
        )
        state = rename_state_dict(
            load_checkpoint_state(
                model_directory / GENERATOR_CHECKPOINT, "generator", device
            ),
            prefix_renames,
            RESIDUAL_UNIT_INFIX_RENAMES + TRANSFORMER_INFIX_RENAMES,
        )
        state |= {
            f"semantic_features.{name}": tensor
            for name, tensor in semantic_features.state_dict().items()
        }
        to_internal_kernel = state.pop("to_internal_kernel").cpu()
        with torch.device("meta"):
            encoder = cls(semantic_features, to_internal_kernel)
        encoder.load_state_dict(state, strict=True, assign=True)
        encoder = encoder.to(device).eval()
        # note (zhaochenyang20): autocast would cast these convolution and linear weights to bfloat16 on every call; holding them in bfloat16 gives the same products without the casts, while norms, scales and the quantizers keep float32.
        for module in (
            encoder.input_conv,
            encoder.downsample_blocks,
            encoder.transformer_stages,
            encoder.semantic_encoder,
            encoder.quantizer.semantic_head,
        ):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear | nn.Conv1d):
                    submodule.to(torch.bfloat16)
                else:
                    pass
        # The channels-last convolutions read each (O, I, K) weight in (O, K, I) memory.
        for module in encoder.downsample_blocks.modules():
            if isinstance(module, CausalConv1d):
                module.weight = nn.Parameter(
                    module.weight.detach().transpose(1, 2).contiguous().transpose(1, 2),
                    requires_grad=False,
                )
            else:
                pass
        for quantizer in (
            encoder.quantizer.semantic_quantizer,
            *encoder.quantizer.residual_quantizers,
        ):
            quantizer.cache_normalized_codebook()
        assert all(
            spec.model_dimension // spec.head_count == ENCODER_HEAD_DIMENSION
            for spec in ENCODER_TRANSFORMER_STAGES
        ), ENCODER_TRANSFORMER_STAGES
        # note (zhaochenyang20): the tables depend only on the frame count, so one prefix-sliced pair replaces the ten small kernels each stage of every encode spent rebuilding them, bit for bit.
        encoder.rotary_cos, encoder.rotary_signed_sin = rotary_tables(
            torch.arange(
                LANE_FRAMES * FIRST_STAGE_FRAMES_PER_CODE_FRAME,
                device=device,
                dtype=torch.float32,
            ),
            ENCODER_HEAD_DIMENSION,
        )
        if current_platform.is_cuda():
            compute_capability_major, _ = torch.cuda.get_device_capability(
                encoder.input_conv.weight.device
            )
            # A ROCm device also reports type "cuda", but its libdevice lacks the kernels' rounding intrinsics.
            encoder.uses_fused_encoder = (
                current_platform.is_cuda()
                and compute_capability_major >= FUSED_ENCODER_MIN_DEVICE_MAJOR
            )
        else:
            pass
        return encoder

    def lane_codes(
        self, internal_waveform: torch.Tensor, teacher_waveform: torch.Tensor
    ) -> torch.Tensor:
        """(10, bucket frames) codes of one lane's device waveforms; the teacher waveform always spans the whole lane."""
        return self.lane_codes_from_features(
            internal_waveform,
            self.semantic_features.filterbank_features(teacher_waveform),
        )

    def lane_codes_from_features(
        self, internal_waveform: torch.Tensor, input_features: torch.Tensor
    ) -> torch.Tensor:
        """(10, bucket frames) codes of one lane's device waveform and its (1, teacher frames, 160) filterbank features."""
        device_type = internal_waveform.device.type
        assert self.rotary_cos is not None and self.rotary_signed_sin is not None
        with torch.autocast(device_type, dtype=torch.bfloat16, cache_enabled=False):
            hidden_states = F.conv1d(
                F.pad(internal_waveform[:, None], (self.input_conv.left_padding, 0)),
                self.input_conv.weight,
            )
            bias = self.input_conv.bias
            for block in self.downsample_blocks:
                hidden_states, bias = block(
                    hidden_states, bias, self.uses_fused_encoder
                )
            hidden_states = biased_patch_down(
                hidden_states, bias, WAVEFORM_PATCH_SIZE, self.uses_fused_encoder
            )
            for stage_index, stage in enumerate(self.transformer_stages):
                if stage_index == 0:
                    pass
                else:
                    hidden_states = patch_down(hidden_states, STAGE_PATCH_SIZE)
                hidden_states = encoder_transformer_forward(
                    stage,
                    hidden_states,
                    self.rotary_cos,
                    self.rotary_signed_sin,
                    self.uses_fused_encoder,
                )
        semantic_states = self.semantic_features.forward_from_features(
            input_features, LANE_FRAMES
        )[..., : hidden_states.shape[-1]]
        with torch.autocast(device_type, dtype=torch.bfloat16, cache_enabled=False):
            semantic_states = self.semantic_encoder(semantic_states)
        semantic_codes, semantic_quantized = self.quantizer.semantic_quantizer.encode(
            semantic_states.float()
        )
        with torch.autocast(device_type, dtype=torch.bfloat16, cache_enabled=False):
            semantic_prediction = self.quantizer.semantic_head(semantic_quantized)
        residual_codes = self.quantizer.encode_residual(
            hidden_states, semantic_prediction
        )
        return torch.cat([semantic_codes[:, None], residual_codes], dim=1)[0]

    @torch.inference_mode()
    def capture_encode_graphs(self) -> None:
        """Capture every lane length bucket before serving, so no capture overlaps the vocoder's.

        No graph is captured on a device whose platform has no graph backend; a capture that fails leaves every lane on the eager path rather than failing startup.
        """
        device = self.input_conv.weight.device
        graph_backend = current_platform.get_device_graph_backend(device)
        if graph_backend is None:
            return
        else:
            pass
        device_module = torch.get_device_module(device)
        # note (zhaochenyang20): the filterbank needs float64 kernels; where the device lacks them it runs on the host, so the graph takes its features as an input instead of the teacher waveform.
        captures_filterbank = current_platform.is_float64_supported()
        graph_pool = graph_backend.graph_pool_handle()
        graph_stream = (
            device_module.Stream(device=device) if has_device_streams(device) else None
        )
        # note (zhaochenyang20): Ascend loads and compiles the kernels of a shape on its first pass, so every bucket warms up before its capture; CUDA keeps the single pass the serving path already paid.
        warmup_passes = 2 if device.type == "npu" else 1

        def encode_lane(
            internal_waveform: torch.Tensor,
            teacher_waveform: torch.Tensor | None,
            input_features: torch.Tensor | None,
        ) -> torch.Tensor:
            if input_features is None:
                assert teacher_waveform is not None
                return self.lane_codes(internal_waveform, teacher_waveform)
            else:
                return self.lane_codes_from_features(internal_waveform, input_features)

        graphs: dict[int, ReferenceEncodeGraph] = {}
        with device_guard(device):
            try:
                # note (zhaochenyang20): the graphs share one pool, so longest first each shorter bucket reuses blocks the longer ones freed; on one H100 the reference encoder reserved 8.8 GiB shortest first and 5.3 GiB longest first.
                for bucket_frames in reversed(
                    (
                        *range(
                            ENCODE_GRAPH_BUCKET_FRAMES,
                            LANE_FRAMES,
                            ENCODE_GRAPH_BUCKET_FRAMES,
                        ),
                        LANE_FRAMES,
                    )
                ):
                    internal_waveform = torch.zeros(
                        1, bucket_frames * INTERNAL_SAMPLES_PER_FRAME, device=device
                    )
                    if captures_filterbank:
                        teacher_waveform = torch.zeros(
                            1, TEACHER_LANE_SAMPLES, device=device
                        )
                        input_features = None
                    else:
                        teacher_waveform = None
                        input_features = torch.zeros_like(
                            self.semantic_features.filterbank_features(
                                torch.zeros(1, TEACHER_LANE_SAMPLES)
                            ),
                            device=device,
                        )
                    if graph_stream is not None:
                        graph_stream.wait_stream(device_module.current_stream(device))
                    else:
                        pass
                    with torch.inference_mode(), (
                        device_module.stream(graph_stream)
                        if graph_stream is not None
                        else nullcontext()
                    ):
                        for _ in range(warmup_passes):
                            encode_lane(
                                internal_waveform, teacher_waveform, input_features
                            )
                    if graph_stream is not None:
                        device_module.current_stream(device).wait_stream(graph_stream)
                    else:
                        pass
                    with graph_backend.capture(
                        pool=graph_pool,
                        stream=graph_stream,
                        thread_local_errors=True,
                    ) as graph:
                        codes = encode_lane(
                            internal_waveform, teacher_waveform, input_features
                        )
                    graphs[bucket_frames] = ReferenceEncodeGraph(
                        graph=graph,
                        internal_waveform=internal_waveform,
                        teacher_waveform=teacher_waveform,
                        input_features=input_features,
                        codes=codes,
                    )
            except Exception as exc:
                logger.warning(
                    "Zonos3 reference encoder graph capture disabled: "
                    f"{type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                self.encode_graphs = {}
                return
        self.encode_graphs = graphs
        logger.info(
            "Zonos3 reference encoder graphs captured for %d lane buckets",
            len(graphs),
        )

    def reference_lanes(self, waveform: torch.Tensor) -> list[ReferenceLane]:
        """Cut a (1, samples) 44.1 kHz CPU waveform into lanes; both resamplers run on the CPU here."""
        assert waveform.dim() == 2 and waveform.shape[0] == 1, waveform.shape
        waveform = waveform.float()
        frame_count = math.ceil(waveform.shape[-1] / EXTERNAL_SAMPLES_PER_FRAME)
        external_waveform = F.pad(
            waveform, (0, frame_count * EXTERNAL_SAMPLES_PER_FRAME - waveform.shape[-1])
        )
        internal_waveform = sinc_resample(
            waveform,
            self.to_internal_kernel,
            EXTERNAL_RESAMPLE_RATIO,
            INTERNAL_RESAMPLE_RATIO,
        )
        internal_waveform = F.pad(
            internal_waveform,
            (0, frame_count * INTERNAL_SAMPLES_PER_FRAME - internal_waveform.shape[-1]),
        )
        lanes: list[ReferenceLane] = []
        output_start_frame = 0
        while output_start_frame < frame_count:
            source_frame = max(output_start_frame - LANE_CONTEXT_FRAMES, 0)
            output_end_frame = min(frame_count, source_frame + LANE_FRAMES)
            lane_frames = output_end_frame - source_frame
            bucket_frames = min(
                math.ceil(lane_frames / ENCODE_GRAPH_BUCKET_FRAMES)
                * ENCODE_GRAPH_BUCKET_FRAMES,
                LANE_FRAMES,
            )
            lanes.append(
                ReferenceLane(
                    internal_waveform=F.pad(
                        internal_waveform[
                            :,
                            source_frame
                            * INTERNAL_SAMPLES_PER_FRAME : output_end_frame
                            * INTERNAL_SAMPLES_PER_FRAME,
                        ],
                        (0, (bucket_frames - lane_frames) * INTERNAL_SAMPLES_PER_FRAME),
                    ),
                    teacher_waveform=self.semantic_features.teacher_waveform(
                        F.pad(
                            external_waveform[
                                :,
                                source_frame
                                * EXTERNAL_SAMPLES_PER_FRAME : output_end_frame
                                * EXTERNAL_SAMPLES_PER_FRAME,
                            ],
                            (
                                0,
                                (LANE_FRAMES - lane_frames)
                                * EXTERNAL_SAMPLES_PER_FRAME,
                            ),
                        )
                    ),
                    source_frame=source_frame,
                    output_start_frame=output_start_frame,
                    output_end_frame=output_end_frame,
                )
            )
            output_start_frame = output_end_frame
        return lanes

    @torch.inference_mode()
    def encode_lanes(self, lanes: list[ReferenceLane]) -> torch.Tensor:
        """(10, frames) device codes of a clip's lanes, one graph replay of the lane's bucket (or an eager pass) each."""
        device = self.input_conv.weight.device
        codes = torch.empty(
            (CODEBOOK_COUNT, lanes[-1].output_end_frame),
            dtype=torch.int64,
            device=device,
        )
        with device_guard(device):
            for lane in lanes:
                output_start_frame = lane.output_start_frame
                output_end_frame = lane.output_end_frame
                source_slice = slice(
                    output_start_frame - lane.source_frame,
                    output_end_frame - lane.source_frame,
                )
                encode_graph = self.encode_graphs.get(
                    lane.internal_waveform.shape[1] // INTERNAL_SAMPLES_PER_FRAME
                )
                if encode_graph is None:
                    codes[:, output_start_frame:output_end_frame] = self.lane_codes(
                        lane.internal_waveform.to(device),
                        lane.teacher_waveform.to(device),
                    )[:, source_slice]
                else:
                    # note (zhaochenyang20): the read of the graph's codes must stay under the lock, or another worker's replay overwrites them first.
                    with self.encode_lock:
                        encode_graph.internal_waveform.copy_(lane.internal_waveform)
                        if encode_graph.input_features is None:
                            encode_graph.teacher_waveform.copy_(lane.teacher_waveform)
                        else:
                            encode_graph.input_features.copy_(
                                self.semantic_features.filterbank_features(
                                    lane.teacher_waveform
                                )
                            )
                        encode_graph.graph.replay()
                        codes[:, output_start_frame:output_end_frame] = (
                            encode_graph.codes[:, source_slice]
                        )
        return codes
