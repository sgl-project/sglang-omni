"""Reference DCP backend for the Duo 5-to-17 pipeline."""

import logging
from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from typing import Literal

import msgspec
import torch
from data.inference_budget import budget_index
from sglang.multimodal_gen.runtime.models.dits.duo.model import RealizationSegment
from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
    NativeDCPRealization,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    NativeDCPOptions,
)
from sglang.multimodal_gen.runtime.pipelines.duo.loader import load_realization_engine
from sglang.multimodal_gen.runtime.pipelines.duo.session import (
    NativeBudgetMediaSession,
    TokenWindowMediaSession,
)
from torchtitan.models.duo_realization.modules import intent_conditioning
from torchtitan.models.duo_realization.pipelines.streaming import shared_condition

from sglang_omni.models.duo.streaming import LiveSessionInput
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DCPState:
    settings: LiveSessionInput
    features: dict[str, object]
    session: NativeBudgetMediaSession | None = None


class DCPRealizationSessionHooks(SessionHooks):
    def __init__(
        self,
        args: Namespace,
        *,
        device: torch.device,
        use_sglang_kernels: bool = True,
        native_backbone: bool = False,
        rendezvous: str = "env://",
        sp_mode: Literal["ulysses", "ring"] = "ulysses",
        tp_size: int = 1,
        kv_window_tokens: int | None = None,
        reference_window_chunks: int = 0,
        native_options: NativeDCPOptions | None = None,
    ) -> None:
        # This inferencer runs Realization only. Its Intent identity must describe
        # the checkpoint, independently of the separately loaded live planner.
        self.kv_window_tokens = kv_window_tokens
        self.reference_window_chunks = reference_window_chunks
        self.device = device
        self.args, self.engine = load_realization_engine(
            args,
            device=device,
            use_sglang_kernels=use_sglang_kernels,
            native_backbone=native_backbone,
            rendezvous=rendezvous,
            sp_mode=sp_mode,
            tp_size=tp_size,
            native_options=native_options,
        )
        self.states: dict[SessionIdentity, DCPState] = {}

    @property
    def graph_counts(self) -> tuple[int, int]:
        if isinstance(self.engine.model, NativeDCPRealization):
            return self.engine.model.graph_captures, self.engine.model.graph_replays
        else:
            return 0, 0

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        settings = msgspec.convert(request.inputs, type=LiveSessionInput, strict=True)
        if (settings.height, settings.width) != (self.args.height, self.args.width):
            raise ValueError(
                "DCP session dimensions must match the configured Realization geometry"
            )
        else:
            pass
        self.states[session_identity] = DCPState(
            settings=settings,
            features={
                "sample_id": settings.sample_id,
                "codec": {"mode": "duo_5_17"},
                "protocol": intent_conditioning.PROTOCOL,
                "blocks": [],
                "text_groups": [],
            },
        )

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if context.cancelled.is_set():
            raise RuntimeError("Realization session cancelled")
        else:
            pass
        state = self.states[context.session_identity]
        features = state.features
        plan: RealizationSegment = payload.data["plan"]
        shared: RealizationSegment = payload.data["shared"]
        video: RealizationSegment = payload.data["video"]
        reference: RealizationSegment = payload.data["reference"]
        height, width = state.settings.height // 16, state.settings.width // 16
        if video.tokens.shape[0] != 5 * height * width // 4:
            raise ValueError("This checkpoint requires duo_5_17 blocks")
        else:
            pass
        block = chunk.seq
        if state.session is None:
            features["reference"] = (
                reference.tokens.reshape(1, height // 2, width // 2, 24, 2, 2)
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(24, 1, height, width)
                .cpu()
            )
            features["h50"] = shared.tokens.cpu()
            description_tokens = payload.data["description_tokens"]
            features["text_groups"] = [
                dict(
                    start=0,
                    end=description_tokens,
                    role="system",
                    block=-1,
                    phase=-1,
                    section="assistant_description",
                ),
                dict(
                    start=description_tokens,
                    end=shared.tokens.shape[0],
                    role="system",
                    block=-1,
                    phase=-1,
                    section="speech_cue",
                ),
            ]
            args = copy(self.args)
            args.seed = state.settings.seed
            if self.kv_window_tokens is None:
                state.session = NativeBudgetMediaSession(
                    self.engine.model,
                    features,
                    args,
                    self.device,
                    reference_window_chunks=self.reference_window_chunks,
                )
            else:
                state.session = TokenWindowMediaSession(
                    self.engine.model,
                    features,
                    args,
                    self.device,
                    self.kv_window_tokens,
                    reference_window_chunks=self.reference_window_chunks,
                )
            state.session.shared = shared_condition(features)
        else:
            pass
        start = features["h50"].shape[0]
        features["h50"] = torch.cat((features["h50"], plan.tokens.cpu()))
        features["text_groups"].append(
            dict(
                start=start,
                end=start + plan.tokens.shape[0],
                role="assistant",
                block=block,
                phase=0,
                section="assistant_plan",
            )
        )
        features["blocks"].append(
            dict(
                block_id=block,
                start_ms=chunk.t_start_ms,
                end_ms=chunk.t_start_ms + chunk.duration_ms,
                frame_start=17 * block,
                frame_stop=17 * (block + 1),
                audio_start_sample=round(block * 17 / 24 * 32000),
                audio_stop_sample=round((block + 1) * 17 / 24 * 32000),
            )
        )
        state.session.index = budget_index(
            features, self.args.height, self.args.width, "duo_5_17"
        )
        state.session.pending_kv_reset = payload.data.get("reset_realization_kv", False)
        latent, metrics, _ = state.session.generate(features, block)
        logger.info(
            "Realization DCP block: %s",
            dict(session=str(context.session_identity), block=block, **metrics),
        )
        video_latent = latent["video"].reshape(24, 5, height // 2, 2, width // 2, 2)
        payload.data = {
            "video_latent": video_latent.permute(1, 2, 4, 0, 3, 5).reshape(-1, 96),
            "audio_latent": latent["audio"].permute(0, 2, 1).reshape(-1, 32),
            "metrics": metrics,
        }
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)
        if not self.states and isinstance(self.engine.model, NativeDCPRealization):
            self.engine.model.clear_execution_cache()
        else:
            pass

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states.get(session_identity)
        if state is None or state.session is None or state.session.kv is None:
            return ResourceUsage()
        else:
            return ResourceUsage(
                kv_tokens=state.session.kv.length, bytes=state.session.kv.nbytes
            )
