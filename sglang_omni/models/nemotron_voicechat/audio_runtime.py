# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful NeMo perception and RVQ-VAE stages for VoiceChat."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

from .payload_types import FRAME_SAMPLES, OUTPUT_FRAME_SAMPLES


class _PerceptionHolder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        from omegaconf import DictConfig

        self.cfg = DictConfig(config)


def _checkpoint_files(model_path: str) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(model_path).expanduser().resolve()
    if not (root / "config.json").is_file() and (root / "checkpoint").is_dir():
        root = root / "checkpoint"
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"VoiceChat config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"VoiceChat weights not found: {weights_path}")
    return root, weights_path, json.loads(config_path.read_text())


def _copy_checkpoint_prefix(module: nn.Module, checkpoint: Path, prefix: str) -> int:
    targets = module.state_dict()
    copied: set[str] = set()
    unexpected: list[str] = []
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        # ``safe_open`` exposes keys but is not itself iterable.
        for checkpoint_name in handle.keys():  # noqa: SIM118
            if not checkpoint_name.startswith(prefix):
                continue
            name = checkpoint_name[len(prefix) :]
            target = targets.get(name)
            if target is None:
                unexpected.append(checkpoint_name)
                continue
            source = handle.get_tensor(checkpoint_name)
            if source.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {checkpoint_name}: checkpoint "
                    f"{tuple(source.shape)}, module {tuple(target.shape)}"
                )
            target.copy_(source)
            copied.add(name)
    if not copied:
        raise ValueError(f"No VoiceChat tensors matched checkpoint prefix {prefix!r}")
    missing = sorted(set(targets) - copied)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {len(missing)} module tensors: {missing[:5]}")
        if unexpected:
            details.append(
                f"unexpected {len(unexpected)} checkpoint tensors: {unexpected[:5]}"
            )
        raise ValueError(
            f"Incomplete VoiceChat checkpoint coverage for prefix {prefix!r}: "
            + "; ".join(details)
        )
    return len(copied)


def _validate_max_sessions(max_sessions: int) -> int:
    value = int(max_sessions)
    if value < 1:
        raise ValueError("VoiceChat max_sessions must be positive")
    return value


@dataclass
class _PerceptionSession:
    audio_buffer: torch.Tensor
    cache: Any
    next_frame_index: int = 0


class VoiceChatPerceptionRuntime:
    """Streaming 16 kHz PCM -> one acoustic embedding per 80 ms frame."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        use_cudagraph: bool = True,
        max_sessions: int = 1,
    ) -> None:
        from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
            PerceptionCacheManager,
        )
        from nemo.collections.speechlm2.parts.pretrained import setup_speech_encoder

        _, weights, config = _checkpoint_files(model_path)
        self.device = torch.device(device)
        holder = _PerceptionHolder(config["model"]["stt"]["model"])
        setup_speech_encoder(holder, pretrained_weights=False)
        self.loaded_tensors = _copy_checkpoint_prefix(
            holder.perception, weights, "stt_model.perception."
        )
        self.model = holder.perception.to(self.device, torch.float32).eval()
        self.model.requires_grad_(False)
        model_view = SimpleNamespace(stt_model=SimpleNamespace(perception=self.model))
        self.cache_manager = PerceptionCacheManager(
            model_view,
            device=self.device,
            dtype=torch.float32,
            use_cudagraph=use_cudagraph,
        )
        if not self.cache_manager.setup():
            raise RuntimeError("The VoiceChat perception encoder is not streamable")
        self.max_sessions = _validate_max_sessions(max_sessions)
        self.sessions: dict[str, _PerceptionSession] = {}

    def _session(self, session_id: str) -> _PerceptionSession:
        session = self.sessions.get(session_id)
        if session is not None:
            return session
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("VoiceChat perception session capacity is exhausted")
        session = _PerceptionSession(
            audio_buffer=torch.empty((1, 0), dtype=torch.float32, device=self.device),
            cache=self.cache_manager.get_initial_state(batch_size=1),
        )
        self.sessions[session_id] = session
        return session

    @torch.inference_mode()
    def step(
        self, session_id: str, frame_index: int, pcm16_base64: str
    ) -> torch.Tensor:
        session = self._session(session_id)
        if frame_index != session.next_frame_index:
            raise ValueError(
                f"VoiceChat perception expected frame {session.next_frame_index}, "
                f"got {frame_index}"
            )
        raw = base64.b64decode(pcm16_base64, validate=True)
        pcm = np.frombuffer(raw, dtype="<i2")
        if pcm.size != FRAME_SAMPLES:
            raise ValueError(
                f"VoiceChat expects {FRAME_SAMPLES} PCM16 samples per frame; "
                f"got {pcm.size}"
            )
        frame = torch.from_numpy(pcm.astype(np.float32) / 32768.0).to(self.device)
        session.audio_buffer = torch.cat(
            (session.audio_buffer, frame.unsqueeze(0)), dim=1
        )
        encoded, session.cache, _ = self.cache_manager.step(
            audio_input=session.audio_buffer,
            frame_idx=frame_index,
            num_frames_per_chunk=1,
            perception_cache=session.cache,
        )
        if encoded.shape[1] != 1:
            raise RuntimeError(
                f"Expected one encoded frame, got {tuple(encoded.shape)}"
            )
        session.next_frame_index += 1
        return encoded[:, 0, :].float().cpu()

    def close(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


@dataclass
class _CodecSession:
    cache: Any
    next_frame_index: int = 0


class VoiceChatCodecRuntime:
    """Streaming 31-code RVQ frames -> 22.05 kHz float PCM."""

    def __init__(self, model_path: str, *, device: str, max_sessions: int = 1) -> None:
        from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel
        from omegaconf import DictConfig

        _, weights, config = _checkpoint_files(model_path)
        self.device = torch.device(device)
        codec_config = config["model"]["speech_generation"]["model"]["codec_config"]
        self.model = RVQVAEModel(DictConfig(codec_config))
        self.loaded_tensors = _copy_checkpoint_prefix(
            self.model, weights, "tts_model.audio_codec."
        )
        self.model = self.model.to(self.device, torch.float32).eval()
        self.model.requires_grad_(False)
        with safe_open(weights, framework="pt", device="cpu") as handle:
            self.control_codes = handle.get_tensor("tts_model._control_codes").to(
                self.device
            )
            self.silence_codes = handle.get_tensor("tts_model.codec_silence_tokens").to(
                self.device
            )
        self.max_sessions = _validate_max_sessions(max_sessions)
        self.sessions: dict[str, _CodecSession] = {}

    def _session(self, session_id: str) -> _CodecSession:
        from nemo.collections.speechlm2.modules.ear_tts_vae_codec import (
            CausalConv1dCache,
        )

        session = self.sessions.get(session_id)
        if session is not None:
            return session
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("VoiceChat codec session capacity is exhausted")
        session = _CodecSession(cache=CausalConv1dCache())
        self.sessions[session_id] = session
        return session

    @torch.inference_mode()
    def step(self, session_id: str, frame_index: int, codes: list[int]) -> np.ndarray:
        from nemo.collections.speechlm2.models.duplex_ear_tts import (
            replace_control_speech_codes,
        )

        session = self._session(session_id)
        if frame_index != session.next_frame_index:
            raise ValueError(
                f"VoiceChat codec expected frame {session.next_frame_index}, "
                f"got {frame_index}"
            )
        if len(codes) != int(self.silence_codes.numel()):
            raise ValueError(
                f"VoiceChat codec expects {self.silence_codes.numel()} codes, "
                f"got {len(codes)}"
            )
        tensor = torch.tensor(codes, dtype=torch.long, device=self.device).reshape(
            1, 1, -1
        )
        tensor = replace_control_speech_codes(
            tensor, self.control_codes, self.silence_codes
        )
        lengths = torch.ones(1, dtype=torch.long, device=self.device)
        audio, _ = self.model.decode(tensor, lengths, cache=session.cache)
        result = audio.reshape(-1).float().cpu().numpy()
        if result.size != OUTPUT_FRAME_SAMPLES:
            raise RuntimeError(
                f"VoiceChat codec produced {result.size} samples; "
                f"expected {OUTPUT_FRAME_SAMPLES}"
            )
        session.next_frame_index += 1
        return np.clip(result, -1.0, 1.0)

    def close(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


__all__ = ["VoiceChatCodecRuntime", "VoiceChatPerceptionRuntime"]
