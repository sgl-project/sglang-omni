# SPDX-License-Identifier: Apache-2.0
"""Reference-audio encoder wrapper for MOSS-TTS preprocessing."""

from __future__ import annotations

import base64
import inspect
import io
import logging
import re
import types
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:audio/[^;,]+;base64,(?P<data>.+)$", re.DOTALL)
_MISSING = object()


class MossReferenceAudioEncoder:
    """Own the compiled MOSS audio-tokenizer entrypoint and reference encoding."""

    def __init__(
        self,
        processor: Any,
        *,
        enable_torch_compile: bool = True,
        compile_mode: str | None = "default",
        compile_warmup_seconds: Sequence[float] | None = None,
    ) -> None:
        """Create a reference-audio encoder wrapper.

        Args:
            processor: MOSS processor that owns `audio_tokenizer`.
            enable_torch_compile: When true, compile the fixed target
                `audio_tokenizer.quantizer.forward`. When false, keep the
                processor eager and only use this wrapper for reference-audio
                loading/encoding.
            compile_mode: Value passed to `torch.compile(mode=...)`. Common
                modes are "default", "reduce-overhead", "max-autotune", and
                "max-autotune-no-cudagraphs". Use None to call `torch.compile`
                without a mode argument.
            compile_warmup_seconds: Audio durations to warm after compiling.
                None uses the default (1.0,); for example, [1.0, 3.0] warms
                1-second and 3-second reference-audio shapes.
        """
        self.processor = processor
        self.audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        self.enable_torch_compile = enable_torch_compile
        self.compile_mode = compile_mode
        self.compile_warmup_seconds = (
            self._normalize_warmup_seconds(compile_warmup_seconds)
            if enable_torch_compile
            else ()
        )
        if self.enable_torch_compile:
            self._compile_audio_tokenizer()

    def encode_reference_audio(self, ref_audio: str) -> Any:
        match = _DATA_URI_RE.match(ref_audio)
        if match is not None:
            wav, sample_rate = self._decode_data_uri(match)
        else:
            wav, sample_rate = self._load_audio_path(ref_audio)
        return self._encode_wav(wav, sample_rate)

    def _compile_audio_tokenizer(self) -> None:
        audio_tokenizer = getattr(self.processor, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise RuntimeError("MOSS processor has no audio_tokenizer to torch.compile")
        self._compile_quantizer_forward(audio_tokenizer)

    def _compile_quantizer_forward(self, audio_tokenizer: Any) -> None:
        quantizer = getattr(audio_tokenizer, "quantizer", None)
        if quantizer is None:
            raise RuntimeError("MOSS audio_tokenizer has no quantizer to torch.compile")
        target_forward = getattr(quantizer, "forward", None)
        if not callable(target_forward):
            raise RuntimeError("MOSS audio_tokenizer quantizer has no callable forward")

        original_audio_tokenizer = audio_tokenizer
        original_instance_forward = self._get_instance_attr(quantizer, "forward")
        compile_target = self._bind_unwrapped_callable(target_forward, quantizer)
        target_desc = "audio_tokenizer.quantizer.forward"
        compile_kwargs = {
            "fullgraph": False,
            "dynamic": False,
        }
        if self.compile_mode is not None:
            compile_kwargs["mode"] = self.compile_mode

        try:
            setattr(
                quantizer, "forward", torch.compile(compile_target, **compile_kwargs)
            )
            self._warmup()
        except Exception as exc:
            self._restore_instance_attr(quantizer, "forward", original_instance_forward)
            self.processor.audio_tokenizer = original_audio_tokenizer
            self.audio_tokenizer = original_audio_tokenizer
            logger.exception(
                "MOSS audio encoder torch.compile failed for %s", target_desc
            )
            raise RuntimeError(
                f"MOSS audio encoder torch.compile failed for {target_desc}"
            ) from exc

        logger.info(
            "Enabled MOSS audio encoder torch.compile for %s (mode=%s)",
            target_desc,
            self.compile_mode,
        )

    @staticmethod
    def _get_instance_attr(instance: Any, name: str) -> Any:
        instance_dict = getattr(instance, "__dict__", None)
        if isinstance(instance_dict, dict) and name in instance_dict:
            return instance_dict[name]
        return _MISSING

    @staticmethod
    def _restore_instance_attr(instance: Any, name: str, value: Any) -> None:
        if value is _MISSING:
            try:
                delattr(instance, name)
            except AttributeError:
                pass
        else:
            setattr(instance, name, value)

    @staticmethod
    def _bind_unwrapped_callable(callable_obj: Any, owner: Any | None = None) -> Any:
        callable_obj = inspect.unwrap(callable_obj)
        if inspect.ismethod(callable_obj):
            callable_obj = callable_obj.__func__
        if owner is not None and inspect.isfunction(callable_obj):
            return types.MethodType(callable_obj, owner)
        return callable_obj

    def _warmup(self) -> None:
        model_config = getattr(self.processor, "model_config", None)
        sample_rate = int(getattr(model_config, "sampling_rate", 24000) or 24000)
        for seconds in self.compile_warmup_seconds:
            num_samples = max(1, int(round(float(seconds) * sample_rate)))
            self._encode_wav(
                torch.zeros((1, num_samples), dtype=torch.float32), sample_rate
            )

    @staticmethod
    def _normalize_warmup_seconds(
        warmup_seconds: Sequence[float] | None,
    ) -> tuple[float, ...]:
        """Normalize compile warmup durations.

        Examples:
        - None -> (1.0,)
        - [1, 3.0, 5] -> (1.0, 3.0, 5.0)
        - [] or [0] -> RuntimeError
        """
        values = tuple(float(value) for value in (warmup_seconds or (1.0,)))
        if not values:
            raise RuntimeError("MOSS audio encoder torch.compile warmup is empty")
        for value in values:
            if value <= 0:
                raise RuntimeError(
                    "MOSS audio encoder torch.compile warmup seconds must be positive"
                )
        return values

    def _encode_wav(self, wav: torch.Tensor, sample_rate: int) -> Any:
        encode_audios = getattr(self.processor, "encode_audios_from_wav", None)
        if not callable(encode_audios):
            raise RuntimeError("MOSS processor.encode_audios_from_wav is unavailable")
        return encode_audios([wav], int(sample_rate))[0]

    def _decode_data_uri(self, match: re.Match[str]) -> tuple[torch.Tensor, int]:
        raw = base64.b64decode(match.group("data"))
        return self._read_audio(io.BytesIO(raw))

    def _load_audio_path(self, ref_audio: str) -> tuple[torch.Tensor, int]:
        return self._read_audio(ref_audio)

    @staticmethod
    def _read_audio(source: str | io.BytesIO) -> tuple[torch.Tensor, int]:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "MOSS-TTS reference audio encoding requires soundfile"
            ) from exc

        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T), int(sample_rate)
