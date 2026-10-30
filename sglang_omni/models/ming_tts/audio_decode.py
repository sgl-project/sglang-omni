"""Ming-Omni-TTS payload adapter for the shared AudioVAE decoder."""

from sglang_omni.models.ming_tts.payload_types import (
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.modules.ming_audio.audio_decode import MingAudioDecoder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload


def decode_ming_tts_audio_payload(
    payload: StagePayload, decoder: MingAudioDecoder, *, keep_latents: bool = False
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""
    state = load_ming_tts_state(payload)
    waveform = decoder.decode_full(state.generated_latents)
    state.sample_rate = int(decoder.sample_rate)
    state.duration_s = float(waveform.numel() / int(decoder.sample_rate))
    if not keep_latents:
        state.generated_latents = None
    else:
        pass
    payload = store_ming_tts_state(payload, state)
    payload.data.update(
        audio_waveform_payload(
            waveform,
            sample_rate=int(decoder.sample_rate),
            modality="audio",
            source_hint="Ming-Omni-TTS",
        )
    )
    usage = build_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    else:
        pass
    return payload


__all__ = ["MingAudioDecoder", "decode_ming_tts_audio_payload"]
