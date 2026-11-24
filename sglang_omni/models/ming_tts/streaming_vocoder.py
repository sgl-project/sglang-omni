"""Ming-Omni-TTS payload adapter for shared AudioVAE streaming."""

from functools import partial

from sglang_omni.models.ming_tts.audio_decode import decode_ming_tts_audio_payload
from sglang_omni.models.ming_tts.payload_types import load_ming_tts_state
from sglang_omni.modules.ming_audio.audio_decode import MingAudioDecoder
from sglang_omni.modules.ming_audio.streaming_vocoder import (
    AudioVAEStreamingVocoder,
    StreamState,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage


class MingTTSStreamingVocoderScheduler(AudioVAEStreamingVocoder):
    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        patch_size: int,
        latent_dim: int,
        initial_chunk_patches: int,
        steady_chunk_patches: int,
        keep_latents: bool = False,
    ) -> None:
        super().__init__(
            decoder,
            patch_size=patch_size,
            latent_dim=latent_dim,
            initial_chunk_patches=initial_chunk_patches,
            steady_chunk_patches=steady_chunk_patches,
            compute=partial(
                decode_ming_tts_audio_payload,
                decoder=decoder,
                keep_latents=keep_latents,
            ),
            source_hint="Ming-Omni-TTS",
        )

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: StreamState,
    ) -> dict[str, object]:
        del request_id
        final_state = load_ming_tts_state(payload)
        final_state.sample_rate = int(self.decoder.sample_rate)
        final_state.duration_s = float(
            state.emitted_samples / int(self.decoder.sample_rate)
        )
        data = final_state.to_dict()
        data["modality"] = "audio"
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        return data


__all__ = ["MingTTSStreamingVocoderScheduler"]
