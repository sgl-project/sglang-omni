# SPDX-License-Identifier: Apache-2.0
"""The FireRedTTS3 reference encode service and its payload conditioning."""

from __future__ import annotations

import concurrent.futures
import logging

from sglang_omni.models.firered_tts3.payload_types import (
    LEFT_CONTEXT_METADATA_KEY,
    MAX_EDIT_SECONDS,
    PROMPT_CHUNK_METADATA_KEY,
    load_firered_tts3_state,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceBundle,
    FireRedReferenceHook,
    ReferenceAudioInput,
    ReferenceConditioning,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.reference_encoder import (
    BatchQueueEntry,
    ReferenceEncodeService,
)

logger = logging.getLogger(__name__)
CACHE_MAX_ITEMS = 256
CACHE_MAX_BYTES = 64 * 1024 * 1024
BATCH_WORKER_NAME = "fireredtts3-ref-encode-batch"


class FireRedReferenceService(
    ReferenceEncodeService[
        ReferenceAudioInput,
        ReferenceConditioning,
        ReferenceConditioning,
        ReferenceAudioInput,
    ]
):
    """The shared service plus a solo thread and caller-side decoding."""

    def __init__(
        self,
        hook: FireRedReferenceHook,
        *,
        max_batch_size: int,
        max_batch_wait_ms: float,
    ) -> None:
        super().__init__(
            hook,
            max_items=CACHE_MAX_ITEMS,
            max_bytes=CACHE_MAX_BYTES,
            log_prefix="fireredtts3",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_worker_name=BATCH_WORKER_NAME,
        )
        self.reference_hook: FireRedReferenceHook = hook
        self.closing: bool = False
        self.encodes_in_flight: int = 0
        self.solo_encodes: int = 0
        self.solo_future: concurrent.futures.Future[ReferenceConditioning] | None = None
        # note (yucheng): Lone requests share one warm thread.
        self.solo_executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{BATCH_WORKER_NAME}-solo"
            )
            if self.batching
            else None
        )

    def close(self) -> None:
        # note (yucheng): Admission checks closing under this lock.
        with self.lock:
            self.closing = True
        if self.solo_executor is not None:
            self.solo_executor.shutdown(wait=False)
            solo_future = self.solo_future
            if solo_future is not None:
                concurrent.futures.wait([solo_future], timeout=5.0)
            else:
                pass
        else:
            pass
        super().close()

    def encode_leader(self, item: ReferenceAudioInput) -> ReferenceConditioning:
        if self.batch_queue is None:
            return self.reference_hook.encode_one(item)
        else:
            pass
        solo_future: concurrent.futures.Future[ReferenceConditioning] | None = None
        with self.lock:
            if self.closing:
                raise RuntimeError("reference encode service is shutting down")
            else:
                pass
            # note (yucheng): Lone requests skip the linger unless solo is busy.
            if (
                self.solo_executor is not None
                and self.encodes_in_flight == 0
                and (self.solo_future is None or self.solo_future.done())
            ):
                solo_future = self.solo_executor.submit(
                    self.reference_hook.encode_one, item
                )
                self.solo_future = solo_future
                self.solo_encodes += 1
            else:
                pass
            # note (yucheng): After submit, so a raising submit leaks no count.
            self.encodes_in_flight += 1
        try:
            if solo_future is not None:
                # note (yucheng): A kept future's traceback can pin tensors.
                solo_future.add_done_callback(self.release_solo_future)
                try:
                    return solo_future.result(timeout=self.timeout_s)
                except concurrent.futures.TimeoutError:
                    self.warn_timeout("solo thread")
                    raise
                finally:
                    if solo_future.done():
                        self.release_solo_future(solo_future)
                    else:
                        pass
            else:
                pass
            entry: BatchQueueEntry[ReferenceAudioInput, ReferenceConditioning] = (
                BatchQueueEntry(
                    queued_input=self.reference_hook.prepare_batch_input(item),
                    future=concurrent.futures.Future(),
                )
            )
            with self.lock:
                if self.closing:
                    raise RuntimeError("reference encode service is shutting down")
                else:
                    self.batch_queue.put(entry)
            try:
                return entry.future.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError:
                # note (yucheng): The entry stays queued; drop its input now.
                entry.abandon()
                self.warn_timeout("batch queue")
                raise
        finally:
            with self.lock:
                self.encodes_in_flight -= 1

    def release_solo_future(
        self, future: concurrent.futures.Future[ReferenceConditioning]
    ) -> None:
        with self.lock:
            if self.solo_future is future:
                self.solo_future = None
            else:
                pass

    def warn_timeout(self, path: str) -> None:
        assert self.batch_queue is not None
        solo_future = self.solo_future
        solo_state = (
            "busy" if solo_future is not None and not solo_future.done() else "idle"
        )
        logger.warning(
            f"FireRedTTS3 reference encode timed out after {self.timeout_s:.0f} s "
            f"on the {path}: {self.encodes_in_flight} in flight, "
            f"{self.batch_queue.qsize()} queued, solo thread {solo_state}"
        )

    def stats(self) -> dict[str, int]:
        stats = super().stats()
        with self.lock:
            stats["solo_encodes"] = self.solo_encodes
        return stats


class FireRedReferenceEncoder:
    """Apply cached reference conditioning to pipeline payloads."""

    def __init__(
        self,
        bundle: FireRedReferenceBundle,
        *,
        model_id: str,
        max_batch_size: int,
        max_batch_wait_ms: float,
        cache_enabled: bool,
    ) -> None:
        self.bundle: FireRedReferenceBundle = bundle
        self.service: FireRedReferenceService = FireRedReferenceService(
            FireRedReferenceHook(
                bundle, model_id=model_id, cache_enabled=cache_enabled
            ),
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def close(self) -> None:
        self.service.close()

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_firered_tts3_state(payload)
        if state.prompt_audio_path is not None:
            artifact = self.service.get_or_encode(
                ReferenceAudioInput(
                    audio_path=state.prompt_audio_path, max_audio_seconds=None
                ),
                desc=repr(state.prompt_audio_path),
            )
            state.prompt_latents = artifact["prompt_latents"]
            state.speaker_embedding = artifact.get("speaker_embedding")
            state.left_context_size = int(state.prompt_latents.shape[1])
        else:
            pass
        if state.audio_in_path is not None:
            artifact = self.service.get_or_encode(
                ReferenceAudioInput(
                    audio_path=state.audio_in_path, max_audio_seconds=MAX_EDIT_SECONDS
                ),
                desc=repr(state.audio_in_path),
            )
            state.audio_in_latents = artifact["prompt_latents"]
        else:
            pass
        if state.reference_pad_token_id is not None:
            reference_latents = (
                state.prompt_latents
                if state.task_type == "tts"
                else state.audio_in_latents
            )
            assert reference_latents is not None
            assert reference_latents.shape[1] % self.bundle.config.patch_size == 0
            reference_patch_count = (
                reference_latents.shape[1] // self.bundle.config.patch_size
            )
            reference_pad_position = state.input_ids.index(state.reference_pad_token_id)
            state.input_ids[reference_pad_position : reference_pad_position + 1] = [
                state.reference_pad_token_id
            ] * reference_patch_count
            state.reference_pad_token_id = None
        else:
            pass
        return store_firered_tts3_state(payload, state)


def prompt_stream_message(payload: StagePayload) -> OutgoingMessage | None:
    """The prompt-latent chunk handed to the vocoder ahead of the talker."""
    params = payload.request.params
    if not isinstance(params, dict) or not params.get("stream"):
        return None
    else:
        pass
    state = load_firered_tts3_state(payload)
    if state.prompt_latents is None or state.prompt_latents.numel() == 0:
        return None
    else:
        pass
    return OutgoingMessage(
        request_id=payload.request_id,
        type="stream",
        target="vocoder",
        data=state.prompt_latents,
        metadata={
            "modality": "audio_latents",
            "stream": True,
            PROMPT_CHUNK_METADATA_KEY: True,
            LEFT_CONTEXT_METADATA_KEY: int(state.left_context_size),
        },
    )


__all__ = [
    "FireRedReferenceEncoder",
    "FireRedReferenceService",
    "prompt_stream_message",
]
