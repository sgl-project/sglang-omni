# SPDX-License-Identifier: Apache-2.0
"""Reads chunks from relay and delivers them to ChunkMailbox."""
from __future__ import annotations

import logging

from sglang_omni.pipeline.chunk.mailbox import ChunkItem, ChunkMailbox
from sglang_omni.pipeline.worker.data_plane import _restore_tensors
from sglang_omni.proto.messages import ChunkReadyMessage

logger = logging.getLogger(__name__)


class ChunkReceiver:
    """Receives chunk notifications and reads tensor data from relay into mailbox."""

    def __init__(self, data_plane, mailbox: ChunkMailbox):
        self._data_plane = data_plane
        self._mailbox = mailbox

    async def deliver(self, msg: ChunkReadyMessage) -> None:
        """Process a ChunkReadyMessage: read blob from relay, put in mailbox."""
        request_id = msg.request_id

        if msg.error:
            self._mailbox.put_error(
                request_id, RuntimeError(msg.error), from_stage=msg.from_stage
            )
            return

        if msg.is_chunks_done:
            self._mailbox.put_chunks_done(request_id, from_stage=msg.from_stage)
            return

        try:
            blob_key = f"{request_id}:chunk:{msg.chunk_id}"
            tensor = await self._data_plane.read_blob(blob_key, msg.relay_metadata)
            chunk_metadata = msg.relay_metadata.get("chunk_metadata")
            metadata_tensor_blobs = msg.relay_metadata.get("chunk_metadata_tensors")
            if isinstance(chunk_metadata, dict) and isinstance(metadata_tensor_blobs, dict):
                tensor_dict = {}
                for path, info in metadata_tensor_blobs.items():
                    if not isinstance(path, str) or not isinstance(info, dict):
                        continue
                    meta_blob_key = info.get("blob_key")
                    meta_metadata = info.get("relay_metadata")
                    if not isinstance(meta_blob_key, str) or not isinstance(
                        meta_metadata, dict
                    ):
                        continue
                    tensor_dict[path] = await self._data_plane.read_blob(
                        meta_blob_key, meta_metadata
                    )
                chunk_metadata = _restore_tensors(chunk_metadata, tensor_dict)
            self._mailbox.put(
                request_id,
                ChunkItem(
                    chunk_id=msg.chunk_id,
                    tensor=tensor,
                    metadata=chunk_metadata,
                    from_stage=msg.from_stage,
                ),
            )
        except Exception as exc:
            logger.error("ChunkReceiver read_blob failed for %s: %s", request_id, exc)
            self._mailbox.put_error(request_id, exc, from_stage=msg.from_stage)
