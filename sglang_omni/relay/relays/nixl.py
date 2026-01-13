# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any

from sglang_omni.relay.descriptor import Descriptor
from sglang_omni.relay.nixl import (
    Connector,
    RdmaMetadata,
    ReadableOperation,
    ReadOperation,
)
from sglang_omni.relay.relays.base import Relay

logger = logging.getLogger(__name__)


class NIXLRelay(Relay):
    """NIXL-based relay implementation using dynamo.nixl_connect."""

    def __init__(self, config: dict[str, Any]):
        if Connector is None:
            raise ImportError("dynamo.nixl_connect not available")

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.metadata_server = config.get(
            "metadata_server", "http://127.0.0.1:8080/metadata"
        )
        self.device_name = config.get("device_name", "")
        self.gpu_id = config.get("gpu_id", 0)

        self.connector: Connector | None = None
        self._pending_operations: dict[str, tuple[Any, asyncio.Event]] = {}

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_sync()

    def _run_maybe_async(self, coro_or_result):
        """Run a coroutine if it's a coroutine, otherwise return the result directly."""
        if asyncio.iscoroutine(coro_or_result):
            try:
                asyncio.get_running_loop()
                raise RuntimeError(
                    "Cannot run async operation synchronously while in an async context. "
                    "If the connector uses async methods, you must be in a sync context or use asyncio.run() at a higher level."
                )
            except RuntimeError as e:
                if "no running event loop" in str(e) or "no current event loop" in str(
                    e
                ):
                    try:
                        return asyncio.run(coro_or_result)
                    except RuntimeError as run_error:
                        if "cannot be called from a running event loop" in str(
                            run_error
                        ):
                            raise RuntimeError(
                                "Cannot run async operation synchronously while in an async context. "
                                "If the connector uses async methods, you must be in a sync context."
                            ) from run_error
                        raise
                else:
                    raise
            except AttributeError:
                return asyncio.run(coro_or_result)
        else:
            return coro_or_result

    def _init_sync(self):
        """Initialize NIXL connector synchronously."""
        try:
            self.connector = Connector(worker_id=self.config.get("worker_id"))
            logger.info("NIXLRelay initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize NIXL connector: %s", e)
            raise

    def put(self, descriptors: list[Descriptor]) -> ReadableOperation:
        """
        Put descriptors into the distributed store.

        Parameters
        ----------
        descriptors : list[Any]
            List of Descriptor objects containing tensor data

        Returns
        -------
        Any
            Readable operation object with metadata() and wait_for_completion() methods
        """
        if not self.connector:
            raise RuntimeError("Connector not initialized")

        try:
            result = self.connector.create_readable(descriptors)
            readable_op = self._run_maybe_async(result)

            total_size = 0
            for desc in descriptors:
                total_size += desc.size

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += total_size

            logger.info(
                "NIXLRelay: created readable for %d descriptors, %d bytes",
                len(descriptors),
                total_size,
            )

            return readable_op

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("NIXLRelay put failed: %s", e)
            raise

    def get(self, metadata: Any, descriptors: list[Descriptor]) -> Any:
        """
        Get data from the distributed store using metadata and descriptors.

        Parameters
        ----------
        metadata : Any
            Metadata from readable operation (returned by put)
        descriptors : list[Any]
            List of Descriptor objects for receiving data

        Returns
        -------
        Any
            Read operation object with wait_for_completion() method
        """
        if not self.connector:
            logger.error("Connector not initialized")
            raise RuntimeError("Connector not initialized")

        try:
            result = self.connector.begin_read(metadata, descriptors)
            read_op = self._run_maybe_async(result)

            total_size = 0
            for desc in descriptors:
                total_size += desc.size

            self._metrics["gets"] += 1

            logger.debug(
                "NIXLRelay: began read for %d descriptors, %d bytes",
                len(descriptors),
                total_size,
            )

            return read_op

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("NIXLRelay get failed: %s", e)
            raise

    async def get_async(
        self, metadata: RdmaMetadata, descriptors: list[Descriptor]
    ) -> ReadOperation:
        """
        Get data from the distributed store using metadata and descriptors.

        Parameters
        ----------
        metadata : Any
            Metadata from readable operation (returned by put)
        descriptors : list[Any]
            List of Descriptor objects for receiving data

        Returns
        -------
        ReadOperation
            Read operation object with wait_for_completion() method
        """
        if not self.connector:
            logger.error("Connector not initialized")
            raise RuntimeError("Connector not initialized")

        try:
            read_op = await self.connector.begin_read(metadata, descriptors)  # 异步调用

            total_size = 0
            for desc in descriptors:
                total_size += desc.size

            self._metrics["gets"] += 1

            logger.debug(
                "NIXLRelay: began read for %d descriptors, %d bytes",
                len(descriptors),
                total_size,
            )

            return read_op

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("NIXLRelay get failed: %s", e)
            raise

    def cleanup(self, request_id: str) -> None:
        """Clean up resources for a request."""
        if not self.connector:
            return

        # Clean up pending operations
        keys_to_remove = [k for k in self._pending_operations.keys() if request_id in k]
        for key in keys_to_remove:
            del self._pending_operations[key]

        logger.debug("NIXLRelay: cleanup requested for %s", request_id)

    def health(self) -> dict[str, Any]:
        """Get connector health status."""
        if not self.connector:
            return {"status": "unhealthy", "error": "Connector not initialized"}

        return {
            "status": "healthy",
            "host": self.host,
            "metadata_server": self.metadata_server,
            "device_name": self.device_name,
            "gpu_id": self.gpu_id,
            **self._metrics,
        }

    def close(self) -> None:
        """Clean shutdown."""
        if self.connector:
            try:
                if hasattr(self.connector, "close"):
                    result = self.connector.close()
                    self._run_maybe_async(result)

                self.connector = None
                logger.info("NIXLRelay closed")
            except Exception as e:
                logger.error("Error closing NIXL connector: %s", e)

    async def put_async(self, descriptors: list[Descriptor]) -> ReadableOperation:
        """
        Put descriptors into the distributed store.

        Parameters
        ----------
        descriptors : list[Any]
            List of Descriptor objects containing tensor data

        Returns
        -------
        Any
            Readable operation object with metadata() and wait_for_completion() methods
        """
        if not self.connector:
            logger.error("Connector not initialized")
            raise RuntimeError("Connector not initialized")

        try:
            readable_op = await self.connector.create_readable(descriptors)

            total_size = 0
            for desc in descriptors:
                total_size += desc.size

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += total_size

            logger.info(
                "NIXLRelay: created readable for %d descriptors, %d bytes",
                len(descriptors),
                total_size,
            )

            return readable_op

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("NIXLRelay put failed: %s", e)
            raise
