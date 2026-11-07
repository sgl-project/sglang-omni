"""Bounded teardown of local asyncio plumbing, never native resource ownership."""

import asyncio
from collections.abc import Iterable

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

# Note (Junnan Li): A second cancellation interrupts the reader's shielded cleanup wait;
# native release remains coordinator-owned.
CANCELLATION_ROUNDS = 2


async def cancel_local_tasks(
    tasks: Iterable[asyncio.Task[None] | asyncio.Task[bool] | None], timeout_s: float
) -> None:
    local_tasks = {
        task
        for task in tasks
        if task is not None and task is not asyncio.current_task()
    }
    unfinished_tasks = {task for task in local_tasks if not task.done()}
    for _ in range(CANCELLATION_ROUNDS):
        if not unfinished_tasks:
            break
        else:
            pass
        for task in unfinished_tasks:
            task.cancel()
        _, unfinished_tasks = await asyncio.wait(unfinished_tasks, timeout=timeout_s)
    finished_tasks = local_tasks - unfinished_tasks
    if finished_tasks:
        await asyncio.gather(*finished_tasks, return_exceptions=True)
    else:
        pass
    if unfinished_tasks:
        raise RuntimeError("local realtime task did not acknowledge cancellation")
    else:
        pass


async def close_websocket(websocket: WebSocket) -> None:
    """Close the socket unless either side has already ended it."""
    if (
        websocket.client_state == WebSocketState.CONNECTED
        and websocket.application_state == WebSocketState.CONNECTED
    ):
        try:
            await websocket.close()
        except WebSocketDisconnect:
            # note (Qing Guo): A client that vanishes during the close leaves nothing to close.
            pass
    else:
        pass
