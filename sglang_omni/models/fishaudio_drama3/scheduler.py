# SPDX-License-Identifier: Apache-2.0
"""Drama 3 engine scheduler that picks each prefill's decode interval online."""
from __future__ import annotations

import logging
import time

from sglang.srt.managers.schedule_batch import ScheduleBatch

from sglang_omni.models.fishaudio_drama3.prefill_cadence import PrefillCadence
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)

REPORT_INTERVAL_S = 30.0


class Drama3Scheduler(OmniScheduler[Drama3RequestData]):
    """Upstream's interval countdown, armed with the interval the cadence picks per prefill."""

    def install_cadence(self, cadence: PrefillCadence) -> None:
        self.cadence = cadence
        self.round_started_s: float | None = None
        self.last_batch_was_decode = False
        self.last_batch_was_extend = False
        self.last_batch_size = 0
        self.reported_s = time.perf_counter()

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        cadence = self.cadence
        now_s = time.perf_counter()
        if self.round_started_s is None:
            pass
        elif self.last_batch_was_decode:
            cadence.observe_decode_step(now_s - self.round_started_s, self.last_batch_size)
        elif self.last_batch_was_extend:
            cadence.observe_prefill(now_s - self.round_started_s, self.last_batch_size)
        else:
            pass
        self.release_if_due(now_s)
        if now_s - self.reported_s >= REPORT_INTERVAL_S:
            logger.info(cadence.report())
            self.reported_s = now_s
        else:
            pass
        self.round_started_s = now_s
        batch = super().get_next_batch_to_run()
        self.last_batch_was_decode = batch is not None and batch.forward_mode.is_decode()
        self.last_batch_was_extend = batch is not None and batch.forward_mode.is_extend()
        self.last_batch_size = len(batch.reqs) if batch is not None else 0
        return batch

    def is_prefill_expected(self) -> bool:
        # note (0xtoward): async decode predicts this round's prefill, so the wait cap applies first.
        self.release_if_due(time.perf_counter())
        return super().is_prefill_expected()

    def release_if_due(self, now_s: float) -> None:
        """Cut the wait short for a request that would otherwise miss its first-audio SLO."""
        if (
            self._prefill_decode_interval_remaining > 0 and self.waiting_queue
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            oldest_wait_s = max(
                now_s
                - request._coalesce_enqueue_t  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                for request in self.waiting_queue
            )
            free_slots = self.max_running_requests - len(self.running_batch.reqs)
            if self.cadence.should_release(
                len(self.waiting_queue), oldest_wait_s, free_slots
            ):
                self._prefill_decode_interval_remaining = 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            else:
                pass
        else:
            pass

    def _arm_prefill_decode_interval(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self, batch: ScheduleBatch | None
    ) -> None:
        if batch is not None and batch.forward_mode.is_extend():
            self._prefill_decode_interval_remaining = (  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                self.cadence.choose_interval()
            )
        else:
            pass
