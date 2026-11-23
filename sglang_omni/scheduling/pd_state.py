# SPDX-License-Identifier: Apache-2.0
"""Resource checks and metrics for Omni's committed KV transfers."""

from collections import Counter
from dataclasses import fields, replace

import torch
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.load_snapshot import DisaggregationMetrics
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.state_check import CheckPhase, Severity
from sglang.srt.observability.forward_pass_metrics import (
    QueuedRequestMetrics,
    WelfordAccumulator,
)
from sglang.srt.observability.metrics_collector import QueueCount


class PDInvariantChecker(SchedulerInvariantChecker):
    """Include transferred slots before they acquire a request-table row."""

    def __init__(self, checker, scheduler) -> None:
        self.pd_scheduler = scheduler
        super().__init__(
            **{
                item.name: getattr(checker, item.name)
                for item in fields(checker)
                if item.init
            }
        )
        self.get_skip_idle_pool_checks = scheduler.pd_holds_kv

    def run(self, phase, batch=None) -> None:
        with self.pd_scheduler.pd_lifecycle_lock:
            if phase is CheckPhase.IDLE and self.pd_scheduler.pd_holds_kv():
                self._check_busy_pools()  # noqa: leading-underscore  # reuse the resource conservation check
                if envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.get():
                    self.check_pd_slots()
                else:
                    pass
                owners = (
                    self._requests_owning_rows()
                )  # noqa: leading-underscore  # inspect actual row owners
                rows = [request.kv.req_pool_idx for request in owners]
                if (
                    len(rows) != len(set(rows))
                    or len(rows) + self.req_to_token_pool.available_size()
                    != self.req_to_token_pool.size
                ):
                    self.reporter.report(
                        Severity.CORRUPTION,
                        "PD request rows do not conserve pool capacity",
                    )
                else:
                    pass
            else:
                pass
            super().run(phase, batch)

    def _get_total_uncached_sizes(
        self, windows=()
    ):  # noqa: leading-underscore  # override the checker extension point
        full, window = super()._get_total_uncached_sizes(
            windows
        )  # noqa: leading-underscore  # upstream extension point
        full += sum(
            admission.allocation.slots.numel()
            for admission in self.pd_scheduler.pd_owned_allocations()
        )
        return full, window

    def _check_kv_page_invariants(
        self,
    ):  # noqa: leading-underscore  # override the checker extension point
        super()._check_kv_page_invariants()  # noqa: leading-underscore  # upstream extension point
        self.check_pd_slots()

    def check_pd_slots(self) -> None:
        allocations = self.pd_scheduler.pd_owned_allocations()
        if not allocations:
            return
        else:
            pass
        slots = torch.cat([admission.allocation.slots for admission in allocations])
        free = self.token_to_kv_pool_allocator.get_all_free_pages()
        table = self.req_to_token_pool.req_to_token
        bound = [
            table[request.kv.req_pool_idx, : request.kv.kv_allocated_len]
            for request in self._requests_owning_rows()
        ]  # noqa: leading-underscore  # inspect actual row owners
        overlap = bool(torch.isin(slots, torch.cat(bound)).any()) if bound else False
        if (
            torch.unique(slots).numel() != slots.numel()
            or bool(torch.isin(slots, free).any())
            or overlap
        ):
            self.reporter.report(
                Severity.CORRUPTION,
                "PD transfer slots alias free, bound, or other transfer slots",
            )
        else:
            pass


class PDLoadInquirer:
    """Combine common engine metrics with the actual Omni transport state."""

    def __init__(self, common, scheduler) -> None:
        self.common = replace(common, disaggregation_mode=DisaggregationMode.NULL)
        self.scheduler = scheduler

    def _get_num_pending_tokens(
        self, chunk_deduct=0
    ):  # noqa: leading-underscore  # preserve the load-inquirer interface
        return self.common._get_num_pending_tokens(
            chunk_deduct
        )  # noqa: leading-underscore  # upstream extension point

    def get_num_waiting_uncached_tokens(self):
        if self.scheduler.scheduler_role == "decode":
            return 0
        else:
            return self.common.get_num_waiting_uncached_tokens()

    def get_loads(self):
        with self.scheduler.pd_lifecycle_lock:
            result = self.common.get_loads()
            stats = self.scheduler.metrics_reporter.stats
            result.disaggregation = DisaggregationMetrics(
                mode=self.scheduler.scheduler_role,
                kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
            )
            if self.scheduler.scheduler_role == "prefill":
                result.disaggregation.prefill_inflight_queue_reqs = len(
                    self.scheduler.disagg_prefill_transfer_queue
                )
            else:
                receiver = self.scheduler.pd_receiver
                result.disaggregation.decode_transfer_queue_reqs = len(
                    receiver.reservations
                )
                result.queues.prealloc_ready = len(receiver.committed)
                result.num_prealloc_ready_tokens = sum(
                    item.allocation.seq_len for item in receiver.committed.values()
                )
                result.num_waiting_reqs = (
                    len(self.scheduler.waiting_queue)
                    + len(receiver.reservations)
                    + len(receiver.committed)
                )
                result.num_waiting_uncached_tokens = 0
                result.num_total_tokens = result.num_used_tokens
                result.num_active_tokens = result.num_used_tokens - sum(
                    item.allocation.seq_len for item in receiver.reservations.values()
                )
            return result


class PDMetricsReporter(SchedulerMetricsReporter):
    """Report transfers without pretending to have native transport queues."""

    def _update_queue_stats(
        self, priority_enabled
    ):  # noqa: leading-underscore  # override the reporter extension point
        super()._update_queue_stats(
            priority_enabled
        )  # noqa: leading-underscore  # upstream extension point
        if self.scheduler.scheduler_role == "prefill":
            self.stats.num_prefill_bootstrap_queue_reqs = QueueCount()
            self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                list(self.scheduler.disagg_prefill_transfer_queue), priority_enabled
            )
        else:
            pass

    def _update_decode_queue_stats(
        self, priority_enabled
    ):  # noqa: leading-underscore  # override the reporter extension point
        with self.scheduler.pd_lifecycle_lock:
            receiver = self.scheduler.pd_receiver
            receiving = tuple(receiver.reservations.values())
            self.stats.num_decode_prealloc_queue_reqs = QueueCount()
            self.stats.num_decode_transfer_queue_reqs = QueueCount(
                total=len(receiving),
                by_priority=(
                    dict(Counter(item.continuation.priority for item in receiving))
                    if priority_enabled
                    else None
                ),
            )
            self.stats.num_decode_host_receive_queue_reqs = QueueCount()
            self.stats.num_decode_deferred_kv_release_reqs = 0
            self.stats.pending_prealloc_token_usage = 0.0
            capacity = self.scheduler.pool_stats_observer.get_pool_stats().full_capacity
            self.stats.prealloc_full_token_usage = (
                sum(item.allocation.seq_len for item in receiver.owned_allocations())
                / capacity
                if capacity
                else 0.0
            )
            self.stats.prealloc_window_token_usage = {}
            self.stats.prealloc_swa_token_usage = 0.0

    def _decode_queue_log_suffix(
        self,
    ):  # noqa: leading-underscore  # override the reporter extension point
        return (
            f"pre-allocated usage: {self.stats.prealloc_full_token_usage:.2f}, "
            f"#transfer-req: {self.stats.num_decode_transfer_queue_reqs.total}, "
        )

    def _build_queued_request_metrics(
        self,
    ):  # noqa: leading-underscore  # override the reporter extension point
        prefill, decode = WelfordAccumulator(), WelfordAccumulator()
        with self.scheduler.pd_lifecycle_lock:
            if self.scheduler.scheduler_role == "prefill":
                for request in self.scheduler.waiting_queue:
                    prefill.add(len(request.origin_input_ids))
            else:
                for admission in self.scheduler.pd_owned_allocations():
                    decode.add(admission.allocation.seq_len)
                for request in self.scheduler.waiting_queue:
                    decode.add(request.kv.kv_allocated_len)
        return QueuedRequestMetrics(
            num_prefill_requests=prefill.count,
            sum_prefill_tokens=prefill.total,
            var_prefill_length=prefill.variance(),
            num_decode_requests=decode.count,
            sum_decode_kv_tokens=decode.total,
            var_decode_kv_tokens=decode.variance(),
        )


class PDDecodeBatchResultProcessor(SchedulerBatchResultProcessor):
    """Publish the transferred sample before the first actual decode step."""

    def __init__(self, processor) -> None:
        super().__init__(
            **{
                item.name: getattr(processor, item.name)
                for item in fields(processor)
                if item.init
            }
        )

    def process_batch_result_prebuilt(self, batch) -> None:
        super().process_batch_result_prebuilt(batch)
        self.output_streamer.stream_output(
            [request for request in batch.reqs if not request.finished()],
            batch.return_logprob,
        )
