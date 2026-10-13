# SPDX-License-Identifier: Apache-2.0
"""Read host-side SGLang scheduler statistics without device synchronization."""

import time
from typing import Protocol, TypedDict


class MetricsSnapshotSource(Protocol):
    enable_metrics: bool


class BackendMetricSample(TypedDict):
    name: str
    labels: dict[str, str]
    value: float


class BackendMetricFamily(TypedDict):
    name: str
    documentation: str
    type: str
    samples: list[BackendMetricSample]


class SchedulerMetricsSnapshot(TypedDict, total=False):
    num_running_reqs: int
    num_queue_reqs: int
    engine_gauges: dict[str, int | float]
    backend_metrics: list[BackendMetricFamily]


class MetricsAdminStageResult(TypedDict):
    stage: str
    success: bool
    data: SchedulerMetricsSnapshot


class MetricsAdminResponse(TypedDict, total=False):
    success: bool
    results: list[MetricsAdminStageResult]


ENGINE_GAUGES = (
    "gen_throughput",
    "cache_hit_rate",
    "token_usage",
    "full_token_usage",
    "swa_token_usage",
    "mamba_usage",
    "num_used_tokens",
    "kv_available_tokens",
    "kv_evictable_tokens",
    "max_total_num_tokens",
)


def engine_metrics_snapshot(
    scheduler: MetricsSnapshotSource,
) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    observer = getattr(scheduler, "pool_stats_observer", None)
    if observer is not None:
        pool = observer.get_pool_stats()
        used, usage = pool.get_kv_token_stats()
        values.update(
            num_used_tokens=used,
            token_usage=usage,
            full_token_usage=pool.full_token_usage,
            kv_available_tokens=pool.full_available_size,
            kv_evictable_tokens=pool.full_evictable_size,
        )
        for name in ("swa_token_usage", "mamba_usage"):
            value = getattr(pool, name, None)
            if value is not None:
                values[name] = value
            else:
                pass
    else:
        pass
    capacity = getattr(scheduler, "max_total_num_tokens", None)
    if capacity is not None:
        values["max_total_num_tokens"] = capacity
    else:
        pass
    reporter = getattr(scheduler, "metrics_reporter", None)
    if reporter is not None:
        throughput = getattr(
            reporter, "_current_gen_throughput", None
        )  # noqa: leading-underscore - backend reporter API
        if throughput is not None:
            values["gen_throughput"] = throughput(time.perf_counter())
        else:
            pass
        # Prefix-cache statistics are maintained by the backend, not inferred
        # from completed HTTP usage or the current contents of the KV pool.
        if getattr(scheduler, "enable_metrics", False):
            values["cache_hit_rate"] = reporter.stats.cache_hit_rate
        else:
            pass
    else:
        pass
    return values


def backend_metrics_snapshot(
    scheduler: MetricsSnapshotSource,
) -> list[BackendMetricFamily]:
    """Serialize existing backend collectors; do not register new worker metrics."""
    if not getattr(scheduler, "enable_metrics", False):
        return []
    else:
        pass
    reporter = getattr(scheduler, "metrics_reporter", None)
    cache = getattr(scheduler, "tree_cache", None)
    collectors = (
        getattr(reporter, "metrics_collector", None),
        getattr(cache, "metrics_collector", None),
    )
    families: list[BackendMetricFamily] = []
    seen: set[int] = set()
    metrics = [
        metric
        for collector in collectors
        if collector is not None
        for metric in vars(collector).values()
    ]
    for metric in metrics:
        collect = getattr(metric, "collect", None)
        if not callable(collect) or id(metric) in seen:
            continue
        else:
            pass
        seen.add(id(metric))
        for family in collect():
            if not family.name.startswith("sglang:"):
                continue
            else:
                pass
            families.append(
                {
                    "name": family.name,
                    "documentation": family.documentation,
                    "type": family.type,
                    "samples": [
                        {
                            "name": sample.name,
                            "labels": dict(sample.labels),
                            "value": sample.value,
                        }
                        for sample in family.samples
                    ],
                }
            )
    return families
