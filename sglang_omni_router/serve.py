# SPDX-License-Identifier: Apache-2.0
"""Serve the external Omni router process."""

from __future__ import annotations

import argparse
import copy
import logging
import logging.config
import shlex
from collections.abc import Sequence
from typing import Any, get_args

import uvicorn
from pydantic import ValidationError

from sglang_omni_router.app import create_app
from sglang_omni_router.config import (
    DEFAULT_CAPABILITIES,
    Capability,
    RouterConfig,
    RoutingPolicy,
    WorkerConfig,
    build_router_config,
    load_worker_configs,
)
from sglang_omni_router.launcher import (
    LocalLauncher,
    LocalLauncherConfig,
    load_launcher_config,
)

logger = logging.getLogger("sglang_omni_router.serve")

# Note (Jiaxin Deng): each in-flight request holds a client and an upstream
# socket; the headroom covers listeners, health checks, and log files.
_NOFILE_HEADROOM = 64


def _read_nofile_soft_limit() -> int | None:
    try:
        import resource
    except ImportError:  # non-POSIX platform, nothing to check
        return None
    soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft_limit == resource.RLIM_INFINITY:
        return None
    return soft_limit


def check_file_descriptor_limit(config: RouterConfig, *, strict: bool = False) -> None:
    soft_limit = _read_nofile_soft_limit()
    if soft_limit is None:
        return
    pool_size = config.upstream_pool_size
    required = 2 * pool_size + _NOFILE_HEADROOM
    if soft_limit >= required:
        return
    # Note (Jiaxin Deng): name the flag that binds the pool max(); an explicit
    # --max-connections == --max-inflight tie binds both (lowering either alone
    # leaves the other holding the pool), while a derived max_inflight (unset)
    # follows --max-connections.
    max_connections = config.max_connections
    max_inflight = config.effective_max_inflight
    if config.max_inflight is not None and max_inflight == max_connections:
        remediation = "lower both --max-connections and --max-inflight"
    elif max_inflight > max_connections:
        remediation = "lower --max-inflight"
    else:
        remediation = "lower --max-connections"
    message = (
        f"nofile soft limit {soft_limit} is below {required} "
        f"(2 x upstream_pool_size={pool_size} + {_NOFILE_HEADROOM} headroom, where "
        f"upstream_pool_size = max(--max-connections={max_connections}, "
        f"--max-inflight={max_inflight})); under load the relay exhausts file "
        f"descriptors and clients see raw connection errors. Raise the limit "
        f"(ulimit -n {required}) or {remediation}."
    )
    if strict:
        raise ValueError(message)
    logger.warning(message)


def normalize_log_level(log_level: str) -> str:
    normalized_level = log_level.upper()
    if not isinstance(getattr(logging, normalized_level, None), int):
        return "INFO"
    return normalized_level


def build_log_config(log_level: str) -> dict[str, Any]:
    normalized_level = normalize_log_level(log_level)
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"]["sglang_omni_router"] = {
        "handlers": ["default"],
        "level": normalized_level,
        "propagate": False,
    }
    log_config["loggers"]["httpx"] = {
        "handlers": ["default"],
        "level": "WARNING",
        "propagate": False,
    }
    log_config["loggers"]["httpcore"] = {
        "handlers": ["default"],
        "level": "WARNING",
        "propagate": False,
    }
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        if logger_name in log_config["loggers"]:
            log_config["loggers"][logger_name]["level"] = normalized_level
    return log_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the SGLang-Omni Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker-urls", nargs="+", default=None)
    parser.add_argument("--worker-config", default=None)
    parser.add_argument("--launcher-config", default=None)
    parser.add_argument(
        "--policy",
        choices=get_args(RoutingPolicy),
        default="round_robin",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--request-timeout-secs", type=int, default=1800)
    parser.add_argument("--max-payload-size", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help=(
            "Admission bound: maximum concurrent in-flight model requests "
            "before the router fast-rejects with 503. Default: auto, "
            "128 x workers, capped at 4096. The upstream connection pool is "
            "sized to at least this value, so admitted requests never queue "
            "inside the pool. Explicit values below 64 x workers can "
            "under-feed the pool."
        ),
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help=(
            "Advanced override: decouple the admission bound from "
            "--max-connections (default: equal to it). The upstream pool is "
            "sized to the larger of the two."
        ),
    )
    parser.add_argument("--health-failure-threshold", type=int, default=3)
    parser.add_argument("--health-success-threshold", type=int, default=2)
    parser.add_argument("--health-check-timeout-secs", type=int, default=5)
    parser.add_argument("--health-check-interval-secs", type=int, default=10)
    parser.add_argument("--health-check-endpoint", default="/health")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--strict-limits",
        action="store_true",
        help=(
            "Fail startup instead of warning when the nofile soft limit is too "
            "low for the resolved upstream pool size "
            "(max of --max-connections and --max-inflight)."
        ),
    )
    parser.add_argument(
        "--admin-api-key",
        default=None,
        help=(
            "Bearer token required for all admin endpoints "
            "(pause_generation, update_weights_from_disk, weights_checker, etc.). "
            "Can also be set via the SGLANG_OMNI_ADMIN_KEY environment variable. "
            "If neither is set, admin endpoints are unauthenticated."
        ),
    )
    return parser


def validate_worker_source_args(args: argparse.Namespace) -> None:
    if args.launcher_config:
        if args.worker_urls:
            raise ValueError("--launcher-config cannot be used with --worker-urls")
        if args.worker_config:
            raise ValueError("--launcher-config cannot be used with --worker-config")
        if args.model is not None:
            raise ValueError(
                "--model cannot be used with --launcher-config; set model_name "
                "in the launcher YAML"
            )
    if args.worker_config and args.model is not None:
        raise ValueError("--model cannot be used with --worker-config")


def build_config_from_args(
    args: argparse.Namespace,
    *,
    managed_worker_urls: list[str] | None = None,
    managed_model: str | None = None,
    managed_worker_capabilities: set[Capability] | None = None,
) -> RouterConfig:
    validate_worker_source_args(args)
    if args.launcher_config and managed_worker_urls is None:
        raise ValueError("managed worker URLs are required for --launcher-config")
    workers = load_worker_configs(args.worker_config) if args.worker_config else None
    worker_urls = managed_worker_urls if args.launcher_config else args.worker_urls
    model = managed_model if args.launcher_config else args.model
    if args.launcher_config and managed_worker_urls is not None:
        workers = [
            WorkerConfig(
                url=worker_url,
                model=model,
                capabilities=set(managed_worker_capabilities or DEFAULT_CAPABILITIES),
            )
            for worker_url in managed_worker_urls
        ]
        worker_urls = None
    return build_router_config(
        worker_urls=worker_urls,
        workers=workers,
        host=args.host,
        port=args.port,
        policy=args.policy,
        model=model,
        request_timeout_secs=args.request_timeout_secs,
        max_payload_size=args.max_payload_size,
        max_connections=args.max_connections,
        max_inflight=args.max_inflight,
        health_failure_threshold=args.health_failure_threshold,
        health_success_threshold=args.health_success_threshold,
        health_check_timeout_secs=args.health_check_timeout_secs,
        health_check_interval_secs=args.health_check_interval_secs,
        health_check_endpoint=args.health_check_endpoint,
    )


def resolve_managed_worker_capabilities(
    launcher_config: LocalLauncherConfig,
) -> set[Capability]:
    if launcher_config.worker_capabilities is not None:
        return set(launcher_config.worker_capabilities)

    extra_args = shlex.split(launcher_config.worker_extra_args)
    if "--text-only" in extra_args:
        return set(DEFAULT_CAPABILITIES) - {"speech", "audio_output"}

    return set(DEFAULT_CAPABILITIES)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_level = normalize_log_level(args.log_level)
    log_config = build_log_config(args.log_level)
    logging.config.dictConfig(log_config)
    launcher: LocalLauncher | None = None
    try:
        validate_worker_source_args(args)
        if args.launcher_config:
            launcher_config = load_launcher_config(args.launcher_config)
            launcher = LocalLauncher(launcher_config)
            logger.info(f"Starting managed Omni V1 workers from {args.launcher_config}")
            managed_worker_urls = launcher.launch_and_wait()
            config = build_config_from_args(
                args,
                managed_worker_urls=managed_worker_urls,
                managed_model=launcher_config.model_name,
                managed_worker_capabilities=resolve_managed_worker_capabilities(
                    launcher_config
                ),
            )
        else:
            config = build_config_from_args(args)

        check_file_descriptor_limit(config, strict=args.strict_limits)
        logger.info(f"Starting SGLang-Omni Router on {config.host}:{config.port}")
        logger.info(
            f"Router configuration: workers={len(config.workers)} | "
            f"policy={config.policy} | "
            f"max_payload_size={config.max_payload_size} | "
            f"max_connections={config.max_connections} | "
            f"max_inflight={config.effective_max_inflight} | "
            f"upstream_pool={config.upstream_pool_size} | "
            f"health_failure_threshold={config.health_failure_threshold} | "
            f"health_success_threshold={config.health_success_threshold} | "
            f"health_check_endpoint={config.health_check_endpoint} | "
            f"health_check_interval_secs={config.health_check_interval_secs} | "
            f"health_check_timeout_secs={config.health_check_timeout_secs} | "
            f"readiness_requires_routable_worker=true"
        )
        uvicorn.run(
            create_app(config, admin_api_key=getattr(args, "admin_api_key", None)),
            host=config.host,
            port=config.port,
            log_level=log_level.lower(),
            log_config=log_config,
        )
    except (ValueError, ValidationError) as exc:
        parser.error(str(exc))
    except (RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130)
    finally:
        if launcher is not None:
            logger.info("Stopping managed Omni V1 workers")
            launcher.shutdown()


if __name__ == "__main__":
    main()
