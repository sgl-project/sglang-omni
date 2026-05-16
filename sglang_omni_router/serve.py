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
    parser.add_argument("--max-connections", type=int, default=100)
    parser.add_argument("--health-failure-threshold", type=int, default=3)
    parser.add_argument("--health-success-threshold", type=int, default=2)
    parser.add_argument("--health-check-timeout-secs", type=int, default=5)
    parser.add_argument("--health-check-interval-secs", type=int, default=10)
    parser.add_argument("--health-check-endpoint", default="/health")
    parser.add_argument("--log-level", default="info")
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

        logger.info(f"Starting SGLang-Omni Router on {config.host}:{config.port}")
        logger.info(
            f"Router configuration: workers={len(config.workers)} | "
            f"policy={config.policy} | "
            f"max_payload_size={config.max_payload_size} | "
            f"max_connections={config.max_connections} | "
            f"health_failure_threshold={config.health_failure_threshold} | "
            f"health_success_threshold={config.health_success_threshold} | "
            f"health_check_endpoint={config.health_check_endpoint} | "
            f"health_check_interval_secs={config.health_check_interval_secs} | "
            f"health_check_timeout_secs={config.health_check_timeout_secs} | "
            f"readiness_requires_routable_worker=true"
        )
        uvicorn.run(
            create_app(config),
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
