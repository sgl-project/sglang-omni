# SPDX-License-Identifier: Apache-2.0
"""Serve the external Omni router process."""

from __future__ import annotations

import argparse
import copy
import logging
import logging.config
from collections.abc import Sequence
from typing import Any, get_args

import uvicorn
from pydantic import ValidationError

from sglang_omni_router.app import create_app
from sglang_omni_router.config import (
    RouterConfig,
    RoutingPolicy,
    build_router_config,
    load_worker_configs,
)

logger = logging.getLogger(__name__)


def normalize_log_level(log_level: str) -> str:
    normalized_level = log_level.upper()
    if not isinstance(getattr(logging, normalized_level, None), int):
        return "INFO"
    return normalized_level


def build_log_config(log_level: str) -> dict[str, Any]:
    normalized_level = normalize_log_level(log_level)
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = "%(levelprefix)s %(name)s:%(message)s"
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


def build_config_from_args(args: argparse.Namespace) -> RouterConfig:
    if args.worker_config and args.model is not None:
        raise ValueError("--model cannot be used with --worker-config")
    workers = load_worker_configs(args.worker_config) if args.worker_config else None
    return build_router_config(
        worker_urls=args.worker_urls,
        workers=workers,
        host=args.host,
        port=args.port,
        policy=args.policy,
        model=args.model,
        request_timeout_secs=args.request_timeout_secs,
        max_payload_size=args.max_payload_size,
        max_connections=args.max_connections,
        health_failure_threshold=args.health_failure_threshold,
        health_success_threshold=args.health_success_threshold,
        health_check_timeout_secs=args.health_check_timeout_secs,
        health_check_interval_secs=args.health_check_interval_secs,
        health_check_endpoint=args.health_check_endpoint,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_level = normalize_log_level(args.log_level)
    log_config = build_log_config(args.log_level)
    logging.config.dictConfig(log_config)
    try:
        config = build_config_from_args(args)
    except (ValueError, ValidationError) as exc:
        parser.error(str(exc))
    logger.info(
        f"Starting SGLang-Omni Router on {config.host}:{config.port} | "
        f"workers={len(config.worker_urls)} | policy={config.policy} | "
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


if __name__ == "__main__":
    main()
