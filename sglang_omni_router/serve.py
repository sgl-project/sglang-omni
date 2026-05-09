# SPDX-License-Identifier: Apache-2.0
"""Serve the external Omni router process."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import get_args

import uvicorn

from sglang_omni_router.app import create_app
from sglang_omni_router.config import RoutingPolicy, build_router_config

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the SGLang-Omni Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker-urls", nargs="+", required=True)
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


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    config = build_router_config(
        worker_urls=args.worker_urls,
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
    logger.info(
        f"Starting SGLang-Omni Router on {config.host}:{config.port} | "
        f"workers={len(config.worker_urls)} | policy={config.policy} | "
        f"max_payload_size={config.max_payload_size} | "
        f"max_connections={config.max_connections} | "
        f"health_failure_threshold={config.health_failure_threshold} | "
        f"health_success_threshold={config.health_success_threshold} | "
        f"health_check_interval_secs={config.health_check_interval_secs} | "
        f"health_check_timeout_secs={config.health_check_timeout_secs}"
    )
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
