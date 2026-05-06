# SPDX-License-Identifier: Apache-2.0
"""Configuration and validation for the external Omni router."""

from __future__ import annotations

import ipaddress
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator, model_validator

Capability = Literal["chat", "speech", "streaming"]
RoutingPolicy = Literal["round_robin", "least_request", "random"]

DEFAULT_CAPABILITIES: set[Capability] = {"chat", "speech", "streaming"}
CLOUD_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal"}


def normalize_worker_url(url: str) -> str:
    """Validate and normalize a worker base URL."""
    if not isinstance(url, str):
        raise ValueError("worker URL must be a string")

    raw_url = url.strip()
    if not raw_url:
        raise ValueError("worker URL must not be empty")

    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("worker URL scheme must be http or https")
    if not parsed.netloc:
        raise ValueError("worker URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("worker URL must not include user-info")
    if parsed.path not in {"", "/"}:
        raise ValueError("worker URL must be a base URL without a path")
    if parsed.query or parsed.fragment:
        raise ValueError("worker URL must not include query or fragment")

    host = parsed.hostname
    if host is None:
        raise ValueError("worker URL must include a host")

    normalized_host = host.lower()
    if normalized_host in CLOUD_METADATA_HOSTS:
        raise ValueError(f"worker URL host {host!r} is not allowed")

    try:
        ip = ipaddress.ip_address(normalized_host.strip("[]"))
    except ValueError:
        ip = None
    if ip is not None and ip.is_link_local:
        raise ValueError(f"worker URL host {host!r} is link-local")

    netloc = normalized_host
    if ":" in normalized_host and not normalized_host.startswith("["):
        netloc = f"[{normalized_host}]"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


class WorkerConfig(BaseModel):
    url: str
    model: str | None = None
    capabilities: set[Capability] = Field(
        default_factory=lambda: set(DEFAULT_CAPABILITIES)
    )

    @field_validator("url")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        return normalize_worker_url(value)

    @field_validator("capabilities")
    @classmethod
    def _validate_capabilities(cls, value: set[Capability]) -> set[Capability]:
        if not value:
            raise ValueError("worker capabilities must not be empty")
        return value


class RouterConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    worker_urls: list[WorkerConfig]
    policy: RoutingPolicy = "round_robin"
    model: str | None = None
    request_timeout_secs: int = 1800
    max_payload_size: int = 512 * 1024 * 1024
    health_failure_threshold: int = 3
    health_success_threshold: int = 2
    health_check_timeout_secs: int = 5
    health_check_interval_secs: int = 60
    health_check_endpoint: str = "/health"
    route_log_path: str | None = None

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if value <= 0 or value > 65535:
            raise ValueError("port must be in [1, 65535]")
        return value

    @field_validator(
        "request_timeout_secs",
        "max_payload_size",
        "health_failure_threshold",
        "health_success_threshold",
        "health_check_timeout_secs",
        "health_check_interval_secs",
    )
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0")
        return value

    @field_validator("health_check_endpoint")
    @classmethod
    def _validate_health_endpoint(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("health_check_endpoint must start with /")
        if "?" in value or "#" in value:
            raise ValueError("health_check_endpoint must not include query or fragment")
        return value

    @model_validator(mode="after")
    def _validate_workers(self) -> "RouterConfig":
        if not self.worker_urls:
            raise ValueError("at least one worker URL is required")
        urls = [worker.url for worker in self.worker_urls]
        duplicates = sorted({url for url in urls if urls.count(url) > 1})
        if duplicates:
            raise ValueError(f"duplicate worker URLs: {', '.join(duplicates)}")
        return self


def build_router_config(
    *,
    worker_urls: list[str],
    host: str = "0.0.0.0",
    port: int = 8000,
    policy: RoutingPolicy = "round_robin",
    model: str | None = None,
    request_timeout_secs: int = 1800,
    max_payload_size: int = 512 * 1024 * 1024,
    health_failure_threshold: int = 3,
    health_success_threshold: int = 2,
    health_check_timeout_secs: int = 5,
    health_check_interval_secs: int = 60,
    health_check_endpoint: str = "/health",
    route_log_path: str | None = None,
) -> RouterConfig:
    workers = [WorkerConfig(url=url, model=model) for url in worker_urls]
    return RouterConfig(
        host=host,
        port=port,
        worker_urls=workers,
        policy=policy,
        model=model,
        request_timeout_secs=request_timeout_secs,
        max_payload_size=max_payload_size,
        health_failure_threshold=health_failure_threshold,
        health_success_threshold=health_success_threshold,
        health_check_timeout_secs=health_check_timeout_secs,
        health_check_interval_secs=health_check_interval_secs,
        health_check_endpoint=health_check_endpoint,
        route_log_path=route_log_path,
    )
