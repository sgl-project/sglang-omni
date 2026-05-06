from __future__ import annotations

from unittest.mock import patch

from sglang_omni_router.launcher import build_parser, main


def test_router_arg_parser_accepts_worker_urls_after_single_flag() -> None:
    args = build_parser().parse_args(
        [
            "--worker-urls",
            "http://127.0.0.1:8101",
            "http://127.0.0.1:8102",
            "--policy",
            "least_request",
        ]
    )

    assert args.worker_urls == ["http://127.0.0.1:8101", "http://127.0.0.1:8102"]
    assert args.policy == "least_request"


def test_router_main_builds_app_and_runs_uvicorn() -> None:
    with patch("sglang_omni_router.launcher.uvicorn.run") as run:
        main(
            [
                "--port",
                "8123",
                "--worker-urls",
                "http://127.0.0.1:8101",
                "http://127.0.0.1:8102",
                "--policy",
                "round_robin",
            ]
        )

    run.assert_called_once()
    _, kwargs = run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8123
    assert kwargs["log_level"] == "info"
