# SPDX-License-Identifier: Apache-2.0
"""CLI entry point wiring the parser, sender, responder, and builder together.

Run as python -m stress_test_pack.run_stress from the workspace root.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Sequence
from typing import Any

from stress_test_pack.builder import (
    print_console_report,
    write_jsonl,
    write_summary_json,
)
from stress_test_pack.config_parser import StressConfig, describe_plan, parse_config
from stress_test_pack.responder import Responder

try:
    from stress_test_pack.sender import StopSignal, run_load
except ImportError as exc:
    raise SystemExit(f"cannot start the stress client: {exc}") from None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_config(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if config.dry_run:
        print(describe_plan(config))
        return 0

    logging.basicConfig(
        level=logging.WARNING if config.quiet else logging.INFO,
        format="%(message)s",
    )
    responder = Responder(config)
    stop = StopSignal()
    stats = asyncio.run(execute(config, responder, stop))
    wall_clock_s = stats.pop("wall_clock_s", 0.0)
    report = responder.build_report(wall_clock_s=wall_clock_s, load_stats=stats)

    output = write_jsonl(config, responder.records, report)
    summary_path = write_summary_json(config, report)
    if not config.quiet:
        print_console_report(config, report, output, summary_path)

    if stop.stopped and stop.reason == "sigint":
        return 130
    if config.strict and (report.failed or report.broken_streams):
        return 1
    return 0


async def execute(
    config: StressConfig, responder: Responder, stop: StopSignal
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop.request, "sigint")
    except (NotImplementedError, RuntimeError, ValueError):
        signal.signal(signal.SIGINT, lambda *_: stop.request("sigint"))
    return await run_load(config, responder, stop=stop)


if __name__ == "__main__":
    sys.exit(main())
