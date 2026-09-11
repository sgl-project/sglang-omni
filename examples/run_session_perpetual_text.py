# SPDX-License-Identifier: Apache-2.0
"""Perpetual text generation over one streaming session (MiniCPM-o thinker).

Drives the text pipeline with ``enable_streaming_session`` on: opens a
session, runs several turns that each stop at a boundary token while the
thinker keeps the turn's KV in its SessionSlot, extends the session with
appended input between turns, and closes the session at the end.

Turn 1 sends a chat-templated prompt; every later turn sends only the
appended token ids — the scheduler concatenates them onto the previous
turn's context via the session, so prefill covers just the delta on top of
the retained KV. After close the demo checks the KV pool returns to its
baseline.

Usage:
    python examples/run_session_perpetual_text.py \
        --model-path openbmb/MiniCPM-o-4_5 --rounds 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

THINKER = "thinker"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="openbmb/MiniCPM-o-4_5",
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell a long story about a lighthouse keeper, one paragraph at a time.",
        help="Opening user prompt.",
    )
    parser.add_argument(
        "--extend-text",
        type=str,
        default=None,
        help=(
            "Text appended to the session before each follow-up turn. "
            "Defaults to the boundary token itself, so the model resumes "
            "the next paragraph right where the previous turn stopped."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of generation turns over the one session (default: 3).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per turn (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--boundary-token-id",
        type=int,
        action="append",
        default=None,
        help=(
            "Token id that ends a turn while keeping the session's KV; "
            "repeatable. Defaults to the tokenizer's newline-pair token when "
            "resolvable, else turns end at max-new-tokens/EOS."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-turn pipeline timeout in seconds (default: 300).",
    )
    return parser


def _result_text(result: object) -> str:
    """Find the terminal text in the nested per-stage result dict."""
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str) and text:
            return text
        for event in result.get("events") or []:
            if isinstance(event, dict) and isinstance(event.get("text"), str):
                return event["text"]
        for value in result.values():
            found = _result_text(value)
            if found:
                return found
    return ""


def _stage_data(admin_result: dict, stage: str = THINKER) -> dict:
    for item in admin_result["results"]:
        if item["stage"] == stage:
            return item["data"]
    raise KeyError(f"no admin result from stage {stage!r}: {admin_result}")


async def _thinker_info(coordinator) -> dict:
    return _stage_data(await coordinator.model_info(stages=[THINKER]))


def _resolve_boundary_ids(tokenizer, override: list[int] | None) -> list[int]:
    if override:
        return list(override)
    ids = tokenizer.encode("\n\n", add_special_tokens=False)
    return ids if len(ids) == 1 else []


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from sglang_omni.models.minicpm_o.config import MiniCPMOPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    try:
        from examples.launchers._common import apply_stage_factory_updates
    except ModuleNotFoundError:
        from launchers._common import apply_stage_factory_updates

    config = MiniCPMOPipelineConfig(model_path=args.model_path)
    apply_stage_factory_updates(
        config,
        stage_name=THINKER,
        server_arg_updates={"enable_streaming_session": True},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    boundary_ids = _resolve_boundary_ids(tokenizer, args.boundary_token_id)
    if boundary_ids:
        logger.info("Turn boundary token ids: %s", boundary_ids)
    else:
        logger.warning("No boundary token resolved; turns end at max-new-tokens/EOS")

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting text pipeline with streaming sessions enabled...")
    await runner.start(timeout=600)
    try:
        baseline = await _thinker_info(runner.coordinator)
        # Evictable radix-cache entries are reclaimable on demand, so the
        # effective free-pool baseline counts them alongside available tokens.
        kv_baseline = (
            baseline["kv_available_tokens"] + baseline["kv_evictable_tokens"]
        )
        logger.info("KV baseline: %d free tokens", kv_baseline)

        opened = await runner.coordinator.open_session(stages=[THINKER])
        session_id = _stage_data(opened)["session_id"]
        assert session_id, f"open_session failed: {opened}"
        logger.info("Opened session %s", session_id)

        params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "stop_token_ids": boundary_ids,
            "session_params": {"session_id": session_id},
        }
        if args.extend_text is not None:
            extend_ids = tokenizer.encode(args.extend_text, add_special_tokens=False)
        else:
            # Feed the boundary token back: the turn stopped right before
            # emitting it, so appending it resumes the next paragraph.
            extend_ids = list(boundary_ids)
        assert extend_ids, "nothing to append between turns; pass --extend-text"

        for turn in range(args.rounds):
            if turn == 0:
                inputs = {"messages": [{"role": "user", "content": args.prompt}]}
            else:
                # Appended token ids only: the session supplies everything
                # before them from its retained KV.
                inputs = {"messages": list(extend_ids)}
            result = await asyncio.wait_for(
                runner.coordinator.submit(
                    f"session-turn-{uuid.uuid4().hex[:8]}",
                    OmniRequest(inputs=inputs, params=dict(params)),
                ),
                timeout=args.timeout,
            )
            text = _result_text(result)
            info = await _thinker_info(runner.coordinator)
            logger.info(
                "Turn %d done: %d chars, session holds %d KV tokens",
                turn + 1,
                len(text),
                info["session_held_tokens"],
            )
            print(f"\n--- turn {turn + 1} ---\n{text}")
            assert info["session_held_tokens"] > 0, (
                "session should retain KV between turns"
            )

        closed = await runner.coordinator.close_session(
            {"session_id": session_id}, stages=[THINKER]
        )
        assert closed["success"], f"close_session failed: {closed}"
        final = await _thinker_info(runner.coordinator)
        kv_free = final["kv_available_tokens"] + final["kv_evictable_tokens"]
        logger.info(
            "Closed session %s: KV free %d/%d, %d held by sessions",
            session_id,
            kv_free,
            kv_baseline,
            final["session_held_tokens"],
        )
        assert final["session_held_tokens"] == 0, "close must release session KV"
        assert kv_free == kv_baseline, (
            "KV pool must return to baseline after close"
        )
        print("\nSession demo passed: KV retained across turns, released on close.")
    finally:
        await runner.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
