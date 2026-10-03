# SPDX-License-Identifier: Apache-2.0
"""Verify custom judge isolation, identities, retained outcomes and denominators."""

from __future__ import annotations

import copy
import fcntl
import json

import pytest
from pydantic import ValidationError

from benchmarks.duplex import reference_behavior, reference_core, reference_custom_judge
from benchmarks.duplex.reference_source import load_official_behavior, verify_reference
from benchmarks.eval import benchmark_duplex_reference as cli_module
from tests.unit_test.benchmarks.test_duplex_reference import (
    REF,
    build_trees,
    response,
    tree_digest,
    write_transcripts,
)


@pytest.fixture
def custom_run(tmp_path):
    if REF is None:
        pytest.skip("Set FDB_REFERENCE_SOURCE to the pinned external checkout")
    trees = build_trees(tmp_path)
    source = tmp_path / "reference-scores"
    for name, tree in trees.items():
        engine = reference_core.Engine(
            source, name, tree, tree / "reference-manifest.json", None
        )
        for sid in engine.samples:
            sample = engine.sample_dir(sid)
            sample.mkdir(parents=True)
            write_transcripts(sample)
            for stem in ("input", "clean_input", "output", "clean_output"):
                reference_core.atomic_write_json(
                    sample / "receipts" / f"asr-{stem}.json",
                    {
                        "status": "ok",
                        "transcript_sha256": reference_core.sha256_file(
                            sample / f"{stem}.json"
                        ),
                        "audio_sha256": reference_core.sha256_file(
                            tree / sid / f"{stem}.wav"
                        ),
                    },
                )
            reference_core.atomic_write_json(
                sample / "judge" / "request.json", {"official": "request"}
            )
            reference_core.atomic_write_json(
                sample / "judge" / "result.json", {"official": "result"}
            )
            reference_core.atomic_write_json(
                sample / "content_tag.json", {"official": "tag"}
            )
    reference_core.atomic_write_json(source / "summary.json", {"official": "summary"})
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"model_revision": "a" * 40, "dtype": "bfloat16"}))
    config = tmp_path / "judge.json"
    reference_core.atomic_write_json(
        config,
        {
            "model_id": "Qwen/Qwen3-32B",
            "model_revision": "a" * 40,
            "tokenizer_id": "Qwen/Qwen3-32B",
            "tokenizer_revision": "a" * 40,
            "served_model": "Qwen3-32B-custom-judge",
            "precision": "bf16",
            "enable_thinking": False,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": 512,
            },
            "seeds": [1, 2, 3],
            "server_launch_receipt": launch.name,
            "server_launch_receipt_sha256": reference_core.sha256_file(launch),
        },
    )
    argv = [
        "custom-judge",
        "--reference-source",
        str(REF),
        "--source-scores",
        str(source),
        "--out",
        str(tmp_path / "custom"),
        "--judge-config",
        str(config),
        "--base-url",
        "http://localhost:8109/v1",
        "--retry-sleep-s",
        "0",
        *[f"--tree={name}={tree}" for name, tree in trees.items()],
    ]
    args = cli_module.build_parser().parse_args(argv)
    return args, cli_module.open_engines(args), verify_reference(REF)


def custom_response(
    content='{"behaviour": ["C_RESPOND"]}',
    *,
    model="Qwen3-32B-custom-judge",
    finish="stop",
):
    result = response(content, model)
    result["choices"][0]["finish_reason"] = finish
    return result


def test_custom_payload_resume_and_source_bytes_unchanged(custom_run):
    args, engines, paths = custom_run
    before = tree_digest(args.source_scores)
    audio_before = {engine.name: tree_digest(engine.tree) for engine in engines}
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    sent = []

    def transport(body, seed):
        sent.append((body, seed))
        return custom_response()

    assert reference_custom_judge.run_custom(args, engines, paths, transport) == {
        "valid": 3,
        "variant_ineligible": 1,
    }
    expected = [
        reference_behavior.build_request(official, engine.sample_dir(sid))["body"][
            "messages"
        ]
        for engine in engines
        for sid in sorted(engine.samples)
        if all(engine.eligible(sid, v) for v in reference_core.VARIANTS)
    ]
    assert [body["messages"] for body, seed in sent] == expected
    assert all(
        body["model"] == "Qwen3-32B-custom-judge" and seed == 1 for body, seed in sent
    )
    assert sent[0][0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert sent[0][0]["max_tokens"] == 512 and sent[0][0]["temperature"] == 0
    reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: pytest.fail("resent")
    )
    args.phase = "custom-summarize"
    reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: pytest.fail("API call")
    )
    assert tree_digest(args.source_scores) == before
    assert {engine.name: tree_digest(engine.tree) for engine in engines} == audio_before
    assert not list(args.out.rglob("content_tag.json"))
    request = reference_core.read_json(next(args.out.rglob("request.json")))
    assert len(request["audio_sha256"]) == len(request["transcript_sha256"]) == 4
    summary = reference_core.read_json(args.out / "summary.json")
    assert summary["scope"] == "custom_behavior_judge_non_official"
    assert summary["engines"]["vllm"]["all"]["selected_pairs"] == 2
    assert summary["engines"]["vllm"]["all"]["eligible_pairs"] == 1
    for group in ("all", "user_interruption"):
        proportions = summary["engines"]["vllm"][group]["valid_label_proportions"]
        assert proportions["C_RESPOND"]["ci95"] == pytest.approx(
            [0.2065493143772374, 1.0]
        )
        assert proportions["C_RESUME"]["ci95"] == pytest.approx(
            [0.0, 0.7934506856227626]
        )


@pytest.mark.parametrize(
    "field",
    ["model_revision", "tokenizer_revision", "temperature", "seeds", "served_model"],
)
def test_custom_config_drift_refuses_resume(custom_run, field):
    args, engines, paths = custom_run
    args.phase = "custom-summarize"
    reference_custom_judge.run_custom(args, engines, paths)
    config = reference_core.read_json(args.judge_config)
    if field == "temperature":
        config["decoding"][field] = 0.7
    elif field == "seeds":
        config[field] = [4, 5, 6]
    elif field == "served_model":
        config[field] = "another-served-name"
    else:
        config[field] = "b" * 40
    reference_core.atomic_write_json(args.judge_config, config)
    with pytest.raises(SystemExit, match="identity changed"):
        reference_custom_judge.run_custom(args, engines, paths)


@pytest.mark.parametrize("kind", ["transcript_bytes", "audio", "asr_receipt"])
def test_custom_input_drift_does_not_reuse_result(custom_run, kind):
    args, engines, paths = custom_run
    reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: custom_response()
    )
    engine, sid = engines[0], sorted(engines[0].samples)[0]
    sample = engine.sample_dir(sid)
    receipt_path = sample / "receipts" / "asr-output.json"
    receipt = reference_core.read_json(receipt_path)
    if kind == "transcript_bytes":
        transcript = sample / "output.json"
        transcript.write_text(transcript.read_text() + "\n")
        receipt["transcript_sha256"] = reference_core.sha256_file(transcript)
    elif kind == "audio":
        audio = engine.source_audio(sid, "output.wav")
        audio.write_bytes(audio.read_bytes() + b"\0")
        receipt["audio_sha256"] = reference_core.sha256_file(audio)
    else:
        receipt["recorded_at"] = "changed"
    reference_core.atomic_write_json(receipt_path, receipt)
    counts = reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: pytest.fail("called")
    )
    assert counts == {"stale_request": 1, "valid": 2, "variant_ineligible": 1}


@pytest.mark.parametrize(
    "reply,status",
    [
        (custom_response(finish="length"), "invalid_finish"),
        (custom_response(finish=None), "invalid_finish"),
        (custom_response(model="wrong"), "model_mismatch"),
        (custom_response('{"behaviour": ["C_FOO"]}'), "invalid_label"),
        (custom_response("malformed"), "failed"),
    ],
)
def test_custom_invalid_results_preserve_denominators(custom_run, reply, status):
    args, engines, paths = custom_run
    args.limit = 1
    counts = reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: copy.deepcopy(reply)
    )
    assert counts == {status: 1, "not_judged": 2, "variant_ineligible": 1}
    summary = reference_core.read_json(args.out / "summary.json")
    for engine in summary["engines"].values():
        all_rows = engine["all"]
        assert sum(all_rows["status"].values()) == all_rows["selected_pairs"] == 2
        assert all_rows["valid_n"] == 0
        assert all(
            row["proportion"] is None
            for row in all_rows["valid_label_proportions"].values()
        )
        assert all(
            row["ci95"] is None for row in all_rows["valid_label_proportions"].values()
        )
    assert summary["engines"]["sgl"]["all"]["attempted_pairs"] == 1
    assert summary["engines"]["sgl"]["all"]["attempts"] == (
        3 if status == "failed" else 1
    )


def test_custom_failed_retry_keeps_all_attempts(custom_run):
    args, engines, paths = custom_run
    args.limit = 1
    reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: custom_response("malformed")
    )
    args.retry_failed = True
    reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: custom_response()
    )
    result = reference_core.read_json(next(args.out.rglob("result.json")))
    assert result["status"] == "valid"
    assert [attempt["seed"] for attempt in result["attempts"]] == [1, 2, 3, 1]
    assert all("error" in attempt for attempt in result["attempts"][:3])


def test_custom_preserves_reference_embedded_json_policy(custom_run):
    args, engines, paths = custom_run
    content = 'Explanation before {"behaviour": ["C_RESUME"]} and after'
    counts = reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: custom_response(content)
    )
    assert counts == {"valid": 3, "variant_ineligible": 1}
    for path in args.out.rglob("result.json"):
        result = reference_core.read_json(path)
        assert result["label"] == "C_RESUME"
        assert (
            result["attempts"][0]["response"]["choices"][0]["message"]["content"]
            == content
        )


def test_custom_source_missing_receipt_never_writes_source(custom_run):
    args, _, _ = custom_run
    (args.source_scores / "engines" / "sgl" / "manifest-receipt.json").unlink()
    before = tree_digest(args.source_scores)
    with pytest.raises(SystemExit, match="lacks existing"):
        cli_module.open_engines(args)
    assert tree_digest(args.source_scores) == before


def test_custom_asr_blocked_pairs_remain_in_summary(custom_run):
    args, engines, paths = custom_run
    engine, sid = engines[0], sorted(engines[0].samples)[0]
    reference_core.atomic_write_json(
        engine.sample_dir(sid) / "receipts" / "asr-output.json", {"status": "failed"}
    )
    counts = reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: custom_response()
    )
    assert counts == {"asr_failed": 1, "valid": 2, "variant_ineligible": 1}
    summary = reference_core.read_json(args.out / "summary.json")
    assert summary["engines"]["sgl"]["all"]["asr_ready_pairs"] == 1
    assert summary["engines"]["sgl"]["all"]["eligible_pairs"] == 2


def test_custom_result_request_mismatch_does_not_reuse(custom_run):
    args, engines, paths = custom_run
    reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: custom_response()
    )
    path = next(args.out.rglob("result.json"))
    result = reference_core.read_json(path)
    result["request_hash"] = "b" * 64
    reference_core.atomic_write_json(path, result)
    counts = reference_custom_judge.run_custom(
        args, engines, paths, lambda *_: pytest.fail("called")
    )
    assert counts == {"result_request_mismatch": 1, "valid": 2, "variant_ineligible": 1}


def test_custom_concurrent_writer_is_rejected(custom_run):
    args, engines, paths = custom_run
    args.out.mkdir()
    with open(args.out / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="another custom judge phase"):
            reference_custom_judge.run_custom(
                args, engines, paths, lambda *_: pytest.fail("called")
            )


def test_custom_direct_call_rejects_source_output_alias(custom_run):
    args, engines, paths = custom_run
    before = tree_digest(args.source_scores)
    args.out = args.source_scores
    with pytest.raises(SystemExit, match="independent"):
        reference_custom_judge.run_custom(args, engines, paths)
    assert tree_digest(args.source_scores) == before


def test_custom_launch_receipt_requires_matching_hash(custom_run):
    args, engines, paths = custom_run
    (args.judge_config.parent / "launch.json").write_text("{}")
    with pytest.raises(SystemExit, match="launch receipt differs"):
        reference_custom_judge.run_custom(
            args, engines, paths, lambda *_: pytest.fail("called")
        )


@pytest.mark.parametrize(
    "field,value",
    [("model_revision", "main"), ("precision", "fp8"), ("enable_thinking", True)],
)
def test_custom_config_rejects_unpinned_or_wrong_mode(custom_run, field, value):
    args, engines, paths = custom_run
    config = reference_core.read_json(args.judge_config)
    config[field] = value
    reference_core.atomic_write_json(args.judge_config, config)
    with pytest.raises(ValidationError):
        reference_custom_judge.run_custom(args, engines, paths)


def test_custom_cli_requires_endpoint_and_independent_out(custom_run, tmp_path):
    args, _, _ = custom_run
    argv = [
        "custom-judge",
        "--reference-source",
        str(REF),
        "--source-scores",
        str(args.source_scores),
        "--out",
        str(args.out),
        "--judge-config",
        str(args.judge_config),
        *[f"--tree={v}" for v in args.tree],
    ]
    with pytest.raises(SystemExit):
        cli_module.build_parser().parse_args(argv)
    link = tmp_path / "scores-link"
    link.symlink_to(args.source_scores, target_is_directory=True)
    argv[argv.index("--out") + 1] = str(link)
    with pytest.raises(SystemExit):
        cli_module.main([*argv, "--base-url", "http://localhost:8109/v1"])
