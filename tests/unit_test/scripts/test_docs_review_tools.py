# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for bounded Git inspection in documentation reviews."""

from __future__ import annotations

import json
from dataclasses import replace
from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from scripts.docs_review.collect import collect_review_bundle, review_input_sha256
from scripts.docs_review.github_comment import render_advisory_comment
from scripts.docs_review.models import (
    CollectedFile,
    FileExcerpt,
    Finding,
    ModelSettings,
    ReviewArtifact,
    ReviewRequest,
)
from scripts.docs_review.policy import DOCS_MATCH_CODE, STYLE_GUIDE_PATH, read_prompt
from scripts.docs_review.reporting import render_markdown
from scripts.docs_review.review import (
    SystemClock,
    citation_matches,
    prepare_review,
    run_review,
)
from scripts.docs_review.tools import GitReadOnlyToolbox
from scripts.docs_review.transport import OpenAICompatibleReviewer, ReviewerError
from tests.unit_test.fixtures.docs_review import (
    commit_all,
    sample_budget,
    start_repository,
    write_file,
)

LEAF_PATH = "internal/codec_constants.py"
DOCUMENT_PATH = "docs/cookbook/demo.md"
RATE_DEFINITION = "AUDIO_SAMPLE_RATE = 24000"
DOCUMENT_CLAIM = "Use `cli.py`. The default sample rate is 48000 Hz."


@pytest.fixture
def review_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewRequest:
    repository = tmp_path / "repo"
    start_repository(repository)
    write_file(repository, "cli.py", "from pipeline_defaults import sample_rate\n")
    write_file(
        repository,
        "pipeline_defaults.py",
        "from internal.codec_constants import AUDIO_SAMPLE_RATE as sample_rate\n",
    )
    write_file(repository, LEAF_PATH, RATE_DEFINITION + "\n")
    write_file(repository, "flags.txt", "--sample-rate\n")
    write_file(repository, "--exec=id", "tracked but forbidden\n")
    write_file(
        repository,
        "large.py",
        "\n".join(f"line {line_number} " + "é" * 60 for line_number in range(1, 301)),
    )
    (repository / "binary.bin").write_bytes(b"\xff\x00\xfe")
    base_commit_sha = commit_all(repository, "add audio defaults")
    write_file(repository, DOCUMENT_PATH, "# Demo\n\n" + DOCUMENT_CLAIM + "\n")
    head_commit_sha = commit_all(repository, "document CLI defaults")
    monkeypatch.setenv("DOCS_REVIEW_TEST_KEY", "fake-key")
    return ReviewRequest(
        repository=repository,
        base_commit_sha=base_commit_sha,
        head_commit_sha=head_commit_sha,
        pull_request_number=7,
        pull_request_text=None,
        model_settings=ModelSettings(
            model_name="reviewer",
            endpoint_url="https://example.invalid/chat",
            max_output_tokens=512,
            api_key_environment_variable="DOCS_REVIEW_TEST_KEY",
            max_tool_rounds=2,
        ),
        budget=sample_budget(),
        ledger_path=tmp_path / "ledger.sqlite",
    )


class ScriptedEndpoint:
    def __init__(self, responses: list[bytes | int]) -> None:
        self.responses = iter(responses)
        self.bodies: list[bytes] = []

    def open(self, request: Request, timeout: float) -> BytesIO:
        assert 0 < timeout <= 60
        assert isinstance(request.data, bytes)
        self.bodies.append(request.data)
        response = next(self.responses)
        if isinstance(response, int):
            raise HTTPError(
                request.full_url, response, "model request rejected", Message(), None
            )
        else:
            return BytesIO(response)


def call_response(
    name: str, arguments: dict[str, str | int], finish_reason: str | None
) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": name + "-1",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": finish_reason,
                }
            ]
        }
    ).encode()


def final_response(findings: list[Finding]) -> bytes:
    content = json.dumps([finding.model_dump() for finding in findings])
    return json.dumps(
        {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    ).encode()


@pytest.mark.parametrize("read_evidence", [True, False], ids=["read-leaf", "grep-only"])
def test_multihop_evidence_must_be_read_before_citation(
    review_request: ReviewRequest, monkeypatch: pytest.MonkeyPatch, read_evidence: bool
) -> None:
    finding = Finding(
        rule_id=DOCS_MATCH_CODE,
        file_path=DOCUMENT_PATH,
        line_start=3,
        line_end=3,
        quote=DOCUMENT_CLAIM,
        claim="The documented rate disagrees with the default.",
        evidence_file_path=LEAF_PATH,
        evidence_line_start=1,
        evidence_line_end=1,
        evidence_quote=RATE_DEFINITION,
        impact="Playback uses the wrong rate.",
        suggested_fix="Document 24000 Hz.",
        change_relation="introduced",
        certainty="supported",
    )
    responses: list[bytes | int] = [
        call_response("git_grep", {"query": "AUDIO_SAMPLE_RATE"}, None)
    ]
    if read_evidence:
        responses.append(
            call_response(
                "read_file_lines",
                {"file_path": LEAF_PATH, "line_start": 1, "line_end": 20},
                "tool_calls",
            )
        )
    else:
        pass
    responses.append(final_response([finding]))
    endpoint = ScriptedEndpoint(responses)
    monkeypatch.setattr("urllib.request.urlopen", endpoint.open)
    write_file(review_request.repository, LEAF_PATH, "working tree content\n")
    result = run_review(review_request, None, SystemClock())
    assert result.report.execution == "complete"
    assert result.report.model_configuration is not None
    assert result.report.model_configuration.max_tool_rounds == 2
    assert result.payload.model_configuration == result.report.model_configuration
    first, second = [json.loads(body) for body in endpoint.bodies[:2]]
    assert LEAF_PATH not in first["messages"][1]["content"]
    assert {tool["function"]["name"] for tool in first["tools"]} == {
        "git_grep",
        "read_file_lines",
        "list_files",
    }
    assert first["tool_choice"] == "auto"
    grep_result = second["messages"][-1]
    assert grep_result["role"] == "tool"
    assert grep_result["tool_call_id"] == second["messages"][-2]["tool_calls"][0]["id"]
    assert f"{LEAF_PATH}:1:{RATE_DEFINITION}" in grep_result["content"]
    if read_evidence:
        last = json.loads(endpoint.bodies[2])
        assert "tools" not in last and "tool_choice" not in last
        read_result = [
            message for message in last["messages"] if message["role"] == "tool"
        ][-1]
        assert read_result["tool_call_id"] == "read_file_lines-1"
        assert f"1: {RATE_DEFINITION}" in read_result["content"]
        assert result.report.findings == [finding]
        assert "Tool calls: 2" in render_markdown(result.report)
        assert result.report.tool_calls_used == 2
        assert result.report.tool_files_read == [LEAF_PATH]
        artifact = ReviewArtifact(
            report=result.report,
            pull_request_number=7,
            run_id="1",
            run_attempt=1,
            stale=False,
            publishable=True,
        )
        assert (
            f"/blob/{review_request.head_commit_sha}/{LEAF_PATH}#L1)"
            in render_advisory_comment("https://example.invalid/repo", artifact)
        )
    else:
        assert result.report.findings == []
        assert result.report.dropped_citation_count == 1
        assert result.report.tool_files_read == []


def test_new_excerpt_preserves_existing_citations(
    review_request: ReviewRequest,
) -> None:
    bundle = collect_review_bundle(review_request)
    lines = (review_request.repository / "large.py").read_text().splitlines()
    first_excerpt = FileExcerpt(
        line_start=1, line_end=30, content="\n".join(lines[:30])
    )
    bundle.files.append(
        CollectedFile(
            file_path="large.py",
            change="included",
            truncated=True,
            content=first_excerpt.content,
            excerpts=(first_excerpt,),
        )
    )
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    arguments = json.dumps(
        {"file_path": "large.py", "line_start": 200, "line_end": 220}
    )
    for attempt in range(2):
        assert "210: line 210 " in toolbox.execute("read_file_lines", arguments)
    assert citation_matches(bundle, "large.py", 210, 210, lines[209])
    assert citation_matches(bundle, "large.py", 30, 30, lines[29])
    assert not citation_matches(bundle, "large.py", 50, 50, lines[49])
    matches = [file for file in bundle.files if file.file_path == "large.py"]
    assert len(matches) == 1
    assert len(matches[0].excerpts) == 2
    assert toolbox.files_read() == ["large.py"]


@pytest.mark.parametrize(
    "file_path",
    ["../outside", "/etc/passwd", "--exec=id", "untracked.py", "binary.bin"],
)
def test_read_rejects_unsafe_untracked_and_binary_paths(
    review_request: ReviewRequest, file_path: str
) -> None:
    write_file(review_request.repository, "untracked.py", "working tree secret\n")
    bundle = collect_review_bundle(review_request)
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    result = toolbox.execute(
        "read_file_lines",
        json.dumps({"file_path": file_path, "line_start": 1, "line_end": 1}),
    )
    assert result and "1: " not in result
    assert not toolbox.files_read()
    assert not any(file.file_path == file_path for file in bundle.files)


def test_discovery_is_limited_to_tracked_paths_and_literal_queries(
    review_request: ReviewRequest,
) -> None:
    write_file(review_request.repository, "internal/untracked.py", "--sample-rate\n")
    bundle = collect_review_bundle(review_request)
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    assert toolbox.execute(
        "list_files", '{"glob_pattern":"internal/*.py"}'
    ).splitlines() == [LEAF_PATH]
    assert "flags.txt:1:--sample-rate" in toolbox.execute(
        "git_grep", '{"query":"--sample-rate"}'
    )
    assert (
        toolbox.execute("git_grep", '{"query":"unknown phrase"}') == "No matches found."
    )
    assert "flags.txt" not in toolbox.execute(
        "git_grep", '{"query":"--sample-rate","path_prefix":"internal"}'
    )
    assert "invalid arguments" in toolbox.execute("git_grep", "{").lower()
    assert "unknown" in toolbox.execute("unrecognized", "{}").lower()
    assert not toolbox.files_read()


def test_head_files_cannot_replace_review_input_or_trusted_base(
    review_request: ReviewRequest,
) -> None:
    write_file(review_request.repository, STYLE_GUIDE_PATH, "Base style policy.\n")
    review_request.base_commit_sha = commit_all(
        review_request.repository, "set style policy"
    )
    for path in (STYLE_GUIDE_PATH, "comments.md"):
        write_file(review_request.repository, path, "Untrusted head text.\n")
    review_request.head_commit_sha = commit_all(
        review_request.repository, "modify overlapping paths"
    )
    review_request.comment_text = "Review comment."
    bundle = collect_review_bundle(review_request)
    protected = [
        file for file in bundle.files if file.change in ("trusted_base", "review_input")
    ]
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    for path in (STYLE_GUIDE_PATH, "comments.md"):
        toolbox.execute(
            "read_file_lines",
            json.dumps({"file_path": path, "line_start": 1, "line_end": 1}),
        )
    assert protected == [
        file for file in bundle.files if file.change in ("trusted_base", "review_input")
    ]
    assert {file.content for file in protected} >= {
        "Base style policy.\n",
        "Review comment.",
    }


@pytest.mark.parametrize("limit", ["calls", "bytes"])
def test_exhausted_budget_stops_git(review_request: ReviewRequest, limit: str) -> None:
    bundle = collect_review_bundle(review_request)
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    if limit == "calls":
        for attempt in range(8):
            assert LEAF_PATH in toolbox.execute(
                "list_files", '{"glob_pattern":"internal/*"}'
            )
        assert toolbox.calls_used() == 8
    else:
        total_bytes = 0
        for attempt in range(4):
            output = toolbox.execute(
                "read_file_lines",
                '{"file_path":"large.py","line_start":1,"line_end":120}',
            )
            byte_count = len(output.encode())
            assert 0 < byte_count <= 12_000
            total_bytes += byte_count
        if total_bytes < 40_000:
            output = toolbox.execute(
                "git_grep", '{"query":"line ","path_prefix":"large.py"}'
            )
            total_bytes += len(output.encode())
        else:
            pass
        assert 39_999 <= total_bytes <= 40_000
    with patch(
        "subprocess.Popen", side_effect=AssertionError("Git ran after exhaustion")
    ):
        assert (
            "budget exhausted"
            in toolbox.execute("git_grep", '{"query":"default"}').lower()
        )


@pytest.mark.parametrize(
    ("status", "after_call", "fallback"),
    [
        (400, False, True),
        (422, False, True),
        (401, False, False),
        (500, False, False),
        (400, True, False),
    ],
)
def test_unsupported_tools_fallback_is_limited_to_initial_request(
    review_request: ReviewRequest,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    after_call: bool,
    fallback: bool,
) -> None:
    responses: list[bytes | int] = []
    if after_call:
        responses.append(call_response("list_files", {"glob_pattern": "*.py"}, "stop"))
    else:
        pass
    responses.extend([status, final_response([])])
    endpoint = ScriptedEndpoint(responses)
    monkeypatch.setattr("urllib.request.urlopen", endpoint.open)
    bundle = collect_review_bundle(review_request)
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    assert review_request.model_settings is not None
    reviewer = OpenAICompatibleReviewer(
        review_request.model_settings, "Review this.", toolbox=toolbox
    )
    if fallback:
        assert reviewer.review("Repository inputs") == "[]"
        assert len(endpoint.bodies) == 2
        retried = json.loads(endpoint.bodies[1])
        assert "tools" not in retried and "tool_choice" not in retried
        assert len(retried["messages"]) == 2
    else:
        with pytest.raises(ReviewerError):
            reviewer.review("Repository inputs")
        assert len(endpoint.bodies) == 1 + int(after_call)
    assert "tools" in json.loads(endpoint.bodies[0])


def test_cache_identity_tracks_available_evidence_only_when_tools_enabled(
    review_request: ReviewRequest,
) -> None:
    enabled = prepare_review(review_request)
    assert review_request.model_settings is not None
    review_request.model_settings.max_tool_rounds = 0
    disabled = prepare_review(review_request)
    assert disabled.prompt_version == sha256(read_prompt().encode()).hexdigest()
    assert disabled.input_hash != enabled.input_hash
    legacy_bundle = replace(
        disabled.bundle,
        repository_identity="legacy-repo",
        policy_version="test-policy",
        document_scopes=[],
        files=[],
        added_documentation_text="",
        style_guide_limitation="",
    )
    assert review_input_sha256(
        legacy_bundle, review_request.model_settings, "test-prompt"
    ) == ("32a9c0c1eda9c7a2714489447c80092b237964c9c7537300c8a385ccd355fe89")
    other_head = replace(disabled.bundle, head_commit_sha="f" * 40)
    assert (
        review_input_sha256(
            other_head, review_request.model_settings, disabled.prompt_version
        )
        == disabled.input_hash
    )
    review_request.model_settings.max_tool_rounds = 2
    assert (
        review_input_sha256(
            other_head, review_request.model_settings, enabled.prompt_version
        )
        != enabled.input_hash
    )


def test_batch_tool_calls_get_paired_results_and_final_turn_refuses_more_tools(
    review_request: ReviewRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = json.loads(
        call_response("list_files", {"glob_pattern": "internal/*"}, "tool_calls")
    )
    message = response["choices"][0]["message"]
    first_call = message["tool_calls"][0]
    message["tool_calls"] = [dict(first_call, id=str(index)) for index in range(9)]
    endpoint = ScriptedEndpoint(
        [
            json.dumps(response).encode(),
            call_response("git_grep", {"query": "default"}, "stop"),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", endpoint.open)
    result = run_review(review_request, None, SystemClock())
    assert result.report.execution == "incomplete"
    assert result.report.tool_calls_used == 8
    assert len(endpoint.bodies) == 2
    final_turn = json.loads(endpoint.bodies[1])
    assert "tools" not in final_turn and "tool_choice" not in final_turn
    tool_results = [
        message for message in final_turn["messages"] if message["role"] == "tool"
    ]
    assert [result["tool_call_id"] for result in tool_results] == [
        str(index) for index in range(9)
    ]
    assert all(result["content"] == LEAF_PATH for result in tool_results[:8])
    assert "budget exhausted" in tool_results[-1]["content"].lower()


def test_response_arriving_after_deadline_is_discarded(
    review_request: ReviewRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed_seconds = 0.0

    def slow_endpoint(request: Request, timeout: float) -> BytesIO:
        nonlocal elapsed_seconds
        elapsed_seconds += 241
        return BytesIO(final_response([]))

    monkeypatch.setattr("urllib.request.urlopen", slow_endpoint)
    monkeypatch.setattr(
        "scripts.docs_review.transport.time.monotonic", lambda: elapsed_seconds
    )
    bundle = collect_review_bundle(review_request)
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    assert review_request.model_settings is not None
    reviewer = OpenAICompatibleReviewer(
        review_request.model_settings, "Review this.", toolbox=toolbox
    )
    with pytest.raises(ReviewerError, match="duration"):
        reviewer.review("Repository inputs")


def test_grep_mixed_encodings_preserves_readable_paths(
    review_request: ReviewRequest,
) -> None:
    repository = review_request.repository
    (repository / "latin.py").write_bytes(b"# caf\xe9 SAMPLE_KEY\n")
    path = "docs/中文.md"
    write_file(repository, path, "SAMPLE_KEY = 1\n")
    commit_sha = commit_all(repository, "add mixed encoding search inputs")
    bundle = collect_review_bundle(
        replace(review_request, base_commit_sha=commit_sha, head_commit_sha=commit_sha)
    )
    toolbox = GitReadOnlyToolbox(repository, commit_sha, bundle)
    matches = toolbox.execute("git_grep", '{"query":"SAMPLE_KEY"}').splitlines()
    assert "latin.py:1:# caf� SAMPLE_KEY" in matches
    assert f"{path}:1:SAMPLE_KEY = 1" in matches
    toolbox.execute(
        "read_file_lines",
        json.dumps({"file_path": path, "line_start": 1, "line_end": 1}),
    )
    assert (
        toolbox.execute(
            "read_file_lines", '{"file_path":"latin.py","line_start":1,"line_end":1}'
        )
        == "Binary file cannot be read."
    )
    assert toolbox.files_read() == [path]
    assert citation_matches(bundle, path, 1, 1, "SAMPLE_KEY = 1")


@pytest.mark.parametrize("second_start", [25, 31])
def test_multiline_citation_across_reads(
    review_request: ReviewRequest, second_start: int
) -> None:
    bundle = collect_review_bundle(review_request)
    toolbox = GitReadOnlyToolbox(
        review_request.repository, bundle.head_commit_sha, bundle
    )
    for start, end in [(second_start, 50), (10, 30)]:
        toolbox.execute(
            "read_file_lines",
            json.dumps({"file_path": "large.py", "line_start": start, "line_end": end}),
        )
    lines = (review_request.repository / "large.py").read_text().splitlines()
    assert citation_matches(bundle, "large.py", 20, 35, "\n".join(lines[19:35]))


def test_byte_budget_forces_final_synthesis(
    review_request: ReviewRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.docs_review.tools.MAX_TOTAL_TOOL_OUTPUT_BYTES", len(LEAF_PATH.encode())
    )
    endpoint = ScriptedEndpoint(
        [
            call_response("list_files", {"glob_pattern": "internal/*"}, "tool_calls"),
            final_response([]),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", endpoint.open)
    result = run_review(review_request, None, SystemClock())
    assert result.report.execution == "complete"
    assert result.report.tool_calls_used == 1
    first, final = map(json.loads, endpoint.bodies)
    assert first["tools"]
    assert "tools" not in final and "tool_choice" not in final
