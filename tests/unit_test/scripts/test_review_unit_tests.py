# SPDX-License-Identifier: Apache-2.0
"""Replay contract-review inputs from the unit-test cleanup."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from scripts.docs_review import github_comment
from scripts.docs_review.github_pilot import detect_review_scopes
from scripts.docs_review.models import (
    BudgetConfig,
    Finding,
    Report,
    ReviewArtifact,
    ReviewRequest,
)
from scripts.docs_review.reporting import render_markdown
from scripts.docs_review.review import SystemClock, run_review

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "unit_test_review_2874.json"
)
CLI_TEST_IMPORTS = (
    "import yaml\n"
    "from sglang_omni.config.manager import ConfigManager\n"
    "from sglang_omni.config.sources import dump_user_config\n"
)
CLI_PARSER_SOURCE = (
    "\n\nfrom sglang_omni.config.manager import ConfigManager\n\n"
    "def resolve_sources(baseline, argv):\n"
    "    return ConfigManager(baseline).parse_extra_args(list(argv))\n"
)


class HistoricalCase(BaseModel):
    test_path: str
    name: str
    before: str
    after: str
    contract_path: str
    contract_source: str


class HistoricalFixtures(BaseModel):
    before_commit: str
    after_commit: str
    examples: list[HistoricalCase]


class OfflineReviewer:
    def __init__(self, response: list[Finding] | str) -> None:
        if isinstance(response, str):
            self.response = response
        else:
            self.response = json.dumps([finding.model_dump() for finding in response])
        self.inputs: list[str] = []

    def review(self, bundle_text: str) -> str:
        self.inputs.append(bundle_text)
        return self.response


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Review Test",
            "-c",
            "user.email=review@example.com",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_file(repository: Path, file_path: str, content: str) -> None:
    target = repository / file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def commit(repository: Path, message: str) -> str:
    git(repository, "add", ".")
    git(repository, "commit", "-qm", message)
    return git(repository, "rev-parse", "HEAD")


@pytest.fixture
def historical_cases() -> list[HistoricalCase]:
    return HistoricalFixtures.model_validate_json(FIXTURE_PATH.read_text()).examples


def historical_revisions(
    repository: Path, example: HistoricalCase
) -> tuple[str, str, str]:
    git(repository, "init", "-q")
    write_file(repository, example.contract_path, example.contract_source)
    if example.test_path.startswith("tests/unit_test/config/"):
        write_file(
            repository,
            "sglang_omni/config/manager.py",
            "class ConfigManager:\n    pass\n",
        )
        write_file(
            repository,
            "sglang_omni/config/sources.py",
            "def dump_user_config(config):\n    return config\n",
        )
        write_file(
            repository,
            "tests/unit_test/config/conftest.py",
            "import pytest\n\n"
            "@pytest.fixture\ndef plain_config_file():\n    return 'config.yaml'\n\n"
            "@pytest.fixture\ndef runner():\n    return None\n\n"
            "@pytest.fixture\ndef stage():\n    return 'audio'\n",
        )
    elif example.test_path.startswith("tests/unit_test/router/"):
        write_file(
            repository,
            "sglang_omni_router/python/worker.py",
            "def build_workers(configs):\n    return configs\n",
        )
    else:
        pass
    base_commit = commit(repository, "contract")
    write_file(repository, example.test_path, example.before)
    before_commit = commit(repository, "before")
    write_file(repository, example.test_path, example.after)
    after_commit = commit(repository, "after")
    return base_commit, before_commit, after_commit


def request_for(
    repository: Path, base_commit: str, head_commit: str, number: int
) -> ReviewRequest:
    return ReviewRequest(
        repository=repository,
        base_commit_sha=base_commit,
        head_commit_sha=head_commit,
        pull_request_number=number,
        pull_request_text="",
        review_scope="unit_tests",
        ledger_path=repository / ".git" / "review.sqlite",
        budget=BudgetConfig(
            max_cost_cents_per_call=1,
            max_cost_cents_per_pull_request=1,
            max_cost_cents_per_day=10,
            call_cost_cents=1,
        ),
    )


@pytest.mark.parametrize(
    (
        "example_index",
        "rule_id",
        "test_quote",
        "contract_quote",
        "public_contract",
        "regression",
        "strong_assertion",
    ),
    [
        (
            0,
            "UNIT_TEST_SELF_DERIVED_EXPECTATION",
            "assert yaml.safe_load(result.stdout) == dump_user_config(expected)",
            "Takes the same arguments as `sgl-omni serve`, including dotted overrides",
            "config resolve writes the requested stage override into YAML.",
            "If the CLI and ConfigManager both ignore the override, the original "
            "comparison can pass while the emitted YAML retains the default.",
            'assert printed["stages"][stage]["engine"]["mem_fraction_static"] == 0.35',
        ),
        (
            1,
            "UNIT_TEST_INTERNAL_ASSERTION",
            "restore_admin_disabled_state(workers, previous, outcome_safe=False)",
            "an earlier weight update did not complete",
            "An interrupted weight broadcast keeps target workers disabled and "
            "a retry returns HTTP 409 after restart until the update is resolved.",
            "If interruption cleanup forgets the pending update, a restarted router "
            "can accept a retry while worker weights are still uncertain.",
            "assert retry.status_code == 409",
        ),
    ],
)
def test_replays_weak_and_strengthened_contract_cases(
    tmp_path: Path,
    historical_cases: list[HistoricalCase],
    example_index: int,
    rule_id: str,
    test_quote: str,
    contract_quote: str,
    public_contract: str,
    regression: str,
    strong_assertion: str,
) -> None:
    example = historical_cases[example_index]
    base_commit, before_commit, after_commit = historical_revisions(tmp_path, example)
    test_line = next(
        index
        for index, line in enumerate(example.before.splitlines(), 1)
        if test_quote in line
    )
    contract_line = next(
        index
        for index, line in enumerate(example.contract_source.splitlines(), 1)
        if contract_quote in line
    )
    weak_finding = Finding(
        rule_id=rule_id,
        file_path=example.test_path,
        line_start=test_line,
        line_end=test_line,
        quote=test_quote,
        claim="This case can pass without detecting the contract regression.",
        evidence_file_path=example.contract_path,
        evidence_line_start=contract_line,
        evidence_line_end=contract_line,
        evidence_quote=contract_quote,
        impact=regression,
        suggested_fix=f"Exercise the entry point and check the outcome using {strong_assertion}.",
        change_relation="introduced",
        certainty="supported",
        public_contract=public_contract,
        regression=regression,
    )

    before_reviewer = OfflineReviewer([weak_finding])
    before = run_review(
        request_for(tmp_path, base_commit, before_commit, 1),
        before_reviewer,
        SystemClock(),
    )
    assert before.report.outcome == "findings", before.report.incomplete_reason
    assert before.report.findings == [weak_finding]
    assert public_contract in render_markdown(before.report)
    assert regression in render_markdown(before.report)
    assert contract_quote in before_reviewer.inputs[0]

    after_reviewer = OfflineReviewer([])
    after_request = request_for(tmp_path, before_commit, after_commit, 2)
    after_request.include_paths.append(example.contract_path)
    after = run_review(after_request, after_reviewer, SystemClock())
    assert after.report.outcome == "no_findings"
    assert [case.name for case in after.payload.unit_test_cases] == [example.name]
    assert test_quote in after.payload.unit_test_cases[0].previous_source
    assert strong_assertion in after_reviewer.inputs[0]
    assert contract_quote in after_reviewer.inputs[0]


@pytest.mark.parametrize("case_variant", ["weak", "strengthened", "different_parser"])
def test_cli_override_review_survives_an_empty_model_response(
    tmp_path: Path,
    historical_cases: list[HistoricalCase],
    case_variant: str,
) -> None:
    historical_case = historical_cases[0]
    parser_source = (
        CLI_PARSER_SOURCE.replace("parse_extra_args", "parse_args_for_preview")
        if case_variant == "different_parser"
        else CLI_PARSER_SOURCE
    )
    example = historical_case.model_copy(
        update={
            "before": CLI_TEST_IMPORTS
            + historical_case.before.replace(
                "engine.mem-fraction-static", "engine.mem_fraction_static"
            ).replace('"0.35"', '"0.42"'),
            "after": CLI_TEST_IMPORTS + historical_case.after.replace("0.35", "0.42"),
            "contract_source": historical_case.contract_source + parser_source,
        }
    )
    base_commit, before_commit, after_commit = historical_revisions(tmp_path, example)
    review = run_review(
        request_for(
            tmp_path,
            base_commit,
            after_commit if case_variant == "strengthened" else before_commit,
            1,
        ),
        OfflineReviewer('{"findings":[]}'),
        SystemClock(),
    )
    assert review.report.execution == "complete", review.report.incomplete_reason
    assert review.report.dropped_citation_count == 0
    assert review.report.outcome == "no_findings"
    assert not review.report.findings


@pytest.mark.parametrize(
    "failure",
    ["unchanged", "empty_regression", "bad_evidence", "unknown", "formatting"],
)
def test_incomplete_candidates_stay_out_of_the_review(
    tmp_path: Path, failure: str
) -> None:
    git(tmp_path, "init", "-q")
    test_path = "tests/unit_test/test_output.py"
    contract_path = "sglang_omni/api.py"
    write_file(tmp_path, contract_path, "def speak():\n    return b'RIFF'\n")
    write_file(
        tmp_path,
        test_path,
        "from sglang_omni.api import speak\n\n"
        "def test_existing():\n    assert speak().startswith(b'RIFF')\n",
    )
    base = commit(tmp_path, "existing contract test")
    original = (tmp_path / test_path).read_text()
    write_file(
        tmp_path,
        test_path,
        original + "\n\ndef test_new():\n    assert speak().endswith(b'WAV')\n",
    )
    head = commit(tmp_path, "weaker overlapping test")
    finding = Finding(
        rule_id="UNIT_TEST_INTERNAL_ASSERTION",
        file_path=test_path,
        line_start=8,
        line_end=8,
        quote="assert speak().endswith(b'WAV')",
        claim="The test checks a suffix instead of the WAV header.",
        evidence_file_path=contract_path,
        evidence_line_start=2,
        evidence_line_end=2,
        evidence_quote="return b'RIFF'",
        public_contract="speak returns a WAV payload.",
        regression="Nonempty raw PCM would be accepted as a WAV payload.",
        impact="A WAV caller cannot decode the result.",
        suggested_fix="Assert the WAV header.",
        change_relation="introduced",
        certainty="supported",
    )
    if failure == "unchanged":
        finding.line_start = finding.line_end = 4
        finding.quote = "assert speak().startswith(b'RIFF')"
    elif failure == "empty_regression":
        finding.regression = " "
    elif failure == "bad_evidence":
        finding.evidence_quote = "return b'OggS'"
    elif failure == "formatting":
        finding.line_start = finding.line_end = 9
        finding.quote = "`assert speak().endswith(b'WAV')`"
        finding.evidence_line_start = finding.evidence_line_end = 1
        finding.evidence_quote = "2:     return b'RIFF'"
    else:
        finding.rule_id = "UNIT_TEST_INSUFFICIENT_EVIDENCE"
        finding.certainty = "uncertain"
        finding.evidence_quote = ""
    result = run_review(
        request_for(tmp_path, base, head, 1), OfflineReviewer([finding]), SystemClock()
    )
    assert [case.name for case in result.report.unit_test_cases] == ["test_new"]
    if failure == "formatting":
        assert result.report.outcome == "findings", result.report.incomplete_reason
        verified = result.report.findings[0]
        assert (verified.line_start, verified.line_end) == (8, 8)
        assert verified.quote.strip() == "assert speak().endswith(b'WAV')"
        assert (verified.evidence_line_start, verified.evidence_line_end) == (2, 2)
        assert verified.evidence_quote.strip() == "return b'RIFF'"
        return
    else:
        assert result.report.findings == []
        assert result.report.outcome == "incomplete"
    if failure == "unknown":
        assert "insufficient contract evidence" in result.report.incomplete_reason
    else:
        assert result.report.dropped_citation_count == 1


def test_pure_test_file_rename_does_not_call_the_reviewer(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    original = "tests/unit_test/test_response.py"
    write_file(
        tmp_path, original, "def test_status():\n    assert response.status == 200\n"
    )
    base = commit(tmp_path, "original test")
    git(tmp_path, "mv", original, "tests/unit_test/test_status.py")
    head = commit(tmp_path, "rename only")
    reviewer = OfflineReviewer([])
    result = run_review(request_for(tmp_path, base, head, 1), reviewer, SystemClock())
    assert result.report.outcome == "no_findings"
    assert result.report.unit_test_cases == []
    assert result.report.test_hygiene_notes == []
    assert reviewer.inputs == []


def test_without_contract_source_the_review_stays_incomplete(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    write_file(tmp_path, "README.md", "Baseline.\n")
    base = commit(tmp_path, "base")
    write_file(
        tmp_path,
        "tests/unit_test/test_output.py",
        "def test_output(unresolved_fixture):\n    assert unresolved_fixture()\n",
    )
    head = commit(tmp_path, "test via a fixture")
    reviewer = OfflineReviewer([])
    result = run_review(request_for(tmp_path, base, head, 1), reviewer, SystemClock())
    assert result.report.outcome == "incomplete"
    assert "No contract source" in result.report.incomplete_reason
    assert reviewer.inputs == []


@pytest.mark.parametrize(
    "fixture_path",
    ["conftest.py", "tests/conftest.py", "tests/unit_test/audio/conftest.py"],
)
def test_changed_conftest_reviews_an_unchanged_case_with_change_evidence(
    tmp_path: Path,
    fixture_path: str,
) -> None:
    git(tmp_path, "init", "-q")
    test_path = "tests/unit_test/audio/test_output.py"
    contract_path = "sglang_omni/api.py"
    write_file(tmp_path, contract_path, "def speak():\n    return b'RIFF'\n")
    write_file(
        tmp_path,
        fixture_path,
        "import pytest\nfrom sglang_omni.api import speak\n\n"
        "@pytest.fixture\ndef audio():\n    return speak()\n",
    )
    write_file(
        tmp_path,
        test_path,
        "def test_output(audio):\n    assert len(audio) > 0\n",
    )
    base = commit(tmp_path, "base")
    write_file(
        tmp_path,
        fixture_path,
        "import pytest\nfrom sglang_omni.api import speak\n\n"
        "@pytest.fixture\ndef audio():\n    return speak()[:1]\n",
    )
    head = commit(tmp_path, "change fixture")
    finding = {
        "rule_id": "UNIT_TEST_INTERNAL_ASSERTION",
        "file_path": test_path,
        "line_start": 2,
        "line_end": 2,
        "quote": "assert len(audio) > 0",
        "claim": "The test accepts a truncated WAV header.",
        "change_evidence": {
            "file_path": fixture_path,
            "line_start": 6,
            "line_end": 6,
            "quote": "    return speak()[:1]",
        },
        "evidence_file_path": contract_path,
        "evidence_line_start": 2,
        "evidence_line_end": 2,
        "evidence_quote": "return b'RIFF'",
        "public_contract": "speak returns the WAV header.",
        "regression": "A truncated header reaches the caller.",
        "impact": "The caller cannot decode the result.",
        "suggested_fix": "Assert the complete WAV header.",
        "change_relation": "worsened",
        "certainty": "supported",
    }
    reviewer = OfflineReviewer(json.dumps({"findings": [finding]}))
    result = run_review(request_for(tmp_path, base, head, 1), reviewer, SystemClock())
    assert detect_review_scopes([fixture_path]) == ["unit_tests"]
    assert [case.name for case in result.report.unit_test_cases] == ["test_output"]
    assert result.report.execution == "complete", result.report.incomplete_reason
    assert result.report.findings[0].change_evidence.file_path == fixture_path
    assert f"Changed dependency: `{fixture_path}`" in render_markdown(result.report)


def test_reviewable_cases_run_when_another_case_has_no_contract(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    write_file(tmp_path, "sglang_omni/api.py", "def speak():\n    return b'RIFF'\n")
    base = commit(tmp_path, "base")
    for index in range(9):
        write_file(
            tmp_path,
            f"tests/unit_test/test_ready_{index}.py",
            "from sglang_omni.api import speak\n\n"
            f"def test_ready_{index}():\n    assert speak() == b'RIFF'\n",
        )
    write_file(
        tmp_path,
        "tests/unit_test/test_missing.py",
        "from sglang_omni.api import speak\n\n"
        "def test_missing(unresolved_fixture):\n    assert speak() == unresolved_fixture()\n",
    )
    head = commit(tmp_path, "ten changed cases")
    reviewer = OfflineReviewer('{"findings": []}')
    result = run_review(request_for(tmp_path, base, head, 1), reviewer, SystemClock())
    assert len(reviewer.inputs) == 1
    assert all(f"test_ready_{index}" in reviewer.inputs[0] for index in range(9))
    assert '"name": "test_missing"' not in reviewer.inputs[0]
    assert "File tests/unit_test/test_missing.py" not in reviewer.inputs[0]
    assert result.report.execution == "incomplete"
    assert result.report.outcome == "incomplete"
    assert "test_missing" in result.report.incomplete_reason


def test_changed_imported_helper_selects_an_unchanged_test(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    helper_path = "tests/unit_test/fixtures/speech.py"
    test_path = "tests/unit_test/test_speech.py"
    write_file(tmp_path, "sglang_omni/api.py", "def speak():\n    return b'RIFF'\n")
    write_file(
        tmp_path,
        helper_path,
        "from sglang_omni.api import speak\n\n" "def audio():\n    return speak()\n",
    )
    write_file(
        tmp_path,
        test_path,
        "from tests.unit_test.fixtures.speech import audio as get_audio\n\n"
        "def test_audio():\n    assert get_audio() == b'RIFF'\n",
    )
    base = commit(tmp_path, "base")
    write_file(
        tmp_path,
        helper_path,
        "from sglang_omni.api import speak\n\n"
        "def audio():\n    return speak()[:1]\n",
    )
    head = commit(tmp_path, "change helper")
    reviewer = OfflineReviewer('{"findings": []}')
    result = run_review(request_for(tmp_path, base, head, 1), reviewer, SystemClock())
    assert result.report.execution == "complete", result.report.incomplete_reason
    assert [case.name for case in result.report.unit_test_cases] == ["test_audio"]
    assert helper_path in reviewer.inputs[0]


def test_valid_finding_survives_another_unreviewable_case(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    write_file(tmp_path, "sglang_omni/api.py", "def speak():\n    return b'RIFF'\n")
    base = commit(tmp_path, "base")
    test_path = "tests/unit_test/test_ready.py"
    write_file(
        tmp_path,
        test_path,
        "from sglang_omni.api import speak\n\n"
        "def test_audio():\n    assert len(speak()) > 0\n",
    )
    write_file(
        tmp_path,
        "tests/unit_test/test_missing.py",
        "def test_missing(unresolved_fixture):\n    assert unresolved_fixture()\n",
    )
    head = commit(tmp_path, "mixed evidence")
    finding = Finding(
        rule_id="UNIT_TEST_INTERNAL_ASSERTION",
        file_path=test_path,
        line_start=4,
        line_end=4,
        quote="assert len(speak()) > 0",
        claim="A nonempty response can have a corrupt WAV header.",
        evidence_file_path="sglang_omni/api.py",
        evidence_line_start=2,
        evidence_line_end=2,
        evidence_quote="return b'RIFF'",
        public_contract="speak returns a WAV header.",
        regression="The caller receives a corrupt header.",
        impact="The caller cannot decode the audio.",
        suggested_fix="Assert the RIFF header.",
        change_relation="introduced",
        certainty="supported",
    )
    result = run_review(
        request_for(tmp_path, base, head, 1), OfflineReviewer([finding]), SystemClock()
    )
    assert result.report.execution == "incomplete"
    assert result.report.findings == [finding]
    assert "test_missing" in result.report.incomplete_reason


def test_unresolved_changed_shared_fixture_produces_incomplete_report(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init", "-q")
    fixture_path = "tests/unit_test/conftest.py"
    write_file(tmp_path, fixture_path, "pytest_plugins = ['external_plugin']\n")
    base = commit(tmp_path, "base")
    write_file(tmp_path, fixture_path, "pytest_plugins = ['new_external_plugin']\n")
    head = commit(tmp_path, "change plugin")
    result = run_review(request_for(tmp_path, base, head, 1), None, SystemClock())
    assert result.report.execution == "incomplete"
    assert fixture_path in result.report.incomplete_reason


def test_dynamic_fixture_lookup_is_reported_as_unresolved(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    fixture_path = "tests/unit_test/conftest.py"
    write_file(
        tmp_path,
        fixture_path,
        "import pytest\n@pytest.fixture\ndef audio():\n    return 'old'\n",
    )
    write_file(
        tmp_path,
        "tests/unit_test/test_audio.py",
        "def test_audio(request, selected):\n    assert request.getfixturevalue(selected)\n",
    )
    base = commit(tmp_path, "base")
    write_file(
        tmp_path,
        fixture_path,
        "import pytest\n@pytest.fixture\ndef audio():\n    return 'new'\n",
    )
    head = commit(tmp_path, "change fixture")
    result = run_review(request_for(tmp_path, base, head, 1), None, SystemClock())
    assert result.report.execution == "incomplete"
    assert "dynamic" in result.report.incomplete_reason.lower()


@pytest.mark.parametrize("edit_test", [False, True])
def test_changed_fixture_selects_its_test_even_when_the_body_is_unchanged(
    tmp_path: Path, edit_test: bool
) -> None:
    git(tmp_path, "init", "-q")
    file_path = "tests/unit_test/test_output.py"
    original = (
        "import pytest\n\n"
        "@pytest.fixture\ndef default_voice():\n    return 'first'\n\n"
        "def test_output(default_voice):\n    assert default_voice\n"
    )
    contract_path = "sglang_omni/api.py"
    write_file(tmp_path, file_path, original)
    write_file(tmp_path, contract_path, "def speak():\n    return b'RIFF'\n")
    base = commit(tmp_path, "base")
    edited = original.replace("'first'", "'second'")
    if edit_test:
        edited = edited.replace(
            "assert default_voice", "assert default_voice == 'second'"
        )
    else:
        pass
    write_file(tmp_path, file_path, edited)
    head = commit(tmp_path, "fixture edit")
    reviewer = OfflineReviewer([])
    request = request_for(tmp_path, base, head, 1)
    request.include_paths.append(contract_path)
    result = run_review(request, reviewer, SystemClock())
    assert result.report.outcome == "no_findings"
    assert [case.name for case in result.report.unit_test_cases] == ["test_output"]
    assert result.report.unit_test_cases[0].reviewed
    assert len(reviewer.inputs) == 1


@pytest.mark.parametrize(
    "citation_shape",
    ["flat", "nested", "line", "pinned_repeat", "ambiguous", "out_of_bounds"],
)
def test_resolves_a_model_citation_without_lines_or_quote(
    tmp_path: Path,
    historical_cases: list[HistoricalCase],
    citation_shape: str,
) -> None:
    case = historical_cases[0]
    contract_quote = (
        "Takes the same arguments as `sgl-omni serve`, including dotted overrides"
    )
    if citation_shape in {"ambiguous", "pinned_repeat"}:
        case = case.model_copy(
            update={"contract_source": case.contract_source + f"\n# {contract_quote}\n"}
        )
    else:
        pass
    base_commit, before_commit, _ = historical_revisions(tmp_path, case)
    test_quote = "expected = manager.merge_config(manager.parse_extra_args(argv))"
    payload: dict[str, object] = {
        "rule_id": "UNIT_TEST_SELF_DERIVED_EXPECTATION",
        "file_path": case.test_path,
        "line_start": 169,
        "line_end": 171,
        "quote": test_quote,
        "claim": "The expected YAML is computed by the same resolver.",
        "public_contract": "The CLI emits the requested stage override.",
        "regression": "A dropped override leaves the default in stdout.",
        "impact": "The preview gives the caller an incorrect configuration.",
        "suggested_fix": "Assert the emitted stage memory fraction equals 0.35.",
        "change_relation": "introduced",
        "certainty": "supported",
    }
    if citation_shape == "nested":
        payload["evidence"] = {"file_path": case.contract_path, "quote": contract_quote}
    elif citation_shape in {"line", "pinned_repeat"}:
        payload["evidence"] = {
            "file_path": case.contract_path,
            "line_start": next(
                number
                for number, line in enumerate(case.contract_source.splitlines(), 1)
                if contract_quote in line
            ),
        }
        if citation_shape == "pinned_repeat":
            payload["evidence_quote"] = contract_quote
        else:
            pass
    elif citation_shape == "out_of_bounds":
        payload["evidence"] = {
            "file_path": case.contract_path,
            "line_start": len(case.contract_source.splitlines()) + 1,
        }
    else:
        payload["evidence_file_path"] = case.contract_path
        payload["evidence_quote"] = contract_quote
    response = json.dumps({"findings": [payload]})

    result = run_review(
        request_for(tmp_path, base_commit, before_commit, 1),
        OfflineReviewer(response),
        SystemClock(),
    )

    if citation_shape in {"ambiguous", "out_of_bounds"}:
        assert result.report.execution == "incomplete"
        assert result.report.findings == []
    else:
        assert result.report.outcome == "findings", result.report.incomplete_reason
        finding = result.report.findings[0]
        assert finding.evidence_file_path == case.contract_path
        assert finding.evidence_quote.strip() == contract_quote
        assert finding.evidence_line_start == finding.evidence_line_end
        assert (
            contract_quote
            in case.contract_source.splitlines()[finding.evidence_line_start - 1]
        )
        assert test_quote in case.before.splitlines()[finding.line_start - 1]
        markdown = render_markdown(result.report)
        assert "A dropped override leaves the default in stdout." in markdown
        assert "Assert the emitted stage memory fraction equals 0.35." in markdown


def test_new_internal_mocks_are_reported_including_helpers(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    file_path = "tests/unit_test/test_output.py"
    write_file(
        tmp_path,
        file_path,
        "from sglang_omni import api\n\n"
        "def install_fakes(monkeypatch):\n"
        "    monkeypatch.setattr(api, 'speak', lambda: None)\n\n"
        "def test_output():\n"
        "    assert True\n",
    )
    base = commit(tmp_path, "base")
    write_file(
        tmp_path,
        file_path,
        "import subprocess\n"
        "from unittest.mock import patch\n"
        "from sglang_omni import api\n\n"
        "def install_fakes(monkeypatch):\n"
        "    monkeypatch.setattr(api, 'speak', lambda: None)\n"
        "    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)\n"
        "    patch.object(api, 'listen', lambda: None)\n\n"
        "@patch('sglang_omni.api.export')\n"
        "def test_output(monkeypatch):\n"
        "    monkeypatch.setattr('sglang_omni.api.transcribe', lambda: None)\n"
        "    assert True\n",
    )
    head = commit(tmp_path, "add patches")
    result = run_review(
        request_for(tmp_path, base, head, 1), OfflineReviewer([]), SystemClock()
    )
    notes = "\n".join(result.report.test_hygiene_notes)
    assert "sglang_omni.api.listen" in notes
    assert "sglang_omni.api.export" in notes
    assert "sglang_omni.api.transcribe" in notes
    assert "sglang_omni.api.speak" not in notes
    assert "subprocess.run" not in notes
    assert "UNIT_TEST_INTERNAL_MOCK" in render_markdown(result.report)


@pytest.mark.parametrize(
    ("test_line_count", "product_line_count", "justified", "reported"),
    [
        (100, 1, False, True),
        (100, 1, True, False),
        (99, 0, False, False),
        (100, 50, False, False),
    ],
)
def test_unit_test_volume_is_reported_only_past_the_threshold(
    tmp_path: Path,
    test_line_count: int,
    product_line_count: int,
    justified: bool,
    reported: bool,
) -> None:
    git(tmp_path, "init", "-q")
    write_file(tmp_path, "README.md", "Base\n")
    base = commit(tmp_path, "base")
    if product_line_count:
        write_file(
            tmp_path,
            "sglang_omni/api.py",
            "".join(
                f"value_{index} = {index}\n" for index in range(product_line_count)
            ),
        )
    else:
        pass
    write_file(
        tmp_path,
        "tests/unit_test/test_rows.py",
        "".join(f"value_{index} = {index}\n" for index in range(test_line_count)),
    )
    head = commit(tmp_path, "grow tests")
    request = request_for(tmp_path, base, head, 1)
    if justified:
        request.pull_request_text = (
            "Test volume: the rows are one generated contract.\n"
        )
    else:
        pass
    result = run_review(request, None, SystemClock())
    notes = "\n".join(result.report.test_hygiene_notes)
    assert ("UNIT_TEST_VOLUME" in notes) is reported
    assert ("UNIT_TEST_VOLUME" in render_markdown(result.report)) is reported


def test_hygiene_note_is_posted_when_the_model_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: list[str] = []

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        assert endpoint.startswith("repos/owner/repo/issues/")
        if method == "GET":
            return "[]"
        else:
            assert body is not None
            posted.append(json.loads(body)["body"])
            return json.dumps({"id": 1})

    monkeypatch.setattr(github_comment, "github_api", api)
    report = Report.model_construct(
        execution="complete",
        head_commit_sha="a" * 40,
        findings=[],
        test_hygiene_notes=[
            "UNIT_TEST_VOLUME: adds 100 test lines and 1 product line."
        ],
        review_scope="unit_tests",
    )
    artifact = ReviewArtifact.model_construct(
        pull_request_number=1,
        run_id="123",
        run_attempt=2,
        stale=False,
        publishable=True,
        review_scope="unit_tests",
        report=report,
    )
    github_comment.publish_reports(
        "owner/repo", "https://github.com/owner/repo", "123", 2, [artifact]
    )
    assert len(posted) == 1
    assert "UNIT_TEST_VOLUME" in posted[0]
    assert github_comment.ADVISORY_REVIEW_MARKER in posted[0]
