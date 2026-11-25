# SPDX-License-Identifier: Apache-2.0
"""The PR description check reports only the template sections that still need the author's text."""

import importlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / ".github/scripts/check_pr_description.py"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/pr-description.yml"
TEMPLATE_MOTIVATION_COMMENT = (
    "<!-- Explain the purpose of this PR and the goals it aims to achieve. -->"
)
TEMPLATE_MODIFICATIONS_COMMENT = "<!-- Describe the changes made in this PR. -->"

# Verbatim bodies of the pull requests this check was accepted on.
PR_4015_BODY = """Use pull_request_target with append-only addLabels so manual and run-* labels stay.

<!-- Thank you for your contribution! We appreciate it. The following guidelines will help improve your pull request and facilitate feedback. If anything is unclear, don't hesitate to submit your pull request and ask the maintainers for assistance. -->

## Motivation

<!-- Explain the purpose of this PR and the goals it aims to achieve. -->

## Modifications

<!-- Describe the changes made in this PR. -->

## Related Issues

<!-- Link to any related issues here. e.g. "Fixes #123" or "Closes #456" -->

## Accuracy Test

<!-- If this PR affects model-side code (e.g., kernels, model architecture), please provide accuracy test results. Ref: https://docs.sglang.io/docs/developer_guide/evaluating_new_models -->

## Benchmark & Profiling

<!-- If this PR is expected to impact performance, please provide benchmark and profiling results. Ref: https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling -->

## Checklist

- [ ] Format your code according with pre-commit.
- [ ] Add unit tests.
- [ ] Update documentation / docstrings / example tutorials as needed.
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

Lint runs on GitHub-hosted runners for every PR, and Test Layout, Docs Check, Playground Test and Rust Router for every PR that changes more than the MLX path (`sglang_omni_mlx/`, `Voxt/` and their workflows); the NPU image build runs for PRs that change its Dockerfile, installer, pyproject files or workflows, and `lint` must pass before merging. The rest of CI runs on self-hosted runners, starts only after lint passes, and requires a maintainer to add CI labels. A PR that changes only the MLX path does not start Omni CI, but Mac CI fails without a label: `run-mac-ci` runs its Mac tests, and `run-lint-only` runs none. Once labeled, every subsequent push re-triggers CI as long as the labels remain. Draft PRs are skipped even if labeled.

- `run-lint-only` runs lint only. The self-hosted gates succeed and run no tests. Use `/tag-and-rerun-ci lint` to add it. It does not turn off a stage label that is also present.
- A stage label runs one stage, and there is no label that runs every stage: `run-asr-ci`, `run-tts-ci`, `run-mps-ci`, `run-omni-ci`, `run-native-duplex`, `run-turn-based-duplex` and `run-cascaded-duplex` on H100, `run-intel-cpu-ci` (Intel Xeon), `run-intel-xpu-ci` (Intel GPU) and `run-mac-ci` (Apple Silicon). The hardware stages still skip a PR that touches none of their paths.
- Any stage label except `run-mac-ci` also runs the H100 unit tests.
- `run-unit-test-ci-only` runs just the H100 unit tests after shared setup when no other stage labels are present. Use `/tag-and-rerun-ci unit-test` to add it and re-run completed CI.
- A model label picks the model of every selected stage that accepts it. It does not select a stage. Without one, a selected stage draws its model at random. Keep at most one model label per stage.
  - ASR: `run-fun-asr-nano-2512`, `run-qwen3-asr-1.7b`, `run-whisper-large-v3`
  - TTS and MPS: `run-higgs-tts-3-4b`, `run-moss-tts-local-transformer-v1.5`. `run-mps-ci` is the only MPS stage label. With both stage labels, one model label selects that model for both. With only `run-tts-ci`, MPS does not run.
  - TTS: `run-qwen3-tts-12hz-1.7b-base`, `run-qwen3-tts-12hz-1.7b-customvoice` (label only, never drawn), `run-fun-cosyvoice3-0.5b-2512`
  - Omni: `run-qwen3-omni-30b-a3b-instruct`, `run-minicpm-o-4.5`

Use `/tag-and-rerun-ci` with short targets, for example `/tag-and-rerun-ci tts asr` or `/tag-and-rerun-ci qwen3-tts qwen3-asr`. Bare `/tag-and-rerun-ci` and full GitHub label arguments are not accepted. Existing stage selections stay enabled. Use `/rerun-failed-ci` to retry failed jobs.

The rules live in `.github/scripts/resolve_ci_labels.py`, and [docs/developer_reference/ci_labels.md](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/ci_labels.md) describes what every stage covers."""
PR_4015_BODY_WITH_MOTIVATION_FILLED = PR_4015_BODY.replace(
    TEMPLATE_MOTIVATION_COMMENT,
    "The documentation label should only mark pull requests that change documentation.",
)
PR_4489_BODY = """## Motivation

The `documentation` label should identify PRs that only change documentation. A model PR that also updates its Cookbook currently receives the same label as a docs-only PR.

## Modifications

- Add `documentation` only when every changed path is under `docs/` or is the root `README.md` or `CONTRIBUTING.md`. For renames, check both paths.
- Remove the title-prefix rule for `documentation` so titles such as `[Docs]` cannot label a mixed PR.
- Preserve the existing CI label behavior, including `run-lint-only`.

## Related Issues

Follow-up to #3993.

## Validation

- `pytest -q tests/unit_test/ci/test_pr_labeler.py` (21 passed).
- `pre-commit run --all-files`.
- `git diff --check`.

## Accuracy Test

Not applicable to the PR labeler.

## Benchmark & Profiling

Not applicable to the PR labeler.

## Checklist

- [x] Format the changes with pre-commit.
- [x] Cover mixed changes, docs-only changes, title prefixes, and cross-directory renames in the labeler tests."""


@pytest.fixture(scope="module")
def checker() -> Iterator[ModuleType]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(REPO_ROOT / ".github/scripts"))

        yield importlib.import_module("check_pr_description")

        sys.modules.pop("check_pr_description")


@pytest.mark.parametrize(
    ("body", "expected_problems"),
    [
        pytest.param(
            "## Motivation\n\n" + TEMPLATE_MOTIVATION_COMMENT + "\n\n"
            "## Modifications\n\n" + TEMPLATE_MODIFICATIONS_COMMENT + "\n",
            [("Motivation", "empty"), ("Modifications", "empty")],
            id="template_comments_only",
        ),
        pytest.param(
            "## Motivation\n\n"
            "Explain the purpose of this PR and the goals it aims to achieve.\n\n"
            "## Modifications\n\nDescribe the changes made in this PR.\n",
            [("Motivation", "empty"), ("Modifications", "empty")],
            id="template_prompt_text_kept_outside_the_comment",
        ),
        pytest.param(
            "## Motivation\n\n"
            + TEMPLATE_MOTIVATION_COMMENT
            + "\nThis check reports the section, not the quality of its text.\n\n"
            "## Modifications\n\n" + TEMPLATE_MODIFICATIONS_COMMENT + "\n"
            "- Add the parser.\n",
            [],
            id="one_sentence_or_list_item_beside_the_template_comment_passes",
        ),
        pytest.param(
            "Fixes the check that reports unfilled template sections.\n\n"
            "## Motivation\n\n" + TEMPLATE_MOTIVATION_COMMENT + "\n\n"
            "## Modifications\n\n" + TEMPLATE_MODIFICATIONS_COMMENT + "\n\n"
            "## Related Issues\n\nFixes #4015\n",
            [("Motivation", "empty"), ("Modifications", "empty")],
            id="summary_above_the_headings_and_a_filled_related_issues_do_not_count",
        ),
        pytest.param(
            "## Motivation\n\nThe label should only mark documentation changes.\n\n"
            "## Modifications\n\n## Related Issues\n\nFixes #3993\n",
            [("Modifications", "empty")],
            id="the_section_ends_at_the_next_heading_of_the_same_level",
        ),
        pytest.param(
            "```markdown\n## Motivation\n\nExplain why this change is needed.\n```\n\n"
            "## Modifications\n\n- Add the check.\n",
            [("Motivation", "missing")],
            id="a_heading_in_a_code_sample_is_not_a_section",
        ),
        pytest.param(
            "~~~~markdown\n~~~\n## Motivation\nExample only.\n~~~~\n"
            "## Modifications\nCheck the body.\n",
            [("Motivation", "missing")],
            id="a_shorter_fence_does_not_end_an_example",
        ),
        pytest.param(
            "~~~html\n<!--\n~~~\n"
            "  ## Motivation\nExplain the problem.\n"
            "## Modifications\nCheck the body.\n",
            [],
            id="a_literal_comment_in_code_cannot_hide_real_sections",
        ),
        pytest.param(
            "## Modifications\n\n- Add the check.\n\n"
            "> ## Motivation\n>\n> Explain why this change is needed.\n",
            [("Motivation", "missing")],
            id="a_heading_in_a_blockquote_is_not_a_section",
        ),
        pytest.param(
            "<!--\n## Motivation\n\nExplain why this change is needed.\n-->\n\n"
            "## Modifications\n\n- Add the check.\n",
            [("Motivation", "missing")],
            id="a_heading_in_an_html_comment_is_not_a_section",
        ),
        pytest.param(
            "##  motivation\n\nThis check reports the section.\n\n"
            "## Modifications  \n\n- Add the check.\n",
            [],
            id="heading_case_and_surrounding_whitespace_are_tolerated",
        ),
        pytest.param(
            "### Motivation\n\nThis check reports the section.\n\n"
            "## Modifications\n\n- Add the check.\n",
            [("Motivation", "missing")],
            id="a_third_level_heading_is_not_the_template_section",
        ),
        pytest.param(
            "## Motivation\n\n### Background\n\n"
            "The labeler marked mixed pull requests as documentation only.\n\n"
            "## Modifications\n\n- Add the check.\n",
            [],
            id="a_deeper_heading_stays_inside_the_section",
        ),
        pytest.param(
            "## Motivation\n\n这个改动让检查在作者没有填写动机时给出明确提示。\n\n"
            "## Modifications\n\n- 新增解析脚本。\n- 新增 GitHub Actions 检查。\n",
            [],
            id="chinese_text_passes",
        ),
        pytest.param(
            "## Motivation\n\n<!-- keep this note for later -->\n\n"
            "## Modifications\n\n   \n",
            [("Motivation", "empty"), ("Modifications", "empty")],
            id="whitespace_and_arbitrary_comments_are_empty",
        ),
        pytest.param(
            "Use pull_request_target with append-only addLabels.\n",
            [("Motivation", "missing"), ("Modifications", "missing")],
            id="a_description_without_template_headings",
        ),
        pytest.param(
            PR_4015_BODY,
            [("Motivation", "empty"), ("Modifications", "empty")],
            id="pull_request_4015",
        ),
        pytest.param(
            PR_4015_BODY_WITH_MOTIVATION_FILLED,
            [("Modifications", "empty")],
            id="pull_request_4015_after_filling_motivation_only",
        ),
        pytest.param(
            PR_4489_BODY,
            [],
            id="pull_request_4489",
        ),
    ],
)
def test_the_check_names_the_sections_that_still_need_the_author_text(
    checker: ModuleType,
    body: str,
    expected_problems: list[tuple[str, str]],
) -> None:
    problems = checker.find_problems(body)

    actual_problems = [(problem.section, problem.reason) for problem in problems]
    assert actual_problems == expected_problems


@pytest.mark.parametrize(
    ("body", "expected_returncode", "expected_summary_fragments"),
    [
        pytest.param(
            PR_4015_BODY,
            1,
            ["`## Motivation` is empty", "`## Modifications` is empty"],
            id="pull_request_4015",
        ),
        pytest.param(
            PR_4489_BODY,
            0,
            [],
            id="pull_request_4489",
        ),
    ],
)
def test_the_command_reports_every_problem_in_the_step_summary(
    tmp_path: Path,
    body: str,
    expected_returncode: int,
    expected_summary_fragments: list[str],
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"body": body}}))
    summary_path = tmp_path / "summary.md"

    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH)],
        capture_output=True,
        text=True,
        env={
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_STEP_SUMMARY": str(summary_path),
        },
    )

    assert result.returncode == expected_returncode
    summary = summary_path.read_text() if summary_path.exists() else ""
    for fragment in expected_summary_fragments:
        assert fragment in summary


def test_the_workflow_reruns_whenever_the_description_or_the_revision_changes() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    trigger = workflow.get("on", workflow.get(True))
    job = workflow["jobs"]["description"]

    assert set(trigger["pull_request"]["types"]) == {
        "opened",
        "reopened",
        "ready_for_review",
        "edited",
        "synchronize",
        "converted_to_draft",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "draft" in job["if"]
    assert any(CHECKER_PATH.name in step.get("run", "") for step in job["steps"])
