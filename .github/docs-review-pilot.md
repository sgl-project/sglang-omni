# Advisory pull request review bot

**Advisory pull request review** runs from the default branch on
`pull_request_target` when `ADVISORY_REVIEW_ENABLED` is `true`. For each matching
pull request it may review documentation changes and changed Python unit tests,
call the configured model, and update one sticky comment with the findings. The
workflow never checks out or runs code from the pull request; it only fetches
fixed commits as data.

Maintainers can still dispatch the workflow manually for offline collection or a
second authorized call on one pull request.

## Configuration

Set repository variable `ADVISORY_REVIEW_ENABLED=true` to turn automatic reviews on.

Automatic and model-backed manual runs need repository secret `DOCS_REVIEW_API_KEY`
and these repository variables:

- `DOCS_REVIEW_MODEL` is JSON with `model_name`, `endpoint_url`,
  `max_output_tokens`, and `api_key_environment_variable` (use
  `DOCS_REVIEW_API_KEY`). Set `reasoning_effort` when the model supports it.
- `UNIT_TEST_REVIEW_MODEL` optionally overrides the docs settings for unit-test
  review. It uses the same credential and budget.
- `DOCS_REVIEW_BUDGET` holds `call_cost_cents`, `max_cost_cents_per_call`,
  `max_cost_cents_per_pull_request`, and `max_cost_cents_per_day`.

Create a `docs-review-ledger` branch with root-level `docs-review-ledger.json`
containing `{"version":1,"reservations":[]}` before the first live review.

## Automatic scope

A push-triggered run selects scopes from changed paths between merge base and head:

- `docs/**`, root `README.md`, or `CONTRIBUTING.md` enable documentation review.
- Added or changed files under `tests/unit_test/` whose names match `test_*.py`
  or `*_test.py` enable unit-test contract review.

When the review input hash is unchanged since the last complete run, the workflow
skips another model call.

## Manual dispatch

```sh
gh workflow run pr-advisory-review.yaml -f pull_request_number=123 -f offline=true
```

Add `-f review_scope=unit_tests` and optional
`-f unit_test_context_paths=sglang_omni/cli/config.py` for unit-test-only runs.
Set `offline=false` for one live call. Use `follow_up=true` for a second call when
the input hash changed.

Reports upload as `docs-review-RUN_ID-ATTEMPT` artifacts. Automatic runs post
complete findings directly on the pull request and update the same comment on
later pushes.

Unit-test runs also add deterministic hygiene notes without a model call. A newly added patch of `sglang_omni`, `sglang_omni_router`, `sglang_omni_mlx`, `examples`, or `scripts` is reported as `UNIT_TEST_INTERNAL_MOCK`, including a patch inside a helper; patches of network, subprocess, time, and filesystem modules are not. When the pull request adds at least 100 Python lines under `tests/unit_test/` and more than twice the Python product lines, the comment includes `UNIT_TEST_VOLUME` unless the description has a line starting with `Test volume:`.

Unit-test review reads fixtures and helpers from the frozen Git revisions and selects tests with statically identified dependencies on changed code. Each case records its evidence limitations and whether a model reviewed it. The report remains incomplete if any selected case or dependency could not be reviewed. Dynamic fixture lookup, plugins, and unsupported class or module dependencies may require additional evidence.

The automatic comment records each scope separately for the reviewed base and head. A completed scope replaces its earlier findings, including when no findings remain. An incomplete scope is marked incomplete, and stale revisions cannot replace a current report. Each model call reserves its existing per-call, per-pull-request, and daily budget before execution.
