# Advisory pull request review bot

**Advisory pull request review** runs from the default branch on
`pull_request_target` when `ADVISORY_REVIEW_ENABLED` is `true`. It reviews
documentation changes, calls the configured model, and posts findings in one
sticky PR comment. The workflow never checks out or runs code from the pull
request; it only fetches fixed commits as data.

Maintainers can still run the same workflow manually for offline collection or a
second authorized call on one pull request.

## Configuration

Set repository variable `ADVISORY_REVIEW_ENABLED=true` to turn automatic reviews on.

Automatic and model-backed manual runs need repository secret `DOCS_REVIEW_API_KEY`
and these repository variables:

- `DOCS_REVIEW_MODEL` is JSON with `model_name`, `endpoint_url`,
  `max_output_tokens`, and `api_key_environment_variable` (use
  `DOCS_REVIEW_API_KEY`). Set `reasoning_effort` when the model supports it.
- `DOCS_REVIEW_BUDGET` holds `call_cost_cents`, `max_cost_cents_per_call`,
  `max_cost_cents_per_pull_request`, and `max_cost_cents_per_day`. The workflow
  reserves the estimated cost before each call.

Create a `docs-review-ledger` branch with root-level `docs-review-ledger.json`
containing `{"version":1,"reservations":[]}` before the first live review.

## Manual dispatch

```sh
gh workflow run pr-advisory-review.yaml -f pull_request_number=123 -f offline=true
```

Set `offline=false` for one live call. Use `follow_up=true` for a second call on
the same pull request when the input hash changed. The pilot caps model inputs at
100,000 bytes.

Reports upload as `docs-review-RUN_ID-ATTEMPT` artifacts with `report.md` and
`report.json`. Automatic runs post complete findings directly on the pull request
and update the same comment on later pushes.