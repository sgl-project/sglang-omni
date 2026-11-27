# Advisory pull request review bot

**Advisory pull request review** runs from the default branch on `pull_request_target` when `ADVISORY_REVIEW_ENABLED` is `true`. During the pilot, automatic runs trigger only for non-draft, non-bot pull requests opened by `OWNER`, `MEMBER`, or `COLLABORATOR` authors, or for external pull requests labeled `run-docs-review`. It reviews documentation changes, calls the configured model, and posts findings in one sticky PR comment owned by `github-actions[bot]`. The workflow never checks out or runs code from the pull request; it only fetches pinned commit SHAs as data.

Runs for the same pull request queue (`cancel-in-progress: false`) so an in-flight review finishes and updates its ledger reservation instead of leaking quota. When an earlier sticky comment exists and a later complete run returns no findings, the bot updates that comment to report that findings are cleared while staying silent on first-time clean pull requests.

Maintainers can still run the same workflow manually for offline collection or a second authorized call on one pull request.

## Configuration

Set repository variable `ADVISORY_REVIEW_ENABLED=true` to turn automatic reviews on.

Automatic and model-backed manual runs need repository secret `DOCS_REVIEW_API_KEY` and these repository variables:

- `DOCS_REVIEW_MODEL` is JSON with `model_name`, `endpoint_url`, `max_output_tokens`, and `api_key_environment_variable` (use `DOCS_REVIEW_API_KEY`). Set `reasoning_effort` when the model supports it. Set `max_tool_rounds` to 1–4 to let the model look up evidence in the pinned head commit. The default of 0 keeps the single-request review.
- `DOCS_REVIEW_BUDGET` holds `call_cost_cents`, `max_cost_cents_per_call`, `max_cost_cents_per_pull_request`, and `max_cost_cents_per_day`. The workflow reserves the estimated cost before each review attempt. For tool-assisted reviews, size `call_cost_cents` for up to `1 + max_tool_rounds` model requests per attempt, including tool results in later requests.

Each tool-assisted review can run up to eight Git lookups. The reviewer returns at most 12,000 bytes per lookup and 40,000 bytes in total. A lookup stops if its Git output exceeds 8 MiB. The report records the call count and the paths whose lines were read.

Create a `docs-review-ledger` branch with root-level `docs-review-ledger.json` containing `{"version":1,"reservations":[]}` before the first live review.

## Manual dispatch

```sh
gh workflow run pr-advisory-review.yaml -f pull_request_number=123 -f offline=true
```

Set `offline=false` for one live call. Use `follow_up=true` for a second call on the same pull request when the reviewed documentation or code inputs changed. The pilot includes unified documentation diffs, auto-discovered referenced code or config files, and hunk/keyword excerpts to keep model inputs within the 100,000-byte cap.

Reports upload as `docs-review-RUN_ID-ATTEMPT` artifacts with `report.md` and `report.json`. Automatic runs post complete findings directly on the pull request and update the same comment on later pushes.
