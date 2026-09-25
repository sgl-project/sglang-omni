# Refactor Net Deletion Tracking

Use `scripts/refactor_net_deletions.py` to track whether a refactor is reducing
non-test code over time. The TTS refactor roadmap lives in
[#985](https://github.com/sgl-project/sglang-omni/issues/985); this script is
the line-count companion for PRs and issue updates.

The progress target is:

```text
non-test net deleted = deleted non-test lines - added non-test lines
```

The target is met when `non-test net deleted > 0`. Test changes are still
reported, but they are excluded from the progress target so that added coverage
does not hide whether the refactor is actually shrinking implementation code.

The HTML dashboard uses conventional diff signs for readability:
`net change = added lines - deleted lines`. A net code reduction is therefore
displayed as a negative number even though the internal progress target above is
expressed as a positive `net deleted` value.

## Test File Detection

A changed file is treated as a test file when either of these rules match:

- Any parent directory is named `test`, `tests`, `unit_test`, `unit_tests`,
  `integration_test`, or `integration_tests`.
- The basename is `conftest.py`, starts with `test_`, or ends with a common test
  suffix such as `_test.py`, `_tests.py`, `.test.ts`, or `.spec.tsx`.

Because the whole path under `tests/` is excluded, fixtures, test data, helper
modules, and CI-only test utilities do not count toward the non-test deletion
target.

## Common Commands

For a PR branch, compare against the merge base with `origin/main`:

```bash
python3 scripts/refactor_net_deletions.py \
  --base origin/main \
  --head HEAD \
  --format markdown \
  --list-test-files
```

For local tracked work before committing, include the working tree:

```bash
python3 scripts/refactor_net_deletions.py \
  --base origin/main \
  --head HEAD \
  --mode worktree \
  --list-test-files \
  --list-non-test-files
```

For an issue or PR comment, use Markdown output. For automation, use JSON:

```bash
python3 scripts/refactor_net_deletions.py --format json
```

If a future CI job should enforce the target, add `--fail-on-nonpositive`.
Leave that flag off for normal tracking because some intermediate refactor PRs
may add shared infrastructure before later PRs delete model-local code.

## HTML Dashboard

The current dashboard is published at
[TTS Refactor Progress](https://sgl-project.github.io/sglang-omni/tts-refactor/).
The docs workflow rebuilds it from `main` on every documentation deployment.

The same script can write a static dashboard. Serve the output directory with a
plain local HTTP server, then expose that localhost port with
[Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
or [ngrok](https://ngrok.com/docs/getting-started).

The dashboard also includes a `Reusable shared surfaces` panel that links the
TTS refactor design doc and the shared files developers should check before
adding new model-local code.

The dashboard uses commit-sum mode rather than diffing the baseline against the
latest `main`. Its reviewed allowlist is
`scripts/tts_refactor_commits.json`. This follows the original #985 snapshot
method and prevents unrelated work in shared directories from changing the
result. Model integrations such as #858 and #1112 are excluded. File counts in
this mode are commit file touches, not unique paths. Add a merged refactor PR's
immutable merge SHA to the allowlist to include it in future deployments.
Stacked PR #930 is represented by its landing PR #927 rather than counted
twice; its standalone non-test net contribution in the old snapshot was zero.

The parent of the first landed TTS refactor PR is the stable history boundary:

```bash
git rev-parse 4e4c98a^1
```

Generate the dashboard once:

```bash
python3 scripts/refactor_net_deletions.py \
  --base 4e4c98a^1 \
  --head origin/main \
  --mode commit-sum \
  --commit-file scripts/tts_refactor_commits.json \
  --format html \
  --output /data/jaxan/tts-refactor-dashboard/index.html \
  --title "TTS Refactor Progress" \
  --refresh-seconds 300 \
  --scope-note "Counts only the reviewed TTS refactor commit allowlist; model integrations and unrelated main commits are excluded." \
  --list-test-files \
  --list-non-test-files
```

For an H100 host, keep the checkout and dashboard under persistent storage such
as `/data/jaxan`. Run the refresher and server in separate `tmux` panes:

```bash
while true; do
  git -C /data/jaxan/sglang-omni fetch origin main
  python3 /data/jaxan/sglang-omni/scripts/refactor_net_deletions.py \
    --repo /data/jaxan/sglang-omni \
    --base 4e4c98a^1 \
    --head origin/main \
    --mode commit-sum \
    --commit-file scripts/tts_refactor_commits.json \
    --format html \
    --output /data/jaxan/tts-refactor-dashboard/index.html \
    --title "TTS Refactor Progress" \
    --refresh-seconds 300 \
    --scope-note "Counts only the reviewed TTS refactor commit allowlist; model integrations and unrelated main commits are excluded." \
    --list-test-files \
    --list-non-test-files
  sleep 300
done
```

```bash
python3 -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory /data/jaxan/tts-refactor-dashboard
```

Expose it temporarily with Cloudflare:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

Or with ngrok:

```bash
ngrok http 127.0.0.1:8765
```

Only expose the generated dashboard directory. Do not serve the full checkout or
any directory containing credentials, caches, checkpoints, or private artifacts.
