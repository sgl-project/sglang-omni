#!/usr/bin/env bash
# Deploy or remove the GitHub Pages preview of one pull request.
#
# Docs Preview (.github/workflows/docs-preview.yaml) runs this from the default
# branch with a write token. The site directory is the pull request's own build
# and is treated as hostile: symlinks, nested .git and .github directories are
# dropped, the size is capped, every page gets a noindex tag and a banner, and
# the result lands only under pr-preview/pr-<number>/ on the gh-pages checkout.
# Previews of closed pull requests and previews beyond MAX_PREVIEWS are pruned
# on every run, so the published site stays well under the GitHub Pages limit.
#
#   docs_preview.sh deploy <number> <head_sha> <site_dir> <gh_pages_dir>
#   docs_preview.sh remove <number> <gh_pages_dir>
#
# Needs GH_TOKEN and GITHUB_REPOSITORY for the pull request state lookups.
set -euo pipefail

MAX_PREVIEWS="${MAX_PREVIEWS:-15}"
MAX_BYTES="${MAX_BYTES:-60000000}"
MAX_FILES="${MAX_FILES:-1500}"

action="${1:?deploy or remove}"
number="${2:?pull request number}"
[[ "$number" =~ ^[0-9]+$ ]] || { echo "pull request number must be numeric: $number" >&2; exit 1; }

case "$action" in
  deploy)
    head_sha="${3:?head sha}"
    site="${4:?site dir}"
    pages="${5:?gh-pages dir}"
    [[ "$head_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "head sha must be 40 hex digits: $head_sha" >&2; exit 1; }
    # The script changes into the gh-pages checkout later.
    site=$(cd "$site" && pwd)
    ;;
  remove)
    pages="${3:?gh-pages dir}"
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 1
    ;;
esac

sanitize_site() {
  find "$site" -type l -delete
  find "$site" -type d \( -name .git -o -name .github \) -prune -exec rm -rf {} +
  rm -rf "$site/_sources" "$site/_downloads"
  local bytes files
  bytes=$(du -sb "$site" | cut -f1)
  files=$(find "$site" -type f | wc -l)
  if [ "$bytes" -gt "$MAX_BYTES" ] || [ "$files" -gt "$MAX_FILES" ]; then
    echo "refusing to deploy: $bytes bytes in $files files (limits $MAX_BYTES bytes, $MAX_FILES files)" >&2
    exit 1
  fi
  # Keep previews out of search engines and make their status obvious.
  local banner="<div style=\"position:sticky;top:0;z-index:9999;background:#fff3cd;color:#664d03;border-bottom:1px solid #ffe69c;padding:6px 12px;font:14px system-ui,sans-serif;text-align:center\">Unreviewed preview of pull request #${number} at commit ${head_sha:0:12}. Nothing here is an approval.</div>"
  find "$site" -type f -name '*.html' -print0 | xargs -0 -r sed -i \
    -e '0,/<head[^>]*>/s//&<meta name="robots" content="noindex, nofollow">/' \
    -e "0,/<body[^>]*>/s||&${banner}|"
  printf '{"pull_request": %s, "head_sha": "%s", "deployed_at": "%s"}\n' \
    "$number" "$head_sha" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$site/.preview.json"
}

pr_is_open() {
  local state
  state=$(gh api "repos/${GITHUB_REPOSITORY}/pulls/$1" --jq .state 2>/dev/null || echo unknown)
  [ "$state" != closed ]
}

# Rebuild pr-preview/ in the gh-pages working tree from the current state.
apply_changes() {
  mkdir -p pr-preview
  if [ "$action" = deploy ]; then
    rm -rf "pr-preview/pr-${number}"
    cp -a "$site" "pr-preview/pr-${number}"
  else
    rm -rf "pr-preview/pr-${number}"
  fi
  # Drop previews whose pull request is closed.
  local dir n
  for dir in pr-preview/pr-*/; do
    [ -d "$dir" ] || continue
    n="${dir#pr-preview/pr-}"
    n="${n%/}"
    [[ "$n" =~ ^[0-9]+$ ]] || { rm -rf "$dir"; continue; }
    [ "$n" = "$number" ] && continue
    pr_is_open "$n" || { echo "pruning preview of closed pull request #$n"; rm -rf "$dir"; }
  done
  # Keep only the newest MAX_PREVIEWS, never dropping the one just deployed.
  local keep=0
  for dir in $(for d in pr-preview/pr-*/; do
      [ -d "$d" ] || continue
      printf '%s %s\n' "$(sed -n 's/.*"deployed_at": "\([^"]*\)".*/\1/p' "$d/.preview.json" 2>/dev/null || true)" "$d"
    done | sort -r | awk '{print $NF}'); do
    n="${dir#pr-preview/pr-}"
    n="${n%/}"
    if [ "$n" = "$number" ] && [ "$action" = deploy ]; then continue; fi
    keep=$((keep + 1))
    if [ "$keep" -ge "$MAX_PREVIEWS" ]; then
      echo "pruning preview of pull request #$n: more than $MAX_PREVIEWS previews"
      rm -rf "$dir"
    fi
  done
}

if [ "$action" = deploy ]; then
  sanitize_site
fi

cd "$pages"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

if [ "$action" = deploy ]; then
  message="Deploy docs preview for PR #${number} (${head_sha})"
else
  message="Remove docs preview for PR #${number}"
fi

# Docs Publish and other previews push to gh-pages too; retry on a lost race.
for attempt in 1 2 3 4 5; do
  apply_changes
  git add -A pr-preview
  if git diff --cached --quiet; then
    echo "gh-pages already up to date"
    exit 0
  fi
  git commit -q -m "$message"
  if git push -q origin HEAD:gh-pages; then
    echo "pushed on attempt $attempt"
    exit 0
  fi
  echo "push rejected, retrying ($attempt/5)"
  git fetch -q --depth=1 origin gh-pages
  git reset -q --hard origin/gh-pages
done
echo "could not push to gh-pages after 5 attempts" >&2
exit 1
