#!/usr/bin/env bash
# actnow-publish.sh: publish a MyGo build to actnow.dev through the site's
# release API. Every file goes through the site (it checks, stores and
# publishes them); CI needs no R2 credentials, only ACTNOW_INGEST_TOKEN.
#
#   ACTNOW_INGEST_TOKEN=… scripts/actnow-publish.sh --app audiosmith --channel beta \
#     --dir build/darwin-arm64 --changelog CHANGELOG.md
#
# Needs bash (3.2, as on macOS, is enough), curl, jq (1.6+) and shasum or
# sha256sum. Steps:
#   1. finds the build's files: *.dmg, *.tar.gz, *.delta, update-*.json
#      (DMG names with spaces, as MyGo writes them, are sent with dashes)
#   2. the version: --version, else the manifests' "version"
#   3. the notes: this version's "## <version>" section of --changelog;
#      English up to the first heading written in Chinese ("### 中文",
#      "### 新增"…), the rest as notesZh
#   4. POST /api/releases/sessions, upload each file (big ones in parts),
#      POST /api/releases/<id>/complete; the site publishes when the app's
#      settings say so (--no-publish stops at "verified")
# It fails with the site's message on any error, and aborts the session
# when an upload fails.
#
# Options:
#   --app <slug>          pik | auralink | dicta | audiosmith | voxtnew (required)
#   --channel <name>      stable | beta (default: beta for x.y.z-pre, else stable)
#   --dir <path>          MyGo's output directory, e.g. build/darwin-arm64 (required;
#                         searched two levels deep, so build/ works for several platforms)
#   --changelog <file>    CHANGELOG.md to take the notes from (default: none)
#   --notes-file <file>   English notes as a file instead (and --notes-zh-file)
#   --version <x.y.z>     default: from update-*.json
#   --date <YYYY-MM-DD>   release date (default: today on the site)
#   --min-macos <ver>     e.g. 15.0
#   --no-publish          stop at "verified" (publish later in the panel)
#   --url <base>          default $ACTNOW_URL or https://actnow.dev
#   --dry-run             check and print the session request, send nothing
set -euo pipefail

APP="" CHANNEL="" DIR="" CHANGELOG="" NOTES_FILE="" NOTES_ZH_FILE="" VERSION=""
DATE="" MIN_MACOS="" PUBLISH="" DRY_RUN=0
BASE_URL="${ACTNOW_URL:-https://actnow.dev}"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '%s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --app) APP="$2"; shift 2 ;;
    --channel) CHANNEL="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --changelog) CHANGELOG="$2"; shift 2 ;;
    --notes-file) NOTES_FILE="$2"; shift 2 ;;
    --notes-zh-file) NOTES_ZH_FILE="$2"; shift 2 ;;
    --version) VERSION="${2#v}"; shift 2 ;;
    --date) DATE="$2"; shift 2 ;;
    --min-macos) MIN_MACOS="$2"; shift 2 ;;
    --no-publish) PUBLISH=false; shift ;;
    --url) BASE_URL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
BASE_URL="${BASE_URL%/}"

command -v jq >/dev/null || die "jq is required"
command -v curl >/dev/null || die "curl is required"
[ -n "$APP" ] || die "--app is required"
[ -n "$DIR" ] || die "--dir is required"
[ -d "$DIR" ] || die "no directory $DIR"
if [ "$DRY_RUN" = 0 ] && [ -z "${ACTNOW_INGEST_TOKEN:-}" ]; then
  die "ACTNOW_INGEST_TOKEN is not set"
fi

sha256_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
size_of() { wc -c < "$1" | tr -d ' '; }

# ---- 1. files ---------------------------------------------------------------
FILES=()
while IFS= read -r -d '' file; do FILES+=("$file"); done < <(
  find "$DIR" -maxdepth 2 -type f \( -name '*.dmg' -o -name '*.tar.gz' -o -name '*.delta' -o -name 'update-*.json' \) -print0 | sort -z
)
[ "${#FILES[@]}" -gt 0 ] || die "no .dmg, .tar.gz, .delta or update-*.json in $DIR"

MANIFESTS=()
for file in "${FILES[@]}"; do
  case "$(basename "$file")" in update-*.json) MANIFESTS+=("$file") ;; esac
done
# bash 3.2 with set -u: "${a[@]}" of an empty array is an error.
manifests() { [ "${#MANIFESTS[@]}" -eq 0 ] || printf '%s\0' "${MANIFESTS[@]}"; }

# ---- 2. version ---------------------------------------------------------------
if [ -z "$VERSION" ]; then
  [ "${#MANIFESTS[@]}" -gt 0 ] || die "no update-*.json to read the version from: pass --version (or set MYGO_UPDATER_PRIVATE_KEY so mygo build writes the manifests)"
  VERSION="$(jq -r '.version' "${MANIFESTS[0]}")"
fi
while IFS= read -r -d '' manifest; do
  v="$(jq -r '.version' "$manifest")"
  [ "$v" = "$VERSION" ] || die "$(basename "$manifest") is version $v, not $VERSION"
done < <(manifests)
if [ -z "$CHANNEL" ]; then
  case "$VERSION" in *-*) CHANNEL=beta ;; *) CHANNEL=stable ;; esac
fi
case "$CHANNEL" in stable|beta) ;; *) die "--channel is stable or beta" ;; esac

# The manifests must point at this site's feed, or installed apps would
# never find the update here.
FEED_DIR="$APP"; [ "$CHANNEL" = beta ] && FEED_DIR="$APP/beta"
while IFS= read -r -d '' manifest; do
  url="$(jq -r '.url' "$manifest")"
  case "$url" in
    "$BASE_URL/updates/$FEED_DIR/"*) ;;
    *) die "$(basename "$manifest") points at $url, not $BASE_URL/updates/$FEED_DIR/…: set updates.url to $BASE_URL/updates/$FEED_DIR in the build's mygo.json" ;;
  esac
done < <(manifests)

# ---- 3. notes -----------------------------------------------------------------
NOTES="" NOTES_ZH=""
if [ -n "$CHANGELOG" ]; then
  [ -f "$CHANGELOG" ] || die "no changelog $CHANGELOG"
  split="$(jq -Rs --arg v "$VERSION" '
    def strip: sub("^\\s+"; "") | sub("\\s+$"; "");
    def headver: (if startswith("[") then .[1:] | split("]")[0] else . end) | sub("^v"; "");
    reduce (split("\n")[] | sub("\r$"; "")) as $l ({in: false, done: false, out: []};
      if .done then .
      elif ($l | startswith("## ")) then
        (if .in then .done = true
         else .in = (($l[3:] | split(" ") | map(select(. != "")) | .[0] // "") | headver) == $v end)
      elif .in then .out += [$l] else . end)
    | .out as $lines
    | ([$lines | to_entries[]
        | select((.value | test("^###?\\s")) and (.value | sub("^#+\\s*"; "") | test("\\p{Han}")))
        | .key] | first) as $i
    | if $i == null then {en: ($lines | join("\n") | strip), zh: null}
      else ($lines[$i] | sub("^#+\\s*"; "") | strip) as $h
        | {en: ($lines[:$i] | join("\n") | strip),
           zh: ((if ($h | test("^(中文|简体中文|繁體中文|Chinese|zh|zh-CN)$"; "i"))
                 then $lines[$i + 1:] else $lines[$i:] end) | join("\n") | strip)}
      end' "$CHANGELOG")"
  NOTES="$(jq -r '.en' <<<"$split")"
  NOTES_ZH="$(jq -r '.zh // empty' <<<"$split")"
  [ -n "$NOTES" ] || die "$CHANGELOG has no \"## $VERSION\" section (or it is empty)"
fi
[ -n "$NOTES_FILE" ] && NOTES="$(cat "$NOTES_FILE")"
[ -n "$NOTES_ZH_FILE" ] && NOTES_ZH="$(cat "$NOTES_ZH_FILE")"

# ---- the session request --------------------------------------------------------
manifest_platform=""
[ "${#MANIFESTS[@]}" -eq 1 ] && manifest_platform="$(basename "${MANIFESTS[0]}" .json)" && manifest_platform="${manifest_platform#update-}"

ARTIFACTS="[]"
NAMES=()   # the names sent, parallel to FILES
path_of() {
  local i
  for i in "${!NAMES[@]}"; do
    [ "${NAMES[$i]}" = "$1" ] && { printf '%s' "${FILES[$i]}"; return 0; }
  done
  return 1
}
for file in "${FILES[@]}"; do
  base="$(basename "$file")"
  name="$(printf '%s' "$base" | tr -s ' ' '-')"
  platform=""
  case "$name" in
    *.dmg)
      # MyGo: "<Name> <version> <arch>.dmg", no arch for universal builds.
      case "$name" in
        *-arm64.dmg) platform=darwin-arm64 ;;
        *-amd64.dmg|*-x86_64.dmg) platform=darwin-amd64 ;;
        *) platform="${manifest_platform:-darwin-universal}" ;;
      esac ;;
  esac
  if [ "${#NAMES[@]}" -gt 0 ] && path_of "$name" >/dev/null; then
    die "two files are both sent as $name"
  fi
  NAMES+=("$name")
  ARTIFACTS="$(jq --arg f "$name" --arg p "$platform" --argjson s "$(size_of "$file")" --arg h "$(sha256_of "$file")" \
    '. + [{filename: $f, size: $s, sha256: $h} + (if $p != "" then {platform: $p} else {} end)]' <<<"$ARTIFACTS")"
done

REQUEST="$(jq -n \
  --arg app "$APP" --arg version "$VERSION" --arg channel "$CHANNEL" \
  --arg notes "$NOTES" --arg notesZh "$NOTES_ZH" --arg date "$DATE" \
  --arg minMacOS "$MIN_MACOS" --arg publish "$PUBLISH" --argjson artifacts "$ARTIFACTS" '
  {app: $app, version: $version, channel: $channel, notes: $notes, artifacts: $artifacts}
  + (if $notesZh != "" then {notesZh: $notesZh} else {} end)
  + (if $date != "" then {date: $date} else {} end)
  + (if $minMacOS != "" then {minMacOS: $minMacOS} else {} end)
  + (if $publish == "false" then {publish: false} else {} end)')"

log "ActNow release: $APP $VERSION ($CHANNEL) → $BASE_URL"
jq -r '.artifacts[] | "  \(.filename)  \(.size) bytes  \(.sha256[0:12])…"' <<<"$REQUEST" >&2
if [ "$DRY_RUN" = 1 ]; then
  jq '.notes |= (.[0:200] + (if length > 200 then "…" else "" end)) | if .notesZh then .notesZh |= .[0:200] else . end' <<<"$REQUEST"
  log "dry run: nothing sent"
  exit 0
fi

# ---- 4. API --------------------------------------------------------------------
TMP="$(mktemp -d)"
RELEASE_ID=""
COMPLETED=0
cleanup() {
  local status=$?
  if [ "$status" -ne 0 ] && [ -n "$RELEASE_ID" ] && [ "$COMPLETED" = 0 ]; then
    log "aborting the upload session $RELEASE_ID"
    curl -sS -o /dev/null -X DELETE -H "authorization: Bearer $ACTNOW_INGEST_TOKEN" \
      "$BASE_URL/api/releases/$RELEASE_ID" || true
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

# call <method> <url> [curl args…]: prints the body; dies with the site's
# message on an error; retries network errors and 5xx (3 times).
call() {
  local method="$1" url="$2"; shift 2
  local attempt code
  for attempt in 1 2 3 4; do
    code="$(curl -sS -o "$TMP/body" -w '%{http_code}' -X "$method" \
      -H "authorization: Bearer $ACTNOW_INGEST_TOKEN" "$@" "$url" 2>"$TMP/err")" || code=000
    case "$code" in
      2??) cat "$TMP/body"; return 0 ;;
      000|5??)
        if [ "$attempt" -lt 4 ]; then
          log "  $method $url: ${code/000/network error}, retrying ($attempt)…"
          sleep $((attempt * 3)); continue
        fi ;;
    esac
    break
  done
  local message
  message="$(jq -r '[.message // .error // empty, (.details.errors // [] | .[] | "  - " + .)] | join("\n")' "$TMP/body" 2>/dev/null || true)"
  die "$method ${url#"$BASE_URL"} → HTTP $code: ${message:-$(head -c 500 "$TMP/body" 2>/dev/null; cat "$TMP/err")}"
}

SESSION="$(call POST "$BASE_URL/api/releases/sessions" -H 'content-type: application/json' --data-binary "$REQUEST")"
RELEASE_ID="$(jq -r '.release.id' <<<"$SESSION")"
log "session $RELEASE_ID opened"

while IFS= read -r artifact; do
  name="$(jq -r '.filename' <<<"$artifact")"
  method="$(jq -r '.upload.method' <<<"$artifact")"
  url="$(jq -r '.upload.url' <<<"$artifact")"
  file="$(path_of "$name")" || die "the site asked for $name, which is not in $DIR"
  if [ "$method" = PUT ]; then
    result="$(call PUT "$url" -H 'content-type: application/octet-stream' --data-binary "@$file")"
  else
    part_size="$(jq -r '.upload.partSize' <<<"$artifact")"
    part_count="$(jq -r '.upload.partCount' <<<"$artifact")"
    call POST "$url" >/dev/null
    parts="[]"
    for ((n = 1; n <= part_count; n++)); do
      dd if="$file" of="$TMP/part" bs="$part_size" skip=$((n - 1)) count=1 2>/dev/null
      part="$(call PUT "$url?part=$n" -H 'content-type: application/octet-stream' \
        -H "x-actnow-part-sha256: $(sha256_of "$TMP/part")" --data-binary "@$TMP/part")"
      parts="$(jq --argjson p "$part" '. + [{partNumber: $p.partNumber, etag: $p.etag}]' <<<"$parts")"
      log "  $name: part $n/$part_count"
    done
    result="$(call POST "$url/complete" -H 'content-type: application/json' \
      --data-binary "$(jq -n --argjson parts "$parts" '{parts: $parts}')")"
  fi
  log "  ✓ $name ($(jq -r '.size' <<<"$result") bytes, sha256 verified)"
done < <(jq -c '.artifacts[]' <<<"$SESSION")

RESULT="$(call POST "$BASE_URL/api/releases/$RELEASE_ID/complete")"
COMPLETED=1
STATUS="$(jq -r '.release.status' <<<"$RESULT")"
jq -r '.warnings // [] | .[] | "warning: " + .' <<<"$RESULT" >&2
jq -r '.feeds // [] | .[] | select(.changed) | "  feed /updates/\(.key) → \(.version // "removed")"' <<<"$RESULT" >&2
log "$APP $VERSION ($CHANNEL): $STATUS"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### ActNow: $APP $VERSION ($CHANNEL) — $STATUS"
    jq -r '.feeds // [] | .[] | select(.changed) | "- `/updates/\(.key)` → \(.version // "removed")"' <<<"$RESULT"
    [ "$STATUS" = published ] || echo "Publish it in $BASE_URL/panel/releases."
  } >> "$GITHUB_STEP_SUMMARY"
fi
jq '{release: .release | {id, app, version, channel, status}, published, feeds: [.feeds // [] | .[] | select(.changed)]}' <<<"$RESULT"
