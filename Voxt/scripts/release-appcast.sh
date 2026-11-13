#!/usr/bin/env bash
# What Voxt 1.x's in-app update needs to install a Voxt release, and the
# Homebrew cask (docs/release.md, "Voxt 1.x 的应用内更新和 Homebrew").
# Voxt 1.x updates with Sparkle 2.10 from the Voxt update service's
# appcast: Sparkle offers an entry whose sparkle:version is above the
# running app's CFBundleVersion (1.16.3: 101600399), and installs the
# Voxt.app of its ZIP only when that app's own CFBundleVersion is not below
# it and its Info.plist has an EdDSA key (Sparkle's SUPlainInstaller.m,
# SUUpdateValidator.m). On an Intel Mac, Sparkle skips an entry whose
# sparkle:hardwareRequirements is arm64 (SPUAppcastItemStateResolver.m), as
# Voxt 2 does not open there. scripts/release.sh writes those keys,
# .github/workflows/voxt-release.yaml names the files with this script,
# and .github/workflows/voxt-appcast.yaml makes the entry and the cask.
#
#   scripts/release-appcast.sh info-plist 2.0.1    # {"CFBundleVersion":"200000199","SUPublicEDKey":"…"}
#   scripts/release-appcast.sh assets 2.0.1        # Voxt-2.0.1-macOS.zip Voxt-2.0.1-arm64.dmg
#   scripts/release-appcast.sh entry beta 2.0.1    # the update service's entry (beta or stable), JSON
#   scripts/release-appcast.sh notes < NOTES.md    # the English release notes as HTML
#   scripts/release-appcast.sh notes-by-locale < NOTES.md   # the release notes as HTML by locale, JSON
#   scripts/release-appcast.sh items < appcast.xml # each item's sparkle:version and hardware requirement
#   scripts/release-appcast.sh hardware-kept BUILD REPLY < APPCASTS   # fails unless the update service kept arm64
#   scripts/release-appcast.sh cask 2.0.1 SHA256 OWNER/REPO   # Homebrew's Casks/voxt.rb
set -euo pipefail

# Voxt 1.x's SUPublicEDKey: the pair of the SPARKLE_PRIVATE_KEY secret that signs its updates.
public_key="iMt1voRdgfUUpKxdY7DliGLaOl17gJFB8Jhlo5ox2eE="
# Voxt 2 runs on Apple silicon only.
hardware=arm64

usage() {
  sed -n 's/^#   //p' "$0" >&2
  exit 2
}

fail() {
  echo "release-appcast.sh: $1" >&2
  exit 2
}

json_python() {
  if python3 -c 'pass' >/dev/null 2>&1; then
    python3 "$@"
  else
    python "$@"
  fi
}

# Voxt 1.x's build number of a version: major×10⁸ + minor×10⁵ + patch×100
# + 99, or + 0 for a pre-release (X.Y.Z-nightly.…), which comes before its
# release.
build_number() {
  if [[ ! "$1" =~ ^([0-9]+)\.([0-9]{1,3})\.([0-9]{1,3})(-[0-9A-Za-z.-]+)?$ ]]; then
    fail "a version is X.Y.Z or X.Y.Z-PRERELEASE with minor and patch below 1000, got '$1'"
  else
    :
  fi
  local last=99
  if [[ -n "${BASH_REMATCH[4]}" ]]; then
    last=0
  else
    :
  fi
  echo $((10#${BASH_REMATCH[1]} * 100000000 + 10#${BASH_REMATCH[2]} * 100000 + 10#${BASH_REMATCH[3]} * 100 + last))
}

zip_name() { echo "Voxt-$1-macOS.zip"; }
dmg_name() { echo "Voxt-$1-arm64.dmg"; }

# The items of an appcast on stdin, in its order, one line each: the item's
# sparkle:version and its sparkle:hardwareRequirements, or none.
items() {
  awk '
    function tag(s, name) {
      if (!match(s, "<" name ">[^<]*</" name ">")) return ""
      s = substr(s, RSTART + length(name) + 2, RLENGTH - 2 * length(name) - 5)
      gsub(/^[ \t]+|[ \t]+$/, "", s)
      return s
    }
    { doc = doc " " $0 }
    END {
      n = split(doc, parts, /<item[ >]/)
      for (i = 2; i <= n; i++) {
        item = parts[i]
        sub(/<\/item>.*/, "", item)
        version = tag(item, "sparkle:version")
        needs = tag(item, "sparkle:hardwareRequirements")
        if (needs == "") needs = "none"
        if (version != "") print version, needs
      }
    }
  '
}

notes_html() {
  awk -v locale="$1" '
    function inline_html(text,    html) {
      gsub(/&/, "\\&amp;", text); gsub(/</, "\\&lt;", text); gsub(/>/, "\\&gt;", text); gsub(/"/, "\\&quot;", text)
      while (match(text, /\*\*[^*]+\*\*/)) {
        html = html substr(text, 1, RSTART - 1) "<strong>" substr(text, RSTART + 2, RLENGTH - 4) "</strong>"
        text = substr(text, RSTART + RLENGTH)
      }
      return html text
    }
    function end_list() { if (in_list) print "</ul>"; in_list = 0 }
    BEGIN { section_locale = "en" }
    { sub(/^[ \t]+/, ""); sub(/[ \t\r]+$/, "") }
    /^## / { exit }
    $0 == "### 中文" { end_list(); section_locale = "zh-Hans"; next }
    /^### / { end_list(); section_locale = "en" }
    section_locale != locale { next }
    $0 == "" { end_list(); next }
    /^### / { print "<h3>" inline_html(substr($0, 5)) "</h3>"; next }
    /^#### / {
      end_list()
      heading = locale == "zh-Hans" ? "h3" : "h4"
      print "<" heading ">" inline_html(substr($0, 6)) "</" heading ">"
      next
    }
    /^[-*] / { if (!in_list) print "<ul>"; in_list = 1; print "<li>" inline_html(substr($0, 3)) "</li>"; next }
    { end_list(); print "<p>" inline_html($0) "</p>" }
    END { end_list() }
  '
}

case "${1:-}" in
  info-plist)
    (($# == 2)) || usage
    build="$(build_number "$2")"
    printf '{"CFBundleVersion":"%s","SUPublicEDKey":"%s"}\n' "$build" "$public_key"
    ;;
  assets)
    (($# == 2)) || usage
    build_number "$2" >/dev/null
    echo "$(zip_name "$2") $(dmg_name "$2")"
    ;;
  entry)
    (($# == 3)) || usage
    case "$2" in
      beta | stable) ;;
      *) fail "the channel is beta or stable, got '$2'" ;;
    esac
    build="$(build_number "$3")"
    printf '{"channel":"%s","version":"%s","sparkleVersion":"%s","hardwareRequirements":"%s","assetFileName":"%s","assetContentType":"application/zip","dmgFileName":"%s","dmgContentType":"application/x-apple-diskimage"}\n' \
      "$2" "$3" "$build" "$hardware" "$(zip_name "$3")" "$(dmg_name "$3")"
    ;;
  items)
    (($# == 1)) || usage
    items
    ;;
  hardware-kept)
    (($# == 3)) || usage
    [[ "$2" =~ ^[0-9]+$ ]] || fail "'$2' is not a build number"
    # The update service's reply to the entry: ok, with the requirement in the entry it kept (data),
    # so that a reply that echoes the request fails. Or the item of the build in the appcasts on stdin.
    if json_python -c 'import json, sys
try:
    reply = json.loads(sys.argv[1])
except (ValueError, TypeError):
    sys.exit(1)
data = reply.get("data") if isinstance(reply, dict) else None
sys.exit(0 if isinstance(reply, dict) and reply.get("ok") is True and isinstance(data, dict) and data.get("hardwareRequirements") == sys.argv[2] else 1)' "$3" "$hardware" >/dev/null 2>&1; then
      echo "The update service keeps hardwareRequirements $hardware"
    elif items | grep -xF "$2 $hardware" >/dev/null; then
      echo "The appcast's item of build $2 has sparkle:hardwareRequirements $hardware"
    else
      echo "release-appcast.sh: the update service does not output hardwareRequirements yet, so Voxt 1.x on an Intel Mac would get a version it cannot open: add it on the server (/api/pkg/update, and <sparkle:hardwareRequirements>$hardware</sparkle:hardwareRequirements> in the appcast)" >&2
      exit 1
    fi
    ;;
  notes)
    (($# == 1)) || usage
    notes_html en
    ;;
  notes-by-locale)
    (($# == 1)) || usage
    release_body="$(cat)"
    english_notes="$(notes_html en <<< "$release_body")"
    [[ -n "$english_notes" ]] || fail "release notes must include English content before the first level-two heading"
    chinese_notes="$(notes_html zh-Hans <<< "$release_body")"
    ENGLISH_NOTES="$english_notes" CHINESE_NOTES="$chinese_notes" json_python -c 'import json, os
payload = {"en": os.environ["ENGLISH_NOTES"]}
if os.environ["CHINESE_NOTES"]:
    payload["zh-Hans"] = os.environ["CHINESE_NOTES"]
print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))'
    ;;
  cask)
    (($# == 4)) || usage
    version="$2" sha256="$3" repo="$4"
    build_number "$version" >/dev/null
    [[ "$sha256" =~ ^[0-9a-f]{64}$ ]] || fail "'$sha256' is not a SHA-256"
    [[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || fail "'$repo' is not OWNER/REPO"
    # No auto_updates: Voxt 2 does not update itself, so brew upgrade updates it.
    cat <<CASK
cask "voxt" do
  version "$version"
  sha256 "$sha256"

  url "https://github.com/$repo/releases/download/voxt-v#{version}/$(zip_name '#{version}')"
  name "Voxt"
  desc "Voice input and translation menu bar app for macOS"
  homepage "https://github.com/$repo/tree/main/Voxt"

  livecheck do
    url :url
    regex(/^voxt-v(\d+(?:\.\d+)+)$/i)
    strategy :github_releases
  end

  depends_on arch: :arm64
  depends_on macos: :sequoia

  app "Voxt.app"
end
CASK
    ;;
  *) usage ;;
esac
