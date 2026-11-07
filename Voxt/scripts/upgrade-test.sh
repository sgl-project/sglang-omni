#!/usr/bin/env bash
# The Voxt 1.x → 2.x upgrade test of .github/workflows/voxt-upgrade-test.yaml
# (docs/release.md, "1.x → 2.x 升级测试"): Voxt 1.x, the sandboxed Swift app
# (com.voxt.Voxt), updates itself with its own Sparkle from the beta appcast
# to Voxt 2 (dev.hehehai.voxtnew), which then imports 1.x's data. Only on a
# fresh GitHub macOS runner: it installs Voxt 1.x in /Applications and
# writes its container, ~/Library/Containers/com.voxt.Voxt, which no fake
# HOME keeps apart from the real one. Never run it on a Mac someone uses.
#
#   scripts/upgrade-test.sh start|install|prepare|update|check-app|check-import|evidence
#
# FROM_TAG is the Voxt 1.x release (voxt-v1.16.3), TO_VERSION the Voxt 2
# version the beta appcast offers, OUT the folder the workflow keeps as its
# artifact. Each check adds a row to $OUT/results.md; one that fails says
# where it stopped (::error::) and exits 1.
set -euo pipefail

: "${FROM_TAG:?the Voxt 1.x tag, e.g. voxt-v1.16.3}" "${TO_VERSION:?the Voxt 2 version, X.Y.Z}" "${OUT:?the artifact folder}"
APP=/Applications/Voxt.app
LEGACY_ID=com.voxt.Voxt
NEW_ID=dev.hehehai.voxtnew
TEAM=RY2TQTYSKQ
DATA="$HOME/Library/Containers/$LEGACY_ID/Data"
PREFS="$DATA/Library/Preferences/$LEGACY_ID" # defaults(1) takes it without .plist
VOXT_DIR="$DATA/Library/Application Support/Voxt" # 1.x's voxt.sqlite and Logs/
BETA_APPCAST=https://voxt.actnow.dev/updates/beta/appcast.xml
# What Voxt 1.16.3 wrote after three terms were added (the rows only in its
# WAL); TestSmokeLegacy expects the same line from it.
FIXTURE=testdata/legacy/real
WANT_IMPORT="smoke: legacy import: 3 terms, 0 history entries"

row() { printf '| %s | %s | %s |\n' "$1" "$2" "${3//|/\\|}" >> "$OUT/results.md"; }
pass() { row "$1" "✅" "$2"; echo "PASS $1: $2"; }
note() { row "$1" "ℹ️" "$2"; echo "NOTE $1: $2"; }
fail() { row "$1" "❌" "$2"; echo "::error title=$1::$2"; exit 1; }
# shot NAME: the screen now; without Screen Recording for the runner, the desktop only.
shot() { screencapture -x "$OUT/shots/$(date +%H%M%S)-$1.png" 2> /dev/null || echo "screencapture $1 failed"; }
info() { /usr/libexec/PlistBuddy -c "Print :$1" "$APP/Contents/Info.plist" 2> /dev/null || true; }
installed() { [[ "$(info CFBundleIdentifier)" == "$NEW_ID" ]]; }
not_running() { ! pgrep -x Voxt > /dev/null; }
installer_waiting() { pgrep -x Autoupdate > /dev/null; }
update_log() { cat "$VOXT_DIR"/Logs/*.log 2> /dev/null || true; } # AppUpdateManager's lines
start_1x() { open -g --env VOXT_UPDATE_CHANNEL=beta "$APP"; } # 1.x's own variable (resolvedUpdateChannel)

# wait_for SECONDS COMMAND...: runs COMMAND until it succeeds; 1 after SECONDS.
wait_for() {
  local deadline=$((SECONDS + $1))
  shift
  until "$@"; do
    if ((SECONDS >= deadline)); then
      return 1
    else
      sleep 2
    fi
  done
}

# write_prefs YES|NO: at its next start 1.x's Sparkle checks the beta appcast
# at once; with YES it downloads the update and installs it when 1.x quits,
# with NO it shows its update window. Written while 1.x is not running.
write_prefs() {
  defaults write "$PREFS" SUEnableAutomaticChecks -bool YES
  defaults write "$PREFS" SUAutomaticallyUpdate -bool "$1"
  defaults write "$PREFS" betaUpdatesEnabled -bool YES
  defaults write "$PREFS" SUHasLaunchedBefore -bool YES # Sparkle does not check on a first start
  defaults write "$PREFS" SULastCheckTime -date "2020-01-01 00:00:00 +0000"
  killall cfprefsd 2> /dev/null || true # so that 1.x reads the file, not a cached copy of its domain
  defaults read "$PREFS" > "$OUT/voxt-1x-prefs-written.txt" 2>&1 || true
}

# click_install prints the name of the Install button it clicked in a
# window of Voxt (Sparkle's update window), or nothing.
click_install() {
  osascript - 2>> "$OUT/click-errors.txt" << 'EOF' || true
tell application "System Events"
  if not (exists process "Voxt") then return ""
  repeat with w in windows of process "Voxt"
    repeat with e in (entire contents of w)
      if class of e is button and name of e is in {"Install and Relaunch", "Install Update", "Install"} then
        set clicked to name of e
        click e
        return clicked
      end if
    end repeat
  end repeat
end tell
return ""
EOF
}

case "${1:-}" in
  start)
    mkdir -p "$OUT/shots"
    date '+%Y-%m-%d %H:%M:%S' > "$OUT/start-time"
    {
      echo "Voxt $FROM_TAG → $TO_VERSION on $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion)), $(uname -m), runner image ${ImageVersion:-unknown}"
      echo
      echo "| Check | | Detail |"
      echo "|---|---|---|"
    } > "$OUT/results.md"
    ;;

  install)
    version="${FROM_TAG#voxt-v}" downloads="$RUNNER_TEMP/voxt-1x" mount="$RUNNER_TEMP/voxt-1x-dmg"
    gh release download "$FROM_TAG" --repo "$GITHUB_REPOSITORY" --dir "$downloads" --clobber \
      --pattern "Voxt-$version.dmg" --pattern "Voxt-$version.dmg.sha256" \
      || fail install "the release $FROM_TAG has no Voxt-$version.dmg and .sha256"
    (cd "$downloads" && shasum -a 256 -c "Voxt-$version.dmg.sha256") || fail install "Voxt-$version.dmg does not match its .sha256"
    mkdir -p "$mount"
    hdiutil attach -nobrowse -readonly -mountpoint "$mount" "$downloads/Voxt-$version.dmg" > /dev/null
    [[ -d "$mount/Voxt.app" ]] || fail install "the dmg has no Voxt.app: $(find "$mount" -maxdepth 1 | tr '\n' ' ')"
    rm -rf "$APP"
    ditto "$mount/Voxt.app" "$APP"
    hdiutil detach "$mount" > /dev/null
    [[ "$(info CFBundleIdentifier) $(info CFBundleShortVersionString)" == "$LEGACY_ID $version" ]] \
      || fail install "the dmg's Voxt.app is $(info CFBundleIdentifier) $(info CFBundleShortVersionString), not $LEGACY_ID $version"
    spctl -a -vv -t exec "$APP" > "$OUT/spctl-1x.txt" 2>&1 || fail install "Gatekeeper refuses Voxt $version: $(tr '\n' ' ' < "$OUT/spctl-1x.txt")"
    pass install "Voxt $version (build $(info CFBundleVersion)) in /Applications: SHA-256 and Gatekeeper pass"
    ;;

  prepare)
    open -g "$APP"
    wait_for 60 test -d "$DATA/Library" || fail container "Voxt 1.x started, but macOS made no container $DATA in 60 s"
    wait_for 30 test -f "$VOXT_DIR/voxt.sqlite" || note container "Voxt 1.x wrote no voxt.sqlite in 30 s; the fixture goes in anyway"
    shot first-start
    pkill -x Voxt || true
    wait_for 30 not_running || fail container "Voxt 1.x still runs 30 s after SIGTERM"
    mkdir -p "$VOXT_DIR"
    rm -f "$VOXT_DIR/voxt.sqlite" "$VOXT_DIR/voxt.sqlite-wal" "$VOXT_DIR/voxt.sqlite-shm"
    cp "$FIXTURE"/voxt.sqlite* "$VOXT_DIR/"
    # The update step reads 1.x's log from here on: no line of this start.
    mv "$VOXT_DIR/Logs" "$OUT/voxt-1x-first-start-logs" 2> /dev/null || true
    pass container "the first start made $DATA; its voxt.sqlite is now $FIXTURE (3 terms)"
    ;;

  update)
    curl -fsS "$BETA_APPCAST" -o "$OUT/appcast-beta.xml" || fail appcast "could not read $BETA_APPCAST"
    grep -Eq "[>\"]${TO_VERSION//./\\.}[<\"]" "$OUT/appcast-beta.xml" \
      || fail appcast "the beta appcast does not offer $TO_VERSION: run Voxt Appcast for voxt-v$TO_VERSION (channel beta) first"
    pass appcast "the beta appcast offers $TO_VERSION"
    write_prefs YES
    start_1x
    shot started
    state=timeout deadline=$((SECONDS + 1200))
    stopped='Sparkle (did not find update|update aborted|update abort treated|failed to download update|finished update cycle with)'
    while ((SECONDS < deadline)); do
      log="$(update_log)"
      if grep -q "Sparkle finished downloading update. version=$TO_VERSION" <<< "$log"; then
        state=downloaded
        break
      elif grep -qE "$stopped" <<< "$log"; then
        state=stopped
        break
      elif not_running; then
        state=exited
        break
      else
        sleep 5
      fi
    done
    log="$(update_log)"
    grep -i sparkle <<< "$log" | tail -n 20 || true
    if grep -q "feedURL=$BETA_APPCAST" <<< "$log"; then
      pass feed "Voxt 1.x reads the beta appcast"
    else
      note feed "Voxt 1.x's log names no beta feed URL"
    fi
    case "$state" in
      downloaded) pass download "Sparkle in Voxt 1.x found and downloaded $TO_VERSION" ;;
      stopped) fail download "Sparkle in Voxt 1.x stopped: $(grep -E "$stopped" <<< "$log" | tail -n 1)" ;;
      exited) fail download "Voxt 1.x quit before Sparkle downloaded $TO_VERSION (see sparkle.log in the artifact)" ;;
      *) fail download "Sparkle in Voxt 1.x had not downloaded $TO_VERSION in 20 minutes; its last line: $(grep Sparkle <<< "$log" | tail -n 1)" ;;
    esac
    # Sparkle's installer prepares the update, then waits for 1.x to quit.
    if wait_for 120 installer_waiting; then
      note installer "Sparkle's installer (Autoupdate) is waiting for Voxt 1.x to quit"
    else
      note installer "no Autoupdate process 2 minutes after the download"
    fi
    shot downloaded

    # Way 1, a user with automatic updates: quit Voxt 1.x; Sparkle installs on quit.
    osascript -e "tell application id \"$LEGACY_ID\" to quit" > "$OUT/quit-osascript.txt" 2>&1 &
    quitter=$!
    if wait_for 60 not_running; then
      how="AppleScript quit"
    else
      kill "$quitter" 2> /dev/null || true
      note quit "osascript did not quit Voxt 1.x in 60 s ($(tr '\n' ' ' < "$OUT/quit-osascript.txt")); sending SIGTERM"
      pkill -TERM -x Voxt || true
      wait_for 30 not_running || fail quit "Voxt 1.x still runs 30 s after SIGTERM"
      how="SIGTERM"
    fi
    if wait_for 300 installed; then
      shot installed
      pass install "way 1: Voxt 1.x quit ($how) and Sparkle installed $(info CFBundleShortVersionString) on quit"
      exit 0
    else
      note install "way 1: Voxt 1.x quit ($how), but Voxt.app is still $(info CFBundleIdentifier) $(info CFBundleShortVersionString) 5 minutes later"
    fi

    # Way 2, a user who clicks: Sparkle's window, Install Update, then Install and Relaunch.
    if osascript -e 'tell application "System Events" to get name of UI element 1 of process "Dock"' > "$OUT/accessibility.txt" 2>&1; then
      note accessibility "the runner's osascript can read other apps' windows"
    else
      fail install "neither way installed $TO_VERSION: way 1 left Voxt 1.x in place, and way 2 needs Accessibility, which the runner's osascript lacks ($(tr '\n' ' ' < "$OUT/accessibility.txt"))"
    fi
    write_prefs NO
    start_1x
    clicked="" deadline=$((SECONDS + 900))
    until installed || ((SECONDS >= deadline)); do
      button="$(click_install)"
      if [[ -n "$button" ]]; then
        clicked+="$button; "
        shot clicked
      else
        :
      fi
      sleep 5
    done
    if installed; then
      shot installed
      pass install "way 2: clicked ${clicked}Sparkle installed $(info CFBundleShortVersionString)"
    elif [[ -n "$clicked" ]]; then
      fail install "neither way installed $TO_VERSION: way 1 left Voxt 1.x in place; in way 2 the clicks ($clicked) did not install it in 15 minutes"
    else
      fail install "neither way installed $TO_VERSION: way 1 left Voxt 1.x in place; in way 2 no Install button showed in Voxt's windows in 15 minutes (click-errors.txt, shots)"
    fi
    ;;

  check-app)
    failed=0
    check() { # check NAME DETAIL COMMAND...: a row; the step fails at the end
      local name="$1" detail="$2"
      shift 2
      if "$@" > /dev/null 2>&1; then
        pass "$name" "$detail"
      else
        row "$name" "❌" "$detail"
        echo "::error title=$name::$detail"
        failed=1
      fi
    }
    id="$(info CFBundleIdentifier)" version="$(info CFBundleShortVersionString)"
    check bundle-id "$id (want $NEW_ID)" test "$id" = "$NEW_ID"
    check version "$version, build $(info CFBundleVersion) (want $TO_VERSION)" test "$version" = "$TO_VERSION"
    codesign -dv --verbose=2 "$APP" > "$OUT/codesign.txt" 2>&1 || true
    check team "$(grep -E '^(Authority=Developer ID|TeamIdentifier=)' "$OUT/codesign.txt" | tr '\n' ' ')(want $TEAM)" grep -qx "TeamIdentifier=$TEAM" "$OUT/codesign.txt"
    codesign --verify --deep --strict --verbose=2 "$APP" > "$OUT/codesign-verify.txt" 2>&1 && valid=1 || valid=0
    check signature "codesign --verify --deep --strict" test "$valid" = 1
    spctl -a -vv -t exec "$APP" > "$OUT/spctl.txt" 2>&1 || true
    check gatekeeper "$(tr '\n' ' ' < "$OUT/spctl.txt")" grep -q "source=Notarized Developer ID" "$OUT/spctl.txt"
    xcrun stapler validate "$APP" > "$OUT/stapler.txt" 2>&1 && stapled=1 || stapled=0
    check staple "$(tail -n 1 "$OUT/stapler.txt")" test "$stapled" = 1
    exit "$failed"
    ;;

  check-import)
    pkill -x Voxt || true # Voxt 2, if way 2 relaunched it
    wait_for 30 not_running || true
    # Through open, not this shell: macOS lets an app read its team's other
    # containers, judging the app LaunchServices starts; a child of this
    # shell would be judged as the runner.
    open -n -W -g --env VOXTNEW_SMOKE_LEGACY=1 --stdout "$OUT/smoke.txt" --stderr "$OUT/smoke-stderr.txt" \
      "$APP" --args --smoke "$OUT/smoke" &
    opener=$!
    opener_done() { ! kill -0 "$opener" 2> /dev/null; }
    if wait_for 180 opener_done; then
      :
    else
      pkill -x Voxt || true
      fail import "the import check did not finish in 3 minutes"
    fi
    out="$(cat "$OUT/smoke.txt" 2> /dev/null || true)"
    echo "$out"
    if grep -qF "$WANT_IMPORT" <<< "$out"; then
      pass import "Voxt $TO_VERSION read Voxt 1.x's container without asking and imported ${WANT_IMPORT#smoke: legacy import: }"
    elif grep -q "not readable without asking" <<< "$out"; then
      fail import "macOS did not let Voxt $TO_VERSION read Voxt 1.x's container without asking: users get the open panel"
    elif [[ -z "$out" ]]; then
      fail import "Voxt $TO_VERSION printed nothing (smoke-stderr.txt)"
    elif ! grep -qiE "legacy|previous Voxt" <<< "$out"; then
      fail import "Voxt $TO_VERSION ran the full smoke, not the import check: it predates VOXTNEW_SMOKE_LEGACY, so the import is not checked"
    else
      fail import "the import check printed: $(grep '^smoke:' <<< "$out" | tr '\n' ' ')"
    fi
    ;;

  evidence)
    shot end
    log show --start "$(cat "$OUT/start-time")" --info --debug --style compact --predicate \
      'subsystem BEGINSWITH "org.sparkle-project" OR process == "Autoupdate" OR process == "Updater" OR process BEGINSWITH "Installer" OR subsystem == "com.voxt.Voxt" OR (subsystem == "com.apple.TCC" AND eventMessage CONTAINS[c] "voxt")' \
      > "$OUT/sparkle.log" 2>&1 || echo "log show failed"
    cp -R "$VOXT_DIR/Logs" "$OUT/voxt-1x-logs" 2> /dev/null || echo "no Voxt 1.x logs"
    plutil -convert xml1 -o "$OUT/voxt-1x-prefs.plist" "$PREFS.plist" 2> /dev/null || echo "no Voxt 1.x preferences"
    find "$DATA/Library/Caches" -ipath '*sparkle*' > "$OUT/sparkle-files.txt" 2> /dev/null || true
    cp "$APP/Contents/Info.plist" "$OUT/installed-Info.plist" 2> /dev/null || true
    cat "$OUT/results.md" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
    ;;

  *)
    echo "usage: scripts/upgrade-test.sh start|install|prepare|update|check-app|check-import|evidence" >&2
    exit 2
    ;;
esac
