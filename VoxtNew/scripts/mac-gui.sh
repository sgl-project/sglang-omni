#!/usr/bin/env bash
# Runs a command on this Mac in the logged-in user's GUI (Aqua) session,
# from ssh: as a one-off LaunchAgent (launchctl bootstrap gui/<uid>), with
# its output streamed here and its exit code returned.
#
# Why: over ssh, codesign cannot use a Developer ID key of the login
# keychain (errSecInternalComponent: the ssh session's security session
# has the keychain locked). In the GUI session of the user logged in at
# the screen, the keychain is unlocked and codesign signs without a
# prompt (mac-air, 10-17: both identities of the team RY2TQTYSKQ, which
# signed VoxtNew until 10-19). Nothing asks for a password; nothing
# appears on the screen. It needs the same user logged in at the Mac's
# screen (a locked screen is fine).
#
#   scripts/mac.sh 'scripts/mac-gui.sh scripts/release.sh --runtimes $HOME/voxtnew-runtime'
#
# The job starts in the current directory with this shell's PATH, HOME and
# Go/proxy settings, and these variables when set: VOXTNEW_SIGN_IDENTITY,
# VOXTNEW_UPDATE_KEY, MYGO_UPDATER_PRIVATE_KEY, ACTNOW_INGEST_TOKEN,
# ACTNOW_URL, MAC_GUI_PASS (more names, space-separated). They reach it
# through a file in a private (0700) temporary directory that the job
# deletes as it starts; no value is printed or put on a command line.
# Ctrl-C (or the ssh session ending) stops the job.
set -euo pipefail

(($#)) || { echo "usage: scripts/mac-gui.sh COMMAND [ARGS…]" >&2; exit 2; }
uid="$(id -u)"
if [[ "$(stat -f %Su /dev/console)" != "$(id -un)" ]]; then
  echo "mac-gui.sh: $(id -un) is not logged in at the Mac's screen: no GUI session to run in" >&2
  exit 1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/voxtnew-gui.XXXXXX")"
chmod 700 "$work"
label="dev.hehehai.voxtnew.gui.$$"
job_pid=""
cleanup() {
  launchctl bootout "gui/$uid/$label" >/dev/null 2>&1 || true
  [[ -n "$job_pid" ]] && kill -TERM -- "-$job_pid" 2>/dev/null || true
  rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

names="PATH HOME USER LANG GOTOOLCHAIN GOPROXY GOFLAGS GOPATH http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
  VOXTNEW_SIGN_IDENTITY VOXTNEW_UPDATE_KEY MYGO_UPDATER_PRIVATE_KEY ACTNOW_INGEST_TOKEN ACTNOW_URL ${MAC_GUI_PASS:-}"
umask 077
{
  for n in $names; do
    if [[ -n "${!n+x}" ]]; then
      printf 'export %s=%q\n' "$n" "${!n}"
    fi
  done
  printf 'cd %q\n' "$PWD"
} >"$work/env.sh"
printf '%q ' "$@" >"$work/cmd"
cat >"$work/run.sh" <<'EOF'
#!/bin/bash
work="$(dirname "$0")"
. "$work/env.sh"
rm -f "$work/env.sh"
echo $$ >"$work/pid"
eval "$(cat "$work/cmd")" >>"$work/log" 2>&1 </dev/null
echo $? >"$work/exit"
EOF
chmod 700 "$work/run.sh"
: >"$work/log"
cat >"$work/job.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$work/run.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>LimitLoadToSessionType</key><string>Aqua</string>
  <key>ProcessType</key><string>Interactive</string>
</dict></plist>
EOF
launchctl bootstrap "gui/$uid" "$work/job.plist"

# Stream the log until the job writes its exit code.
offset=0
while :; do
  [[ -z "$job_pid" && -f "$work/pid" ]] && job_pid="$(cat "$work/pid")"
  size="$(stat -f %z "$work/log")"
  if ((size > offset)); then
    tail -c "+$((offset + 1))" "$work/log" | head -c "$((size - offset))"
    offset=$size
  fi
  if [[ -f "$work/exit" ]]; then
    size="$(stat -f %z "$work/log")"
    ((size > offset)) && tail -c "+$((offset + 1))" "$work/log"
    job_pid=""
    exit "$(cat "$work/exit")"
  fi
  if ! launchctl print "gui/$uid/$label" >/dev/null 2>&1; then
    echo "mac-gui.sh: the job is gone without an exit code" >&2
    exit 1
  fi
  sleep 1
done
