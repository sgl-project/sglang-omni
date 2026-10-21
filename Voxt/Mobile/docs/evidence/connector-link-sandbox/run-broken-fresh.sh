#!/bin/bash
# The broken-seal bundle against a server that never saw the good one.
APP="/Applications/Voxt LinkCheck.app"
SP=""
trap '[ -n "$SP" ] && kill "$SP" 2>/dev/null; rm -rf "$APP"; echo "cleanup: $( [ -e "$APP" ] && echo still-there || echo removed )"' EXIT
[ -e "$APP" ] && { echo "refusing: exists"; exit 1; }
ditto "LinkCheck-stage/Voxt LinkCheck.app" "$APP"
mkdir -p "$APP/Contents/Resources" && echo x > "$APP/Contents/Resources/added.txt"
./LinkServer.app/Contents/MacOS/LinkServer > server.log 2>&1 & SP=$!
sleep 2
perl -e 'alarm 15; exec { $ARGV[0] } @ARGV' "$APP/Contents/MacOS/linkclient" 2>&1 | tail -1
