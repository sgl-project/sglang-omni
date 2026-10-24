#!/bin/bash
# One run: the sandboxed link server, the signed client bundle in /Applications (accepted), the same
# bundle with a broken seal (rejected), and the client outside /Applications (rejected). The bundle
# in /Applications exists only while this script runs.
set -u
APP="/Applications/Voxt LinkCheck.app"
SP=""
cleanup() { [ -n "$SP" ] && kill "$SP" 2>/dev/null; rm -rf "$APP"; }
trap cleanup EXIT
[ -e "$APP" ] && { echo "refusing: $APP already exists"; exit 1; }
./LinkServer.app/Contents/MacOS/LinkServer > server.log 2>&1 & SP=$!
sleep 2
ditto "LinkCheck-stage/Voxt LinkCheck.app" "$APP"
echo "== client in /Applications, signed"
perl -e 'alarm 15; exec { $ARGV[0] } @ARGV' "$APP/Contents/MacOS/linkclient" 2>&1; echo "exit=$?"
echo "== same bundle, seal broken by an added resource"
mkdir -p "$APP/Contents/Resources" && echo x > "$APP/Contents/Resources/added.txt"
perl -e 'alarm 15; exec { $ARGV[0] } @ARGV' "$APP/Contents/MacOS/linkclient" 2>&1; echo "exit=$?"
echo "== client outside /Applications"
perl -e 'alarm 15; exec { $ARGV[0] } @ARGV' ./linkclient 2>&1; echo "exit=$?"
