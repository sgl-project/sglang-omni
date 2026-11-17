#!/bin/sh
set -eu

exec /usr/bin/env python3 "$(dirname "$0")/profile_loop.py" "$@"
