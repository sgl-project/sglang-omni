#!/bin/bash
# Focused review regressions; uses offline fixtures and a disposable loopback server.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
bash "$project_dir/local_setup_test.sh"
bash "$project_dir/verify_all_test.sh" PolishFidelity Personalization ServiceConfiguration ClientComposition HTTPTransport
