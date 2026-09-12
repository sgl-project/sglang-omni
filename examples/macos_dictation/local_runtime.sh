#!/bin/bash
# Shared defaults for the source installer and foreground launcher. No side effects on source.
dictation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dictation_repo="$(cd "$dictation_dir/../.." && pwd)"
dictation_venv="${SGLANG_OMNI_VENV:-$dictation_repo/.venv-apple}"
dictation_python="$dictation_venv/bin/python"
dictation_app="$dictation_dir/.build/client/OmniDictation.app"
dictation_logs="$dictation_dir/.build/runtime-logs"
asr_repository="mlx-community/Qwen3-ASR-0.6B-4bit"
asr_model="Qwen/Qwen3-ASR-0.6B"
ollama_model="openbmb/minicpm5-2b:q4_K_M"
asr_url="http://127.0.0.1:8000"
ollama_url="http://127.0.0.1:11434"
owned_pids=()

die() { printf '错误：%s\n' "$*" >&2; exit 1; }
note() { printf '[Omni 听写] %s\n' "$*"; }
show_command() {
    local argument
    printf '  '
    # Bash 3.2 printf %q can split UTF-8 bytes in Chinese application paths.
    for argument in "$@"; do
        printf "'"
        while [[ "$argument" == *"'"* ]]; do
            printf '%s' "${argument%%\'*}" "'\\''"
            argument="${argument#*\'}"
        done
        printf "%s' " "$argument"
    done
    printf '\n'
}

check_platform() {
    [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die '需要 Apple Silicon Mac。'
    local version
    version="$(sw_vers -productVersion)"
    [[ "${version%%.*}" -ge 14 ]] || die '需要 macOS 14 或更新版本。'
    [[ "$dictation_venv" == /* ]] || die 'SGLANG_OMNI_VENV 必须是绝对路径。'
}

find_brew() {
    if [[ -x /opt/homebrew/bin/brew ]]; then brew_bin=/opt/homebrew/bin/brew
    else brew_bin="$(command -v brew || true)"; fi
    [[ -n "$brew_bin" ]] || die '请先从 https://brew.sh 安装 Homebrew，再运行此脚本。'
}

find_ollama() {
    ollama_bin="$(command -v ollama || true)"
    local candidate
    for candidate in /opt/homebrew/bin/ollama /Applications/Ollama.app/Contents/Resources/ollama \
        "$HOME/Applications/Ollama.app/Contents/Resources/ollama"; do
        if [[ -z "$ollama_bin" && -x "$candidate" ]]; then ollama_bin="$candidate"; fi
    done
    [[ -n "$ollama_bin" ]]
}

check_python() {
    [[ -x "$dictation_python" ]] || die '缺少 Python 环境，请先运行 install_local.sh。'
    "$dictation_python" -c 'import sys; assert sys.version_info[:2] == (3, 12) and sys.prefix != sys.base_prefix, "需要独立的 Python 3.12 环境"'
}

configure_mlx() {
    find_brew
    local ffmpeg_prefix
    ffmpeg_prefix="$("$brew_bin" --prefix ffmpeg@7)"
    [[ -d "$ffmpeg_prefix/lib" ]] || die '缺少 ffmpeg@7，请先运行 install_local.sh。'
    export DYLD_LIBRARY_PATH="$ffmpeg_prefix/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export SGLANG_USE_MLX=1
}

# Use the same HF cache for installation and startup. Startup never downloads weights.
asr_snapshot() {
    "$dictation_python" - "$asr_repository" "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], local_files_only=sys.argv[2] == "offline"))
PY
}

local_get() { curl --noproxy '*' --fail --silent --show-error --max-time 3 "$1"; }

has_model() {
    local kind="$1" model="$2" url="$3"
    local_get "$url" 2>/dev/null | "$dictation_python" -c '
import json, sys
try:
    payload = json.load(sys.stdin)
    key, field = ("data", "id") if sys.argv[1] == "asr" else ("models", "name")
    sys.exit(0 if any(item.get(field) == sys.argv[2] for item in payload[key]) else 1)
except (ValueError, KeyError, TypeError, AttributeError):
    sys.exit(1)
' "$kind" "$model"
}

ollama_ready() {
    local_get "$ollama_url/api/tags" 2>/dev/null | "$dictation_python" -c '
import json, sys
try:
    sys.exit(0 if isinstance(json.load(sys.stdin).get("models"), list) else 1)
except (ValueError, AttributeError):
    sys.exit(1)
'
}

asr_ready() {
    local_get "$asr_url/health" 2>/dev/null | "$dictation_python" -c '
import json, sys
try:
    sys.exit(0 if json.load(sys.stdin).get("status") == "healthy" else 1)
except (ValueError, AttributeError):
    sys.exit(1)
' || return 1
    has_model asr "$asr_model" "$asr_url/v1/models"
}

port_in_use() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

cleanup_owned() {
    local pid
    for pid in ${owned_pids[@]+"${owned_pids[@]}"}; do
        if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
        wait "$pid" 2>/dev/null || true
    done
}

wait_ready() {
    local pid="$1" check="$2" label="$3" log_path="$4"
    local deadline=$((SECONDS + 300))
    while ((SECONDS < deadline)); do
        kill -0 "$pid" 2>/dev/null || die "$label 已退出，请查看 $log_path"
        if "$check"; then return; fi
        sleep 1
    done
    die "$label 启动超时，请查看 $log_path"
}

ensure_ollama() {
    if ollama_ready; then note '复用已运行的 Ollama。'; return; fi
    port_in_use 11434 && die '端口 11434 已占用，但不是可用的 Ollama；请检查原服务。'
    find_ollama || die '未找到 Ollama，请先运行 install_local.sh。'
    mkdir -p "$dictation_logs"
    OLLAMA_HOST=127.0.0.1:11434 "$ollama_bin" serve >"$dictation_logs/ollama.log" 2>&1 &
    local pid=$!
    owned_pids+=("$pid")
    wait_ready "$pid" ollama_ready Ollama "$dictation_logs/ollama.log"
}

link_app() {
    local destination="$1"
    if [[ -L "$destination" && "$(readlink "$destination")" == "$dictation_app" ]]; then return; fi
    [[ ! -e "$destination" && ! -L "$destination" ]] || die "入口已存在且不属于本项目，不会覆盖：$destination"
    mkdir -p "$(dirname "$destination")"
    ln -s "$dictation_app" "$destination"
}
