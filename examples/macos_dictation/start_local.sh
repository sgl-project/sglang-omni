#!/bin/bash
# Foreground owner of only the backend processes this invocation starts.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/local_runtime.sh"
dry_run=false
asr_only=false
for option in "$@"; do
    case "$option" in
        --dry-run) dry_run=true ;;
        --asr-only) asr_only=true ;;
        -h|--help)
            printf 'Usage: bash start_local.sh [--dry-run] [--asr-only]\n'
            printf '启动或复用默认本机服务，检查模型后打开应用。Ctrl+C 只停止本次启动的服务。\n'
            exit 0 ;;
        *) die "未知参数：$option" ;;
    esac
done
if "$dry_run"; then
    note '仅展示计划，不启动服务、不下载模型、不打开应用。'
    show_command env SGLANG_USE_MLX=1 'DYLD_LIBRARY_PATH=<ffmpeg@7>/lib' "$dictation_venv/bin/sgl-omni" serve \
        --model-path "$asr_repository" --model-name "$asr_model" --asr.engine.max_running_requests 1 --host 127.0.0.1 --port 8000
    if ! "$asr_only"; then
        show_command env OLLAMA_HOST=127.0.0.1:11434 ollama serve
        note "检查已下载的纠正模型：$ollama_model"
    fi
    show_command open "$dictation_app"
    note '启动前检查已有服务和模型。实际 ASR 使用已下载的本地快照路径。'
    exit 0
fi

check_platform
check_python
[[ -d "$dictation_app" ]] || die '应用尚未构建，请先运行 install_local.sh。'
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$dictation_logs"

if asr_ready; then
    note '复用已运行的 Omni ASR。'
else
    port_in_use 8000 && die '端口 8000 已占用，但服务未就绪或模型不匹配；请检查原服务。'
    configure_mlx
    [[ -x "$dictation_venv/bin/sgl-omni" ]] || die '缺少 sgl-omni 命令，请先运行 install_local.sh。'
    model_path="$(asr_snapshot offline)" || die '找不到已缓存的 ASR 模型，请先运行 install_local.sh。'
    [[ -d "$model_path" ]] || die 'ASR 模型缓存路径无效。'
    "$dictation_venv/bin/sgl-omni" serve --model-path "$model_path" --model-name "$asr_model" \
        --asr.engine.max_running_requests 1 --host 127.0.0.1 --port 8000 >"$dictation_logs/asr.log" 2>&1 &
    pid=$!
    owned_pids+=("$pid")
    wait_ready "$pid" asr_ready 'Omni ASR' "$dictation_logs/asr.log"
fi
if ! "$asr_only"; then
    ensure_ollama
    has_model ollama "$ollama_model" "$ollama_url/api/tags" || die '缺少默认纠正模型，请先运行 install_local.sh --skip-runtime。'
fi
open "$dictation_app"
note '服务已就绪，已打开 Omni 听写。应用窗口激活时按 ⌘, 进入设置。'
note '脚本使用默认模型和端口；应用保存的自定义配置不会被覆盖。'
if [[ ${#owned_pids[@]} -eq 0 ]]; then
    note '所有服务原本已运行，本次无需管理后台进程。'
    exit 0
fi
note "请保持此终端打开；Ctrl+C 只停止本次启动的服务。日志：$dictation_logs"
while true; do
    for pid in "${owned_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || die "后台服务已退出，请查看 $dictation_logs"
    done
    sleep 2
done
