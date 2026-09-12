#!/bin/bash
# Install the local speech stack and build the app; reuse Omni's Apple installer.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/local_runtime.sh"
dry_run=false
skip_runtime=false
asr_only=false
for option in "$@"; do
    case "$option" in
        --dry-run) dry_run=true ;;
        --skip-runtime) skip_runtime=true ;;
        --asr-only) asr_only=true ;;
        -h|--help)
            printf 'Usage: bash install_local.sh [--dry-run] [--skip-runtime] [--asr-only]\n'
            printf '安装依赖、下载默认模型、构建应用并创建桌面入口。\n'
            printf '已有可用的 Omni Python 环境时，用 --skip-runtime 跳过根目录安装器。\n'
            exit 0 ;;
        *) die "未知参数：$option" ;;
    esac
done

if "$dry_run"; then
    note "仅展示计划，不安装、不下载、不修改文件。Python 环境：$dictation_venv"
    if ! "$skip_runtime"; then show_command env -u SGLANG_OMNI_PROJECT_DIR SGLANG_OMNI_VENV="$dictation_venv" bash "$dictation_repo/install.sh"; fi
    show_command "$dictation_venv/bin/hf" download "$asr_repository"
    if ! "$asr_only"; then
        printf '  检查已有 Ollama；缺少时 brew install ollama；按需临时启动本机服务。\n'
        show_command env OLLAMA_HOST=127.0.0.1:11434 ollama pull "$ollama_model"
    fi
    show_command bash "$dictation_dir/build_client.sh"
    show_command ln -s "$dictation_app" "$HOME/Applications/Omni 听写.app"
    show_command ln -s "$dictation_app" "$HOME/Desktop/Omni 听写.app"
    exit 0
fi

check_platform
xcrun --find swift >/dev/null 2>&1 || die '请先运行 xcode-select --install 安装命令行工具。'
if pgrep -x OmniDictation >/dev/null 2>&1; then
    die '请先从 Omni 听写菜单退出应用，再运行安装脚本，避免替换正在运行的版本。'
fi
if ! "$skip_runtime"; then
    port_in_use 8000 && die '请先停止 8000 端口的 ASR，或使用 --skip-runtime 保留已安装环境。'
    note "使用根目录安装器。目标环境：$dictation_venv；当前 conda 环境：${CONDA_DEFAULT_ENV:-无}。"
    env -u SGLANG_OMNI_PROJECT_DIR SGLANG_OMNI_VENV="$dictation_venv" bash "$dictation_repo/install.sh"
fi
check_python
configure_mlx
"$dictation_python" -c 'import mlx.core as mx; from torchcodec.decoders import AudioDecoder; assert mx.metal.is_available(), "MLX Metal 不可用"'
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

note "下载或复用 ASR 模型：$asr_repository"
asr_snapshot online
if ! "$asr_only"; then
    if ! find_ollama; then "$brew_bin" install ollama; fi
    find_ollama || die 'Ollama 安装后仍无法找到命令，请检查 Homebrew。'
    ensure_ollama
    if has_model ollama "$ollama_model" "$ollama_url/api/tags"; then
        note "复用已下载的模型：$ollama_model"
    else
        OLLAMA_HOST=127.0.0.1:11434 "$ollama_bin" pull "$ollama_model"
    fi
    has_model ollama "$ollama_model" "$ollama_url/api/tags" || die 'Ollama 模型下载未完成。'
fi
bash "$dictation_dir/build_client.sh"
link_app "$HOME/Applications/Omni 听写.app"
link_app "$HOME/Desktop/Omni 听写.app"
note '安装完成。麦克风和辅助功能权限仍需在 macOS 中手动允许。'
note '日常使用以下命令启动；无需再次安装或下载：'
if "$asr_only"; then show_command bash "$dictation_dir/start_local.sh" --asr-only
else show_command bash "$dictation_dir/start_local.sh"; fi
