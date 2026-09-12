# 安装与启动 Omni 听写

这些脚本把 SGLang、SGLang-Omni、Ollama、模型下载和听写应用构建串起来。
它们是本仓库的源码安装工具，不包含模型权重，也不是独立分发的安装包。

## 首次安装

需要 Apple Silicon Mac、macOS 14 或更新版本、[Homebrew](https://brew.sh)
以及 Xcode Command Line Tools。缺少命令行工具时先运行 `xcode-select --install`。
脚本不会代替用户安装 Homebrew，也不会自动授予系统权限。

进入已经 clone 的 `sglang-omni` 仓库后运行：

```bash
# 可选：先查看会执行哪些步骤，不安装或下载
bash examples/macos_dictation/install_local.sh --dry-run

# 退出正在运行的 Omni 听写和 ASR 服务后，执行完整安装
bash examples/macos_dictation/install_local.sh
```

完整安装会：

1. 调用仓库根目录的 `install.sh`，在 `.venv-apple` 中安装 Python 3.12、
   SGLang 的 Apple Silicon 依赖与本仓库的 SGLang-Omni。版本跟随根安装器，
   不另行维护一套依赖，也不安装到系统 Python 或当前 conda 环境。
2. 检查 MLX Metal 与 FFmpeg 加载，下载或复用 Hugging Face 的 ASR 模型缓存。
3. 查找已有 Ollama；缺少时通过 `brew install ollama` 安装。按需临时启动本机
   Ollama，下载默认纠正模型，已有同名模型则保留。临时服务在安装结束后退出。
4. 调用 `build_client.sh` 构建应用，并在桌面和 `~/Applications` 创建
   **Omni 听写.app** 快捷入口。已有同目标入口可以复用，其他同名文件不会被覆盖。

**如果你之前已跑通 Omni，使用下面的命令保留现有 Python 环境，跳过第一步：**

```bash
bash examples/macos_dictation/install_local.sh --skip-runtime
```

仍需先退出听写应用，因为构建会更新应用文件。已有 ASR 可继续运行。
不要删除或移动仓库：快捷入口指向仓库里的 `.build/client/OmniDictation.app`。

## 日常启动

```bash
bash examples/macos_dictation/start_local.sh
```

该命令检查默认端口和模型，复用已有服务，按需启动缺少的服务，再打开听写应用。
启动不会安装依赖或主动下载模型；找不到缓存时会提示先运行安装脚本。
ASR 和脚本启动的 Ollama 都只监听 `127.0.0.1`。

如果脚本启动了服务，请保持终端打开。按 `Ctrl+C` 只停止**本次脚本启动的服务**，
不会关闭原先已经运行的 ASR 或 Ollama，也不会退出听写应用。
若全部服务原本已运行，脚本打开应用后直接退出。

服务日志位于 `examples/macos_dictation/.build/runtime-logs/`，不提交到 Git。
这是后端进程的标准输出和错误输出，内容由后端决定；客户端仍不保存听写历史。

## 默认模型与地址

| 用途 | 模型 | 地址 |
| --- | --- | --- |
| ASR 下载仓库 | `mlx-community/Qwen3-ASR-0.6B-4bit` | Hugging Face 缓存 |
| ASR 服务名称 | `Qwen/Qwen3-ASR-0.6B` | `http://127.0.0.1:8000` |
| 文本纠正 | `openbmb/minicpm5-2b:q4_K_M` | `http://127.0.0.1:11434` |

这些默认值与应用一致。脚本不会覆盖应用保存的快捷键、个人背景、模型配置或开关。
如果之前更改了应用中的模型或地址，请确保它们与实际服务匹配。
模型仓库的默认 revision 和 Ollama tag 可能更新；这套脚本不保证跨日期下载相同权重。

只使用 ASR、不安装或启动 Ollama：

```bash
bash examples/macos_dictation/install_local.sh --asr-only
bash examples/macos_dictation/start_local.sh --asr-only
```

应用设置中的 **轻度整理** 也应关闭；脚本不会替你修改开关。
`--asr-only` 可以和安装脚本的 `--skip-runtime` 一起使用。

如果原来使用的是其他虚拟环境，对安装和启动都传入同一个绝对路径：

```bash
SGLANG_OMNI_VENV=/absolute/path/to/venv bash examples/macos_dictation/install_local.sh --skip-runtime
SGLANG_OMNI_VENV=/absolute/path/to/venv bash examples/macos_dictation/start_local.sh
```

## 打开设置与授权

双击桌面的 **Omni 听写**；应用窗口激活时按 `⌘,` 打开设置，也可使用顶部菜单中的
**Omni 听写 → 设置…**。当前应用只显示菜单栏图标，不显示底部 Dock 图标。
如果首次启动只出现菜单栏图标，再双击一次应用入口可打开结果窗口。

首次录音需要允许麦克风访问。自动回填需要在
**系统设置 → 隐私与安全性 → 辅助功能** 中允许 Omni 听写，并回到应用刷新。
本地构建使用临时签名，重新构建后可能需要移除旧授权条目，再添加新构建的应用。

## 单独使用原始命令

以下是脚本整合的主要命令，方便理解和排查：

```bash
# SGLang、Omni、MLX 和 FFmpeg：复用仓库 Apple Silicon 安装器
bash install.sh
source .venv-apple/bin/activate

# ASR 模型
hf download mlx-community/Qwen3-ASR-0.6B-4bit

# ASR 服务：在一个终端中保持运行
export SGLANG_USE_MLX=1
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
sgl-omni serve \
  --model-path mlx-community/Qwen3-ASR-0.6B-4bit \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --host 127.0.0.1 --port 8000
```

```bash
# 缺少 Ollama 时安装；已有可用版本时跳过
brew install ollama

# 没有运行中的 Ollama 时，在另一个终端保持运行
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

```bash
# Ollama 已启动后，在新终端下载模型并构建应用
OLLAMA_HOST=127.0.0.1:11434 ollama pull openbmb/minicpm5-2b:q4_K_M
bash examples/macos_dictation/build_client.sh
open examples/macos_dictation/.build/client/OmniDictation.app
```

依赖安装规范见 [Apple Silicon 安装说明](../../docs/get_started/installation.md#macos-apple-silicon)，
ASR 启动规范见 [Qwen3-ASR MLX](../../docs/cookbook/qwen3_asr.md#apple-silicon-mlx)，
Ollama CLI 安装方式见 [Homebrew formula](https://formulae.brew.sh/formula/ollama)。

## 验证脚本

```bash
bash examples/macos_dictation/local_setup_test.sh
```

测试使用临时目录和模拟服务响应，不安装软件、不下载模型、不打开应用。
它覆盖预览模式、模型检查、已有服务复用、离线快照查找、入口冲突和进程清理。
完整的新机器安装仍需要真实网络和 Apple Silicon 环境验证。
