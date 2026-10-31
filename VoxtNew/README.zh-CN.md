# VoxtNew

[English](README.md) · **简体中文**

**只用本地模型**的 macOS 语音输入：在任意 App 里按下快捷键说话，文字出现在光标处。

- **转录**：把说的话输入到光标处，说话时实时显示文字，可以再由语言模型整理（去掉口头语和说了一半的话，补上标点）。
- **翻译**：把选中的文字，或者说的话，译成另一种语言。
- **转写**：说一句怎么改（“短一点”“正式一点”）来改写选中的文字，或者按语音要求写新内容；在答案卡片里可以继续追问。
- **应用增强**：把 App 和网站分组，每组有自己的场景提示词和插入后的按键（聊天时口语一些，插入后按回车发送）。
- **词典**：人名和专业术语，语音识别模型提前知道，语言模型照你的写法输出。

还有历史、浮窗、快捷键、菜单栏、设置引导、设置和更新。

**快捷键**（在各功能页里可以改）：单按**右 Option** 转录，右 Option 加 Shift 翻译，Control 加右 Option 转写；Control-Option-V 粘贴上一条结果。VoxtNew 是 [ActNow](https://actnow.dev) 家族的 app：这套按键把 fn 留给 Auralink，两个 app 可以同时使用。

**官网** [actnow.dev/voxtnew](https://actnow.dev/voxtnew) · **下载** [actnow.dev/download/voxtnew](https://actnow.dev/download/voxtnew)（需要先登录）· 更新也来自 actnow.dev。

> **状态：0.1.0**（第一个 beta：`0.1.0-beta.1`）。上面的功能都已在本地模型上可用，见 [CHANGELOG.md](CHANGELOG.md)。

## 在你的 Mac 上运行什么

所有听到、读到你的话的部分都在本机：

- **语音识别**和**声音检测**跑在 sglang-omni 的原生 MLX 服务上（[`sglang_omni_mlx/native`](../sglang_omni_mlx/native)，C++，不用 Python），每个模型一个进程，用到时启动，10 分钟不用就停（本地模型页的“高级”里可以改）。
- **语言模型**跑在 VoxtNew 自己的 Swift 进程 `voxtnew-llm` 里（MLX，其中一个模型用 llama.cpp），5 分钟不用就停（同样可以改）。
- 一个小的 Swift 辅助进程负责要用苹果框架的事：麦克风、全局快捷键、辅助功能、粘贴。

VoxtNew 只提供少数几个为它调好的模型，在你需要时从 Hugging Face（或镜像）下载，每个文件都按固定的 SHA-256 校验：

| 模型 | 擅长什么 | 下载大小 |
|---|---|---|
| 精准语音识别（推荐） | 最准确；口音、方言、人名、一句话里夹几种语言；实时显示文字 | 2.5 GB |
| 快速语音识别 | 更快、更省内存，适合在安静环境里清晰地说话 | 713 MB |
| 均衡语言模型（推荐） | 整理、翻译和改写：质量和速度最均衡 | 3.3 GB |
| 快速语言模型 | 回答更快、内存更少；简短的整理和简单的改写 | 1.7 GB |
| 翻译专用模型 | 专门做翻译，快而自然；只有翻译用它 | 1.1 GB |

声音检测（1.2 MB）随第一个语音识别模型一起安装。语音识别模型支持 30 种语言和 22 种中文方言；通用语言模型中英文最好。

**需要** Apple 芯片、macOS 15 及以上的 Mac。app 约 280 MB；推荐的两个模型共需下载 5.8 GB（五个模型全装 9.4 GB）。两个推荐模型都加载时约占 7 GB 内存，16 GB 内存的 Mac 比较从容；8 GB 内存的 Mac 请选快速的模型。

## 隐私

音频和文字不会离开这台 Mac：识别、整理、翻译和改写都在本机完成。VoxtNew 没有自己的服务器，不需要账号，也不做统计。它只在两种情况下联网：下载你选择的模型（从 Hugging Face，走系统代理；连不上时用 hf-mirror.com），以及在 [actnow.dev](https://actnow.dev) 检查更新。历史、可选保存的录音、词典和设置都在 `~/Library/Application Support/VoxtNew/`。

## 从源码构建

需要 Go 1.27.1（`GOTOOLCHAIN=auto` 会自动下载）；构建 app 需要 Apple 芯片、macOS 15 及以上的 Mac，Xcode（编译辅助进程和 `voxtnew-llm`），以及 [uv](https://docs.astral.sh/uv/) 或 Python 3（Omni 运行时的构建工具：`scripts/build-omni.sh` 用固定版本的 MLX 编译 `sglang_omni_mlx/native`，CMake 会从 GitHub 下载 C++ 依赖，需要时走 shell 的 `https_proxy`）。在 `VoxtNew/` 下执行：

```sh
scripts/build-helper.sh              # Swift 辅助进程，输出到 resources/darwin-arm64/helper/
scripts/build-omni.sh                # Omni 运行时（语音识别模型），输出到 resources/darwin-arm64/omni/
scripts/build-llm.sh                 # voxtnew-llm（语言模型），输出到 resources/darwin-arm64/llm/
go tool mygo dev                     # 热重载运行
go tool mygo build                   # 生成 build/darwin-arm64/VoxtNew.app 和 dmg
```

`go tool mygo build` 和 `mygo dev` 之前必须先编好辅助进程和 `voxtnew-llm`：`mygo.json` 给这两者指定了 entitlement，缺了就会失败。没有 Omni 运行时 app 也能运行，但不能识别语音（也可以把 `VOXTNEW_OMNI_RUNTIME` 指向 `scripts/build-omni.sh --out DIR` 编出的运行时；`VOXTNEW_LLM` 对应 `scripts/build-llm.sh --out DIR`）。

**发布版本**在 Mac 上用一条命令构建：`scripts/release.sh` 编译这三样，打包、签名 app 并检查结果（`--runtimes DIR` 直接用编好的运行时；有 779M533DL5 团队（和家族其他 app 一样）的 Developer ID 时，`--notarize` 和 `--publish` 公证并发布到 actnow.dev；ssh 里经 `scripts/mac-gui.sh` 运行）。见 [docs/release.md](docs/release.md)。

在 Linux 上，`scripts/mac.sh <命令>` 会把项目同步到 Mac（`MAC_HOST` 默认 `mac-air`，`MAC_DIR` 默认 `code/VoxtNew`）并在那里执行，例如 `scripts/mac.sh 'scripts/build-helper.sh && scripts/build-llm.sh && go tool mygo build'`。

<details>
<summary>开发时的签名（重新构建后保留权限）</summary>

macOS 按签名记住麦克风和辅助功能的授权。ad hoc 签名的构建（默认）每次重新构建签名都会变，所以重建以后 macOS 会忘掉授权：麦克风变成**拒绝且不再询问**，辅助功能在系统设置里看着是开的，却不起作用。辅助进程的授权就是 app 的（macOS 把 VoxtNew.app 当作它的责任进程），所以它也一样。解决办法：

- **只改辅助进程时**：保留一个构建好的 app，用 bundle 外面的辅助进程运行它。辅助进程放在哪里都继承 app 的授权，重编它不受影响：
  ```sh
  scripts/build-helper.sh --out ~/voxtnew-dev
  open --env VOXTNEW_HELPER=$HOME/voxtnew-dev/voxtnew-helper build/darwin-arm64/VoxtNew.app
  ```
- **重新开始**：`tccutil reset Microphone dev.hehehai.voxtnew`（`Accessibility` 同理），再打开 app 重新授权。如果旧的辅助功能记录开着却不起作用，在系统设置里用 − 删掉再重新授权。
- **用固定的身份签名**：Apple Development 证书能让授权在重建后保留：`go tool mygo build -sign "Apple Development: …"`（或者在本地的 mygo.json 里写 `macos.signingIdentity`；`scripts/build-helper.sh` 和 `scripts/build-llm.sh` 用 `VOXTNEW_SIGN_IDENTITY`）。通过 ssh 时，`codesign` 会报 `errSecInternalComponent`，直到允许不经图形界面使用私钥：在 Mac 的“终端”里执行一次 `security set-key-partition-list -S apple-tool:,apple: -s ~/Library/Keychains/login.keychain-db`。

在自己编的版本和发布版（Developer ID）之间切换，也要这样 reset 一次（[docs/release.md](docs/release.md) §4）。下载好的模型会保留。

</details>

## 开发

VoxtNew 是基于 [MyGo](https://github.com/egoist/mygo) 的 Go 应用（用 Go 写原生 macOS 界面，不用 cgo）。由 Go 做决定；辅助进程和模型运行时照指令办事，通过 stdin/stdout 上的 JSON 行（辅助进程、`voxtnew-llm`）或回环地址上的 HTTP（Omni 服务）通信。除了 Swift 代码，全部是普通的 Go，在 Linux 上用辅助进程和运行时的假实现测试。在 `VoxtNew/` 下执行：

```sh
CGO_ENABLED=0 go test ./...          # 全部测试，Linux 上可跑
go tool mygo vet .                   # go vet 和 MyGo 的 UI 生命周期检查
VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .   # 每个页面截图，浅色/深色、英文/简体中文
cd helper && swift test              # Swift 辅助进程的测试（在 Mac 上），用同一套协议 fixture
scripts/build-llm.sh --test          # voxtnew-llm 的 Swift 测试（在 Mac 上）
scripts/mac.sh scripts/mac-e2e.sh    # Mac 上的无界面测试：真实的辅助进程，假的模型
scripts/mac.sh scripts/mac-regress.sh   # Mac 上的真实模型：运行时、评测、打包好的 app
```

存储的测试用 MyGo 的 SQLite 插件，它会下载一次自己的库（按 SHA-256 校验），或者用 `MYGO_SQLITE_LIBRARY`。在 Linux 上，浮窗中文文字的像素检查需要 CJK 字体（Ubuntu：`fonts-noto-cjk`）。

- [docs/architecture.md](docs/architecture.md)：包结构、线程、会话、运行时、磁盘上的数据、测试。
- [docs/plan.md](docs/plan.md)：VoxtNew 是什么，以及里程碑。
- [docs/behavior.md](docs/behavior.md)：VoxtNew 和 Voxt 行为上的不同，以及原因。
- [docs/llm-models.md](docs/llm-models.md)：语言模型、它们的参数和实测。
- [docs/eval.md](docs/eval.md)：模型在评测集上的质量，以及由此做的决定。
- [docs/helper-protocol.md](docs/helper-protocol.md)、[docs/llm-protocol.md](docs/llm-protocol.md)：辅助进程和 `voxtnew-llm` 的协议。
- [docs/manual-test.md](docs/manual-test.md)：在 Mac 上手动验证的清单，以及自动的真机测试。
- [docs/release.md](docs/release.md)：构建、签名、公证和发布，以及更新。
- [AGENTS.md](AGENTS.md)：改动规则（文字、线程、协议）。
- [CHANGELOG.md](CHANGELOG.md)：各版本的变化。[CONTRIBUTING.md](CONTRIBUTING.md)：怎么提交改动。

VoxtNew 由 [Auralink](https://github.com/hehehai/Auralink) 复制而来。Auralink 是同一作者基于远程模型的应用，它用 Go 和 MyGo 以 [AudioSmith](https://github.com/hehehai/AudioSmith) 的界面重写了 [Voxt](https://github.com/hehehai/voxt)（本仓的 [`../Voxt`](../Voxt)）；VoxtNew 的本地模型运行时和请求策略沿用 Voxt 的做法。

## 许可

Apache License 2.0（[LICENSE](LICENSE)），与本仓其余部分一致。版权归 ActNow 和 SGLang-Omni 团队所有。第三方代码、运行时及其许可见 [NOTICE](NOTICE)；模型按各自的许可下载。
