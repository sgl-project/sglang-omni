# 构建和开发 Voxt

[English](development.md) · **简体中文**

怎么从源码构建 Voxt、在开发时签名、运行测试。安装和使用见 [README](../README.zh-CN.md)；提交改动见 [CONTRIBUTING.md](../CONTRIBUTING.md#参与贡献)。

## 里面有什么

- **语音识别**和**声音检测**跑在 sglang-omni 的原生 MLX 服务上（[`sglang_omni_mlx/native`](../../sglang_omni_mlx/native)，C++，不用 Python），每个模型一个进程，用到时启动，10 分钟不用就停（语音识别模型页的“高级”里可以改）。
- 一个小的 Swift 辅助进程负责要用苹果框架的事：麦克风、全局快捷键、辅助功能、粘贴。

## 从源码构建

需要 Go 1.27.1（`GOTOOLCHAIN=auto` 会自动下载）；构建 app 需要 Apple 芯片、macOS 15 及以上的 Mac，Xcode（编译辅助进程），以及 [uv](https://docs.astral.sh/uv/) 或 Python 3（Omni 运行时的构建工具：`scripts/build-omni.sh` 用固定版本的 MLX 编译 `sglang_omni_mlx/native`，CMake 会从 GitHub 下载 C++ 依赖，需要时走 shell 的 `https_proxy`）。在 `Voxt/` 下执行：

```sh
scripts/build-helper.sh              # Swift 辅助进程，输出到 resources/darwin-arm64/helper/
scripts/build-omni.sh                # Omni 运行时（语音识别模型），输出到 resources/darwin-arm64/omni/
go tool mygo dev                     # 热重载运行
go tool mygo build                   # 生成 build/darwin-arm64/Voxt.app 和 dmg
```

`go tool mygo build` 和 `mygo dev` 之前必须先编好辅助进程：`mygo.json` 给它指定了 entitlement，缺了就会失败。没有 Omni 运行时 app 也能运行，但不能识别语音（也可以把 `VOXTNEW_OMNI_RUNTIME` 指向 `scripts/build-omni.sh --out DIR` 编出的运行时）。

**发布版本**在 Mac 上用一条命令构建：`scripts/release.sh` 编译辅助进程和 Omni 运行时，打包、签名 app 并检查结果（`--runtimes DIR` 直接用编好的运行时；有 Developer ID 时，`--notarize` 做公证（Voxt 2 没有更新通道，不用 `--publish`）；ssh 里经 `scripts/mac-gui.sh` 运行）。正式版就是在 GitHub 上跑这个构建：推一个 `voxt-vX.Y.Z` tag，Voxt Release 工作流用 RY2TQTYSKQ 团队的 Developer ID 签名、公证，把磁盘映像放进 GitHub 的草稿 Release。见 [docs/release.md](release.md)。

在 Linux 上，`scripts/mac.sh <命令>` 会把项目同步到 Mac（`MAC_HOST` 默认 `mac-air`，`MAC_DIR` 默认 `code/VoxtNew`）并在那里执行，例如 `scripts/mac.sh 'scripts/build-helper.sh && go tool mygo build'`。

<details>
<summary>开发时的签名（重新构建后保留权限）</summary>

macOS 按签名记住麦克风和辅助功能的授权。ad hoc 签名的构建（默认）每次重新构建签名都会变，所以重建以后 macOS 会忘掉授权：麦克风变成**拒绝且不再询问**，辅助功能在系统设置里看着是开的，却不起作用。辅助进程的授权就是 app 的（macOS 把 Voxt.app 当作它的责任进程），所以它也一样。解决办法：

- **只改辅助进程时**：保留一个构建好的 app，用 bundle 外面的辅助进程运行它。辅助进程放在哪里都继承 app 的授权，重编它不受影响：
  ```sh
  scripts/build-helper.sh --out ~/voxtnew-dev
  open --env VOXTNEW_HELPER=$HOME/voxtnew-dev/voxtnew-helper build/darwin-arm64/Voxt.app
  ```
- **重新开始**：`tccutil reset Microphone dev.hehehai.voxtnew`（`Accessibility` 同理），再打开 app 重新授权。如果旧的辅助功能记录开着却不起作用，在系统设置里用 − 删掉再重新授权。
- **用固定的身份签名**：Apple Development 证书能让授权在重建后保留：`go tool mygo build -sign "Apple Development: …"`（或者在本地的 mygo.json 里写 `macos.signingIdentity`；`scripts/build-helper.sh` 用 `VOXTNEW_SIGN_IDENTITY`）。通过 ssh 时，`codesign` 会报 `errSecInternalComponent`，直到允许不经图形界面使用私钥：在 Mac 的“终端”里执行一次 `security set-key-partition-list -S apple-tool:,apple: -s ~/Library/Keychains/login.keychain-db`。

在自己编的版本和发布版（Developer ID）之间切换，也要这样 reset 一次（[docs/release.md](release.md) §4）。下载好的模型会保留。

</details>

## 开发

Voxt 是基于 [MyGo](https://github.com/egoist/mygo) 的 Go 应用（用 Go 写原生 macOS 界面，不用 cgo）。由 Go 做决定；辅助进程和模型运行时照指令办事，通过 stdin/stdout 上的 JSON 行（辅助进程）或回环地址上的 HTTP（Omni 服务）通信。除了 Swift 代码，全部是普通的 Go，在 Linux 上用辅助进程和运行时的假实现测试。在 `Voxt/` 下执行：

```sh
CGO_ENABLED=0 go test ./...          # 全部测试，Linux 上可跑
go tool mygo vet .                   # go vet 和 MyGo 的 UI 生命周期检查
VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .   # 每个页面截图，浅色/深色、英文/简体中文
cd helper && swift test              # Swift 辅助进程的测试（在 Mac 上），用同一套协议 fixture
scripts/mac.sh scripts/mac-e2e.sh    # Mac 上的无界面测试：真实的辅助进程，假的模型
scripts/mac.sh scripts/mac-regress.sh   # Mac 上的真实模型：运行时、评测、打包好的 app
```

存储的测试用 MyGo 的 SQLite 插件，它会下载一次自己的库（按 SHA-256 校验），或者用 `MYGO_SQLITE_LIBRARY`。在 Linux 上，浮窗中文文字的像素检查需要 CJK 字体（Ubuntu：`fonts-noto-cjk`）。

- [docs/architecture.md](architecture.md)：包结构、线程、会话、运行时、磁盘上的数据、测试。
- [docs/plan.md](plan.md)：这个 app 是什么，以及里程碑。
- [docs/behavior.md](behavior.md)：这个 app 和 Swift 版 Voxt（[hehehai/voxt](https://github.com/hehehai/voxt)）行为上的不同，以及原因。
- [docs/llm-models.md](llm-models.md)：已退役的本地语言模型、它们的参数和实测（历史记录）。
- [docs/eval.md](eval.md)：模型在评测集上的质量，以及由此做的决定。
- [docs/helper-protocol.md](helper-protocol.md)：辅助进程的协议。
- [docs/history-access.md](history-access.md)：让这台 Mac 上的脚本和 agent 读取听写历史（默认关闭）。
- [docs/manual-test.md](manual-test.md)：在 Mac 上手动验证的清单，以及自动的真机测试。
- [docs/release.md](release.md)：构建、签名、公证和发布，以及更新。
- [AGENTS.md](../AGENTS.md)：改动规则（文字、线程、协议）。
- [CHANGELOG.md](../CHANGELOG.md)：各版本的变化。[CONTRIBUTING.md](../CONTRIBUTING.md)：怎么提交改动。

Voxt 取代了本仓在 2.0.0 之前的 Swift 版 Voxt（1.x）：装上它会替换“应用程序”里那个 Voxt.app，两者的数据互不相干；Voxt 可以把旧版的词典、历史、下载好的语音识别模型，以及语言模型服务和语音服务的配置导入过来（[docs/voxt-migration.md](voxt-migration.md)）。内部有些名字仍叫 VoxtNew（Swift 包、进程、bundle id 和数据目录）。它由 [Auralink](https://github.com/hehehai/Auralink) 复制而来。Auralink 是同一作者基于远程模型的应用，它用 Go 和 MyGo 以 [AudioSmith](https://github.com/hehehai/AudioSmith) 的界面重写了 [Voxt](https://github.com/hehehai/voxt)；VoxtNew 的本地模型运行时和请求策略沿用 Voxt 的做法。
