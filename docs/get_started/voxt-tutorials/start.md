# 安装 Voxt 和下载语音识别模型

[English](en/start.md)

使用 Apple 芯片的 Mac 需要 macOS 15 或更新版本。第一次安装，选 **Voxt 2.0.3 正式版**。需要验证最新改动时，选下方的 Beta 路线。

## 下载 2.0.3 正式版

[一键下载 Voxt 2.0.3（Apple 芯片）](https://github.com/sgl-project/sglang-omni/releases/download/voxt-v2.0.3/Voxt-2.0.3-arm64.dmg)

也可以从以下入口找到同一个版本号的安装包。

- 仓库首页往下到 **News**，找到 **Voxt 2.0**，点 **Download**。 [SGLang Omni News](https://github.com/sgl-project/sglang-omni#news)
- 在 **Voxt v2.0.3** 发布页展开 **Assets**，点 `Voxt-2.0.3-arm64.dmg`。 [GitHub Releases](https://github.com/sgl-project/sglang-omni/releases/tag/voxt-v2.0.3)
- 在 **v2.0.3** 条目点 **Download**。Changelog 可能给这条加上 **Beta** 标签，选择文件名为 `Voxt-2.0.3-arm64.dmg` 的包。 [Voxt Changelog](https://voxt.actnow.dev/changelog)

## 下载 Beta

`main` 每合入一次 Voxt 的改动，就会自动构建一个 Beta，版本号形如 `2.0.3-beta.YYYYMMDD.N`（日期加构建序号）。Beta 比正式版新，也可能不稳定。

当前最新的是 **Voxt Beta #40**：[下载 `Voxt-2.0.3-beta.20261112.40-arm64`](https://github.com/sgl-project/sglang-omni/actions/runs/43128787320/artifacts/13904657773)（约 33 MB，需要登录 GitHub，2026 年 11 月 26 日过期）。过期后，或者想找更新的构建：

1. 打开 [Voxt Beta 构建列表](https://github.com/sgl-project/sglang-omni/actions/workflows/voxt-beta.yaml)。最上面一条带绿色对勾、分支是 `main` 的就是最新的成功构建。黄色圆点表示还在构建，红叉表示失败，都跳过。
2. 点进这次构建，往下翻到 **Artifacts**，点 `Voxt-…-beta.…-arm64` 下载 ZIP。构建产物只保留 14 天。
3. 双击 ZIP 解压，打开里面的 `.dmg`。文件名里带 `-beta.`，可以用它区分每次构建和正式版。

<div class="vx-anim" data-scene="beta-artifact" data-lang="zh"></div>

Beta 不会自己更新：想换更新的 Beta，或者回到正式版，重新下载对应的 `.dmg` 安装即可。

## 安装并允许权限

1. 打开 `.dmg`，把 **Voxt** 拖到 **Applications（应用程序）**，再从应用程序打开 Voxt。有旧版时先退出旧版，按需决定是否替换或导入。

   <div class="vx-anim" data-scene="install-drag" data-lang="zh"></div>

2. 在**设置引导**点**开始设置**。在**权限**一步，**麦克风**点**请求授权**，在系统弹窗里点**允许**；**辅助功能**点**打开系统设置**，在“隐私与安全性 → 辅助功能”里打开 Voxt 的开关。两项都显示**已允许**后点**继续**。如系统要求重启 Voxt，关闭后从应用程序重新打开。

   <div class="vx-anim" data-scene="guide-start" data-lang="zh"></div>

## 下载 ASR

1. 到**模型**一步，点**下载并继续**。推荐的本地 ASR 是 **Qwen3-ASR 1.7B (8bit)**，约 2.5 GB。模型在后台下载，可以先继续引导，进度条走满就能用了。

   <div class="vx-anim" data-scene="guide-model" data-lang="zh"></div>

2. 如果已跳过引导，回侧栏**语音识别模型**，在**本机**里找到同一个型号并点**下载**，等它显示**已安装**。需要重新打开引导时，去**设置 → 通用 → 打开设置引导**。

   <div class="vx-anim" data-scene="asr-download" data-lang="zh"></div>

3. 打开一个空白文本框，轻按右 `⌥ Option`，说「明天下午三点开会」，再轻按右 `⌥ Option`。看到文本出现在光标处，就完成了本地转录的基本设置。

   <div class="vx-anim" data-scene="dictate" data-lang="zh"></div>

接下来可以[配置自己的 API Key](byok.md)，用于翻译、改写和可选的转录润色。

## 交给 Agent

> 按本页在 Apple 芯片 Mac 上安装 Voxt 2.0.3 正式版。只有我明确要求 Beta 时才下载 Actions 的 Beta 包。遇到替换旧版、导入历史或 macOS 权限授权时让我确认。下载推荐的本地语音识别模型，等到显示可用。用空白文本框和一句测试内容确认转录能输出，告诉我安装的版本和结果。
