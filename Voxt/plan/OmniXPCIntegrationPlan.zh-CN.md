# Voxt Omni XPC 集成方案

## 1. 目标

将 Voxt 当前的 Omni Native backend 从“App 启动本地 HTTP/WebSocket server”逐步演进为：

```text
Voxt.app
  -> Omni XPC Service
      -> Omni Native Core
          -> MLX / Metal
              -> 本地模型权重缓存
```

用户只安装一个 `Voxt.app`。Native runtime 代码随 App 构建、签名和发布；模型权重作为数据按需下载并缓存，不放进 App bundle。

现有 HTTP server 保留为开发、CI、CLI 和模型 golden test 的 transport adapter，不作为正式 Voxt 产品路径。

## 2. 当前基线

当前 Voxt 的 Omni 路径是：

```text
MLXModelManager
  -> OmniASRRuntime
      -> Process()
          -> qwen3_asr_server / whisper_server / cohere_transcribe_server /
             moss_transcribe_diarize_server
              -> localhost HTTP/WebSocket
                  -> C++ / MLX inference
```

已具备的基础：

- Native C++/MLX ASR、VAD、Sortformer runtime；
- supervised process launch protocol；
- HTTP、SSE、WebSocket API；
- Swift 侧模型加载、lease、drain、retire、restart 逻辑；
- 模型下载、校验、缓存和删除逻辑；
- Native golden、API、生命周期和 Voxt Omni 测试；
- GitHub Apple Silicon CI 现场构建 Native runtime 并运行测试。

当前主要限制：

- Xcode 工程没有 Omni XPC target；
- Release App 没有嵌入 Native runtime；
- `VOXT_OMNI_RUNTIME` 仍然指向外部 binary；
- Native server 的 HTTP transport、进程控制和推理核心尚未分层；
- GitHub CI 会临时构建 runtime，但未产出可发布的 runtime artifact 或 packaged Voxt.app；
- Qwen、Whisper、Cohere、MOSS 仍共用 Voxt 的 MLXAudio 上层兼容类型。

## 3. 目标架构

### 3.1 代码分层

```text
Omni Core
  - 模型加载
  - 音频预处理
  - MLX 推理
  - session 状态
  - final / streaming / VAD / diarization 结果

Omni XPC Service
  - XPC listener
  - Swift/C++ 或 Objective-C++ bridge
  - session 生命周期
  - 音频 chunk 和 event 转发
  - 模型目录访问

Omni HTTP Adapter
  - 当前 HTTP/SSE/WebSocket API
  - Native CI
  - CLI 和开发调试

Voxt XPC Client
  - Swift async API
  - transport-independent request/result 类型
  - XPC 断连、服务重启、取消和错误映射
```

### 3.2 App bundle 形态

```text
Voxt.app/
└── Contents/
    ├── MacOS/Voxt
    ├── XPCServices/OmniInference.xpc/
    │   └── Contents/
    │       ├── MacOS/OmniInference
    │       ├── Frameworks/libOmniCore.dylib
    │       ├── Frameworks/libMLX*.dylib
    │       ├── Resources/*.metallib
    │       └── Info.plist
    └── Frameworks/
```

模型权重位于 App Group 或 App Sandbox 的 Application Support 容器中：

```text
Application Support/Voxt/Models/<model-repo>/
```

XPC 服务不需要用户单独安装，也不需要启动常驻 LaunchAgent。它由 macOS 按需启动，Voxt 退出或服务空闲时释放。

## 4. 方案决策

### 4.1 正式产品路径

采用“内嵌 XPC Service + Native Core”。

- App 内嵌并签名 XPC executable、Omni Core、MLX dylib 和 Metal resources；
- App 和 XPC Service 共享 App Group 模型目录；
- 主 App 负责下载、校验、删除和展示模型状态；
- XPC Service 负责加载和执行模型；
- 生产路径不依赖 localhost port 和外部环境变量。

### 4.2 开发和 CI 路径

保留当前 HTTP server：

```text
Python / CLI / golden test -> HTTP -> Omni Core
Voxt release              -> XPC  -> Omni Core
```

不做运行时自动 fallback。HTTP 只作为明确的开发/测试 transport，避免两条生产路径产生不可控的行为差异。

### 4.3 迁移策略

先完成“可发布的内嵌 Helper/Runtime”验证，再切换 XPC transport：

1. 将 Native runtime 构建和签名纳入 Voxt Release；
2. 先让 App bundle 能够定位并启动内嵌 Native binary；
3. 抽离 Omni Core；
4. 增加 Omni XPC Service；
5. 将生产 Voxt client 切换到 XPC；
6. 保留 HTTP adapter 给 CI 和开发。

这样可以把“打包问题”和“通信架构重写问题”分开验收。

## 5. 实施阶段

### Phase 0：冻结接口和验收矩阵

输出一张 capability matrix，覆盖每个模型的：

- model load/unload；
- final transcription；
- streaming/live preview；
- timestamps；
- speaker segments；
- language/prompt/hotwords；
- cancellation；
- service crash/restart；
- VAD 和 diarization 并行运行。

同时确定最低 macOS、Apple Silicon 架构、App Sandbox、App Store/Developer ID 两种发布方式。

### Phase 1：Runtime 构建和发布链路

目标：不改变 HTTP 协议，先让 Native runtime 成为可发布产物。

工作项：

- 将 `build_runtime.sh` 的 CMake 构建接入 Xcode/Release pipeline；
- 明确只打包 server 必需产物，评估是否排除 CLI transcribe binary；
- 生成 Release arm64 runtime；
- 对 executable、dylib、Metal library 做 strip 和依赖审计；
- 将 runtime 嵌入 App 的标准位置；
- 删除生产环境对 `VOXT_OMNI_RUNTIME` 的依赖；
- 加入签名、notarization 和 packaged app smoke test；
- 输出真实体积清单：App、XPC、Native Core、MLX、Metal、模型数据分别统计。

验收：从全新机器安装 Voxt.app，不设置环境变量，不安装 Python/uv，能够启动一个 Native server 并完成 Qwen final transcription。

### Phase 2：抽离 Omni Core

目标：让 HTTP server 和 XPC Service 共用同一套推理代码。

工作项：

- 从现有 `*_server` 中分离模型加载、session、推理和结果结构；
- 形成 `OmniCore` C++ library 或稳定 C ABI；
- HTTP server 改成调用 `OmniCore`；
- 保持现有 API 和 golden 输出不变；
- 明确线程、MLX stream、模型切换和 session 的所有权；
- 不把 MLX C++ 模板类型直接暴露给 Swift。

验收：HTTP golden/API 测试结果与当前一致；同一 Native Core 能被一个最小 CLI 调用。

### Phase 3：增加 Omni XPC Service

目标：建立 App 内部的正式 Native transport。

工作项：

- 新增 `OmniInference.xpc` Xcode target；
- 增加 XPC listener 和 App-side client；
- 定义 load、start session、push audio、finish、cancel、unload、health 和 event API；
- 音频使用二进制 Data/dispatch data，不使用 JSON；
- 将 final、preview、VAD、speaker event 映射为 Voxt 自己的类型；
- 实现服务断连、崩溃、重启和未完成请求的错误处理；
- 将 App 和 XPC Service 统一到 App Group 模型目录；
- 保持服务无常驻后台状态，按需加载和卸载模型。

验收：XPC 服务可以独立崩溃并被 Voxt 识别；重连后下一次请求可以重新加载模型；App 主进程不被 Native crash 影响。

### Phase 4：切换 Voxt 生产客户端

目标：正式 Voxt 运行不再依赖 HTTP 和外部 runtime path。

工作项：

- 将 `OmniASRRuntime` 重构为 transport-independent runtime client；
- 增加 `OmniXPCClient`；
- 将 `MLXModelManager` 的 Omni 分支切换到 XPC；
- 保留 `OmniHTTPClient` 给开发和测试；
- 保持模型切换、idle unload、delete、quit、cancel 和 crash recovery 语义；
- 清理 `VOXT_ASR_BACKEND`、`VOXT_OMNI_RUNTIME` 在 Release 中的使用。

验收：Release 构建的 Voxt.app 在干净环境中完成 Qwen、Whisper、Cohere、MOSS 的已支持能力，并能正确使用 VAD/Sortformer。

### Phase 5：体积、性能和稳定性验收

必须分别测量：

- 安装包下载大小；
- 解压后 App 大小；
- XPC runtime 大小；
- MLX dylib 和 Metal kernel 大小；
- 首次启动耗时；
- 首次模型加载耗时；
- 模型切换耗时；
- ASR + VAD + Sortformer 并行内存；
- 服务崩溃恢复时间；
- 长时间会议内存增长。

模型权重单独统计，不计入固定 App runtime 体积。

## 6. 需要修改的主要位置

### Voxt

- `Voxt/Voxt.xcodeproj/project.pbxproj`：新增 XPC target、target dependency、Embed XPC Service、签名配置；
- `Voxt/Voxt/Transcription/OmniASRRuntime.swift`：从 Process/HTTP client 演进为 transport-independent client；
- `Voxt/Voxt/Transcription/OmniASRBackend.swift`：从环境变量路径配置改为 bundled XPC 配置；
- `Voxt/Voxt/Transcription/OmniNativeStreamingSession.swift`：切换到 XPC event stream；
- `Voxt/Voxt/Transcription/OmniVoiceActivity.swift`：切换 VAD session transport；
- `Voxt/Voxt/Transcription/OmniSpeakerDiarization.swift`：切换 diarization transport；
- `Voxt/Voxt/Transcription/MLXModelManager.swift`：统一模型目录和 XPC service 生命周期；
- `Voxt/Voxt/Core/Models/*Download*`：保留下载逻辑，改为 App Group/正式缓存路径；
- `Voxt/.github/workflows/*`：增加 Release runtime 构建、打包、签名和 smoke test。

### sglang-omni Native

- `sglang_omni_mlx/native/CMakeLists.txt`：增加 Omni Core library/XPC 可链接产物；
- `sglang_omni_mlx/native/scripts/build_runtime.sh`：区分 CI runtime、开发 runtime 和 Release embedded runtime；
- `sglang_omni_mlx/native/*_server`：保留 HTTP adapter；
- 新增 Omni Core 的稳定 C/C++ API 和 XPC host target。

## 7. 关键风险

### 体积重复

Voxt 已经使用 MLXAudio Swift；Native C++ MLX 可能带来另一套 MLX dylib。必须通过 Release 构建确认是否重复，不能假设两者可以直接共享。

### Sandbox 和模型目录

XPC Service 有自己的 sandbox。模型目录应使用 App Group，或者由主 App 通过安全作用域 bookmark 明确授予访问权限。

### Native 代码签名

XPC executable、Omni Core、MLX dylib、Metal resources 的签名顺序和 entitlements 必须纳入自动化，不能依赖手工复制。

### Streaming 和背压

XPC 传输音频 chunk 和 transcript event 时要明确队列上限、取消语义和服务断连行为，防止录音过程中积压数据。

### 模型能力不完整

XPC 只是 transport 替换，不会自动补齐 Parakeet、Nemotron、SenseVoice，也不会自动解决 Cohere realtime 和 Whisper realtime 的能力差异。

## 8. 完成标准

方案完成不以“XPC target 能编译”为准，而需要满足：

1. 用户只安装一个 Voxt.app；
2. 不需要 Python、uv、环境变量或外部 runtime path；
3. Native runtime、MLX、Metal kernel 全部随 App 正确签名；
4. 模型权重独立下载、校验、缓存和删除；
5. App 和 XPC Service 通过共享目录访问模型；
6. Qwen、Whisper、Cohere、MOSS 的既有 Native 能力通过 App 端到端测试；
7. XPC 服务崩溃不会导致 Voxt 主进程退出；
8. 服务可以被取消、卸载、重启，且不留下孤儿进程；
9. HTTP API 继续通过 Native CI golden/API 测试；
10. Release artifact 有真实的体积、签名、notarization 和干净机器安装记录。

## 9. 当前建议

先完成 Phase 0 和 Phase 1，再决定是否马上进入 XPC。

原因是当前最大的未知项不是 XPC API，而是：

- Native runtime 的 Release 体积；
- MLX 依赖是否重复；
- Xcode 能否稳定完成嵌套签名；
- App Sandbox 下 XPC 是否能稳定读取模型缓存；
- 当前 server 代码中哪些部分需要从 HTTP 层抽离。

如果 Phase 1 证明内嵌 Native Helper 的体积和签名可接受，可以在此基础上继续迁移到 XPC；如果 Native Core 抽取成本过高，则短期先发布内嵌 Helper + HTTP，XPC 作为后续 transport 重构。

## 10. Phase 0/1 前置验证记录

验证日期：2026-10-10。以下结果来自 Apple Silicon 本机的 Release 构建和产物检查。

### 10.1 Phase 0 结果

当前 Native 能力还不是完整的 Omni 替换矩阵：

| 能力 | 当前 Native 情况 | 结论 |
|---|---|---|
| Qwen3-ASR | final、realtime | 可以由 Native 替换 |
| Whisper | final、batch preview | 可以替换，但没有 Native realtime |
| Cohere | final、batch preview | 可以替换，但 realtime 需要继续保留 Swift 或补齐 Native |
| MOSS | final、realtime、speaker segments | 可以由 Native 替换 |
| VAD / Sortformer | 已有 Native probe/runtime | 可以作为 Native 依赖使用 |
| Parakeet / Nemotron / SenseVoice | 当前仍由 Swift MLX 实现 | 不能宣称已完成全模型替换 |

平台验证确认当前 Native runtime 的最低系统版本为 macOS 26.2：`qwen3_asr_server` 最低系统版本约为 macOS 26.0，`libmlx.dylib` 和 `libjaccl.dylib` 为 macOS 26.2。因此本验证版本将 Voxt 的最低系统版本统一设为 macOS 26.2；后续若要恢复 macOS 15 支持，需要重新构建兼容的 MLX 和 Native runtime。

当前 App 开启了 App Sandbox，但 entitlements 没有 App Group。现有模型目录默认是：

```text
~/Library/Application Support/Voxt/model-storage
```

同时支持用户选择外部目录，并用 security-scoped bookmark 保存授权。对于 XPC，默认目录应迁移到 App Group；用户选择的外部目录则需要由主 App 统一管理授权，不能直接假设 XPC 能复用主 App 的 bookmark。这个问题尚未通过验证。

### 10.2 Phase 1 结果

已完成的本机检查：

- `build_runtime.sh` 成功生成 arm64 Release runtime；
- 当前完整 runtime 约 `213 MB`；其中 `mlx.metallib` 约 `182 MB`、`libmlx.dylib` 约 `20 MB`、`libjaccl.dylib` 约 `1.5 MB`；
- 只保留四个 server binary 时约 `208 MB`，因此生产包不应携带 CLI transcribe/probe binary；
- Voxt 当前 Release App 构建成功，未嵌入 Native runtime 时约 `125 MB`；
- App 与 server-only runtime 合并后，未压缩体积估算约 `333 MB`，还没有计算模型权重；
- App 内已有 Swift MLX 的 `default.metallib`，约 `3.6 MB`。App 使用 `mlx-swift 0.31.6`，Native 使用 Python wheel 中的 MLX `0.32.3`，两套 MLX 不能简单视作同一个可共享二进制；重复的 Native MLX dylib 和 Metal kernel 已经确认是主要体积来源；
- 当前 Xcode 工程只有 Voxt 和 VoxtTests target，没有 `OmniInference.xpc` target，也没有 XPC service 的嵌套签名链；本次构建使用 `CODE_SIGNING_ALLOWED=NO`，所以还没有验证 Developer ID/App Store 签名、notarization 或 XPC 嵌套签名。

### 10.3 阶段判断

Phase 0：能力矩阵已经足够支持继续做接口设计，但“完整 Omni 替换”尚未成立，且最低 macOS 版本必须先定下来。

Phase 1：Native 构建和体积测量已通过；发布链路尚未通过，具体阻塞为：

1. MLX/native runtime 目前只支持 macOS 26.2 及以上；
2. 生产包还没有只包含 server 的正式布局；
3. 没有 XPC target、App Group 和真实签名验证；
4. 模型目录的主 App/XPC 共享访问策略尚未落地。

因此现在不应直接进入“把现有 HTTP server 改成 XPC”的大规模重构。下一步应先固定最低 macOS 支持范围，在该环境重新构建并检查 `libmlx.dylib`，然后做一个最小的 signed XPC package smoke test，再决定是否继续抽离 Omni Core。

## 11. Native Runtime 验证版本实现记录

在当前验证分支中，目标调整为先交付 macOS 26.2+ 的单 App Native Runtime 版本，不等待 XPC。

实现内容：

- `OmniASRBackend` 优先读取开发环境中的 `VOXT_OMNI_RUNTIME`，没有环境变量时自动查找 App Bundle 内的 `OmniRuntime/bin/qwen3_asr_server`；
- `VOXT_ASR_BACKEND=swift` 可以显式关闭 Native backend，方便开发和回归测试；
- `package_local_app.sh --native-runtime <runtime-dir>` 会把 runtime 注入 `Contents/Resources/OmniRuntime`；
- 打包脚本会验证 Qwen、Whisper、Cohere、MOSS server 以及 MLX dylib/Metal library 是否齐全；
- Developer ID 打包会按嵌套顺序签名 runtime、framework、Sparkle XPC 和 App；
- 模型权重仍然独立下载到用户模型目录，不会进入 App 安装包。

本机验证结果：

- Voxt Release App 构建通过；
- Native runtime 成功注入 App Bundle；
- Developer ID Application 签名通过；
- `codesign --verify --deep --strict` 通过；
- 验证 App 未压缩约 `337 MB`；
- ZIP/PKG 约 `105 MB`，DMG 约 `120 MB`；
- 当前 runtime 仍包含 CLI/probe binary，后续正式发布前可以继续裁剪；
- Native runtime 和 App 的实际最低系统版本均按 macOS 26.2 处理；Intel 版本只使用 Remote backend，不启动 arm64 Native runtime。

因此这一版的产品形态已经从“外部环境变量指定的开发 server”变成“App Bundle 内置 runtime、App 自动启动 localhost server、用户只安装一个 App”。XPC 仍然保留为后续隔离 Native crash 和跨进程模型访问的演进方案。
