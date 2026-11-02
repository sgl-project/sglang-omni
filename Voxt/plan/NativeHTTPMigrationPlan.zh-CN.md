# Voxt Native HTTP 迁移边界与剩余验证

## 已确认的架构决策

- 不做 XPC，不把它列为 MLX Audio 替换或 release 的前置条件。
- 不扩展共享 `sglang_omni_mlx/native` 包。撤回本任务加入的 C ABI、typed realtime engine 提取、checked destroy、配置/数值边界修改及配套替身测试，恢复到已合入官方 main 的 `e3d7dc378c749914ffca07576c00b16d32b08739`。后续只正常合入官方更新，不把 Voxt 的封装职责放进共享包。
- 删除 Voxt XPC helper/client/contracts、配置、App Group/smoke/嵌入链路以及对应测试/CI 选择，不保留等待生产接入的死代码。
- 保留 Voxt 已有 Native HTTP/WebSocket 子进程路线和完整 AudioSTT/VAD/Core 退包成果。Native 不可用明确报错，不恢复 Swift 数值 fallback。
- 原 catalog、保存选择、安装权重、Remote ASR/LLM、MLX Swift LLM/VLM/MLX、Tokenizers/HuggingFace、llama.swift 不变，原 `model-storage/mlx-audio/...` 路径不重命名。
- 已发布历史不重写；撤回使用正常修正提交。原 XPC 计划及其测试记录仅属于此前 Git 历史，不代表当前待完成任务。

## 保留的 App 职责

| 范围 | 边界 |
| --- | --- |
| ASR | 七个既有 Native server 对应十一 catalog checkpoints；Qwen/MOSS/Nemotron realtime，其余 family 保留 batch-preview/final 路线 |
| 音频/结果 | 既有 family options、language/hints、采样/rounding、chunk cadence、结构化 segments 与最终展示；不自动 replay/truncate/切成30秒拼接 |
| Silero | HTTP whole-audio offline，包括超过30秒输入；stream保持completed-frame/nil/tail/reset语义，上层原energy降级警告不伪装Native成功 |
| Sortformer | 原约5秒feed/FIFO/cache/PCM offset scheduler；Foundation读取调度配置不依赖Audio/XPC，也不改Native模型settings/admission cap |
| 生命周期 | 原子进程、endpoint作用域及directory reader由Voxt持有；reader在spawn前取得，实际child exit后释放，不把cancel/socket/task end作为GPU barrier |
| 存储 | 独立OmniModelArtifacts的bounded config/safetensors header验证、cooperating shared/exclusive reader、sibling atomic publication、source/staging复制协调 |

未调用的 `MeetingRealtimeDiarizationStage` 已在前一 checkpoint 经caller/selector/resource审计后删除。不重新引入24秒whole-window替代，也不修改Native的既有feed cap逃避真实限制。

## 修正后的验证清单

- [x] 当前修正index与working tree的120 Native files、mode/blob/内容与官方版本一致（subtree `86848cefe4cda5902b25248f6c7b77cdb511cf02`）；修正commit后再用`git diff e3d7dc37 HEAD -- sglang_omni_mlx/native`检查零差异。
- [x] App/project/source/test/CI无 abandoned XPC helper/Contracts/Core链接、selector、entrypoint或构建phase。实际项目仅Voxt/VoxtTests targets及Debug/Release/TestDebug配置；第三方Sparkle/EventKit系统XPC组件保留。
- [x] Audio product/package/pin缺失；严格Mac build后Package.resolved字节不变（SHA256 `a5a7e57da071226128a869cff06c69ed9720e32f955cf23d1870477cbb300999`），没有重新解析升级pins。
- [x] Python工具39、source/resolved audit、全量hooks及`git diff --check`通过。独立删除review也fresh39工具+22 Swift；portable替身限制明确。
- [x] 独立OmniModelArtifacts Foundation/POSIX fresh79，0失败。
- [x] 真实Mac普通App build/完整TestDebug：1975（1933通过/42skip/0失败），isolated test home、unsigned、严格lockfile，未覆盖正式App/私人数据。
- [x] 官方未修改Native runtime fresh构建、七server/resource inventory与App实际artifact audit通过，无OmniCore/helper/Audio modules残留。首次依赖下载curl52失败保留；使用SHA/commit验证的既有nlohmann3.11.3、PCRE2/civetweb缓存通过CMake支持的source override重跑，未改Native脚本/源码。
- [x] 现有Qwen0.6B与公开PCM16：official/installed CLI相同22-token英文text+stop；真实Native-only Silero HTTP detector独立旧capture对照、reset/recovery/reader10，0skip/0fail，不用旧Core结果替代。
- [x] 独立review确认17 protected HTTP/catalog/storage/LLM/VLM/Remote/DSP/Artifacts source groups不变，source退包保留、XPC删除完整。Mac冻结输入1047与本机snapshot hash一致，官方Native120 source文件亦单独比对。

额外Qwen App final/live benchmark尝试两次均各2项skip：现有bf16安装由catalog迁移规则规范化为4bit，未安装目标4bit，因此不是HTTP成功/质量结果。未改catalog、把bf16冒充4bit、弱化gate或下载新权重；当前Qwen真实证明限定于CLI，Silero HTTP10是实际执行。CLI耗时仅单次观察，不当性能结论。

前一冻结源码Mac TestDebug2023（1981通过/42skip/0fail）、Core ABI10/helper/signed copy/portable peer结果保留为历史，不能自动继承给此次修正。服务重启丢失的`/tmp`结果不冒充fresh执行。新证据放durable task cache，限并行子任务TOTAL≤3（含descendants）；连接/限流失败不报review成功。

## 数字对照与剩余风险

保留真实旧AudioCore compiled product的42 DSP输出fixture（Float32 bits、输入/平台来源/hash），不是新代码生成oracle。Silero/Sortformer parity要求独立旧实现capture，绑定model/decoded samples/profile/settings；缺reference显式skip，不做Native自比较。现有真实fixture仅覆盖Qwen和Silero，不认证其他family全部模型。

目录reader只覆盖cooperating正常操作：不保证外部blob不可变、hostile namespace、文件hash完整性、fsync/power-loss、App crash后Native orphan清理或cache/repair自动恢复。TCP/WebSocket完成不代表每个stream的GPU工作清空；可观察的整模型退休仍是所拥有child实际退出。

全部model质量、GUI/权限、长会议、memory/performance、Intel/Release/clean-machine、cache/blob/orphan/recovery与最终release矩阵仍需独立验证。不注册App Group、不重新使用上传签名/公证/release凭据，不自动push/PR merge/部署或开启CI/release门槛。
