# Sortformer 长文件分析退化修复

状态：代码与测试已补充，macOS 编译、模型测试、质量及性能复测待执行。不是已完成真机验收。

## 证据和前次判断更正

2026-09-21 用户日志中，ASR 全部 121 个窗口完成；03:55:36 UTC 进入说话人分析。每个 60 秒窗口的耗时逐渐从几秒升至 31、124、163 秒；第 21 个窗口开始后记录中断，04:10:17 出现新进程启动。没有业务 `task-failed`，所以无法从这份日志确定是用户强制退出、系统杀进程还是崩溃，仍需系统诊断报告。

温度 nominal/fair 不能证明热降频就是根因。RSS 大幅下降也不能证明内存使用降低：日志缺少当时的 footprint、压缩内存、MLX allocator 和 swap 数据。

408 个原始窗口片段变成 386 个最终片段，可能是 `MeetingTranscriptPostProcessor.process` 的合并/整理；检查点保存的是其前的 raw segments，不能只按数量认定检查点丢失数据。

## 已核实的调用约束问题

检查的依赖为 app 锁定版本 `mlx-audio-swift` commit `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5`，没有修改依赖 pin 或本地依赖源码。

`Sortformer.feed` 在一次 `streamingStep` 后仅调用一次 `maybeCompressState`。后者在 AOSC 模式下：

```text
popLen = min(fifoLen - fifoMax, spkcacheUpdatePeriod)
```

因此 `fifoMax` 不是任意大小 feed 的硬上限：一次输入的新帧大于最多能移除的帧，FIFO 就会逐次积累。`streamingStep` 又把全部 FIFO、speaker cache 和当前块一起送入后续计算。

以 hop=160、subsampling=8、16kHz、updatePeriod=188 为例：60 秒约 750 个输出帧，每次只移除最多 188 帧；20 次 feed 后 FIFO 可达到 11240 帧，而不是目标 188 帧。这是按依赖代码推导的示例，实际值由模型配置决定；新增日志打印实际配置及每次返回的帧数。该结构性问题与持续变慢吻合，仍需真机确认修复后的资源和准确率表现。

## 本次修复

1. **分开读盘窗口与推理窗口**：保留最多 60 秒的读取，feed 上限取 5 秒、模型 chunkLen、AOSC updatePeriod（减去 padding 余量）的最小值，并对齐 hop × subsampling。
2. **不重置说话人身份**：连续 descriptor 复用同一个 streaming state；不每 5/10 分钟清空 state。每个 feed 前后校验 FIFO≤188，speaker cache≤配置上限，异常立即停止后续提交，不静默修剪或重置标签。
3. **时间轴对齐**：将 feed 的帧累计偏移映射回当前真实样本偏移，裁剪末尾 padding，避免小块边缘帧导致整场时间漂移。feed 内不提前丢弃短标签片段，后续沿用已有平滑/文本组装；质量需专项验证。
4. **每块准入**：新增 fileSpeakerAnalysis 工作类别，复用现有资源协调器和可取消等待。模型加载和每次短 feed 都获取/释放许可；首次和恢复后均复查内存、温度、录音与执行通道，不保持整场许可。
5. **协作式停止**：短 feed 的原生执行时间超过 30 秒时，等待该原生调用真正返回再报错，停止下一块，保留 ASR 检查点。资源等待时间不计入这个阈值；不启动脱离管理的超时任务，不虚称硬实时终止 Metal。
6. **资源所有权**：文件使用独立 Sortformer 引擎；所有退出路径释放该引擎缓存的模型，避免共享说话人引擎在文件结束后一直保留权重。不通过每块全局清空 MLX 缓存来掩盖仍被引用的状态。
7. **真实错误语义**：文件入口使用 throwing 说话人分析，不把模型/输入/安全上限错误吞掉后写成成功历史。实时会议入口仍保留原有 fallback 行为。
8. **转录可查看**：说话人执行、等待或失败时，“查看转录”直接打开现有会议详情窗口的只读文件草稿模式，使用完整 ASR 检查点的 segments 保留时间轴、搜索、选择和复制；明确标记“说话人分析尚未完成”。临时纯文本弹窗已删除。不创建未完成的普通历史、不启动摘要/翻译；草稿暂不提供缓存音频播放、编辑和导出。任务成功后，已打开的同一窗口切换为正式历史结果与归档音频，不强制抢焦点，也不重开已关闭窗口。重试继续复用现有 ASR 检查点，重新执行说话人阶段。
9. **测量补充**：详细日志增加 physical footprint、compressed bytes；说话人窗口边界记录 MLX active/cache，feed 边界记录 fifo/cache 帧数和实际块大小。均为进程或 allocator 的采样，不能混为同一指标，也不能当作完整峰值。

## 明确不做

- 不序列化 Sortformer 的内部张量/缓存；异常或重启后说话人从头重跑，ASR 不必重跑。
- 不每个窗口重置 speaker ID，不按 RSS 单一数值判定无压力。
- 不保证不会出现系统级 OOM；同步原生调用的卡死仍可能需要独立进程隔离，当前实现未引入该复杂度。
- 不自动重试任意异常、不改变模型选择、不删除源视频或有效 ASR 检查点。
- 状态写入/读取身份的完整加固、阶段增量存储和模型状态持久化不在本修复范围。

## 后续修复：89.23% 长时间等待与闲置 MLX 缓存

2026-09-21 第二次日志显示 FIFO/cache 均稳定为 188，窗口 60–75 仍正常推进；但 MLX 的 `activeMemory` 约 237 MB，allocator `cacheMemory` 约 1.788 GB，footprint 约 2.22 GB。注意这里的 speaker cache **帧数**与 MLX allocator cache **字节数**不是同一个缓存。

13:20:46 UTC 开始 `memoryPressure=true`，音频推进停在约 4529.76 秒，总进度 89.23%。随后约 20 分钟只有心跳，footprint 仍约 2.17 GB；不是持续推理。原先等待循环只睡眠与检查布尔值，没有主动回收闲置 allocator 缓存，也没有复核可能陈旧的压力事件。日志不能单独证明系统所有压力都来自本进程，也不能断定后续 Metal 编译服务 XPC 错误是直接原因。

上一轮小范围修复（以下记录当时行为；当前已调整为下一节的固定低开销链路）：

- `MeetingFileInferenceCache` 在文件 ASR/说话人工作的安全边界检查闲置 cache；**超过 256 MiB 才清理**，小缓存保留，避免无条件每块清缓存。只调用 MLX `Memory.clearCache()` 释放 allocator-owned unused buffers，不删除权重或重置 streaming state，不修改全局 `cacheLimit`/`memoryLimit`。
- 返回/异常/取消的清理由 `withPermit` 的 defer 在原生工作真正退出后、释放许可之前执行。继承的旧大缓存在首次准入前也检查；内存压力等待期间、协调器通道空闲时，每次压力周期额外清理一次闲置缓存。其他工作占有通道时不在等待线程进行维护。
- 256 MiB 是**工作单元之间的缓存保留阈值**，不是硬性总内存/单次推理峰值上限。全局 allocator 可能被其他模型重新填充；其余调用者仍拥有自己的活跃数据，不能声称该协调器覆盖所有 MLX 调用。
- 文件准入/等待使用只读 `kern.memorystatus_vm_pressure_level` 可选探测，按 dispatch 的 normal/warning/critical 值解释。系统确实报告 normal 才能解除旧标志；探测失败、未知值或仍有压力时保留原保护，不以低 RSS 或等待超时猜测恢复。该接口可用性仍需目标 macOS 验证。
- 当时每 15 秒输出压力和通道状态，并记录缓存回收前后数值。该高频诊断已在后续整理中默认关闭，不再作为当前日志策略。

证据：锁定的 mlx-swift 0.31.6 `Source/MLX/Memory.swift` 中 `clearCache()` 通过 eval lock 调用 `mlx_clear_cache`；XNU `bsd/kern/kern_memorystatus_notify.c` 的 sysctl handler 将当前压力转换为 NOTE_MEMORYSTATUS 单值后输出。仅做只读探测，有明确不支持时回退，未更改任何依赖或系统设置。

复测关注：出现 `file-cache-reclaimed` 后 cache/footprint 是否实际下降；`file-permit-wait` 是否仍显示真实压力；压力解除后能否出现下一条 feed 完成。若系统确实长期处于 warning/critical，仍应保持暂停而非绕过保护。本地尚未执行 macOS/XCTest，不能宣称此样本已完整处理成功。

## 当前简化：直接压低开销，不再将 warning 视为不可执行

最新附件含旧进程记录，不能把其中 1.788 GB 旧缓存当作新进程占用。22:40 新进程在模型加载之前被 `source=current-sample` 阻塞，footprint 约 61 MB，通道空闲。该日志没有保留 warning/critical 原始等级，不能仅凭低 footprint 判断系统无压力；但继续叠加缓存回收/轮询不能解决把两级都视为硬阻塞的策略问题。

当前做法保持单任务、短输入和既有检查点，只修改实际资源使用方式：

1. 每个文件推理单元持有许可期间，MLX unused `cacheLimit` 临时取原值与 **128 MiB** 的较小值；清理超限旧缓存，异常/取消/成功退出均恢复原设置。不修改 active `memoryLimit`，不每块卸载模型。实际分配仍受 MLX 的回收时机影响，这不是整个进程内存峰值的硬上限。同步参与的文件任务共用既有串行许可；全局缓存设置在工作单元期间可能影响其他 MLX 使用者的缓存复用性能，但不会删除他们的活跃张量。
2. 文件路径只在已知 **critical** 时阻塞；normal/warning 下执行固定有界工作并保持缓存上限，不做可用内存猜测和自动预算扩张。未知探测保留通知保护，取消/录音/严重温度/磁盘安全检查不移除。其他后台任务的 warning 行为不被文件任务修改；不可执行的其他 waiter 不再阻塞已符合准入条件的文件任务。
3. 同样的 critical-only 策略用于流式文件准备和最终归档复制，避免转录结束后又因普通 warning 卡在保存阶段。
4. ASR 分块先记录样本范围，使用 lazy collection 在消费当前块时才复制 PCM。22 秒上限、1 秒重叠、静音规划和时间轴保持不变，不同时持有一整个窗口的所有分块副本。
5. 文件音频已标准化为有限值的 16 kHz PCM，说话人阶段不再对同一窗口做一次全量 `map`/重采样复制。模型的 streaming state 连续保留，不通过重置 speaker 身份节约内存。
6. ASR 转录器局部作用域在转录后结束，阶段边界取消已结束的残余工作、清空转录器引用，再释放 manager 的闲置 ASR 权重，避免 VAD/转录器缓存跟随整个说话人阶段。
7. 分析只读队列缓存；原先分析前的完整 WAV 副本改到成功后的保存阶段才创建，且仅在任务开始分析时开启了历史音频存储才复制。历史仍取得独立副本，不会移动队列缓存。取消/失败只清理本次拥有的临时文件。

不承诺内存、CPU、GPU 三者同时达到绝对最小：更小的 unused cache 可能增加重新分配；这里优先减少闲置保留和重复复制，不增加更多模型调用、激进静音裁剪或新并发任务。关键验收是完整运行的 footprint/MLX cache 降低且质量不退化、普通 warning 下能推进、critical 下不强行运行；不是只看 RSS 或总进度。尚未在 macOS 执行验证。

## 验收

纯逻辑/桩测试：
- `MeetingSpeakerFeedPolicyTests`：12 小时 FIFO 递推上限、旧 60 秒 feed 反例、小 updatePeriod、非法配置、超限和时间轴 padding。
- `MeetingFileSpeakerFailureTests`：文件说话人错误不能退化成成功的无说话人结果。
- `MeetingLocalInferenceCoordinatorTests`：文件说话人压力等待/取消，等待通道期间出现压力时不误准入。
- `MeetingFileInferenceCacheTests`：缓存阈值、作用域退出恢复、warning/critical 文件策略、其他调用者保护不变、成功/错误/取消后的维护和未知/持续 critical 不强行放行。使用注入的 allocator/probe，不在普通单测初始化 GPU。
- `MeetingTranscriptAssemblyTests`：惰性范围计划、按需 PCM、重叠和时间轴回归；保留已有静音和严格错误测试。
- `SortformerBoundedFeedIntegrationTests` 补充 cacheLimit 恢复与 live memoryLimit 不变验证，长流式测试采用同样的工作单元缓存作用域。

模型测试（默认跳过，需要本地已安装 Sortformer）：
- `SortformerBoundedFeedIntegrationTests`：20 分钟合成音频，实际锁定模型持续 feed 的 fifo/cache 上限和帧进度。合成测试不替代真实多人质量验证。

macOS：
```bash
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO \
  -only-testing:VoxtTests/MeetingSpeakerFeedPolicyTests \
  -only-testing:VoxtTests/MeetingFileSpeakerFailureTests \
  -only-testing:VoxtTests/MeetingLocalInferenceCoordinatorTests \
  -only-testing:VoxtTests/MeetingFileInferenceCacheTests

VOXT_RUN_MODEL_TESTS=1 VOXT_MODEL_STORAGE_ROOT='/path/to/models' \
  xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO \
  -only-testing:VoxtTests/SortformerBoundedFeedIntegrationTests
```

用户原始 2 小时样本至少同机复测：不清理有效任务/检查点，点击重试应跳过 ASR；确认 `speaker-feed-policy` 与 `speaker-feed-completed` 帧数受限；不再随处理分钟数逐步扩大单次计算上下文。记录窗口耗时分布、footprint/MLX/cache、系统 swap/内存压力、CPU/GPU、温度、取消响应，以及跨 5 秒和 60 秒边界的 speaker 一致性、DER、时间轴和短发言保留。

另外验证：普通 warning 下文件模型加载和后续小块能执行；critical/录音时继续暂停且可取消；关闭历史音频存储不创建副本；分析失败后原缓存仍有效；开启存储仅在保存阶段出现 archive-copy 日志，历史回放有效。另查只读转录预览不触发模型/历史写入，清理任务后找不到检查点时明确提示。不要把仅编译成功或纯递推测试通过称为长文件性能已达标。
