# VAD / ASR 手工验收矩阵

本文档用于阶段 6 上线保护。自动测试通过只说明策略、偏好和启动保护可用；真实音频效果必须在 App 内完成验收。

## 前置条件

- 使用当前分支构建的 Release app。
- 设置页 `General > Advanced > VAD` 可切换 `Automatic`、`Silero`、`Energy`、`Off`。
- 模型调试页能看到 VAD mode、Frame VAD、Local Gate。
- 本地预检优先运行 `tools/run_vad_phase6_smoke.sh`。

## 会议场景

| 场景 | 模式 | 步骤 | 必须通过 |
| --- | --- | --- | --- |
| 单人清晰讲话 | Automatic / Silero | 5 分钟会议，包含开始静音、正常讲话、结束静音 | 讲话进入 ASR；静音不批量产生空转写；停止后最终稿完整 |
| 静音会议 | Automatic / Silero | 3 分钟全程静音或房间底噪 | 不产生大量空 ASR；CPU/RSS 稳定；停止后无空会议正文 |
| 键盘/鼠标噪声 | Automatic / Silero | 无人声，持续打字 2 分钟 | 不误判成长语音段；停止、ESC、语音结束指令可用 |
| 背景音乐/视频 | Automatic / Silero | 背景音 2 分钟，再加入人声 | 背景音不导致持续 ASR；人声开始后能恢复识别 |
| 远场低音量讲话 | Automatic / Silero | 麦克风 1.5 米以上，低音量讲话 3 分钟 | 不应因 gate 过严导致整段丢失 |
| 双人重叠讲话 | Automatic / Silero | 两人轮流和短暂重叠讲话 5 分钟 | VAD 不应切碎到影响 ASR 和最终稿 |
| Energy 对照 | Energy | 重复清晰讲话和静音场景 | 结果可用于对照，不作为高级默认效果 |
| VAD 关闭 | Off | 重复清晰讲话和静音场景 | 录音、最终 ASR、停止、取消都正常；允许静音更容易进入 ASR |

## 录音场景

| 产品面 | 模式 | 必须通过 |
| --- | --- | --- |
| 转录 | Automatic / Silero / Off | 最终文本完整；中间转写不因静音产生明显空刷屏；Off 不影响最终识别 |
| 翻译 | Automatic / Silero / Off | 识别、翻译、结果窗口行为正常；静音不触发大量无效翻译 |
| 转写/改写 | Automatic / Silero / Off | ASR、LLM 调用、插入或结果展示正常；Off 不影响停止和取消 |
| 长录音 | Automatic / Silero | 至少 30 分钟；chunk 数量、内存和最终文本稳定 |

## 操作与稳定性

| 场景 | 必须通过 |
| --- | --- |
| 20 次快速开始/停止 | 无采集死锁、重复提交或残留录音状态 |
| ESC 取消 | 捕获、VAD、ASR、LLM 均取消，不写入历史或提交文本 |
| 语音结束指令 | 与 ESC 取消平级设置生效，不被 VAD 静音判断阻塞 |
| 设备热插拔 | 会议和录音期间切换 USB 麦克风/耳机后能恢复或明确报错 |
| 睡眠唤醒 | idle 和录音中各执行一次，唤醒后快捷键与采集链路可用 |
| 麦克风权限撤销/恢复 | 不崩溃；权限恢复后能重新开始 |
| Release idle | 300 秒进程存活；CPU 低位稳定；RSS 无线性增长；无 VAD fatal/error 和隐私日志 |
| 损坏 Silero 缓存 | `tools/run_vad_damaged_cache_smoke.sh --app <Voxt.app>` 不崩溃；真实 app 不阻塞启动或停止 |

## 阻断标准

- 任意崩溃、采集死锁、无法停止/取消、重复 final commit、批量空 ASR 都阻断发布。
- 任意原始音频、完整转写、完整提示词或未脱敏用户内容进入日志都阻断发布。
- Automatic / Silero 明显降低转录、翻译、转写、会议任一核心体验时，默认回到更保守参数后再验收。
- Off 模式不能破坏最终 ASR、ESC 取消、语音结束指令或历史写入边界。

## 推荐命令

```bash
tools/run_vad_phase6_smoke.sh --duration 300 --interval 30
tools/validate_vad_acceptance_report.sh --report <report.md> --allow-unsigned
```
