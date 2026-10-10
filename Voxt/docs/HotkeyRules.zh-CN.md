# 快捷键规则核对文档

本文档描述当前代码中的快捷键模型、路由优先级、触发方式、业务动作和已知待确认点。内容基于 `Voxt/Hotkey`、`Voxt/App/HotkeyLifecycle.swift`、`Voxt/App/Recording/RecordingSessionFlow.swift`、`Voxt/App/MeetingSessionFlow.swift` 与现有 `VoxtTests/*Hotkey*Tests.swift`。

## 术语

- 业务：快捷键触发的功能类型，包括转录、翻译、改写、会议、最近结果粘贴。
- Binding：一条快捷键配置，结构是 `hotkey + behavior`。同一个业务可以有多条 binding。
- Hotkey：输入本身，包括键盘键、modifier-only 组合或鼠标按键，以及修饰键。
- Behavior：触发方式，目前有 `tap`、`longPress`、`doubleTap`。
- modifier-only：没有普通键，只由修饰键组成的快捷键，例如 `fn`、右 `Command`、`fn+Shift`。
- common stop：录音或会议进行中，部分快捷键不再按原业务启动，而是统一触发停止/关闭流程。

## 默认快捷键

默认 preset 是 `fnCombo`：

| 业务         | 默认快捷键          | 默认触发方式    |
| ------------ | ------------------- | --------------- |
| 转录         | `fn`                | `tap`           |
| 翻译         | `fn+Shift`          | `tap`           |
| 改写         | `fn+Control`        | `tap`           |
| 会议         | `fn+Option`         | `tap`           |
| 最近结果粘贴 | `Control+Command+V` | 固定按 tap 处理 |

`commandCombo` preset：

| 业务         | 快捷键              |
| ------------ | ------------------- |
| 转录         | 右 `Command`        |
| 翻译         | 右 `Command+Shift`  |
| 改写         | 右 `Command+Option` |
| 会议         | 右 `Command+L`      |
| 最近结果粘贴 | `Control+Command+V` |

`mouseMiddleFnShift` preset：

| 业务         | 快捷键              | 备注            |
| ------------ | ------------------- | --------------- |
| 转录         | 鼠标中键            | `tap`           |
| 翻译         | `fn+Shift`          | `tap`           |
| 改写         | 鼠标中键            | `doubleTap`     |
| 会议         | `fn+Option`         | `tap`           |
| 最近结果粘贴 | `Control+Command+V` | 固定按 tap 处理 |

## 存储模型

每个主业务使用 binding 列表：

- `transcriptionHotkeyBindings`
- `translationHotkeyBindings`
- `meetingHotkeyBindings`
- `rewriteHotkeyBindings`

每条 binding 都有独立的 `behavior`。因此“转录有两条快捷键，一条 `fn` 用 `tap`，一条 `⌘X` 用 `doubleTap`”在模型层是允许表达的。

普通键盘键可以作为无修饰全局快捷键，例如单独的 `F`。这类快捷键会在所有 App 中监听并消费对应按键，因此应避免绑定日常输入频繁使用的字母。modifier-only 快捷键仍必须至少包含一个修饰键；中键及以上鼠标按键不受此规则影响。

旧版单快捷键配置会迁移成一条 binding。为了兼容旧设置，代码仍保存每个业务的第一条 binding 到旧 scalar keys 中，但运行时主要读取 binding 列表。

## 匹配与优先级

事件路由会把所有业务的 bindings 合并排序。排序规则是：

1. 快捷键更具体者优先。
   - 普通键或鼠标按键比 modifier-only 更具体。
   - 修饰键数量越多越具体，例如 `fn+Shift` 比 `fn` 更具体。
   - 开启侧边区分时，带具体左右侧修饰键的配置更具体。
2. 同等快捷键具体度时，触发方式优先级是 `doubleTap` > `longPress` > `tap`。
3. 同等具体度和触发方式时，业务优先级是：翻译 > 改写 > 会议 > 最近结果粘贴 > 转录。

示例：

- 如果转录是 `fn`，翻译是 `fn+Shift`，按 `fn+Shift` 应触发翻译，不应误触发转录。
- 如果两个业务使用同一个快捷键和同一个触发方式，翻译会优先于转录。
- 如果同一个快捷键同时存在 `tap` 和 `doubleTap`，第一次释放会等待系统双击窗口；窗口内没有第二击则触发 `tap`，出现第二击则取消待执行的 `tap` 并触发 `doubleTap`。

## 输入类型规则

### modifier-only

modifier-only 通过 `flagsChanged` 处理。按下时先记录候选状态，释放时根据 behavior 决定是否触发。

- `tap`：通常在释放时触发 down 回调。
- `longPress`：按下后触发 down，释放后触发 up。若存在更具体的 modifier-only 组合，低优先级长按会延迟约 80ms，以等待更具体组合是否出现。
- `doubleTap`：第一次释放只记录候选；第二次释放发生在系统 double-click interval 内才触发 down。

`fn` 有特殊兼容逻辑，因为部分键盘的 `fn` 事件 flags 会抖动。代码会专门识别 `kVK_Function` 和 `.maskSecondaryFn`。

### 普通键盘键

普通键盘快捷键通过 `keyDown/keyUp` 处理。普通键可以不带修饰键，并按精确修饰键状态匹配；例如裸 `F` 不会匹配 `Command+F`。无修饰全局快捷键会消费其它 App 中对应的正常输入，配置时需要谨慎。

- `tap`：keyDown 触发 down，keyUp 触发 up。
- `longPress`：keyDown 触发 down，keyUp 触发 up。
- `doubleTap`：keyDown 不触发业务 down，keyUp 参与 double tap 判定；第二次 keyUp 才触发 down。
- 自动重复 keyDown 不会重复触发。
- tap 普通键必须先收到匹配 keyDown，才会消费匹配 keyUp；孤立 keyUp 不触发。

### 鼠标按键

鼠标快捷键通过 `otherMouseDown/otherMouseUp` 处理。

- `tap`：mouseDown 触发 down，mouseUp 触发 up。最近结果粘贴是例外，它在 release 时触发粘贴动作。
- `longPress`：mouseDown 触发 down，mouseUp 触发 up。
- `doubleTap`：release 参与 double tap 判定，第二次 release 才触发 down。

## 转录规则

### 空闲状态

`tap` 或 `doubleTap` 的转录 down：

- 如果没有活跃 session，开始转录。
- 如果选中文本可直接加入词典，则优先执行词典逻辑，不开始录音。
- 记录本次转录是由哪个 behavior 启动，用于决定录音期间是否启用 common stop。

`longPress` 的转录 down：

- 空闲时先安排一个延迟启动任务。
- 如果在延迟期间释放，则取消启动。
- 如果保持按住到延迟结束，则开始转录。

### 转录中

`tap` 或 `doubleTap` 的转录 down：

- 如果 session 活跃，且不在 tap-stop guard interval 内，并且 session 还没进入停止流程，则停止录音。

`longPress` 的转录 up：

- 如果 pending start 还没执行，则取消 pending start。
- 如果已经开始转录，则停止录音。

### 当前 common stop 行为

开始录音后，如果以下条件之一成立，会启用 common stop：

- 当前 session 是翻译。
- 当前 session 是改写。
- 当前 session 是转录，且启动它的转录 behavior 是 `tap` 或 `doubleTap`。

common stop 的 App 层行为：

- 如果会议活跃，转到会议快捷键处理。
- 如果录音 session 活跃且 mode 是翻译、改写或转录，则停止录音。
- 会取消 pending transcription start。
- 会取消 pending double tap candidate，避免停止后第二下又启动。

当前实现里，`HotkeyManager` 只会把“转录业务的 modifier-only binding”作为 common stop 输入。也就是说：

- 录音中单击 `fn` 这类 modifier-only 转录 binding，可以触发 common stop。
- 录音中单击右 `Command` 这类 modifier-only 转录 binding，可以触发 common stop。
- 录音中单击 `fn+Command` 这类 modifier-only 组合，也可以触发 common stop。
- 录音中单击普通键盘组合 `⌘X` 或鼠标按键形式的转录 binding，目前不会通过 `emitCommonStopIfNeeded` 触发 common stop。

## 翻译规则

翻译 down：

- 先取消 pending transcription start。
- 如果已有 session 活跃，则忽略翻译启动。
- 如果可以走选中文本翻译，则执行选中文本翻译。
- 否则开始麦克风翻译 session。

翻译 up：

- 只有 `longPress` 会处理 up。
- 如果是选中文本翻译流程，不通过 keyUp 停止。
- 如果当前活跃 session 是翻译，则停止录音。

翻译快捷键优先级高于转录。默认 `fn+Shift` 会压过 `fn`，避免组合键触发时误触发转录。

## 改写规则

改写 down：

- 先取消 pending transcription start。
- 空闲时开始改写录音。
- 如果已有 session 活跃，通常忽略。
- 特例：如果刚刚由转录 tap 启动，且仍处于 tap-stop guard interval 内，改写 down 可以把刚开始的转录 session 取消并改为改写。

改写 up：

- 只有 `longPress` 会处理 up。
- 如果当前活跃 session 是改写，则停止录音。

旧的“双击转录快捷键唤起改写”会迁移为改写业务的一条 `doubleTap` binding。启用该旧模式时，全局 legacy trigger mode 会强制为 `tap`。

## 会议规则

会议 down：

- 先取消 pending transcription start。
- 如果会议已活跃：
  - 如果关闭确认框已显示，则再次按会议快捷键会取消关闭确认。
  - 否则显示关闭确认。
- 如果会议未活跃，但普通录音 session 活跃，则提示先结束当前录音。
- 如果会议未活跃且没有普通录音 session，则开始会议。

会议启动成功后会启用 common stop。common stop 触发时，如果会议活跃，会转到会议快捷键处理，即显示或取消关闭确认。

会议快捷键没有通用 keyUp 停止规则；会议结束由确认/取消流程处理。

## 最近结果粘贴规则

最近结果粘贴只在 `customPasteHotkeyEnabled` 为 true 时生效。

- 它没有 binding 列表，运行时包装成一条 `tap` binding。
- 键盘形式的 custom paste 在 release 时触发，避免 keyDown 过早注入。
- 普通键盘 custom paste 会清掉 sided modifiers，避免左右侧修饰键影响粘贴快捷键。
- 它和其它业务的 tap binding 使用相同快捷键时，设置页会给出重复警告。

## 冲突和重复校验

设置页会提示常见系统快捷键冲突：

- `fn+Space`：可能和 Globe / 输入源切换冲突。
- `Command+Space`：Spotlight。
- `Option+Command+Space`：Finder 搜索。
- `Command+Tab`：App Switcher。
- `Command+\``：窗口切换。
- `Command+Q`：退出。
- `Command+H`：隐藏。
- `Command+M`：最小化。
- `Command+W`：关闭窗口。
- `Command+V`：粘贴。

重复校验：

- 同一业务内不能使用相同 hotkey，多条相同 hotkey 即使 behavior 不同也会提示重复。
- 不同业务之间，相同 hotkey 且相同 behavior 会提示重复。
- custom paste 和其它业务的 tap binding 相同会提示重复。

注意：重复校验是设置页提示，不等于运行时完全无法处理。运行时仍有确定性优先级。

## 状态恢复和保护规则

- 快捷键录制中，global hotkey routing 会暂停，避免录制快捷键时触发业务。
- Voxt 自己注入的键盘事件会带特殊 marker，事件路由会忽略它们，避免粘贴/注入结果反过来触发快捷键。
- 事件 tap 被系统禁用时会清 transient state 并尝试重新启用。
- modifier-only tap 在按住期间如果出现普通键，会取消 modifier-only tap candidate，避免 `fn+A` 后释放 `fn` 误触发 `fn` 转录。
- 存在 stale state 恢复逻辑：长时间空闲后检测到旧的 tap 状态，会清理 transient state。
- Escape 在 overlay answer 模式下可关闭 answer；在 tap 模式且普通录音 session 活跃时可取消录音。`longPress` 模式下 Escape 不作为普通录音取消快捷键。

## 关键场景矩阵

| 场景                                                                 | 期望/当前行为                                                                                                             |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 转录绑定 A=`fn tap`，绑定 B=`⌘X doubleTap`，空闲时单击 `fn`          | 当前应开始转录。                                                                                                          |
| 转录绑定 A=`fn tap`，绑定 B=`⌘X doubleTap`，空闲时双击 `⌘X`          | 当前应在第二次 release 后开始转录。                                                                                       |
| 转录绑定 A=`fn tap`，绑定 B=`⌘X doubleTap`，空闲时单击 `⌘X`          | 当前不应开始转录，只记录/等待 double tap candidate。                                                                      |
| 由 `fn tap` 开始转录后，单击 `fn`                                    | 当前应触发 common stop 并结束录音。                                                                                       |
| 由 `fn tap` 开始转录后，单击 `⌘X`                                    | 按目标规则应结束录音；当前实现疑似不会结束，因为普通键 binding 不走 common stop。                                         |
| 由 `⌘X doubleTap` 开始转录后，单击 `fn`                              | 当前应触发 common stop 并结束录音。                                                                                       |
| 由 `⌘X doubleTap` 开始转录后，单击 `⌘X`                              | 按目标规则应结束录音；当前实现疑似不会结束，因为普通键 doubleTap binding 不走 common stop。                               |
| 转录=`fn`，翻译=`fn+Shift`，按 `fn+Shift`                            | 当前应触发翻译，不触发转录。                                                                                              |
| 转录=`fn longPress`，翻译=`fn+Shift tap`，先按 `fn` 后很快加 `Shift` | 当前应触发翻译，并取消/避免转录 longPress。                                                                               |
| 转录=`fn longPress`，没有更具体组合出现                              | 当前约 80ms 后触发转录 down，释放后触发 up/停止。                                                                         |
| 改写=`同一快捷键 doubleTap`，转录=`同一快捷键 tap`                   | 单击在双击窗口结束后触发转录；双击取消待执行的转录并触发改写；录音活跃时单击直接触发 common stop。设置页对跨业务同 hotkey 不同行为不报重复。 |
| 会议已活跃时按会议快捷键                                             | 当前显示关闭确认；如果确认框已显示则取消确认。                                                                            |
| 会议活跃时尝试开始普通转录/翻译/改写                                 | 当前会阻止非会议录音，并重新显示会议 overlay。                                                                            |

## 当前待核对/待修复点

1. 目标规则是否是：“录音中，所有转录 bindings 都应该退化为单击停止键，不论原 behavior 是 `tap`、`doubleTap` 还是 `longPress`，也不论输入是 modifier-only、普通键还是鼠标键”？
2. 如果上一条成立，当前实现只覆盖了 modifier-only 转录 binding；普通键组合如 `⌘X doubleTap` 作为停止键的场景需要补测试并修复。
3. `longPress` 转录 binding 在录音中是否也允许“单击/短按结束”，还是必须保持 release-to-end 语义？目前 App 层 `longPress` 主要靠 keyUp 停止。
4. 同一业务内相同 hotkey 但不同 behavior，设置页当前会报重复；运行时可以 deterministic 处理。是否要继续禁止这种配置，需要产品确认。
5. 跨业务相同 hotkey 但不同 behavior，当前设置页不按重复处理；运行时按 behavior priority 处理。是否允许这种配置，也需要产品确认。

验证

1. 点按没有问题
2. 长按没有问题
3. 双击没有问题
4. 单功能，多个快捷键，都是点按，没有问题

问题

1. 转录 fn（点按），right command（长按），fn 点按下没反应，没唤醒（异常），right command 长按 唤醒了，但是松开按键后，未结束（异常）
1. 转录 fn（点按），right command（双击），fn 点按下没反应，没唤醒（异常），right command 双击 唤醒了，单击 right command 结束（正常）
