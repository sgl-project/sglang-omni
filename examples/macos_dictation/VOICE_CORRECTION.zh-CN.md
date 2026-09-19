# 双击 Option 语音更正

完成一次听写后，在原输入框连续轻按同一侧的 Option 两次，说出修改意见，再双击 Option 结束。浮条上的停止按钮也能结束录音。示例：“把张三改成张珊，珊瑚的珊”“把下午两点改成下午三点”。成功后提示“已更正”，用户仍需自行发送。

此功能独立于“轻度整理”开关，使用设置中的本机 ASR 转写修改意见。取消、明确整段删除、唯一匹配的拼写以及原词和新词都带引号的字面替换可在客户端执行；普通替换、局部删除和语义编辑使用设置中的本机纠错模型。个人术语背景只在其开关开启时参与更正。没有上一段、所需服务不可用或修改意见不明确时，保留原文并显示原因。

个人背景是参考，不是每次都必须替换成的答案。本次明确拼出 `S G L A N G` 时使用 `SGLang`，不会因为背景里写了 `SGLang-Omni` 就补上 `-Omni`；说“按背景里的项目名改”时，才允许使用背景中的完整名称。唯一的拼音匹配可由规则完成；背景或原文包含多个匹配时，应明确指定。未执行修改意见的结果不等于已更正。

普通“轻度整理”仍只放行明确的“错词 → 正确词”映射；仅填写一个项目名不会授权自动改写其他词。主动语音更正使用独立的编辑规则。

## 输入位置与替换

- 普通听写真正粘贴前，尝试读取当前输入框、窗口和光标范围；粘贴后验证实际文字。读取失败仍可普通粘贴，但后续更正只能提供结果供复制。
- 粘贴确认依据记录的偏移、完整听写片段和前后内容，不要求光标仍停在段尾。兼容编辑器在段落外添加或移除一个末尾换行；听写片段本身必须逐字匹配，不忽略其换行，不搜索其他位置。
- 若粘贴前的占位内容或选区快照与实际不一致，但粘贴后同一输入框的内容发生变化，且全部内容恰好等于本轮听写原文（允许段落外多一个末尾换行），则将该段位置确认为从头开始。仍只替换更正的片段；包含其他文字或重复段落时，不通过搜索猜测位置。
- 只记录最近一次听写段落的位置。新一轮普通听写、清空或退出会移除该记录；窗口渐隐不会移除。安装新版本前已粘贴的文字没有位置记录。
- 更正开始时核对原输入框及该段内容。允许用户已在这段之后追加文字，但不会替换追加部分。
- 更正处理期间监听输入框和窗口变化，写入前再次校验。输入内容被改写、切换输入框、原窗口关闭、权限失效或定位失败时，提供更正结果供复制。
- 设置更正选区后，会短暂等待辅助功能接口读回该选区，再发送粘贴；等待期间继续核对原文和焦点。超时会显示预期/实际选区并明确尚未粘贴，避免将界面更新延迟直接误判成用户改变选区。
- 客户端按 Unicode 字素边界计算最小连续改动区间，再换算为 UTF-16 选区。多处修改用一次连续替换，保留输入框中的其他文字。例如“张三→张珊”只替换“三”。
- 非空替换使用一次向目标进程发送的 Cmd+V，并读回确认；确认失败不自动重试。剪贴板仍由本次操作持有时，尝试将临时片段更新为完整更正文本；用户的新复制优先。
- 纯删除只在输入框支持设置 AXSelectedText 时操作已校验的选区，不发送估算数量的退格，也不使用 Cmd+A。纯删除不会修改剪贴板。明确删除整段后，没有可继续更正的上一段。

辅助功能接口与跨进程输入不是原子事务。富文本、终端、密码框以及不暴露完整文本和可写选区的编辑器不在自动更正的首版保证范围内；按实际接口能力降级，不能从一个编辑器通过推断全部应用兼容。

## 快捷键与取消

两次完整的 Option 按下/松开，单次按住不超过 250 ms，间隔不超过 350 ms 才触发。左右 Option 均可，但混用两侧、长按、Option+Space、其他组合键和鼠标操作会清除候选手势。这些数值是交互配置参数，不是性能观测。

普通听写录音、识别或回填期间不能同时启动更正。更正识别、模型请求和回填期间忽略重复手势。更正期间的 Control+Shift+Escape 或浮条关闭按钮取消更正；已发送给输入框的修改不会被盲目撤销。

全局键盘监听需要辅助功能权限。授权或更新应用后，可在 Omni 设置中点击“刷新”，重新注册监听。

## 模型契约与隐私

需要模型时，只发送上一段文本、修改意见和已启用的术语背景，不发送输入框中的其他段落。使用独立纠错 Prompt 和 JSON Schema，允许用户明确要求的用词、数字或否定表达修改，不套用轻度整理的保真过滤规则。

模型输出 `scope` 和 `text`：

- `local`：局部修改，`text` 是修改后的完整上一段，例如“把张三改成李四吧”。
- `rewrite`：整段重写，`text` 是重写后的完整上一段，例如“整段合成一句话”“整段翻译成英文”。整段仅指上一次听写，不包括输入框其他内容。
- `clarify`：无法确定怎么改，`text` 是澄清问题，仅显示在 Omni 中，不会写进输入框。用户重新录制更具体的指令；当前没有多轮澄清对话历史。

提示词包含四组固定的 user/assistant 示例，覆盖原文缺字段、含糊指令、单字更正和明确拼写优先于背景。示例不来自录音历史。真实请求始终位于最后，保留中文标签；`think=false`、`temperature=0`，输出受 JSON Schema 约束。

`CorrectionPlan` 对能识别的字面目标检查修改边界：匹配数字 `300` 不会命中 `1300`；“第二个张三”只定位第二处；没有指定位置而存在多个相同字面目标时要求明确。冒号格式的“负责人”“备注”等字段定位其值，保留标签、空白和句末标点。找不到带引号的原词时直接拒绝；短字面原词没有精确匹配或同音匹配时也拒绝，例如原文只有“杭州和苏州”，不能执行“把南京改成北京”。唯一的精确同音候选可以约束模型改动，但不直接替换。原文没有可识别的电话或邮箱证据时，不能凭空新增字段。原词和新词都用引号明确指定时，可直接完成替换，不请求模型。

识别到局部目标后，模型不得改变目标之外的 UTF-16 序列，也不能用 `rewrite` 绕过该边界。识别到“最后一个字是珊瑚的珊”等单字要求时，检查差异只涉及一个字且新字正确；原文还没有这个字却原样返回时，也视为未执行要求。复杂指代、没有明确字面锚点的局部编辑和重写意图仍依赖模型判断，不能保证所有无关修改都能拦截。

修改意见只用于编辑，不能追加或复述成正文。客户端检查完整指令复述、额外代码围栏、局部越界和明确替换文字。内容检查失败时，携带具体原因最多重新生成一次，仍使用同一份原文和指令；第二次仍失败则保留原文。网络错误、畸形响应和澄清不会重试。两次生成都发生在输入框写入之前；粘贴未确认时不会重新粘贴。

对于完整的“ASCII 名称是逐字母拼写”指令，客户端先检查唯一的“拉丁字母 + 汉字拼音”精确匹配。例如 `S.G.浪` 与 `S G L A N G` 完全对应时直接更正，不等待模型。背景只帮助选择同一拼写的大小写，不补充后缀；明确说“按背景里的项目名改”时，才允许使用背景中的完整名称。该规则不做模糊编辑距离匹配，也不代表模型独立纠错能力。

也支持完整的“名称的后缀是逐字母拼写”说明，如 `S.G. Lang 的 Lang 是 L.A.N.G.`；只在能唯一匹配原词时直接更正。如果指定原文内部片段，如 `S G浪的浪是 L A N G`，则只改“浪”，保留前面的 `S G` 和空格；想统一整个名称时需给出完整拼写。

在设置中选择已安装的模型即可更换纠错模型，轻度整理共用该配置。公共安装脚本默认仍是 `openbmb/minicpm5-2b:q4_K_M`；最终版 Q8 的准确标签是 `openbmb/minicpm5-2b:q8_0`，不是 `:q8`。MiniCPM5-2B-SFT 是另一训练版本，本地导入后需填写 `ollama list` 中的实际名称；本机别名和权重不会随 PR 发布。

输入设有保守的字节预算，过长的上一段、意见或术语背景会被拒绝并提示精简，避免默默截断。音频、上一段、输入框快照和纠错结果只保存在进程内存中，不写入偏好设置或日志。

## 代码与验证

- `DictationCore/CorrectionSession.swift`：上一段、独立更正状态、取消和迟到结果隔离。
- `DictationCore/OllamaCorrector.swift`：Prompt、结构化请求与响应校验。
- `DictationCore/CorrectionPlan.swift`：明确指令、精确拼写匹配、局部边界、重写与输出校验。
- `DictationCore/SpokenSpelling.swift`：明确字母拼写的格式处理。
- `DictationCore/TextRevision.swift`：文本差异和段落范围校验。
- `DictationCore/DoubleOptionGesture.swift`、`DictationApp/DoubleOptionShortcut.swift`：手势规则及系统事件接入。
- `DictationApp/EditableTextAnchor.swift`：原输入框身份、选区替换、读回确认。
- `DictationApp/CorrectionViews.swift`：录制修改意见、结果和复制入口。

```bash
bash examples/macos_dictation/correction_test.sh
bash examples/macos_dictation/revision_target_test.sh
bash examples/macos_dictation/correction_integration_test.sh
bash examples/macos_dictation/correction_feedback_test.sh
bash examples/macos_dictation/ollama_correction_test.sh
bash examples/macos_dictation/correction_plan_test.sh
bash examples/macos_dictation/correction_instruction_echo_test.sh
bash examples/macos_dictation/spelling_correction_test.sh
bash examples/macos_dictation/background_correction_test.sh
bash examples/macos_dictation/revision_confirmation_test.sh
bash examples/macos_dictation/revision_anchor_compatibility_test.sh
bash examples/macos_dictation/revision_selection_delay_test.sh
```

离线测试不使用麦克风、真实输入事件或模型服务器。还需在常用输入框实测双击手感、同音字更正、删除、连续更正，以及处理中切换窗口和输入新内容的降级行为。

后缀拼写例子现在由客户端直接处理；即使设置 `OMNI_CORRECTION_LIVE_TEST=1`，命中明确规则时也不会请求模型。

已启动本机 Ollama 时，可选运行合成文本检查：

```bash
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/correction_live_test.sh
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/background_correction_test.sh
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/correction_accuracy_test.sh
OMNI_CORRECTION_LIVE_TEST=1 OMNI_CORRECTION_EXTRA_CASES=1 bash examples/macos_dictation/correction_accuracy_test.sh
OMNI_CORRECTION_LIVE_TEST=1 bash examples/macos_dictation/semantic_editing_live_test.sh
```

该检查使用默认模型（可用 `OMNI_CORRECTION_MODEL` 和 `OMNI_CORRECTION_URL` 覆盖）验证指定的人名、数字和否定表达修改，不读取个人偏好，不代表整体纠错准确率或真实麦克风、输入框端到端验收。

`correction_accuracy_test.sh` 的两组固定回归案例分别覆盖既有失败模式和不同措辞，默认各重复三轮，报告完整输出精确匹配及耗时的 mean ± sample std。明确规则、模型结果以及预期澄清均计入整条纠错流程；网络或格式错误不能算正确澄清；它不是模型单独准确率。可通过 `OMNI_CORRECTION_SOURCE_DIR` 指向旧版 `DictationCore` 源码目录，在相同模型环境运行对照。比较耗时时应串行运行，排除模型首次加载，不与其他推理或编译任务争用资源。

也可以用原始听写和修改意见两个本地音频文件测试实际识别、整理、纠错及内存中的片段替换：

```bash
OMNI_CORRECTION_AUDIO_TEST=1 bash examples/macos_dictation/audio_correction_test.sh \
  --saved-settings /absolute/path/to/original.m4a /absolute/path/to/correction.m4a '预期完整结果'
```

该命令会读取已保存的模型配置、整理开关和已启用背景，将两个文件发送给本机服务，并向终端打印转写和结果；不录音、不读取或写入真实输入框，也不修改偏好。去掉 `--saved-settings` 时使用默认配置且关闭整理及个人背景；最后的预期结果参数可省略。此检查不能代替目标应用中的自动替换验收。
可用 `OMNI_CORRECTION_MODEL` 临时指定本机已经安装的另一个文本模型，仅影响此次测试的整理和更正，不保存到应用配置。

官方接口依据：[AppKit 事件监听](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/EventOverview/MonitoringEvents/MonitoringEvents.html)、[Ollama 结构化输出](https://docs.ollama.com/capabilities/structured-outputs)。
