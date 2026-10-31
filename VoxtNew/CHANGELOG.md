# Changelog

All notable changes to VoxtNew. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/). Each version's section is what the update window and actnow.dev show: English first, then the same in Chinese under `### 中文`, with `####` headings below it (docs/release.md §2.1).

## [Unreleased]

### Added

- History entries have a context menu: right-click (or Control-click) one to copy its result or its recognised text, show its details, show its recording in Finder, or delete it. Home's recent results have the same menu. In History a selected entry also takes ⌘C (copy) and Delete.
- **Keep models loaded** (Local models → Advanced): how long a model stays in memory unused. Automatic, the default, is as before (speech 10 minutes, language models 5); you can choose 5, 15 or 30 minutes, 1 hour, or until VoxtNew quits, so that every dictation starts at once. Other ActNow apps can still ask VoxtNew to free the memory of models it is not using. A change applies at once.
- **Sounds to choose** (Settings → Audio → Play sounds): the start and the stop sound are each chosen from the ActNow family's sound library (from React Bits Sound Lab, in the builds from actnow.dev; Tally and Wrap by default), or macOS's system sounds, or none. A compact picker searches them (names and tags), filters by type, plays each with ▶ (again to stop) and sets their volume in steps; a Try button plays the choice. A Voxt pair chosen in an earlier build is kept as those system sounds. A build from source without the library plays Pop and Tink.
- **Mute system audio while recording** (Settings → Audio, on by default): the current output device is muted from just after the start sound until the recording stops, so that videos, music or calls are not recorded; a device you muted stays muted, one you unmute during the recording stays on, and a device that cannot be muted leaves the recording as it is (the overlay says so). No permission is needed. If VoxtNew or its helper stops while muting, the sound comes back at once, or at the next start.
- **A notice when your Mac is short of memory**: when the language model was slow, or did not answer in time, while the Mac's swap was over 90% full, the overlay says "Low memory slows the model: quit unused apps." after the text went in. It shows at most once each time VoxtNew runs.
- **History settings** (the button beside History's search field): keep the recordings (the same switch as Settings → Privacy), choose the folder they are kept in (the recordings already kept move there, each one copied and checked before the original is deleted; if the folder is on a drive that is not connected, new recordings go to the default folder until it is back), see how much room they take, export them (all, or those of the tab shown) into a folder you choose, named by their date and first words, or delete them all and keep the texts. **Automatic clean-up**, on as in Voxt: entries older than 3 months are deleted with their recordings; you can choose 1 week to 1 year, keep at most 50 to 1000 entries of each feature, or turn it off. A new setting that would delete entries now asks first, with their number.
- **Retry** a history entry from its recording: a failed entry (or one with no text) shows Retry on its row instead of Failed, in its details and in its context menu; other entries with a recording have Transcribe again. The entry is updated in place (text, status, time); nothing is pasted. The overlay's notice of a failed recognition opens the entry and retries it.
- **Long dictations are recognised while you speak**: after a minute, finished stretches are recognised in the background, cut at pauses, so Stop leaves only the last half-minute or so (on a busy Mac, 104 s of speech: 26 s → 2.4 s after Stop).
- **The overlay says what a recording is waiting for.** When the microphone is slow to start (the first dictation after VoxtNew starts on a Mac short of memory), the microphone icon pulses inside a ring and the overlay says "Starting the microphone…": what you say before the bars light up is not recorded. While the speech model is still loading (the first dictation, or after it was unloaded), only the icon pulses and the bars keep showing your voice: the recording works, and the live text comes once the model is ready.

### Changed

- **Models are kept in ~/Library/Application Support/ActNow/models**, shared with the other ActNow apps, instead of the Hugging Face cache (~/.cache/huggingface), which cleaning tools treat as a cache. Models already there are brought over at the first start without downloading them again (hard links: the cache stays as it is). Time Machine leaves models out; Settings → Storage → "Back up models with Time Machine" turns it on for all ActNow apps.
- The note of which output device VoxtNew muted is kept in Application Support instead of ~/Library/Caches, so that a cache cleaner run after a crash cannot leave your speakers muted.
- With a dictionary of more than 30 terms, clean-up, translation and rewriting answer sooner: the first words come 1.4–1.9 s sooner on the balanced language model and 0.7 s on the fast one. The terms that a dictation mentions now go with the dictation instead of changing the model's instructions each time.
- The fast speech model gets the 30 most recently changed terms of your dictionary. A longer list made it slower and less accurate; the accurate model still gets up to 500.
- **The result goes where you are typing when it is ready.** If you switch to another app during or after the recording and click into a text field there, the text is inserted there; the app you started in is not brought back. Nothing is typed into a password field or into VoxtNew's own windows, and when no text field is focused (Finder's desktop, a web page you are reading), the result waits in the answer card with Insert and Copy, which now also inserts into the app in front. A selection is still replaced only in the app it was selected in, and app enhancement's auto key (Return…) is pressed only there. The history shows the app the text went into.

### Fixed

- A result that was pasted but did not show up (in a web page in Arc) is no longer lost without a word. VoxtNew now checks that the focused text field changed after the paste. When it did not, the result waits in the answer card and stays on the clipboard (press ⌘V), the auto key is not pressed, and the card says so; when another app had secure input on (a password field, a terminal's Secure Keyboard Entry), the card says that. Where the field cannot tell (some web editors), the text is pasted as before. Your clipboard now comes back as soon as the text shows, instead of half a second later.
- A long dictation no longer fails after 30 seconds of recognition and loses everything (a 4m51s dictation on a Mac short of memory). Recognition now gets 60 s plus 1 s per second of recording; a model that stops answering is still caught after 15 s. If full recognition fails anyway, the live text is used (cleaned up and inserted as usual) and the overlay says so.
- The speech model uses much less memory on long recordings: under 4 GB for a 5-minute dictation (Accurate model) instead of up to 10 GB, which pushed your other apps into swap and slowed the Mac. Recognising a recording again is faster, too.
- When recognition fails or finds no text, the entry keeps the recording for 7 days even when Keep the recordings is off, so that you can retry it; a successful retry deletes it.
- Dictionary terms no longer appear at the end of a dictation where you did not say them ("… 嗯。Voxt, Omni, SGLang."). The speech model wrote out the dictionary it was given when part of a recording had no speech, such as the pause before Stop at the end of a long dictation. Such an ending is now checked against the same audio recognised without the dictionary, and terms you did not say are removed; terms you did say stay, also at the end of a sentence.

### 中文

#### 新增

- 历史记录有了右键菜单：右键（或按住 Control 点）一条记录，可以复制结果或识别原文、查看详情、在访达中显示录音，或者删除它。首页的最近结果也有同样的菜单。在历史页选中一条记录后，也可以按 ⌘C 复制、按 Delete 删除。
- **保持模型载入**（本地模型 → 高级）：模型闲置多久后从内存卸载。默认“自动”与以前相同（语音识别 10 分钟、语言模型 5 分钟）；也可以选 5、15、30 分钟、1 小时或“直到退出 VoxtNew”，让每次口述都立即开始。其他 ActNow app 仍然可以请 VoxtNew 释放闲置模型占用的内存。修改立即生效。
- **可选的提示音**（设置 → 音频 → 播放提示音）：开始和结束提示音分别从 ActNow 家族提示音库里选（来自 React Bits Sound Lab，只在 actnow.dev 的官方构建里；默认“开录”和“收工”），也可以选 macOS 系统声音或无声。选择器很紧凑：可以搜索（名字和标签）、按类型筛选、用 ▶ 逐个试听（再点一下停止）、按档位调音量；试听按钮播放当前选择。以前版本选的 Voxt 预设保留为对应的系统声音。没有提示音库的源码构建播放 Pop 和 Tink。
- **录音时静音系统声音**（设置 → 音频，默认开启）：从开始提示音之后到停止录音为止，当前输出设备静音，免得视频、音乐、通话的声音被录进去；你自己静音的设备保持静音，录音中你取消静音后不会再被静音，不支持软件静音的设备照常录音（浮窗会提示）。不需要任何权限。VoxtNew 或辅助进程在静音期间意外退出时，声音会立即恢复，最迟在下次启动时恢复。
- **内存不足时的提示**：Mac 的 swap 用了九成以上、语言模型因此变慢或没有及时响应时，文字插入之后浮窗提示“内存不足，模型变慢：可退出不用的应用。”。每次运行 VoxtNew 最多提示一次。
- **历史设置**（历史页搜索框右边的按钮）：保存录音（和 设置 → 隐私 是同一个开关）；选择录音存放的文件夹（已有录音会一起移过去，每段先复制并校验再删原文件；文件夹在没接上的外接盘上时，新录音先存到默认文件夹，等它回来）；查看录音占用的空间；把录音（全部，或当前 tab 的）导出到你选的文件夹，以日期和开头几个词命名；或者全部删除、保留文字。**自动清理**，和 Voxt 一样默认开启：超过 3 个月的记录连同录音一起删除；可以选 1 周到 1 年，或每种功能最多保留 50 到 1000 条，也可以关掉。新设置会立即删掉已有记录时，先告诉你会删几条并请你确认。
- **重试**：用历史记录保存的录音再识别一次。失败或没有文字的记录，行上的“失败”换成“重试”，详情和右键菜单里也有；其他有录音的记录是“重新识别”。结果原地更新这条记录（文字、状态、用时），不会粘贴。浮窗里识别失败的提示，点一下打开这条记录并重试。
- **长口述边说边识别**：说到 1 分钟以后，已经说完的部分在停顿处切开、在后台识别，停止后只剩最后半分钟左右（Mac 内存紧张时，104 秒口述停止后的等待 26 秒 → 2.4 秒）。
- **浮窗说明录音在等什么。** 麦克风启动慢时（Mac 内存紧张时 VoxtNew 启动后的第一次口述），麦克风图标在一个圆环里闪动，浮窗显示“正在启动麦克风…”：波形亮起之前说的话不会被录下。语音识别模型还在加载时（第一次口述，或者模型被卸载之后），只有图标在动，波形照常跟着你的声音：录音不受影响，模型就绪后实时文字随即出现。

#### 变更

- **模型放在 ~/Library/Application Support/ActNow/models**，与其他 ActNow app 共用，不再放在会被清理工具当成缓存的 Hugging Face 缓存（~/.cache/huggingface）里。那里已有的模型在首次启动时直接拿过来（硬链接，原缓存不动），不用重新下载。Time Machine 默认不备份模型，可在 设置 → 存储 →“用 Time Machine 备份模型”中为所有 ActNow app 打开。
- VoxtNew 静音了哪个输出设备的记录改放在 Application Support，不再放在 ~/Library/Caches，崩溃后即使清理了缓存，扬声器也不会一直静音。
- 词典超过 30 条时，整理、翻译和转写更快开始输出：均衡语言模型快 1.4–1.9 秒，快速模型约快 0.7 秒。口述里提到的词条现在随口述一起发给模型，不再每次改动模型的指令。
- 快速语音识别模型只使用词典里最近修改的 30 条。更长的列表会让它更慢、更不准；精准模型仍可使用最多 500 条。
- **结果插入到它出来时你正在输入的地方。** 录音中或录完后切到另一个 App、点进那里的输入框，文字就插到那里，不会再切回开始录音的 App。密码框和 VoxtNew 自己的窗口里不会输入任何东西；没有选中输入框时（访达的桌面、正在看的网页），结果留在答案卡片里，可以插入或复制，卡片的“插入”现在也插进当时在前台的 App。替换选区仍然只在选中文字的那个 App 里进行，应用增强的自动按键（Return 等）也只在那里按。历史记录仍记在开始录音的 App 下，详情里另外写明文字插入到了哪个 App（“已插入到 X”）。

#### 修复

- 粘贴了却没有出现的结果（Arc 里的网页）不会再悄无声息地丢掉。VoxtNew 现在会确认粘贴后焦点所在的输入框确实变了。没有变时，结果留在答案卡片里，并且留在剪贴板上（按 ⌘V 即可粘贴），不按自动按键，卡片里写明原因；另一个 app 开启了安全输入（密码框、终端的“安全键盘输入”）时，卡片会说明这一点。输入框无法告知内容的地方（部分网页编辑器）照常粘贴。原来的剪贴板在文字出现后立即恢复，不再固定等半秒。
- 长口述不会再因为识别超过 30 秒而失败、什么都没留下（内存紧张的 Mac 上一段 4 分 51 秒的口述）。识别时限改为 60 秒加录音每秒 1 秒；模型停止响应仍在 15 秒内发现。完整识别仍然失败时，使用实时识别的文字（照常整理、插入），浮窗会说明。
- 长录音时语音识别模型占用的内存大大减少：5 分钟的口述（精准模型）不到 4 GB，以前最多 10 GB，会把其他应用挤进 swap、拖慢 Mac。重新识别一段录音也更快了。
- 识别失败或没有识别出文字时，即使“保存录音”关着，这条记录也保留录音 7 天，方便重试；重试成功后删除。
- 词典里的词不会再出现在你没说过它们的口述末尾（“……嗯。Voxt，Omni，SGLang。”）。录音中没有说话的部分（比如长口述结束前、按停止之前的停顿）会让语音模型把给它的词典原样写出来。现在这样的结尾会和不带词典识别的同一段音频核对，删掉你没说过的词条；说过的词条照常保留，在句末也一样。

## [0.1.0-beta.1] - 2026-10-17

### Added

- The first version of VoxtNew: voice input for macOS on local models only. Press a shortcut in any app, speak, and the text appears where your cursor is. Audio and text never leave your Mac.
- Local speech recognition: two models, accurate (recommended) and fast, with the text shown live as you speak, and a voice detector that sends only speech to them.
- Local language models: a balanced one (recommended), a fast one and a translation model, for clean-up, translation and rewriting.
- Transcribe, with an optional clean-up of fillers, false starts and punctuation; Translate, the selected text or what you say; Rewrite, the selected text or new text from a spoken request, with follow-ups in the answer card.
- App enhancement: groups of apps and websites with their own scene prompt and a key pressed after inserting.
- Dictionary: its terms help the speech model hear your names and jargon, and the language model write them as you spell them.
- The Local models page: download, pause and resume, every file verified, delete; the memory a loaded model uses. Downloads use the system's proxy, or a mirror when Hugging Face cannot be reached.
- A setup guide that installs the two recommended models and tries each feature; history, the overlay, the menu bar, shortcuts and settings.
- Shortcuts: tap the right Option key to transcribe, right Option with Shift to translate, Control with right Option to rewrite; Control-Option-V pastes the last result. They leave fn to Auralink, so the two apps run side by side.
- Updates from actnow.dev, with a Beta channel (Settings → About). The app is signed and notarised.

### 中文

#### 新增

- VoxtNew 的第一个版本：只用本地模型的 macOS 语音输入。在任意 App 里按下快捷键说话，文字出现在光标处。音频和文字不会离开你的 Mac。
- 本地语音识别：精准（推荐）和快速两个模型，说话时实时显示文字；声音检测只把说话的部分交给它们。
- 本地语言模型：均衡（推荐）、快速和翻译专用三个模型，用于整理、翻译和改写。
- 转录，可以整理掉口头语、说了一半的话并补上标点；翻译，选中的文字或说的话；转写，改写选中的文字或按语音要求写新内容，并可在答案卡片里继续追问。
- 应用增强：把 App 和网站分组，每组有自己的场景提示词和插入后的按键。
- 词典：词条帮助语音识别模型听准你的人名和术语，并让语言模型照你的写法输出。
- 本地模型页：下载、暂停和继续下载，每个文件都经过校验，可删除；显示已加载模型占用的内存。下载走系统代理，连不上 Hugging Face 时改用镜像。
- 设置引导：安装推荐的两个模型，并把每个功能试一遍；还有历史、浮窗、菜单栏、快捷键和设置。
- 快捷键：单按右 Option 转录，右 Option 加 Shift 翻译，Control 加右 Option 转写；Control-Option-V 粘贴上一条结果。fn 留给 Auralink，两个 app 可以同时使用。
- 从 actnow.dev 更新，有 Beta 通道（设置 → 诊断与关于）。app 已签名并经过公证。
