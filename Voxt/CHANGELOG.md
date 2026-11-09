# Changelog

All notable changes to Voxt. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/). Each version's section is its release notes: English first, then the same in Chinese under `### 中文`, with `####` headings below it (docs/release.md §2.1).

## [Unreleased]

**How a recording starts.** Pressing the shortcut starts the microphone at once; Voxt does not wait for the speech model. The speech model and the voice detector load in the background while you speak, on the first dictation after Voxt opens and again after ten idle minutes unload them. Everything you say after the microphone has started is recorded and kept in memory until the model is ready, then sent to it: the live text appears later, but the result has all of it, up to the ten-minute limit of one recording. If the overlay says "Starting the microphone…", the microphone has not produced sound yet and words said then are not recorded; while only the icon pulses and the bars follow your voice, the model is still loading and the recording is fine.

### Added

- **The preview pill keeps only live speech recognition text** while you record and while recognition finishes. The offline Final and the language model no longer replay in the pill; results still insert or show in the answer card.
- **Editable full system prompts** (Transcribe, Translate, Rewrite → Your rules): edit the whole system prompt in one field (default shows the built-in text). What you save is what Voxt sends. Reset restores the built-in prompt. A preview shows the example user message used for cleanup and formatting when you use the built-in layers. **Settings search**: ⌘F in Settings focuses the search box and filters by title, description, and English keywords.
- **Setup guide: language model on Translate and Rewrite try steps** (FB22): when no language model is set up, the step explains what is missing and offers **Set up a language model** (the guide stays open). Speech model downloads show progress on the models and try steps; you can continue while a model downloads.
- **While you record, the overlay pill names the mode** (Transcribe, Translate or Rewrite) next to the icon. The shortcut you pressed to start sets the mode; pressing another feature's shortcut while recording only stops, the same as Voxt 1.x common stop (`hotkey.Route`).
- **Notes** (the new Notes page), as in Voxt 1.x: tap right ⌘ in any app, speak, and tap it again. Voxt saves what you said as a note instead of typing it. The Notes page lists your notes, newest first, with their time, and you can delete them. The shortcut changes on that page (off, or other keys); if another shortcut already uses right ⌘, the note shortcut starts off: pick keys on the Notes page. Notes stay out of History, and clearing History keeps them; a note whose recognition failed, or that you cancelled, stays in History with its recording, so you can retry it.
- **Shortcut presets** (Settings → Shortcuts → Preset; also in the setup guide), as in Voxt 1.x: Right ⌥ (the default), fn (Auralink's keys), ⌘ combination (right ⌘, right ⌘⇧, right ⌘⌥, ⌃⌘V), or Middle mouse + fn⇧ (middle click to transcribe, double middle click to rewrite; fn⇧ to translate) set Transcribe, Translate, Rewrite and Paste the last result at once. A shortcut you changed yourself stays as it is. With fn, the page says that Auralink and macOS's Globe key react to fn too.
- **Mouse buttons as shortcuts**, as in Voxt 1.x: in a shortcut's Custom… field type Mouse3 (the middle button), Mouse4 or Mouse5 (the side buttons most mice have) up to Mouse32, alone or with modifiers (Option+Mouse4). Tap, double tap and long press work as with keys, and the button's clicks no longer reach the app in front. Like fn and single modifiers, mouse buttons need Accessibility.
- **Fallback model for Translate** (Translate → Models): when the language model fails, Voxt translates once more with the model you choose here. History names the model that gave the translation.
- **Press the keys to set a shortcut**, as in Voxt 1.x: Custom… on the Transcribe, Translate, Rewrite and Notes pages (and for pasting the last result and cancelling) now takes the keys you press, such as ⌥Space, and saves them at once; your shortcuts pause while it listens. With Accessibility, the helper's **hotkey.record** also captures fn, a left or right modifier and mouse buttons (modifier-only chords too); without it the in-app field still records plain keys and you can Type the keys instead. Shortcuts show Esc, Space, Return, Delete and Tab by name instead of ⎋ ↩ ⌫ ⇥, and Esc is in the Cancel transcription list.
- **The answer card switch** (Answer card on the Transcribe and Translate pages): Show the result in a card applies to transcribe and translate and is the same switch on both pages. Off, the default, results type in or replace the selection; Rewrite's answers still always wait in the answer card, as in Voxt 1.x, with follow-ups in the card. Who had it on for Translate keeps it on. Before, turning it off in Translate left Rewrite's card with no way to turn it off, and Transcribe had no such option.
- **Recording goes on when the microphone goes away**, as in Voxt 1.x: unplug the microphone in use while you speak and Voxt records on with the first connected one in Settings → Audio's list, else the system's default input; the overlay says "Switched the microphone to X." for a moment. The recording ends only when no input is left.
- **Hide a speech model** (Speech models): a model's card has Hide, which leaves the model out of the features' speech model menus; the card moves to Hidden models at the foot of the page, where Show puts it back. A model a feature uses cannot be hidden.
- **The update from Voxt 1.x is tested on a clean Mac**: a GitHub Actions workflow (Voxt Upgrade Test) installs Voxt 1.x, lets it update itself to the new Voxt from the beta channel, then checks the installed app (its bundle ID, Developer ID signature and notarization) and that it imports Voxt 1.x's dictionary without asking you to choose a folder.
- **Voxt 1.x on Apple silicon updates to Voxt 2 by itself**: each release also has the app as `Voxt-X.Y.Z-macOS.zip`, which Voxt 1.x's in-app update installs on a Mac with Apple silicon (first for those who turned on Beta Updates, then for everyone there) and the Homebrew cask (`brew install --cask hehehai/tap/voxt`, Apple silicon only) installs. Voxt 1.x on an Intel Mac is not offered Voxt 2, which does not run there, and stays on Voxt 1.x. After the update, allow the microphone and accessibility again: macOS sees Voxt 2 as another app.
- **Proxy user name and password** (Speech models → Network… → Custom), as in Voxt 1.x: a custom proxy (HTTP, HTTPS or SOCKS5) takes a user name and a password, sent only when both are set, with model downloads and the requests to remote services. The password is kept in the Keychain only, and only Voxt uses them: the other apps that share the model library do not get them.
- **Announcements**, as in Voxt 1.x: when Voxt starts and each time you switch to it (at most every 10 minutes), it reads Voxt's announcements. A new one shows as a card above Settings in the sidebar: Read shows it whole, and once you dismiss it, it does not come back. The request sends nothing about you (no version, language or identifier), and Settings → Privacy says so.
- **Add selected words to the dictionary**, as in Voxt 1.x: select a few words in any app (at most five, or ten Chinese, Japanese or Korean characters in a row) and press the transcribe shortcut. Voxt adds them to the dictionary instead of recording, and the overlay says "Added to the dictionary" or "Already in the dictionary". With a longer selection or nothing selected the shortcut records as usual, and with the dictionary off it always records.
- **Dictionary suggestions**: terms Voxt suggests appear above the list on the Dictionary page, with what it heard instead and where the suggestion came from, and the sidebar shows how many wait. Add puts the term in the dictionary without aliases, Edit… lets you change it first, and Ignore hides it for 90 days.
- **How the dictionary works** (Dictionary): a folded guide on the page says where a term applies: speech hints, the language model's glossary, and aliases replaced in the recognised text.
- **Scan history for suggestions** (Dictionary → Scan history…), as in Voxt 1.x's one-click import: Voxt sends the Transcribe results it has not read yet, 12 at a time, to the language model Transcribe uses and suggests the names and terms it finds. The dialog says how many results go to which model before anything is sent; you add each term yourself, and nothing of your dictionary is sent.
- **Learn from my corrections** (Dictionary): off by default. After Voxt inserts a Transcribe result into another app, it reads the text around that spot for up to 30 seconds; when you correct a word there, it suggests your spelling on the Dictionary page. Local rules only, nothing is sent, what it reads is not saved or logged, and password fields are never read.
- **Microphone in the menu bar**, back from Voxt 1.x: the menu bar icon's menu has a Microphone submenu that lists the connected inputs and checks the one recording uses. Choose one to move it to the top of Settings → Audio's list, so the next recording uses it. The list is read when Voxt starts, after each recording and when Settings → Audio opens.
- **End with a word** (Transcribe → Shortcuts), back from Voxt 1.x: choose over, end, 完毕, 好了 or a word of your own, say it last and pause, and the recording stops as if you pressed the shortcut. The word is left out of the text. It works with speech models that show the text as you speak (the local Qwen3-ASR models, System dictation, Doubao, Aliyun Bailian, Gemini), in Transcribe, Translate and Rewrite.
- **Doubao's flash file recognition** (Speech models → Doubao → Model: `volc.bigasr.auc_turbo`), which Voxt 1.x could run but did not list: with the same App ID and token, Doubao gets the whole recording once you stop and answers in one go, instead of streaming. Your account needs the resource opened in the Volcengine console. Recordings over 100 MB (about 52 minutes) are not sent.
- **Hide the features you do not use** (Settings → General → Features → Shown features), as in Voxt 1.x: uncheck Translate, Rewrite or App enhancement and it leaves the sidebar, ⌘K and Home's counts, and its shortcuts and scene rules stop. Its settings stay; check it again to bring it back.
- **Use the live text as the result** (Settings → Audio → Recognition → Recognise the whole recording again, on by default): off, a local speech model that shows the text as you speak (Qwen3-ASR) does not recognise the whole recording once more after you stop; it finishes its last words and the live text is the result, sooner and a little less accurate. If the live session broke off, Voxt still recognises the whole recording.
- **Live text in lines** (Settings → Overlay style → Live text layout → Lines): the overlay starts wide and wraps the words as a text box does, growing up to ten lines, then shows the newest ones, so the start of what you said stays in view. One line, which scrolls sideways, stays the default.
- **Answer card sizes** (Transcribe, Translate, Rewrite → Answer card → Card size): Small (as before), Medium or Large, each feature its own, so a long Rewrite answer can have a large card while translations keep a small one. Show the result in a card stays one switch for Transcribe and Translate.
- **Voxt 1.x's sounds in one click** (Settings → Audio → Play sounds → Use Voxt 1.x's sounds): Pop when recording starts and Tink when it stops, the pair Voxt 1.x played by default. Every other sound is still in the pickers.
- **Per-service recognition prompts** (Speech models → a remote service's dialog): optional text after the global speech context, with `{{USER_MAIN_LANGUAGE}}`, `{{USER_OTHER_LANGUAGES}}` and `{{DICTIONARY_TERMS}}` filled at recognition time. Services that take no prompt ignore it.
- **Voxt 1.x's other local speech models** (Speech models): Qwen3-ASR 1.7B (6bit), Whisper Large v3 Turbo, Whisper Large v3, Whisper Small, Parakeet TDT 0.6B v3, SenseVoice Small, Cohere Transcribe 03-2026, Nemotron 3.5 ASR 0.6B and MOSS-Transcribe-Diarize, each on its own native server. Nemotron and MOSS show the text as you speak; the others show it about once a second while you speak, and Voxt recognises the whole recording again when you stop. Only Qwen3-ASR takes your dictionary (the cards say "No dictionary"). The import brings the models Voxt 1.x downloaded without a download (a Whisper model with its tokenizer), and a feature that used one of them in Voxt 1.x uses it again; who imported with Voxt 2.0.x gets those downloads at the next start.

### Changed

- **Automatic says which model it uses**: in a feature's speech model and language model menus, and Translate's target language, the Automatic entry reads "Automatic (now: Qwen3-ASR 1.7B (8bit))", so it no longer looks like a second copy of the model listed below it.
- **The setup guide picks up where you left it**: reopened, it shows the step you left (until you finish it), and Set up a language model or Choose models… leaves a "Back to the setup guide" bar on that page.
- **Ask for the required permissions in one go**: the permissions are grouped as Required (microphone, Accessibility) and Optional (Speech Recognition), and "Ask for the required permissions" shows macOS's prompts one after another.
- **A service that needs its API key says so**: its card reads "API key needed" with Complete setup, and when no language model is ready the sidebar card names the service and opens its dialog. Services brought over from Voxt 1.x show this at once.
- **The import from Voxt 1.x reads at a glance**: its result is grouped as Imported, Left out (with why) and To do, each to-do with Complete setup; an import from outside the setup guide shows it in a dialog.
- The rules box and the Speech hints checkbox say which stage they act on: hints go to the speech model, rules shape the text after recognition.
- The setup guide says what each feature is before you try it: Transcribe writes down your words, Translate puts them into another language, Rewrite writes what you ask for. The Permissions step says why Voxt needs the microphone and Accessibility.
- **Language models sample with the service's defaults**, as in Voxt 1.x: Voxt no longer sends its own temperature (0.1 for clean-up, 0.2 for Translate, 0.5 for Rewrite), for every service. Temperature, Top P, Top K, Min P and the penalties are all in the service dialog's Advanced section, folded by default; an empty field reads "Service default" and sends nothing. Values you set before are kept and still sent. Reasoning models that accept only their default temperature no longer refuse the first request of each dictation.
- **An ⓘ opens on a click**: clicking an ⓘ shows its tip and keeps it until you click elsewhere or press Esc; resting the pointer on it shows the tip as before.
- **The rules field grows with your prompt, and ⌘F finds in it** (Transcribe, Translate, Rewrite → Your rules): the field is as high as its text, up to about 36 lines, before it scrolls. ⌘F on a feature page opens Your rules with Find in the prompt: it counts the matches, ignoring case, shows them in the accent colour, and Return goes to the next one.
- **Rewrite answers questions about the selection**: select text and say "explain this" or "what does this mean", and the built-in prompt has the model answer in the card, in the language you spoke, instead of rewriting the text. Your own full prompt (Rewrite → Your rules) is sent as you wrote it.
- **Clean-up drops rambling repeats out of the box**: the clean-up's example now also repeats a phrase and keeps it once, so the same words or point said again right away ("I need it before five, I really need it before five") come out once, with fillers and self-corrections as before.
- **The setup guide shows where the shortcut's keys are**: under the shortcut on the Transcribe, Translate and Rewrite steps, the two bottom rows of a Mac keyboard light up the keys to press, so Right ⌥ is the Option key right of the space bar.
- **Pages sit in the middle of a wide window**: in full screen or a wide window, Home, Transcribe, Translate, Rewrite, Speech models and Language models keep their column of content in the middle of the page, with the title and the page's buttons lined up with it, instead of hugging the left with the buttons at the far edge.
- **The live text stays while the language model works**: after you stop, the overlay keeps the words it showed while you spoke (Rewrite's instruction too) until the result is typed or shows in the card; it no longer drops to a bare loader. The model's draft still does not replay there.
- **Rewrite's answer writes into the card**: once the model starts answering, the overlay turns into the answer card at its full height and the answer appears in it as it is written, instead of a loader until the whole answer is in. Close stops it. With Show live text off, the card comes when the answer is done, as before.
- **Scroll back through live text in lines**: when the live text is longer than the overlay's ten lines, the overlay takes the wheel: scroll up to read what you said before, and new words leave it there; scroll back to the end and it follows the newest words again.
- **One Request parameters object in Advanced**: the language model dialog has no Expert section any more; at the end of the folded Advanced section, one JSON object takes every custom request parameter, in place of Extra Body JSON and Ollama's separate Options JSON. It is applied to the request last as a JSON merge patch (RFC 7386): its keys win, an object merges into the one Voxt sends (Ollama's model options go in `"options"`), and `null` leaves a field out. A saved or imported Options JSON moves into it as `"options"`.
- **History, Dictionary, Notes and App enhancement in a wide window**: in full screen or a wide window these pages keep their lists and panes up to 1120 points wide in the middle, with the title and buttons lined up with them; Logs still uses the whole width.

### Fixed

- **⌘W closes the window.** With "Keep running when the window closes" on (the default), Voxt stays in the menu bar and its shortcuts keep working, as the red × does; only ⌘Q quits. The Window menu had no Close Window item before.
- **Dictation types into Voxt's own window again**, as in Voxt 1.x: with the cursor in a field of Voxt (a feature's Your rules, say), the shortcut records as in any other app, with the overlay, and the text goes into that field. Before, the shortcut on Home or a feature's page started a try on the page (no overlay; the result showed in its Try it area), and on other pages the result went to the answer card.
- **Scrolling past Your rules keeps scrolling the page**: a scroll that starts on the page goes on when the rules field comes under the pointer; only a scroll that starts over the field scrolls the field, as elsewhere on macOS. A long prompt also opens at its start, not its end.
- **Rewrite takes long selections**: Rewrite reads the selection while you speak and now gives it up to 6 seconds (and a copy 1.5 seconds) instead of 1.5, so a long selection in a web editor is no longer missed, with the answer then written from your instruction alone.

### Fixed

- **Browser text boxes (GitHub and similar)** stop showing "The text did not land where you were typing" when ⌘V actually worked: Chromium-style web editors often report the same Accessibility value after a paste, so the check treats an unchanged web area or contenteditable reading as unverified instead of failed (#3644 FB36).
- **The preview leaves when the result goes in**: the live text no longer stays over the app for a moment while the result is pasted and checked; a result that cannot go in still comes back in the card.
- **The command palette in a small window** keeps its margin and rounded corners at the largest interface size, and **App enhancement in a wide window** fills the page under its header instead of stopping 760 points wide.
- **The wheel scrolls a long preview as soon as it passes ten lines**: when the measured text turned out longer than the first guess, the overlay kept letting the pointer through, and the wheel scrolled the app underneath instead.
- **Follow-ups on a long selection no longer time out**: a follow-up in the answer card sends the earlier selection and answers again, but its wait for the model's first words counted only what you just said; it now counts all of it.
- **Rewrite's card follows the answer as it is written**: a long answer used to stream on below the card's last line, out of sight; the card now shows the newest words, unless you scroll up to read.
- **Resize the answer card from the card**: a small button in the card's bottom-right corner steps it through Small, Medium and Large (and back), the same size the feature's page sets. On a short screen a Medium or Large card keeps its text to the room the screen has, instead of reaching past its edge.
- **Your rules is one long field, no scroll bar of its own**: the rules field grows with the whole prompt, so a long prompt reads down the page under one scroll bar; the page follows the cursor as you type or move it, and Find scrolls the page to the match.
- **Transcribe, Translate and Rewrite in a wide window read in two columns**: shortcuts and models on the left, the feature's own settings, answer card, rules and Try it on the right, so the page fills a wide window instead of a long column in the middle of blank space.
- **A copy pass over the English text**: about 45 strings that read as translated now read as plain English: a "test" instead of a "try" in the setup guide, "Voxt does not have microphone access" instead of "Voxt may not use the microphone", "frontmost app" instead of "the app in front", "Group prompt" instead of "Scene prompt", "Cancel recording" instead of "Cancel transcription" (it cancels Translate and Rewrite too), "Import from Voxt 1.x" instead of "the previous Voxt", "Model ID", "Thinking". The three American spellings left in a British-spelled app (Recognizing, Optimize, optimization) follow the rest. The Chinese text keeps its wording.
- **Language model debug page** (command palette, or the bottom of the Language models page), as Voxt 1.x's LLM debug window: choose a language model you set up and a request (Transcribe, Translate, Rewrite or an App enhancement group), edit its prompt there without saving, type the text, and Run shows the answer, its time and every message sent. Use this prompt saves the edit as the feature's rules or the group's prompt.

### 中文

**录音是怎么开始的。** 按下快捷键的那一刻麦克风就开始录，Voxt 不等语音模型。语音模型和人声检测在你说话的同时后台加载：Voxt 打开后的第一次听写要加载，空闲十分钟卸载后的第一次也要。麦克风开始之后你说的每一句都录下来，模型就绪前先存在内存里，就绪后一并送过去：实时文字会晚一点出现，结果一句不少，上限是一次录音的十分钟。浮窗写着“正在启动麦克风…”时，麦克风还没出声音，那时说的话录不下来；只有图标在呼吸、波形跟着你的声音动时，是模型还在加载，录音没问题。

#### 新增

- **浮窗预览只显示实时语音识别文字**：录音和识别收尾时保留流式内容；整段 Final 和语言模型整理/翻译不在胶囊里回放，结果照常插入或在答案卡片里出现。
- **可编辑完整 system prompt**（转录、翻译、转写 → 你的规则）：在一个文本框里编辑整段 system prompt（默认显示内置文字）；保存后按所写原样发送；恢复默认回到内置；内置分层模式下另有示例用户消息预览（清理与格式）。**设置搜索**：在设置里按 ⌘F 聚焦搜索框，按标题、说明和英文关键词过滤。
- **设置引导：翻译和转写试用步就地配置语言模型**（FB22）：没配语言模型时，这一步会说明缺什么，并提供「设置语言模型」按钮（引导不关闭）。语音模型下载时，模型步和试用步会显示进度；下载中也可以点继续。
- **录音时浮窗胶囊在图标旁显示当前模式**（转录、翻译或转写）。开始录音的快捷键决定模式；录音中按别的功能的快捷键只会停止，和 Voxt 1.x 的 common stop 一样（`hotkey.Route`）。
- **笔记**（新的“笔记”页），和 Voxt 1.x 一样：在任意 app 里单按右 ⌘，说完再按一次，说的内容会存成一条笔记，不会输入到光标处。笔记页按时间从新到旧列出笔记和时间，可以删除。快捷键在笔记页里改（可以关掉，也可以换成别的键）；右 ⌘ 已经被别的快捷键占用时，笔记键一开始是关的，去笔记页选一个。笔记不进历史，清空历史也不会删掉笔记；识别失败或取消的笔记会留在历史里，带着录音，可以重试。
- **快捷键预设**（设置 → 快捷键 → 预设；设置引导里也可选），和 Voxt 1.x 一样：选“右 ⌥”（默认）、“fn”（和 Auralink 相同）、“⌘ 组合”（右 ⌘ 一套）或“鼠标中键 + fn⇧”（中键单击转录、双击转写，fn⇧ 翻译），一次设好转录、翻译、转写和粘贴最近结果的快捷键。你自己改过的快捷键保持不变。选 fn 时，页面会提示 Auralink 和 macOS 的地球键也会响应 fn。
- **鼠标按钮当快捷键**，同 Voxt 1.x：在快捷键的“自定义…”里输入 Mouse3（中键）、Mouse4 或 Mouse5（多数鼠标的两个侧键），最多到 Mouse32，可以单独用，也可以加修饰键（Option+Mouse4）。单按、双击、长按和键盘快捷键一样；这个按钮按下时不再传给前台 App。和 fn、单个修饰键一样，鼠标快捷键需要辅助功能权限。
- **翻译的备用模型**（翻译 → 模型）：语言模型出错时，Voxt 用这里选的模型再翻译一次。历史记录里写的是实际给出译文的模型。
- **按键录入快捷键**，同 Voxt 1.x：转录、翻译、转写、笔记页（以及粘贴最近结果、取消转录）的“自定义…”改成直接按下组合键，比如 ⌥空格，按下就保存；录制时你的快捷键暂停。开了辅助功能时，辅助进程的 **hotkey.record** 还能录 fn、分左右的修饰键和鼠标按钮（只按修饰键也行）；没有辅助功能时应用内录制框仍可用，也可以点“改为手动输入”。快捷键里的 Esc、空格、回车、删除、Tab 写成文字，不再是 ⎋ ↩ ⌫ ⇥；取消转录的下拉菜单里有 Esc。
- **答案卡片的开关**（转录、翻译页的“答案卡片”一节）：“在卡片中显示结果”对转录和翻译生效，两页上是同一个开关。关着（默认）时直接插入，有选区就替换；转写的回答照旧一律进答案卡片，和 Voxt 1.x 一样，追问也在卡片里。之前在翻译里开着的，更新后仍是开的。以前在翻译里关掉，转写的卡片照样出现，也找不到关的地方；转录没有这个选项。
- **录音中麦克风断开时接着录**，同 Voxt 1.x：说话时拔掉正在用的麦克风，Voxt 换到 设置 → 音频 列表里第一个还连着的麦克风，再没有就用系统默认输入，接着录；浮窗显示一会儿“已切换到 X”。一个输入都没有了才结束录音。
- **隐藏语音模型**（语音模型）：模型卡片上有“隐藏”，隐藏后功能页的语音识别菜单里不再列出它；卡片移到页面底部的“已隐藏的模型”，点“显示”放回去。功能正在用的模型不能隐藏。
- **在一台全新的 Mac 上测试从 Voxt 1.x 更新**：GitHub Actions 工作流（Voxt Upgrade Test）装好 Voxt 1.x，让它从 beta 通道自己更新到新的 Voxt，再检查装好的 app（bundle ID、Developer ID 签名和公证），以及它不用你选文件夹就能导入 Voxt 1.x 的词典。
- **Apple 芯片上的 Voxt 1.x 自己更新到 Voxt 2**：每个版本还附带 app 的 `Voxt-X.Y.Z-macOS.zip`，在 Apple 芯片的 Mac 上，Voxt 1.x 的应用内更新安装它（先给打开了“Beta 更新”的用户，再给其他人），Homebrew 的 cask（`brew install --cask hehehai/tap/voxt`，只支持 Apple 芯片）也装它。Intel Mac 上的 Voxt 1.x 不会收到 Voxt 2（Voxt 2 不能在 Intel 上运行），继续用 Voxt 1.x。更新后要重新允许麦克风和辅助功能：macOS 把 Voxt 2 当成另一个 app。
- **代理的用户名和密码**（语音模型 → 网络… → 自定义），和 Voxt 1.x 一样：自定义代理（HTTP、HTTPS 或 SOCKS5）可以填用户名和密码，两个都填了才使用，模型下载和远程服务的请求都会带上。密码只保存在钥匙串里；它们只给 Voxt 用，共用模型库的其他 app 不会用到。
- **公告**，和 Voxt 1.x 一样：Voxt 启动时和每次切回 Voxt 时（最多 10 分钟一次）读取 Voxt 的公告。新公告显示为侧边栏“设置”上方的一张卡片：点“查看全文”看完整内容，关掉以后不会再出现。读取公告不会发送任何关于你的信息（没有版本号、语言或标识），设置 → 隐私里也写明了。
- **把选中的词加进词典**，和 Voxt 1.x 一样：在任意 app 里选中几个词（最多五个词，或连续最多十个中日韩字），按转录快捷键，Voxt 把它加进词典，不开始录音，浮窗显示“已加入词典”或“词典里已经有”。选中的文字更长或者没有选中文字时照常录音；词典关着时总是录音。
- **词典建议**：Voxt 建议的词显示在词典页的列表上方，写明听成了什么、来自哪里，侧栏显示待处理的条数。“添加”只把词条加入词典，不加别名；“编辑…”可以先改再加入；“忽略”后 90 天内不再出现。
- **词典的用法**（词典页）：页面上有一段可以展开的说明，讲清词条在哪里生效：语音识别提示、语言模型术语表，以及别名在识别文字里的替换。
- **扫描历史，找词条建议**（词典 → 扫描历史…），对应 Voxt 1.x 的一键导入：Voxt 把还没读过的转录结果每 12 条发给转录用的语言模型，把找到的人名和术语放进建议。开始之前对话框写明要把多少条发给哪个模型；词条要你自己点“添加”，词典内容不会发出去。
- **从我的改正中学习**（词典页）：默认关。把转录结果插入别的 App 后，Voxt 会在 30 秒内读取插入处附近的文字；你改了其中某个词，它在词典页建议你改后的写法。只用本地规则，不发送任何文字；读到的文字不保存、不写日志，也不读密码框。
- **菜单栏选麦克风**，Voxt 1.x 有过的功能：菜单栏图标的菜单里有“麦克风”子菜单，列出已连接的输入设备，录音会用的那个打勾。点一个就把它移到 设置 → 音频 列表的第一位，下一次录音用它。设备列表在 Voxt 启动时、每次录音结束后和打开 设置 → 音频 时读取。
- **说结束词停止**（转录 → 快捷键），Voxt 1.x 有过的功能：选 over、end、完毕、好了，或者自己填一个词，最后说出它并停顿一下，录音就像按了快捷键一样停止，这个词不会写进文字。边说边出文字的语音模型（本地 Qwen3-ASR、系统听写、豆包、阿里云百炼、Gemini）才能用，转录、翻译、转写都管用。
- **豆包极速版文件识别**（语音模型 → 豆包 → 模型：`volc.bigasr.auc_turbo`），Voxt 1.x 能用但菜单里没有：用同一组 App ID 和 Access Token，停止录音后把整段录音交给豆包，一次返回结果，不走流式。账号要在火山引擎控制台开通这个资源。超过 100 MB（约 52 分钟）的录音不发送。
- **隐藏不用的功能**（设置 → 通用 → 功能 →“显示的功能”），和 Voxt 1.x 一样：取消勾选翻译、转写或应用增强，它就不出现在侧边栏、⌘K 和首页的统计里，快捷键和场景规则也停用。它的设置会保留，重新勾选就回来。
- **直接用实时文字当结果**（设置 → 音频 → 识别 →“再识别一次整段录音”，默认开）：关掉后，边说边出字的本地语音模型（Qwen3-ASR）停止后不再把整段录音识别一遍，识别完最后几个字，实时文字就是结果，出得更快，准确率略低。实时会话中途断了时，Voxt 仍会识别整段录音。
- **实时文字可以多行显示**（设置 → 浮窗样式 →“实时文字的样式”→“多行”）：浮窗一开始就是宽框，像输入框一样满行换行，最多长到十行，之后显示最新的几行，开头说的话不会被滚出去。默认仍是单行、横向滚动。
- **答案卡片的大小**（转录、翻译、转写 → 答案卡片 → 卡片大小）：小（和以前一样）、中、大，每个功能各自记住，比如转写的长回答用大卡片，翻译仍用小卡片。“在卡片中显示结果”仍是转录和翻译共用的一个开关。
- **一键换回 Voxt 1.x 的提示音**（设置 → 音频 → 播放提示音 →“使用 Voxt 1.x 的提示音”）：开始录音时 Pop、停止时 Tink，是 Voxt 1.x 默认的一对。其他提示音照样在选择器里。
- **每家远程服务的识别提示词**（语音模型页 → 某远程服务的设置对话框）：可选文字，接在全局「识别上下文」后面；识别时填入 `{{USER_MAIN_LANGUAGE}}`、`{{USER_OTHER_LANGUAGES}}` 和 `{{DICTIONARY_TERMS}}`。不收提示词的语音服务会忽略它。
- **Voxt 1.x 的其他本地语音识别模型**（语音模型页）：Qwen3-ASR 1.7B (6bit)、Whisper Large v3 Turbo、Whisper Large v3、Whisper Small、Parakeet TDT 0.6B v3、SenseVoice Small、Cohere Transcribe 03-2026、Nemotron 3.5 ASR 0.6B 和 MOSS-Transcribe-Diarize，每个用自己的原生服务运行。Nemotron 和 MOSS 边说边出文字；其他模型在你说话时大约每秒更新一次文字，停止后再识别一遍整段录音。只有 Qwen3-ASR 使用你的词典（卡片上标着“不用词典”）。导入时，Voxt 1.x 下载过的模型直接带过来，不用重新下载（Whisper 连同它的分词器），在 Voxt 1.x 里用着这些模型的功能会继续用它；用 Voxt 2.0.x 导入过的，下次启动时这些下载也会带过来。

#### 变更

- **“自动”写明现在用哪一个**：功能页的语音模型、语言模型菜单和翻译的目标语言菜单里，“自动”一项写成“自动（现在用：Qwen3-ASR 1.7B (8bit)）”，不再像是下面那个模型的重复项。
- **设置引导接着上次的地方**：再打开时回到离开时的那一步（走完之前），点“设置语言模型”或“选择模型…”离开后，那一页顶部有“回到设置引导”。
- **一次请求必需的权限**：权限分成“必需”（麦克风、辅助功能）和“可选”（语音识别），“依次请求必需的权限”会一个接一个地弹出 macOS 的提示。
- **缺 API Key 的服务会说清楚**：卡片上写“需要 API Key”，带“完成设置”；没有可用的语言模型时，侧边栏的卡片写明是哪个服务，点一下打开它的设置。从 Voxt 1.x 导入的服务马上就显示这个状态。
- **从 Voxt 1.x 导入的结果一眼看清**：分成“已导入”“没有导入”（写明原因）和“还要做”，每件要做的事带“完成设置”；不是在设置引导里导入时，结果显示在一个对话框里。
- “你的规则”和“语音识别提示”各自写明作用阶段：提示给语音模型，规则在识别之后整理文字。
- 设置引导每一步先说明这个功能是什么再让你试：转录照录你说的话，翻译把它译成另一种语言，转写按你的要求写出内容。权限一步写明为什么需要麦克风和辅助功能。
- **语言模型的采样用服务自己的默认值**，和 Voxt 1.x 一样：所有服务上，Voxt 都不再自己发温度（之前清理 0.1、翻译 0.2、转写 0.5）。温度、Top P、Top K、Min P 和几种惩罚项都放进服务对话框里默认折叠的“高级”；留空的输入框写着“服务默认”，什么也不发。之前填过的值保留，照常发送。只接受默认温度的推理模型，不会再在每次听写的第一个请求就拒绝。
- **点一下 ⓘ 就能看说明**：点 ⓘ 显示它的提示，直到点别处或按 Esc 才收起；鼠标停在上面照旧显示。
- **规则框跟着提示词变高，⌘F 可以在里面查找**（转录、翻译、转写 → 你的规则）：输入框随内容增高，大约 36 行以后才在框内滚动。在功能页按 ⌘F 会展开“你的规则”并聚焦“在提示词里查找”：显示有几处（不分大小写），匹配处用强调色标出，按回车跳到下一处。
- **转写能回答关于选中文字的问题**：选中文字后说“解释一下这段”“这是什么意思”，内置提示词会让模型用你说话的语言在卡片里回答，而不是改写原文。你自己写的完整提示词（转写 → 你的规则）照原样发送。
- **清理默认去掉车轱辘话**：清理的示例里加了一句重复的话、只保留一次，所以紧接着又说一遍的同样词句或同一个意思（“今天下班前一定要发，就是说今天下班前一定要发出去”）只留一次；语气词和改口照旧去掉。
- **设置引导画出快捷键在键盘上的位置**：转录、翻译、转写这几步的快捷键下面，画着 Mac 键盘最下面两排，要按的键亮起，比如“右 ⌥”就是空格键右边的 Option 键。
- **宽窗口里页面居中**：全屏或窗口很宽时，首页、转录、翻译、转写、语音模型和语言模型的内容栏放在页面中间，标题和页面按钮跟内容对齐，不再贴着左边、按钮却在最右边。
- **语言模型处理时实时文字留在浮窗里**：停止录音后，浮窗保留说话时显示的文字（转写的指令也是），直到结果输入进去或出现在卡片里，不再只剩一个加载动画。模型写的草稿仍然不在浮窗里回放。
- **转写的回答直接写进卡片**：模型一开始回答，浮窗就变成满高的答案卡片，回答边写边出现在里面，不再一直转圈等整段写完。点关闭会停止。关掉“显示实时文字”时，和以前一样写完才出卡片。
- **多行实时文字可以往回翻**：实时文字超过浮窗的十行时，浮窗接收滚轮，可以往上翻看前面说的话，新的字不会把它拉回去；滚回底部就重新跟着最新的字。
- **“高级”里只有一个“请求参数”对象**：语言模型对话框不再有“专家”一节；默认折叠的“高级”最后是一个 JSON 对象，所有自定义请求参数都写在这里，取代原来的额外请求体 JSON 和 Ollama 单独的 Options JSON。它按 JSON merge patch（RFC 7386）最后合并进请求：同名的键以这里为准，对象会并进 Voxt 发的同名对象（Ollama 的模型选项写在 `"options"` 里），写 `null` 就不发这个字段。已保存或从 1.x 导入的 Options JSON 会移进来，成为其中的 `"options"`。
- **宽窗口里历史、词典、笔记和应用增强也居中**：全屏或窗口很宽时，这几页的列表和面板最宽 1120 点、放在中间，标题和按钮跟着对齐；日志页仍然占满宽度。

#### 修复

- **⌘W 关闭窗口。** 开着“关闭窗口后继续运行”（默认开）时，Voxt 留在菜单栏里，快捷键照常可用，和红色 × 一样；只有 ⌘Q 退出。之前窗口菜单里没有“关闭窗口”。
- **在 Voxt 自己的窗口里听写，文字又能输进去了**，和 Voxt 1.x 一样：光标在 Voxt 的输入框里（比如功能页的“你的规则”）时，按快捷键和在别的 app 里一样录音、显示浮窗，文字输入到这个框里。以前在首页和功能页上按快捷键会在页面里开始一次试一试（没有浮窗，结果显示在“试一试”区域），在其他页面上结果进了答案卡片。
- **滚过“你的规则”时页面接着滚**：在页面上开始的滚动，规则框移到指针下面时页面照样往下滚；只有在规则框里开始的滚动才滚规则框，和 macOS 别处一样。长提示词展开时也从开头显示，不再停在结尾。
- **转写能处理长选段**：转写边录音边读选中的文字，读取时限从 1.5 秒放宽到 6 秒（复制兜底 1.5 秒），网页编辑器里的长选段不会再读不到、只按指令写。

#### 修复

- **浏览器输入框（GitHub 等）**：⌘V 已经粘进输入框时，不再误报“文字没有出现在输入位置”。Chromium 类网页编辑器常在粘贴后仍让无障碍读到相同长度；网页区域或 contenteditable 里若对比结果不变，按“无法校验”处理，仍算插入（#3644 FB36）。
- **插入时预览立刻收起**：结果开始往 App 里粘贴时浮窗就淡出，不再在刚插入的文字上停一会儿；插不进去的照样回到卡片里。
- **小窗口里的命令面板**在界面放到最大时也留着边距和圆角；**宽窗口里的应用增强**铺满表头下的整页，不再停在 760 点宽。
- **预览一超过十行就能用滚轮翻**：实际量出的文字比最初估计的长时，浮窗还让指针穿过去，滚轮滚的是下面的 App。
- **长选区的追问不再超时**：在答案卡片里追问时，之前的选区和回答会一起再发给模型，但等第一个字的时限只按你刚说的话算；现在全部算进去。
- **转写的卡片跟着回答往下走**：长回答以前会一直写到卡片最后一行下面，看不到；现在卡片显示最新写出的字，你往上滚着读时不打断。
- **在卡片上直接调大小**：卡片右下角的小按钮依次切换小、中、大（再回到小），和功能页上设的是同一个大小。屏幕较矮时，中、大卡片的文字高度收到屏幕放得下为止，卡片不再伸出屏幕。
- **“你的规则”不再有自己的滚动条**：规则框随整段提示词长高，长提示词顺着页面往下读，只有页面一个滚动条；打字或移动光标时页面跟着光标走，查找时页面滚到匹配处。
- **宽窗口里转录、翻译、转写页分两栏**：左边是快捷键和模型，右边是功能自己的设置、答案卡片、规则和试一试，页面铺满宽窗口，不再是空白中间一条长长的单栏。
- **英文文案过了一遍**：约 45 条读起来像翻译腔的英文改成了地道的说法，例如设置引导里的 “try” 改成 “test”，“Voxt may not use the microphone” 改成 “Voxt does not have microphone access”，“the app in front” 改成 “frontmost app”，“Scene prompt” 改成 “Group prompt”，“Cancel transcription” 改成 “Cancel recording”（它也取消翻译和转写），“the previous Voxt” 改成 “Voxt 1.x”。英文整体用英式拼写，剩下的三处美式拼写也改了过来。中文文案只把“场景提示词”改成“分组提示词”，和英文一致。
- **语言模型调试页**（命令面板，或语言模型页底部），对应 1.x 的语言模型调试窗口：选一个已设置的语言模型和一个请求（转录、翻译、转写或某个应用增强分组），在页内改提示词、不保存，输入文字后点“运行”，显示回答、用时和发出的全部消息。“使用这个提示词”把改动存成该功能的规则或分组提示词。

## [2.0.1] - 2026-11-07

### Added

- **Remote language models** (the new Language models page): Transcribe's clean-up, Translate and Rewrite use a language model of a service you set up, over its OpenAI-compatible API: OpenAI, Anthropic, Google Gemini, DeepSeek, Qwen (Alibaba Cloud), Kimi, GLM, Doubao (Volcengine), MiniMax, StepFun, Xiaomi MiMo, xAI, OpenRouter, any other compatible server (Custom), or one on this Mac (Ollama, LM Studio, oMLX), so that the text stays here. Each service takes an API key, an address and a model, with Test connection; the key is kept in the macOS Keychain only, never in Voxt's settings or logs. Every feature's language model menu lists the services set up; Automatic uses the first one, and each card says which features use it. The page groups the servers that run on this Mac (free) apart from the online services. Only the recognised or selected text is sent, never audio.
- **Import from the Swift Voxt 1.x.** When Voxt 1.x's data is on the Mac, the setup guide starts with a card that brings over its dictionary (replacement terms become "heard as" aliases, disabled terms stay disabled), its dictation, translation and rewrite history, and the speech models it downloaded that Voxt uses, so they are not downloaded again (its language models are not: Voxt no longer runs language models on this Mac). Voxt 1.x's data is only read. If you set up 2.0.0, which had no import, a card in the sidebar offers it once; it is also in the dictionary's ••• menu ([docs/voxt-migration.md](docs/voxt-migration.md)).
- Import terms takes the dictionary file Voxt 1.x exports (Dictionary → More → Export): its terms, with their replacement terms as aliases.
- **System dictation** is back as a speech model (Speech models → On this Mac): macOS's own speech recognition, on this Mac, with nothing to download, and the text shown as you speak. It hears your main language (Settings → General; without one, the Mac's dictation language), and your dictionary's terms help it. Choose it on a feature's page; Automatic still uses a speech model you downloaded. The first time, macOS asks for Speech Recognition (Settings → Permissions has it too); a language this Mac cannot recognise on device fails with a notice that says so.
- **Remote speech recognition** (the Speech models page's Remote services): OpenAI Transcribe, Doubao (Volcengine), Zhipu GLM-ASR, Aliyun Bailian ASR, StepFun, Xiaomi MiMo and Google Gemini, as in Voxt 1.x. Doubao, Aliyun Bailian and Gemini show the text while you speak (tagged Realtime); the others get the recording once you stop. Set one up with its key (Doubao also takes its App ID), address and model, and test the connection before saving; keys stay in the Mac's Keychain, never in Voxt's settings or logs. OpenAI Transcribe, StepFun and Zhipu pointed at a server on this Mac (a local Whisper server) need no key. A service that is set up is offered after the speech models of this Mac in every feature's speech model menu, and used when you choose it there, or when this Mac has no speech model (system dictation counts as one). Doubao, Gemini and Aliyun Bailian's Fun-ASR and Qwen-Audio models get your dictionary's terms as hot words; OpenAI Transcribe, StepFun and Zhipu get them in their prompt; Xiaomi MiMo takes none.
- **Choose the model folder** (Settings → Storage → Model folder → Change…): keep the models in another folder or on another disk. Voxt brings the models there first (hard links on the same disk, so no space is used; copies on another disk), including the other ActNow apps' models, wherever each app kept them, so that none is downloaded again; then those apps use the new folder too, and Voxt restarts. The old folders stay as they were: delete them in Finder once no app uses them.
- **Generation settings for remote language models**, field for field as Voxt 1.x had them for each service: Think (default, off, on, effort, budget), max output tokens, temperature, top P, the service's own options (OpenAI's verbosity, Ollama's keep alive, oMLX's usage in stream events), and the folding Advanced (response format, top K, min P, seed, penalties, logprobs, stop sequences) and Expert (Extra Body JSON) sections. A service shows only the fields it takes.
- **Show in the Dock** (Settings → General): off, Voxt is in the menu bar only, and its menu bar icon stays on.
- **Also copy the result to the clipboard** (Settings → General): each result stays on the clipboard, also after it was pasted, and a result the answer card shows is copied too.
- **Show live text** (Settings → Overlay style): off, the overlay shows no words while you speak or while the model writes.

### Changed

- In Activity Monitor the voice detector (Silero VAD) runs as `silero_vad_server`, apart from the speech model's `qwen3_asr_server`, so that a recording no longer looks like two copies of the speech model.
- **Models are named by what they are**: the Speech models page, the feature pages' model menus, the setup guide and History name a model by its family, size and quantisation (Qwen3-ASR 1.7B (8bit), Qwen3-ASR 0.6B (4bit)), as Voxt 1.x did, instead of "Accurate speech recognition"; its card has short labels (Local, Accurate or Fast, Realtime, whether it knows your main language), says which features use it, and has no rating.
- **Language models are remote only.** Voxt no longer downloads or runs language models on this Mac: set one up on the Language models page (Ollama runs one here, for free). Those Voxt 2.0.0 downloaded are listed on the Speech models page under "Language models no longer used", to delete. Settings → Privacy, the setup guide and Home say where the text goes.
- **The Local models page is now Speech models**: the speech models of this Mac, then the remote speech services, each tagged Local or Remote; language models are on the Language models page.
- **Models are kept in ~/Library/Application Support/SGLang Omni/models**, named after SGLang Omni, whose runtime serves them, instead of the ActNow model library (~/Library/Application Support/ActNow/models). The models Voxt 2.0.0 downloaded are brought over at the first start without downloading them again (hard links: no space is used), and so are the other ActNow apps' (AudioSmith, Dicta): the family's model folder becomes SGLang Omni/models, so every app keeps its models there. The old folder stays as it was. A folder chosen in Settings → Storage applies to every ActNow app, and their models are brought there.

### Fixed

- Import terms no longer adds each line of a JSON file as a term: Voxt 1.x's export is read as a dictionary, and any other JSON file is refused.
- Clicking beside the dialog of a remote service, a dictionary term or the network settings, or pressing Escape, no longer closes it while it holds what you typed: a key you had just pasted was lost when the click that brought Voxt's window back landed beside the dialog. Cancel and × still close it.
- VoiceOver now reaches the Retry button of a history entry and the Test button of a remote service's card, and reads the name of Storage's Time Machine switch.

### 中文

#### 新增

- **远程语言模型**（新的“语言模型”页）：转录的整理、翻译和转写使用你设置的服务的语言模型，通过它的 OpenAI 兼容 API 调用：OpenAI、Anthropic、Google Gemini、DeepSeek、通义千问（阿里云百炼）、Kimi、智谱 GLM、豆包（火山方舟）、MiniMax、阶跃星辰、小米 MiMo、xAI、OpenRouter、其他兼容的服务（自定义），或者本机的服务（Ollama、LM Studio、oMLX），让文字留在这台 Mac 上。每个服务填 API Key、地址和模型，可以测试连接；Key 只保存在 macOS 钥匙串里，不会写进 Voxt 的设置或日志。各功能的语言模型菜单列出设置好的服务，“自动”用第一个，卡片上写明使用位置。页面把本机运行的服务（免费）和在线服务分开列出。只发送识别出的文字或选中的文字，不发送音频。
- **从 Swift 版 Voxt 1.x 导入。** 这台 Mac 上有 Voxt 1.x 的数据时，设置引导第一页有一张卡片，把它的词典（替换词变成"可能被听成"，停用的词还是停用）、听写、翻译和改写的历史，以及它下载过、Voxt 也在用的语音模型带过来，不用再下载（它的语言模型不带：Voxt 不再在本机运行语言模型）。Voxt 1.x 的数据只读不改。装过 2.0.0（那一版没有导入）的，侧栏会出一次卡片；词典的 ••• 菜单里也有（[docs/voxt-migration.md](docs/voxt-migration.md)）。
- "导入词条"认 Voxt 1.x 导出的词典文件（词典 → 更多 → 导出）：词条，以及作为别名的替换词。
- **系统听写**回来了，作为语音识别模型（语音模型 → 本机）：macOS 自带的语音识别，在这台 Mac 上运行，无需下载，说话时实时显示文字。它识别你的主要语言（设置 → 通用；没有设置时用这台 Mac 的听写语言），词典里的词条也会帮它识别。在功能页里选择它；“自动”仍然优先用你下载的语音识别模型。第一次使用时 macOS 会请求“语音识别”权限（设置 → 权限里也有）；这台 Mac 不能在本机识别的语言会提示失败原因。
- **远程语音识别**（语音模型页的“远程服务”）：OpenAI Transcribe、豆包（火山引擎）、智谱 GLM-ASR、阿里云百炼 ASR、阶跃星辰、小米 MiMo 和 Google Gemini，与 Voxt 1.x 相同。豆包、阿里云百炼和 Gemini 边说边出文字（标为“实时”），其他服务在你说完后识别整段录音。填写密钥（豆包还可以填 App ID）、地址和模型即可设置，保存前可以测试连接；密钥只保存在 Mac 的钥匙串里，不会写进 Voxt 的设置或日志。OpenAI Transcribe、阶跃星辰和智谱指向本机的服务（本地 Whisper 服务）时不需要密钥。设置好的服务排在这台 Mac 上的语音模型之后，出现在每个功能的语音识别模型菜单里；在那里选中它，或者这台 Mac 上没有语音模型（系统听写也算一个）时使用。豆包、Gemini 以及阿里云百炼的 Fun-ASR 和 Qwen-Audio 模型把词典里的词当作热词，OpenAI Transcribe、阶跃星辰和智谱把它们放进提示词，小米 MiMo 不使用词典。
- **自己选模型文件夹**（设置 → 存储 → 模型文件夹 → 更改…）：把模型放到别的文件夹或别的磁盘。Voxt 先把模型放过去（同一磁盘上建硬链接，不占新空间；其他磁盘上复制），也包括其他 ActNow app 的模型，不管它们原来放在哪里，谁都不用重新下载；之后这些 app 也改用新文件夹，Voxt 随即重启。原来的文件夹保持不变：没有 app 再用它们之后，可以在访达里删掉。
- **远程语言模型的生成设置**，和 1.x 一样按服务提供：思考（默认、关闭、开启、推理强度、预算）、最大输出 Token 数、温度、Top P、服务专属的选项（OpenAI 的详略程度、Ollama 的模型保留时长、oMLX 的流式用量），以及可以展开的“高级”（返回格式、Top K、Min P、随机种子、惩罚项、Logprobs、停止序列）和“专家”（额外请求体 JSON）。每个服务只显示它支持的字段。
- **在程序坞中显示**（设置 → 通用）。关掉后 Voxt 只在菜单栏里，菜单栏图标会一直开着。
- **同时把结果复制到剪贴板**（设置 → 通用）。每次的结果都留在剪贴板里，粘贴过的也一样；只出现在答案卡片里的结果也会复制。
- **显示实时文字**（设置 → 浮窗样式）。关掉后，你说话时和模型写回答时，浮窗里都不显示文字。

#### 变更

- 活动监视器里，声音检测（Silero VAD）的进程叫 `silero_vad_server`，和语音识别模型的 `qwen3_asr_server` 分开，录音时不再像是开了两个语音识别进程。
- **模型写清楚是哪一个**：语音模型页、各功能页的模型菜单、设置引导和历史都按系列、尺寸和量化称呼模型（Qwen3-ASR 1.7B (8bit)、Qwen3-ASR 0.6B (4bit)），和 Voxt 1.x 一样，不再写“精准语音识别”；卡片上有简短的标签（本地、准确或快速、实时、是否支持主语言）和使用位置，没有评分。
- **语言模型只用远程服务。** Voxt 不再在这台 Mac 上下载和运行语言模型：请在“语言模型”页设置一个（用 Ollama 可以在本机免费运行）。Voxt 2.0.0 下载过的语言模型列在语音模型页的“不再使用的语言模型”里，可以删除。设置 → 隐私、设置引导和首页都写明了文字会发到哪里。
- **本地模型页改名为语音模型**：先列这台 Mac 上的语音模型，再列远程语音识别服务，各自标为“本地”或“远程”；语言模型在语言模型页。
- **模型放在 ~/Library/Application Support/SGLang Omni/models**，以运行这些模型的 SGLang Omni 命名，不再放在 ActNow 模型库（~/Library/Application Support/ActNow/models）。Voxt 2.0.0 下载的模型在第一次启动时搬过来，不用重新下载（硬链接，不占新空间）；其他 ActNow app（AudioSmith、Dicta）的模型也一起搬过来：家族的模型文件夹改成 SGLang Omni/models，所有 app 都把模型放在这里。原来的文件夹保持不变。在“设置 → 存储”里选的文件夹对所有 ActNow app 生效，它们的模型也会一起搬过去。

#### 修复

- "导入词条"不再把 JSON 文件的每一行当成一个词：Voxt 1.x 的导出文件按词典读，别的 JSON 文件会被拒绝。
- 远程服务、词条和网络设置的对话框里填了内容时，点对话框旁边或按 Esc 不会再把它关掉。以前刚粘贴进去的 key，可能因为点回 Voxt 窗口时点在了对话框旁边而丢失。“取消”和 × 照常关闭。
- 读屏（VoiceOver）现在能点到历史记录里的“重试”和远程服务卡片上的“测试”，也能读出 设置 → 存储 里 Time Machine 开关的名字。

## [2.0.0] - 2026-11-06

### Added

- Voxt 2.0.0, rebuilt from scratch: voice input for macOS on local models only. Press a shortcut in any app, speak, and the text appears where your cursor is. Audio and text never leave your Mac.
- Local speech recognition: two models, accurate (recommended) and fast, with the text shown live as you speak, and a voice detector that sends only speech to them.
- Local language models: a balanced one (recommended), a fast one and a translation model, for clean-up, translation and rewriting.
- Transcribe, with an optional clean-up of fillers, false starts and punctuation; Translate, the selected text or what you say; Rewrite, the selected text or new text from a spoken request, with follow-ups in the answer card.
- App enhancement: groups of apps and websites with their own scene prompt and a key pressed after inserting.
- Dictionary: its terms help the speech model hear your names and jargon, and the language model write them as you spell them.
- The Local models page: download, pause and resume, every file verified, delete; the memory a loaded model uses. Downloads use the system's proxy, or a mirror when Hugging Face cannot be reached.
- A setup guide that installs the two recommended models and tries each feature; history, the overlay, the menu bar, shortcuts and settings.
- Shortcuts: tap the right Option key to transcribe, right Option with Shift to translate, Control with right Option to rewrite; Control-Option-V pastes the last result. They leave fn to Auralink, so the two apps run side by side.
- The app is signed and notarised. This version does not update itself: Settings → About says so, and a new version is downloaded by hand.
- History entries have a context menu: right-click (or Control-click) one to copy its result or its recognised text, show its details, show its recording in Finder, or delete it. Home's recent results have the same menu. In History a selected entry also takes ⌘C (copy) and Delete.
- **Keep models loaded** (Local models → Advanced): how long a model stays in memory unused. Automatic, the default, is as before (speech 10 minutes, language models 5); you can choose 5, 15 or 30 minutes, 1 hour, or until Voxt quits, so that every dictation starts at once. Other apps that use this model library can still ask Voxt to free the memory of models it is not using. A change applies at once.
- **Sounds to choose** (Settings → Audio → Play sounds): the start and the stop sound are each chosen from a sound library (from React Bits Sound Lab, in the official builds; Tally and Wrap by default), or macOS's system sounds, or none. A compact picker searches them (names and tags), filters by type, plays each with ▶ (again to stop) and sets their volume in steps; a Try button plays the choice. A Voxt pair chosen in an earlier build is kept as those system sounds. A build from source without the library plays Pop and Tink.
- **Mute system audio while recording** (Settings → Audio, on by default): the current output device is muted from just after the start sound until the recording stops, so that videos, music or calls are not recorded; a device you muted stays muted, one you unmute during the recording stays on, and a device that cannot be muted leaves the recording as it is (the overlay says so). No permission is needed. If Voxt or its helper stops while muting, the sound comes back at once, or at the next start.
- **A notice when your Mac is short of memory**: when the language model was slow, or did not answer in time, while the Mac's swap was over 90% full, the overlay says "Low memory slows the model: quit unused apps." after the text went in. It shows at most once each time Voxt runs.
- **History settings** (the button beside History's search field): keep the recordings (the same switch as Settings → Privacy), choose the folder they are kept in (the recordings already kept move there, each one copied and checked before the original is deleted; if the folder is on a drive that is not connected, new recordings go to the default folder until it is back), see how much room they take, export them (all, or those of the tab shown) into a folder you choose, named by their date and first words, or delete them all and keep the texts. **Automatic clean-up**, on as in Voxt: entries older than 3 months are deleted with their recordings; you can choose 1 week to 1 year, keep at most 50 to 1000 entries of each feature, or turn it off. A new setting that would delete entries now asks first, with their number.
- **Retry** a history entry from its recording: a failed entry (or one with no text) shows Retry on its row instead of Failed, in its details and in its context menu; other entries with a recording have Transcribe again. The entry is updated in place (text, status, time); nothing is pasted. The overlay's notice of a failed recognition opens the entry and retries it.
- **Long dictations are recognised while you speak**: after a minute, finished stretches are recognised in the background, cut at pauses, so Stop leaves only the last half-minute or so (on a busy Mac, 104 s of speech: 26 s → 2.4 s after Stop).
- **The overlay says what a recording is waiting for.** When the microphone is slow to start (the first dictation after Voxt starts on a Mac short of memory), the microphone icon pulses inside a ring and the overlay says "Starting the microphone…": what you say before the bars light up is not recorded. While the speech model is still loading (the first dictation, or after it was unloaded), only the icon pulses and the bars keep showing your voice: the recording works, and the live text comes once the model is ready.
- **A customizable cancel shortcut**: Settings → Shortcuts lists Cancel transcription; change its key or turn it off on the Transcribe page (Escape by default). The key is taken only while a recording runs or the answer card shows, and a cancelled recording now stays in History, ready to Retry.

### Changed

- **Voxt 2.0.0 replaces the Swift Voxt 1.x.** Installing it replaces Voxt.app in Applications. The settings, dictionary and history of Voxt 1.x stay in its own folder and are not brought over, and Voxt asks for its permissions again (the setup guide walks you through them).
- **Models are kept in ~/Library/Application Support/ActNow/models**, shared with other apps that use this model library, instead of the Hugging Face cache (~/.cache/huggingface), which cleaning tools treat as a cache. Models already there are brought over at the first start without downloading them again (hard links: the cache stays as it is). Time Machine leaves models out; Settings → Storage → "Back up models with Time Machine" turns it on for all of them.
- The note of which output device Voxt muted is kept in Application Support instead of ~/Library/Caches, so that a cache cleaner run after a crash cannot leave your speakers muted.
- With a dictionary of more than 30 terms, clean-up, translation and rewriting answer sooner: the first words come 1.4–1.9 s sooner on the balanced language model and 0.7 s on the fast one. The terms that a dictation mentions now go with the dictation instead of changing the model's instructions each time.
- The fast speech model gets the 30 most recently changed terms of your dictionary. A longer list made it slower and less accurate; the accurate model still gets up to 500.
- **The result goes where you are typing when it is ready.** If you switch to another app during or after the recording and click into a text field there, the text is inserted there; the app you started in is not brought back. Nothing is typed into a password field or into Voxt's own windows, and when no text field is focused (Finder's desktop, a web page you are reading), the result waits in the answer card with Insert and Copy, which now also inserts into the app in front. A selection is still replaced only in the app it was selected in, and app enhancement's auto key (Return…) is pressed only there. The history shows the app the text went into.

### Fixed

- A result that was pasted but did not show up (in a web page in Arc) is no longer lost without a word. Voxt now checks that the focused text field changed after the paste. When it did not, the result waits in the answer card and stays on the clipboard (press ⌘V), the auto key is not pressed, and the card says so; when another app had secure input on (a password field, a terminal's Secure Keyboard Entry), the card says that. Where the field cannot tell (some web editors), the text is pasted as before. Your clipboard now comes back as soon as the text shows, instead of half a second later.
- A long dictation no longer fails after 30 seconds of recognition and loses everything (a 4m51s dictation on a Mac short of memory). Recognition now gets 60 s plus 1 s per second of recording; a model that stops answering is still caught after 15 s. If full recognition fails anyway, the live text is used (cleaned up and inserted as usual) and the overlay says so.
- The speech model uses much less memory on long recordings: under 4 GB for a 5-minute dictation (Accurate model) instead of up to 10 GB, which pushed your other apps into swap and slowed the Mac. Recognising a recording again is faster, too.
- When recognition fails or finds no text, the entry keeps the recording for 7 days even when Keep the recordings is off, so that you can retry it; a successful retry deletes it.
- Dictionary terms no longer appear at the end of a dictation where you did not say them ("… 嗯。Voxt, Omni, SGLang."). The speech model wrote out the dictionary it was given when part of a recording had no speech, such as the pause before Stop at the end of a long dictation. Such an ending is now checked against the same audio recognised without the dictionary, and terms you did not say are removed; terms you did say stay, also at the end of a sentence.
- Voxt no longer crashes now and then while it checks its Microphone and Accessibility permissions (when the helper starts or restarts). The check could finish on a different system thread from the one it started on.
- The global shortcut tap is no longer reported ready when macOS created it but stripped its keyboard events (Accessibility granted without Input Monitoring): the granted events are read back and verified, a partial tap is refused and retried every 2 s, plain shortcuts keep working through the fallback, and the Shortcuts settings show the permission notice.
- Chinese dictation is no longer turned into English on a Mac whose first language is English. Without a main language set in Settings → General, the speech model was told to expect the interface's language (English there) and the Fast model translated most Chinese recordings; now it detects the language itself unless you set a main language or use Voxt in Chinese.
- Translating what you say no longer removes 额 and 唔 from inside words first: 金额 (amount) became 金, and Cantonese 唔 (not) was dropped, which reversed the meaning. They are now removed only as a hesitation that opens a clause, before a pause.
- A translation or rewrite that stopped at the language model's length limit is no longer inserted, nor put over your selection: it shows in the answer card, which says it may be incomplete.
- **Clear the history** (Settings → Privacy) now also removes the cleared texts from the database file. Before, they stayed in its free space until it was reused, readable by anyone who opened the file.
- A long recording recognised in parts (Transcribe again and Retry in History) no longer loses the space where two parts meet ("the Republic.Socrates begins"); Chinese still gets none.
- A dictation of over a minute no longer ends with a word you did not say ("Okay.") when you pause before you stop: the pause at the end is no longer cut off and recognised on its own.

### 中文

#### 新增

- Voxt 2.0.0，从头重写：只用本地模型的 macOS 语音输入。在任意 App 里按下快捷键说话，文字出现在光标处。音频和文字不会离开你的 Mac。
- 本地语音识别：精准（推荐）和快速两个模型，说话时实时显示文字；声音检测只把说话的部分交给它们。
- 本地语言模型：均衡（推荐）、快速和翻译专用三个模型，用于整理、翻译和改写。
- 转录，可以整理掉口头语、说了一半的话并补上标点；翻译，选中的文字或说的话；转写，改写选中的文字或按语音要求写新内容，并可在答案卡片里继续追问。
- 应用增强：把 App 和网站分组，每组有自己的场景提示词和插入后的按键。
- 词典：词条帮助语音识别模型听准你的人名和术语，并让语言模型照你的写法输出。
- 本地模型页：下载、暂停和继续下载，每个文件都经过校验，可删除；显示已加载模型占用的内存。下载走系统代理，连不上 Hugging Face 时改用镜像。
- 设置引导：安装推荐的两个模型，并把每个功能试一遍；还有历史、浮窗、菜单栏、快捷键和设置。
- 快捷键：单按右 Option 转录，右 Option 加 Shift 翻译，Control 加右 Option 转写；Control-Option-V 粘贴上一条结果。fn 留给 Auralink，两个 app 可以同时使用。
- app 已签名并经过公证。这个版本不会自动更新：设置 → 诊断与关于里有说明，新版本需要自己下载。
- 历史记录有了右键菜单：右键（或按住 Control 点）一条记录，可以复制结果或识别原文、查看详情、在访达中显示录音，或者删除它。首页的最近结果也有同样的菜单。在历史页选中一条记录后，也可以按 ⌘C 复制、按 Delete 删除。
- **保持模型载入**（本地模型 → 高级）：模型闲置多久后从内存卸载。默认“自动”与以前相同（语音识别 10 分钟、语言模型 5 分钟）；也可以选 5、15、30 分钟、1 小时或“直到退出 Voxt”，让每次口述都立即开始。使用这个模型库的其他 app 仍然可以请 Voxt 释放闲置模型占用的内存。修改立即生效。
- **可选的提示音**（设置 → 音频 → 播放提示音）：开始和结束提示音分别从提示音库里选（来自 React Bits Sound Lab，只在官方构建里；默认“开录”和“收工”），也可以选 macOS 系统声音或无声。选择器很紧凑：可以搜索（名字和标签）、按类型筛选、用 ▶ 逐个试听（再点一下停止）、按档位调音量；试听按钮播放当前选择。以前版本选的 Voxt 预设保留为对应的系统声音。没有提示音库的源码构建播放 Pop 和 Tink。
- **录音时静音系统声音**（设置 → 音频，默认开启）：从开始提示音之后到停止录音为止，当前输出设备静音，免得视频、音乐、通话的声音被录进去；你自己静音的设备保持静音，录音中你取消静音后不会再被静音，不支持软件静音的设备照常录音（浮窗会提示）。不需要任何权限。Voxt 或辅助进程在静音期间意外退出时，声音会立即恢复，最迟在下次启动时恢复。
- **内存不足时的提示**：Mac 的 swap 用了九成以上、语言模型因此变慢或没有及时响应时，文字插入之后浮窗提示“内存不足，模型变慢：可退出不用的应用。”。每次运行 Voxt 最多提示一次。
- **历史设置**（历史页搜索框右边的按钮）：保存录音（和 设置 → 隐私 是同一个开关）；选择录音存放的文件夹（已有录音会一起移过去，每段先复制并校验再删原文件；文件夹在没接上的外接盘上时，新录音先存到默认文件夹，等它回来）；查看录音占用的空间；把录音（全部，或当前 tab 的）导出到你选的文件夹，以日期和开头几个词命名；或者全部删除、保留文字。**自动清理**，和 Voxt 一样默认开启：超过 3 个月的记录连同录音一起删除；可以选 1 周到 1 年，或每种功能最多保留 50 到 1000 条，也可以关掉。新设置会立即删掉已有记录时，先告诉你会删几条并请你确认。
- **重试**：用历史记录保存的录音再识别一次。失败或没有文字的记录，行上的“失败”换成“重试”，详情和右键菜单里也有；其他有录音的记录是“重新识别”。结果原地更新这条记录（文字、状态、用时），不会粘贴。浮窗里识别失败的提示，点一下打开这条记录并重试。
- **长口述边说边识别**：说到 1 分钟以后，已经说完的部分在停顿处切开、在后台识别，停止后只剩最后半分钟左右（Mac 内存紧张时，104 秒口述停止后的等待 26 秒 → 2.4 秒）。
- **浮窗说明录音在等什么。** 麦克风启动慢时（Mac 内存紧张时 Voxt 启动后的第一次口述），麦克风图标在一个圆环里闪动，浮窗显示“正在启动麦克风…”：波形亮起之前说的话不会被录下。语音识别模型还在加载时（第一次口述，或者模型被卸载之后），只有图标在动，波形照常跟着你的声音：录音不受影响，模型就绪后实时文字随即出现。
- **取消转录的按键可自定义**：设置→快捷键里新增“取消转录”一行，在转录页可以换成别的键或关掉（默认 Esc）。只在录音或答案卡片显示时拦截这个键；取消的录音现在会留在历史里，可以用“重试”找回。

#### 变更

- **Voxt 2.0.0 取代 Swift 版 Voxt 1.x。** 装上它会替换“应用程序”里的 Voxt.app。Voxt 1.x 的设置、词典和历史留在它自己的文件夹里，不会带过来；Voxt 会重新请求权限（设置引导会带你完成）。
- **模型放在 ~/Library/Application Support/ActNow/models**，与使用这个模型库的其他 app 共用，不再放在会被清理工具当成缓存的 Hugging Face 缓存（~/.cache/huggingface）里。那里已有的模型在首次启动时直接拿过来（硬链接，原缓存不动），不用重新下载。Time Machine 默认不备份模型，可在 设置 → 存储 →“用 Time Machine 备份模型”中为所有这些 app 打开。
- Voxt 静音了哪个输出设备的记录改放在 Application Support，不再放在 ~/Library/Caches，崩溃后即使清理了缓存，扬声器也不会一直静音。
- 词典超过 30 条时，整理、翻译和转写更快开始输出：均衡语言模型快 1.4–1.9 秒，快速模型约快 0.7 秒。口述里提到的词条现在随口述一起发给模型，不再每次改动模型的指令。
- 快速语音识别模型只使用词典里最近修改的 30 条。更长的列表会让它更慢、更不准；精准模型仍可使用最多 500 条。
- **结果插入到它出来时你正在输入的地方。** 录音中或录完后切到另一个 App、点进那里的输入框，文字就插到那里，不会再切回开始录音的 App。密码框和 Voxt 自己的窗口里不会输入任何东西；没有选中输入框时（访达的桌面、正在看的网页），结果留在答案卡片里，可以插入或复制，卡片的“插入”现在也插进当时在前台的 App。替换选区仍然只在选中文字的那个 App 里进行，应用增强的自动按键（Return 等）也只在那里按。历史记录仍记在开始录音的 App 下，详情里另外写明文字插入到了哪个 App（“已插入到 X”）。

#### 修复

- 粘贴了却没有出现的结果（Arc 里的网页）不会再悄无声息地丢掉。Voxt 现在会确认粘贴后焦点所在的输入框确实变了。没有变时，结果留在答案卡片里，并且留在剪贴板上（按 ⌘V 即可粘贴），不按自动按键，卡片里写明原因；另一个 app 开启了安全输入（密码框、终端的“安全键盘输入”）时，卡片会说明这一点。输入框无法告知内容的地方（部分网页编辑器）照常粘贴。原来的剪贴板在文字出现后立即恢复，不再固定等半秒。
- 长口述不会再因为识别超过 30 秒而失败、什么都没留下（内存紧张的 Mac 上一段 4 分 51 秒的口述）。识别时限改为 60 秒加录音每秒 1 秒；模型停止响应仍在 15 秒内发现。完整识别仍然失败时，使用实时识别的文字（照常整理、插入），浮窗会说明。
- 长录音时语音识别模型占用的内存大大减少：5 分钟的口述（精准模型）不到 4 GB，以前最多 10 GB，会把其他应用挤进 swap、拖慢 Mac。重新识别一段录音也更快了。
- 识别失败或没有识别出文字时，即使“保存录音”关着，这条记录也保留录音 7 天，方便重试；重试成功后删除。
- 词典里的词不会再出现在你没说过它们的口述末尾（“……嗯。Voxt，Omni，SGLang。”）。录音中没有说话的部分（比如长口述结束前、按停止之前的停顿）会让语音模型把给它的词典原样写出来。现在这样的结尾会和不带词典识别的同一段音频核对，删掉你没说过的词条；说过的词条照常保留，在句末也一样。
- Voxt 检查麦克风和辅助功能权限时（辅助进程启动或重启时）不会再偶尔崩溃。原因是这次检查可能在另一个系统线程上结束，而不是开始时的那个线程。
- 全局快捷键的 event tap 不再在 macOS 创建了它却剥掉键盘事件时误报就绪（只给了辅助功能、没给输入监控）：现在会回读并校验实际授予的事件，不完整的 tap 会被拒绝并每 2 秒重试；普通快捷键继续走备用通道，快捷键设置页会显示权限提示。
- 系统首选语言是英文的 Mac 上，中文口述不会再被转成英文。以前没在 设置 → 通用 里设主语言时，语音模型会被告知按界面语言（那里是英文）识别，快速模型把大部分中文录音翻成了英文；现在除非你设了主语言或者用中文界面，否则由模型自己判断语言。
- 口述翻译不会再先把词里的“额”“唔”去掉：“金额”变成了“金”，粤语的“唔”（不）被去掉、意思反了。现在只有分句开头、后面有停顿的“额”“唔”才当作语气词去掉。
- 翻译或改写的回答到了语言模型的长度上限就停了时，不再插入，也不再替换你选中的文字：它显示在答案卡片里，并提示可能不完整。
- **清空历史**（设置 → 隐私）现在也会把清掉的文字从数据库文件里抹掉。以前它们留在文件的空闲空间里，直到被覆盖，打开文件的人都能读到。
- 分段识别的长录音（历史里的“重新识别”和“重试”）在两段相接的地方不会再丢掉空格（“the Republic.Socrates begins”）；中文照旧不加空格。
- 一分钟以上的口述，停止前停顿一下时，结尾不会再多出一个没说过的词（“Okay.”）：结尾的停顿不再被单独切出来识别。
