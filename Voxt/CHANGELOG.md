# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

## [1.15.0] - 2026-09-26

### English

#### Added
- Added Google Gemini live transcription for voice input, with language hints, custom vocabulary, and live previews.
- Added a six-step onboarding guide covering model selection, permissions, shortcuts, and guided practice.
- Added streaming file transcription preprocessing with resource-aware scheduling, recovery, and reusable transcription previews.

#### Changed
- Improved local MLX model installation, loading, cleanup, and meeting transcription lifecycle handling.
- Added a tray history shortcut and more compact history entries in the status menu.
- Added model visibility controls and a Hidden filter for unused models in model settings and feature selectors.
- Moved the Settings entry into the settings sidebar for clearer navigation.
- Improved wake-up responsiveness by showing the wake-up interface before sound playback and moving system audio muting off the UI path.

#### Fixed
- Fixed Qwen multilingual finalization and several file task, speaker analysis, and final transcription recovery issues.
- Fixed shortcut cancellation and recording mute timing so wake-up sounds play reliably before recording starts.
- Reduced the delay before system audio is muted after the wake-up sound finishes.

### 简体中文

#### 新增
- 新增 Google Gemini 实时语音转写，支持语言提示、自定义词汇和实时预览。
- 新增六步引导流程，覆盖模型选择、权限设置、快捷键配置和练习操作。
- 新增文件转录流式预处理，支持资源感知调度、失败恢复和可复用的转录预览。

#### 改进
- 改进本地 MLX 模型安装、加载、清理以及会议转录的生命周期处理。
- 托盘菜单新增历史记录快捷入口，并精简状态菜单中的历史记录条目。
- 模型设置和功能模型选择器支持隐藏暂时不用的模型，并可通过“已隐藏”筛选查看。
- 将“设置”入口移入设置侧边栏，让导航更清晰。
- 优先显示唤醒界面再播放提示音，并将系统音频静音移出界面响应路径，提升唤醒响应速度。

#### 修复
- 修复 Qwen 多语言最终转写，以及文件任务、说话人分析和最终转录恢复中的多个问题。
- 修复快捷键取消和录音静音时序问题，确保录音开始前可靠播放唤醒提示音。
- 缩短唤醒提示音播放完成后系统音频静音的等待时间，减少录音启动延迟。

### 日本語

#### 追加
- Google Gemini のリアルタイム音声文字起こしに対応し、言語ヒント、カスタム語彙、ライブプレビューを追加しました。
- モデル選択、権限設定、ショートカット設定、練習操作を含む6ステップのオンボーディングガイドを追加しました。
- リソースを考慮したスケジューリング、復旧、再利用可能な文字起こしプレビューに対応した、ファイル文字起こしのストリーミング前処理を追加しました。

#### 変更
- ローカル MLX モデルのインストール、読み込み、解放、会議文字起こしのライフサイクル処理を改善しました。
- トレイメニューに履歴へのショートカットを追加し、ステータスメニューの履歴項目をよりコンパクトにしました。
- モデル設定と機能モデル選択画面で、使用しないモデルを非表示にし、「非表示」フィルターで確認できるようにしました。
- 「設定」項目を設定サイドバーに移動し、ナビゲーションを分かりやすくしました。
- ウェイクアップ画面を音声再生より先に表示し、システム音声のミュート処理を UI の応答経路から切り離して応答性を改善しました。

#### 修正
- Qwen の多言語最終処理、およびファイルタスク、話者分析、最終文字起こしの復旧に関する複数の問題を修正しました。
- ショートカットのキャンセルと録音ミュートのタイミングを修正し、録音開始前にウェイクアップ音が確実に再生されるようにしました。
- ウェイクアップ音の再生終了後にシステム音声をミュートするまでの遅延を短縮しました。

## [1.15.0-beta.3] - 2026-09-25

### English

#### Changed
- Added model visibility controls and a Hidden filter for unused models in model settings and feature selectors.
- Moved the Settings entry into the settings sidebar for clearer navigation.

#### Fixed
- Reduced the delay before system audio is muted after the wake-up sound finishes.

### 简体中文

#### 改进
- 模型设置和功能模型选择器支持隐藏暂时不用的模型，并可通过“已隐藏”筛选查看。
- 将“设置”入口移入设置侧边栏，让导航更清晰。

#### 修复
- 缩短唤醒提示音播放完成后系统音频静音的等待时间，减少录音启动延迟。

### 日本語

#### 変更
- モデル設定と機能モデル選択画面で、使用しないモデルを非表示にし、「非表示」フィルターで確認できるようにしました。
- 「設定」項目を設定サイドバーに移動し、ナビゲーションを分かりやすくしました。

#### 修正
- ウェイクアップ音の再生終了後にシステム音声をミュートするまでの遅延を短縮しました。

## [1.15.0-beta.2] - 2026-09-24

### English

#### Fixed
- Fixed shortcut cancellation and recording mute timing so wake-up sounds play reliably before recording starts.
- Improved wake-up responsiveness by rendering the wake-up interface before sound playback and moving system audio muting off the UI path.
- Unified system-audio layout and permission guidance in the permissions settings.

### 简体中文

#### 修复
- 修复快捷键取消和录音静音时序问题，确保唤醒提示音可靠播放后再开始录音。
- 优化唤醒响应速度，优先显示唤醒界面，并将系统音频静音移出界面响应路径。
- 统一权限设置中的系统音频布局和权限说明。

### 日本語

#### 修正
- ショートカットのキャンセルと録音ミュートのタイミングを修正し、録音開始前にウェイクアップ音が確実に再生されるようにしました。
- ウェイクアップ画面を音声再生より先に表示し、システム音声のミュート処理を UI の応答経路から切り離して応答性を改善しました。
- 権限設定におけるシステム音声のレイアウトと権限案内を統一しました。

## [1.15.0-beta.1] - 2026-09-22

### English

#### Added
- Added Google Gemini live transcription for voice input, with language hints, custom vocabulary, and live previews.
- Added a six-step onboarding guide with model selection, permission setup, shortcut configuration, and guided practice.
- Added streaming file transcription preprocessing with resource-aware scheduling, recovery, and reusable transcription previews.

#### Changed
- Improved local MLX model installation, loading, cleanup, and meeting transcription lifecycle handling.
- Added a tray history shortcut and more compact history entries in the status menu.

#### Fixed
- Fixed Qwen multilingual finalization and several file task, speaker analysis, and final transcription recovery issues.

### 简体中文

#### 新增
- 新增 Google Gemini 实时语音转写，支持语言提示、自定义词汇和实时预览。
- 新增六步引导流程，覆盖模型选择、权限设置、快捷键配置和练习操作。
- 新增文件转录流式预处理，支持资源感知调度、失败恢复和可复用的转录预览。

#### 改进
- 改进本地 MLX 模型安装、加载、清理以及会议转录的生命周期处理。
- 托盘菜单新增历史记录快捷入口，并精简状态菜单中的历史记录条目。

#### 修复
- 修复 Qwen 多语言最终转写，以及文件任务、说话人分析和最终转录恢复中的多个问题。

### 日本語

#### 追加
- Google Gemini のリアルタイム音声文字起こしに対応し、言語ヒント、カスタム語彙、ライブプレビューを追加しました。
- モデル選択、権限設定、ショートカット設定、練習操作を含む6ステップのオンボーディングガイドを追加しました。
- リソースを考慮したスケジューリング、復旧、再利用可能な文字起こしプレビューに対応した、ファイル文字起こしのストリーミング前処理を追加しました。

#### 変更
- ローカル MLX モデルのインストール、読み込み、解放、会議文字起こしのライフサイクル処理を改善しました。
- トレイメニューに履歴へのショートカットを追加し、ステータスメニューの履歴項目をよりコンパクトにしました。

#### 修正
- Qwen の多言語最終処理、およびファイルタスク、話者分析、最終文字起こしの復旧に関する複数の問題を修正しました。

## [1.14.1] - 2026-08-17

### English

#### Added
- Added model-aware remote ASR configuration for StepFun and Aliyun, including model-specific prompt, hotword, and realtime options.

#### Changed
- Kept the main window on its assigned macOS Space when opened from the status menu.

#### Fixed
- Fixed replacement dictionary entries not refreshing correctly after creation or update.

### 简体中文

#### 新增
- 新增 StepFun 和阿里云远程 ASR 的模型感知配置，支持按模型设置提示词、热词和实时识别选项。

#### 改进
- 从状态栏菜单打开主窗口时，保持窗口位于其分配的 macOS 工作区（Space）。

#### 修复
- 修复创建或更新替换词典条目后，条目未正确刷新的问题。

### 日本語

#### 追加
- StepFun と Aliyun のリモート ASR でモデルに応じた設定に対応し、モデル別のプロンプト、ホットワード、リアルタイム認識オプションを追加しました。

#### 変更
- ステータスメニューからメインウィンドウを開いたとき、割り当てられた macOS の Space にウィンドウを保持するようにしました。

#### 修正
- 置換辞書エントリの作成または更新後に、エントリが正しく更新されない問題を修正しました。

## [1.14.0] - 2026-08-07

### English

#### Added
- Added imported meeting audio processing with full transcription, speaker diarization, progress, cancellation, and completion notifications.
- Added editable meeting transcripts with waveform timelines, audio playback, and more durable meeting history.

#### Changed
- Improved meeting capture and history browsing with smoother long-transcript handling and clearer task status.
- Improved selected-text flows in Obsidian and Apple Intelligence model availability detection.

### 简体中文

#### 新增
- 新增会议音频文件处理，支持完整转写、说话人分离、进度、取消和完成通知。
- 新增可编辑会议转写，支持波形时间轴、音频播放，并提升会议历史持久性。

#### 改进
- 改进会议录制与历史浏览，长转写处理更流畅，任务状态更清晰。
- 改进 Obsidian 中的选中文本流程和 Apple Intelligence 模型可用性检测。

### 日本語

#### 追加
- 会議音声ファイルの処理に対応し、完全文字起こし、話者分離、進捗、キャンセル、完了通知を追加しました。
- 編集可能な会議文字起こし、波形タイムライン、音声再生を追加し、会議履歴の永続性を向上しました。

#### 変更
- 会議のキャプチャと履歴閲覧を改善し、長い文字起こしをよりスムーズに扱えるようにし、タスク状態を分かりやすくしました。
- Obsidian の選択テキストフローと Apple Intelligence モデルの利用可能性検出を改善しました。

## [1.13.9] - 2026-08-04

### English

#### Fixed
- Fixed excessive CPU usage after recording by clearing stale model initialization state and stopping hidden overlay animations.

### 简体中文

#### 修复
- 修复录音结束后 Voxt 空闲 CPU 占用过高：清理残留的模型初始化状态，并停止隐藏 Overlay 的动画刷新。

### 日本語

#### 修正
- 録音終了後に Voxt がアイドル状態でも CPU 使用率が高くなる問題を修正し、残留したモデル初期化状態を解消して非表示 Overlay のアニメーション更新を停止しました。

## [1.13.8] - 2026-08-03

### English

#### Added
- Added website and Sponsor links on the About page, and a Settings home Sponsor entry next to Changelog.
- Added meeting completion notifications for model and file jobs, including sandboxed drag-and-drop import results.
- Added an option to cap meeting history by count so older meetings can be cleaned up automatically.

#### Changed
- Made long meeting transcripts more responsive with a virtualized detail list.
- Reduced Final ASR stop-to-result latency while keeping the offline Final quality path.
- Kept MOSS dictionary Hotwords on Final only so short live windows no longer echo prompt terms mid-stream.

### 简体中文

#### 新增
- 关于页增加官网与赞助链接，设置首页在更新日志旁增加赞助入口。
- 会议模型与文件任务完成后支持系统通知，含沙盒拖放导入结果。
- 新增按条数限制会议历史，可自动清理较早的会议记录。

#### 改进
- 会议详情使用虚拟列表，长转写更流畅。
- 降低 Final ASR 从停止录音到出结果的等待，同时保留离线 Final 质量路径。
- 直播流不再注入 MOSS 词典热词，仅在 Final 阶段使用，避免短窗口中途回显提示词。

### 日本語

#### 追加
- About ページに公式サイトとスポンサーリンクを追加し、設定ホームの Changelog 横にもスポンサー入口を追加しました。
- 会議のモデル／ファイル処理完了時にシステム通知を表示し、サンドボックスでのドラッグ＆ドロップ取り込み結果にも対応しました。
- 会議履歴を件数で上限設定できるようにし、古い会議を自動整理できます。

#### 変更
- 会議詳細を仮想リスト化し、長い文字起こしでも応答性を改善しました。
- オフライン Final の品質パスは維持しつつ、Final ASR の停止から結果表示までの待ち時間を短縮しました。
- MOSS 辞書 Hotwords をライブストリームから外し Final のみに適用することで、短いライブ窓でのプロンプト語の途中エコーを防ぎます。

## [1.13.7] - 2026-07-31

### English

#### Fixed
- Fixed browser selection false positives for dictionary, note, and translation hotkeys so empty web inputs no longer claim those shortcuts, while real in-field and page selections still work on Accessibility-blind browsers like Arc.

### 简体中文

#### 修复
- 修复浏览器中查词、笔记、翻译快捷键的选区误判：空网页输入框不再抢占这些快捷键，同时在 Arc 等 Accessibility 受限浏览器中仍能正确识别输入框内与页面上的真实选区。

### 日本語

#### 修正
- ブラウザでの辞書・ノート・翻訳ショートカットの選択誤検知を修正し、空の Web 入力欄がホットキーを奪わないようにしつつ、Arc など Accessibility が制限されたブラウザでも入力欄内およびページ上の実際の選択を引き続き認識します。

## [1.13.6] - 2026-07-28

### English

#### Changed
- Redesigned the Feature overview cards with compact config badges, click-to-open feature pages, App Enhancement above Notes, and a non-scrolling overview layout.
- App Enhancement cards can show recent configured app icons when available.

### 简体中文

#### 改进
- 重做「功能」总览卡片：展示精简配置徽章、点击进入对应功能页，「应用增强」移到「笔记」上方，总览页不再滚动。
- 「应用增强」卡片在可用时显示最近配置的应用图标。

### 日本語

#### 変更
- 「機能」概要カードを刷新し、コンパクトな設定バッジ表示、クリックで各機能ページへ移動、「アプリ強化」を「ノート」より上に配置、概要ページのスクロールを停止しました。
- 「アプリ強化」カードで、利用可能な場合は最近設定したアプリのアイコンを表示できます。

## [1.13.5] - 2026-07-27

### English

#### Fixed
- Fixed selected-text hotkeys for browsers and Accessibility-limited editors, including recovering real mid-line selections in line-copy editors without inventing empty selections.
- Fixed Silero VAD MLX evaluation failures so local transcription startup throws a recoverable error instead of crashing.

### 简体中文

#### 修复
- 修复选中文本快捷键：在浏览器和 Accessibility 受限编辑器中更可靠地读取真实选区，并在整行复制类编辑器中恢复行内选区，避免在无选区时误造选中文本。
- 修复 Silero VAD 的 MLX 评估失败：本地转写启动时改为可恢复错误，而不再直接崩溃。

### 日本語

#### 修正
- テキスト選択ショートカットを修正し、ブラウザや Accessibility が制限されたエディタでも実際の選択を読み取り、行コピー系エディタでは行内選択を回復しつつ、未選択時に選択を捏造しないようにしました。
- Silero VAD の MLX 評価失敗を修正し、ローカル文字起こし開始時にクラッシュせず回復可能なエラーとして扱うようにしました。

## [1.13.4] - 2026-07-25

### English

#### Added
- Added Custom → Feature master toggles so translation, rewrite, notes, app enhancement, and meeting can be disabled from a two-column overview without affecting always-on transcription.
- Added imported meeting file analysis with bounded audio windows, full transcription, continuous speaker diarization, progress and cancellation, and history audio preference handling.

#### Changed
- Upgraded MLX Audio and related model runtimes so live ASR sessions end with structured STT output, and unreliable chunk timing is no longer treated as final timestamps.

#### Fixed
- Fixed selected-text hotkeys so editors like VS Code no longer treat the current line as a selection when nothing is actually selected.
- Fixed meeting microphone start when AVAudioEngine reports a sample rate that disagrees with the hardware input device.
- Fixed Create Hot Word so adding a term that already exists in the same scope reinforces it instead of failing with a SQLite duplicate-scope error.
- Fixed local model download authorization so security-scoped folder access is required and can be reauthorized from the main window sidebar.
- Fixed status-menu activation so the main window moves to the active Space when opened from the menu bar.
- Fixed app usage attribution so metrics track the final text injection destination and exclude meeting transcripts from branch counts.

### 简体中文

#### 新增
- 在「自定义 → 功能」中增加总开关，可从双列概览关闭翻译、改写、笔记、应用增强和会议，且不影响始终可用的转写。
- 支持导入会议音频文件分析：按窗口切分、完整转写、连续说话人分离，并提供进度、取消以及按历史音频偏好保存结果。

#### 改进
- 升级 MLX Audio 及相关模型运行时，使实时 ASR 会话以结构化 STT 输出结束，并不再将不可靠的分片时间戳当作最终时间戳。

#### 修复
- 修复选中文本快捷键：在未真正选中内容时，不再把 VS Code 等编辑器的当前行误当作选区。
- 修复会议麦克风启动：当 AVAudioEngine 报告的采样率与硬件输入设备不一致时也能正常开麦。
- 修复创建热词：同一作用域下已存在的词条会增强计数，而不再因 SQLite 唯一约束报错。
- 修复本地模型下载授权：下载前校验安全作用域文件夹访问，并可在主窗口侧栏重新授权。
- 修复状态栏菜单唤起主窗口时，窗口会切到当前桌面空间。
- 修复应用使用统计：按最终注入目标统计，并将会议转写排除在分支计数之外。

### 日本語

#### 追加
- 「カスタム → 機能」にマスタートグルを追加し、翻訳、リライト、ノート、アプリ強化、会議を二列の概要から無効化できるようにしました。常時利用の文字起こしには影響しません。
- 会議音声ファイルのインポート分析を追加しました。音声を区間に分割して完全文字起こしと連続話者分離を実行し、進捗・キャンセル、履歴音声の設定に沿った保存に対応します。

#### 変更
- MLX Audio と関連モデルランタイムを更新し、ライブ ASR セッションが構造化 STT 出力で終了するようにし、信頼できないチャンク時刻を最終タイムスタンプとして扱わないようにしました。

#### 修正
- テキスト選択ショートカットを修正し、実際に選択がないときに VS Code などのエディタで現在行を選択扱いにしないようにしました。
- AVAudioEngine のサンプルレートとハードウェア入力が一致しない場合でも、会議マイクを開始できるように修正しました。
- ホットワード作成を修正し、同じスコープに既存の用語がある場合は SQLite の重複エラーではなく強化するようにしました。
- ローカルモデルダウンロード前にセキュリティスコープ付きフォルダ認可を必須にし、メインウィンドウのサイドバーから再認可できるように修正しました。
- メニューバーから開いたときにメインウィンドウがアクティブな Space へ移動するよう修正しました。
- アプリ利用統計を修正し、最終的なテキスト注入先を追跡し、会議の文字起こしをブランチ集計から除外するようにしました。

## [1.13.3] - 2026-07-19

### English

#### Changed
- Reduced idle memory after local ASR and LLM use by reclaiming voice-activity, transcription, MLX cache, and allocator resources after model timeout, and by loading prequantized weights directly.

#### Fixed
- Prevented concurrent local ASR and LLM preparation from loading duplicate model instances, and ensured app shutdown waits for canceled model loads to finish.
- Ensured delayed resource reclamation still runs after meetings, temporary background work, model changes or deletion, and remote VAD-only sessions without resetting a newly started meeting's voice activity state.

### 简体中文

#### 改进
- 本地 ASR 和 LLM 模型超时卸载后，继续回收语音活动检测、转写器、MLX 缓存和分配器资源，并直接加载预量化权重，降低应用空闲内存占用。

#### 修复
- 修复本地 ASR 和 LLM 并发准备时可能重复加载模型的问题，并确保应用退出时等待已取消的模型加载任务真正结束。
- 确保会议、临时后台任务、模型切换或删除，以及仅使用远程模型和 VAD 的会话结束后仍会执行延迟资源回收，且不会重置新开始会议的语音活动状态。

### 日本語

#### 変更
- ローカル ASR と LLM のモデルがタイムアウトで解放された後、音声区間検出、文字起こし、MLX キャッシュ、アロケータのリソースも回収し、量子化済みモデルの重みを直接読み込むことでアイドル時のメモリ使用量を削減しました。

#### 修正
- ローカル ASR と LLM の同時準備でモデルが重複して読み込まれる問題を修正し、アプリ終了時にキャンセル済みのモデル読み込みが実際に終了するまで待機するようにしました。
- 会議、一時的なバックグラウンド処理、モデルの変更や削除、リモートモデルと VAD のみを使用するセッションの後も遅延リソース回収を実行し、新しく開始した会議の音声活動状態をリセットしないよう修正しました。

## [1.13.3-beta.3] - 2026-07-18

### English

#### Changed
- Further reduced idle memory after local ASR and LLM use by loading prequantized model weights without temporary graphs and returning released allocator pages more aggressively.

#### Fixed
- Prevented delayed VAD resource reclamation from resetting a newly started meeting and losing its initial voice activity state.

### 简体中文

#### 改进
- 直接加载预量化模型权重并更积极地归还已释放的内存页，进一步降低本地 ASR 和 LLM 使用后的空闲内存占用。

#### 修复
- 修复延迟回收 VAD 资源可能重置新开始的会议并丢失开头语音活动状态的问题。

### 日本語

#### 変更
- 量子化済みモデルの重みを一時グラフなしで読み込み、解放済みメモリページをより積極的に返却することで、ローカル ASR と LLM 使用後のアイドルメモリをさらに削減しました。

#### 修正
- 遅延した VAD リソース回収が新しく開始した会議をリセットし、冒頭の音声活動状態を失う可能性がある問題を修正しました。

## [1.13.3-beta.2] - 2026-07-18

### English

#### Changed
- Reduced memory retained after repeated Qwen3 local transcription by loading quantized weights directly without temporary full-model quantization graphs.

#### Fixed
- Prevented concurrent local ASR and LLM preparation from loading duplicate model instances, and ensured app shutdown waits for canceled model loads to finish.

### 简体中文

#### 改进
- 直接加载 Qwen3 量化权重，避免创建临时的全模型量化计算图，降低重复使用本地转写后的内存残留。

#### 修复
- 修复本地 ASR 和 LLM 并发准备时可能重复加载模型的问题，并确保应用退出时等待已取消的模型加载任务真正结束。

### 日本語

#### 変更
- Qwen3 の量子化済み重みを直接読み込み、一時的なモデル全体の量子化グラフを作成しないことで、ローカル文字起こしを繰り返した後のメモリ残留を削減しました。

#### 修正
- ローカル ASR と LLM の同時準備でモデルが重複して読み込まれる問題を修正し、アプリ終了時にキャンセル済みのモデル読み込みが実際に終了するまで待機するようにしました。

## [1.13.3-beta.1] - 2026-07-17

### English

#### Changed
- Reduced idle memory after local ASR and LLM use by releasing voice-activity, transcription, MLX cache, and allocator resources after the model timeout.

#### Fixed
- Prevented delayed memory reclamation from being skipped after meetings, temporary background work, model changes or deletion, and remote VAD-only sessions.

### 简体中文

#### 改进
- 本地 ASR 和 LLM 模型超时卸载后，继续释放语音活动检测、转写器、MLX 缓存和分配器资源，降低应用空闲内存占用。

#### 修复
- 修复会议、临时后台任务、模型切换或删除，以及仅使用远程模型和 VAD 的会话可能永久错过延迟内存回收的问题。

### 日本語

#### 変更
- ローカル ASR と LLM のモデルがタイムアウトで解放された後、音声区間検出、文字起こし、MLX キャッシュ、アロケータのリソースも解放し、アイドル時のメモリ使用量を削減しました。

#### 修正
- 会議、一時的なバックグラウンド処理、モデルの変更や削除、リモートモデルと VAD のみを使用するセッションの後に、遅延メモリ回収が実行されない場合がある問題を修正しました。

## [1.13.2] - 2026-07-17

### English

#### Changed
- Streamlined onboarding with inline microphone validation, main-language selection, clearer model choices, and improved settings controls.
- Redesigned the DMG installer with a branded drag-to-Applications layout.

#### Fixed
- Made quitting more reliable by canceling recordings, model tasks, permission callbacks, updates, and sync work before releasing native resources.
- Updated MLX Swift LM to fix local language model weight loading.

### 简体中文

#### 改进
- 简化首次引导，整合麦克风检测、主要语言选择、更清晰的模型选项和改进后的设置控件。
- 重新设计 DMG 安装界面，提供品牌化的拖拽安装布局。

#### 修复
- 在释放原生资源前取消录音、模型任务、权限回调、更新和同步工作，提升应用退出的可靠性。
- 更新 MLX Swift LM，修复本地语言模型权重加载问题。

### 日本語

#### 変更
- マイクの動作確認、メイン言語の選択、分かりやすいモデル候補、改善された設定コントロールを統合し、初回ガイドを簡素化しました。
- ブランドに合わせた、Applications フォルダへドラッグするレイアウトに DMG インストーラーを刷新しました。

#### 修正
- ネイティブリソースを解放する前に、録音、モデル処理、権限コールバック、更新、同期処理をキャンセルし、アプリ終了時の信頼性を改善しました。
- MLX Swift LM を更新し、ローカル言語モデルの重みの読み込みを修正しました。

## [1.13.1] - 2026-07-16

### English

#### Fixed
- Fixed rewrite conversations so follow-up context is preserved, canceled sessions restore the latest answer, and repeated or malformed responses are rejected.
- Kept remote model credentials out of metadata loads and resolved protected values only when sending requests.
- Fixed history clearing and retention so removing one category preserves other entries, including meeting history.

### 简体中文

#### 修复
- 修复改写对话的后续上下文保留、取消会话后的最新答案恢复，以及重复或格式异常响应的拦截。
- 避免在元数据加载时读取远程模型凭据，仅在发送请求时解析受保护的凭据值。
- 修复历史记录清理与保留逻辑，删除单个类别时不再误删其他记录，包括会议历史。

### 日本語

#### 修正
- 書き換え会話でフォローアップの文脈を保持し、キャンセル後に最新の回答を復元するとともに、重複または不正な形式の応答を拒否するよう修正しました。
- リモートモデルの認証情報をメタデータ読み込みから除外し、リクエスト送信時にのみ保護された値を解決するよう修正しました。
- 履歴の消去と保持処理を修正し、1 つのカテゴリを削除しても会議履歴を含む他の記録が保持されるようにしました。

## [1.13.0] - 2026-07-15

### English

#### Added
- Added meeting mode with live capture, speaker-aware transcripts, summaries, translation, resilient audio history, and meeting detail views.
- Rebuilt voice notes with configurable hotkeys, transcription enhancement, richer floating panels, list and timeline history views, editing, pagination, and improved external sync.
- Added OmniVAD, expanded MLX Audio and Sherpa ONNX local ASR support, Xiaomi MiMo remote ASR providers, dictionary categories, selected-text dictionary shortcuts, and guided setup.

#### Changed
- Reduced CPU and memory pressure during long meetings by bounding audio analysis, local inference, real-time translation, and summary workloads.
- Improved MOSS meeting transcription with voice-activity-gated local sessions, preserved speech onsets, structured speaker segments, and cleaned display text.
- Refined onboarding, model settings and download feedback, prompt presets, notifications, meeting history, translation language selection, and history status filtering.
- Moved remote provider credentials to the Data Protection Keychain and strengthened endpoint validation and diagnostic log redaction.

#### Fixed
- Improved meeting finalization recovery and durable history persistence so captured audio and transcripts survive interrupted processing more reliably.
- Fixed Qwen, MLX, and MOSS streaming transcript finalization and revision ordering to reduce duplicated, missing, stale, or protocol-formatted text.
- Fixed hotkey routing, long-press release handling, prompt editor autosave, GGUF UTF-8 decoding, remote ASR uploads, and recording startup after idle.

### 简体中文

#### 新增
- 新增会议模式，支持实时采集、按说话人整理的转写、摘要、翻译、可靠的音频历史记录和会议详情视图。
- 重构语音笔记，新增可配置快捷键、转写增强、更丰富的悬浮面板、列表与时间线历史视图、编辑、分页及外部同步改进。
- 新增 OmniVAD，扩展 MLX Audio 与 Sherpa ONNX 本地 ASR 支持，并加入小米 MiMo 远程 ASR provider、词典分类、选中文本词典快捷键和设置引导。

#### 改进
- 限制音频分析、本地推理、实时翻译和摘要任务的并发与负载，降低长时间会议中的 CPU 和内存压力。
- 通过语音活动门控的本地会话、语音起始音频保留、结构化说话人片段和文本清理，提升 MOSS 会议转写质量。
- 优化首次引导、模型设置与下载反馈、提示词预设、通知、会议历史、翻译语言选择和历史状态筛选。
- 将远程 provider 凭据迁移至 Data Protection Keychain，并加强 endpoint 校验和诊断日志脱敏。

#### 修复
- 加强会议结束恢复和历史记录持久化，使处理中断后已采集的音频与转写更可靠地保留。
- 修复 Qwen、MLX 与 MOSS 流式转写的结束处理和修订顺序，减少文本重复、缺失、过期或混入协议标记的问题。
- 修复快捷键路由、长按释放处理、提示词编辑器自动保存、GGUF UTF-8 解码、远程 ASR 上传和应用闲置后的录音启动问题。

### 日本語

#### 追加
- ライブキャプチャ、話者別の文字起こし、要約、翻訳、信頼性の高い音声履歴、会議詳細表示に対応した会議モードを追加しました。
- 音声ノートを再構築し、設定可能なホットキー、文字起こしの強化、より充実したフローティングパネル、リストとタイムラインの履歴表示、編集、ページ分割、外部同期の改善を追加しました。
- OmniVAD を追加し、MLX Audio と Sherpa ONNX のローカル ASR 対応を拡張するとともに、Xiaomi MiMo リモート ASR provider、辞書カテゴリ、選択テキスト用の辞書ショートカット、セットアップガイドを追加しました。

#### 変更
- 音声解析、ローカル推論、リアルタイム翻訳、要約処理の同時実行と負荷を制御し、長時間の会議における CPU とメモリの負荷を軽減しました。
- 音声活動で制御するローカルセッション、発話開始部分の保持、構造化された話者セグメント、表示テキストの整理により、MOSS の会議文字起こしを改善しました。
- オンボーディング、モデル設定とダウンロードのフィードバック、プロンプトプリセット、通知、会議履歴、翻訳言語の選択、履歴ステータスの絞り込みを改善しました。
- リモート provider の認証情報を Data Protection Keychain に移行し、endpoint 検証と診断ログの機密情報除去を強化しました。

#### 修正
- 会議終了時の復旧と履歴の永続化を強化し、処理が中断された場合でも収録済みの音声と文字起こしをより確実に保持できるようにしました。
- Qwen、MLX、MOSS のストリーミング文字起こしの終了処理と修正順序を改善し、テキストの重複、欠落、古い内容、プロトコル表記の混入を軽減しました。
- ホットキーのルーティング、長押し解除、プロンプトエディタの自動保存、GGUF の UTF-8 デコード、リモート ASR のアップロード、アイドル後の録音開始を修正しました。

## [1.13.0-beta.4] - 2026-07-14

### English

#### Changed
- Reduced CPU and memory pressure during long meetings by bounding audio analysis, local inference, real-time translation, and summary workloads.
- Refined meeting history and detail views, standardized translation language selection, and expanded history status filtering.

#### Fixed
- Improved meeting finalization recovery and durable history persistence so captured audio and transcripts survive interrupted processing more reliably.
- Fixed Qwen and MLX streaming transcript updates and translation revision ordering to reduce duplicated, missing, or stale text.
- Localized unavailable-audio messages and removed unusable playback controls from transcription details.

### 简体中文

#### 改进
- 限制音频分析、本地推理、实时翻译和摘要任务的并发与负载，降低长时间会议中的 CPU 和内存压力。
- 优化会议历史与详情界面，统一翻译语言选择方式，并扩展历史状态筛选范围。

#### 修复
- 加强会议结束恢复和历史记录持久化，使处理中断后已采集的音频与转写更可靠地保留。
- 修复 Qwen 与 MLX 流式转写更新及翻译修订顺序问题，减少文本重复、缺失或回退。
- 本地化音频不可用提示，并移除转写详情中不可用的播放控件。

### 日本語

#### 変更
- 音声解析、ローカル推論、リアルタイム翻訳、要約処理の同時実行と負荷を制御し、長時間の会議における CPU とメモリの負荷を軽減しました。
- 会議履歴と詳細画面を改善し、翻訳言語の選択方法を統一するとともに、履歴ステータスの絞り込み範囲を拡張しました。

#### 修正
- 会議終了時の復旧と履歴の永続化を強化し、処理が中断された場合でも収録済みの音声と文字起こしをより確実に保持できるようにしました。
- Qwen と MLX のストリーミング文字起こし更新、および翻訳の修正順序を修正し、テキストの重複、欠落、巻き戻りを軽減しました。
- 音声を利用できない場合のメッセージをローカライズし、文字起こし詳細画面から利用できない再生コントロールを削除しました。

## [1.13.0-beta.3] - 2026-07-14

### English

#### Added
- Rebuilt voice notes with configurable hotkeys, transcription enhancement, richer floating panels, list and timeline history views, editing, pagination, and improved external sync.
- Added OmniVAD support, expanded MLX Audio model capabilities, and automatic Silero VAD model provisioning for transcription and meetings.

#### Changed
- Refined onboarding, settings, model selection and download feedback, prompt presets, notifications, and history presentation.
- Migrated remote provider credentials to the Data Protection Keychain and strengthened endpoint validation and privacy protections for diagnostics.

#### Fixed
- Improved meeting finalization, speaker transcript preservation, audio-only recording persistence, MLX streaming latency, and recording startup after idle.
- Fixed GGUF UTF-8 output decoding, MOSS live transcription behavior, prompt editor autosave, and remote ASR upload reliability.

### 简体中文

#### 新增
- 重构语音笔记，新增可配置快捷键、转写增强、更丰富的悬浮面板、列表与时间线历史视图、编辑、分页及外部同步改进。
- 新增 OmniVAD 支持，扩展 MLX Audio 模型能力，并为转写和会议自动配置 Silero VAD 模型。

#### 改进
- 优化首次引导、设置、模型选择与下载反馈、提示词预设、通知和历史记录展示。
- 将远程 provider 凭据迁移至 Data Protection Keychain，并加强 endpoint 校验和诊断日志隐私保护。

#### 修复
- 改进会议结束处理、说话人转写保留、纯音频录音持久化、MLX 流式延迟，以及应用闲置后的录音启动表现。
- 修复 GGUF UTF-8 输出解码、MOSS 实时转写、提示词编辑器自动保存和远程 ASR 上传可靠性问题。

### 日本語

#### 追加
- 音声ノートを再構築し、設定可能なホットキー、文字起こしの強化、より充実したフローティングパネル、リストとタイムラインの履歴表示、編集、ページ分割、外部同期の改善を追加しました。
- OmniVAD 対応、MLX Audio モデル機能の拡張、文字起こしと会議向けの Silero VAD モデル自動準備を追加しました。

#### 変更
- オンボーディング、設定、モデル選択とダウンロードのフィードバック、プロンプトプリセット、通知、履歴表示を改善しました。
- リモート provider の認証情報を Data Protection Keychain に移行し、endpoint 検証と診断ログのプライバシー保護を強化しました。

#### 修正
- 会議終了処理、話者別文字起こしの保持、音声のみの録音保存、MLX ストリーミング遅延、アイドル後の録音開始を改善しました。
- GGUF の UTF-8 出力デコード、MOSS のライブ文字起こし、プロンプトエディタの自動保存、リモート ASR アップロードの信頼性を修正しました。

## [1.13.0-beta.2] - 2026-07-02

### English

#### Fixed
- Fixed beta release packaging so Sherpa ONNX local ASR is enabled in the signed app build.

### 简体中文

#### 修复
- 修复 beta 发布包打包流程，现在签名后的 App 会启用 Sherpa ONNX 本地 ASR。

### 日本語

#### 修正
- beta リリースのパッケージングを修正し、署名済みアプリで Sherpa ONNX ローカル ASR が有効になるようにしました。

## [1.13.0-beta.1] - 2026-07-02

### English

#### Added
- Added meeting mode with live capture, speaker-aware transcripts, summaries, translation support, and meeting detail windows.
- Added Sherpa ONNX local ASR support and Xiaomi MiMo remote ASR providers.
- Added dictionary categories, selected-text dictionary shortcuts, and richer app-context prompts for rewrite and enhancement workflows.
- Added setup guide onboarding and structured app logging for easier troubleshooting.

#### Changed
- Refined model settings, local model catalogs, and download status presentation across ASR and local translation models.
- Improved remote ASR audio upload preprocessing and voice activity handling for more reliable long-form transcription.

#### Fixed
- Fixed hotkey routing, long-press release handling, and overlay shortcut responsiveness.
- Improved logger write reliability and prompt fallback behavior.

### 简体中文

#### 新增
- 新增会议模式，支持实时采集、按说话人整理的转写、摘要、翻译以及会议详情窗口。
- 新增 Sherpa ONNX 本地 ASR 支持和 Xiaomi MiMo 远程 ASR provider。
- 新增词典分类、选中文本词典快捷键，以及改写和增强流程中的更完整应用上下文提示词。
- 新增设置引导和结构化应用日志，方便首次配置与问题排查。

#### 改进
- 优化 ASR 与本地翻译模型的模型设置、目录展示和下载状态呈现。
- 改进远程 ASR 音频上传预处理和语音活动处理，让长音频转写更可靠。

#### 修复
- 修复快捷键路由、长按释放处理和悬浮窗快捷键响应问题。
- 提升日志写入可靠性，并改进提示词回退行为。

### 日本語

#### 追加
- 会議モードを追加し、ライブキャプチャ、話者別の文字起こし、要約、翻訳、会議詳細ウィンドウに対応しました。
- Sherpa ONNX ローカル ASR と Xiaomi MiMo リモート ASR provider を追加しました。
- 辞書カテゴリ、選択テキスト用の辞書ショートカット、書き換えと強化フロー向けのより詳しいアプリコンテキストプロンプトを追加しました。
- 初期設定ガイドと構造化アプリログを追加し、設定とトラブルシューティングを進めやすくしました。

#### 変更
- ASR とローカル翻訳モデルのモデル設定、カタログ表示、ダウンロード状態の表示を見直しました。
- リモート ASR の音声アップロード前処理と音声活動処理を改善し、長時間の文字起こしをより安定させました。

#### 修正
- ホットキーのルーティング、長押し解除処理、オーバーレイのショートカット応答性を修正しました。
- ログ書き込みの信頼性とプロンプトのフォールバック動作を改善しました。

## [1.12.1] - 2026-07-02

### English

#### Fixed
- Fixed recording startup with 44.1 kHz input devices across MLX, WhisperKit, and remote ASR providers by matching capture taps to the active microphone hardware sample rate.

### 简体中文

#### 修复
- 修复输入设备为 44.1 kHz 时 MLX、WhisperKit 和远程 ASR 录音启动失败的问题，现在采集 tap 会匹配当前麦克风硬件采样率。

### 日本語

#### 修正
- 入力デバイスが 44.1 kHz の場合に MLX、WhisperKit、リモート ASR の録音開始が失敗する問題を修正し、キャプチャ tap が現在のマイクのハードウェアサンプルレートに合わせられるようにしました。

## [1.12.1-beta.2] - 2026-07-01

### English

#### Fixed
- Fixed recording startup with 44.1 kHz input devices across MLX, WhisperKit, and remote ASR providers by matching capture taps to the active microphone hardware sample rate.

### 简体中文

#### 修复
- 修复输入设备为 44.1 kHz 时 MLX、WhisperKit 和远程 ASR 录音启动失败的问题，现在采集 tap 会匹配当前麦克风硬件采样率。

### 日本語

#### 修正
- 入力デバイスが 44.1 kHz の場合に MLX、WhisperKit、リモート ASR の録音開始が失敗する問題を修正し、キャプチャ tap が現在のマイクのハードウェアサンプルレートに合わせられるようにしました。

## [1.12.1-beta.1] - 2026-06-30

### English

#### Fixed
- Fixed Direct Dictation recording startup with 44.1 kHz input devices by matching the recording tap to the microphone hardware sample rate.

### 简体中文

#### 修复
- 修复输入设备为 44.1 kHz 时 Direct Dictation 录音启动失败的问题，现在录音 tap 会匹配麦克风硬件采样率。

### 日本語

#### 修正
- 入力デバイスが 44.1 kHz の場合に Direct Dictation の録音開始が失敗する問題を修正し、録音 tap がマイクのハードウェアサンプルレートに合わせられるようにしました。

## [1.12.0] - 2026-06-17

### English

#### Added
- Added native live preview support for local Qwen3 MLX ASR models.

#### Changed
- Refined ASR and model settings copy, layout, and recommendation badges so local model options are easier to understand and configure.
- Improved SenseVoice integration and related ASR settings behavior across the app.

#### Fixed
- Fixed MLX capture startup so global hotkeys stay responsive while recording begins.
- Fixed StepFun SSE final transcript handling so completed realtime results are captured more reliably.
- Fixed SenseVoice finalization so completed results and related settings flows are more reliable.

### 简体中文

#### 新增
- 为本地 Qwen3 MLX ASR 模型新增原生实时预览支持。

#### 改进
- 优化 ASR 与模型设置的文案、布局和推荐标识，让本地模型选项更容易理解和配置。
- 改进 SenseVoice 集成以及相关 ASR 设置在应用内的整体行为。

#### 修复
- 修复 MLX 采集启动流程，让录音开始时全局快捷键保持响应。
- 修复 StepFun SSE 最终转写结果处理，让实时转写的完成结果更可靠。
- 修复 SenseVoice 收尾流程，让最终结果和相关设置流程更加稳定。

### 日本語

#### 追加
- ローカル Qwen3 MLX ASR モデル向けに、ネイティブのライブプレビュー対応を追加しました。

#### 変更
- ASR とモデル設定の文言、レイアウト、推奨バッジを見直し、ローカルモデルの設定内容を把握しやすくしました。
- SenseVoice の統合と関連する ASR 設定の挙動をアプリ全体で改善しました。

#### 修正
- MLX のキャプチャ開始処理を修正し、録音開始中もグローバルホットキーが応答し続けるようにしました。
- StepFun SSE の最終文字起こし結果の処理を修正し、リアルタイム結果の完了内容をより確実に取得できるようにしました。
- SenseVoice の終了処理を修正し、最終結果と関連する設定フローの安定性を向上させました。

## [1.12.0-beta.1] - 2026-06-04

### English

#### Added
- Added native live preview support for local Qwen3 MLX ASR models.

#### Changed
- Refined ASR and model settings copy and layout so local model options are easier to understand and configure.
- Improved SenseVoice integration and related ASR settings behavior across the app.

#### Fixed
- Fixed SenseVoice finalization so completed results and related settings flows are more reliable.

### 简体中文

#### 新增
- 为本地 Qwen3 MLX ASR 模型新增原生实时预览支持。

#### 改进
- 优化 ASR 和模型设置的文案与布局，让本地模型选项更容易理解和配置。
- 改进 SenseVoice 集成以及相关 ASR 设置在应用内的整体行为。

#### 修复
- 修复 SenseVoice 收尾流程，让最终结果和相关设置流程更加稳定。

### 日本語

#### 追加
- ローカル Qwen3 MLX ASR モデル向けに、ネイティブのライブプレビュー対応を追加しました。

#### 変更
- ASR とモデル設定の文言とレイアウトを見直し、ローカルモデルの設定内容を把握しやすくしました。
- SenseVoice の統合と関連する ASR 設定の挙動をアプリ全体で改善しました。

#### 修正
- SenseVoice の終了処理を修正し、最終結果と関連する設定フローの安定性を向上させました。

## [1.11.9] - 2026-05-30

### English

#### Added
- Added StepFun as a remote ASR provider with setup validation and hint support for realtime transcription.
- Added a log viewer in General > Logging with refresh, copy, and export actions plus appended app metadata for debugging.

#### Changed
- Expanded microphone startup and live capture diagnostics so recording issues are easier to trace from logs and settings.

#### Fixed
- Fixed the custom browser timeout flow so browser automation permission setup no longer gets blocked by launch timeout settings.
- Fixed model catalog empty states and log viewer preview loading so settings and exported logs refresh more reliably.

### 简体中文

#### 新增
- 新增 StepFun 远程 ASR provider，支持连接校验和实时转写提示词配置。
- 在通用 > 日志中新增日志查看器，支持刷新、复制、导出，并在日志末尾附带应用元数据，方便排查问题。

#### 改进
- 补强麦克风启动和实时采集诊断信息，让录音问题更容易结合日志和设置页面定位。

#### 修复
- 修复自定义浏览器超时流程，避免浏览器自动化权限设置被启动超时配置卡住。
- 修复模型目录空状态和日志查看器预览加载问题，让设置刷新和日志导出结果更稳定。

### 日本語

#### 追加
- StepFun をリモート ASR provider として追加し、接続確認とリアルタイム文字起こし向けヒント設定に対応しました。
- 一般 > ログにログビューアを追加し、更新・コピー・エクスポート操作と、デバッグ用のアプリメタデータ付与に対応しました。

#### 変更
- マイク起動時とライブキャプチャの診断情報を拡充し、録音トラブルをログや設定画面から追いやすくしました。

#### 修正
- カスタムブラウザのタイムアウト設定が原因で、ブラウザ自動操作の権限設定が止まってしまう問題を修正しました。
- モデルカタログの空状態とログビューアのプレビュー読み込みを修正し、設定更新とログ出力の反映を安定させました。

## [1.11.9-beta.1] - 2026-05-24

### English

#### Added
- Added a log viewer in General > Logging with refresh, copy, and export actions plus appended app metadata for debugging.
- Added richer remote ASR debug logs so realtime session startup, partial results, final results, and stop handling are easier to diagnose.

### 简体中文

#### 新增
- 在通用 > 日志中新增日志查看器，支持刷新、复制、导出，并在日志末尾附带应用元数据，方便排查问题。
- 补充更完整的远程 ASR 调试日志，让实时会话启动、中间结果、最终结果和停止流程更容易定位。

### 日本語

#### 追加
- 一般 > ログにログビューアを追加し、更新・コピー・エクスポート操作と、デバッグ用のアプリメタデータ付与に対応しました。
- リモート ASR のデバッグログを拡充し、リアルタイムセッションの開始、中間結果、最終結果、停止処理を追いやすくしました。

## [1.11.8] - 2026-05-23

### English

#### Changed
- Rebuilt the settings window with a new sidebar home, clearer dashboard cards, and cleaner About and update controls.
- Expanded model and provider presentation across settings so catalogs, install states, and feature model pickers are easier to scan.
- Moved local model storage out of the cache directory while keeping compatibility with existing installs.

#### Fixed
- Preserved settings list scroll position and prompt edits more reliably while settings views refresh.
- Surfaced local model load failures more clearly and reduced stale refresh issues after unloading local models.
- Started Whisper capture before model loading so recording startup stays more reliable.

### 简体中文

#### 改进
- 重构设置窗口，带来新的侧边栏首页、更清晰的 Dashboard 卡片，以及更干净的 About 和更新控制。
- 补全设置中的模型与 provider 展示，让模型目录、安装状态和功能模型选择器更易浏览。
- 将本地模型存储迁出缓存目录，同时保持对现有安装的兼容。

#### 修复
- 改善设置视图刷新时的状态保持，更稳定地保留设置列表滚动位置和提示词编辑内容。
- 更清晰地展示本地模型加载失败信息，并减少卸载本地模型后的陈旧刷新问题。
- 在模型加载前启动 Whisper capture，让录音启动流程更稳定。

### 日本語

#### 変更
- 設定ウィンドウを刷新し、新しいサイドバーのホーム、分かりやすい Dashboard カード、より整理された About と更新操作を追加しました。
- 設定内のモデルと provider の表示を拡充し、カタログ、インストール状態、機能別モデル選択を見やすくしました。
- 既存インストールとの互換性を保ちながら、ローカルモデルの保存先をキャッシュディレクトリ外へ移しました。

#### 修正
- 設定ビューの再読み込み中でも、設定リストのスクロール位置とプロンプト編集内容をより安定して保持するようにしました。
- ローカルモデルの読み込み失敗をより分かりやすく表示し、ローカルモデル解除後の古い更新状態が残る問題を減らしました。
- モデル読み込み前に Whisper capture を開始し、録音開始をより安定させました。

## [1.11.8-beta.1] - 2026-05-21

### English

#### Changed
- Rebuilt the settings window with a new sidebar, home dashboard, and cleaner About and update controls.
- Refreshed history and dashboard pages with clearer cards, empty states, and navigation for dictation, translation, rewrite, and notes.
- Expanded model and provider visuals across settings so feature model pickers and catalogs are easier to scan.

### 简体中文

#### 改进
- 重构设置窗口，带来新的侧边栏首页、主 Dashboard，以及更清晰的 About 和更新控制。
- 刷新历史和 Dashboard 页面，为听写、翻译、改写和笔记提供更清晰的卡片、空状态和导航结构。
- 补全设置中的模型与 provider 视觉标识，让功能模型选择器和模型目录更易浏览。

### 日本語

#### 変更
- 設定ウィンドウを刷新し、新しいサイドバーのホーム、メイン Dashboard、より分かりやすい About と更新操作を追加しました。
- 履歴と Dashboard ページを見直し、ディクテーション、翻訳、リライト、ノート向けにカード、空状態、ナビゲーションを分かりやすくしました。
- 設定内のモデルと provider の視覚表示を拡充し、機能別モデル選択とカタログを見やすくしました。

## [1.11.7] - 2026-05-17

### English

#### Added
- Improved automatic dictionary learning prompts for better correction term extraction.
- Added Codex provider controls for custom reasoning effort and remote model debugging.

#### Changed
- ASR hints now use a larger, usage-sorted dictionary candidate set for providers that support richer hints.
- Dictionary terms found in final inserted text now gain usage weight, improving future ASR hint ranking.
- Reduced idle model resource usage after recording and model selection changes.

#### Fixed
- Fixed translation fallback protection for very short same-language outputs that are not strict text prefixes.
- Fixed release uploads so DMG assets are attached to GitHub releases.

### 简体中文

#### 新增
- 优化自动学习词典提示词，提升纠错词汇提取效果。
- Codex provider 新增自定义 reasoning effort 和远端模型调试控制。

#### 改进
- ASR hints 现在会为支持更丰富热词的 provider 使用更大的、按使用情况排序的词典候选集。
- 最终插入文本中出现的已有词条现在会增加使用权重，让后续 ASR hint 排序更贴近真实使用。
- 降低录音结束和模型选择变更后的空闲模型资源占用。

#### 修复
- 修复同语言翻译输出很短但不是严格前缀时没有触发 fallback 保护的问题。
- 修复 GitHub release 上传流程，确保 DMG 资源会附加到 release。

### 日本語

#### 追加
- 自動辞書学習のプロンプトを改善し、修正語句を抽出しやすくしました。
- Codex provider にカスタム reasoning effort とリモートモデルデバッグの設定を追加しました。

#### 変更
- ASR hints は、より多くのヒントを扱える provider で、使用状況順に並べた大きな辞書候補セットを使うようになりました。
- 最終的に挿入されたテキストに既存の辞書語句が含まれる場合、その使用重みを増やして今後の ASR hint の並びを改善します。
- 録音終了後やモデル選択変更後のアイドル状態でのモデルリソース使用量を削減しました。

#### 修正
- 同一言語の翻訳結果が非常に短く、厳密なプレフィックスでない場合にも fallback 保護が働くようにしました。
- GitHub release に DMG アセットが添付されるよう、release アップロード処理を修正しました。

## [1.11.6] - 2026-05-15

### English

#### Added
- Added a Beta Updates switch on the About page so opted-in users can check beta appcast updates.
- Added beta-aware update feed selection for both manual checks and Sparkle background checks.
- Added a Codex auth file picker so users can choose the local `auth.json` used by ChatGPT Codex.

#### Changed
- Configuration export and import now preserves the Beta Updates preference.
- Switching update channels now clears stale update status before the next check.

#### Fixed
- Fixed Codex auth lookup in sandboxed builds by resolving the real user home and preserving selected auth file access.

### 简体中文

#### 新增
- 在 About 页新增 Beta 更新开关，让选择加入的用户可以检查 beta appcast 更新。
- 手动检查更新和 Sparkle 后台检查现在都支持按开关选择 beta 更新通道。
- 在 Codex 配置中新增 auth 文件选择器，用户可以选择 ChatGPT Codex 使用的本地 `auth.json`。

#### 改进
- 配置导入和导出现在会保留 Beta 更新偏好。
- 切换更新通道时会清空旧的更新状态，避免展示过期结果。

#### 修复
- 修复 sandbox 构建中 Codex auth 查找路径不正确的问题，现在会解析真实用户 home 并保留已选择 auth 文件的访问权限。

### 日本語

#### 追加
- About ページにベータ更新スイッチを追加し、希望するユーザーが beta appcast 更新を確認できるようにしました。
- 手動更新確認と Sparkle のバックグラウンド確認で、スイッチに応じた beta 更新チャンネルを使えるようにしました。
- Codex 設定に auth ファイルピッカーを追加し、ChatGPT Codex が使うローカル `auth.json` を選択できるようにしました。

#### 変更
- 設定のエクスポートとインポートで Beta Updates の設定を保持するようにしました。
- 更新チャンネルを切り替えたとき、次回確認前に古い更新状態を消去するようにしました。

#### 修正
- sandbox ビルドで Codex auth の参照先が誤る問題を修正し、実ユーザーの home 解決と選択済み auth ファイルへのアクセス保持に対応しました。

## [1.11.6-beta.2] - 2026-05-15

### English

#### Added
- Added a Codex auth file picker so users can choose the local `auth.json` used by ChatGPT Codex.

#### Changed
- Beta GitHub releases are now marked as prereleases.

#### Fixed
- Fixed Codex auth lookup in sandboxed builds by resolving the real user home and preserving selected auth file access.

### 简体中文

#### 新增
- 在 Codex 配置中新增 auth 文件选择器，用户可以选择 ChatGPT Codex 使用的本地 `auth.json`。

#### 改进
- Beta GitHub release 现在会标记为 prerelease。

#### 修复
- 修复 sandbox 构建中 Codex auth 查找路径不正确的问题，现在会解析真实用户 home 并保留已选择 auth 文件的访问权限。

### 日本語

#### 追加
- Codex 設定に auth ファイルピッカーを追加し、ChatGPT Codex が使うローカル `auth.json` を選択できるようにしました。

#### 変更
- Beta の GitHub release を prerelease としてマークするようにしました。

#### 修正
- sandbox ビルドで Codex auth の参照先が誤る問題を修正し、実ユーザーの home 解決と選択済み auth ファイルへのアクセス保持に対応しました。

## [1.11.6-beta.1] - 2026-05-15

### English

#### Added
- Added a Beta Updates switch on the About page so opted-in users can check beta appcast updates.
- Added beta-aware update feed selection for both manual checks and Sparkle background checks.
- Added beta appcast publishing support in release CI and documented the Voxt Server update contract.

#### Changed
- Configuration export and import now preserves the Beta Updates preference.
- Switching update channels now clears stale update status before the next check.

### 简体中文

#### 新增
- 在 About 页新增 Beta 更新开关，让选择加入的用户可以检查 beta appcast 更新。
- 手动检查更新和 Sparkle 后台检查现在都支持按开关选择 beta 更新通道。
- 发布 CI 新增 beta appcast 发布支持，并补充 Voxt Server 更新协议文档。

#### 改进
- 配置导入和导出现在会保留 Beta 更新偏好。
- 切换更新通道时会清空旧的更新状态，避免展示过期结果。

### 日本語

#### 追加
- About ページにベータ更新スイッチを追加し、希望するユーザーが beta appcast 更新を確認できるようにしました。
- 手動更新確認と Sparkle のバックグラウンド確認で、スイッチに応じた beta 更新チャンネルを使えるようにしました。
- リリース CI に beta appcast 公開サポートを追加し、Voxt Server の更新契約を文書化しました。

#### 変更
- 設定のエクスポートとインポートで Beta Updates の設定を保持するようにしました。
- 更新チャンネルを切り替えたとき、次回確認前に古い更新状態を消去するようにしました。

## [1.11.5] - 2026-05-15

### English

#### Fixed
- Fixed a startup crash that could affect existing users after the SQLite storage update.

### 简体中文

#### 修复
- 修复 SQLite 存储更新后可能影响已有用户的启动崩溃问题。

### 日本語

#### 修正
- SQLite ストレージ更新後に既存ユーザーで発生する可能性がある起動クラッシュを修正しました。

## [1.11.4] - 2026-05-15

### English

#### Added
- Added Codex as an OAuth-backed remote LLM provider.
- Added middle mouse button support as a transcription hotkey trigger.

#### Changed
- Updated the default cleanup prompts for more reliable dictated text handling.
- Updated the permission settings helper used to open newer macOS System Settings destinations.

#### Fixed
- Fixed a database startup crash caused by WAL setup on new or migrated storage.
- Improved stability around settings controls, transcription passes, and dictionary diff handling.

### 简体中文

#### 新增
- 新增 Codex 作为支持 OAuth 的远端 LLM provider。
- 新增鼠标中键作为转录快捷键触发方式。

#### 改进
- 更新默认清理提示词，让口述文本处理更稳定。
- 更新用于打开新版 macOS 系统设置位置的权限设置辅助库。

#### 修复
- 修复新建或迁移存储时 WAL 设置导致的数据库启动崩溃。
- 提升设置控件、转录处理和词典 diff 处理的稳定性。

### 日本語

#### 追加
- OAuth 対応のリモート LLM provider として Codex を追加しました。
- 文字起こしホットキーのトリガーとしてマウス中ボタンに対応しました。

#### 変更
- 口述テキスト処理の安定性を高めるため、既定のクリーンアッププロンプトを更新しました。
- 新しい macOS システム設定の移動先を開くための権限設定ヘルパーを更新しました。

#### 修正
- 新規または移行済みストレージで WAL 設定によりデータベース起動がクラッシュする問題を修正しました。
- 設定コントロール、文字起こし処理、辞書 diff 処理まわりの安定性を改善しました。

## [1.11.3] - 2026-05-14

### English

#### Added
- Added SQLite-backed dictionary and transcription history storage for faster large-list browsing and searching.

#### Changed
- Optimized the default prompts and cleanup rules for more reliable transcription handling.
- Improved history and dictionary performance, including lower idle overhead and fewer full dictionary scans during recognition.
- History and dictionary data now migrates into the new storage format on first use; opening history for the first time may be slow, with later launches expected to improve.
- Reduced idle CPU usage in model settings, model catalog refresh, and transcription polling.

#### Fixed
- Fixed ASR prompt echoes so prompt text is less likely to appear in the recording overlay output.
- The rewrite follow-up overlay now hides the continue button while follow-up recording is active.

### 简体中文

#### 新增
- 新增基于 SQLite 的词典与转录历史存储，大量记录下的浏览和搜索更快。

#### 改进
- 优化了默认提示词与清理规则，让转录文本处理更可靠。
- 优化了历史记录和词典性能，包括降低空闲开销，并减少识别过程中的全量词典扫描。
- 历史记录和词典数据会在首次使用时迁移到新的存储格式；首次打开历史记录可能较慢，后续启动和使用会改善。
- 降低了模型设置、模型目录刷新和转录轮询的空闲 CPU 占用。

#### 修复
- 修复了 ASR 提示词回显问题，提示词文本不再容易出现在录音浮窗输出中。
- 改写追问录音进行中时，现在会隐藏继续按钮。

### 日本語

#### 追加
- SQLite ベースの辞書と文字起こし履歴ストレージを追加し、大量の履歴でも閲覧と検索が速くなりました。

#### 変更
- 既定のプロンプトとクリーンアップルールを最適化し、文字起こしテキストの処理をより安定させました。
- 履歴と辞書のパフォーマンスを改善し、待機中の負荷を減らし、認識中の辞書全件スキャンを抑えました。
- 履歴と辞書データは初回利用時に新しい保存形式へ移行されます。初めて履歴を開くと時間がかかる場合がありますが、その後の起動と利用は改善されます。
- モデル設定、モデルカタログ更新、文字起こしポーリングの待機中 CPU 使用率を下げました。

#### 修正
- ASR プロンプトのエコーを修正し、プロンプト文が録音オーバーレイ出力に表示されにくくなりました。
- リライトの追加録音中は、続行ボタンを非表示にするようにしました。

## [1.11.2] - 2026-05-13

### English

#### Changed
- Updated the prompt system with simpler template variables. If you customized prompts, reset them to the new defaults before editing again.
- Improved transcription capture, long-form recognition, and LLM execution planning so transcript, translation, dictionary, and rewrite flows share more reliable text handling.
- Folded the deprecated Meeting workflow into transcript records and removed the separate Meeting entry points.
- Refined oMLX and remote model configuration, including localization, compatibility checks, and provider option handling.
- Removed the Phi 3.5 MoE local model option from the catalog.

#### Fixed
- Fixed fn combination tap hotkeys so they no longer fall through into unrelated tap shortcut behavior.

### 简体中文

#### 改进
- 更新了提示词体系，并大幅精简了模板变量。如果你自定义过提示词，建议先重置为新版默认提示词后再继续调整。
- 改进了转录采集、长音频识别与 LLM 执行规划，让转录、翻译、词典和改写流程共用更稳定的文本处理链路。
- 将已废弃的会议工作流收敛到转录记录中，并移除了独立的会议入口。
- 优化了 oMLX 与远端模型配置，包括多语言文案、兼容性检查和 provider 选项处理。
- 从本地模型目录中移除了 Phi 3.5 MoE 选项。

#### 修复
- 修复了 fn 组合键点击热键会继续触发其他点击快捷键行为的问题。

### 日本語

#### 変更
- プロンプト体系を更新し、テンプレート変数を大幅に簡素化しました。カスタムプロンプトを使っている場合は、新しい既定プロンプトにリセットしてから再編集することをおすすめします。
- 文字起こしキャプチャ、長時間認識、LLM 実行計画を改善し、文字起こし、翻訳、辞書、リライトのテキスト処理がより安定するようにしました。
- 廃止予定だった Meeting ワークフローを文字起こし履歴へ統合し、独立した Meeting 入口を削除しました。
- oMLX とリモートモデル設定を改善し、ローカライズ、互換性チェック、provider オプション処理を調整しました。
- ローカルモデル一覧から Phi 3.5 MoE オプションを削除しました。

#### 修正
- fn 組み合わせのタップホットキーが、別のタップショートカット動作へ続けて処理される問題を修正しました。

## [1.11.1] - 2026-05-10

### English

#### Added
- Added automatic corrected-term learning, so Voxt can watch quick manual fixes after text insertion and save confirmed terms to the dictionary automatically.
- Added a double-tap wake mode for Content Rewrite, so you can trigger rewrite directly from the transcription shortcut.

#### Changed
- Expanded dictionary and history correction tools with one-click ingest actions, richer correction details, and cleaner settings flows.
- Model settings now allow multiple MLX local model downloads at the same time, with clearer per-model progress and refresh behavior.
- Added advanced Ollama options in remote model settings, including response format, think mode, keep-alive, logprobs, and custom native options.

#### Fixed
- Fixed model settings refresh and download badges so local model state updates more reliably while downloads are running or storage paths change.
- Fixed Ollama connectivity and request handling so endpoint validation, structured output, and newer option combinations behave more reliably.

### 简体中文

#### 新增
- 新增自动学习纠正词条功能，Voxt 会在插入文本后短暂观察你的手动修正，并将确认后的词条自动写入词典。
- 为内容改写新增双击唤醒模式，现在可以通过双击转录快捷键直接触发改写。

#### 改进
- 扩展了词典与历史纠错工具，加入一键录入入口、更完整的纠错详情，以及更顺手的设置流程。
- Model 设置现在支持同时下载多个 MLX 本地模型，并提供更清晰的逐模型进度与刷新反馈。
- 远端模型设置新增了更完整的 Ollama 高级选项，包括响应格式、思考模式、保活时间、logprobs 和原生自定义选项。

#### 修复
- 修复了模型设置页的刷新与下载徽标状态问题，现在下载进行中或模型存储路径变化时，本地模型状态会更稳定地更新。
- 修复了 Ollama 的连通性检测与请求处理问题，现在端点校验、结构化输出和较新的参数组合表现更稳定。

### 日本語

#### 追加
- 自動修正語彙学習を追加し、Voxt がテキスト挿入後の短い手動修正を監視して、確定した語彙を辞書へ自動登録できるようになりました。
- Content Rewrite 向けのダブルタップ起動モードを追加し、文字起こしショートカットのダブルタップで直接リライトを開始できるようになりました。

#### 変更
- 辞書と履歴修正まわりを拡張し、ワンクリック取り込み、より詳しい修正表示、より整理された設定フローを追加しました。
- Model 設定で複数の MLX ローカルモデルを同時にダウンロードできるようにし、モデルごとの進捗表示と更新挙動もわかりやすくしました。
- リモートモデル設定に Ollama の詳細オプションを追加し、レスポンス形式、Think モード、keep-alive、logprobs、ネイティブオプションを設定できるようにしました。

#### 修正
- モデル設定の更新処理とダウンロードバッジを修正し、ダウンロード中や保存先変更時でもローカルモデル状態がより確実に反映されるようにしました。
- Ollama の接続確認とリクエスト処理を修正し、エンドポイント検証、構造化出力、新しいオプション組み合わせでも安定して動作するようにしました。

## [1.11.0] - 2026-05-08

### English

- New local model tuning and ASR pipeline updates are included in this release. If you run into issues, please report them in GitHub Issues.
- The current Meeting feature will be removed in a future release because it no longer matches the app direction. If there is enough demand, I plan to ship it later as a separate app.

#### Added
- Added a deprecation badge to the Start Meeting toggle in transcription settings so the upcoming removal is easier to notice.

#### Changed
- Localized more of Settings across General, Model, Dictionary, History, and About, including tabs, dialogs, action buttons, helper copy, and history info popovers.
- Simplified onboarding by removing the Meeting step and shortening the English back action label to `Prev`.
- Refined the settings sidebar and history tab spacing so English labels align more consistently.

#### Fixed
- Fixed mixed-language text in dictionary creation, model download settings, history audio settings, and history detail popovers so they follow the selected in-app UI language more reliably.
- Replaced the History clear-all text action with an icon button and added confirmation before deleting all history or notes.

### 简体中文

- 这个版本包含了新的本地模型调优与 ASR 链路优化。如果遇到问题，请到 GitHub Issues 反馈。
- 现有会议功能因与 App 后续发展方向不一致，后续版本会删除；如果确实有持续需求，我会考虑将其独立做成一个单独的 App。

#### 新增
- 为转录设置中的“开启会议”开关新增了即将删除提示徽标，方便提前感知后续下线计划。

#### 改进
- 继续完善了 General、Model、Dictionary、History、About 等设置页的多语言覆盖，补齐了 tabs、弹窗、操作按钮、说明文案和历史详情信息弹层。
- 精简了引导流程，移除了会议步骤，并将英文界面的返回按钮文案缩短为 `Prev`。
- 调整了设置侧栏与历史页 tabs 的间距和对齐，让英文界面的菜单排版更稳定。

#### 修复
- 修复了词典创建弹窗、模型下载设置、历史音频设置和历史详情信息弹层中的混合语言问题，现在会更稳定地跟随应用内所选语言。
- 将历史记录中的清空全部文本按钮改为图标按钮，并在删除全部历史或笔记前增加了二次确认。

### 日本語

- このリリースには、ローカルモデルの調整と ASR パイプラインの最適化が含まれています。問題があれば GitHub Issues で報告してください。
- 現在の Meeting 機能は、今後のアプリ方針と合わないため将来的に削除予定です。需要があれば、後に独立した別アプリとして提供する想定です。

#### 追加
- 文字起こし設定の Start Meeting トグルに廃止予定バッジを追加し、今後の削除予定がわかりやすくなりました。

#### 変更
- General、Model、Dictionary、History、About など設定画面の多言語対応をさらに拡張し、タブ、ダイアログ、操作ボタン、補助文言、履歴詳細ポップオーバーまで対応しました。
- オンボーディングを簡素化し、Meeting ステップを削除するとともに、英語 UI の戻る操作ラベルを `Prev` に短縮しました。
- 設定サイドバーと履歴タブの余白と整列を調整し、英語 UI でもメニュー配置がより安定するよう改善しました。

#### 修正
- 辞書作成ダイアログ、モデルダウンロード設定、履歴音声設定、履歴詳細ポップオーバーで混在していた言語表示を修正し、アプリ内で選択した UI 言語により確実に追従するようにしました。
- 履歴画面の全削除操作をアイコンボタンに変更し、履歴またはノートをすべて削除する前に確認を挟むようにしました。

## [1.10.4] - 2026-05-05

### English

#### Added
- Added a save action to prompt sections in settings so custom edits persist immediately when you leave the page. Thanks @zhaochenyang20.
- Added a direct Settings entry for MLX ASR models in the catalog so recognition context, preset, and ITN options are easier to reach. Thanks @yxs.

#### Changed
- Made the main window resizable with a safer minimum size and more stable traffic-light placement. Thanks @yxs.
- Localized the default dictionary ingest prompt to match the current UI language, so the prompt stays fully in one language and is easier for models to follow.
- Limited the dictionary advanced settings sheet height and added scrolling so long prompts stay manageable.

#### Fixed
- Fixed dictionary auto-correction so approved replacements apply before text is inserted, and delivered text, history, and rewrite payload stay consistent.
- Fixed model downloads so the active catalog row updates correctly during download, pause, and resume without switching the selected model automatically.
- Fixed one-click dictionary ingest so common JSON response formats, legacy line-based output, and truncated term lists are handled more reliably.

### 简体中文

#### 新增
- 为设置页中的提示词分区新增了保存操作，现在离开页面时，自定义编辑会立即持久化。感谢 @zhaochenyang20。
- 为 MLX ASR 模型目录新增了直接进入设置的入口，现在可以更方便地调整识别上下文、预设和 ITN 选项。感谢 @yxs。

#### 改进
- 让主窗口支持自由调整大小，同时设置了更稳妥的最小尺寸，并让左上角控制按钮的位置在布局变化时保持更稳定。感谢 @yxs。
- 让默认词典录入提示词跟随当前 UI 语言，整段提示词会保持单一语言风格，便于模型更稳定地理解。
- 为词典高级设置弹窗增加了最大高度限制和滚动支持，长提示词现在也更容易查看和编辑。

#### 修复
- 修复了词典自动纠偏无效的问题，现在命中的替换会在文本插入前生效，最终投递文本、历史记录和 rewrite payload 也会保持一致。
- 修复了模型下载状态显示问题，现在下载、暂停和继续时，对应目录行会正确更新，且不会自动切换当前选中模型。
- 修复了一键词典录入在常见 JSON 返回格式、旧版逐行输出和截断词条列表下的兼容性问题。

### 日本語

#### 追加
- 設定画面のプロンプト項目に保存操作を追加し、ページを離れたときにカスタム編集内容がすぐ保持されるようにしました。Thanks @zhaochenyang20.
- MLX ASR モデルのカタログに設定への直接入口を追加し、認識コンテキスト、プリセット、ITN オプションへすばやくアクセスできるようにしました。Thanks @yxs.

#### 変更
- メインウィンドウをリサイズ可能にし、より安全な最小サイズを設定したうえで、左上のトラフィックライト位置もレイアウト変更時に安定するよう改善しました。Thanks @yxs.
- 既定の辞書取り込みプロンプトを現在の UI 言語に合わせるようにし、プロンプト全体が単一言語で保たれてモデルが解釈しやすくなりました。
- 辞書の詳細設定ダイアログに最大高さ制限とスクロール対応を追加し、長いプロンプトでも扱いやすくしました。

#### 修正
- 辞書の自動補正が効かない問題を修正し、承認済みの置換がテキスト挿入前に適用され、最終出力、履歴、rewrite payload が一致するようにしました。
- モデルダウンロード時の状態表示を修正し、ダウンロード、一時停止、再開に応じて対象行が正しく更新され、選択中モデルが自動で切り替わらないようにしました。
- ワンクリック辞書取り込みを修正し、一般的な JSON 応答形式、旧来の行単位出力、途中で切れた候補一覧にもより安定して対応できるようにしました。

## [1.10.3] - 2026-05-02

### English

#### Added
- Added app-aware enhancement icons to the transcription overlay, so matched enhancement groups can show the active app or site icon while enhancement is running.

#### Changed
- Refined the recording overlay visuals with a calmer loading indicator, tighter waveform motion, and more stable card layout during processing.
- Updated onboarding and model management flows so local model downloads, pause and continue actions, and related status handling stay more consistent across the app.

#### Fixed
- Fixed remote model configuration copy so provider-specific search descriptions, active-provider notices, and custom model placeholders are localized more consistently.
- Fixed remote provider connectivity checks and DeepSeek defaults so newer DeepSeek models and structured-output requests behave more reliably in settings and enhancement flows.

### 简体中文

#### 新增
- 为转录悬浮卡片新增了应用增强图标提示，命中增强分组时，增强处理中会显示当前应用或站点图标。

#### 改进
- 优化了录音悬浮层的视觉表现，新的加载指示器更克制，声波动态更紧凑，处理中卡片布局也更稳定。
- 优化了引导与模型管理流程，让本地模型下载、暂停与继续操作，以及相关状态反馈在应用内保持更一致的体验。

#### 修复
- 修复了远端模型配置弹窗中的多语言文案问题，现在不同提供商的搜索说明、当前使用提示和自定义模型占位文案都会更一致地本地化。
- 修复了远端模型连通性检查和 DeepSeek 默认配置，使较新的 DeepSeek 模型与结构化输出请求在设置页和增强链路中表现更稳定。

### 日本語

#### 追加
- 文字起こしオーバーレイにアプリ強化アイコン表示を追加し、強化グループに一致した場合は処理中に現在のアプリやサイトのアイコンが表示されるようにしました。

#### 変更
- 録音オーバーレイの見た目を改善し、より落ち着いたローディング表示、引き締まった波形アニメーション、処理中でも安定したカードレイアウトに整えました。
- オンボーディングとモデル管理フローを見直し、ローカルモデルのダウンロード、一時停止・再開操作、関連する状態表示がアプリ全体でより一貫するようにしました。

#### 修正
- リモートモデル設定ダイアログの多言語文言を修正し、プロバイダーごとの検索説明、現在利用中の案内、カスタムモデルのプレースホルダーがより一貫してローカライズされるようにしました。
- リモートプロバイダーの接続チェックと DeepSeek の既定設定を修正し、新しい DeepSeek モデルと構造化出力リクエストが設定画面と強化フローでより安定して動作するようにしました。

## [1.10.2] - 2026-05-01

### English

#### Changed
- Improved local model downloads so large files can resume from partial progress, recover better from stalled transfers, and support pause, continue, and cancel actions directly in the app.
- Improved the software update flow so repeated clicks while the update window is opening are ignored and the update window stays in front more reliably once it appears.

#### Fixed
- Fixed the dictionary term editor so existing replacement aliases appear immediately when you open an entry for editing.

### 简体中文

#### 改进
- 优化了本地模型下载，大文件现在可以基于已下载进度继续传输，更稳地从卡住的下载中恢复，并直接支持暂停、继续和取消操作。
- 优化了应用更新流程，现在更新窗口打开过程中会忽略重复点击，更新窗口弹出后也能更稳定地保持在前台。

#### 修复
- 修复了词典词汇编辑弹窗，现在编辑已有词汇时，已存在的替换别名会在弹窗打开后立即显示。

### 日本語

#### 変更
- ローカルモデルのダウンロードを改善し、大きなファイルでも途中から再開できるようになり、停止したダウンロードからの復旧が安定し、アプリ内で一時停止・再開・キャンセルを直接行えるようにしました。
- アプリ更新フローを改善し、更新ウィンドウを開いている途中の連打を無視し、表示後も更新ウィンドウがより安定して前面に残るようにしました。

#### 修正
- 辞書用語の編集ダイアログを修正し、既存の置換エイリアスが編集画面を開いた直後に表示されるようにしました。

## [1.10.1] - 2026-04-29

### English

#### Added
- Added a custom paste shortcut option and configurable hotkey so you can insert the latest Voxt result into the current input field with Control + Command + V.

#### Changed
- Moved selected-text direct translation settings into the Translation feature menu and unified the related toggles with the app's switch style.
- Updated release notes formatting so GitHub releases now show separate English, Simplified Chinese, and Japanese sections instead of mixed multilingual bullets.

#### Fixed
- Fixed the Translation feature settings tab so opening it no longer crashes and the selected-text translation options keep their expected card layout.
- Fixed Obsidian note sync so renaming a synced note keeps the edited Markdown body stored in your vault.

### 简体中文

#### 新增
- 新增“自定义粘贴快捷键”开关和可配置热键，现在可以通过 Control + Command + V 将最近一次 Voxt 结果直接注入当前输入框。

#### 改进
- 将“选中文本直接翻译”相关设置移动到了功能子菜单的“翻译”页，并统一为应用内一致的 Switch 开关样式。
- 调整了发布说明格式，GitHub Release 现在会按英文、简体中文、日文分区显示，不再混合排列多语言条目。

#### 修复
- 修复了功能子菜单“翻译”设置页的崩溃问题，同时保持选中文本翻译相关卡片的原有布局样式。
- 修复了 Obsidian 笔记同步问题，现在重命名已同步笔记时，会保留 vault 中已经编辑过的 Markdown 正文。

### 日本語

#### 追加
- カスタム貼り付けショートカットの切り替えと設定可能なホットキーを追加し、Control + Command + V で最新の Voxt 結果を現在の入力欄へ直接挿入できるようにしました。

#### 変更
- 選択テキストの直接翻訳に関する設定を翻訳機能メニューへ移動し、関連トグルをアプリ内で統一された Switch スタイルにそろえました。
- リリースノートの形式を見直し、GitHub Release では英語・簡体字中国語・日本語の各セクションを分けて表示するようにしました。

#### 修正
- 翻訳機能設定タブを開いたときに発生していたクラッシュを修正し、選択テキスト翻訳オプションのカードレイアウトも想定どおり維持されるようにしました。
- Obsidian メモ同期を修正し、同期済みメモの名前を変更しても vault 内で編集済みの Markdown 本文が保持されるようにしました。

## [1.10.0] - 2026-04-28

### Added
- EN: Added Voxt Notes so you can split a live transcription into separate floating notes with short AI titles without stopping the current recording session.
- 简体中文：新增 Voxt 笔记功能，现在可以在不中断当前录音的情况下，把实时转录拆分成多条独立悬浮笔记，并为每条生成简短的 AI 标题。
- 日本語：Voxt メモ機能を追加し、現在の録音セッションを止めずにリアルタイム文字起こしを複数のフローティングメモへ分割し、それぞれに短い AI タイトルを付けられるようにしました。
- EN: Added Obsidian sync for Voxt Notes so selected notes can export directly into your chosen vault folder as Markdown files.
- 简体中文：为 Voxt 笔记新增 Obsidian 同步，所选笔记现在可以直接以 Markdown 文件导出到你指定的 vault 文件夹。
- 日本語：Voxt メモ向けに Obsidian 同期を追加し、選択したメモを指定した vault フォルダへ Markdown ファイルとして直接書き出せるようにしました。
- EN: Added Apple Reminders sync for Voxt Notes so synced notes can create and update reminders in the list you choose.
- 简体中文：为 Voxt 笔记新增 Apple 提醒事项同步，现在同步后的笔记可以在你选择的列表中创建和更新提醒。
- 日本語：Voxt メモ向けに Apple Reminders 同期を追加し、同期したメモから選択したリスト内のリマインダーを作成・更新できるようにしました。

### Changed
- EN: Improved local model pickers by grouping variants within each model family, making large model catalogs easier to browse and compare.
- 简体中文：优化了本地模型选择器，现在会按模型家族分组显示各个变体，大型模型目录更容易浏览和比较。
- 日本語：ローカルモデルの選択画面を改善し、各モデルファミリー内でバリアントをまとめて表示することで、大きなモデル一覧でも比較しやすくしました。
- EN: Simplified history cleanup settings so transcription and meeting history retention is easier to review, configure, and carry across devices.
- 简体中文：简化了历史清理设置，现在转录和会议历史的保留策略更容易查看、配置和随设备迁移。
- 日本語：履歴クリーンアップ設定を簡素化し、文字起こし履歴と会議履歴の保持ルールを確認・設定・デバイス間移行しやすくしました。

### Fixed
- EN: Fixed MLX recording stability so microphone switching behaves more reliably and the live transcript no longer shows internal prompt-only guidance text.
- 简体中文：修复了 MLX 录音稳定性问题，现在切换麦克风时表现更可靠，实时转录里也不会再显示仅供内部提示使用的引导文本。
- 日本語：MLX 録音の安定性を修正し、マイク切り替えがより確実になり、リアルタイム文字起こしに内部向けのガイダンステキストが表示されないようにしました。
- EN: Fixed Obsidian note sync so renaming a synced note no longer overwrites the edited Markdown body stored in your vault.
- 简体中文：修复了 Obsidian 笔记同步问题，现在重命名已同步笔记时，不会再覆盖 vault 中已经编辑过的 Markdown 正文。
- 日本語：Obsidian メモ同期を修正し、同期済みメモの名前を変更しても vault 内で編集済みの Markdown 本文が上書きされないようにしました。

## [1.9.12] - 2026-04-27

### Added
- EN: Added support for custom OpenAI-compatible ASR model IDs in Remote ASR settings, so you can test or switch to newer transcription models without waiting for a built-in preset.
- 简体中文：在 Remote ASR 设置中新增了自定义 OpenAI 兼容 ASR 模型 ID 支持，现在无需等待内置预设更新，也能直接测试或切换到新的转录模型。
- 日本語：Remote ASR 設定で OpenAI 互換 ASR のカスタムモデル ID を指定できるようにし、内蔵プリセットを待たずに新しい文字起こしモデルを試したり切り替えたりできるようにしました。

### Changed
- EN: Improved transcription enhancement so language cleanup now follows your primary and secondary language preferences without unexpectedly translating bilingual or non-primary-language speech into the primary language.
- 简体中文：优化了转录增强逻辑，现在会按照你的第一语言和副语言偏好进行清理优化，不会再把双语内容或非第一语言内容意外翻译成第一语言。
- 日本語：文字起こしの強化処理を改善し、第一言語と副言語の設定に沿って整形を行うようにして、バイリンガル入力や第一言語以外の発話が勝手に第一言語へ翻訳されないようにしました。

### Fixed
- EN: Fixed the General settings section so opening it no longer causes the app window to lose focus or jump behind other apps.
- 简体中文：修复了 General 设置分区的聚焦问题，现在打开该分区时应用窗口不会再失焦或突然跑到其他应用后面。
- 日本語：General 設定セクションを開いたときにアプリウィンドウがフォーカスを失ったり他のアプリの背後へ回ったりする問題を修正しました。
- EN: Fixed selected-text translation so the translation result overlay stays available for follow-up actions, avoids duplicate session-end sounds, and cleans up any residual microphone capture after the result appears or the overlay is dismissed.
- 简体中文：修复了选中文本翻译流程，现在翻译结果浮层会保留以便继续操作，并避免重复结束提示音，同时会在结果出现或关闭浮层后正确清理残留的麦克风采集。
- 日本語：選択テキスト翻訳のフローを修正し、翻訳結果オーバーレイを後続操作に使えるよう維持したうえで、終了音の重複を防ぎ、結果表示後やオーバーレイを閉じた後に残留したマイク収集も正しく解放するようにしました。
- EN: Fixed meeting hotkey gating so the dedicated meeting shortcut is ignored when Meeting Notes is disabled, and `fn+option` no longer falls back to the plain `fn` transcription hotkey.
- 简体中文：修复了会议快捷键的开关控制，现在未开启 Meeting Notes 时不会再响应专用会议快捷键，同时 `fn+option` 也不会再回退触发普通 `fn` 转录。
- 日本語：Meeting Notes が無効なときは専用会議ショートカットに反応しないよう修正し、`fn+option` が通常の `fn` 文字起こしショートカットへフォールバックしてしまう問題も解消しました。
- EN: Fixed custom remote model selection so custom ASR and meeting model IDs are preserved correctly instead of snapping back to built-in defaults.
- 简体中文：修复了远程模型自定义选择逻辑，现在自定义 ASR 和会议模型 ID 会被正确保留，不会再错误回退到内置默认值。
- 日本語：リモートモデルのカスタム選択処理を修正し、ASR と会議モデルのカスタム ID が正しく保持され、内蔵デフォルトへ勝手に戻らないようにしました。

## [1.9.11] - 2026-04-22

### Changed
- EN: Improved macOS permission guidance so Settings and onboarding now lead Accessibility and Input Monitoring through clearer System Settings flows, including direct guided handoff where drag-in authorization is required.
- 简体中文：优化了 macOS 权限引导，现在设置页和 onboarding 会以更清晰的系统设置流程引导辅助功能和输入监控权限，并在需要拖入授权的页面提供直接引导。
- 日本語：macOS の権限案内を改善し、Settings とオンボーディングからアクセシビリティ権限と入力監視権限をより分かりやすいシステム設定フローへ案内し、ドラッグでの許可が必要な画面ではそのまま誘導できるようにしました。

### Fixed
- EN: Fixed system audio capture permission guidance so features that need macOS audio capture now open the correct Privacy & Security page instead of leaving you at a generic settings screen.
- 简体中文：修复了系统音频捕获权限的引导流程，现在需要 macOS 音频捕获权限的功能会直接打开正确的“隐私与安全性”页面，而不是停留在通用设置首页。
- 日本語：システム音声キャプチャ権限の案内を修正し、macOS の音声キャプチャを必要とする機能が汎用設定画面ではなく、正しい「プライバシーとセキュリティ」のページを直接開くようにしました。

## [1.9.10] - 2026-04-21

### Fixed
- EN: Fixed long history info popovers so detailed text now wraps cleanly instead of stretching the panel too wide.
- 简体中文：修复了历史记录信息弹层的长文本显示问题，详细内容现在会自动换行，不会再把弹层横向撑得过宽。
- 日本語：履歴情報ポップオーバーの長文表示を修正し、詳細テキストが自動で折り返されてパネルが横に広がりすぎないようにしました。
- EN: Fixed the main window's first-launch positioning so it opens centered on the current primary screen instead of appearing offset near the top-right area.
- 简体中文：修复了主窗口首次启动时的定位问题，现在会直接居中显示在当前主屏上，不会再偏到顶部靠右的位置。
- 日本語：メインウィンドウの初回起動時の位置を修正し、右上寄りにずれて表示されることなく、現在のメイン画面中央に開くようにしました。
- EN: Fixed reopened main windows so they preserve the user's last window position, only falling back to recentering when the saved frame is invalid or off-screen.
- 简体中文：修复了主窗口重新打开时的定位行为，现在会保留用户上次摆放的位置，只有在保存的位置无效或跑出屏幕时才会回退到居中。
- 日本語：メインウィンドウを再表示した際の位置復元を修正し、前回の配置を保持しつつ、保存位置が無効または画面外の場合にのみ中央へ戻すようにしました。

## [1.9.9] - 2026-04-20

### Changed
- EN: Switched Sparkle auto-updates from package installer payloads to regular app archive updates, reducing unnecessary installer-style authorization prompts during app upgrades.
- 简体中文：将 Sparkle 自动更新从安装包载体切换为常规应用归档更新，减少了应用升级时不必要的安装器式授权提示。
- 日本語：Sparkle の自動更新をインストーラパッケージ方式から通常のアプリアーカイブ更新へ切り替え、アップグレード時の不要なインストーラ認証プロンプトを減らしました。

## [1.9.8] - 2026-04-20

### Changed
- EN: Improved transcription and meeting history details so they keep the full ongoing conversation in a single record, making follow-up context easier to review and continue.
- 简体中文：优化了转写和会议历史详情，现在同一轮持续对话会保存在同一条记录中，后续查看和继续追问时上下文更完整。
- 日本語：転写と会議の履歴詳細を改善し、継続中の対話全体を 1 件の履歴に保持するようにしたことで、見返しや続きを行う際の文脈がより完整になりました。
- EN: Expanded configuration import and export so more general settings and model setup can move between devices with fewer manual reconfiguration steps.
- 简体中文：扩展了配置导入导出范围，更多通用设置和模型配置现在都能一起迁移，减少了手动重新设置的步骤。
- 日本語：設定のインポートとエクスポート対象を拡張し、より多くの一般設定とモデル構成をまとめて移行できるようにして、手動での再設定を減らしました。

### Fixed
- EN: Fixed remote provider credential storage so saving one provider no longer clears the saved keys, tokens, or IDs from another provider.
- 简体中文：修复了远程服务商凭据保存逻辑，现在保存某一个服务商时，不会再清空其他服务商已保存的密钥、令牌或 ID。
- 日本語：リモートプロバイダの資格情報保存を修正し、あるプロバイダを保存した際に別のプロバイダのキー、トークン、ID が消えないようにしました。
- EN: Fixed Aliyun realtime ASR endpoint handling and model switching, so FunASR and Qwen realtime models use the correct WebSocket route more reliably.
- 简体中文：修复了阿里云实时 ASR 的端点处理和模型切换逻辑，FunASR 与 Qwen 实时模型现在会更可靠地使用正确的 WebSocket 地址。
- 日本語：Aliyun リアルタイム ASR のエンドポイント処理とモデル切り替えを修正し、FunASR と Qwen のリアルタイムモデルがより確実に正しい WebSocket ルートを使うようにしました。
- EN: Fixed the remote model selector so already configured models are recognized correctly in filters and setup prompts, reducing false "not configured" states.
- 简体中文：修复了远程模型选择器的状态判断，已配置完成的模型现在会在筛选和提示中被正确识别，减少误报“未配置”的情况。
- 日本語：リモートモデルセレクタの状態判定を修正し、設定済みモデルがフィルタや案内で正しく認識されるようにして、誤った「未設定」表示を減らしました。

## [1.9.7] - 2026-04-20

### Fixed
- EN: Improved Settings responsiveness by keeping sidebar sections warm, moving dictionary and history reload work off the main thread, and paginating large dictionary lists so switching between sections stays smoother.
- 简体中文：优化了设置页响应速度，通过保留侧边栏分区状态、将字典与历史记录的重载移出主线程，并为大词典列表加入分页，让分区切换更加顺滑。
- 日本語：Settings の応答性を改善し、サイドバー各セクションの状態を保持したまま、辞書と履歴の再読み込みをメインスレッド外へ移し、大きな辞書一覧にはページングを追加したことで、セクション切り替えがより滑らかになりました。
- EN: Fixed Settings sidebar rows so clicking anywhere across the highlighted row switches sections more reliably, and kept the Dictionary `Clear All` button at a stable width.
- 简体中文：修复了设置页侧边栏条目的点击区域，现在点击高亮整行内的任意位置都能更稳定地切换分区，并让词典里的 `清空全部` 按钮保持稳定宽度。
- 日本語：Settings サイドバーの行クリック領域を修正し、ハイライトされた行のどこをクリックしてもより確実にセクションを切り替えられるようにし、辞書内の `Clear All` ボタン幅も安定させました。
- EN: Fixed sided modifier shortcut detection so hotkeys that distinguish left and right Command, Option, Control, or Shift keys trigger more consistently.
- 简体中文：修复了区分左右修饰键的快捷键识别，现在区分左右 Command、Option、Control 或 Shift 的热键触发会更加稳定。
- 日本語：左右の修飾キーを区別するショートカット検出を修正し、左右の Command、Option、Control、Shift を区別するホットキーがより安定して発火するようにしました。

## [1.9.6] - 2026-04-19

### Added
- EN: Added the local Cohere Transcribe model to the MLX speech model library, so you can install another multilingual on-device ASR option directly from Settings.
- 简体中文：在 MLX 语音模型库中新增了本地 Cohere Transcribe 模型，现在你可以直接在设置里安装这一多语言本地 ASR 选项。
- 日本語：MLX 音声モデルライブラリにローカル版 Cohere Transcribe を追加し、Settings から多言語対応のオンデバイス ASR を新たに導入できるようにしました。

### Changed
- EN: Expanded local ASR settings with per-model tuning dialogs, including recognition presets, main-language following, and model-specific context or prompt controls where supported.
- 简体中文：扩展了本地 ASR 设置，支持按模型分别调整识别预设、跟随主语言，以及模型支持时的上下文或提示词控制。
- 日本語：ローカル ASR 設定を拡張し、モデルごとのチューニングダイアログから認識プリセット、主言語追随、対応モデルでの context / prompt 制御を調整できるようにしました。
- EN: Improved the model catalog so local ASR entries show whether they support your current primary language, making it easier to choose the right model before installing.
- 简体中文：改进了模型目录，本地 ASR 条目现在会显示是否支持你当前的主语言，安装前更容易选对模型。
- 日本語：モデルカタログを改善し、ローカル ASR エントリに現在の主言語への対応状況を表示するようにしたため、インストール前に適切なモデルを選びやすくなりました。

### Fixed
- EN: Reduced CPU usage during local model downloads and added an in-progress download badge in Settings so active installs stay visible and can jump back to the model list quickly.
- 简体中文：降低了本地模型下载过程中的 CPU 占用，并在设置中新增下载中的提示徽标，让正在安装的模型始终可见并能快速跳回模型列表。
- 日本語：ローカルモデルのダウンロード中に発生していた CPU 使用率を抑え、Settings に進行中ダウンロードのバッジを追加して、インストール状況の確認とモデル一覧への復帰をしやすくしました。
- EN: Fixed local ASR configuration sheets so Whisper and other tuning controls follow the app language more consistently across labels, presets, and helper text.
- 简体中文：修复了本地 ASR 配置弹窗的多语言显示，现在 Whisper 等调参项的标签、预设和说明文案会更一致地跟随界面语言。
- 日本語：ローカル ASR 設定シートの多言語表示を修正し、Whisper などのチューニング項目でラベル、プリセット、補助テキストがより一貫してアプリ言語に追随するようにしました。

## [1.9.5] - 2026-04-16

### Fixed
- EN: Fixed the Settings permission badge so it no longer warns about unopened permissions when the currently enabled features only require access you have already granted.
- 简体中文：修复了设置页的权限提示徽标，当当前启用的功能只依赖已授权权限时，不会再错误显示还有权限未开启。
- 日本語：設定画面の権限バッジを修正し、現在有効な機能がすでに許可済みの権限だけを必要とする場合は、未許可の警告が表示されないようにしました。
- EN: Fixed local model lists so known file sizes stay visible more reliably, and model metadata or downloads automatically retry through the mirror when the primary Hugging Face endpoint is rate-limited or unavailable.
- 简体中文：修复了本地模型列表的文件大小显示，已知大小会更稳定地展示；当 Hugging Face 主站遇到限流或不可用时，模型元数据与下载会自动改走镜像重试。
- 日本語：ローカルモデル一覧のファイルサイズ表示を修正し、既知のサイズをより安定して表示するようにしました。あわせて、Hugging Face の本家エンドポイントがレート制限または利用不可の場合は、モデル情報取得とダウンロードをミラー経由で自動再試行するようにしました。

## [1.9.4] - 2026-04-13

### Changed
- EN: Moved one-click dictionary ingest model and prompt controls into Dictionary Advanced Settings and added cancel support while a scan is running.
- 简体中文：将一键录入使用的模型和提示词移至词典高级设置，并支持在扫描过程中取消录入。
- 日本語：ワンクリック取り込みのモデルとプロンプト設定を辞書の詳細設定へ移し、走査中のキャンセルにも対応しました。
- EN: Tightened dictionary ingest term selection so common words, mixed-language filler, route details, and obvious transcript fragments are filtered out more aggressively.
- 简体中文：进一步收紧了词典录入的筛词规则，更积极地排除常见词、混合语言口语填充、路线信息和明显的转写片段。
- 日本語：辞書取り込みの語句選別をさらに厳しくし、一般語、混在言語のつなぎ語、経路情報、明らかな書き起こし断片をより強く除外するようにしました。

### Fixed
- EN: Fixed dictionary ingest parsing by requiring validated structured array output from supported language models before terms are written into the dictionary.
- 简体中文：修复了词典录入解析流程，要求受支持的大模型先返回经过校验的结构化数组结果后才写入词典。
- 日本語：辞書取り込みの解析を修正し、対応する言語モデルが検証済みの構造化配列を返した場合にのみ辞書へ書き込むようにしました。
- EN: Reduced idle memory in Settings and fixed transcription finalization work that could hit newer Swift concurrency isolation checks.
- 简体中文：降低了设置页的空闲内存占用，并修复了转写收尾流程在新版 Swift 并发隔离检查下可能出现的问题。
- 日本語：設定画面の待機時メモリ使用量を抑え、新しい Swift の並行性分離チェックで転写確定処理が不安定になる問題を修正しました。

## [1.9.3] - 2026-04-12

### Fixed
- Fixed rewrite follow-up answers so Aliyun-backed continue conversations no longer collapse into empty fallback responses.
- Improved rewrite conversation stability with safer prompt assembly and overlay teardown handling during longer answer sessions.

## [1.9.2] - 2026-04-11

### Added
- Added a dedicated transcription detail window with chat-style history, timestamps, copy actions, and follow-up questions.

### Changed
- Reworked rewrite follow-up interactions so continue mode keeps the action available, streams updates more clearly, and follows new content more reliably.
- Improved remote LLM provider handling and streaming parsing for rewrite conversations, including better compatibility with Aliyun chat-completions style responses.

### Fixed
- Fixed feature-specific model routing so transcription, translation, rewrite, and meeting workflows use their own configured models more consistently in runtime and history.
- Fixed ASR runtime switching so changing the selected speech model no longer leaves the first recording attempt unresponsive.

## [1.9.1] - 2026-04-10

### Fixed
- Fixed remote ASR realtime sessions so Doubao and Aliyun connections release network resources more cleanly during long-running use.

## [1.9.0] - 2026-04-09

### Added
- Added filtered feature-specific model pickers so you can quickly choose installed, configured, local, remote, and in-use models for transcription, translation, rewrite, and meeting workflows.
- Added dedicated meeting workflow controls for meeting-only ASR, summary model, prompt, realtime translation target, and screen-sharing visibility in onboarding and Settings.
- Added Doubao dictionary boosting controls so active Voxt dictionary hotwords and corrections can be sent with each ASR request when needed.

### Changed
- Reworked Settings and onboarding around feature-focused flows, with clearer guidance for permissions, model setup, downloads, and shortcut configuration.
- Improved remote provider setup with endpoint presets, meeting ASR testing, and clearer provider-specific guidance.

## [1.8.6] - 2026-04-07

### Fixed
- Fixed Doubao remote ASR streaming shutdown so successful transcriptions no longer trigger protocol sequence mismatch warnings when you stop a session.

## [1.8.5] - 2026-04-07

### Added
- Added dedicated Direct Dictation settings for locale selection, contextual phrases, on-device recognition, punctuation, and partial-result preferences.
- Added live download progress, cancel actions, and clearer status details for MLX, Whisper, and custom local model downloads in Settings and onboarding.

### Changed
- Improved onboarding and model settings so download states stay visible, demo previews handle loading more gracefully, and model lists fit the available space more reliably.

### Fixed
- Fixed Doubao remote ASR live sessions so transcription and meeting captures keep streaming updates and final transcript segments more reliably.
- Fixed Doubao remote ASR retries and transcript assembly so trailing text is less likely to be dropped after interrupted uploads.
- Fixed remote ASR failure handling so network, timeout, authentication, quota, and availability problems now show clearer user-facing guidance.

## [1.8.4] - 2026-04-06

### Added
- Added a new Chinese voice end command preset for saying `好了`.

### Changed
- Expanded the built-in local MLX speech model list with more Qwen3 ASR, Voxtral, Parakeet, Granite, FireRed, and SenseVoice options.

### Fixed
- Fixed remote provider credentials so API keys and tokens are stored in the macOS keychain instead of exported preferences.
- Fixed sided modifier shortcut recording so left and right modifier keys are captured more reliably.

## [1.8.3] - 2026-03-27

### Fixed
- Fixed permission warnings so Settings only asks for permissions that are actually required by enabled features.
- Fixed App Branch matching so groups with an empty prompt no longer override the default enhancement flow.
- Fixed rewrite output handling so the answer card stays available more reliably when that mode is enabled.
- Improved meeting detail timing and summary defaults for clearer post-meeting review.

## [1.8.2] - 2026-03-26

### Fixed
- Fixed the local Whisper transcription dependency so Voxt builds reliably with the current Xcode toolchain.
- Fixed custom select controls in Settings so choosing a different option updates immediately again.

## [1.8.1] - 2026-03-25

### Changed
- Lowered the minimum supported macOS version to 15.0 so Voxt can run on more Macs.

### Fixed
- Improved compatibility on macOS 15 by gracefully falling back when Apple Intelligence features or newer system audio APIs are unavailable.
- Added clearer diagnostics around microphone connect and disconnect handling, device priority evaluation, and automatic microphone switching.

## [1.8.0] - 2026-03-24

### Added
- Added an AI meeting summary sidebar with saved summaries and follow-up chat directly inside Meeting details.
- Added a screen sharing toggle in the meeting overlay so meeting captures can include shared-screen context when needed.

### Changed
- Refreshed the settings and Meeting detail interfaces with more consistent controls, layouts, and localized guidance.

### Fixed
- Improved meeting summary and onboarding reliability with tighter prompt handling and more isolated preference test coverage.

## [1.7.1] - 2026-03-23

### Added
- Added a first-run setup guide in the main window with step-by-step onboarding for language, models, transcription, translation, rewrite, app enhancement, and meeting notes.

### Changed
- Refined the onboarding flow with simpler shortcut presets, contextual permission prompts, inline demo videos, and localized guidance across English, Simplified Chinese, and Japanese.

### Fixed
- Fixed meeting transcript translation updates so existing translated text stays visible while background refreshes complete, instead of flashing a loading state on every update.

## [1.7.0] - 2026-03-23

### Added
- Added a separate on-device Whisper engine powered by WhisperKit, with built-in Whisper model downloads and configurable realtime, VAD, timestamp, and temperature options.
- Added Meeting Notes (Beta), a dedicated long-running meeting capture flow with its own shortcut, floating meeting card, Meeting history entries, and detail window review/export support.
- Added meeting-specific Remote ASR setup for Doubao ASR and Aliyun Bailian ASR, including dedicated Meeting ASR model selection and request-path testing.

### Changed
- Refined the meeting capture experience with clearer model initialization states, pause/resume controls, timestamped segments, click-to-copy, and smoother long-running overlay behavior.
- Expanded localization and configuration transfer coverage for the new meeting workflow and Whisper settings across English, Simplified Chinese, and Japanese.

### Fixed
- Fixed `fn` hotkey recovery after idle and hardened recording start handling so shortcut-triggered capture sessions resume more reliably.
- Improved Whisper startup and meeting control stability during longer transcription sessions.

## [1.6.6] - 2026-03-20

### Added
- Added a General setting to cancel the active overlay with `Esc`, plus optional overlay appearance controls for opacity, corner radius, and screen edge distance.
- Added manual dictionary replacement match terms so custom aliases can map directly to a standard term.

### Changed
- Refined the recording waveform so the voice bars now feel like audio waves moving from left to right.

### Fixed
- Reduced idle memory after local MLX transcription or LLM use by unloading on-device models after they sit unused.
- Fixed General configuration export/import so the latest `Esc` cancel and overlay appearance settings are preserved.
- Fixed the permissions page so Speech Recognition only appears when the selected transcription engine actually needs system dictation.

## [1.6.5] - 2026-03-19

### Added
- Added built-in Doubao ASR 2.0 and 1.0 model options, with 2.0 now used as the default selection for new setups.

### Fixed
- Fixed Doubao ASR realtime routing so 2.0 now connects through the supported streaming endpoint, while file and connectivity-test flows keep using the compatible endpoint and payload format.
- Reduced extra Doubao diagnostic log noise during normal recording and settings connectivity tests.

## [1.6.4] - 2026-03-19

### Fixed
- Reduced idle and active CPU usage in the status menu and recording overlay by replacing broad menu rebuild triggers with targeted updates and by stopping hidden overlay animations from continuing to drive SwiftUI layout work.
- Improved recording waveform feedback so the voice bars animate more smoothly and remain visually clearer during transcription while keeping the lower CPU overhead.
- Removed answer card button tooltips to reduce redundant hover chrome in the rewrite result UI.

## [1.6.3] - 2026-03-19

### Fixed
- Reissued the latest patch release with Sparkle-compatible plain-text auto-update release notes so update feeds no longer depend on Markdown-formatted release bodies.
- Includes the rewrite overlay, answer injection, dictionary ingest, history UI, and `fn` hotkey fixes shipped in the 1.6.2 patch line.

## [1.6.2] - 2026-03-19

### Changed
- Simplified rewrite output behavior so rewrite always shows the answer card, while keeping the General settings UI aligned with the new fixed behavior.
- Improved dictionary ingest and history surfaces by reducing candidate-only UI noise and surfacing direct dictionary hits more clearly.

### Fixed
- Fixed rewrite answer card actions and loading feedback, including the inject action, loading spinner sizing, and icon alignment in the recording overlay.
- Fixed rewrite answer injection availability by improving focused input detection and adding a safer fallback for apps that do not expose standard accessibility focused elements.
- Fixed modifier-only hotkey handling so `fn` no longer steals unrelated combos such as `fn+1`, while preserving dedicated `fn+shift` and `fn+control` shortcuts.
- Fixed overlay session startup so stale transcription text is cleared before a new recording card appears.

## [1.6.1] - 2026-03-17

### Added
- Added a configurable dictionary ingest flow with localized settings copy and model selection controls.

### Changed
- Refined rewrite answer card behavior and related recording overlay handling for rewrite and translation result flows.

### Fixed
- Persisted dictionary ingest model selection across launches and configuration export/import.
- Localized rewrite setting labels consistently across English, Simplified Chinese, and Japanese, and fixed settings window stability issues.

## [1.6.0] - 2026-03-16

### Added
- Added a dictionary workflow with scoped terms, history-based candidate suggestions, one-click ingestion, and prompt-time dictionary guidance.
- Added user main language selection plus engine hint settings for MLX and remote ASR providers, including provider-specific language handling for OpenAI, GLM, Doubao, and Aliyun.
- Added menu bar microphone switching and a General setting that can mute other apps' media audio during recording after system audio capture permission is granted.

### Changed
- Expanded configuration export/import so it now covers dictionary data, voice end command settings, user main language, ASR hint settings, and the latest General settings additions.
- Improved settings organization and localization for the new dictionary, language, and ASR hint workflows across English, Simplified Chinese, and Japanese.

### Fixed
- Fixed custom hotkey recording so modifier-heavy shortcuts are captured more reliably and no longer leak into active global hotkey handling while recording.

## [1.5.1] - 2026-03-14

### Fixed
- Improved Hotkey settings shortcut capture so newly recorded shortcuts stay pending until the user explicitly confirms them.
- Prevented global hotkey handlers from firing while recording a shortcut in Hotkey settings.
- Improved modifier-only shortcut capture reliability, including better handling for repeated modifier changes during recording.

## [1.5.0] - 2026-03-13

### Added
- Added configurable voice end commands in Hotkey settings with presets for `over`, `end`, `完毕`, plus custom command text.
- Added automatic stop from spoken end commands when the command appears at the transcript tail and is followed by about 1 second of silence.
- Added feedback entry points in the About tab and status bar menu, both linked to the GitHub issue chooser.

### Changed
- Updated the About tab tagline to `Voice to Thought` with localized Simplified Chinese copy `思想之声`.
- Refactored voice end command handling into focused recording/session and settings components to reduce coupling across `AppDelegate` and settings views.
- Simplified remote ASR stop flows and session task cleanup by extracting reusable helpers for streaming shutdown and recording lifecycle control.

### Fixed
- Fixed trailing end-command matching so surrounding punctuation, including Asian punctuation such as `，。！？`, is ignored reliably.
- Fixed final transcription output so spoken control commands are stripped from committed text, including when the user manually stops after the command.
- Fixed remote ASR file-recording shutdown to release the microphone capture object immediately after stop instead of holding it until upload/transcription completes.

## [1.4.8] - 2026-03-11

### Added
- Added configuration export/import in General settings for app preferences, models, app branch rules, and hotkeys.
- Added model setup warning badges after configuration import to guide users to incomplete provider or model setup.

### Fixed
- Fixed Sparkle no-update results so "already up to date" no longer appears as update check failure in settings.
- Fixed app branch configuration export so group and URL entries are serialized from their stored data payloads.

## [1.4.7] - 2026-03-10

### Added
- Added localized prompt template variable chips with copy interaction and hover tips in settings.
- Added app branch prompt templating support with `{{RAW_TRANSCRIPTION}}`.
- Added LLM debug log output for prompt input and model output content.
- Added shortcut preset support for `fn` and right-side `Command` combinations.
- Added optional left/right modifier distinction for shortcuts, including recording and display support.

### Changed
- Updated app branch prompt delivery so matched branch prompts can be sent as direct user messages.
- Refined prompt variable help UI with system popover tooltip behavior and improved hover persistence.
- Improved shortcut settings UI with preset selection and left/right modifier controls.

### Fixed
- Fixed remote realtime ASR start/stop races that could desync hotkey state and recording UI.
- Fixed proxy-disabled networking so WebSocket traffic no longer relies on legacy proxy behavior alone.
- Fixed accessibility permission prompting/registration flow so installed apps register more reliably in macOS Accessibility settings.
- Fixed hotkey matching so right-side modifier shortcuts no longer trigger from left-side keys when left/right distinction is enabled.
- Fixed app branch prompt handling and prompt editor guidance to align with current enhancement behavior.

## [1.4.2] - 2026-03-09

### Added
- Added prompt template variable support for enhancement and translation:
  - Enhancement: `{{RAW_TRANSCRIPTION}}`
  - Translation: `{{TARGET_LANGUAGE}}`, `{{SOURCE_TEXT}}`
- Added prompt variable hints below prompt textareas in model settings (localized in English, Simplified Chinese, and Japanese).
- Added OpenAI ASR chunk pseudo-realtime preview option (default off) with explicit usage-cost hint.

### Changed
- Updated default enhancement prompt to the new structured instruction template with strict output constraints.
- Updated default translation prompt to the new structured template with explicit target/source variable blocks and strict translation rules.
- Improved recording overlay waveform visibility and interaction feedback (stronger amplitude response, higher dynamic range, clearer bar rendering).

### Fixed
- Fixed OpenAI ASR preview text rendering/parsing in overlay to avoid JSON-like raw payload display artifacts.
- Fixed update check/install UX to reduce disruptive failure popups and surface status via settings sidebar badge with detail action.

### Refactored
- Reduced settings-layer coupling by extracting remote provider configuration sheet from `ModelSettingsView`.
- Moved remote provider connectivity test logic to `RemoteProviderConnectivityTester` (support layer).
- Moved remote provider model/endpoint selection policy to `RemoteProviderConfigurationPolicy` (support layer).
- Simplified update state handling and notification flow in `AppUpdateManager`.

## [1.4.1] - 2026-03-09

### Fixed
- Fixed Sparkle update installer configuration and release checks to prevent package install launch failures.

## [1.4.0] - 2026-03-08

- Release v1.4.0.

## [1.3.11] - 2026-03-06

### Fixed
- Fixed Sparkle package installer launch failures (e.g. `code 4005`) by expanding Sparkle entitlements placeholders in the release signing step so `Installer.xpc` can launch with the correct bundle identifier.

## [1.3.10] - 2026-03-06

- Release v1.3.10.

## [1.3.3] - 2026-03-05

### Fixed
- Fixed the About page log export action in menu-bar/dockless contexts by using a window-attached save sheet when possible and adding explicit export status feedback.

## [1.3.2] - 2026-03-05

### Fixed
- Fixed update version comparison by aligning app `CURRENT_PROJECT_VERSION` with Sparkle `sparkle:version`, preventing `1.3.1` from repeatedly showing `1.3.1 (1003001)` as an available update.
- Added detailed Sparkle update lifecycle logs (check source, found/not found, download success/failure, cycle completion, and abort details) to support in-app log export troubleshooting.

## [1.3.1] - 2026-03-05

### Changed
- Refactored app startup and runtime logic into focused `AppDelegate` extensions for better maintainability:
  - `AppDelegate+MenuWindow`
  - `AppDelegate+PreferencesAndHistory`
  - `AppDelegate+EnhancementPrompt`
  - `AppDelegate+RecordingSession`
- Extracted shared settings/domain types and reusable UI components to reduce file size and duplication.

### Fixed
- Sparkle update channel selection now defaults to stable feed unless `VOXT_UPDATE_CHANNEL=beta` is explicitly set, avoiding accidental use of test beta appcast entries that can trigger EdDSA security warnings.

## [1.3.0-beta.1] - 2026-03-04

### Added
- App Branch source card now shows an Apps-tab drag hint in the header.
- Added an experimental custom LLM model path for early `Qwen3.5` evaluation.

### Changed
- Upgraded `mlx-swift-lm` to a newer revision for early `Qwen3.5` model-type support.
- Improved App Branch localization coverage for tab content and related sheets.

### Fixed
- Fixed custom LLM download cancellation UI state not resetting reliably.
- Fixed custom LLM large-file progress display by aligning in-flight progress logic with MLX model download behavior.
- Fixed App Branch language switching inconsistency when switching to English.

## [1.1.8] - 2026-03-02

### Added
- Release v1.1.8.


## [1.1.7] - 2026-03-02

### Added
- Release v1.1.7.

## [1.1.5] - 2026-03-01

### Fixed
- Added a close action for the in-app update dialog so users can dismiss it.

## [1.1.4] - 2026-03-01

### Added
- Persistent application logs with local file storage.
- About page Logs section with last update time and export of latest 2000 entries.

### Changed
- Localized the new Logs export/status copy in English, Japanese, and Simplified Chinese.

### Fixed
- Updated app sandbox user-selected file access to read/write so save panel can be shown.

## [1.1.3] - 2026-03-01

### Added
- Release v1.1.3.


## [1.1.2] - 2026-03-01

### Added
- Release v1.1.2.


### Added
- Test release based on local Voxt.app archive package.
- In-app update checks with menu entry and optional automatic check at launch.
- Release scripts for generating `.app.zip`, `.pkg`, and update manifest.

## [1.1.1] - 2026-03-01

### Added
- Test release based on local Voxt.app archive package.

## [1.1.6] - 2026-03-01

### Added
- Added the ability to skip a specific update version in update checks.

### Fixed
- Added microphone permission checks before starting dictation recording.
- Fixed hotkey event callback ownership handling in the event tap callback.
- Fixed update installer download handling by staging the package before completion callback returns.
