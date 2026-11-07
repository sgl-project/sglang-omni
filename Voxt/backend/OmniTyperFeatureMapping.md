# OmniTyper to Voxt feature mapping

This document lists each OmniTyper feature and what Voxt has for it, for roadmap issue sgl-project/sglang-omni#2537. A later PR removes `OmniTyper/`. A feature that is not in this document is lost at that time.

## Summary

OmniTyper has 106 features. 62 need no work: Voxt has 51 of them, and 11 exist only to support OmniTyper itself. The other 44 need a decision from the maintainers. The proposal is to fill 19 in Voxt and drop 25.

| Section | Rows | Meaning |
| --- | --- | --- |
| [Fill](#fill) | 19 | Voxt has part of the feature or none of it. Add it to Voxt. |
| [Drop](#drop) | 25 | Voxt has part of the feature or none of it. Do not add it. |
| [Equivalent](#equivalent) | 51 | Voxt has the feature. |
| [Obsolete](#obsolete) | 11 | The feature exists only to support OmniTyper itself. |

- Effort of a fill: S is less than 1 day in 1 or 2 files. M is 1 to 3 days in 3 or more files. L is more than 3 days or crosses components.
- An ID that starts with `PR-` comes from an unmerged OmniTyper PR. [Sources](#sources) lists these PRs.
- Each section ends with a collapsed evidence table: the OmniTyper and Voxt source lines, and the searches for features that Voxt does not have.

## Fill

| Feature | OmniTyper | Voxt now | Fill | Effort |
| --- | --- | --- | --- | --- |
| Voice edit (MOD-3) | The user speaks an instruction. OmniTyper replaces the selected text at once. | Shows the result in an answer card. The user must click Inject. | If the user selected text, replace it at once. Keep the card as a setting. | S |
| Warning when cleanup fails (MOD-5) | Inserts the raw transcript and shows a warning. | Inserts the raw transcript. Only the log records the failure. | Show the same warning. | S |
| Pinned model revision (ASR-3) | Downloads one fixed revision of Qwen3-ASR-0.6B-4bit. | Downloads the latest revision (main). | Pin the revision for each Qwen3-ASR checkpoint that runs on the Omni backend. | S |
| Remote sglang-omni server (ASR-11) | Runs a local server only. It rejects other hosts. | The OpenAI Whisper provider can call a remote sglang-omni server. It requires an API key, which sglang-omni does not use. | Make the API key optional for custom endpoints. See [Appendix](#appendix-remote-sglang-omni-server). | S |
| Cut-off cleanup is a failure (LLM-6) | Treats a cut-off LLM answer (finish_reason length) as a failure. | Rejects an incomplete Responses API answer when it does not stream. The chat-completions path and the Responses stream do not check for a cut-off answer. | Treat a cut-off answer in the chat-completions path and the Responses stream as a failure, so it does not reach the cursor. | S |
| Dictionary CSV import and export (DIC-3) | Imports and exports the dictionary as CSV. | Imports and exports JSON only. | Add CSV. This also moves an OmniTyper dictionary into Voxt. | S |
| Casual, formal and concise styles (STY-1) | Has 5 style presets: verbatim, clean, casual, formal and concise. | Has Precise Cleanup and Clear Structure. Enhancement off gives verbatim. | Add Casual, Formal and Concise presets. | S |
| Full clipboard restore (INS-1) | Restores the whole clipboard after it pastes. | Restores text only. It loses images, files and rich text on the clipboard. | Save and restore all clipboard item types. | S |
| Hide results from clipboard managers (INS-2) | Marks its paste as transient and auto-generated, so clipboard managers skip it. | Clipboard managers record every result. | Add the same markers. | S |
| Block secure fields (INS-3) | Does not paste into password fields or during secure input. | No check found. | Check for secure input before start and before paste. | S |
| Stale Accessibility grant (PRM-2) | Detects that a rebuild broke the Accessibility grant and shows the fix. | No detection. Each dev rebuild loses the grant without a message. | Remember the earlier grant. When it is gone, show the remove-and-add-again steps. | S |
| Signing identity for dev builds (DEV-3) | Signs with a chosen identity, so macOS keeps the permission grants after a rebuild. | The Omni dev build signs ad hoc. Each rebuild loses the grants. | Use a local signing identity when the developer sets one. | S |
| Stable signing identity (PR-2242-A) | Same as DEV-3, with sign.sh and test-signing.sh. | Same as DEV-3. | Same fill as DEV-3. | S |
| F1–F20 key names (PR-2250-B) | Shows names for F1–F20 and Fn combinations. | Supports Fn. Shows no names for F1–F20. | Add the F1–F20 names. | S |
| Keep the model loaded (PR-2250-C) | Keeps the speech model loaded until the user unloads it. | Unloads the model after an idle delay of 1200 s or less. | Add a Never option to the idle unload delay. | S |
| Original transcript in history (HIS-3) | Keeps the transcript before LLM cleanup next to the final text. | Keeps the final text only. | Store the original transcript. Show it in the history detail. | M |
| Transcribe again (HIS-7) | Transcribes saved audio again, also for the last session. | No retry from history. | Add Transcribe Again to history entries that have audio. | M |
| Dictionary hints in the live preview (PR-2434) | Sends dictionary terms with the realtime session. | Sends dictionary terms in the final pass only. The live preview gets no hints. | Add the hints to the Voxt realtime session and the native realtime server. | M |
| Omni backend in a normal build (ASR-2) | Runs its speech server in the normal app. | Runs the Omni backend only in the dev build, with 2 environment variables. | Bundle the runtime binary in the app and find it without environment variables. Show the Omni engine in model settings. | L |

<details>
<summary>Evidence and where to change (19 rows)</summary>

| ID | OmniTyper source | Voxt source and notes | Where to change |
| --- | --- | --- | --- |
| MOD-3 | `AppModel.swift:177`<br>`AppModel.swift:310`<br>`backend/text_api.py:23` | `App/SessionTextIO.swift:316-321`<br>`Windows/WaveformAnswerCard.swift:128`<br>Voxt Rewrite always shows an answer card. The user must click Inject into Current Input. OmniTyper replaces the selection directly. | In SessionTextIO.shouldPresentRewriteAnswerOverlay, return false when selected text exists. Add a Settings toggle for alwaysShowRewriteAnswerCard. |
| MOD-5 | `backend/worker.py:283-293` | `App/TranscriptionFlow.swift:95`<br>Voxt falls back to raw text. Voxt only logs the failure. The user sees no warning.<br>`App/Recording/RecordingOverlayFlow.swift:29` | In the TranscriptionFlow.swift:95 fallback branch, call showOverlayReminder (App/Recording/RecordingOverlayFlow.swift:29). |
| ASR-3 | `backend/server.py:24-26` | `Transcription/OmniASRBackend.swift:26-30`<br>`Transcription/MLXModelDownloadSupport.swift:174`<br>`Transcription/MLXModelDownloadSupport.swift:243`<br>`sglang_omni_mlx/native/ci/models.json`<br>Voxt runs Qwen3-ASR 0.6B 4-bit, 1.7B 6-bit and 1.7B 8-bit on the Omni backend. Voxt lists files from tree/main and downloads resolve/main. OmniTyper pins revision 313d850181767edf09f00a9c289becca70e58cd0. The Mac CI pins the same 0.6B revision and one revision for each 1.7B checkpoint in models.json. | Pin the revisions for the 3 Omni repos in MLXModelDownloadSupport.swift:174 and MLXModelDownloadSupport.swift:243. Take them from models.json. |
| ASR-11 | `ASRStream.swift:52`<br>`backend/server.py:65-145` | `Transcription/RemoteASR/RemoteASRFileRequests.swift:9-13`<br>`Transcription/RemoteASR/RemoteASRTextSupport.swift:49-67`<br>`Core/RemoteProviders/RemoteEndpointSecurityPolicy.swift:45-47`<br>`Transcription/RemoteASR/RemoteASRTranscriber.swift:264`<br>`Voxt/docs/RemoteModel.md:7-35`<br>OmniTyper is local only. Voxt OpenAI Whisper provider can call a remote /v1/audio/transcriptions. It needs a non-empty API key. Plain HTTP with a key works only on loopback. No /v1/realtime streaming. | Make the API key optional for custom endpoints in RemoteASRFileRequests.swift:11-13. Do the same in RemoteProviderConfiguration.isConfigured (Core/RemoteProviders/RemoteProviderConfiguration.swift:132-148), which requires an API key or access token. isConfigured gates model selection (Settings/Features/FeatureModelCatalogBuilder.swift:320). sglang-omni inference endpoints need no key. |
| LLM-6 | `backend/text_api.py:161-199` | `Core/Models/RemoteModelConfiguration.swift:364-370`<br>`Core/LLM/RemoteLLMResponsesExecution.swift:161-170`<br>`Core/LLM/RemoteLLMStreamingParser.swift:496-506`<br>`Core/LLM/RemoteLLMRuntimePolicy.swift:93-103`<br>`Core/LLM/LLMVisibleOutputSanitizer.swift:78-83`<br>openAI, codex, volcengine and aliyunBailian use the Responses API. Without streaming, Voxt rejects a Responses status other than completed. The Responses stream does not check the status. Codex always streams. The other providers use the chat-completions path. It does not check finish_reason. Voxt strips <think>. No check for tool calls found in Core/LLM. | Treat finish_reason length as a failure in the chat-completions path (Core/LLM/RemoteLLMCompletionExecution.swift). Check the response status in the Responses stream (RemoteLLMResponsesExecution.swift). A cut-off cleanup must not reach the cursor. |
| DIC-3 | `Store.swift:307`<br>`Store.swift:320`<br>`LibraryViews.swift:98` | `Core/Dictionary/DictionaryTransferManager.swift:9`<br>`Core/Dictionary/DictionaryTransferManager.swift:110`<br>`Core/Dictionary/DictionaryTransferManager.swift:136`<br>Voxt imports and exports JSON only. | Add CSV to DictionaryTransferManager. This also moves an exported OmniTyper dictionary into Voxt. |
| STY-1 | `backend/text_api.py:26-32`<br>`Views.swift:15` | `Core/FeaturePromptPreset.swift:128-140`<br>Voxt has Precise Cleanup and Clear Structure. Verbatim equals enhancement off. No casual, formal or concise preset. | Add Casual, Formal and Concise presets in FeaturePromptPreset.swift. |
| INS-1 | `TextInsertion.swift:134-179` | `App/TextOutputDelivery.swift:54-113`<br>`App/TextOutputDelivery.swift:215`<br>`Core/Utilities/PasteboardTextWriter.swift:18-54`<br>Voxt restores only the previous text. Images, files and rich text on the clipboard are lost. | Snapshot and restore all pasteboard item types in PasteboardTextWriter. |
| INS-2 | `TextInsertion.swift:157`<br>`TextInsertion.swift:212`<br>`TextInsertion.swift:215-216` | None found<br>OmniTyper sets org.nspasteboard.TransientType and org.nspasteboard.AutoGeneratedType. Searched org.nspasteboard, TransientType and AutoGeneratedType in Voxt. No match. Clipboard managers record every Voxt result. | Set org.nspasteboard.TransientType and org.nspasteboard.AutoGeneratedType in PasteboardTextWriter. |
| INS-3 | `TextInsertion.swift:85`<br>`TextInsertion.swift:246`<br>`AppModel.swift:166` | None found<br>Searched SecureEventInput, IsSecureEventInputEnabled, AXSecureTextField and AXProtectedContent. No match. | Check IsSecureEventInputEnabled() and the AXSecureTextField subrole before start and before paste in TextOutputDelivery. |
| PRM-2 | `AppModel.swift:126-144`<br>`Store.swift:57-95` | None found<br>Searched stale, tccutil and re-add guidance in permission code. No match. Config/OmniDev.xcconfig:18 signs ad hoc, so each rebuild loses the grant. | Store a was-trusted flag in AccessibilityPermissionManager. Show remove-and-re-add guidance when the flag is set and trust is gone. |
| DEV-3 | `scripts/build.sh:7-34` | `Voxt/Config/OmniDev.xcconfig:18`<br>`Voxt/Config/Signing.local.xcconfig.example:5-6`<br>OmniTyper reads $CODE_SIGN_IDENTITY. OmniDev.xcconfig forces ad-hoc signing (CODE_SIGN_IDENTITY = -). | Let OmniDev.xcconfig take VOXT_CODE_SIGN_IDENTITY from Signing.local.xcconfig when it exists. |
| PR-2242-A | sgl-project/sglang-omni#2242 | `Voxt/Config/OmniDev.xcconfig:18`<br>Same as DEV-3. | Same fill as DEV-3. |
| PR-2250-B | sgl-project/sglang-omni#2250 | `Hotkey/HotkeySupport.swift:430`<br>`Hotkey/HotkeyPreferencePresentation.swift:44-53`<br>Voxt supports Fn. The key name table has no F1-F20 entries. | Add F1-F20 names in HotkeyPreferencePresentation.swift. |
| PR-2250-C | sgl-project/sglang-omni#2250 | `Settings/Shell/AppPreferenceKey.swift:131-133`<br>The Voxt idle unload delay has a 1200 s maximum.<br>A reload of Qwen3-ASR 0.6B 4-bit is short on the native runtime. On an Apple M5 Pro with warm file caches, it reported ready in 0.15-0.21 s. The first Final of a 4.6 s clip then took 0.10 s. The 1.7B checkpoints were not measured. | Add a Never option to the idle unload delay. |
| HIS-3 | `LibraryViews.swift:59`<br>`Views.swift:234-249` | `Core/History/TranscriptionHistoryModels.swift:71-111`<br>Voxt stores the final text and dictionary correction snapshots. It does not store the transcript before LLM cleanup. | Add an optional raw transcript field to TranscriptionHistoryEntry. Show it in the history detail sheet. |
| HIS-7 | `AppModel.swift:347`<br>`AppModel.swift:356`<br>`LibraryViews.swift:53`<br>`Views.swift:51-56` | None found<br>Searched retranscribe, retryTranscription, transcribeHistoryAudio and reprocessHistory. No history retry found. | Add Transcribe Again to history entries with audio. Reuse the final-pass path in MLXTranscriber. |
| PR-2434 | sgl-project/sglang-omni#2434 | `Transcription/OmniRealtimeTranscriptionSession.swift:212-218`<br>`sglang_omni_mlx/native/src/realtime.cpp:224-230`<br>Voxt sends language only in session.update. The native realtime server reads no prompt. The final pass has hints (ASR-8). | Port the server part of this PR from Python to realtime.cpp. Add prompt to Voxt session.update. |
| ASR-2 | `scripts/setup.sh:7-14`<br>`scripts/build.sh:7-34`<br>`PreferencesView.swift:66`<br>`WorkerClient.swift:314` | `Transcription/OmniASRBackend.swift:37-44`<br>`Transcription/OmniVoiceActivity.swift:9`<br>`Transcription/OmniSpeakerDiarization.swift:9`<br>`Voxt/backend/run_omni_dev.sh`<br>Voxt enables Omni only in the Voxt Omni Dev build. It needs VOXT_ASR_BACKEND=omni and VOXT_OMNI_RUNTIME. The same variables also move Silero VAD and Sortformer speaker analysis to the runtime. | Bundle qwen3_asr_server, libmlx.dylib and mlx.metallib in the app and resolve them without environment variables. Show the Omni engine in model settings. |

</details>

## Drop

| Feature | OmniTyper | Voxt now | Why drop |
| --- | --- | --- | --- |
| Ask mode (MOD-4) | Answers a spoken question in a card. Inserts nothing. | Rewrite with no selection shows the same card. | Rewrite covers Ask. The Continue button adds follow-up turns. |
| Stop and Cancel buttons on the panel (AUD-6) | Shows the elapsed time and Stop and Cancel buttons. | No buttons and no timer on the panel. | The hotkey stops a session, and Esc cancels it. |
| Prepare and Unload buttons (ASR-5) | Loads and unloads the model with buttons. | Loads the model at session start. Unloads it after an idle delay. | Automatic load and unload replace the buttons. PR-2250-C covers keep-loaded. |
| LLM model list (LLM-2) | Gets the model list from the server. | Has preset lists, a custom model ID and a Test button. | A custom model ID and the Test button cover setup. |
| API key in memory only (LLM-3) | Keeps the LLM API key in memory. The user enters it again at each launch. | Stores the key in the Keychain. The dev build may not keep it. | The Keychain is the macOS store for secrets. |
| Endpoint hardening (LLM-5) | Checks the URL, blocks redirects and limits a response to 1 MiB. | Checks the URL. Blocks a key over plain HTTP to a remote host. | The key rule covers the main risk. The other checks are general hardening, not a migration item. |
| Learn a word from a correction (DIC-4) | Adds a word when the user corrects a history entry. | Learns from edits after insert and from history scans. | Automatic learning and manual entry cover it. |
| Direct text write in native apps (INS-4) | Writes text through Accessibility in native apps. | Always pastes with Cmd+V. | Paste works in all apps. After INS-1 and INS-2, paste leaves the clipboard as it was. |
| Same-field check (INS-5) | Checks that the same text field still has focus before it inserts. | Brings back the app only. | Voxt pastes at the current focus by design. History keeps every result. |
| Ignored-paste fallback (INS-7) | Detects a paste that the app ignored and falls back. | No detection. | History keeps the result, and the paste hotkey pastes it again. The detection is a heuristic. |
| Accessibility in Chromium and Electron apps (INS-8) | Turns on Accessibility in Chromium and Electron apps to find the text field. | Does not turn it on. | Voxt does not need the field to paste. Selection reads have other fallbacks. |
| Copy only (INS-9) | Can copy the result without a paste. | Copies only when no text field has focus. | No one asked for copy-only output. The no-field case already copies. |
| History export as JSON (HIS-4) | Exports history as JSON. | Exports audio only. | No migration step needs it. Meeting export and audio export cover the main data. |
| Import OmniTyper data (HIS-10) | Stores history in library.json and an Audio folder. | No import. | OmniTyper is a development demo. DIC-3 moves the dictionary. The old files stay on disk. |
| Open the data folder (HIS-12) | Opens its data folder in Finder. | Has an Open folder button for the history audio folder. No button opens the history or dictionary data folder. | The Logs sheet exports logs. The user chooses the audio folder. |
| Start and stop without a hotkey (APP-2) | Starts, stops and cancels from the menu or the Home button. | The menu has no start, stop or cancel item. | Hotkeys start every Voxt session. Onboarding teaches the hotkey. |
| Appearance setting (APP-3) | Has light, dark and system appearance. | Follows the system appearance only. | The macOS appearance setting covers it. |
| macOS 14 (APP-7) | Runs on macOS 14. | Requires macOS 15. | Voxt upstream targets macOS 15. macOS 14 support is an app-wide port. |
| Hugging Face endpoint setting (PR-2241) | Lets the user set a Hugging Face endpoint or use hf-mirror.com. | Tries huggingface.co and hf-mirror.com and uses the one that answers. | The automatic check covers the mirror case. |
| Optional Hugging Face endpoint (PR-2394) | Same as PR-2241. It fixes sgl-project/sglang-omni#2239. | Same as PR-2241. | The automatic check covers sgl-project/sglang-omni#2239. |
| Hotkey conflict messages (PR-2242-C) | Reports conflicts with system and registered hotkeys. | Checks a fixed list of system shortcuts. | The fixed list covers the common system shortcuts. |
| ModelScope download (PR-2242-D) | Downloads the model from ModelScope. | No ModelScope source. | The hf-mirror.com check covers access from mainland China. |
| Pick a mode by hold and move (PR-2251-A) | The user holds the hotkey and moves to pick a mode. | Has one hotkey for each mode. | Separate hotkeys go to each mode directly. |
| Review panel (PR-2251-B) | Shows a panel with pause, resume, Insert, Copy and Clear. | The answer card has Inject and Copy. Only Meeting Notes can pause. | The answer card covers review. Meeting Notes covers long captures with pause. |
| Voice edit of the whole text box (PR-2251-C) | Edits the whole text box when the user selects nothing. | Rewrite with no selection writes new text. | Select all before Rewrite gives the same result. |

<details>
<summary>Evidence (25 rows)</summary>

| ID | OmniTyper source | Voxt source and notes |
| --- | --- | --- |
| MOD-4 | `AppModel.swift:310`<br>`backend/text_api.py:24` | `Voxt/docs/Rewrite.md`<br>`Windows/WaveformAnswerCard.swift:124`<br>`Windows/WaveformAnswerCard.swift:151`<br>Rewrite without a selection gives the same answer card. With a selection, Voxt rewrites the selection. OmniTyper uses the selection as question context. |
| AUD-6 | `Views.swift:282` | None found<br>No button or elapsed-time view found in Windows/RecordingOverlay*.swift or Windows/Waveform*.swift (searched Button(, elapsed, duration). |
| ASR-5 | `PreferencesView.swift:61-62`<br>`AppModel.swift:372`<br>`AppModel.swift:416` | `App/Recording/RecordingCaptureFlow.swift:58`<br>`Settings/Shell/AppPreferenceKey.swift:131-133`<br>Voxt prewarms at session start and unloads after an idle delay of 10-1200 s. It has no manual buttons. |
| LLM-2 | `PreferencesView.swift:74-98`<br>`AppModel.swift:391`<br>`backend/worker.py:228` | `Core/RemoteProviders/RemoteConnectivityTester.swift:38`<br>Voxt uses preset model lists, a custom model ID and a Test button. No live GET /models list found. |
| LLM-3 | `AppModel.swift:24` | `Core/Security/VoxtSecureStorage.swift:143`<br>Voxt stores the key in the Keychain. The ad-hoc Omni Dev build may not keep it (Voxt/backend/README.md:87-88). |
| LLM-5 | `backend/text_api.py:97` | `Core/RemoteProviders/RemoteEndpointSecurityPolicy.swift:30-47`<br>Voxt validates the URL and blocks credentials over plain HTTP to remote hosts. No redirect block or response size limit found. |
| DIC-4 | `LibraryViews.swift:73`<br>`Store.swift:291` | `Core/Dictionary/DictionaryLearningMonitor.swift`<br>`Settings/Shell/AppPreferenceKey.swift:181`<br>Voxt learns from edits to delivered text and from history scans. It has no manual correction editor. |
| INS-4 | `TextInsertion.swift:123-131` | `App/TextOutputDelivery.swift:215`<br>Voxt always pastes with Cmd+V. |
| INS-5 | `TextInsertion.swift:26-44`<br>`TextInsertion.swift:97` | `App/TextOutputDelivery.swift:19-52`<br>`App/TextInjectionTransaction.swift`<br>Voxt re-activates the session app only. It does not check the field. |
| INS-7 | `TextInsertion.swift:183-196` | `App/TextOutputDelivery.swift:206`<br>Searched pasteWasIgnored. AXNumberOfCharacters appears only in App/TextInputIO.swift:254 for snapshots.<br>`TextOutputDelivery.swift:206` |
| INS-8 | `TextInsertion.swift:54-71` | None found<br>Searched AXManualAccessibility and AXEnhancedUserInterface. No match.<br>`App/SelectedTextProbe.swift:30` |
| INS-9 | `Store.swift:57-95`<br>`PreferencesView.swift:34` | None found<br>Searched paste and clipboard keys in AppPreferenceKey.swift. Only autoCopyWhenNoFocusedInput and the custom paste hotkey exist. |
| HIS-4 | `LibraryViews.swift:26`<br>`LibraryViews.swift:226` | None found<br>Searched exportJSON, export.*History and NSSavePanel in Core/History and Settings/History. Only audio export exists. |
| HIS-10 | `Store.swift:148` | None found<br>Searched OmniTyper and library.json in Voxt/. No match. |
| HIS-12 | `PreferencesView.swift:104-115` | `Settings/History/HistoryAudioSettingsSheet.swift:84-86`<br>`Core/History/HistoryAudioDirectoryManager.swift:54-56`<br>`AppPreferenceKey.swift:128`<br>The Open folder button opens the history audio folder with activateFileViewerSelecting. No button opens the history or dictionary data folder. Searched activateFileViewerSelecting, selectFile, Show in Finder and Open Folder. The other matches open model folders. |
| APP-2 | `OmniTyperApp.swift:46-73`<br>`Views.swift:198-204` | `App/MenuWindowCoordinator.swift:93-158`<br>The Voxt menu has Dashboard, History, Dictionary, Microphone and Quit. It has no start, stop or cancel item. |
| APP-3 | `Views.swift:73`<br>`PreferencesView.swift:123` | `Settings/Shell/SettingsUIStyles.swift:211`<br>Voxt follows the system appearance only. |
| APP-7 | `Package.swift`<br>`Resources/Info.plist` | `Voxt/Voxt.xcodeproj/project.pbxproj:475`<br>Voxt requires macOS 15.0. OmniTyper requires macOS 14.0. |
| PR-2241 | sgl-project/sglang-omni#2241 | `Transcription/MLXModelManager.swift:1089-1101`<br>Voxt probes huggingface.co and hf-mirror.com and picks a reachable source. No custom endpoint field. |
| PR-2394 | sgl-project/sglang-omni#2394 | `Transcription/MLXModelManager.swift:1089-1101`<br>Same as PR-2241. It fixes sgl-project/sglang-omni#2239. |
| PR-2242-C | sgl-project/sglang-omni#2242 | `Settings/HotkeySettingsValidation.swift:14-24`<br>`Hotkey/HotkeyRecorderView.swift:184`<br>Voxt checks a static list of system shortcuts. No query of registered hotkeys found. |
| PR-2242-D | sgl-project/sglang-omni#2242 | None found<br>Searched modelscope and ModelScope in Voxt/. No match. |
| PR-2251-A | sgl-project/sglang-omni#2251 | None found<br>Voxt binds one hotkey to each mode. |
| PR-2251-B | sgl-project/sglang-omni#2251 | `Windows/WaveformAnswerCard.swift:128`<br>`Windows/WaveformAnswerCard.swift:151`<br>`Meeting/MeetingSessionModels.swift`<br>Voxt answer cards have Inject and Copy. Pause exists only in Meeting Notes. |
| PR-2251-C | sgl-project/sglang-omni#2251 | `Voxt/docs/Rewrite.md`<br>Voxt Rewrite without a selection writes new text from the prompt. |

</details>

## Equivalent

| Area | OmniTyper feature | Notes on Voxt |
| --- | --- | --- |
| Voice modes | Dictate: speech to text, optional cleanup, insert at cursor (MOD-1) | Voxt runs cleanup only when EnhancementMode is not off. Both dictate without a text API. |
|  | Translate into a target language (MOD-2) | Voxt also translates selected text with the translation hotkey. |
|  | Translate, edit and ask failures do not insert raw text (MOD-6) | Voxt shows Translation failed or Rewrite failed and commits no text. |
| Shortcuts | Custom global shortcut (default Ctrl+Option+Space) (KEY-1) | The Voxt default is fn. Voxt also accepts mouse buttons. |
|  | Toggle or hold-to-talk (KEY-2) | Voxt also has double-tap. |
|  | Lone modifier shortcuts such as Fn (KEY-3) | The Voxt default shortcut is the lone fn key. |
|  | Hold release from the physical key state (KEY-4) | Both read CGEventSource key state. |
|  | Re-enable the event tap after a timeout (KEY-5) | Both handle tapDisabledByTimeout. |
|  | Esc cancels the active session (KEY-6) | Voxt has a toggle. The default is on. |
|  | One shortcut runs the selected mode; mode picker on Home (KEY-7) | Voxt binds one hotkey to each mode: transcription, translation and rewrite. |
| Audio and panel | Microphone selection (AUD-1) | Voxt adds a priority list and auto switch. OmniTyper fails the session on a device change. |
|  | Input level meter (AUD-2) | Voxt draws a waveform. |
|  | Start and stop sounds (AUD-3) | Voxt has sound presets. |
|  | Non-activating floating panel with live text (AUD-5) | Both panels do not take focus. Voxt live text has an on/off toggle. |
|  | Silence gate returns no text (AUD-7) | Voxt uses local VAD. OmniTyper uses an energy threshold. |
|  | Live transcript preview over /v1/realtime (AUD-8) | Voxt Omni preview updates once per second. |
| Speech recognition | Local speech server on loopback with a random port (ASR-1) | OmniTyper runs sglang_omni.cli serve with SGLANG_USE_MLX=1. Voxt runs qwen3_asr_server, a C++ binary on MLX without Python. |
|  | Use the cached model without network access (ASR-4) | Voxt starts the Omni runtime from its installed model directory. |
|  | Cancel keeps the model loaded; Unload stops the server process group (ASR-6) | Voxt stops earlier runtimes before it starts a new one. |
|  | Readiness wait and preparation progress (ASR-7) | The runtime prints a ready event on stdout when it serves. Voxt waits for it. |
|  | Dictionary hints in the final transcription prompt (ASR-8) | OmniTyper sends the first 20 written forms. Voxt fills {{DICTIONARY_TERMS}}. |
|  | Speech language selection (ASR-9) | Voxt takes the language from the user main languages (userMainLanguageCodes). |
|  | Full-file final pass after streaming (ASR-10) | Both send the full WAV to /v1/audio/transcriptions. |
| Text API (LLM) | OpenAI-compatible text API, default Ollama (LLM-1) | Voxt has 17 LLM providers, Ollama included. |
|  | Request options JSON (LLM-4) | Voxt calls it Extra Body JSON. |
|  | Per-task prompt templates with escaping and examples (LLM-7) | Voxt prompts are user-editable. |
| Dictionary | Preferred spellings as recognition hints (DIC-1) | Voxt adds categories and match counts. |
|  | Literal replacements: longest first, case-insensitive, word boundary (DIC-2) | Voxt replacement terms always apply. |
|  | Entry validation and limits (DIC-5) | Limits differ. OmniTyper allows 200 entries of 120 characters. |
| Writing style | Global custom instructions (STY-2) | Voxt uses an editable prompt. |
|  | Per-app style rule (STY-3) | Voxt groups apps and URL patterns. |
| Text insertion | No editable field or no Accessibility: copy instead (INS-6) | Voxt uses autoCopyWhenNoFocusedInput. OmniTyper also writes a note into the history entry. |
| History and data | History list with search and mode filter (HIS-1) | Voxt filters by transcription, translation, rewrite, note and transcript. |
|  | Copy, delete one, delete all (HIS-2) | Voxt deletes all entries of one kind. |
|  | History on/off and retention (HIS-5) | Voxt has retention by period and by count. |
|  | Keep audio (off by default) and export WAV (HIS-6) | Voxt exports all archives at once. |
|  | Usage stats (HIS-8) | Voxt has a dashboard with time, characters and speed. |
|  | Local data store; a corrupt file is not overwritten (HIS-9) | Voxt keeps history and dictionary in repositories. File mode 0600 was not checked. |
| App shell | Menu bar app; closing the window keeps it running (APP-1) | Voxt has a Show in Dock toggle. |
|  | Interface language with runtime switch (APP-4) | Voxt adds Japanese. |
|  | Open at login (APP-5) | Both use SMAppService. |
| Permissions | Microphone and Accessibility checks with settings links (PRM-1) | Voxt adds an onboarding guide. |
|  | Bounded diagnostics log without user content (PRM-3) | Voxt rotates the log file, redacts secrets and keeps LLM content out of files. |
|  | Microphone usage text and audio-input entitlement (PRM-4) | Same feature. |
| Build and tests | Unit tests (Swift and Python) (DEV-6) | Voxt Mac CI runs the Voxt Omni unit tests and the runtime API tests on an Apple Silicon runner, for PRs with the run-ci label. |
| Unmerged PRs | Pause the shortcut while a new one is captured (PR-2242-B) | Voxt sets hotkeyCaptureInProgress. |
|  | Ask single-turn answer view (PR-2251-D) | Voxt adds Continue for follow-up turns. |
|  | Download percentage and model status (missing, downloaded, loaded) (PR-2256) | Voxt shows download progress and install state. |
|  | Compact waveform capsule popup (PR-2303) | The Voxt overlay is a waveform capsule. Live text off keeps it compact. Click to finish is not in Voxt. |
|  | Start and stop the audio engine off the main actor (PR-2304) | Voxt starts the engine in a detached task with a timeout. |
|  | Meeting notetaker: 3 h recording, chunked transcription, Markdown summary (PR-2523) | Voxt Meeting Notes runs speech recognition, Silero VAD and Sortformer speaker analysis on the Omni runtime. It has summaries and export. It labels Me and Them. Speaker analysis gives each speaker a numbered label, such as Speaker 1. |

<details>
<summary>Evidence (51 rows)</summary>

| ID | OmniTyper source | Voxt source and notes |
| --- | --- | --- |
| MOD-1 | `AppModel.swift:264-310`<br>`backend/text_api.py:20-32` | `App/TranscriptionFlow.swift:95`<br>`Settings/TranscriptionTypes.swift:51`<br>Voxt runs cleanup only when EnhancementMode is not off. Both dictate without a text API. |
| MOD-2 | `backend/text_api.py:22`<br>`Views.swift:212`<br>`PreferencesView.swift:45` | `Settings/Shell/AppPreferenceKey.swift:82-84`<br>`App/TranslationFlow.swift:124-130`<br>Voxt also translates selected text with the translation hotkey. |
| MOD-6 | `backend/worker.py:283-293` | `App/TranslationFlow.swift:124-130`<br>`App/TranslationFlow.swift:384-389`<br>`App/TranslationFlow.swift:428-434`<br>Voxt shows Translation failed or Rewrite failed and commits no text. |
| KEY-1 | `Store.swift:57-95`<br>`PreferencesView.swift:20-25`<br>`GlobalShortcut.swift` | `Hotkey/HotkeySupport.swift:144`<br>`Hotkey/HotkeySupport.swift:341-345`<br>The Voxt default is fn. Voxt also accepts mouse buttons. |
| KEY-2 | `PreferencesView.swift:26`<br>`AppModel.swift:100`<br>`GlobalShortcut.swift:160` | `Hotkey/HotkeySupport.swift:10-13`<br>`Hotkey/HotkeySupport.swift:115-131`<br>Voxt also has double-tap. |
| KEY-3 | `ShortcutCapture.swift:30`<br>`GlobalShortcut.swift:21` | `Hotkey/HotkeySupport.swift:341-345`<br>`Hotkey/HotkeySupport.swift:430`<br>The Voxt default shortcut is the lone fn key. |
| KEY-4 | `GlobalShortcut.swift:72` | `Hotkey/HotkeyManager.swift:120`<br>Both read CGEventSource key state. |
| KEY-5 | `GlobalShortcut.swift:118` | `Hotkey/HotkeyManager.swift:191`<br>Both handle tapDisabledByTimeout. |
| KEY-6 | `GlobalShortcut.swift:124`<br>`AppModel.swift:114` | `Settings/HotkeySettingsView.swift:141`<br>`App/HotkeyLifecycle.swift:235`<br>Voxt has a toggle. The default is on. |
| KEY-7 | `Views.swift:194`<br>`OmniTyperApp.swift:46-73` | `Settings/HotkeySettingsView.swift:590`<br>Voxt binds one hotkey to each mode: transcription, translation and rewrite. |
| AUD-1 | `AudioRecorder.swift:130`<br>`AudioRecorder.swift:160`<br>`PreferencesView.swift:29` | `Core/AudioInputDeviceManager.swift:17`<br>`Core/MicrophonePreferenceManager.swift:119`<br>`App/MenuWindowCoordinator.swift:121`<br>Voxt adds a priority list and auto switch. OmniTyper fails the session on a device change (AudioRecorder.swift:198). |
| AUD-2 | `AudioRecorder.swift:13-106`<br>`Views.swift:282` | `Core/AudioLevelMeter.swift`<br>`Windows/WaveformView.swift`<br>Voxt draws a waveform. |
| AUD-3 | `AppModel.swift:214`<br>`AppModel.swift:223`<br>`PreferencesView.swift:33` | `Core/InteractionSoundPlayer.swift:30`<br>`Core/InteractionSoundPlayer.swift:51`<br>`Settings/Shell/AppPreferenceKey.swift:74-75`<br>Voxt has sound presets. |
| AUD-5 | `OmniTyperApp.swift:94-107`<br>`Views.swift:282` | `Windows/RecordingOverlayWindow.swift:30-35`<br>`Settings/GeneralSettingsView.swift:22`<br>Both panels do not take focus. Voxt live text has an on/off toggle. |
| AUD-7 | `backend/worker.py:196` | `App/Recording/RecordingTextRouting.swift:81`<br>`Transcription/MLXTranscriber.swift:469`<br>`Meeting/Capture/MeetingVoiceActivity.swift:323-325`<br>Voxt uses local VAD. OmniTyper uses an energy threshold. With the Omni backend on, the Voxt Silero VAD runs on the runtime. |
| AUD-8 | `ASRStream.swift:5-122`<br>`Views.swift:224` | `Transcription/OmniRealtimeTranscriptionSession.swift:81`<br>`Transcription/MLXTranscriber.swift:1117`<br>Voxt Omni preview updates once per second (Voxt/backend/README.md). |
| ASR-1 | `backend/server.py:65-145` | `sglang_omni_mlx/native/src/server.cpp:302-303`<br>`sglang_omni_mlx/native/src/server.cpp:522-526`<br>`Transcription/OmniASRRuntime.swift:253-259`<br>OmniTyper runs sglang_omni.cli serve with SGLANG_USE_MLX=1. Voxt runs qwen3_asr_server, a C++ binary on MLX without Python. |
| ASR-4 | `backend/server.py:29-55` | `Transcription/MLXModelManager.swift:835-847`<br>Voxt starts the Omni runtime from its installed model directory. |
| ASR-6 | `AppModel.swift:418`<br>`backend/server.py:205-225` | `Transcription/MLXModelManager.swift:844`<br>`Transcription/OmniASRRuntime.swift:179`<br>Voxt stops earlier runtimes before it starts a new one. |
| ASR-7 | `backend/server.py:65-145`<br>`Views.swift:144`<br>`WorkerClient.swift:221` | `sglang_omni_mlx/native/src/server.cpp:543`<br>`Transcription/OmniASRRuntime.swift:318`<br>`Settings/Models/ModelDownloadStatusView.swift:12-40`<br>The runtime prints a ready event on stdout when it serves. Voxt waits for it. |
| ASR-8 | `backend/server.py:147-203` | `Transcription/OmniASRRuntime.swift:474`<br>`Core/Transcription/ASRHintLocalTuning.swift:486`<br>OmniTyper sends the first 20 written forms. Voxt fills {{DICTIONARY_TERMS}}. |
| ASR-9 | `PreferencesView.swift:41`<br>`ASRStream.swift:84` | `Core/Transcription/ASRHintResolver.swift:34`<br>`Transcription/MLXInferenceConfiguration.swift:50`<br>Voxt takes the language from the user main languages (userMainLanguageCodes). |
| ASR-10 | `backend/worker.py:254`<br>`AppModel.swift:278-290` | `Transcription/MLXTranscriber.swift:1692`<br>`Transcription/OmniASRRuntime.swift:221`<br>Both send the full WAV to /v1/audio/transcriptions. |
| LLM-1 | `Store.swift:26-55`<br>`backend/text_api.py:97` | `Core/Models/RemoteModelConfiguration.swift:343`<br>`Voxt/docs/RemoteModel.md`<br>Voxt has 17 LLM providers, Ollama included. |
| LLM-4 | `Store.swift:26-55`<br>`PreferencesView.swift:92-96` | `Settings/RemoteProviderSheetSections.swift:541`<br>Voxt calls it Extra Body JSON. |
| LLM-7 | `backend/text_api.py:19-53` | `Core/FeaturePromptPreset.swift:128-140`<br>`Resources/Prompts/en/en-enhancement.txt`<br>`Resources/Prompts/en/en-rewrite.txt`<br>Voxt prompts are user-editable. |
| DIC-1 | `Store.swift:97`<br>`backend/server.py:147-203` | `Core/Dictionary/DictionaryModels.swift:139`<br>Voxt adds categories and match counts. |
| DIC-2 | `backend/worker.py:201` | `Core/Dictionary/DictionaryMatchingSupport.swift:314`<br>`Core/Dictionary/DictionaryMatchingSupport.swift:385-394`<br>Voxt replacement terms always apply. |
| DIC-5 | `Store.swift:97`<br>`Store.swift:291` | `Core/Dictionary/EntryValidationSupport.swift`<br>Limits differ. OmniTyper allows 200 entries of 120 characters. |
| STY-2 | `LibraryViews.swift:161` | `Settings/Features/FeatureSettings.swift:270`<br>Voxt uses an editable prompt. |
| STY-3 | `LibraryViews.swift:161`<br>`AppModel.swift:238` | `Settings/AppEnhancement/AppEnhancementModels.swift:40-50`<br>Voxt groups apps and URL patterns. |
| INS-6 | `TextInsertion.swift:97`<br>`TextInsertion.swift:106-108`<br>`AppModel.swift:317` | `Settings/Shell/AppPreferenceKey.swift:111`<br>`App/TextOutputDelivery.swift:54-113`<br>Voxt uses autoCopyWhenNoFocusedInput. OmniTyper also writes a note into the history entry. |
| HIS-1 | `LibraryViews.swift:14` | `Settings/History/HistorySettingsView.swift:34`<br>`Settings/History/HistorySettingsView.swift:314`<br>`Settings/History/HistorySettingsComponents.swift:12-17`<br>Voxt filters by transcription, translation, rewrite, note and transcript. |
| HIS-2 | `LibraryViews.swift:26-27`<br>`LibraryViews.swift:49`<br>`LibraryViews.swift:56` | `Settings/History/HistorySettingsComponents.swift:117`<br>`Settings/History/HistorySettingsComponents.swift:139`<br>`Settings/History/HistorySettingsView.swift:684`<br>`Core/History/TranscriptionHistoryStore.swift:402`<br>Voxt deletes all entries of one kind. |
| HIS-5 | `Store.swift:265`<br>`PreferencesView.swift:104-115` | `Settings/Shell/AppPreferenceKey.swift:123-126`<br>Voxt has retention by period and by count. |
| HIS-6 | `Store.swift:217`<br>`LibraryViews.swift:54`<br>`LibraryViews.swift:226` | `Settings/Shell/AppPreferenceKey.swift:127`<br>`Core/History/HistoryAudioArchiveService.swift:17`<br>`Core/History/HistoryAudioArchiveSupport.swift:10`<br>Voxt exports all archives at once. |
| HIS-8 | `Views.swift:250-254` | `Settings/ReportSettingsView.swift:42-63`<br>Voxt has a dashboard with time, characters and speed. |
| HIS-9 | `Store.swift:148`<br>`Store.swift:185-189`<br>`Store.swift:205` | `Core/History/HistoryRepository.swift`<br>`Core/Dictionary/DictionaryRepository.swift`<br>Voxt keeps history and dictionary in repositories. File mode 0600 was not checked. |
| APP-1 | `Resources/Info.plist`<br>`OmniTyperApp.swift:19`<br>`OmniTyperApp.swift:89-90` | `Settings/Shell/AppPreferenceKey.swift:122`<br>`App/MenuWindowCoordinator.swift:93`<br>Voxt has a Show in Dock toggle. |
| APP-4 | `Localization.swift:17`<br>`PreferencesView.swift:47` | `Settings/Shell/SettingsTypes.swift:22-26`<br>`Core/Utilities/AppLocalization.swift:30`<br>Voxt adds Japanese. |
| APP-5 | `PreferencesView.swift:124` | `Core/AppBehaviorController.swift:42`<br>Both use SMAppService. |
| PRM-1 | `AppModel.swift:126-144`<br>`PreferencesView.swift:131-132`<br>`Views.swift:169-182` | `Settings/PermissionsSettingsView.swift`<br>`Settings/Onboarding/OnboardingGuidePermissions.swift`<br>`Core/Security/AccessibilityPermissionManager.swift`<br>Voxt adds an onboarding guide. |
| PRM-3 | `Diagnostics.swift:5-49` | `Core/Logging/VoxtLogFileStore.swift:12`<br>`Core/Logging/VoxtLog.swift:78-82`<br>`Settings/LogsViewerSheet.swift:85`<br>Voxt rotates the log file, redacts secrets and keeps LLM content out of files. |
| PRM-4 | `Resources/Info.plist:17`<br>`Resources/Entitlements.plist` | `Voxt/Voxt.xcodeproj/project.pbxproj:563`<br>`VoxtOmniDev.entitlements`<br>None. |
| DEV-6 | `Tests/OmniTyperTests`<br>`backend/test_worker.py`<br>`scripts/test.sh:13-14` | `Voxt/VoxtTests/OmniASRRuntimeLaunchTests.swift`<br>`Voxt/VoxtTests/OmniFailurePathTests.swift`<br>`Voxt/VoxtTests/OmniPhase1LifecycleTests.swift`<br>`Voxt/VoxtTests/OmniTranscriptionRequestTests.swift`<br>`sglang_omni_mlx/native/ci/test_server_api.py`<br>`.github/workflows/voxt-mac-ci.yaml:140`<br>Voxt Mac CI runs the Voxt Omni unit tests and the runtime API tests on an Apple Silicon runner, for PRs with the run-ci label. OmniPhase1LifecycleTests needs the model, is opt-in and is not in CI. Voxt/.github/workflows/tests.yml is nested, so GitHub does not run it. |
| PR-2242-B | sgl-project/sglang-omni#2242 | `Hotkey/HotkeyCaptureState.swift:13-30`<br>Voxt sets hotkeyCaptureInProgress. |
| PR-2251-D | sgl-project/sglang-omni#2251 | `Windows/WaveformAnswerCard.swift:124`<br>Voxt adds Continue for follow-up turns. |
| PR-2256 | sgl-project/sglang-omni#2256 | `Settings/Models/ModelDownloadStatusView.swift:12-40`<br>Voxt shows download progress and install state. |
| PR-2303 | sgl-project/sglang-omni#2303 | `Windows/WaveformView.swift`<br>`Settings/GeneralSettingsView.swift:22`<br>The Voxt overlay is a waveform capsule. Live text off keeps it compact. Click to finish is not in Voxt. |
| PR-2304 | sgl-project/sglang-omni#2304 | `Transcription/MLXTranscriber.swift:883-891`<br>Voxt starts the engine in a detached task with a timeout. |
| PR-2523 | sgl-project/sglang-omni#2523 | `Voxt/docs/Meeting.md`<br>`Transcription/MLXTranscriber.swift:975-979`<br>`Meeting/Capture/MeetingVoiceActivity.swift:323-325`<br>`Meeting/SpeakerAnalysis/MeetingSpeakerDiarizationEngines.swift:126-129`<br>`Meeting/SpeakerAnalysis/SpeakerDisplayNameFormatter.swift:35-41`<br>`Core/TranscriptSummarySupport.swift:213`<br>Voxt Meeting Notes runs speech recognition, Silero VAD and Sortformer speaker analysis on the Omni runtime. It has summaries and export. It labels Me and Them. Speaker analysis gives each speaker a numbered label, such as Speaker 1. |

</details>

## Obsolete

- 5-minute recording limit (AUD-4). The limit matched `--audio_chunking.max_total_audio_s 300` of the old `server.py`. Voxt does not use that server.
- Migrate from OpenTypeless (HIS-11). OpenTypeless is the earlier OmniTyper name. No Voxt user has OpenTypeless data.
- Launch arguments --background and --snapshot (APP-6). They are development helpers for the OmniTyper package.
- setup.sh (DEV-1). Voxt has its own runtime build script.
- build.sh (DEV-2). Voxt builds with Xcode.
- Python JSON-lines worker (DEV-4). Voxt calls the server over HTTP and runs text API calls in Swift.
- Runtime path setting (DEV-5). Voxt reads VOXT_OMNI_RUNTIME. ASR-2 covers the user need.
- Real-model smoke tests (DEV-7). They drive the OmniTyper worker protocol. Voxt has its own golden-output check and benchmark tests.
- OmniTyper README (DEV-8). Voxt/backend/README.md documents the Omni build.
- Swift tools 6.0 and icon type fix (PR-2240). The fix applies to the OmniTyper package only.
- Window polish (PR-2250-A). The changes apply to OmniTyper views only.

<details>
<summary>Evidence (11 rows)</summary>

| ID | OmniTyper source | Voxt source and notes |
| --- | --- | --- |
| AUD-4 | `AppModel.swift:80`<br>`backend/worker.py:168` | `Transcription/OmniASRRuntime.swift:79`<br>`sglang_omni_mlx/native/src/server.cpp:306`<br>Voxt has no dictation cap. A Final is split into energy-cut chunks of up to 1200 s. The live preview starts a new segment every 30 s (--max-segment-seconds). |
| HIS-11 | `Store.swift:170-176` | None found<br>OpenTypeless is the earlier OmniTyper name. |
| APP-6 | `OmniTyperApp.swift:38-43`<br>`OmniTyperApp.swift:109` | None found<br>--snapshot renders README screenshots. --background starts hidden. |
| DEV-1 | `scripts/setup.sh:7-14`<br>`backend/requirements.txt` | `sglang_omni_mlx/native/scripts/build_runtime.sh`<br>Voxt builds the runtime with CMake and Ninja from a uv venv pinned in build-tools.lock. |
| DEV-2 | `scripts/build.sh:7-34`<br>`scripts/icon.swift` | `Voxt/backend/run_omni_dev.sh`<br>`Voxt/tools/package_local_app.sh`<br>Voxt builds with Xcode. |
| DEV-4 | `backend/worker.py:306`<br>`WorkerClient.swift:47-314` | `Transcription/OmniASRRuntime.swift`<br>Voxt calls the server over HTTP. Text API calls run in Swift. |
| DEV-5 | `PreferencesView.swift:66`<br>`Store.swift:193-200`<br>`WorkerClient.swift:314` | `Transcription/OmniASRBackend.swift:37-44`<br>Voxt reads VOXT_OMNI_RUNTIME. |
| DEV-7 | `backend/smoke.py`<br>`backend/smoke_stream.py`<br>`Tests/realtime_server.py` | `Voxt/VoxtTests/OmniPhase1BenchmarkTests.swift`<br>`sglang_omni_mlx/native/ci/check_golden.py`<br>The smoke tests drive the OmniTyper worker protocol. |
| DEV-8 | `README.md:62-76` | `Voxt/backend/README.md`<br>The README describes OmniTyper only. |
| PR-2240 | sgl-project/sglang-omni#2240 | None found<br>Voxt builds with Xcode. |
| PR-2250-A | sgl-project/sglang-omni#2250 | `Settings/Shell/AppPreferenceKey.swift:77-80`<br>Voxt has its own UI. The overlay position is top or bottom and does not drag. |

</details>

## Appendix: remote sglang-omni server

OmniTyper cannot use a remote server. Voxt can, with three limits.

- OmniTyper starts a private local server. `ASRStream.swift:52` rejects any host other than 127.0.0.1.
- OmniTyper runs `sglang_omni.cli serve` with `SGLANG_USE_MLX=1`. It does not use the CUDA server.
- The Voxt `OpenAI Whisper` remote provider takes a full custom endpoint and a custom model ID (`Voxt/docs/RemoteModel.md:7-35`).
- It sends a 16 kHz mono WAV with `model`, `response_format=json`, `language` and `prompt` (`Transcription/RemoteASR/RemoteASRTextSupport.swift:49-67`, `Transcription/RemoteASR/RemoteASRFileRequests.swift:597`, `Transcription/RemoteASR/RemoteASRTranscriber.swift:619-625`).
- `sglang_omni/serve/speech_to_text.py:68-76` accepts these fields. So a sglang-omni `/v1/audio/transcriptions` endpoint works.

Limits:

1. Voxt requires a non-empty API key (`RemoteASRFileRequests.swift:11-13`). Without an API key or access token, the provider is not configured and cannot be selected (`RemoteProviderConfiguration.swift:132-148`, `FeatureModelCatalogBuilder.swift:320`). sglang-omni inference endpoints need no key. Enter any placeholder.
2. Voxt rejects plain HTTP with a key on a non-loopback host (`RemoteEndpointSecurityPolicy.swift:45-47`). Use HTTPS or an SSH tunnel to localhost.
3. Voxt does not stream to a remote `/v1/realtime`. The optional pseudo-realtime preview re-uploads WAV snapshots (`RemoteASRTranscriber.swift:264`).

The fill for ASR-11 makes the key optional. Then a LAN server over plain HTTP works without a tunnel.

## Sources

- OmniTyper source: last commit that changes `OmniTyper/` is `c4a0145472be6d5c1a566ccc0963fbf462a37bff` (sgl-project/sglang-omni#2519).
- Main branch for all refs: `db765fe495fdf71c7bd171c8359cfb88f21c6e45`. It includes the merge of sgl-project/sglang-omni#2577 (`a30e33eb`), and sgl-project/sglang-omni#2586, sgl-project/sglang-omni#2587 and sgl-project/sglang-omni#2588.
- Voxt import: hehehai/voxt at `baa6f316` (`Voxt/PROVENANCE.md`).
- No file outside `OmniTyper/` refers to OmniTyper (`git grep -i omnityper`).
- PR data: `gh pr view` and `gh pr diff` for each PR in this document.

Rows by source and status:

| Status | Main branch | Unmerged PRs | Total |
| --- | ---: | ---: | ---: |
| `equivalent` | 45 | 6 | 51 |
| `partial` or `gap` | 33 | 11 | 44 |
| `obsolete` | 9 | 2 | 11 |
| **Total** | **87** | **19** | **106** |

Unmerged OmniTyper PRs. All 11 were open on 2026-10-09. A PR with 2 or more features has one row for each feature. sgl-project/sglang-omni#2434 also changes server code under `sglang_omni/`.

| PR | Title | Rows |
| --- | --- | --- |
| sgl-project/sglang-omni#2240 | [Fix] OmniTyper: correct Swift requirements and icon types | PR-2240 |
| sgl-project/sglang-omni#2241 | [Feature] OmniTyper: configurable Hugging Face endpoint and mirror download UX | PR-2241 |
| sgl-project/sglang-omni#2242 | [Fix] OmniTyper: preserve permissions, report shortcut conflicts, add ModelScope | PR-2242-A, PR-2242-B, PR-2242-C, PR-2242-D |
| sgl-project/sglang-omni#2250 | [OmniTyper] Improve window controls and everyday usability | PR-2250-A, PR-2250-B, PR-2250-C |
| sgl-project/sglang-omni#2251 | [OmniTyper] Redesign mode selection and popup interaction | PR-2251-A, PR-2251-B, PR-2251-C, PR-2251-D |
| sgl-project/sglang-omni#2256 | [Feature] OmniTyper: show speech model download progress and status | PR-2256 |
| sgl-project/sglang-omni#2303 | [Feature] OmniTyper: optional compact waveform capsule as the recording popup | PR-2303 |
| sgl-project/sglang-omni#2304 | [Fix] OmniTyper: keep the popup responsive while the microphone starts and stops | PR-2304 |
| sgl-project/sglang-omni#2394 | feat(omnityper): support Hugging Face mirrors for ASR | PR-2394 |
| sgl-project/sglang-omni#2434 | [ASR] Forward vocabulary hints through realtime transcription | PR-2434 |
| sgl-project/sglang-omni#2523 | [Feature] OmniTyper: meeting notetaker | PR-2523 |

Merged OmniTyper PRs:

| PR | Title | Rows |
| --- | --- | --- |
| sgl-project/sglang-omni#2214 | [Feature] Add OmniTyper with MLX streaming ASR and configurable text APIs | All main rows |
| sgl-project/sglang-omni#2231 | [Fix] OmniTyper: remove the Settings main-thread stall, dictate without a text API, add Simplified Chinese | MOD-1, APP-4 |
| sgl-project/sglang-omni#2237 | [Fix] OmniTyper: record lone modifier shortcuts such as Fn | KEY-3 |
| sgl-project/sglang-omni#2255 | [Fix] OmniTyper: stop reporting successful terminal pastes as ignored | INS-7 |
| sgl-project/sglang-omni#2261 | [Fix] OmniTyper: withdraw the unkept copy promise, show preparation progress in Settings, point at where verbatim is set, stop Unload ASR orphaning the speech server | ASR-5, ASR-6, ASR-7, STY-1 |
| sgl-project/sglang-omni#2275 | [Deps] Bump SGLang to 0.5.20 | DEV-1 |
| sgl-project/sglang-omni#2505 | [Deps] Bump SGLang to 0.5.21 | DEV-1 |
| sgl-project/sglang-omni#2519 | [Fix] OmniTyper: limit Esc cancellation, keep the model loaded on cancel, clipboard history, notices | KEY-6, ASR-6, INS-2 |

Closed without merge: sgl-project/sglang-omni#2247 and sgl-project/sglang-omni#2249. sgl-project/sglang-omni#2250 replaces them.

Paths in the evidence tables:

- A bare `*.swift` name in the OmniTyper column is in `OmniTyper/Sources/OmniTyper/`.
- Other OmniTyper refs are relative to `OmniTyper/`.
- A Voxt ref that starts with `Voxt/` is relative to the repository root.
- Other Voxt refs are relative to `Voxt/Voxt/`.
- `sglang_omni/` and `sglang_omni_mlx/` refs are relative to the repository root.
