# From Voxt 1.x to Voxt 2

Voxt 2 replaces the Swift Voxt 1.x at `/Applications/Voxt.app`. To macOS they are two apps: Voxt 2's bundle id is `dev.hehehai.voxtnew`, Voxt 1.x's is `com.voxt.Voxt`. Both are signed by the same team, RY2TQTYSKQ. Replacing the app leaves Voxt 1.x's data where it was, in its sandbox container `~/Library/Containers/com.voxt.Voxt`. Voxt 2 imports from there and never writes to it.

## What a user does

1. **Quit Voxt 1.x** from its menu bar icon. If it does not quit, use Force Quit (⌥⌘⎋). Finder cannot replace an app that is running: it says Voxt is in use.
2. **Drag Voxt from the disk image to Applications.** Finder asks what to do with the Voxt already there:
   - **Replace** replaces the app only. Voxt 1.x's data stays, and reinstalling Voxt 1.x brings it back as it was.
   - **Keep Both** installs Voxt 2 as "Voxt 2". Both apps run side by side; their default shortcuts do not overlap (Voxt 1.x: fn, ⇧fn, ⌃fn, ⌥fn; Voxt 2: right Option, right Option with Shift, Control with right Option).
3. **Open Voxt.** The setup guide's first step starts with a card, "The previous Voxt is on this Mac". Click **Import from the previous Voxt**.

   ![The setup guide's card](screenshots/migration-1-guide-card.png)

   - When macOS lets Voxt read Voxt 1.x's data, which it should for the signed release since both apps share a team, the import runs at once.
   - Otherwise an open panel opens on Voxt 1.x's data folder. Click **Import** without changing the folder: that click is the consent macOS asks for, and no other prompt follows.

   ![The open panel's message](screenshots/migration-2-panel.png)
   ![The open panel's buttons](screenshots/migration-2-panel-buttons.png)

4. **Read the result** on the card: how many terms are new or updated, how many history entries and which models came over. The button becomes **Import again**, which adds only what is new.

   ![After the import](screenshots/migration-3-imported.png)
   ![The dictionary after the import](screenshots/migration-4-dictionary.png)

5. **Grant the microphone and Accessibility again** in the setup guide's Permissions step. macOS keeps permissions per app, and Voxt 2 is a different app.
6. **Set up a language model** on the Language models page for clean-up, Translate and Rewrite. Voxt 2 no longer runs language models itself, so Voxt 1.x's local language models do not come over: Ollama, LM Studio or oMLX keep the text on this Mac, or pick an online service.

### If you set up 2.0.0 first

Voxt 2.0.0 had no import, and its setup guide does not open again once it is done or skipped. When the previous Voxt's data is on the Mac and the guide will not open, Voxt shows one card in the sidebar; its **Import** runs the import. The card does not come back once dismissed. The import is also in the dictionary's ••• menu, as **Import from the previous Voxt…**.

![The card for who set up 2.0.0](screenshots/migration-5-upgrade-card.png)

The setup guide's card in Simplified Chinese:

![The card in Simplified Chinese](screenshots/migration-zh-guide-card.png)

### Another Mac

On the old Mac, export the dictionary from Voxt 1.x: Dictionary → More → Export. The file is `Voxt-Dictionary.json` (`Voxt-词典.json` in Chinese). On the new Mac, open Voxt's Dictionary → ••• → **Import…** and pick the file. The export holds the terms and their replacement terms only: no history, no models, and no disabled state, so disabled terms come in enabled.

Voxt 2.0.0 reads this file as plain text and adds each line as a term. With 2.0.0, do not import it; if you did, delete the terms that look like JSON (`{`, `"version": 3`, …).

### Going back

Reinstall Voxt 1.16.3 from its release. Its dictionary, history and settings are where Voxt 1.x left them.

## What comes over

| Voxt 1.x (`voxt.sqlite`) | Voxt 2 |
|---|---|
| Dictionary terms | Terms. A term longer than 80 characters, or empty, is left out and counted in the result. |
| Replacement terms | Aliases ("heard as"): the first 10, in order. |
| Disabled terms | Disabled terms. |
| History: dictation, translation, rewrite | Transcribe, Translate and Rewrite entries, with the app, the models, the time and the length. Their ids are `legacy-<old id>`, so importing again adds nothing. |
| History: meeting transcripts | Left out: Voxt 2 has no meetings. |
| Speech models in `model-storage/<owner>_<name>` | All eleven of Voxt 1.16's local models are in Voxt 2's catalogue: Qwen3-ASR 0.6B (4bit), 1.7B (6bit) and 1.7B (8bit), Whisper Large v3 Turbo, Large v3 and Small, Parakeet TDT 0.6B v3, SenseVoice Small, Cohere Transcribe 03-2026, Nemotron 3.5 ASR 0.6B and MOSS-Transcribe-Diarize. Each is hard-linked into the model library when every file matches the version Voxt pins (size, and SHA-256 for the large files), a Whisper model with the tokenizer files Voxt 1.x kept beside it; otherwise left alone, and Voxt downloads it as usual. Voxt 1.x downloaded each repo's latest files, so a repo changed since then downloads again. Who imported with Voxt 2.0.x gets the models Voxt 2 added at its next start, without importing again (each model is looked for once). |
| Language models in `model-storage` | Left alone: Voxt no longer runs language models on this Mac (`legacyReusable` skips retired models). |
| Language model services (Settings → Model, in its preferences) | Setups on the Language models page, with their model, address and generation settings. Only a setup's own service's Voxt 1.x default address becomes the preset's: an OpenAI setup pointed at StepFun keeps StepFun's address. Another address keeps its host and loses its request path (`…/v1/chat/completions` → `…/v1`); an Ollama native address becomes its OpenAI-compatible `…/v1`. Voxt 1.x spoke Anthropic's, Gemini's and MiniMax's own APIs, which Voxt 2 does not: their custom addresses are kept (Gemini's `/v1beta/models` becomes `/v1beta/openai` on the same host) and the import's result asks to check them. An address with no path gets the path Voxt 1.x added for that service (`/v1`, `/api/v3` for Volcengine, `/compatible-mode/v1` for Aliyun Bailian, `/v1beta/openai` for Gemini), and the result asks to check it; a query string is dropped, to check. A `http://` address on another host is kept, but Voxt 2 only connects over https, or http to this Mac: the result's to-do names that rule. No address is replaced by the preset's, which would send requests, and the key pasted later, elsewhere. For OpenAI, only what Voxt 1.x sent comes over: the output limit, response format and Extra Body, and the reasoning effort and verbosity for the models it sent them to; temperature, top_p, top_k, min_p, stop, penalties, logprobs, seed and Ollama's options stay behind. Extra Body (written for Voxt 1.x's Responses request) and the models OpenAI serves only through Responses (`gpt-5.2-pro`, `gpt-5-pro`) come over, to check. A saved thinking budget above Anthropic's output limit raises the limit, as Voxt 1.x did. A service already set up in Voxt 2 keeps its setup. |
| Their API keys | Not imported: Voxt 1.x kept them in the Data Protection Keychain under its own keychain group (`RY2TQTYSKQ.com.voxt.Voxt`), which only an app signed with that group in a provisioning profile can read. The import's result asks to paste them again (not when a key for that service is already in Voxt's Keychain), also for an Ollama, oMLX or LM Studio that had a key saved, though those work without one here. |
| Speech services (Settings → Model, remote speech recognition) | Setups on the Speech models page, for all seven services Voxt 1.x had (OpenAI Transcribe, Doubao, Zhipu GLM-ASR, Aliyun Bailian, StepFun, Xiaomi MiMo, Google Gemini). A model Voxt 2's client of the service has comes over, any model for OpenAI-compatible servers; another becomes the service's default, and the result asks to check the setup. Only the service's own Voxt 1.x defaults become its address here; any other address stays, also one the setup's model cannot use (a wss:// address for a model that uploads, an https:// one for a streaming model), which the result asks to check. Doubao's flash file recognition (`volc.bigasr.auc_turbo`) comes over as a Doubao model. StepFun's SSE address becomes its transcriptions on the same host, to check. Their keys, and Doubao's App ID, which Voxt 1.x kept in the Keychain too, are pasted again. The services' own options (Aliyun's VAD, Doubao's dictionary modes, OpenAI's pseudo-realtime chunks) are not imported. |
| The language model and the speech service each feature used | The feature's model, when it was one of the setups that came over and the feature has none chosen in Voxt 2. A local speech model Voxt 1.x used (`mlx:<repo>`, or `whisper:<id>` from before) becomes the feature's model when it came over and the feature has none chosen; a model Voxt 1.x had retired maps as Voxt 1.x mapped it (other Parakeet models to v3, Whisper Tiny and Base to Small, other Qwen3-ASR 1.7B to 6bit, the rest to Qwen3-ASR 0.6B). System dictation is not carried over: the feature stays on Automatic. |
| Transcription, translation and rewrite prompts | The feature's Your rules, when the user wrote or changed them and the feature has none in Voxt 2; a built-in prompt (the previous Voxt saved a preset's text when one was picked: `legacyBuiltInPrompts` knows them by digest) stays behind, and `{{RAW_TRANSCRIPTION}}`, `{{SOURCE_TEXT}}` and `{{DICTATED_PROMPT}}` go (the input is never in the rules): a line that held one closes up its spacing, and a line left with only its label goes. |
| Transcribe's clean-up switch (`llmEnabled`) | Off in Voxt 1.x leaves Transcribe's clean-up off, when Transcribe has no language model chosen in Voxt 2. Only the first import applies it (the document `legacy-enhance-settled`), so **Import again** does not undo a change made since. Translate and Rewrite have no such switch. |
| Codex setups, and features that used Codex | Left out: the sign-in does not carry over. The result's "Left out" group names Codex and the features that used it, and points to signing in to ChatGPT (Codex) on the Language models page. |
| Dictionary terms scoped to an app group | Global terms: Voxt 2 has no per-group terms. |
| Hosted web search (`searchEnabled`) | Not carried over: Voxt 2 sends no hosted web search tool. |
| The Qwen3-ASR context | The speech context (Speech models page), sent to the speech model before the dictionary's terms, when the user wrote one and Voxt 2 has none; `{{DICTIONARY_TERMS}}` goes. |
| Shortcuts, other settings, app enhancement groups, notes | Not imported. |

Terms merge by name, ignoring case: an existing term gains the aliases it lacks. An import that stops halfway, for example when Voxt is force-quit, leaves the database consistent; importing again adds what is missing.

## How it works

`legacy_import.go` holds the import; `legacy_import_test.go` tests it on databases Voxt 1.x wrote (`testdata/legacy`).

- **Finding Voxt 1.x.** `legacyLibrary` checks that the container directory exists. Checking a container's existence needs no permission; reading inside it does. The card and the menu item show only when it exists.
- **Reading without asking.** `legacyReadable` opens Voxt 1.x's `voxt.sqlite` and reads one byte. macOS lets apps signed by the same team read each other's containers, so the signed release should get through and import at once. An ad hoc build is refused with EPERM and no prompt, and falls back to the open panel. Reading a byte, not only finding the file, matters: a container can let a file be found and not opened.
- **The open panel.** It opens on the container's `Data/Library` (`DefaultPath`), and the folder the user picks is readable for the rest of the run. Any of the three levels down to `Application Support/Voxt` works (`legacyDatabase`).
- **The database** is read read-only together with its `-wal` and `-shm`: Voxt 1.x's rows can live only in the WAL, and opening the main file alone does not see them.
- **Settings.** `legacy_settings.go` reads Voxt 1.x's preferences (`Library/Preferences/com.voxt.Voxt.plist`, binary) with `plutil` (`legacyPreference`): `remoteLLMProviderConfigurations`, `remoteASRProviderConfigurations`, `featureSettings` and `mlxLocalASRTuningSettings`, each a JSON document in a string. `parseLegacySettings` turns them into setups, prompts and the speech context, and `apply` fills only what Voxt 2 lacks. When the preferences are there but `plutil -lint` cannot read them (`legacyPreferencesReadable`), the result's "Left out" group says so, instead of the import looking as if Voxt 1.x had no setups.
- **Models.** `reuseLegacyModels` (`legacy_models.go`) looks for each catalogue model that `legacyReusable` allows (not borrowed, not retired) under Voxt 1.x's `model-storage`, checks every file, hard-links them into the library and claims them (`Manager.Use`). A Whisper model needs its tokenizer (OpenAI's `whisper-large-v3` or `whisper-small` files, an `Aux` of the catalogue in its own repo): it comes from the same folder, and without it the model stays. The import runs before the settings, so that a feature's local speech model is known to be installed (`legacyLocalSpeech`). At each start, `legacyModelsAtStart` does the same for catalogue models it has not looked for yet (the document `legacy-models-checked`), when Voxt 1.x's data reads without asking.
- **The cards.** The setup guide's welcome step shows its card on a fresh install. `offerLegacy` posts the sidebar card at start when the guide will not open, once: the document `legacy-import-offered` records that it was posted.
- **Tests and fake homes.** `VOXTNEW_LEGACY_HOME` replaces the home folder the container is looked for in, for tests that run the app with a fake `HOME`.

## Telling Voxt 1.x users

Voxt 1.x shows notices from `https://voxt.actnow.dev/api/app/notifications` (the `voxt-service` repo) in Settings → Notifications, as plain text. A notice is the way to reach users who never visit GitHub: tell them to quit Voxt 1.x before installing Voxt 2, and that Voxt 2 imports their dictionary, history and settings (their language models' keys excepted).
