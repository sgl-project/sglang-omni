# Voxt's settings, for people and agents who edit them directly

Every setting on Voxt's pages is stored as data on this Mac. This page lists every one of them, where it is stored, which values and types it takes, and how to change one without breaking the app, so that an agent can change Voxt's configuration for you. The settings pages remain the safer way.

## Where things are

| What | Where |
|---|---|
| The settings (everything in the tables below) | One JSON document in the database: table `documents`, row `key = 'settings:app'`, column `doc` |
| App enhancement groups | The same table, row `key = 'appctx:rules'` (not covered here) |
| Dictionary terms | The table `dictionary_terms`, one set per profile in `dictionary_profiles` (use the Dictionary page's Import and Export) |
| API keys and the ChatGPT sign-in | The login Keychain, service `VoxtNew`, accounts `llm.<id>` and `asr.<id>` (`llm.codex` is the ChatGPT sign-in). Never in the settings |
| Interface language, download proxy and Hub endpoints, model folder, while Voxt follows the values it shares with other apps | `~/Library/Application Support/ActNow/settings.json` (or `$ACTNOW_FAMILY_DIR/settings.json`), see [Shared with other apps](#shared-with-other-apps) |

The database is `voxtnew.db` in the data folder, by default `~/Library/Application Support/VoxtNew/`. If the data folder was moved in Settings → Storage, `location.json` in that default folder names the folder in use in its `"dir"` field.

## Changing a setting

Voxt reads the document once when it starts, keeps it in memory, and writes the whole document back on every change it makes. A change made while Voxt runs is therefore not seen, and the next change from the app overwrites it. So:

1. **Quit Voxt** (menu bar icon → Quit Voxt, or ⌘Q). Closing the window is not enough while "Keep running when the window closes" is on.
2. **Find the database and check it is there.** `sqlite3` creates an empty database at a path that does not exist, so check first:

   ```sh
   cat "$HOME/Library/Application Support/VoxtNew/location.json" 2>/dev/null   # a "dir" here is the data folder in use
   DB="$HOME/Library/Application Support/VoxtNew/voxtnew.db"                   # or "<dir>/voxtnew.db"
   test -f "$DB" && echo found || echo "no database at $DB"
   ```

   If the database is not found, stop here. Do not run the following commands until `DB` points to the existing database.

3. **Back up the database**: copy `voxtnew.db`, with `voxtnew.db-wal` and `voxtnew.db-shm` if they exist.
4. **Change one field at a time** with SQLite's JSON functions, instead of replacing the document, and check each step:

   ```sh
   sqlite3 "$DB" "SELECT count(*) FROM documents WHERE key = 'settings:app';"          # must print 1
   sqlite3 "$DB" "UPDATE documents SET doc = json_set(doc, '$.overlay.position', 'top'), updated_at = CAST(strftime('%s','now') AS INTEGER) * 1000 WHERE key = 'settings:app'; SELECT changes();"   # must print 1
   sqlite3 "$DB" "SELECT json_valid(doc), json_type(doc, '$.overlay.position'), json_extract(doc, '$.overlay.position') FROM documents WHERE key = 'settings:app';"   # must print 1|text|top
   ```

   Write each value with the type its table gives: a string as a SQL string (`'top'`), a boolean as `json('true')`, an integer as `json('30')`, a number as `json('0.9')`, an array or object as `json('[...]')` or `json('{...}')`. `json_type` then reports `text`, `true`/`false`, `integer`, `real` (or `integer` for a whole number), `array` or `object`. `json_set` adds a field that is missing.
5. **Start Voxt** and check the page the setting is on.

### What Voxt does with what it reads

- **Valid JSON is not enough: every value must also have its expected type.** A known field with the wrong type, such as `"true"` (a string) for a boolean, `1.5` for an integer, or an object for a field that holds JSON as a string, causes Voxt to load the defaults for all settings; the next settings update persists those defaults, losing every setting, the language model setups and the finished setup guide. A document that is not valid JSON is treated the same way. Check `json_valid` and `json_type` before you start Voxt.
- A field it does not know is dropped at the next save.
- Write names exactly as in the tables (`keepHistory`). `json_set` with `keephistory` adds a second field beside `keepHistory`, and which of the two Voxt then reads is not something to rely on.
- A missing field takes its default.
- Values the tables mark as clamped, snapped or "else …" are put back in range at start.
- Most other values are not checked. A shortcut Voxt cannot parse, or the id of a service you did not set up, leaves that feature without a working shortcut or model until you fix it on its page.

### Rolling back

- Quit Voxt and copy the backup files back (delete `-wal` and `-shm` if the backup had none).
- Without a backup: Settings → Storage → Reset all settings. If Voxt does not start, quit it and delete the row (`DELETE FROM documents WHERE key = 'settings:app';`); the setup guide then runs again. Either way the language model setups are forgotten (their keys stay in the Keychain).

## Shared with other apps

Voxt shares four values with the other apps on this Mac that use its model library, in the `settings.json` listed above. `family.follow` says, per key, whether Voxt uses the shared value (a key that is missing follows). While it follows, the shared value wins and the field below is not used. Editing the shared file changes every app that follows it; this page does not cover its format.

| Shared key | Voxt's own field |
|---|---|
| `general.language` | `general.language` |
| `network.proxy` | `network.proxy` |
| `network.hfEndpoints` | `network.hfEndpoints` |
| `models.root` | none (the model folder) |

To make Voxt use its own value, set `family.follow."<key>"` to `false` as well, for example `json_set(doc, '$.family.follow."general.language"', json('false'), '$.general.language', 'zh-Hans')`.

## The settings

Paths are JSON paths in the document; "Default" is a fresh install's value. Types: `string`, `bool`, `integer` (a whole number; `1.5` is a type error), `number` (may have a fraction), arrays and objects as shown. "absent" means the field is left out of the document.

### Values that several fields use

- **Shortcut**: an object `{"keys": "...", "trigger": "..."}`, never a bare string. `keys` is modifiers and at most one key joined with `+`, for example `RightOption`, `RightOption+Shift`, `Control+RightOption`, `Control+Option+V`, `Fn+Shift`, `Mouse3`. Modifiers: `Control`, `Option`, `Shift`, `Command`, each optionally `Left…` or `Right…`, and `Fn` (no side). Keys: letters, digits, `F1`–`F20`, `Space`, `Return`, `Tab`, `Escape`, `Delete`, arrows, punctuation, `Mouse3`–`Mouse32`. `trigger` is `tap` (the default; anything else becomes `tap`), `double` or `long`. Changing keys here does not check them against other apps' shortcuts; the Shortcuts page does.
- **Language**: a language code such as `en`, `zh`, `zh-Hans`, `zh-Hant`, `ja`.
- **Speech model** (`asr`): a model id from the Speech models page, or the id of a speech service you set up; `""` uses the recommended installed model.
- **Language model** (`llm`): the `id` of an entry of `providers.llm` (`openai`, `codex`, `ollama`, `omlx`, `deepseek`, …); `""` is Automatic, the first one set up. `features.translate.fallback` takes the same ids, but there `""` means no fallback.

### General (Settings → General)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `general.language` | string | `""` | `""` (follow the system), `en`, `zh-Hans` | App language |
| `general.mainLanguage` | string | `""` | a language code; `""` = from the app language | Main language |
| `general.otherLanguages` | array of strings | `[]` | language codes | Other languages |
| `general.copyResult` | bool | `false` | | Also copy the result to the clipboard |
| `general.menuBar` | bool | `true` | | Show in the menu bar |
| `general.hideDock` | bool | `false` | `true` hides Voxt from the Dock | Show in the Dock (inverted) |
| `general.closeToTray` | bool | `true` | | Keep running when the window closes |
| `general.launchAtLogin` | bool | `false` | | Open at login |
| `features.hidden` | array of strings | `[]` | `translate`, `rewrite`, `app-enhancement` | Shown features (inverted). A hidden feature's shortcuts are off |

### Appearance (Settings → Appearance)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `appearance.theme` | string | `system` | `system`, `light`, `dark`, `transparent` (else `system`) | Theme |
| `appearance.palette` | string | `#0d9384` | `#rrggbb` (else the default) | Theme |
| `appearance.windowTransparency` | integer | `8` | 5–15 percent (clamped), used by `transparent` | Theme |
| `appearance.scale` | number | `1` | `0.9`, `1`, `1.1`, `1.25` (snapped to the nearest) | Interface size |
| `appearance.reduceMotion` | bool | `false` | | Reduce motion |
| `appearance.rail` | bool | `false` | `true` shows the sidebar as icons only | Sidebar |
| `appearance.railChosen` | bool | `false` | the user chose names and icons | Sidebar |
| `appearance.sidebarWidth` | number | `0` | 0 (default) or 170–320 (clamped) | Dragging the sidebar's edge |
| `appearance.closedSections` | array of strings | `[]` | sidebar sections the user closed | The sidebar's section headers |

### Privacy (Settings → Privacy)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `privacy.keepHistory` | bool | `true` | | Keep a history |
| `privacy.keepAudio` | bool | `false` | | Keep recordings |
| `privacy.logText` | bool | `false` | | Write text to the log |
| `privacy.sendWindow` | bool | `false` | | Send a screenshot |
| `privacy.sendClipboardImage` | bool | `false` | | Send a copied image |
| `privacy.historyAccess` | bool | `false` | | Let programs on this Mac read the history ([history-access.md](history-access.md)) |
| `privacy.historySelection` | bool | `false` | | Also include the text you selected |

### History (History → History settings)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `history.cleanupOff` | bool | `false` | `true` keeps every entry | Clean up automatically (inverted) |
| `history.keepDays` | integer | `0` (= 90) | `365`, `180`, `90`, `30`, `7`; any other = 90 | Keep entries for |
| `history.keepCount` | integer | `0` (no limit) | `0`, `50`, `100`, `300`, `500`, `1000`; any other = no limit | Keep at most |
| `history.audioDir` | string | `""` | a folder; `""` = `audio/` in the data folder | Storage location |

### Audio (Settings → Audio)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `audio.devices` | array of strings | `[]` | capture device ids, preferred first; `[]` = the system's input | Microphone list (also the menu bar) |
| `audio.deviceNames` | object, string → string | `{}` | device id → name, for `audio.devices` only | (kept by Voxt) |
| `audio.muteOutput` | bool or absent | absent (= on) | | Mute system audio while recording |
| `audio.sounds` | bool | `true` | | Sounds |
| `audio.startSound`, `audio.stopSound` | string | `""` | `""` (default), `none`, `system:<macOS sound name>`, a built-in sound's id | Sounds |
| `audio.soundVolume` | integer | `0` (= 70) | 1–100 percent; negative = muted | Sounds |
| `audio.liveResult` | bool | `false` | `true` keeps the live text instead of recognising the recording again | Recognise the whole recording again (inverted) |
| `audio.vad.off` | bool | `false` | | Skip silence (inverted) |
| `audio.clean.off` | bool | `false` | | Optimise sound (inverted) |
| `audio.uploadMB` | integer | `0` | MB per upload to a speech service that takes whole files; 0 = the service's limit | Upload size limit |
| `audio.maxSeconds` | integer | `0` | seconds; 0 = no limit | Recording length limit (shown in minutes) |

**Silence detection and sound clean-up, for troubleshooting only.** These fields have no UI and are absent from a fresh install's document. Leave them absent: Voxt then uses the effective default in the table. It also uses the effective default for `0` or a value outside the range; a few values are capped instead, as noted.

| Path | Type | Effective default | Range | Meaning |
|---|---|---|---|---|
| `audio.vad.onsetDB` | number | `10` | over 0; above 60 becomes 60 | dB above the noise floor where speech starts |
| `audio.vad.offsetDB` | number | `6` | over 0; above `onsetDB` becomes `onsetDB` | dB above the noise floor where speech continues |
| `audio.vad.minLevelDB` | number | `-60` | under 0; below -120 becomes -120 | dBFS under which a frame is never speech |
| `audio.vad.hangoverMS` | integer | `300` | over 0 | ms of quiet before speech ends |
| `audio.vad.leadMS` | integer | `300` | over 0 | ms kept before the first speech |
| `audio.vad.tailMS` | integer | `300` | over 0 | ms kept after the last speech |
| `audio.vad.longGapMS` | integer | `1000` | over 0 | a pause longer than this many ms is shortened… |
| `audio.vad.gapKeepMS` | integer | `400` | over 0; above `longGapMS` becomes `longGapMS` | …to this many ms |
| `audio.vad.preRollMS` | integer | `300` | over 0 | ms sent to a streaming speech model from before the speech starts |
| `audio.vad.maxHoldMS` | integer | `5000` | over 0 | a stream starts anyway after this many ms without speech |
| `audio.clean.highPassOff` | bool | `false` | | turns the low-cut filter off |
| `audio.clean.levelOff` | bool | `false` | | turns the level stage off |
| `audio.clean.highPassHz` | number | `80` | over 0, at most 1000 | the low-cut filter's corner, Hz |
| `audio.clean.targetDB` | number | `-23` | -120 to under 0 | dBFS a quiet voice is raised to… |
| `audio.clean.maxGainDB` | number | `12` | over 0, at most 40 | …by at most this many dB |
| `audio.clean.loudDB` | number | `-12` | -120 to under 0; below `targetDB` becomes `targetDB` | dBFS above which a loud voice is lowered… |
| `audio.clean.maxCutDB` | number | `6` | over 0, at most 40 | …by at most this many dB |
| `audio.clean.noiseCeilDB` | number | `-50` | -120 to under 0 | dBFS the noise floor is never raised above |
| `audio.clean.ceilingDB` | number | `-1` | -120 to under 0 | dBFS no sample exceeds |

### Overlay (Settings → Overlay style)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `overlay.position` | string | `bottom` | `bottom`, `top` (else `bottom`) | Position |
| `overlay.margin` | number | `96` | 0–400 points (clamped) | Distance from the edge |
| `overlay.opacity` | number | `0.8` | 0.5–1 (clamped) | Opacity |
| `overlay.radius` | number | `24` | 0–40 points (clamped) | Corner radius |
| `overlay.showTime` | bool | `false` | | Show recording time |
| `overlay.hideLiveText` | bool | `false` | | Show live text (inverted) |
| `overlay.liveLines` | bool | `true` | `false` = one line that scrolls | Live text layout |

### Shortcuts (each feature's page)

| Path | Type | Default | On the page |
|---|---|---|---|
| `features.transcribe.hotkeys` | array of shortcuts | `[{"keys": "RightOption", "trigger": "tap"}]` | Transcribe → Shortcut |
| `features.translate.hotkeys` | array of shortcuts | `[{"keys": "RightOption+Shift", "trigger": "tap"}]` | Translate → Shortcut |
| `features.rewrite.hotkeys` | array of shortcuts | `[{"keys": "Control+RightOption", "trigger": "tap"}]` | Rewrite → Shortcut |
| `features.meeting.hotkeys` | array of shortcuts | absent (none) | Meetings |
| `features.pasteLast` | shortcut | `{"keys": "Control+Option+V", "trigger": "tap"}` | Transcribe → Paste the last result |
| `features.cancel` | shortcut | `{"keys": "Escape", "trigger": "tap"}` (tap only; `"keys": ""` = off) | Transcribe → Cancel recording |
| `features.note` | shortcut | `{"keys": "RightCommand", "trigger": "tap"}` (tap only; `"keys": ""` = off) | Notes |
| `features.strictShortcuts` | bool | `false` | Settings → Shortcuts → Strict shortcuts: only the starting feature's shortcut stops a recording |

Settings → Shortcuts → presets set the four main shortcuts at once; no field stores the preset, Voxt recognises it from the keys.

### Transcribe, Translate and Rewrite (each feature's page)

Each of `features.transcribe`, `features.translate` and `features.rewrite` has:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `asr` | string | `""` | Speech model (see above) |
| `llm` | string | `""` | Language model (see above) |
| `prompt` | string | `""` | Your rules: added to the built-in prompt |
| `showCard` | bool | `false`; Rewrite `true` | Show the result in a card instead of inserting it |

And their own fields:

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `features.transcribe.enhance` | bool | `true` | | Clean up with a language model |
| `features.transcribe.cleanup` | bool | `true` | | Basic clean-up |
| `features.transcribe.structure` | bool | `false` | | Structure |
| `features.endWord` | string | `""` | a word that ends the recording; `""` = off (streaming speech models only) | Transcribe → End with a word |
| `features.translate.target` | string | `""` | a language code; `""` = English, or Chinese when the main language is English | Target language |
| `features.translate.cleanup` | bool | `true` | | Basic clean-up |
| `features.translate.fallback` | string | `""` | a language model id; `""` = none | Fallback model |
| `features.rewrite.cleanup` | bool | `true` | | Clean up the instruction |
| `features.cardSizes` | object, string → integer | `{}` | `transcribe`/`translate`/`rewrite` → `0` small, `1` medium, `2` large, `3` fill (the available work-area width) | The answer card's size button |

### Meetings (Meetings → settings)

`features.meeting` has `hotkeys`, `asr`, `llm` and `prompt` as above, and:

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `features.meeting.mode` | string | `""` | `""` (meeting), `subtitles`, `recording` | Capture mode |
| `features.meeting.lang` | string | `""` | a language code; `""` = the main language | Spoken language |
| `features.meeting.hideFromSharing` | bool | `false` | | Hide from screen sharing |
| `features.meeting.collapsed` | bool | `false` | | The overlay's compact layout |
| `features.meeting.translateTarget` | string | `""` (= `en`) | a language code | (no settings row) |
| `features.meeting.speakers` | bool | `false` | | (no settings row) |
| `features.meeting.noFinalPass` | bool | `false` | `true` skips re-recognising the meeting afterwards | (no settings row) |
| `features.meeting.noAutoSummary` | bool | `false` | `true` skips the summary when a meeting ends | (no settings row) |

### Dictionary (Dictionary page)

| Path | Type | Default | Meaning |
|---|---|---|---|
| `dictionary.enabled` | bool | `true` | Use the dictionary at all |
| `dictionary.asrHints` | bool | `true` | Send the terms to the speech model as hints |
| `dictionary.glossary` | bool | `true` | Glossary for the language model |
| `dictionary.speechContext` | string | `""` | Free text about the speech (who speaks, about what) for the speech model |
| `dictionary.notes` | string | `""` | The personal dictionary: Markdown sent to the language model of the three features |
| `dictionary.notesOff` | bool | `false` | Keep the personal dictionary out of the prompt |
| `dictionary.learn` | bool | `false` | Learn from my corrections (suggest terms) |

### App enhancement (App enhancement page)

| Path | Type | Default | Meaning |
|---|---|---|---|
| `appEnhancement.enabled` | bool | `true` | Use App enhancement |
| `appEnhancement.browsers` | array of `{"bundleID": string, "name": string}` | `[]` | Browsers added to the built-in ones, whose tab address Voxt reads |

### Speech models and language models

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `models.keepLoaded` | string | `""` | `""` (Automatic), `5m`, `15m`, `30m`, `1h`, `quit` | Speech models → Keep models loaded |
| `models.hidden` | array of strings | `[]` | speech model ids | Speech models → Hide |
| `models.faster` | object, string → bool | `{}` | speech model id → on; absent = on while the draft is installed | Speech models → Faster transcription |
| `models.llmWait` | string | `""` | `""` (Automatic), `1m`, `2m`, `5m`, `10m`, `30m`, `60m`, `none` (no limit); another `<n>m` for 1–60 also works | Language models → Wait for the language model |
| `providers.llm` | array of providers | `[]` | one entry per language model service | Language models (one card per service set up) |
| `providers.asr` | array of providers | `[]` | one entry per speech service | Speech models → online services |

A provider is an object. Set the key on the page; an entry without one fails its connection test.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | required | The service (`openai`, `codex`, `ollama`, `omlx`, `doubao`, …), one entry per `id` |
| `endpoint` | string | the service's | API root (`https://…/v1`, or `http://localhost:…` for a server on this Mac); for a speech service its URL |
| `model` | string | the service's | Model name |
| `apiFormat` | string | `""` | `""` chat completions, `responses` the Responses API (OpenAI and custom only) |
| `appID` | string | `""` | Doubao's App ID (speech, older console) |
| `recognitionPrompt` | string | `""` | A speech service's recognition hint, with `{{USER_MAIN_LANGUAGE}}`, `{{USER_OTHER_LANGUAGES}}`, `{{DICTIONARY_TERMS}}` |
| `generation` | object | absent | A language model's request settings, below |

In `generation`, an absent field leaves Voxt's or the service's default. A field marked "or absent" distinguishes absent from `0`: `0` is sent as `0`. Voxt does not check the numbers: it sends them as they are, the service decides which values it accepts, and a value it rejects can make requests fail. `jsonSchema` and `extraBodyJSON` that are not a JSON object are left out of the request without an error (the page checks them only when you save there).

| Field | Type | Meaning |
|---|---|---|
| `maxOutputTokens` | integer or absent | Most tokens in an answer |
| `temperature` | number or absent | Sampling temperature |
| `topP` | number or absent | Top P |
| `topK` | integer or absent | Top K |
| `minP` | number or absent | Min P |
| `seed` | integer or absent | Random seed |
| `stop` | array of strings | Stop sequences |
| `presencePenalty` | number or absent | Presence penalty |
| `frequencyPenalty` | number or absent | Frequency penalty |
| `repetitionPenalty` | number or absent | Repetition penalty |
| `logprobs` | bool | Ask for log probabilities |
| `topLogprobs` | integer or absent | How many top log probabilities |
| `responseFormat` | string | `""` plain text, `json`, `jsonSchema` (with `jsonSchema`) |
| `jsonSchema` | string | The schema **as a string of JSON**, not a nested object |
| `thinking` | string | `""` Voxt's default, `off`, `on`, `effort` (with `thinkingEffort`), `budget` (with `thinkingBudget`) |
| `thinkingEffort` | string | `minimal`, `low`, `medium`, `high` |
| `thinkingBudget` | integer | Thinking tokens |
| `thinkingOffUnder` | integer | Inputs shorter than this many characters use Voxt's default thinking; `0` never |
| `verbosity` | string | OpenAI only: `""`, `low`, `medium`, `high` |
| `keepAlive` | string | Ollama only: how long it keeps the model loaded, such as `5m` |
| `includeUsage` | bool or absent | oMLX only: usage in stream events; absent = on |
| `extraBodyJSON` | string | Request parameters: a JSON object **as a string**, merged into the request last (JSON merge patch: its keys win, `null` removes a field) |

### Updates and network (Settings → About, Speech models → Network)

| Path | Type | Default | Allowed values | On the page |
|---|---|---|---|---|
| `updates.beta` | bool or absent | absent (= on while running a beta) | | Beta updates |
| `updates.noAutoCheck` | bool | `false` | | Check for updates automatically (inverted) |
| `network.proxy` | string | `""` | `""` (automatic), `direct`, a proxy URL | Speech models → Network… |
| `network.hfEndpoints` | string | `""` | `""`/`auto`, `official`, `mirror`, or URLs | Speech models → Network… |

### Kept by Voxt

Voxt writes these itself. Do not edit them; they are listed so that every field of the document is known.

| Path | Type | Meaning |
|---|---|---|
| `version` | integer | The document's schema version; Voxt rewrites it |
| `setup.done` | bool | The setup guide is finished |
| `setup.openingShown` | bool | The first-use opening was shown |
| `setup.tried` | bool | A session gave a result once; Home stops offering its Try area |
| `updates.lastCheck` | string (RFC 3339 time) or absent | When the last update check succeeded |
| `updates.dismissed` | string | The version whose "is available" card was dismissed |
| `network.proxyUser` | string | A custom proxy's user name (its password is in the Keychain) |
| `network.proxyAuthHost` | string | The proxy host that user name is for |
| `family.follow` | object, string → bool | Which shared values Voxt follows ([Shared with other apps](#shared-with-other-apps)); the one field here you may change |
| `family.migrated` | bool | The first start with the shared settings is done |
| `family.conflicts` | array of strings | Shared keys whose value differed at that start |
| `family.prompted` | array of strings | Shortcut conflicts already shown |
| `family.keysSince` | object, string → `{"keys": string, "since": RFC 3339 string}` | When each shared shortcut got its keys |
| `audio.soundPreset` | string | An old sound choice, turned into `startSound` and `stopSound` at start |
