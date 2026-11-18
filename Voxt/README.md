<div align="center"><a name="readme-top"></a>

<img src="resources/icon.png" width="128" alt="Voxt app icon">

# Voxt

**Speak in any app on your Mac. Voxt writes it down, translates it, or answers you.**

Open-source voice input for macOS, from the [SGLang-Omni](https://github.com/sgl-project/sglang-omni) team.

**English** · [简体中文](README.zh-CN.md)

[![Latest Voxt release][release-shield]][download]
[![macOS 15 or later, Apple silicon][macos-shield]](#requirements)
[![Apache License 2.0][licence-shield]](LICENSE)

[![Download Voxt for Mac][download-shield]][download]
[![Download Voxt Beta][beta-download-shield]][beta-download]

[Install guide][install-guide] · [Changelog](CHANGELOG.md) · [Website](https://voxt.actnow.dev) · [Report a problem][issues] · [Slack][slack]

https://github.com/user-attachments/assets/8faf2c9b-a500-4d62-befc-0a82086883c1

▶ **Watch the Voxt 2.0 launch video**: [English](https://youtu.be/ktJRE7AB2mU) · [中文](https://youtu.be/mOwWavTSXu0)

</div>

## Voxt 2.0 is here

Today we are releasing **Voxt 2.0**, and with it a new idea of what open-source voice interaction can be.

Voice input built on speech recognition plus a language model's polish. Any-to-any translation by a language model. A local AI assistant that responds to any text you select, the moment you ask out loud. Voxt 2.0 is the new entry point to interactive intelligence that we, the SGLang-Omni team, are proud of.

- **Rebuilt from Swift to Go, and faster where it hurt most.** After a very long dictation, the wait once you stop is **92% shorter** than in Voxt 1.x. Voxt 2 recognises the finished parts of a long recording while you are still speaking, instead of all of it after you stop: for a dictation of 4 min 46 s on a Mac with an M4 and 24 GB, 32.3 s became 2.6 s ([how it was measured](docs/eval.md#long-recordings-on-the-speech-model-10-17-testevallongasr)).
- **Everything from Voxt 1.x, in one click.** Your dictionary, history, compatible speech models and service setups come over, so your habits stay as they were.
- **Nothing you said is lost.** When recognition fails, the recording is kept and **Retry** in History recognises it again.
- **A conversation in the answer card.** A result can wait in a floating card: keep asking follow-ups by voice, then copy or insert the answer.
- **Your models, your data.** Speech runs on your Mac with Qwen3-ASR or MOSS-Transcribe-Diarize, served by the SGLang-Omni MLX runtime, or goes to a cloud speech API you set up. The language model is anything with an OpenAI-compatible API: SGLang MLX or Ollama on your Mac, or a service such as DeepSeek V4.1 and Kimi K3.1, the two our team uses most. Where your data goes is yours to decide.
- **Optional speculative decoding for Qwen3-ASR, MOSS-Transcribe-Diarize, and Whisper Large v3.** Open **Faster transcription** on the model's card to download its draft.
- **And more**: sign in with ChatGPT instead of pasting an API key, a fallback model for Translate, your own prompts, and History export to Markdown or JSONL.

Already on Voxt 1.x? See [Coming from Voxt 1.x](#coming-from-voxt-1x). Every change, version by version: [CHANGELOG.md](CHANGELOG.md).

## What Voxt does

Tap a shortcut in any app, speak, and tap it again. Esc cancels.

### Transcribe: voice input

Tap **right ⌥ Option**. What you say is typed where your cursor is. The text shows live while you speak, and a language model can polish it: fillers and false starts out, punctuation in.

### Translate: any language to any language

**Right ⌥** with **⇧ Shift**. The selected text, or what you say, comes back in another language: 22 target languages, by the language model you choose.

### Rewrite: your AI assistant

**⌃ Control** with **right ⌥**. Select any text and say what you want: "shorter", "more formal", "explain this". Or select nothing and ask for new text. The answer waits in a card, where you can follow up by voice, then copy or insert it.

### Also built in

- **App enhancement**: groups of apps and websites with their own prompt and a key pressed after inserting, such as a casual tone in chats and Return to send.
- **Dictionary**: names and jargon that the speech model expects and the language model writes as you spell them. Voxt suggests new terms from your history and from the corrections you make.
- **Meetings**: choose Start meeting in the menu bar menu or on the Meetings page. Voxt records your microphone, the other side's audio, or both, and shows live captions. With MOSS-Transcribe-Diarize installed, the finished transcript labels the other side as Speaker 1–4 while your lines stay Me. After the meeting, a language model corrects the transcript with your dictionary and writes a summary you can ask follow-up questions about. Recording the other side needs System Audio Recording permission.
- **Notes**: tap right ⌘, speak, and Voxt saves what you said as a note instead of typing it.
- **History**: every result, with search, **Retry** from the recording, and export.

Every shortcut can be changed on its feature's page, and Settings → Shortcuts has presets: right ⌥, fn, ⌘ combinations, or the middle mouse button. ⌃⌥V pastes the last result again.

## Install

### New to Voxt

1. **Download** [`Voxt-2.1.0-arm64.dmg`][download], open it and drag **Voxt** to **Applications**.

2. **Open Voxt and follow the setup guide**: allow the microphone and Accessibility, download the recommended speech model (MOSS-Transcribe-Diarize with its Faster transcription draft, 2.1 GB), and set up a language model for clean-up, Translate and Rewrite.
3. **Tap the right Option key** in any app, speak, and tap it again.

Step by step: [Install Voxt][install-guide].

To install Beta `2.0.3-beta.20261111.38`, sign in to GitHub, download and unzip the ZIP, then open the DMG and drag **Voxt** to **Applications**. If the download does not start, select the arm64 artifact in [Voxt Beta #38][beta-run]. For newer builds, open the latest successful `main` run in [Actions → Voxt Beta][beta-workflow].

### Coming from Voxt 1.x

Take whichever way is easier:

- **From inside Voxt 1.x**: on a Mac with Apple silicon, turn on **Beta Updates** (Settings → About) and choose **Check for Updates…**. It offers Voxt 2 together with the upgrade guide. Without Beta Updates, Voxt 1.x does not offer Voxt 2 yet.
- **Install over it**: quit Voxt 1.x, download the newest disk image and drag Voxt to Applications. Choose **Replace** when Finder asks.

Then, in Voxt 2:

1. In the setup guide, click **Import from Voxt 1.x**. Your dictionary, history, the speech models you already downloaded, and your language model and speech service setups come over. Voxt 1.x's data is only read, never changed.
2. Allow the microphone and Accessibility again: macOS sees Voxt 2 as a new app.
3. Paste your API keys again and check your shortcuts. These two are not imported.

What the import brings, with screenshots: [From Voxt 1.x to Voxt 2](docs/voxt-migration.md).

### Requirements

- A Mac with **Apple silicon** and **macOS 15** or later. Voxt 2 does not run on an Intel Mac; there, keep using Voxt 1.x.
- About 280 MB for the app, and 2.1 GB for the recommended speech model and its draft.
- Voxt 2 tells you when a new version is out (Check for Updates…, and a daily check you can turn off in Settings → Diagnostics & About), but does not install it itself. Install a newer version the same way: from its disk image.

## Your models, your data

### Speech models

Speech is recognised **on your Mac** by the [SGLang-Omni MLX runtime](../sglang_omni_mlx/native): native C++ on MLX, no Python. Voxt downloads a model only when you ask for it, from Hugging Face or from a mirror when Hugging Face cannot be reached, and checks every file against a pinned SHA-256. A model loads when you first dictate and unloads after 10 idle minutes.

| Model | Good at | Download |
|---|---|---|
| **MOSS-Transcribe-Diarize**, recommended | Chinese and English, made for conversations between several people. Live text, with Faster transcription on. Takes no dictionary or Speech context: your dictionary still guides the language model's clean-up. | 2.1 GB with its draft |
| **Qwen3-ASR 1.7B (8bit)** | The most accurate: accents, dialects, names, several languages in one sentence. Live text. | 2.5 GB |
| **Qwen3-ASR 0.6B (4bit)** | Quicker and lighter, for clear speech in a quiet place. Live text. | 713 MB |

- **More local models**: Nemotron 3.5 ASR 0.6B (28 languages, live text), Qwen3-ASR 1.7B (6bit), Whisper Large v3 Turbo, Large v3 and Small (about 100 languages), Parakeet TDT 0.6B v3 (25 European languages), SenseVoice Small (Chinese, Cantonese, English, Japanese, Korean) and Cohere Transcribe 03-2026 (14 languages). macOS's own System dictation is there too, with nothing to download.
- **Qwen3-ASR** knows 30 languages and 22 Chinese dialects, and uses your dictionary; the other models you download do not.
- **Faster transcription** is available on the Qwen3-ASR 1.7B (8bit), Qwen3-ASR 0.6B (4bit), MOSS-Transcribe-Diarize, and Whisper Large v3 cards. Turn on the switch to download the DFlash draft for Qwen3-ASR or MOSS (726 MB, 262 MB, or 262 MB, respectively), or the Distil draft for Whisper (1.5 GB). Turning off the switch resumes ordinary decoding and keeps the draft on disk.
- **Cloud speech APIs**, if you prefer one: OpenAI Transcribe, Doubao (Volcengine), Zhipu GLM-ASR, Aliyun Bailian, StepFun, Xiaomi MiMo and Google Gemini.

### Language models

Clean-up, Translate and Rewrite use the language model you set up on the Language models page, over its OpenAI-compatible API:

- **On your Mac**, so the text stays here: Ollama, LM Studio, oMLX, SGLang MLX, or any other compatible server.
- **A service**: OpenAI, Anthropic, Google Gemini, DeepSeek, Qwen (Alibaba Cloud), Kimi, GLM, Doubao (Volcengine), MiniMax, StepFun, Xiaomi MiMo, xAI and OpenRouter.
- **ChatGPT**: sign in with your own ChatGPT plan in the browser, with no API key.

Without a language model, Transcribe types the text as recognised; Translate and Rewrite need one. Each feature can use its own model and its own prompt (**Your rules**), and Translate can fall back on a second model when the first one fails.

## Privacy

- **Speech stays on your Mac**, unless you set up a cloud speech API for a feature: that feature's recordings then go to that service, and nowhere else.
- **Text goes only to the language model you set up.** With a server on your Mac, it stays here. Audio is never sent to a language model. Images go to it only when you turn on sending the window or a copied image in Settings → Privacy (both off by default).
- **API keys stay in the macOS Keychain**, never in Voxt's settings or logs.
- **No account and no analytics.** Voxt has no server that receives your audio or your text. Otherwise it goes online only to download the models you choose, to read Voxt's announcements and to ask GitHub for the newest version, requests that send nothing about you but Voxt's version.
- **Your data is in two folders**: the history, its optional recordings, the dictionary and the settings in `~/Library/Application Support/VoxtNew/`, and the models in `~/Library/Application Support/SGLang Omni/models`.

## Help and community

- **Installing and upgrading**: [Install Voxt][install-guide] and [From Voxt 1.x to Voxt 2](docs/voxt-migration.md).
- **Something is wrong**: Settings → About → **Report a problem** opens a GitHub issue with the diagnostics filled in. Or [open an issue][issues] with "[Voxt]" in its title.
- **Questions and ideas**: join the SGLang [Slack][slack].
- **For your own scripts and agents**: [read your dictation history](docs/history-access.md) from other programs on your Mac. It is off by default.
- **Transcribe recordings and videos to subtitles**: [`Voxt transcribe`](docs/transcribe.md) turns a local audio or video file into SRT or text with your speech model.

## For developers

Voxt is a Go app on [MyGo](https://github.com/egoist/mygo), with a small Swift helper and the SGLang-Omni native runtime inside. Everything is in this folder:

- [Build from source, sign and test](docs/development.md)
- [How to send a change](CONTRIBUTING.md), and [the architecture](docs/architecture.md)

## Acknowledgements

Voxt 2 takes the place of the Swift [Voxt](https://github.com/hehehai/voxt) 1.x. It started from [Auralink](https://github.com/hehehai/Auralink) by the same author, with the interface of [AudioSmith](https://github.com/hehehai/AudioSmith). Third-party code, runtimes and their licences are listed in [NOTICE](NOTICE); the models are downloaded under their own licences.

## Licence

Apache License 2.0 ([LICENSE](LICENSE)), as the rest of this repository. Copyright the SGLang-Omni Team.

[![Back to top][back-to-top]](#readme-top)

[download]: https://github.com/sgl-project/sglang-omni/releases/download/voxt-v2.1.0/Voxt-2.1.0-arm64.dmg
[beta-download]: https://github.com/sgl-project/sglang-omni/actions/runs/42954481839/artifacts/13825829109
[beta-run]: https://github.com/sgl-project/sglang-omni/actions/runs/42954481839#artifacts
[beta-workflow]: https://github.com/sgl-project/sglang-omni/actions/workflows/voxt-beta.yaml
[install-guide]: https://sgl-project.github.io/sglang-omni/get_started/voxt.html
[issues]: https://github.com/sgl-project/sglang-omni/issues/new/choose
[slack]: https://slack.sglang.io
[release-shield]: https://img.shields.io/github/v/release/sgl-project/sglang-omni?filter=Voxt%20v%2A&display_name=release&label=latest&style=flat-square&labelColor=000000&color=3fb950&logo=github&logoColor=white
[macos-shield]: https://img.shields.io/badge/macOS-15%2B%20%C2%B7%20Apple%20silicon-58a6ff?style=flat-square&labelColor=000000&logo=apple&logoColor=white
[licence-shield]: https://img.shields.io/badge/licence-Apache%202.0-58a6ff?style=flat-square&labelColor=000000&logo=apache&logoColor=white
[download-shield]: https://img.shields.io/badge/Download-Voxt%20for%20Mac-000000?style=for-the-badge&logo=apple&logoColor=white
[beta-download-shield]: https://img.shields.io/badge/Download-Voxt%20Beta-1f6feb?style=for-the-badge&logo=github&logoColor=white
[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square
