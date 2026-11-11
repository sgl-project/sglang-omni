# Install Voxt

[Voxt](https://github.com/sgl-project/sglang-omni/tree/main/Voxt) is voice input for macOS built on SGLang-Omni: press a shortcut in any app, speak, and the text appears where your cursor is. Speech is recognised on your Mac by SGLang-Omni's native MLX runtime.

**Download:** the newest `Voxt-X.Y.Z-arm64.dmg` from [the Voxt releases](https://github.com/sgl-project/sglang-omni/releases?q=voxt-v&expanded=true).

**Needs:** a Mac with Apple silicon and macOS 15 or later. The app is about 280 MB; the recommended speech model is a 2.5 GB download. Voxt 2 does not run on an Intel Mac; there, keep using Voxt 1.x.

## What is new in Voxt 2

Voxt 2 is a new app that replaces the Swift Voxt 1.x. The biggest changes:

- **Speech recognition runs on SGLang-Omni**: the local speech models (Qwen3-ASR 1.7B and 0.6B, and the 1.x models such as Whisper, Parakeet and SenseVoice) run on SGLang-Omni's native MLX servers, with the text shown live as you speak. **Faster transcription** adds a small draft model that speeds up Qwen3-ASR by speculative decoding.
- **You choose the language model**: clean-up, Translate and Rewrite use a service you set up (OpenAI, Anthropic, Gemini, DeepSeek, Qwen, Kimi, GLM, Doubao and others), or one on your Mac with Ollama, LM Studio or oMLX. Voxt 2 no longer runs language models itself.
- **Rewrite keeps a conversation** in the answer card: say a follow-up ("shorter", "now in English"), then copy or insert the answer.
- **The dictionary learns**: Voxt suggests names and terms from your history and from the corrections you make after it types.
- **One-click import from Voxt 1.x**: the dictionary, history, downloaded speech models, and your language model and speech service setups come over.
- **Open source** in this repository, under the Apache License 2.0.

Voxt 2 does not update itself: install each new version from its disk image. The full list of changes is in the [changelog](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/CHANGELOG.md).

## New users

1. Download the disk image and open it.
2. Drag **Voxt** to **Applications**, then open Voxt from Applications.
3. Follow the setup guide:
   - Grant the **microphone** and **Accessibility** permissions.
   - Download the recommended speech model, **Qwen3-ASR 1.7B (8bit)** (2.5 GB).
   - For clean-up, Translate and Rewrite, set up a language model on the **Language models** page.
4. Tap the **right Option** key in any app, speak, and tap it again.

## Upgrading from Voxt 1.x

1. **Quit Voxt 1.x** from its menu bar icon. If it does not quit, use Force Quit (⌥⌘⎋).
2. Open the Voxt 2 disk image and drag **Voxt** to **Applications**. When Finder asks:
   - **Replace** replaces the app. Voxt 1.x's data stays, so reinstalling Voxt 1.x brings it back.
   - **Keep Both** keeps Voxt 1.x next to Voxt 2. Their default shortcuts do not overlap.
3. Open Voxt. The setup guide starts with **Import from the previous Voxt**. Click it. If an open panel appears, click **Import** without changing the folder.
4. The import brings over the dictionary, the history, your language model and speech service setups, and the speech models Voxt 1.x downloaded (no new download). It does not bring over API keys, shortcuts, other settings apart from Show in the Dock, or Voxt 1.x's local language models.
5. **Grant the microphone and Accessibility again.** macOS treats Voxt 2 as a different app, so the permissions of Voxt 1.x do not carry over.
6. Set up a language model on the **Language models** page.

**Other ways to upgrade:** on a Mac with Apple silicon, turn on **Beta Updates** in Voxt 1.x (Settings → About) and choose **Check for Updates…**: it offers Voxt 2 together with the upgrade guide. Without Beta Updates, Voxt 1.x does not offer Voxt 2 yet. Then continue from step 3.

**Another Mac:** in Voxt 1.x, export the dictionary (Dictionary → More → Export); in Voxt 2, import the file (Dictionary → ••• → Import…). Only the dictionary moves this way.

**Going back:** reinstall [Voxt 1.16.3](https://github.com/sgl-project/sglang-omni/releases/tag/voxt-v1.16.3). Its data is where Voxt 1.x left it.

More detail, with screenshots: [From Voxt 1.x to Voxt 2](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/docs/voxt-migration.md).
