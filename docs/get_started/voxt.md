# Voxt for macOS

Dictate, translate, and rewrite text by voice in any Mac app.

[![Download Voxt for Mac][download-shield]][download]
[![Download Voxt Beta][beta-download-shield]][beta-download]

Apple silicon · macOS 15+

## Install

1. Open the DMG, drag **Voxt** to **Applications**, and open it.
2. Follow the setup guide to grant microphone and Accessibility permissions and download the recommended speech model.
3. Tap **Right Option** in any app, speak, and tap it again to finish.

To install Beta `2.0.3-beta.20261111.38`, sign in to GitHub, download and unzip the ZIP, then open the DMG and drag **Voxt** to **Applications**. If the download does not start, select the arm64 artifact in [Voxt Beta #38][beta-run]. For newer builds, open the latest successful `main` run in [Actions → Voxt Beta][beta-workflow].

## Features

- **Dictation (Right Option)**. Type what you say at the cursor.
- **Translate (Shift + Right Option)**. Translate speech or selected text.
- **Rewrite (Control + Right Option)**. Select text and say how you want it rewritten.
- **Notes (Right Command)**. Speak to save a note in Voxt.

Set up a language model in Voxt to use Translate and Rewrite. Voxt also has a custom dictionary, history retry and export, and app-specific prompts.

[Upgrade from Voxt 1.x](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/docs/voxt-migration.md) · [Full feature list](https://github.com/sgl-project/sglang-omni/tree/main/Voxt#what-voxt-does) · [Changelog](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/CHANGELOG.md)

[download]: https://github.com/sgl-project/sglang-omni/releases/download/voxt-v2.0.3/Voxt-2.0.3-arm64.dmg
[beta-download]: https://github.com/sgl-project/sglang-omni/actions/runs/42954481839/artifacts/13825829109
[beta-run]: https://github.com/sgl-project/sglang-omni/actions/runs/42954481839#artifacts
[beta-workflow]: https://github.com/sgl-project/sglang-omni/actions/workflows/voxt-beta.yaml
[download-shield]: https://img.shields.io/badge/Download-Voxt%20for%20Mac-000000?style=for-the-badge&logo=apple&logoColor=white
[beta-download-shield]: https://img.shields.io/badge/Download-Voxt%20Beta-1f6feb?style=for-the-badge&logo=github&logoColor=white
