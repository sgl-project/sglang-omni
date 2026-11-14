# Install Voxt and download a speech model

[中文](../start.md)

Voxt needs a Mac with Apple silicon and macOS 15 or later. For a first install, choose the **Voxt 2.0.3** release. To try the newest changes, use the Beta route below.

## Download the 2.0.3 release

[Download Voxt 2.0.3 (Apple silicon)](https://github.com/sgl-project/sglang-omni/releases/download/voxt-v2.0.3/Voxt-2.0.3-arm64.dmg)

The same package is also linked from:

- **News** on the repository home page: find **Voxt 2.0** and click **Download**. [SGLang Omni News](https://github.com/sgl-project/sglang-omni#news)
- The **Voxt v2.0.3** release: expand **Assets** and click `Voxt-2.0.3-arm64.dmg`. [GitHub Releases](https://github.com/sgl-project/sglang-omni/releases/tag/voxt-v2.0.3)
- The **v2.0.3** entry of the changelog: click **Download**. The changelog may tag this entry **Beta**; take the file named `Voxt-2.0.3-arm64.dmg`. [Voxt Changelog](https://voxt.actnow.dev/changelog)

## Download a Beta

Every change to Voxt merged into `main` builds a Beta, versioned `2.0.3-beta.YYYYMMDD.N` (the date and the build number). A Beta is newer than the release and may be less stable.

The newest is **Voxt Beta #40**: [download `Voxt-2.0.3-beta.20261112.40-arm64`](https://github.com/sgl-project/sglang-omni/actions/runs/43128787320/artifacts/13904657773) (about 33 MB; needs a GitHub sign-in; expires on November 26, 2026). Once it expires, or to find a newer build:

1. Open the [Voxt Beta runs](https://github.com/sgl-project/sglang-omni/actions/workflows/voxt-beta.yaml). The top run with a green check on `main` is the newest successful build. Skip runs with a yellow dot (still building) or a red cross (failed).
2. Open that run, scroll to **Artifacts** and click `Voxt-…-beta.…-arm64` to download a ZIP. Artifacts are kept for 14 days.
3. Double-click the ZIP to unzip it and open the `.dmg` inside. Its name contains `-beta.`, which tells builds and releases apart.

<div class="vx-anim" data-scene="beta-artifact" data-lang="en"></div>

A Beta does not update itself: to move to a newer Beta, or back to the release, download and install that `.dmg`.

## Install and allow permissions

1. Open the `.dmg`, drag **Voxt** to **Applications**, and open Voxt from Applications. If an older Voxt is installed, quit it first, then decide whether to replace it and import its data.

   <div class="vx-anim" data-scene="install-drag" data-lang="en"></div>

2. In the **setup guide**, click **Get started**. On the **Permissions** step, click **Ask** next to **Microphone** and choose **Allow** in the macOS prompt; click **Open System Settings** next to **Accessibility** and turn Voxt on under Privacy & Security → Accessibility. When both show **Allowed**, click **Continue**. If macOS asks you to restart Voxt, quit it and open it again from Applications.

   <div class="vx-anim" data-scene="guide-start" data-lang="en"></div>

## Download a speech model

1. On the **Model** step, click **Download and continue**. The recommended local speech model is **Qwen3-ASR 1.7B (8bit)**, about 2.5 GB. It downloads in the background, so you can continue the guide; it is ready when the bar fills.

   <div class="vx-anim" data-scene="guide-model" data-lang="en"></div>

2. If you skipped the guide, open **Speech models** in the sidebar, find the same model under **On this Mac**, click **Download**, and wait until it shows **Installed**. To reopen the guide, go to **Settings → General → Open setup guide**.

   <div class="vx-anim" data-scene="asr-download" data-lang="en"></div>

3. Open an empty text field, tap Right `⌥ Option`, say “the meeting is tomorrow at 3 pm”, and tap Right `⌥ Option` again. When the text appears at the cursor, local transcription works.

   <div class="vx-anim" data-scene="dictate" data-lang="en"></div>

Next, [set up your own API key](byok.md) for Translate, Rewrite and optional clean-up of transcriptions.

## For agents

> Following this page, install the Voxt 2.0.3 release on an Apple silicon Mac. Download a Beta from Actions only if I ask for one. Stop and let me confirm before replacing an older Voxt, importing its data, or answering a macOS permission prompt. Download the recommended local speech model and wait until it is ready. Check that transcription works with an empty text field and a test sentence, then tell me the installed version and the result.
