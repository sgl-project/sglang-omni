# Windows packaging and updates

## NSIS packaging

MyGo's Windows packager is in `third_party/mygo/cmd/mygo/{build,nsis}.go`; no Voxt-specific NSIS script or extra `mygo.json` section is needed. Build the helper into `resources/windows-amd64/voxtnew-helper.exe`, then run `go tool mygo build -platform windows/amd64` from `Voxt/`. MyGo compiles `Voxt.exe`, copies the contents of that platform resource directory beside it, and packages every staged file into `build/windows-amd64/Voxt Setup <version> amd64.exe`. This also includes the common `LICENSE`, `NOTICE`, and `LICENSES` resources listed in `mygo.json`.

The installer layout is therefore flat: `Voxt.exe` and `voxtnew-helper.exe` are side by side in the installation directory, which is what `internal/helper.Find` already supports. Arbitrary per-application files and directories can be included through `resources/windows-amd64/`; no special helper-executable option exists or is needed.

The generated NSIS installer installs for the current user, defaulting to `%LOCALAPPDATA%\Programs\Voxt`, and needs no administrator rights. It creates a Start Menu shortcut, registers the `voxtnew://` URL scheme, writes `Uninstall.exe`, and registers both interactive and silent uninstall commands under the current user's Windows uninstall registry key. The finish page can create a desktop shortcut and launch Voxt. `/S` selects a silent installation and `/D=<absolute-path>` overrides the installation directory. The uninstaller removes the application directory, shortcuts, URL registration, and uninstall registry entry.

On Windows, MyGo finds an installed `makensis` or downloads the pinned NSIS 3.13 release into the user's MyGo cache and verifies its SHA-256. The vendored `internal/nsis` package implements that download; it is not an application configuration package.

An unsigned installer and unsigned executables have no publisher identity that Windows SmartScreen can recognize. SmartScreen may show a warning or block the first launch; a user testing such a build must explicitly choose to run it. Production signing still needs a Windows code-signing certificate and should sign the app, helper, installer, and uninstaller; MyGo's `windows.certificate` or `windows.signCommand` configuration supports that later step.

## Update manifest and feed

MyGo does not update from the NSIS installer. With `updates` configured and the matching private key available, `mygo build` creates a signed archive named `voxt-<version>-windows-amd64.tar.gz` containing the same flat application directory, plus `update-windows-amd64.json`. The manifest contains `version`, Markdown `notes`, an RFC 3339 `date`, and the archive's `url`, byte `size`, and Ed25519 `signature`; it can also contain signed `deltas` and `previous` archives. The installed app checks the manifest, compares versions, downloads the archive, verifies its size and signature, and replaces the writable per-user installation directory.

For GitHub Releases hosting, the build configuration is `updates: {publicKey: <base64-key>, github: "sgl-project/sglang-omni", tagPrefix: "voxt-v"}`. The matching private key is supplied only through the `MYGO_UPDATER_PRIVATE_KEY` secret. The embedded feed uses the newest published, non-prerelease GitHub Release whose tag starts with `voxt-v`; the archive URL points at that version's own tag. The workflow must attach both the archive and manifest to the same draft release. A draft cannot serve as a live feed: end-to-end update verification becomes possible only after that release is published and an older installed build has the same public key.

Voxt 2.0 has no MyGo update feed: `release/mygo.release.json` is `{}`, and release builds instead only check GitHub's release list and offer a download. Windows packaging must preserve that default. The Windows update feed can be enabled when its channel and dedicated update key pair are provisioned; until then, CI produces the unsigned NSIS installer without embedding an update endpoint or publishing an unusable manifest.

## Local build and installation check

Run `scripts/build-windows.sh` from `Voxt/` in Git Bash on Windows. It compiles the helper as a Windows GUI executable with CGO disabled, stages it under `resources/windows-amd64/`, and invokes MyGo's Windows packager. Extra arguments are passed to `go tool mygo build`.

For a scratch installation, run `"build/windows-amd64/Voxt Setup 2.0.3 amd64.exe" /S "/D=C:\absolute\scratch\Voxt"` from a Windows command prompt, substituting the current version and an empty directory. In Git Bash, prefix the command with `MSYS2_ARG_CONV_EXCL='*'` so it does not rewrite the NSIS flags. `/D` must be the last argument. Use Task Scheduler's `/IT` option for GUI verification; an isolated `APPDATA`, `LOCALAPPDATA`, `TEMP`, and `TMP` also keeps its data and single-instance socket apart from a running development copy.

Verified on the Windows amd64 development host: the script built `Voxt Setup 2.0.3 amd64.exe`; a silent scratch install contained `Voxt.exe`, `voxtnew-helper.exe`, and `Uninstall.exe`; the Start Menu shortcut and uninstall registry values pointed into that directory. A scheduled interactive launch of the installed executable opened Diagnostics, where the self-check reported `Ready (dev)` and the running helper's path was the installed side-by-side executable. The app exited with status 0, its helper exited too, and silent uninstall removed the directory, shortcuts, URL registration, and uninstall registration.
