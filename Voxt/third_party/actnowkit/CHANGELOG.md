# Changelog

## v0.9.1 — 2026-10-20

- `sounds`: for the pickers' second look: `Player.Previewing` (the choice
  the last preview plays and how long it has left; `StopPreview` ends it),
  `UnknownLength`, `Player.SetClock`; `VolumeSteps` (0, 25, 50, 75, 100),
  `NearestVolumeStep`, `VolumeLevel` (muted, low, high: the volume
  button's icon); `Catalog.GroupName`.

## v0.9.0 — 2026-10-20

- New package `sounds`: the family's cue sounds. `Load` reads an app's
  `catalog.json` (schema `actnow.sounds.catalog/1`: groups, tags, uses,
  sounds with names in English and Chinese, durations, pairs, the system
  sounds each replaces, SHA-256) from the app's `fs.FS`; actnowkit ships no
  audio. `Event`, `Prefs` (on, volume, a choice per event: a sound, `none`
  or `system:<name>`), `Target`, `Resolve` with a system-sound fallback for
  builds without the catalogue, `MigrateSystem` / `MigrateClosest` for old
  settings; `Filter`, `GroupCounts`, `TagCounts`, `Suggested` for pickers;
  `Player` (`Preload`, `PlayEvent`, `Preview` cutting off the last one,
  `StopPreview`). macOS output: WAV decoded into an AVAudioEngine with 8
  player nodes (starts in microseconds, repeats overlap, the engine stops
  after 5 s idle), AVAudioPlayer for other files, NSSound for system
  sounds; `Silent` elsewhere; `Recorder` for tests. `DecodeWAV`, `ToneWAV`.
- New command `cmd/actnow-sounds`: `list`, `check`, `subset` (the sounds an
  app ships, with their catalogue and a size limit), `play` (macOS).

## v0.8.0 — 2026-10-19

The family audit's first batch (spec v1.0.9). **The model library moves**:
apps that update must call `Store.Import` before showing model states.

- `models`: **the default `$HUB` is `$FAMILY/models/hub`**
  (`~/Library/Application Support/ActNow/models/hub`, `DefaultHub`,
  `DefaultHubIn`), still in the Hugging Face layout. `ResolveHub` no longer
  reads `HF_HUB_CACHE` / `HF_HOME`: an app started from Finder and one
  started from a shell chose different folders. The Hugging Face caches are
  now import sources (`LegacyHubs`; `EnvHub` keeps its meaning,
  `HFDefaultHub` is `~/.cache/huggingface/hub`, `FromHFDefault` its source).
- `models`: `Store.Import(ctx, from, backup, want...)` brings an app's
  catalogue models (and the claims of its previous library) over from those
  caches: hard links on the same volume, a copy across volumes when it
  leaves `ImportReserve` free; holding the destination repo's lock
  exclusively and the source's shared; refs naming the commit copied;
  the app's previous claims claimed again (`OriginImported`; the
  catalogue models are left to the app's own migration or reuse card).
  Nothing in the source is changed, models no
  family app wants are not touched. Each app records what it handled in
  `$FAMILY/models/imported/<bundle id>.json`, so a model deleted after the
  import does not come back. Deleting a hard-linked blob reports nothing
  freed (`Decision.FreeBytes`, `Result.Freed`): the cache still holds it.
- `models`: Time Machine. `SetBackupExcluded` / `IsBackupExcluded` set the
  sticky attribute `com.apple.metadata:com_apple_backup_excludeItem` (the
  bytes `tmutil addexclusion` writes; pure Go through purego).
  `ApplyBackup(fam, hub, backup)` marks the whole default library, and in a
  folder the user chose only the repos family apps claim; `MarkRepoBackup`
  for one repo. hfhub marks the repos it finishes. `EnsureHub` creates the
  library with `$FAMILY`'s 0700 parents.
- `settings`: `models.backup` (`KeyModelsBackup`, `"on"`/`"off"`, default
  off = excluded; `ModelsBackup(s)`), family-wide without a follow flag.
- `models`: the "installed" check compares sizes. Downloaders record each
  blob's verified size in `models--<owner>--<name>/.actnow/blobs.json`
  (`RecordBlobSizes`, `ReadBlobSizes`; hfhub and `Import` do), and
  `Claim.Sizes` carries catalogue sizes (hfhub fills it from `Entry.Files`):
  a file of another size is listed in `Status.Damaged` / `Local.Damaged`
  and the model is `Partial`. A download repairs it.
- `models`: lost claims are "unsure". `AppState.ModelsRoot` (from the
  manifest, `registry.States`) names the `$HUB` an app keeps models in; an
  installed app with this `$HUB` and no claims file makes `Decide` unsure
  (`Decision.Unclaimed`), so nothing it may use is deleted.
  `Store.Heal(ctx, installed...)` recreates the app's claims file when it is
  missing (claiming the installed models, or writing it empty): call it at
  launch and on activation.
- `redact` (new): `URL`, `Text`, `Error` take API keys and tokens out of URL
  queries (`key=`, `token=`, `X-Amz-Signature=`…), `user:pass@` out of proxy
  URLs, `Authorization` headers, bearer tokens and well-known key shapes
  (`AIza…`, `sk-…`, `hf_…`, `ghp_…`) out of logs, error details and
  diagnostics.
- `runtime`: files carry the writer's executable (`File.Exe`); `Live` (and
  so `ReadAll`, `Others`, `LoadedBy`) ignores a file whose pid now runs
  another executable, or that was written before the system started, so a
  crashed app's models stop counting when its pid is reused.
- `models` texts: `TextBackupModels`, `TextBackupModelsHint`, `TextImported`,
  `TextFolderDefault`, `TextFolderChosen`, `TextDamaged`, `TextUnclaimed`.
- `conformance`: the fixture's `$HUB` is `$FAMILY/models/hub`
  (`Fixture.Legacy` is a Hugging Face cache beside it, `$HF_HUB_CACHE`);
  registered peers write their claims file; new scenario
  `DeleteWithLostClaims`. `Check` skips the library inside `$FAMILY`.
- `hfhub`: `CacheDir` is `models.DefaultHub`; a waiter whose blob holder
  left no owner file tries the lock again before reporting "unknown" (a
  flaky test on macOS).

## v0.7.2 — 2026-10-19

A pins table update; nothing else changed.

- `models`: the Qwen3-ForcedAligner 0.6B 4-bit pin lists Act, which now
  uses it (optional subtitle timing, M5) at the same commit, so the other
  apps do not clean it up while Act may use it.

## v0.7.1 — 2026-10-19

A fix to the default shortcut table; nothing else changed.

- `shortcuts`: Act's Quick Collect moves to `Control+Option+Command+S`
  ("save"). v0.7.0 gave it ⌃⌥⌘A, which Pik already uses for Pick Area
  (off by default, and not in the table). Pik's Pick Area now has its
  row (`pik`/`pickArea`, ⌃⌥⌘A, off), so the disjointness test guards it.

## v0.7.0 — 2026-10-19

Act joins the family, with the capabilities, model roles and pins it
uses; nothing else changed, so code written for v0.6 builds and behaves as
before (new rows and constants only).

- `family`: Act joins the table (`act`, `dev.hehehai.act`, URL scheme
  `act`, https://actnow.dev/act, team `TeamActNow`). Its engine
  (`dev.hehehai.act.engine`) is a component, not an app.
- `shortcuts`: Act's default, Quick Collect on `Control+Option+Command+A`
  (Carbon, enabled), free in the family table (Pik's picker is ⌃⌥⌘P,
  HereNow's panel ⌃⌥⌘H, Dicta's reserved meeting key ⌃⌥⌘R). Its
  screenshot shortcuts have no default keys.
- `registry`: new capabilities `CapLibrary` (`"library"`: a library that
  takes files other apps hand over, `<scheme>://import?path=<file>…` and
  `<scheme>://new?url=<link>`), `CapEmbedLocal` (`"embed.local"`) and
  `CapVLMLocal` (`"vlm.local"`), the strings Act already declares.
- `models`: roles `RoleEmbed` (`"embed"`) and `RoleVLM` (`"vlm"`); the
  pins table gains EmbeddingGemma 2 (`mlx-community/embeddinggemma-2-8bit`
  at `7505ef2`, role `embed`) and lists Act on the three pins its
  catalogue shares (Qwen3-ASR 0.6B, Qwen3.5 2B, Gemma 4 E4B), so the
  other apps do not clean them up while Act may use them.

## v0.6.0 — 2026-10-18

One app and one capability join the family; nothing else changed, so code
written for v0.5 builds and behaves as before (`family.Apps` gains a row,
`registry` a constant).

- `family`: Paralign joins the table (`paralign`, `dev.hehehai.paralign`,
  URL scheme `paralign`, https://actnow.dev/paralign, team `TeamActNow`).
  Its engine (`dev.hehehai.paralign.engine`) is a component, not an app.
  It has no global shortcut, so `shortcuts.Defaults` is unchanged.
- `registry`: new capability `CapLearning` (`"learning"`): the app does
  local language learning from media and opens
  `<scheme>://import?path=<file>[&subtitles=<file>]` for a video or audio
  file another app hands over, optionally with the subtitles to study
  (CaptionCraft's and Dicta's "Study in Paralign").

## v0.5.1 — 2026-10-18

- `mirror`: the ActNow mirror's files are now served by R2 itself
  (Worker responses reach mainland China at a few KB/s, R2's at MB/s), and
  R2 cannot redirect: `Client.Latest` also reads a `200` answer with JSON
  `{"tag": "<tag>"}` (GitHub still redirects). `Client.File` gives a
  source it left for stalling or crawling one more try on a new
  connection (some of a CDN's addresses crawl from China, others do not);
  the last attempt is never left for being slow.

## v0.5.0 — 2026-10-18

Downloads from GitHub for users in China; nothing else changed.

- New package `mirror`: files of GitHub releases from github.com or the
  family's mirror, `https://mirror.actnow.dev/github` (spec §9.6), which
  answers GitHub's paths. `Sources` maps the family's download source
  (`network.hfEndpoints`: automatic → both, `official` → github.com,
  `mirror` → the mirror; `$ACTNOW_GITHUB_SOURCES` replaces them).
  `Client.File` races the sources for each file (the first to deliver
  `Lead` bytes wins), leaves a source that stalls, crawls, fails or serves
  a file with another SHA-256 for the next, and returns `ErrChecksum`
  (`*ChecksumError`) when only wrong files came. `Client.Latest` reads
  the latest release's tag from the `/releases/latest` redirect of the
  first source to answer; `Try` runs a check that needs several files of
  one source (a checksum file and its signature) at every source and keeps
  the first that passes; `Bytes` and `Small` fetch small files. Reference
  wordings `TextSourceAuto`, `TextSourceOfficial`, `TextSourceMirror`,
  `TextSourceNote`.

## v0.4.0 — 2026-10-18

Two apps join the family; nothing else changed, so code written for v0.3
builds and behaves as before (`family.Apps` and `shortcuts.Defaults` gain
rows).

- `family`: HereNow joins the table (`herenow`, `dev.hehehai.herenow`,
  URL scheme `herenow`, https://actnow.dev/herenow, team `TeamActNow`).
- `shortcuts`: HereNow's default, `openPanel` "Open HereNow" on
  ⌃⌥⌘H (`Control+Option+Command+H`, press, Carbon, enabled, not
  exclusive); it collides with no other default.
- `family`: CaptionCraft joins the table (`captioncraft`,
  `dev.hehehai.captioncraft`, URL scheme `captioncraft`,
  https://actnow.dev/captioncraft, team `TeamActNow`). It has no global
  shortcut, so `shortcuts.Defaults` is unchanged.
- `registry`: new capability `CapSubtitles` (`"subtitles"`): the app makes
  video subtitles and opens `<scheme>://import?path=<file>` for a file
  another app hands over (Dicta's "Make subtitles in CaptionCraft").

## v0.3.0 — 2026-10-17

- New package `listsel`: the row selection of the apps' log views, with no
  UI. `Model[K]` holds the rows selected by stable key, the anchor and the
  pointer gesture: a plain click copies its row (`CopyRow`) and sets the
  anchor, Shift-click selects a range, ⌘-click toggles a row, pressing and
  moving more than `DragSlop` selects by dragging (⌘ adds), `SelectAll`,
  `Clear`, `Cancel`, and `Prune` as filters change and lines rotate out.
  `Items` and `Text` format the lines to copy; `TextSelected` … `TextCopied`
  are reference wordings in English and Chinese. Nothing else changed.

## v0.2.2 — 2026-10-17

- `models`: pins: Dicta and AudioSmith share one MOSS Transcribe Diarize
  repo, `vanch007/mlx-MOSS-Transcribe-Diarize-8bit` (Dicta moved to it in
  its 736f427); the `majentik/MOSS-Transcribe-Diarize-MLX-8bit` pin, which
  no app's catalogue has any more, is gone. `PinsOfApp("dicta")` now
  includes the shared repo, so AudioSmith's phase-0 check keeps it for
  Dicta.

## v0.2.1 — 2026-10-17

- macOS: the AppKit lookups (`registry.DefaultSystem.AppPaths`, `Running`,
  `TeamID`, and so `registry.Discover`) lock the OS thread for the life of
  their autorelease pool. Called from a goroutine, the scheduler could move
  the goroutine to another thread between the pool's push and pop, which
  crashed the process ("semasleep on Darwin signal stack"). Callers no
  longer need a workaround of their own.

## v0.2.0 — 2026-10-17

- `models`: a claim names every model of the app that uses the repo
  (`Claim.Models`, a new member: claims files stay `actnow.models.claims/1`,
  and files without it read as before, `model` alone). Claiming merges the
  names (and the file lists); `Store.Release(ref, model)` removes one name,
  and the claim goes with the last; `Store.DeleteModel(ctx, ref, model,
  scope)` removes one model: when other models of the app still use the
  repo, ThisApp keeps the claim and the files (`Result.OwnModels`).
  `Decision.MyModels`. So deleting one of two models that share an
  auxiliary repo no longer drops the other's claim on it.
- `models`: `HeldLock.Downgrade` (exclusive → shared); `family`:
  `Lock.Convert`.
- `hfhub`: `Client.Delete` removes the entry by its id (`DeleteModel`) from
  its main and auxiliary repos.
- `conformance`: the `CatalogApp` interface and the `SharedRepoWithinApp`
  scenario; sample `claims-models.json`.

## v0.1.0 — 2026-10-17

First release: the first phase of the ActNow family protocol (spec v1.0.2).

- `family`: the family directory, JSON files with schema versions and kept
  unknown members, atomic writes, corrupt files set aside, `flock` locks,
  Darwin notifications (purego; in-process elsewhere), the family app table
  (AudioSmith, Dicta, Auralink, VoxtNew, Pik) and bundle-id variants.
- `registry`: manifests, discovery through manifests, LaunchServices and
  `NSRunningApplication`, Info.plist reading (XML and binary), the signing
  Team ID.
- `models`: `$HUB` resolution, claims, repo and blob locks with owner files,
  the installed check, delete decisions and deletion, "delete for all apps"
  tombstones, orphan clean-up, usage, reuse survey, `mlx-audio` clean-up,
  licence acceptances, the family pins table.
- `hfhub` (experimental): the Hugging Face downloader from Dicta's
  `internal/hub`, on the family locks and claims.
- `settings`: `settings.json`, follow flags, first-run migration.
- `credentials`: the vendor table, the `security` Keychain wrapper, the
  consent flow, legacy item migration.
- `shortcuts`: canonical keys, conflicts, prompts, family defaults.
- `runtime`: runtime files, the family memory budget, the pasting flag.
- `conformance`: sample files and cases, fixture, fake Hugging Face
  endpoint, scenario suite, directory check; `cmd/actnow-conformance`.
