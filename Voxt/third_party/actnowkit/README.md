# actnowkit

Shared Go packages for the [ActNow](https://actnow.dev) family of macOS apps
(AudioSmith, Dicta, Auralink, VoxtNew, Pik, HereNow, CaptionCraft, Paralign,
Act).
They let the apps on one Mac find each other and share what can be shared —
the Hugging Face model cache, remote-service API keys, general settings —
without deleting each other's models or fighting over global shortcuts.

actnowkit is the reference implementation of the ActNow family protocol: a
set of JSON files, `flock(2)` locks and Darwin notifications under
`~/Library/Application Support/ActNow` and the Hugging Face cache. An app may
implement the protocol itself instead; the [`conformance`](#conformance)
package checks either.

- Pure Go: builds with `CGO_ENABLED=0`. The macOS parts (Darwin
  notifications, LaunchServices, `NSRunningApplication`, code-signing Team
  ID) use [purego](https://github.com/ebitengine/purego); on other systems
  they compile to no-ops, so the logic is tested on Linux.
- No UI, no dependency on MyGo, no i18n: packages export English and Chinese
  reference strings (`family.Text`) for apps to put in their own catalogues.
- Status: **v0.6** — `hfhub` is experimental; the other packages follow the
  protocol's first phase. APIs may change before v1.0.

```sh
go get github.com/hehehai/actnowkit@v0.7.0
go mod tidy   # adds purego, which only the darwin files import
```

Go 1.27 or later. Apple silicon and Intel Macs, macOS 12 or later.

## Packages

| Package | What it does |
| --- | --- |
| [`family`](family) | The family directory (`$FAMILY`), the JSON file rules (schema versions, unknown members kept, atomic replacement, corrupt files set aside), `flock` locks, change notifications (`notify_post` / `notify_register_file_descriptor`), the table of family apps and bundle-id variants. |
| [`registry`](registry) | Each app's manifest (`apps/<bundle id>.json`: version, path, capabilities, shortcuts, services), and discovery: manifests + LaunchServices + running processes → which family apps are installed and running. |
| [`models`](models) | The shared model library (`$HUB`, the Hugging Face cache): resolving it, claims ("I use repo@commit", naming every model of the app that uses it), repo and blob locks compatible with huggingface_hub's `.locks/`, the installed check, delete decisions and deletion that keep other apps' models, tombstones, orphan clean-up, disk usage, the family pins table, model licence acceptances. |
| [`hfhub`](hfhub) | *Experimental.* A Hugging Face downloader on top of `models`: mirrors raced, parallel ranged downloads, resume (including another app's partial files), verification, and the family locks and claims. |
| [`mirror`](mirror) | Files of GitHub releases (yt-dlp, QuickJS, ffmpeg) from github.com or the family's mirror `https://mirror.actnow.dev/github`: sources raced per file, a stalled, crawling or wrong source left for the next; the caller's checks unchanged. |
| [`settings`](settings) | `settings.json` (language, proxy, Hugging Face endpoints, model folder), "follow the family" flags, the first-run migration of an app's own values. |
| [`credentials`](credentials) | Shared API keys: the vendor table, the login-Keychain item `ActNow` / `vendor.<id>` through `/usr/bin/security`, `credentials.json` (hint, fingerprint, users), the consent flow. |
| [`shortcuts`](shortcuts) | Canonical key combinations, the conflict rule, who shows the conflict prompt, the family's default assignments. |
| [`runtime`](runtime) | Running-process files (`runtime/<bundle id>.json`: loaded models, memory budget, pasting flag) and the family memory budget. |
| [`redact`](redact) | Secrets out of logs, error details and diagnostics: API keys and tokens in URL queries, presigned-URL signatures, `user:pass@` in proxy URLs, `Authorization` headers, well-known key shapes. |
| [`sounds`](sounds) | Cue sounds: the catalogue an app ships (`catalog.json` + audio, read from the app's `fs.FS`; actnowkit carries no audio), each event's choice (a catalogue sound, a macOS system sound, none), filters for a picker (group, tags, text, suggested uses), previews, and a low-latency player (AVAudioEngine on macOS). |
| [`listsel`](listsel) | The row selection of the apps' log views: click to copy a line, Shift/⌘-click and drag to select rows, ⌘A, Esc; rows named by stable keys. No UI. |
| [`conformance`](conformance) | Sample files with expected results, a fixture (`$FAMILY` + `$HUB` in temp dirs), a fake Hugging Face endpoint, and the scenario suite any implementation can run. |
| [`cmd/actnow-sounds`](cmd/actnow-sounds) | `list`, `check`, `subset` (copy the sounds an app ships, with their catalogue), `play` (macOS) for a sound catalogue. |
| [`cmd/actnow-conformance`](cmd/actnow-conformance) | Inspect a real family directory on a Mac: `check`, `discover`, `claims`, `lock`, `watch`, `pins`. |

## The protocol in brief

```
~/Library/Application Support/ActNow/      $FAMILY (0700; files 0600)
├── family.json                            created once
├── apps/<bundle id>.json                  each app's manifest (written by that app only)
├── settings.json                          shared settings (multi-writer)
├── credentials.json                       shared credential metadata, never secrets (multi-writer)
├── models/claims/<bundle id>.json         each app's model claims (that app only)
├── models/removed.json                    "deleted for all apps" tombstones (multi-writer)
├── models/licences.json                   model licence acceptances (multi-writer)
├── models/imported/<bundle id>.json       what the app brought over from Hugging Face caches (that app only)
├── models/hub/                            $HUB, the model library (by default; see below)
├── runtime/<bundle id>.json               running process state (that app only)
└── locks/<name>.lock                      flock locks of the multi-writer files

$HUB = $FAMILY/models/hub (or the family's models.root)       excluded from Time Machine unless models.backup is "on"
├── models--<owner>--<name>/{blobs,snapshots,refs}    the standard Hugging Face layout
├── models--<owner>--<name>/.actnow/blobs.json        verified size of each blob (the "installed" check compares)
├── .locks/models--<owner>--<name>/<blob>.lock        blob locks (the same files huggingface_hub locks)
├── .locks/models--<owner>--<name>/actnow-repo.lock   repo lock: shared to download or claim, exclusive to finish or delete
└── mlx-audio/<owner>_<name>/                         mlx-audio-swift's flat copies
```

Since v0.8.0 (spec v1.0.9) the library is no longer `~/.cache/huggingface/hub`:
`HF_HUB_CACHE`, `HF_HOME` and `~/.cache/huggingface/hub` are only where
`Store.Import` finds models to bring over (hard links on the same volume),
once per app.

- Every file has `"schema": "actnow.<kind>/<major>"`. Readers ignore unknown
  members; writers keep them. A higher major is read-only.
- Writes go through a temporary file and `rename(2)`. Multi-writer files are
  read-modified-written under `flock` on `locks/<name>.lock`.
- After writing, apps post `dev.actnow.family.<topic>` (`apps`, `settings`,
  `credentials`, `models`, `runtime`, `memory`, `account`). Notifications are
  hints: apps also re-read when activated.
- A model is `repo@commit`. Files are deleted only when no other installed
  app claims the snapshot and no running app has it loaded.
- Secrets live only in the login keychain.

Directories other than the real one (tests, `ACTNOW_FAMILY_DIR`) post
notifications under `dev.actnow.family.<hash of the directory>.<topic>`, so
test runs never wake the user's apps.

## Using it in an app

### At start-up

```go
import (
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/settings"
)

const bundleID = "dev.hehehai.dicta" // the running bundle's id (….dev for dev builds)

fam := family.Default()
info, err := fam.Ensure(bundleID) // creates $FAMILY and family.json when missing
if err != nil || info.ReadOnly() {
	// keep working as before; a newer app owns the directory's format
}
fam.CleanAllTemps()

// Shared settings and the model library this app uses.
famSettings, _, _ := settings.Read(fam)
root, _ := famSettings.Get(settings.KeyModelsRoot)
hub, _ := models.ResolveHub(follow.Of(settings.KeyModelsRoot), root, own.ModelsFolder)

// The manifest: at every start, and when the version, shortcuts or services change.
registry.Write(fam, registry.Manifest{
	App: "dicta", BundleID: bundleID, Name: "Dicta", Version: version,
	Path: registry.BundlePath(), TeamID: registry.TeamID(), URLScheme: "dicta",
	Capabilities: []string{registry.CapModelsHF, registry.CapASRLocal, registry.CapLLMLocal, registry.CapMeeting},
	Models:       &registry.ModelsInfo{Root: hub},
	Links:        map[string]string{registry.LinkModels: "dicta://models"},
})

// Who else is here.
found, _ := registry.Discover(fam, registry.DefaultSystem)

store := &models.Store{
	Family: fam, Hub: hub, BundleID: bundleID, AppID: "dicta",
	Apps: func() map[string]models.AppState { return registry.States(found) },
}
store.ApplyTombstones() // drop claims on models another app deleted for all apps

// The library moved (v0.8.0): bring this app's models over from the
// Hugging Face caches, once; nothing is downloaded, the caches stay as they are.
backup := settings.ModelsBackup(famSettings)
store.Import(ctx, models.LegacyHubs(hub), backup, wanted...)
// The claims file lost with the family folder? Claim the installed models again
// (at start-up and when the app becomes active).
store.Heal(ctx, wanted...)
// Time Machine: the library is excluded unless models.backup is "on"
// (again when TopicSettings says it changed).
models.ApplyBackup(fam, hub, backup)

// Re-read on changes (and when the app becomes active).
w, _ := fam.Watch(family.TopicApps, family.TopicSettings, family.TopicModels, family.TopicMemory)
go func() {
	for topic := range w.C {
		reload(topic)
	}
}()
```

### First run with the protocol (migration)

```go
// Settings: merge the app's own values without overriding the user's choices.
outcomes, _ := settings.Migrate(ctx, fam, bundleID, map[string]settings.Local{
	settings.KeyLanguage:    {Value: own.Language, IsDefault: own.Language == ""},
	settings.KeyProxy:       {Value: own.Proxy, IsDefault: own.Proxy == ""},
	settings.KeyHFEndpoints: {Value: own.Endpoints, IsDefault: own.Endpoints == "auto"},
	settings.KeyModelsRoot:  {Value: own.ModelsFolder, IsDefault: own.ModelsFolder == "", InUse: hasModels},
})
for _, o := range outcomes {
	follow[o.Key] = o.Follow // keep the flags in the app's own settings
	if o.Conflict {
		// show settings.TextOtherValue once on that row
		// (for models.root: models.TextMoveModels)
	}
}

// Models: claim what of the catalogue is already installed. Disk only, no network.
for _, e := range catalogue {
	if models.Check(hub, models.Claim{Repo: e.Repo, Revision: e.Revision}).State == models.Installed {
		store.Claim(ctx, models.Claim{Repo: e.Repo, Revision: e.Revision, Model: e.ID, Role: models.RoleASR, Origin: models.OriginMigrated})
	}
}
```

### Models: reuse, download, delete

```go
// First-run reuse card: what of the recommended pack is already here, and from whom.
found := store.Survey(wanted...)            // found[i].By[0].Name == "AudioSmith"
store.Claim(ctx, claimsOf(found)...)        // "Use them": Origin models.OriginReused

// Download with the family locks; the result is claimed.
dl := hfhub.New(hfhub.Options{Store: store, Endpoints: settings.Endpoints(endpoints), Proxy: proxy})
err := dl.Download(ctx, hfhub.Entry{ID: "qwen3-asr-0.6b-4bit", Repo: repo, Revision: commit}, func(p hfhub.Progress) {
	if p.WaitingFor != "" { /* "Downloading in AudioSmith…" (models.TextDownloadingIn) */ }
})

// Delete: ask first who else uses it.
d, _ := store.Decide(models.Ref{Repo: repo, Revision: commit})
if d.InUse() {
	// models.TextRemoveTitle / TextRemoveBody with d.UserNames() and d.FreeBytes;
	// buttons: "Remove from Dicta" (models.ThisApp) or "Delete for all apps…" (models.AllApps)
}
res, err := store.Delete(ctx, d.Ref, models.ThisApp) // res.Deleted is false when others still use it
var busy *models.BusyError
if errors.As(err, &busy) { /* models.TextBusy: another app is downloading it */ }

// One catalogue model of several sharing a repo (an auxiliary tokenizer):
// claim with the model's id, and remove by it. The claim keeps the other
// models' names, and the files stay while any of them uses the repo.
store.Claim(ctx, models.Claim{Repo: tok, Revision: tokCommit, Model: "whisper-turbo", Aux: true})
res, err = store.DeleteModel(ctx, models.Ref{Repo: tok, Revision: tokCommit}, "whisper-turbo", models.ThisApp)
// res.OwnModels lists this app's other models still using it; store.Release only edits the claim.

// Storage page.
usage, _ := store.Usage()       // usage.Mine vs usage.Shared
orphans, _ := store.Orphans()   // models only uninstalled apps claimed; store.CleanOrphans(ctx, bid)
```

`hfhub.Client.Delete(ctx, entry, scope)` does this for an entry's main
and auxiliary repos.

Locks: `models.LockRepo` / `TryLockRepo` (shared to download or claim,
exclusive to finish or delete), `HeldLock.Upgrade` and `HeldLock.Downgrade`
(not atomic: re-check after either), `models.LockBlob` / `TryLockBlob`.

Engines must be given the snapshot path (`models.SnapshotDir(hub, ref)`),
never pick a snapshot themselves, and never resolve repos from
`~/.cache/huggingface` on their own: pass them `hub` too (for mlx-audio's
copies).

`models.Check` compares each file's size with the verified size of its blob
(`.actnow/blobs.json`, written by hfhub and `Store.Import`; an app's own
downloader calls `models.RecordBlobSizes`) and with `Claim.Sizes` from the
catalogue: a truncated file makes the model `Partial` (`Status.Damaged`).

Logs: pass every line, error detail and exported file through
`redact.Text` (or wrap errors with `redact.Error`).

### Shared API keys

```go
cs := &credentials.Store{Family: fam, BundleID: bundleID} // Keychain: service "ActNow"

// Provider: the user saved a key with "Let other ActNow apps on this Mac use this key" checked.
cs.Save(ctx, "deepseek", key, credentials.SaveOptions{})
// keep credentials.Ref("deepseek") as the provider's credentialRef

// Consumer: offer it without reading the secret.
if e, ok := cs.Offer("deepseek"); ok {
	// credentials.TextUseKeyFrom: "Use the DeepSeek key from Auralink (sk-…a1b2)"
}
key, err := cs.Use(ctx, "deepseek")   // only after the user clicked "Use this key"
key, err = cs.Secret(ctx, "deepseek") // later reads; ErrNotConsented before Use
```

The Keychain is written with `security add-generic-password -X <hex>` on
standard input (no escaping, nothing in the process list) and read with
`security find-generic-password -g` (unambiguous for any secret).

### Shortcuts

```go
me := registry.HolderOf(myManifest)
for _, c := range shortcuts.FindConflicts(me, registry.Holders(found)) {
	// warning on the row; when c.Prompt, show shortcuts.TextConflict once per c.Key()
}
hits := shortcuts.UsedBy(recordedKeys, bundleID, registry.Holders(found)) // under the recorder
```

`shortcuts.Defaults` is the family's default assignment; its test fails if
two defaults collide.

### Memory

```go
cap := famruntime.FamilyCap(famruntime.PhysicalMemory())
budget := famruntime.Budget(ownLimit, cap, famruntime.OthersLoadedBytes(fam, bundleID))
famruntime.Write(fam, famruntime.File{BundleID: bundleID, BudgetBytes: budget, Loaded: loaded})
// short on memory: famruntime.RequestMemory(fam); on family.TopicMemory unload models idle for famruntime.UnloadIdleAfter
// at quit: famruntime.Remove(fam, bundleID)
```

(`famruntime` is `github.com/hehehai/actnowkit/runtime` imported under another name.)

### GitHub release downloads

Many users in China get GitHub's release files at a few KB/s.
`https://mirror.actnow.dev/github` (Cloudflare R2, synced every 30 minutes,
signatures and checksums checked when syncing) answers GitHub's paths for
the releases the apps need, so an app only changes the base URL (its
`/releases/latest` is JSON `{"tag"}` instead of a redirect; `Latest`
reads both). The
mirror is not trusted: the app checks what it gets exactly as before.

```go
c := &mirror.Client{Sources: mirror.Sources(hfEndpoints), HTTP: httpClient}
tag, _, err := c.Latest(ctx, "yt-dlp/yt-dlp", validTag)          // sources raced
sums, from, err := mirror.Try(ctx, c, func(ctx context.Context, base string) ([]byte, error) {
	// a checksum file and its signature, from one source, checked together
})
file, from, err := c.File(ctx, "/yt-dlp/yt-dlp/releases/download/"+tag+"/yt-dlp_macos.zip",
	mirror.FileOptions{Dir: dir, SHA256: sum, Progress: report})
```

`mirror.Sources` reads the family's download source (`network.hfEndpoints`):
automatic → github.com and the mirror, `official` → github.com,
`mirror` → the mirror; `$ACTNOW_GITHUB_SOURCES` replaces them. `File`
starts every source at once and keeps the first to deliver 256 KB; a
source that then stalls for 20 s, stays under 32 KB/s for 30 s, fails or
serves a file whose SHA-256 differs is left and the others race again
(a stalled or crawling one once more, on a new connection).
`TextSourceAuto`, `TextSourceOfficial`, `TextSourceMirror` and
`TextSourceNote` are the setting's reference wordings.

### Cue sounds

`sounds` plays the family's cue sounds and models their settings; the
picker's UI is the app's. The sounds themselves are not in actnowkit: the
family's library is a private repository (its audio comes from React Bits
Sound Lab, licensed for use inside apps but not for redistribution on its
own), and each app copies the sounds it offers into its bundle
(`go run github.com/hehehai/actnowkit/cmd/actnow-sounds subset -from … -to
resources/sounds -groups … -uses …`). A build without them (a source build
of an open-source app) passes a nil catalogue and plays every event's
`Fallback` system sound.

```go
cat, err := sounds.Load(resources, "sounds")   // nil and ErrNoCatalog when the build has none
player := sounds.NewPlayer(cat, nil)            // nil: DefaultOutput (macOS: AVAudioEngine; elsewhere silent)

var recordStart = sounds.Event{ID: "record.start", Name: sounds.Text{En: "Start recording", Zh: "开始录音"},
	Default: "tally", Fallback: "Pop", Uses: []string{"record.start"}}

prefs := sounds.DefaultPrefs()                  // {On: true, Volume: 70}: kept in the app's own settings
player.Preload(prefs.Choice(recordStart))       // at start-up, off the main thread
player.PlayEvent(recordStart, prefs)            // returns at once
t := prefs.Target(cat, recordStart)             // what it plays; t.Length(cat) = how long it is heard

// The picker
f := sounds.Filter{Group: "notify", Tags: []string{"soft"}, Query: "bell"}
rows := cat.Filter(f)                           // + cat.GroupCounts(f), cat.TagCounts(f, true) for the chips
suggested := cat.Suggested(recordStart)         // its default, then the sounds of its uses
player.Preview(row.ID, "", prefs.Volume)        // ▶: cuts off the last preview
ch, left := player.Previewing()                 // ▶ in the accent on ch, redraw after left; StopPreview stops it
step := sounds.NearestVolumeStep(prefs.Volume)  // the volume menu's checked step (VolumeSteps), VolumeLevel its icon
prefs.Choose(recordStart, row.ID)               // or sounds.None, or sounds.System("Glass")
player.Preview(prefs.Choice(recordStart), recordStart.Fallback, prefs.Volume) // "Test"
```

A choice is `""` (the default), `"none"`, `"system:<name>"` or a sound id.
Migrating an old setting that named a system sound: `MigrateSystem(ev,
name, wasDefault)` gives users who never changed it the new default and
keeps an explicit choice as `system:<name>`; `MigrateClosest(cat, name)`
maps it to the sound that replaces it (`Sound.Replaces`). On macOS the
catalogue's WAV files are decoded once (`DecodeWAV`) into an AVAudioEngine
with 8 player nodes: a start takes microseconds and quick repeats overlap;
the engine stops after 5 s without a sound. Other files use AVAudioPlayer,
system sounds NSSound. Tests use `sounds.Recorder`.

### Log selection

`listsel.Model[K]` is the selection behind every app's log view, so the
same gestures do the same thing everywhere. `K` is a stable key of a row
(the log's sequence number), never its place in the list. The view reports
the primary button's press, moves and release with the row under the
pointer, and draws `IsSelected`:

| Gesture | What happens |
| --- | --- |
| Click (no modifier, no drag) | `Release` returns `CopyRow`: copy that line. The selection clears; the row becomes the anchor, not shown selected. |
| Shift-click | The rows from the anchor to this one, in display order. |
| ⌘-click (Ctrl elsewhere) | Add the row or take it out; it becomes the anchor. |
| Press and move more than `DragSlop` (4 DIPs) | Select from the row pressed to the row under the pointer, up or down; with ⌘ held, add them. |
| ⌘A / Esc | `SelectAll(shown)` / `Clear()` (the anchor stays). |
| Right-click, Control-click on macOS | The context menu (the app's): not reported to the Model. |

```go
var sel listsel.Model[uint64]          // with the view's state

sel.Prune(shownKeys)                   // every build: lines filtered out or rotated away leave the selection
// pointer down (primary) on a row:
sel.Press(row.Seq, x, y, mods)         // mods: listsel.Extend (Shift) | listsel.Toggle (⌘)
// pointer moved while down:
if sel.Move(x, y, rowUnder, overRow, shownKeys) { redraw() }
// pointer up:
if in, k := sel.Release(shownKeys); in == listsel.CopyRow { copyLines(k) }
// Esc while dragging: sel.Cancel(); else sel.Clear()

picked := listsel.Items(&sel, shown, func(e Entry) uint64 { return e.Seq })
clipboard(listsel.Text(picked, format)) // lines end in "\n"; a single line has none
sel.Clear()                            // copying clears the selection
```

`listsel.TextSelected`, `TextCopy`, `TextDeselect`, `TextCopyLine`,
`TextCopySelected`, `TextCopyShown` and `TextCopied` are the reference
wordings (English and Chinese) for the apps' catalogues. The family's
interaction spec (header, context menu, colours, tests) is VoxtNew's
reference implementation.

## Conformance

Run the scenario suite against your app's own implementation by wrapping it
in an adapter:

```go
func TestFamilyConformance(t *testing.T) {
	conformance.Run(t, func(t testing.TB, fx *conformance.Fixture, bundleID string) any {
		return newAdapter(fx.Family.Root, fx.Hub, bundleID) // implements conformance.ModelApp, CatalogApp, Downloader, SettingsApp…
	})
}
```

The scenarios cover model reuse without downloading, deletion protection
(claims, locks, loaded models), repos shared by several models of one app, deleting for all apps (tombstones),
concurrent and resumed downloads against another app, settings that follow
the family, shortcut conflicts, cleaning up after an uninstalled app, and
forward compatibility. `conformance.KitApp` is the adapter of actnowkit
itself. `conformance/testdata` has a sample of every file with the result a
reader must reach (`cases.json`), for implementations in any language.

On a Mac, `go run github.com/hehehai/actnowkit/cmd/actnow-conformance check`
lists problems in the real family directory.

## Development

```sh
go test -race ./...                          # Linux or macOS
ACTNOWKIT_MAC=1 go test -race ./...          # macOS: also the real notify, LaunchServices, security, scutil, audio (quiet)
ACTNOW_SOUNDS_DIR=../actnow-sounds go test ./sounds   # check a real sound catalogue
go vet ./... && staticcheck ./...
```

Tests never touch the real `~/Library/Application Support/ActNow` or
`~/.cache/huggingface`: they use temporary directories (the Keychain tests a
temporary keychain file).

## Licence

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
