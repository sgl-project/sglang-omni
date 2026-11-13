package conformance

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/hfhub"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/hehehai/actnowkit/settings"
	"github.com/hehehai/actnowkit/shortcuts"
)

// The interfaces the scenarios drive the app under test through. An
// adapter wraps the app's own code (not actnowkit's, unless the app uses
// it) and implements what the app has.

// ModelApp is an app with local models.
type ModelApp interface {
	BundleID() string
	// Reuse is "Use them" on the first-run reuse card: register installed
	// models as used by the app (claims with origin "reused"), downloading
	// nothing.
	Reuse(ctx context.Context, refs ...models.Ref) error
	// Remove is the user deleting a model in the app; allApps is "Delete
	// for all apps…" (after its confirmation). It returns an error when
	// it could not (another app downloads the model).
	Remove(ctx context.Context, ref models.Ref, allApps bool) error
}

// CatalogApp is an app whose catalogue entries can share repos (an
// auxiliary tokenizer used by two models). Claims then name every model
// using the repo (spec §7.3 "models").
type CatalogApp interface {
	BundleID() string
	// UseModel registers installed repos (a model's main repo and its
	// auxiliary repos) as used by one catalogue model, as "Use them" or a
	// finished download does.
	UseModel(ctx context.Context, model string, refs ...models.Ref) error
	// RemoveModel is "Remove from <app>" for one catalogue model: each of
	// its repos is released, and deleted unless something (another app,
	// or another model of this app) still uses it.
	RemoveModel(ctx context.Context, model string, refs ...models.Ref) error
}

// Downloader is an app that downloads models.
type Downloader interface {
	BundleID() string
	// Download fetches ref from the Hugging Face endpoint into $HUB and
	// claims it.
	Download(ctx context.Context, endpoint string, ref models.Ref) error
}

// OrphanCleaner is an app with the storage page's orphan clean-up.
type OrphanCleaner interface {
	// CleanOrphans deletes what only the uninstalled app bundleID used,
	// and its files in $FAMILY.
	CleanOrphans(ctx context.Context, bundleID string) error
}

// SettingsApp is an app with the shared settings.
type SettingsApp interface {
	BundleID() string
	// Set changes a shared setting as the user would in the app (written
	// to the family while the app follows it).
	Set(ctx context.Context, key, value string) error
	// StopFollowing is "Use a different setting for this app".
	StopFollowing(ctx context.Context, key string) error
	// Effective is the value the app uses now, after re-reading the family
	// file (as on activation).
	Effective(ctx context.Context, key string) (string, error)
}

// ShortcutApp is an app with global shortcuts.
type ShortcutApp interface {
	BundleID() string
	// Declare sets the app's shortcuts and writes its manifest.
	Declare(ctx context.Context, sc []shortcuts.Shortcut) error
	// Conflicts lists the bundle ids of installed apps with a shortcut
	// colliding with one of the app's.
	Conflicts(ctx context.Context) ([]string, error)
}

// Factory makes the adapter of the app under test with this bundle id,
// working in fx's directories.
type Factory func(t testing.TB, fx *Fixture, bundleID string) any

// Run runs the scenarios against the app the factory makes. Each runs in
// a fresh fixture.
func Run(t *testing.T, newApp Factory) {
	for _, sc := range Scenarios {
		t.Run(sc.Name, func(t *testing.T) {
			fx := New(t)
			sc.Run(t, fx, newApp)
		})
	}
}

// Scenario is one check.
type Scenario struct {
	Name string
	Spec string // the spec section or §16.1 scenario
	Run  func(t testing.TB, fx *Fixture, newApp Factory)
}

func timeout(t testing.TB) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func as[T any](t testing.TB, v any) T {
	t.Helper()
	x, ok := v.(T)
	if !ok {
		var zero T
		t.Skipf("the app does not implement %T", &zero)
	}
	return x
}

// Repos the scenarios use.
var (
	repoASR = TestRepo("mlx-community/Qwen3-ASR-0.6B-4bit", "313d850181767edf09f00a9c289becca70e58cd0", 300<<10)
	repoLLM = TestRepo("mlx-community/gemma-4-e4b-it-4bit", "475b9088d29754a3379866cf5aeb6b41acd313c2", 200<<10)
	repoBig = TestRepo("mlx-community/whisper-large-v3-turbo-4bit", "0f058d38170d183f9fdee07908f5b515d91793a8", 3<<20)
)

// Scenarios are the automatable checks of spec §16.1 and §5.1.
var Scenarios = []Scenario{
	{Name: "ReuseWithoutDownload", Spec: "§16.1 scenario 1, §7.10.1", Run: reuse},
	{Name: "DeleteKeepsSharedModel", Spec: "§16.1 scenario 2, §7.6", Run: deleteShared},
	{Name: "DeleteUnsharedModel", Spec: "§7.6 step 1.5", Run: deleteUnshared},
	{Name: "DeleteWithLostClaims", Spec: "§7.6 (v1.0.9: lost claims are unsure)", Run: deleteLostClaims},
	{Name: "DeleteForAllApps", Spec: "§7.6 step 1.6", Run: deleteForAll},
	{Name: "SharedRepoWithinApp", Spec: "§7.3 models, §7.6", Run: sharedWithinApp},
	{Name: "DeleteWhileDownloading", Spec: "§7.4", Run: deleteBusy},
	{Name: "DeleteWhileLoaded", Spec: "§7.6 step 1.4, §11.1", Run: deleteLoaded},
	{Name: "ConcurrentDownload", Spec: "§16.1 scenario 3, §7.4", Run: concurrentDownload},
	{Name: "ResumeOtherAppsDownload", Spec: "§16.1 scenario 4, §7.1", Run: resumeDownload},
	{Name: "SettingsFollowFamily", Spec: "§16.1 scenario 5, §9.3", Run: settingsFollow},
	{Name: "ShortcutConflict", Spec: "§16.1 scenario 7, §10.4", Run: shortcutConflict},
	{Name: "UninstalledAppCleanup", Spec: "§16.1 scenario 8, §7.6 step 4", Run: orphans},
	{Name: "ForwardCompatibleClaims", Spec: "§16.1 scenario 9, §5.1", Run: forwardClaims},
	{Name: "ForwardCompatibleSettings", Spec: "§16.1 scenario 9, §5.1", Run: forwardSettings},
}

func reuse(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Register(PeerSmith)
	fx.Install(repoASR)
	peer := fx.Peer(PeerSmith)
	must(t, peer.Claim(ctx, models.Claim{Repo: repoASR.Ref.Repo, Revision: repoASR.Ref.Revision, Origin: models.OriginDownloaded}))
	before := snapshotState(t, fx.Hub)
	if err := app.Reuse(ctx, repoASR.Ref); err != nil {
		t.Fatalf("Reuse: %v", err)
	}
	c, ok := fx.Claims(app.BundleID()).Find(repoASR.Ref)
	if !ok {
		t.Fatalf("the app did not claim %s", repoASR.Ref)
	}
	if c.Origin != models.OriginReused {
		t.Errorf("origin %q, want %q", c.Origin, models.OriginReused)
	}
	if root := fx.Claims(app.BundleID()).Root; !models.SameHub(root, fx.Hub) {
		t.Errorf("claims root %q, want %q", root, fx.Hub)
	}
	if after := snapshotState(t, fx.Hub); after != before {
		t.Errorf("reusing changed the model files")
	}
	if !fx.Claimed(PeerSmith, repoASR.Ref) {
		t.Error("the peer's claim is gone")
	}
}

// snapshotState fingerprints the blobs (names and sizes) under $HUB.
func snapshotState(t testing.TB, hub string) string {
	var b strings.Builder
	filepath.Walk(hub, func(p string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && strings.Contains(p, string(filepath.Separator)+"blobs"+string(filepath.Separator)) {
			b.WriteString(p)
			b.WriteString(":")
			b.WriteString(info.ModTime().String())
			b.WriteString("\n")
		}
		return nil
	})
	return b.String()
}

func deleteShared(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Register(PeerSmith)
	fx.Install(repoASR)
	peer := fx.Peer(PeerSmith)
	must(t, peer.Claim(ctx, models.Claim{Repo: repoASR.Ref.Repo, Revision: repoASR.Ref.Revision}))
	must(t, app.Reuse(ctx, repoASR.Ref))
	if err := app.Remove(ctx, repoASR.Ref, false); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if !fx.Installed(repoASR.Ref) {
		t.Fatal("the app deleted a model another app claims")
	}
	if fx.Claimed(app.BundleID(), repoASR.Ref) {
		t.Error("the app still claims the model it removed")
	}
	// The peer removes it too: nobody uses it, the files go.
	res, err := peer.Delete(ctx, repoASR.Ref, models.ThisApp)
	if err != nil || !res.Deleted {
		t.Fatalf("the peer could not delete the model after the app released it: %+v %v", res, err)
	}
}

// deleteLostClaims: the peer's claims file is gone (a cleaner deleted the
// family folder's models/claims) while its manifest says it keeps models
// in this $HUB: the app must not take that as "nobody uses it".
func deleteLostClaims(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Register(PeerSmith)
	fx.Install(repoASR)
	must(t, fx.Peer(PeerSmith).Claim(ctx, models.Claim{Repo: repoASR.Ref.Repo, Revision: repoASR.Ref.Revision}))
	must(t, app.Reuse(ctx, repoASR.Ref))
	must(t, os.Remove(fx.Family.ClaimsFile(PeerSmith)))
	if err := app.Remove(ctx, repoASR.Ref, false); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if !fx.Installed(repoASR.Ref) {
		t.Fatal("the app deleted a model while another app's claims were missing (lost claims must count as unsure)")
	}
}

func deleteUnshared(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Install(repoASR)
	must(t, os.MkdirAll(models.EngineCopy(fx.Hub, repoASR.Ref.Repo), 0o755))
	must(t, app.Reuse(ctx, repoASR.Ref))
	if err := app.Remove(ctx, repoASR.Ref, false); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := os.Stat(models.RepoDir(fx.Hub, repoASR.Ref.Repo)); !os.IsNotExist(err) {
		t.Error("the repo directory stayed though nobody uses the model")
	}
	if _, err := os.Stat(models.EngineCopy(fx.Hub, repoASR.Ref.Repo)); !os.IsNotExist(err) {
		t.Error("the mlx-audio copy stayed after the last snapshot went (§7.8)")
	}
	if _, err := os.Stat(models.RepoLockFile(fx.Hub, repoASR.Ref.Repo)); err != nil {
		t.Error("the repo lock file was removed or never used (lock files are never deleted, §5.3)")
	}
}

func deleteForAll(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Register(PeerSmith)
	fx.Install(repoLLM)
	peer := fx.Peer(PeerSmith)
	must(t, peer.Claim(ctx, models.Claim{Repo: repoLLM.Ref.Repo, Revision: repoLLM.Ref.Revision}))
	must(t, app.Reuse(ctx, repoLLM.Ref))
	if err := app.Remove(ctx, repoLLM.Ref, true); err != nil {
		t.Fatalf("Remove for all apps: %v", err)
	}
	if fx.Installed(repoLLM.Ref) {
		t.Fatal("files kept after Delete for all apps")
	}
	ts, err := models.ReadTombstones(fx.Family)
	must(t, err)
	if !slices.ContainsFunc(ts, func(x models.Tombstone) bool { return x.Ref() == repoLLM.Ref && x.RemovedBy == app.BundleID() }) {
		t.Errorf("no tombstone by the app in removed.json: %+v", ts)
	}
	if !fx.Claimed(PeerSmith, repoLLM.Ref) {
		t.Error("the app edited the peer's claims file (it must only write a tombstone)")
	}
	hit, err := peer.ApplyTombstones()
	if err != nil || len(hit) != 1 {
		t.Errorf("the peer did not see the tombstone: %v %v", hit, err)
	}
}

func sharedWithinApp(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[CatalogApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	tok := TestRepo("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", 4<<10)
	fx.Install(repoASR)
	fx.Install(repoLLM)
	fx.Install(tok)
	must(t, app.UseModel(ctx, "fast", repoASR.Ref, tok.Ref))
	must(t, app.UseModel(ctx, "slow", repoLLM.Ref, tok.Ref))
	c, ok := fx.Claims(app.BundleID()).Find(tok.Ref)
	if !ok || !slices.Equal(c.Users(), []string{"fast", "slow"}) {
		t.Fatalf("the shared repo's claim names %v (%v), want both models (\"models\": [\"fast\", \"slow\"])", c.Users(), ok)
	}
	must(t, app.RemoveModel(ctx, "fast", repoASR.Ref, tok.Ref))
	if fx.Installed(repoASR.Ref) {
		t.Error("the removed model's own repo stayed though nothing uses it")
	}
	if !fx.Installed(tok.Ref) {
		t.Fatal("removing one model deleted a repo another model of the same app uses")
	}
	c, ok = fx.Claims(app.BundleID()).Find(tok.Ref)
	if !ok || !slices.Equal(c.Users(), []string{"slow"}) {
		t.Errorf("after removing \"fast\", the shared repo's claim names %v (%v), want [slow]", c.Users(), ok)
	}
	must(t, app.RemoveModel(ctx, "slow", repoLLM.Ref, tok.Ref))
	if fx.Installed(tok.Ref) || fx.Claimed(app.BundleID(), tok.Ref) {
		t.Error("the shared repo stayed after its last model went")
	}
}

func deleteBusy(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Install(repoASR)
	must(t, app.Reuse(ctx, repoASR.Ref))
	lock, err := models.TryLockRepo(fx.Hub, repoASR.Ref.Repo, family.Shared, models.Owner{BundleID: PeerSmith, Op: models.OpDownload})
	must(t, err)
	defer lock.Release()
	done := make(chan error, 1)
	go func() { done <- app.Remove(ctx, repoASR.Ref, false) }()
	select {
	case err := <-done:
		if err == nil && !fx.Installed(repoASR.Ref) {
			t.Fatal("the app deleted a model while another app held its repo lock")
		}
		if err == nil {
			t.Log("Remove returned no error; files kept")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Remove waited for the repo lock: deleting must not queue (§7.4)")
	}
	if !fx.Installed(repoASR.Ref) {
		t.Fatal("files deleted while locked")
	}
	if !fx.Claimed(app.BundleID(), repoASR.Ref) {
		t.Error("the app's claim was not restored after the failed delete (§7.6 step 1.2)")
	}
}

func deleteLoaded(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Register(PeerSmith)
	fx.Install(repoASR)
	must(t, app.Reuse(ctx, repoASR.Ref))
	must(t, famruntime.Write(fx.Family, famruntime.File{BundleID: PeerSmith, Loaded: []famruntime.Loaded{{Repo: repoASR.Ref.Repo, Revision: repoASR.Ref.Revision, Bytes: 1 << 20}}}))
	defer famruntime.Remove(fx.Family, PeerSmith)
	if err := app.Remove(ctx, repoASR.Ref, false); err != nil {
		t.Logf("Remove: %v", err)
	}
	if !fx.Installed(repoASR.Ref) {
		t.Fatal("the app deleted a model a running app has loaded (runtime file)")
	}
}

func concurrentDownload(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[Downloader](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	hub := NewFakeHub(repoBig)
	defer hub.Close()
	reached, release := hub.Pause(repoBig.Ref.Repo, "model.safetensors", 1<<20)
	peer := hfhub.New(hfhub.Options{Store: fx.Peer(PeerSmith), Endpoints: []string{hub.URL}, ReserveBytes: -1})
	peerErr := make(chan error, 1)
	go func() {
		peerErr <- peer.Download(ctx, hfhub.Entry{ID: "whisper", Repo: repoBig.Ref.Repo, Revision: repoBig.Ref.Revision}, nil)
	}()
	select {
	case <-reached:
	case err := <-peerErr:
		t.Fatalf("the peer finished early: %v", err)
	case <-ctx.Done():
		t.Fatal("the peer never started")
	}
	appErr := make(chan error, 1)
	go func() { appErr <- app.Download(ctx, hub.URL, repoBig.Ref) }()
	time.Sleep(300 * time.Millisecond) // the app reaches the blob lock and waits
	release()
	if err := <-peerErr; err != nil {
		t.Fatalf("peer download: %v", err)
	}
	if err := <-appErr; err != nil {
		t.Fatalf("app download: %v", err)
	}
	if !fx.Installed(repoBig.Ref) {
		t.Fatal("not installed after both downloads")
	}
	if n, size := hub.Served(), repoBig.Size(); n > size+64<<10 {
		t.Errorf("served %d bytes for a %d-byte model: the app downloaded what the peer was downloading (blob locks, §7.4)", n, size)
	}
	if !fx.Claimed(app.BundleID(), repoBig.Ref) || !fx.Claimed(PeerSmith, repoBig.Ref) {
		t.Error("both apps must claim the model")
	}
	if left := fx.Incomplete(repoBig.Ref.Repo); len(left) > 0 {
		t.Errorf("partial files left: %v", left)
	}
}

func resumeDownload(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[Downloader](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	hub := NewFakeHub(repoBig)
	defer hub.Close()
	const cut = 2 << 20
	reached, release := hub.Pause(repoBig.Ref.Repo, "model.safetensors", cut)
	peerCtx, cancelPeer := context.WithCancel(ctx)
	peer := hfhub.New(hfhub.Options{Store: fx.Peer(PeerSmith), Endpoints: []string{hub.URL}, ReserveBytes: -1})
	peerErr := make(chan error, 1)
	go func() {
		peerErr <- peer.Download(peerCtx, hfhub.Entry{ID: "whisper", Repo: repoBig.Ref.Repo, Revision: repoBig.Ref.Revision}, nil)
	}()
	select {
	case <-reached:
	case <-ctx.Done():
		t.Fatal("the peer never reached the pause")
	}
	time.Sleep(100 * time.Millisecond) // let it write what it got
	cancelPeer()                       // the peer quits mid-download (its locks go with it)
	<-peerErr
	release()
	partial := hub.ServedOf(repoBig.Ref.Repo, "model.safetensors")
	if err := app.Download(ctx, hub.URL, repoBig.Ref); err != nil {
		t.Fatalf("app download: %v", err)
	}
	if !fx.Installed(repoBig.Ref) {
		t.Fatal("not installed")
	}
	again := hub.ServedOf(repoBig.Ref.Repo, "model.safetensors") - partial
	weights := int64(len(repoBig.Files[2].Content))
	if again > weights-partial+64<<10 {
		t.Errorf("the app fetched %d bytes of weights after the peer had %d of %d: it did not resume the partial blob (.incomplete, §7.1)", again, partial, weights)
	}
	if left := fx.Incomplete(repoBig.Ref.Repo); len(left) > 0 {
		t.Errorf("partial files left: %v", left)
	}
}

func settingsFollow(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[SettingsApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	// The app changes the proxy: the family file has it.
	must(t, app.Set(ctx, settings.KeyProxy, "http://127.0.0.1:7897"))
	fam, _, err := settings.Read(fx.Family)
	must(t, err)
	if v, _ := fam.Get(settings.KeyProxy); v != "http://127.0.0.1:7897" {
		t.Fatalf("settings.json network.proxy = %q after the app set it while following", v)
	}
	if fam.Rev < 1 || fam.UpdatedBy != app.BundleID() {
		t.Errorf("rev %d, updatedBy %q", fam.Rev, fam.UpdatedBy)
	}
	// Another app changes the language: the app uses it.
	_, err = settings.Set(ctx, fx.Family, PeerAuralink, settings.KeyLanguage, "zh-Hans")
	must(t, err)
	if v, err := app.Effective(ctx, settings.KeyLanguage); err != nil || v != "zh-Hans" {
		t.Errorf("effective language %q (%v) after another app set zh-Hans", v, err)
	}
	// The app stops following the proxy: others' changes no longer apply.
	must(t, app.StopFollowing(ctx, settings.KeyProxy))
	_, err = settings.Set(ctx, fx.Family, PeerAuralink, settings.KeyProxy, settings.ProxyDirect)
	must(t, err)
	if v, err := app.Effective(ctx, settings.KeyProxy); err != nil || v == settings.ProxyDirect {
		t.Errorf("effective proxy %q (%v): the app follows the family after StopFollowing", v, err)
	}
	// And its own changes stay its own.
	must(t, app.Set(ctx, settings.KeyProxy, "socks5://127.0.0.1:1080"))
	fam, _, _ = settings.Read(fx.Family)
	if v, _ := fam.Get(settings.KeyProxy); v != settings.ProxyDirect {
		t.Errorf("the app wrote its own proxy to the family file (%q)", v)
	}
}

func shortcutConflict(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ShortcutApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Register(PeerAuralink, shortcuts.DefaultsOf("auralink")...)
	must(t, app.Declare(ctx, []shortcuts.Shortcut{{ID: "dictation", Title: "Dictation", Keys: "Fn", Trigger: shortcuts.Hold, Enabled: true, Mechanism: shortcuts.EventTap}}))
	got, err := app.Conflicts(ctx)
	must(t, err)
	if !slices.Contains(got, PeerAuralink) {
		t.Errorf("Fn vs Auralink's Fn: conflicts %v", got)
	}
	must(t, app.Declare(ctx, []shortcuts.Shortcut{{ID: "dictation", Title: "Dictation", Keys: "RightCommand+Space", Trigger: shortcuts.Press, Enabled: true, Mechanism: shortcuts.Carbon}}))
	got, err = app.Conflicts(ctx)
	must(t, err)
	if len(got) != 0 {
		t.Errorf("no shared keys, but conflicts %v", got)
	}
	m, err := readManifest(fx, app.BundleID())
	if err != nil || len(m.Shortcuts) != 1 || m.Shortcuts[0].Keys != "RightCommand+Space" {
		t.Errorf("the manifest does not declare the shortcut in canonical form: %+v %v", m.Shortcuts, err)
	}
}

func orphans(t testing.TB, fx *Fixture, newApp Factory) {
	v := newApp(t, fx, AppUnderTest)
	app := as[ModelApp](t, v)
	cleaner := as[OrphanCleaner](t, v)
	ctx := timeout(t)
	fx.Register(PeerGone)
	fx.Install(repoASR)
	fx.Install(repoLLM)
	gone := fx.Peer(PeerGone)
	must(t, gone.Claim(ctx, models.Claim{Repo: repoASR.Ref.Repo, Revision: repoASR.Ref.Revision}, models.Claim{Repo: repoLLM.Ref.Repo, Revision: repoLLM.Ref.Revision}))
	must(t, app.Reuse(ctx, repoASR.Ref))
	fx.Uninstall(PeerGone)
	if err := cleaner.CleanOrphans(ctx, PeerGone); err != nil {
		t.Fatalf("CleanOrphans: %v", err)
	}
	if fx.Installed(repoLLM.Ref) {
		t.Error("a model only the uninstalled app used was kept")
	}
	if !fx.Installed(repoASR.Ref) {
		t.Error("the app deleted a model it uses itself")
	}
	for _, p := range []string{fx.Family.ClaimsFile(PeerGone), fx.Family.AppFile(PeerGone)} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s left after the clean-up", p)
		}
	}
}

func forwardClaims(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[ModelApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	fx.Install(repoASR)
	fx.Install(repoLLM)
	path := fx.Family.ClaimsFile(app.BundleID())
	future := `{"schema":"actnow.models.claims/1","app":"dicta","bundleId":"` + app.BundleID() + `","updated":"2026-10-20T08:00:00Z","root":` + quote(fx.Hub) + `,"futureTop":{"x":1},
"claims":[{"repo":"` + repoLLM.Ref.Repo + `","revision":"` + repoLLM.Ref.Revision + `","files":null,"aux":false,"since":"2026-10-20T08:00:00Z","origin":"downloaded","futureClaim":"keep"}]}`
	must(t, os.WriteFile(path, []byte(future), 0o600))
	must(t, app.Reuse(ctx, repoASR.Ref))
	data, err := os.ReadFile(path)
	must(t, err)
	for _, want := range []string{`"futureTop"`, `"futureClaim"`, repoASR.Ref.Repo, repoLLM.Ref.Repo} {
		if !strings.Contains(string(data), want) {
			t.Errorf("after the app's rewrite, the claims file lacks %s:\n%s", want, data)
		}
	}
	// A newer major: never written back.
	v2 := `{"schema":"actnow.models.claims/2","app":"dicta","bundleId":"` + app.BundleID() + `","root":` + quote(fx.Hub) + `,"claims":[]}`
	must(t, os.WriteFile(path, []byte(v2), 0o600))
	_ = app.Reuse(ctx, repoLLM.Ref)
	if data, _ := os.ReadFile(path); string(data) != v2 {
		t.Errorf("the app rewrote a claims file of a newer major:\n%s", data)
	}
}

func forwardSettings(t testing.TB, fx *Fixture, newApp Factory) {
	app := as[SettingsApp](t, newApp(t, fx, AppUnderTest))
	ctx := timeout(t)
	future, err := File("settings-future.json")
	must(t, err)
	must(t, os.WriteFile(fx.Family.SettingsFile(), future, 0o600))
	if v, err := app.Effective(ctx, settings.KeyHFEndpoints); err != nil || v != "mirror" {
		t.Errorf("effective endpoints %q %v, want the family's \"mirror\"", v, err)
	}
	must(t, app.Set(ctx, settings.KeyLanguage, "en"))
	data, _ := os.ReadFile(fx.Family.SettingsFile())
	for _, want := range []string{`"theme"`, `"timeoutSeconds"`, `"speech"`, `"experimental"`, `"rev": 100`} {
		if !strings.Contains(strings.ReplaceAll(string(data), `"rev":100`, `"rev": 100`), want) {
			t.Errorf("after the app's write, settings.json lacks %s:\n%s", want, data)
		}
	}
	v2, err := File("settings-v2.json")
	must(t, err)
	must(t, os.WriteFile(fx.Family.SettingsFile(), v2, 0o600))
	_ = app.Set(ctx, settings.KeyLanguage, "zh-Hans")
	if data, _ := os.ReadFile(fx.Family.SettingsFile()); string(data) != string(v2) {
		t.Errorf("the app rewrote settings.json of a newer major:\n%s", data)
	}
}

func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func readManifest(fx *Fixture, bundleID string) (registry.Manifest, error) {
	return registry.Read(fx.Family, bundleID)
}
