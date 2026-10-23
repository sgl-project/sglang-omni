package models

// The Manager against the family's protocol: a temporary $FAMILY and
// $HUB (conformance.New), conformance's fake Hub, and other apps played
// by actnowkit (fx.Peer).

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
)

const (
	voxtnew    = "dev.hehehai.voxtnew"
	audiosmith = "dev.hehehai.audiosmith"
	dicta      = "dev.hehehai.dicta"
)

var (
	repoSpeech = conformance.TestRepo("mlx-community/test-asr", "1111111111111111111111111111111111111111", 96<<10)
	repoVAD    = conformance.TestRepo("mlx-community/test-vad", "2222222222222222222222222222222222222222", 8<<10)
	repoLLM    = conformance.TestRepo("mlx-community/test-llm", "3333333333333333333333333333333333333333", 64<<10)
)

// repoModel is a catalogue model of a fixture repo, its files pinned.
func repoModel(id string, kind Kind, r conformance.Repo) Model {
	m := Model{ID: id, Kind: kind, Runtime: Omni, Repo: r.Ref.Repo, Revision: r.Ref.Revision, Family: "test"}
	for _, f := range r.Files {
		sum := sha256.Sum256(f.Content)
		m.Files = append(m.Files, File{Path: f.Path, Size: int64(len(f.Content)), SHA256: hex.EncodeToString(sum[:]), LFS: f.LFS})
	}
	return m
}

// testCatalog: a speech model that needs a VAD, and a language model.
func testCatalog() []Model {
	asr := repoModel("asr", ASR, repoSpeech)
	asr.Needs = []string{"vad"}
	return []Model{asr, repoModel("vad", VAD, repoVAD), repoModel("llm", LLM, repoLLM)}
}

type testEnv struct {
	fx  *conformance.Fixture
	hub *conformance.FakeHub
	m   *Manager
}

// newEnv is VoxtNew's Manager over a fixture, AudioSmith and Dicta
// installed beside it, downloading from a fake Hub.
func newEnv(t *testing.T, cat []Model, opt Options) *testEnv {
	fx := conformance.New(t)
	fx.Register(audiosmith)
	fx.Register(dicta)
	h := conformance.NewFakeHub(repoSpeech, repoVAD, repoLLM)
	t.Cleanup(h.Close)
	opt.Store, opt.Catalog = fx.Peer(voxtnew), cat
	opt.Network.Endpoints = []string{h.URL}
	if opt.ReserveBytes == 0 {
		opt.ReserveBytes = -1
	}
	m := Open(opt)
	t.Cleanup(m.Close)
	return &testEnv{fx: fx, hub: h, m: m}
}

func ctxT(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// wait waits for model id's download to end.
func wait(t *testing.T, m *Manager, id string) Status {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if s := m.Status(id); !s.Downloading {
			return s
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("%s: still downloading: %+v", id, m.Status(id))
	return Status{}
}

func (e *testEnv) claim(t *testing.T, bundleID string, r conformance.Repo) {
	t.Helper()
	if err := e.fx.Peer(bundleID).Claim(ctxT(t), fammodels.Claim{Repo: r.Ref.Repo, Revision: r.Ref.Revision, Origin: fammodels.OriginDownloaded}); err != nil {
		t.Fatal(err)
	}
}

// TestDownloadClaims: a speech model downloads with its VAD into $HUB,
// both claimed for VoxtNew, and Dir is the model's snapshot.
func TestDownloadClaims(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	if err := e.m.Download("asr"); err != nil {
		t.Fatal(err)
	}
	if s := wait(t, e.m, "asr"); !s.Installed || !s.Mine || s.Err != nil || s.Revision != repoSpeech.Ref.Revision {
		t.Fatalf("%+v", s)
	}
	if !e.m.Installed("vad") || !e.fx.Installed(repoSpeech.Ref) || !e.fx.Installed(repoVAD.Ref) {
		t.Fatal("not installed")
	}
	asr, _ := e.m.Lookup("asr")
	if e.m.Dir("asr") != fammodels.SnapshotDir(e.fx.Hub, repoSpeech.Ref) || Installed(e.fx.Hub, asr) != e.m.Dir("asr") {
		t.Errorf("dir %s", e.m.Dir("asr"))
	}
	for _, r := range []conformance.Repo{repoSpeech, repoVAD} {
		c, ok := e.fx.Claims(voxtnew).Find(r.Ref)
		if !ok || c.Origin != fammodels.OriginDownloaded || c.Role == "" || len(c.Files) != 3 {
			t.Errorf("claim of %s: %+v %v", r.Ref, c, ok)
		}
	}
	if got, want := e.hub.Served(), repoSpeech.Size()+repoVAD.Size(); got != want {
		t.Errorf("served %d bytes, want %d", got, want)
	}
	if e.m.Download("asr") != nil || e.m.Status("asr").Downloading {
		t.Error("an installed model downloads again")
	}
}

// TestReuseWithoutDownload (scenario 1): a model AudioSmith installed is
// on disk for VoxtNew at once, shared with it, and VoxtNew's when "Use
// them" claims it (origin reused), downloading nothing.
func TestReuseWithoutDownload(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	e.fx.Install(repoLLM)
	e.claim(t, audiosmith, repoLLM)
	e.m.Refresh()
	s := e.m.Status("llm")
	if !s.OnDisk || s.Installed || s.Mine || !slices.Equal(s.Users, []string{"AudioSmith"}) {
		t.Fatalf("%+v", s)
	}
	found := e.m.Survey("asr", "llm")
	if len(found) != 1 || found[0].ID != "llm" || !slices.Equal(found[0].By, []string{"AudioSmith"}) || found[0].Bytes != repoLLM.Size() {
		t.Fatalf("survey %+v", found)
	}
	if err := e.m.Use(ctxT(t), "llm", "asr"); err != nil {
		t.Fatal(err)
	}
	c, ok := e.fx.Claims(voxtnew).Find(repoLLM.Ref)
	if !ok || c.Origin != fammodels.OriginReused || c.Model != "llm" || !e.m.Status("llm").Mine || !e.m.Installed("llm") {
		t.Errorf("claim %+v %v", c, ok)
	}
	if e.hub.Served() != 0 {
		t.Errorf("served %d bytes", e.hub.Served())
	}
	if err := e.m.EnsureClaimed(ctxT(t), "llm"); err != nil {
		t.Error(err)
	}
}

// TestDelete (scenario 2): with AudioSmith claiming the model, removing
// it from VoxtNew keeps the files; deleting it for all apps removes them
// and leaves a tombstone, after which AudioSmith's claim goes too. A
// model nobody else uses is deleted.
func TestDelete(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	e.fx.Install(repoLLM)
	e.claim(t, audiosmith, repoLLM)
	if err := e.m.Use(ctxT(t), "llm"); err != nil {
		t.Fatal(err)
	}
	d, err := e.m.Decide("llm")
	if err != nil || !d.InUse() || !d.Mine || !slices.Equal(d.UserNames(), []string{"AudioSmith"}) || d.FreeBytes != repoLLM.Size() {
		t.Fatalf("decision %+v %v", d, err)
	}
	res, err := e.m.Delete(ctxT(t), "llm", fammodels.ThisApp)
	if err != nil || len(res) != 1 || res[0].Deleted || !e.fx.Installed(repoLLM.Ref) || e.fx.Claimed(voxtnew, repoLLM.Ref) {
		t.Fatalf("remove from VoxtNew: %+v %v", res, err)
	}
	if s := e.m.Status("llm"); s.Installed || !s.OnDisk || s.Mine {
		t.Errorf("after removing: %+v (another app's model: on disk, not VoxtNew's)", s)
	}
	if err := e.m.Use(ctxT(t), "llm"); err != nil {
		t.Fatal(err)
	}
	res, err = e.m.Delete(ctxT(t), "llm", fammodels.AllApps)
	if err != nil || len(res) != 1 || !res[0].Deleted || !res[0].Tombstoned || e.fx.Installed(repoLLM.Ref) {
		t.Fatalf("delete for all apps: %+v %v", res, err)
	}
	if hit, _ := e.fx.Peer(audiosmith).ApplyTombstones(); len(hit) != 1 {
		t.Errorf("AudioSmith saw %v", hit)
	}
	if s := e.m.Status("llm"); s.Installed || s.OnDisk {
		t.Errorf("%+v", s)
	}
	// Nobody else: the files go at once.
	e.fx.Install(repoSpeech)
	if err := e.m.Use(ctxT(t), "asr"); err != nil {
		t.Fatal(err)
	}
	if d, _ := e.m.Decide("asr"); d.InUse() {
		t.Errorf("in use: %+v", d)
	}
	if res, err := e.m.Delete(ctxT(t), "asr", fammodels.ThisApp); err != nil || len(res) != 1 || !res[0].Deleted {
		t.Fatalf("%+v %v", res, err)
	}
	if _, err := os.Stat(fammodels.RepoDir(e.fx.Hub, repoSpeech.Ref.Repo)); !os.IsNotExist(err) {
		t.Error("the repo stayed")
	}
}

// TestDeleteWhileAnotherDownloads: a repo another app holds fails the
// delete at once, naming it, and keeps VoxtNew's claim.
func TestDeleteWhileAnotherDownloads(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	e.fx.Install(repoLLM)
	if err := e.m.Use(ctxT(t), "llm"); err != nil {
		t.Fatal(err)
	}
	lock, err := fammodels.TryLockRepo(e.fx.Hub, repoLLM.Ref.Repo, family.Shared, fammodels.Owner{BundleID: dicta, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Release()
	_, err = e.m.Delete(ctxT(t), "llm", fammodels.ThisApp)
	if who, ok := Busy(err); !ok || who != dicta {
		t.Fatalf("err %v (%q)", err, who)
	}
	if !e.fx.Installed(repoLLM.Ref) || !e.fx.Claimed(voxtnew, repoLLM.Ref) {
		t.Error("deleted, or the claim is gone")
	}
}

// TestWaitingFor (scenario 3): while Dicta holds a blob's lock, the
// download says it waits for Dicta; once Dicta lets go, it goes on.
func TestWaitingFor(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	weights := repoLLM.Files[2]
	lock, err := fammodels.TryLockBlob(e.fx.Hub, repoLLM.Ref.Repo, weights.BlobID(), fammodels.Owner{BundleID: dicta, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for e.m.Status("llm").WaitingFor != dicta {
		if time.Now().After(deadline) {
			lock.Release()
			t.Fatalf("not waiting: %+v", e.m.Status("llm"))
		}
		time.Sleep(5 * time.Millisecond)
	}
	lock.Release()
	if s := wait(t, e.m, "llm"); !s.Installed || s.Err != nil || s.WaitingFor != "" {
		t.Fatalf("%+v", s)
	}
}

// TestElsewhere: a download another app runs shows on the model while
// VoxtNew does not download it.
func TestElsewhere(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	lock, err := fammodels.TryLockRepo(e.fx.Hub, repoLLM.Ref.Repo, family.Shared, fammodels.Owner{BundleID: audiosmith, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	e.m.Refresh()
	if s := e.m.Status("llm"); s.Elsewhere != audiosmith {
		t.Errorf("%+v", s)
	}
	lock.Release()
	e.m.Refresh()
	if s := e.m.Status("llm"); s.Elsewhere != "" {
		t.Errorf("%+v", s)
	}
}

// TestAccept: a commit the catalogue accepts, on disk instead of the
// pinned one, is used without downloading.
func TestAccept(t *testing.T) {
	old := conformance.TestRepo(repoLLM.Ref.Repo, "4444444444444444444444444444444444444444", 4<<10)
	cat := testCatalog()
	cat[2].Accept = []string{old.Ref.Revision}
	e := newEnv(t, cat, Options{})
	e.fx.Install(old)
	e.m.Refresh()
	if s := e.m.Status("llm"); !s.Installed || s.Revision != old.Ref.Revision || e.m.Dir("llm") != fammodels.SnapshotDir(e.fx.Hub, old.Ref) {
		t.Fatalf("%+v %s", s, e.m.Dir("llm"))
	}
}

// TestBorrowed: another app's model is on offer while on disk, VoxtNew's
// once it is used; VoxtNew never downloads it.
func TestBorrowed(t *testing.T) {
	cat := testCatalog()
	cat[2].Borrowed, cat[2].Files, cat[2].Bytes = true, nil, repoLLM.Size()
	e := newEnv(t, cat, Options{})
	if !errors.Is(e.m.Download("llm"), ErrBorrowed) {
		t.Error("downloads a borrowed model")
	}
	e.fx.Install(repoLLM)
	e.claim(t, dicta, repoLLM)
	if err := e.m.Migrate(ctxT(t)); err != nil {
		t.Fatal(err)
	}
	if s := e.m.Status("llm"); !s.OnDisk || s.Installed || s.Mine || !slices.Equal(s.Users, []string{"Dicta"}) {
		t.Fatalf("migrated a borrowed model: %+v", s)
	}
	if err := e.m.Use(ctxT(t), "llm"); err != nil {
		t.Fatal(err)
	}
	if s := e.m.Status("llm"); !s.Installed || !s.Mine {
		t.Fatalf("%+v", s)
	}
}

// TestMigrate: the first start claims what of the catalogue is on disk
// and no other app claims, origin migrated, reading the disk only;
// another app's model waits for "Use them".
func TestMigrate(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	e.fx.Install(repoSpeech)
	e.fx.Install(repoVAD)
	e.fx.Install(repoLLM)
	e.claim(t, dicta, repoLLM)
	e.m.Refresh()
	for range 2 { // idempotent
		if err := e.m.Migrate(ctxT(t)); err != nil {
			t.Fatal(err)
		}
	}
	cs := e.fx.Claims(voxtnew)
	if len(cs.Claims) != 2 {
		t.Fatalf("%+v", cs)
	}
	for _, c := range cs.Claims {
		if c.Origin != fammodels.OriginMigrated {
			t.Errorf("%+v", c)
		}
	}
	if e.hub.Served() != 0 {
		t.Error("migrating downloaded")
	}
}

// TestCancelResume: a cancelled download keeps its part (Status.Done
// after a Refresh), and the next one fetches only the rest.
func TestCancelResume(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	reached, release := e.hub.Pause(repoLLM.Ref.Repo, "model.safetensors", 32<<10)
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	<-reached
	go func() { time.Sleep(50 * time.Millisecond); release() }()
	e.m.Cancel("llm")
	e.m.Refresh()
	s := e.m.Status("llm")
	if s.Installed || s.Downloading {
		t.Fatalf("%+v", s)
	}
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	if s := wait(t, e.m, "llm"); !s.Installed || s.Err != nil {
		t.Fatalf("%+v", s)
	}
	if served := e.hub.Served(); served > repoLLM.Size()+(64<<10) {
		t.Errorf("served %d bytes for %d", served, repoLLM.Size())
	}
}

// TestDiskSpace: a download that would leave less than the reserve
// fails before fetching, with hfhub's error.
func TestDiskSpace(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{ReserveBytes: 1 << 62})
	if err := e.m.Download("llm"); err != nil {
		t.Fatal(err)
	}
	if s := wait(t, e.m, "llm"); s.Installed || hfhub.Code(s.Err) != hfhub.CodeDiskSpace {
		t.Fatalf("%+v", s)
	}
	if e.hub.Served() != 0 {
		t.Error("fetched")
	}
}

// TestDeletedBy: Dicta deleting a model for all apps takes it from
// VoxtNew; its status says who did.
func TestDeletedBy(t *testing.T) {
	e := newEnv(t, testCatalog(), Options{})
	e.fx.Install(repoLLM)
	if err := e.m.Use(ctxT(t), "llm"); err != nil {
		t.Fatal(err)
	}
	e.claim(t, dicta, repoLLM)
	if _, err := e.fx.Peer(dicta).Delete(ctxT(t), repoLLM.Ref, fammodels.AllApps); err != nil {
		t.Fatal(err)
	}
	e.m.Refresh()
	if s := e.m.Status("llm"); s.Installed || s.Mine || s.DeletedBy != dicta {
		t.Fatalf("%+v", s)
	}
	if e.fx.Claimed(voxtnew, repoLLM.Ref) {
		t.Error("claim kept")
	}
	e.m.Refresh()
	if s := e.m.Status("llm"); s.DeletedBy != dicta {
		t.Errorf("forgot who deleted it: %+v", s)
	}
}
