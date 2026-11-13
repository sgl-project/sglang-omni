package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/family"
	famruntime "github.com/hehehai/actnowkit/runtime"
)

const (
	repoA = "mlx-community/Qwen3-ASR-0.6B-4bit"
	revA  = "313d850181767edf09f00a9c289becca70e58cd0"
	revA2 = "1111111111111111111111111111111111111111"
	repoB = "mlx-community/gemma-4-e4b-it-4bit"
	revB  = "475b9088d29754a3379866cf5aeb6b41acd313c2"
)

// snap writes a snapshot in the HF layout: blobs named by the sha256 of
// their content, links from snapshots/<rev>/<path>, refs/main.
func snap(t *testing.T, hub, repo, rev string, files map[string]string) {
	t.Helper()
	dir := RepoDir(hub, repo)
	for p, content := range files {
		sum := sha256.Sum256([]byte(content))
		blob := hex.EncodeToString(sum[:])
		bp := filepath.Join(dir, "blobs", blob)
		os.MkdirAll(filepath.Dir(bp), 0o755)
		if err := os.WriteFile(bp, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		link := filepath.Join(dir, "snapshots", rev, filepath.FromSlash(p))
		os.MkdirAll(filepath.Dir(link), 0o755)
		target := strings.Repeat("../", strings.Count(p, "/")+2) + "blobs/" + blob
		if runtime.GOOS == "windows" {
			if err := os.Link(bp, link); err != nil {
				t.Fatal(err)
			}
		} else if err := os.Symlink(target, link); err != nil {
			t.Fatal(err)
		}
	}
	os.MkdirAll(filepath.Join(dir, "refs"), 0o755)
	os.WriteFile(filepath.Join(dir, "refs", "main"), []byte(rev), 0o644)
}

func asrFiles() map[string]string {
	return map[string]string{"config.json": `{"a":1}`, "model.safetensors": strings.Repeat("w", 1000), "tokenizer/vocab.json": "{}"}
}

type env struct {
	fam  family.Dir
	hub  string
	apps map[string]AppState
}

func newEnv(t *testing.T) *env {
	t.Helper()
	root := t.TempDir()
	e := &env{fam: family.At(filepath.Join(root, "ActNow")), hub: filepath.Join(root, "hub"), apps: map[string]AppState{}}
	if _, err := e.fam.Ensure("test"); err != nil {
		t.Fatal(err)
	}
	return e
}

func (e *env) store(bid string) *Store {
	a, _, _ := family.ParseBundleID(bid)
	e.apps[bid] = AppState{BundleID: bid, AppID: a.ID, Name: a.Name, Installed: true}
	return &Store{Family: e.fam, Hub: e.hub, BundleID: bid, AppID: a.ID, Apps: func() map[string]AppState { return e.apps }}
}

func TestResolveHub(t *testing.T) {
	fam := t.TempDir()
	t.Setenv("ACTNOW_FAMILY_DIR", fam)
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HF_HOME", "")
	want := filepath.Join(fam, "models", "hub")
	if h, src := ResolveHub(true, "", ""); h != want || src != FromDefault {
		t.Errorf("%s %s", h, src)
	}
	// The environment no longer chooses $HUB (spec v1.0.9): it names
	// import sources.
	t.Setenv("HF_HOME", "/data/hf")
	t.Setenv("HF_HUB_CACHE", "/cache")
	if h, src := ResolveHub(true, "", "/own"); h != want || src != FromDefault {
		t.Errorf("%s %s", h, src)
	}
	if h, src := EnvHub(); h != "/cache" || src != FromEnvHub {
		t.Errorf("EnvHub %s %s", h, src)
	}
	if h, src := ResolveHub(false, "/fam", "/own/"); h != "/own" || src != FromApp {
		t.Errorf("%s %s", h, src)
	}
	if h, src := ResolveHub(true, "/fam", "/own"); h != "/fam" || src != FromFamily {
		t.Errorf("%s %s", h, src)
	}
}

func TestPaths(t *testing.T) {
	if got := RepoDir("/h", repoA); got != "/h/models--mlx-community--Qwen3-ASR-0.6B-4bit" {
		t.Error(got)
	}
	if got := BlobLockFile("/h", repoA, "abc"); got != "/h/.locks/models--mlx-community--Qwen3-ASR-0.6B-4bit/abc.lock" {
		t.Error(got)
	}
	if got := RepoLockFile("/h", repoA); got != "/h/.locks/models--mlx-community--Qwen3-ASR-0.6B-4bit/actnow-repo.lock" {
		t.Error(got)
	}
	if got := EngineCopy("/h", repoA); got != "/h/mlx-audio/mlx-community_Qwen3-ASR-0.6B-4bit" {
		t.Error(got)
	}
	for _, bad := range []string{"", "a", "a/b/c", "../x", "a/..", "a/.b/../c", "/abs", "a/b c"} {
		if ValidRepo(bad) {
			t.Errorf("%q valid", bad)
		}
	}
	r, err := ParseRef(repoA + "@" + revA)
	if err != nil || r.Repo != repoA || r.Revision != revA || r.String() != repoA+"@"+revA {
		t.Error(r, err)
	}
}

func TestCheck(t *testing.T) {
	hub := t.TempDir()
	c := Claim{Repo: repoA, Revision: revA}
	if st := Check(hub, c); st.State != Missing {
		t.Errorf("empty: %v", st.State)
	}
	os.MkdirAll(filepath.Join(RepoDir(hub, repoA), "blobs"), 0o755)
	os.WriteFile(filepath.Join(RepoDir(hub, repoA), "blobs", "x.incomplete"), []byte("12345"), 0o644)
	if st := Check(hub, c); st.State != Partial || st.Incomplete != 5 {
		t.Errorf("incomplete: %+v", st)
	}
	snap(t, hub, repoA, revA, asrFiles())
	st := Check(hub, c)
	if st.State != Installed || st.Bytes != int64(len(`{"a":1}`)+1000+2) || st.Snapshot != SnapshotDir(hub, c.Ref()) {
		t.Errorf("installed: %+v", st)
	}
	// Files listed must be there; globs match.
	c.Files = []string{"config.json", "tokenizer/*"}
	if st := Check(hub, c); st.State != Installed {
		t.Errorf("files: %+v", st)
	}
	c.Files = []string{"nope.bin"}
	if st := Check(hub, c); st.State != Partial || st.Missing[0] != "nope.bin" {
		t.Errorf("missing file: %+v", st)
	}
	// A dangling link.
	c.Files = nil
	sum := sha256.Sum256([]byte(`{"a":1}`))
	os.Remove(filepath.Join(RepoDir(hub, repoA), "blobs", hex.EncodeToString(sum[:])))
	if st := Check(hub, c); st.State != Partial || len(st.Missing) != 1 {
		t.Errorf("dangling: %+v", st)
	}
	// No weights: partial unless aux.
	snap(t, hub, repoB, revB, map[string]string{"config.json": "{}"})
	if st := Check(hub, Claim{Repo: repoB, Revision: revB}); st.State != Partial {
		t.Errorf("no weights: %v", st.State)
	}
	if st := Check(hub, Claim{Repo: repoB, Revision: revB, Aux: true}); st.State != Installed {
		t.Errorf("aux: %v", st.State)
	}
	// FindInstalled falls back to an accepted commit.
	if rev, _, ok := FindInstalled(hub, Claim{Repo: repoB, Revision: revA2, Aux: true}, []string{revB}); !ok || rev != revB {
		t.Errorf("accept: %s %v", rev, ok)
	}
}

func TestLocks(t *testing.T) {
	hub := t.TempDir()
	who := Owner{BundleID: "dev.hehehai.dicta", Op: OpDownload}
	a, err := TryLockRepo(hub, repoA, family.Shared, who)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TryLockRepo(hub, repoA, family.Shared, Owner{BundleID: "dev.hehehai.audiosmith", Op: OpClaim})
	if err != nil {
		t.Fatal("two shared:", err)
	}
	_, err = TryLockRepo(hub, repoA, family.Exclusive, Owner{BundleID: "x", Op: OpDelete})
	var be *BusyError
	if !errors.As(err, &be) || !errors.Is(err, family.ErrLocked) || be.Owner == nil {
		t.Fatalf("busy: %v", err)
	}
	if o, ok := Activity(hub, repoA, "other"); !ok || o.PID != os.Getpid() {
		t.Errorf("activity: %+v %v", o, ok)
	}
	b.Release()
	done := make(chan error)
	go func() { done <- a.Upgrade(context.Background(), OpFinish) }()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if a.Mode() != family.Exclusive {
		t.Error("not upgraded")
	}
	if o, ok := ReadOwner(a.Path()); !ok || o.Op != OpFinish {
		t.Errorf("owner after upgrade: %+v", o)
	}
	a.Release()
	a.Release()
	if _, ok := ReadOwner(RepoLockFile(hub, repoA)); ok {
		t.Error("owner file left")
	}
	if _, err := os.Stat(RepoLockFile(hub, repoA)); err != nil {
		t.Error("lock file removed")
	}
	// Blob locks.
	l, err := TryLockBlob(hub, repoA, "abc123", who)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := TryLockBlob(hub, repoA, "abc123", who); !errors.Is(err, family.ErrLocked) {
		t.Error("blob lock not exclusive")
	}
	if _, err := TryLockBlob(hub, repoA, "../x", who); err == nil {
		t.Error("bad blob accepted")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := LockBlob(ctx, hub, repoA, "abc123", who); !errors.Is(err, context.DeadlineExceeded) {
		t.Error(err)
	}
	l.Release()
	// A stale owner file of a dead process is ignored.
	os.WriteFile(RepoLockFile(hub, repoA)+".owner", []byte(`{"bundleId":"gone","pid":999999999,"op":"download"}`), 0o644)
	if _, ok := Activity(hub, repoA, ""); ok {
		t.Error("stale owner")
	}
}

func TestClaims(t *testing.T) {
	e := newEnv(t)
	s := e.store("dev.hehehai.dicta")
	ctx := context.Background()
	c := Claim{Repo: repoA, Revision: revA, Model: "qwen3-asr-0.6b-4bit", Family: "qwen3-asr", Role: RoleASR, Origin: OriginDownloaded}
	if err := s.Claim(ctx, c); !errors.Is(err, ErrNotInstalled) {
		t.Fatalf("not installed: %v", err)
	}
	snap(t, e.hub, repoA, revA, asrFiles())
	if err := s.Claim(ctx, c); err != nil {
		t.Fatal(err)
	}
	cf, _ := s.Claims()
	if len(cf.Claims) != 1 || cf.Root != e.hub || cf.App != "dicta" || cf.Claims[0].Bytes == 0 || cf.Schema != "actnow.models.claims/1" {
		t.Fatalf("%+v", cf)
	}
	since := cf.Claims[0].Since
	// Claiming again keeps since and origin.
	c.Origin = OriginReused
	c.Since = since.Add(time.Hour)
	if err := s.Claim(ctx, c); err != nil {
		t.Fatal(err)
	}
	cf, _ = s.Claims()
	if len(cf.Claims) != 1 || !cf.Claims[0].Since.Equal(since) || cf.Claims[0].Origin != OriginDownloaded {
		t.Errorf("reclaim: %+v", cf.Claims)
	}
	// Unknown members survive our rewrite.
	raw, _ := os.ReadFile(e.fam.ClaimsFile(s.BundleID))
	raw = []byte(strings.Replace(string(raw), `"claims": [`, `"future": 1, "claims": [`, 1))
	raw = []byte(strings.Replace(string(raw), `"repo":`, `"later": "x", "repo":`, 1))
	os.WriteFile(e.fam.ClaimsFile(s.BundleID), raw, 0o600)
	snap(t, e.hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})
	if err := s.Claim(ctx, Claim{Repo: repoB, Revision: revB}); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(e.fam.ClaimsFile(s.BundleID))
	if !strings.Contains(string(raw), `"future": 1`) || !strings.Contains(string(raw), `"later": "x"`) {
		t.Errorf("unknown members lost: %s", raw)
	}
	if err := s.Unclaim(Ref{repoB, revB}); err != nil {
		t.Fatal(err)
	}
	cf, _ = s.Claims()
	if len(cf.Claims) != 1 {
		t.Errorf("unclaim: %+v", cf.Claims)
	}
	// A newer major: never written back.
	os.WriteFile(e.fam.ClaimsFile(s.BundleID), []byte(`{"schema":"actnow.models.claims/2","claims":[]}`), 0o600)
	if err := s.Claim(ctx, c); !errors.Is(err, family.ErrNewerSchema) {
		t.Errorf("newer: %v", err)
	}
	os.Remove(e.fam.ClaimsFile(s.BundleID))
	// Another root: claims of the old root are not carried over.
	s.Claim(ctx, c)
	s2 := &Store{Family: s.Family, Hub: t.TempDir(), BundleID: s.BundleID, AppID: s.AppID, Apps: s.Apps}
	snap(t, s2.Hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})
	if err := s2.Claim(ctx, Claim{Repo: repoB, Revision: revB}); err != nil {
		t.Fatal(err)
	}
	cf, _ = s2.Claims()
	if cf.Root != s2.Hub || len(cf.Claims) != 1 || cf.Claims[0].Repo != repoB {
		t.Errorf("new root: %+v", cf)
	}
}

func TestDeleteSharedModel(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	dicta := e.store("dev.hehehai.dicta")
	smith := e.store("dev.hehehai.audiosmith")
	snap(t, e.hub, repoA, revA, asrFiles())
	ref := Ref{repoA, revA}
	if err := smith.Claim(ctx, Claim{Repo: repoA, Revision: revA, Origin: OriginDownloaded}); err != nil {
		t.Fatal(err)
	}
	if err := dicta.Claim(ctx, Claim{Repo: repoA, Revision: revA, Origin: OriginReused}); err != nil {
		t.Fatal(err)
	}
	d, err := dicta.Decide(ref)
	if err != nil || !d.Mine || len(d.Users) != 1 || d.Users[0].Name != "AudioSmith" || !d.InUse() || !d.LastSnapshot || d.FreeBytes == 0 {
		t.Fatalf("decide: %+v %v", d, err)
	}
	// Dicta removes it: AudioSmith uses it, the files stay.
	res, err := dicta.Delete(ctx, ref, ThisApp)
	if err != nil || res.Deleted {
		t.Fatalf("kept: %+v %v", res, err)
	}
	if Check(e.hub, Claim{Repo: repoA, Revision: revA}).State != Installed {
		t.Fatal("files deleted")
	}
	if cf, _ := dicta.Claims(); len(cf.Claims) != 0 {
		t.Error("dicta claim left")
	}
	// AudioSmith removes it: nobody else, the files go.
	res, err = smith.Delete(ctx, ref, ThisApp)
	if err != nil || !res.Deleted || res.Freed == 0 {
		t.Fatalf("deleted: %+v %v", res, err)
	}
	if _, err := os.Stat(RepoDir(e.hub, repoA)); !os.IsNotExist(err) {
		t.Error("repo dir left")
	}
}

func TestDeleteKeepsOtherSnapshotsBlobs(t *testing.T) {
	e := newEnv(t)
	s := e.store("dev.hehehai.dicta")
	files := asrFiles()
	snap(t, e.hub, repoA, revA, files)
	files2 := asrFiles()
	files2["config.json"] = `{"a":2}`
	snap(t, e.hub, repoA, revA2, files2)
	os.MkdirAll(EngineCopy(e.hub, repoA), 0o755)
	res, err := s.Delete(context.Background(), Ref{repoA, revA}, ThisApp)
	if err != nil || !res.Deleted || res.Freed != int64(len(`{"a":1}`)) || res.Decision.LastSnapshot {
		t.Fatalf("%+v %v", res, err)
	}
	if Check(e.hub, Claim{Repo: repoA, Revision: revA2}).State != Installed {
		t.Error("other snapshot broken")
	}
	if _, err := os.Stat(EngineCopy(e.hub, repoA)); err != nil {
		t.Error("engine copy removed while a snapshot remains")
	}
	// A blob being downloaded (its lock held) stays.
	os.WriteFile(filepath.Join(RepoDir(e.hub, repoA), "blobs", "feed.incomplete"), []byte("xx"), 0o644)
	l, _ := TryLockBlob(e.hub, repoA, "feed", Owner{BundleID: "other"})
	res, err = s.Delete(context.Background(), Ref{repoA, revA2}, ThisApp)
	if err != nil || !res.Deleted {
		t.Fatal(res, err)
	}
	if _, err := os.Stat(filepath.Join(RepoDir(e.hub, repoA), "blobs", "feed.incomplete")); err != nil {
		t.Error("locked partial blob removed")
	}
	l.Release()
}

func TestDeleteBusyAndLoaded(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	s := e.store("dev.hehehai.dicta")
	snap(t, e.hub, repoA, revA, asrFiles())
	ref := Ref{repoA, revA}
	s.Claim(ctx, Claim{Repo: repoA, Revision: revA})
	// Another app downloads (shared lock): deleting does not wait.
	l, _ := TryLockRepo(e.hub, repoA, family.Shared, Owner{BundleID: "dev.hehehai.audiosmith", Op: OpDownload})
	_, err := s.Delete(ctx, ref, ThisApp)
	var be *BusyError
	if !errors.As(err, &be) || be.Owner == nil || be.Owner.BundleID != "dev.hehehai.audiosmith" {
		t.Fatalf("busy: %v", err)
	}
	if cf, _ := s.Claims(); len(cf.Claims) != 1 {
		t.Error("claim not restored after busy")
	}
	l.Release()
	// A running app has it loaded: kept.
	famruntime.Write(e.fam, famruntime.File{BundleID: "dev.hehehai.audiosmith", Loaded: []famruntime.Loaded{{Repo: repoA, Revision: revA, Bytes: 1}}})
	res, err := s.Delete(ctx, ref, ThisApp)
	if err != nil || res.Deleted || len(res.Decision.LoadedBy) != 1 {
		t.Fatalf("loaded: %+v %v", res, err)
	}
	famruntime.Remove(e.fam, "dev.hehehai.audiosmith")
	// An unreadable claims file: unsure, kept.
	s.Claim(ctx, Claim{Repo: repoA, Revision: revA})
	os.WriteFile(e.fam.ClaimsFile("dev.hehehai.auralink"), []byte("{broken"), 0o600)
	res, _ = s.Delete(ctx, ref, ThisApp)
	if res.Deleted || !res.Decision.Unsure {
		t.Errorf("unsure: %+v", res)
	}
}

func TestDeleteForAllApps(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	dicta := e.store("dev.hehehai.dicta")
	smith := e.store("dev.hehehai.audiosmith")
	snap(t, e.hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})
	ref := Ref{repoB, revB}
	dicta.Claim(ctx, Claim{Repo: repoB, Revision: revB})
	smith.Now = func() time.Time { return time.Now().Add(time.Second) }
	res, err := smith.Delete(ctx, ref, AllApps)
	if err != nil || !res.Deleted || !res.Tombstoned {
		t.Fatalf("%+v %v", res, err)
	}
	ts, _ := ReadTombstones(e.fam)
	if len(ts) != 1 || ts[0].RemovedBy != smith.BundleID || ts[0].Root != e.hub {
		t.Fatalf("tombstones %+v", ts)
	}
	hit, err := dicta.ApplyTombstones()
	if err != nil || len(hit) != 1 {
		t.Fatalf("apply: %v %v", hit, err)
	}
	if cf, _ := dicta.Claims(); len(cf.Claims) != 0 {
		t.Error("claim kept after tombstone")
	}
	// Downloading again clears the tombstone.
	snap(t, e.hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})
	if err := dicta.Claim(ctx, Claim{Repo: repoB, Revision: revB, Origin: OriginDownloaded}); err != nil {
		t.Fatal(err)
	}
	if ts, _ := ReadTombstones(e.fam); len(ts) != 0 {
		t.Errorf("tombstone kept: %+v", ts)
	}
}

func TestOrphans(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	me := e.store("dev.hehehai.audiosmith")
	gone := e.store("dev.hehehai.dicta")
	snap(t, e.hub, repoA, revA, asrFiles())
	snap(t, e.hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})
	gone.Claim(ctx, Claim{Repo: repoA, Revision: revA}, Claim{Repo: repoB, Revision: revB})
	me.Claim(ctx, Claim{Repo: repoA, Revision: revA})
	os.WriteFile(e.fam.AppFile(gone.BundleID), []byte(`{}`), 0o600)
	e.apps[gone.BundleID] = AppState{BundleID: gone.BundleID, Installed: false}
	// Dicta's claim no longer protects.
	d, _ := me.Decide(Ref{repoB, revB})
	if d.InUse() || len(d.Orphans) != 1 {
		t.Errorf("decide: %+v", d)
	}
	orph, err := me.Orphans()
	if err != nil || len(orph) != 1 || orph[0].Bytes != int64(len("gemma")) || len(orph[0].Claims) != 2 {
		t.Fatalf("orphans: %+v %v", orph, err)
	}
	res, err := me.CleanOrphans(ctx, gone.BundleID)
	if err != nil || len(res) != 2 {
		t.Fatalf("%+v %v", res, err)
	}
	if Check(e.hub, Claim{Repo: repoA, Revision: revA}).State != Installed {
		t.Error("my model deleted")
	}
	if _, err := os.Stat(RepoDir(e.hub, repoB)); !os.IsNotExist(err) {
		t.Error("orphan model kept")
	}
	for _, p := range []string{e.fam.ClaimsFile(gone.BundleID), e.fam.AppFile(gone.BundleID)} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s left", p)
		}
	}
	if _, err := me.CleanOrphans(ctx, me.BundleID); err == nil {
		t.Error("cleaned own claims")
	}
}

func TestUsageAndSurvey(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	dicta := e.store("dev.hehehai.dicta")
	smith := e.store("dev.hehehai.audiosmith")
	snap(t, e.hub, repoA, revA, asrFiles())
	snap(t, e.hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})
	smith.Claim(ctx, Claim{Repo: repoA, Revision: revA})
	found := dicta.Survey(Claim{Repo: repoA, Revision: revA}, Claim{Repo: repoB, Revision: revB}, Claim{Repo: "a/b", Revision: revA})
	if len(found) != 2 || len(found[0].By) != 1 || found[0].By[0].AppID != "audiosmith" || len(found[1].By) != 0 {
		t.Fatalf("survey: %+v", found)
	}
	dicta.Claim(ctx, Claim{Repo: repoA, Revision: revA}, Claim{Repo: repoB, Revision: revB})
	u, err := dicta.Usage()
	if err != nil || u.Mine != int64(len("gemma")) || u.Shared != int64(len(`{"a":1}`)+1000+2) {
		t.Errorf("usage %+v %v", u, err)
	}
}

func TestCleanEngineCopies(t *testing.T) {
	hub := t.TempDir()
	old := EngineCopy(hub, "a/old")
	fresh := EngineCopy(hub, "a/fresh")
	full := EngineCopy(hub, "a/full")
	for _, d := range []string{old, fresh, full} {
		os.MkdirAll(d, 0o755)
	}
	os.WriteFile(filepath.Join(full, "x"), nil, 0o644)
	past := time.Now().Add(-time.Hour)
	os.Chtimes(old, past, past)
	os.Chtimes(full, past, past)
	got, err := CleanEngineCopies(hub, EngineCopyMinAge)
	if err != nil || len(got) != 1 || got[0] != "a_old" {
		t.Errorf("%v %v", got, err)
	}
}

func TestLicences(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	if _, ok := LicenceAccepted(e.fam, "mlx-community/OmniVoice-bf16", "CC-BY-NC-4.0"); ok {
		t.Error("accepted before")
	}
	if err := AcceptLicence(ctx, e.fam, "mlx-community/OmniVoice-bf16", "CC-BY-NC-4.0", "dev.hehehai.audiosmith"); err != nil {
		t.Fatal(err)
	}
	AcceptLicence(ctx, e.fam, "mlx-community/OmniVoice-bf16", "CC-BY-NC-4.0", "dev.hehehai.dicta")
	a, ok := LicenceAccepted(e.fam, "mlx-community/OmniVoice-bf16", "cc-by-nc-4.0")
	if !ok || a.By != "dev.hehehai.dicta" {
		t.Errorf("%+v", a)
	}
	all, _ := ReadLicences(e.fam)
	if len(all) != 1 {
		t.Errorf("duplicates: %+v", all)
	}
}

func TestPins(t *testing.T) {
	seen := map[string]bool{}
	for _, p := range Pins() {
		if !ValidRepo(p.Repo) || !IsCommit(p.Revision) || p.Family == "" || p.Role == "" || len(p.Apps) == 0 || seen[p.Repo] {
			t.Errorf("bad pin %+v", p)
		}
		seen[p.Repo] = true
		for _, a := range p.Apps {
			if _, ok := family.AppByID(a); !ok {
				t.Errorf("%s: unknown app %s", p.Repo, a)
			}
		}
		for _, x := range p.Aux {
			if !ValidRepo(x.Repo) || !IsCommit(x.Revision) {
				t.Errorf("%s: bad aux %+v", p.Repo, x)
			}
		}
	}
	if p, ok := PinOf(repoB); !ok || p.Revision != revB {
		t.Error("gemma pin")
	}
	if len(PinsOfFamily("qwen3-asr")) < 2 || len(PinsOfApp("dicta")) < 9 {
		t.Error("pin queries")
	}
	// Act's catalogue: EmbeddingGemma 2 (its own pin, role embed) and four
	// models it shares with Dicta, AudioSmith and VoxtNew (the forced
	// aligner since v0.7.2).
	if p, ok := PinOf("mlx-community/embeddinggemma-2-8bit"); !ok || p.Role != RoleEmbed || p.Family != "embeddinggemma2" {
		t.Errorf("embeddinggemma pin %+v", p)
	}
	if len(PinsOfApp("act")) != 5 {
		t.Errorf("act pins: %d", len(PinsOfApp("act")))
	}
}

func TestSharedWithinApp(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	s := e.store("dev.hehehai.audiosmith")
	snap(t, e.hub, repoA, revA, asrFiles())                                           // a main repo
	snap(t, e.hub, repoB, revB, map[string]string{"model.safetensors": "gemma"})      // another main repo
	tok := Ref{"openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"} // the shared aux
	snap(t, e.hub, tok.Repo, tok.Revision, map[string]string{"vocab.json": "{}"})
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(s.Claim(ctx, Claim{Repo: repoA, Revision: revA, Model: "turbo"}, Claim{Repo: tok.Repo, Revision: tok.Revision, Model: "turbo", Aux: true, Files: []string{"vocab.json"}}))
	must(s.Claim(ctx, Claim{Repo: repoB, Revision: revB, Model: "large"}, Claim{Repo: tok.Repo, Revision: tok.Revision, Model: "large", Aux: true}))
	cf, _ := s.Claims()
	c, _ := cf.Find(tok)
	if c.Model != "turbo" || !slices.Equal(c.Models, []string{"large", "turbo"}) || c.Files != nil || !c.Aux {
		t.Fatalf("merged claim %+v", c)
	}
	if d, _ := s.Decide(tok); !slices.Equal(d.MyModels, []string{"large", "turbo"}) {
		t.Errorf("my models %v", d.MyModels)
	}
	// Removing "turbo": its main repo goes, the shared aux stays claimed for "large".
	res, err := s.DeleteModel(ctx, Ref{repoA, revA}, "turbo", ThisApp)
	if err != nil || !res.Deleted {
		t.Fatalf("turbo main: %+v %v", res, err)
	}
	res, err = s.DeleteModel(ctx, tok, "turbo", ThisApp)
	if err != nil || res.Deleted || !slices.Equal(res.OwnModels, []string{"large"}) {
		t.Fatalf("turbo aux: %+v %v", res, err)
	}
	cf, _ = s.Claims()
	if c, ok := cf.Find(tok); !ok || c.Model != "large" || !slices.Equal(c.Users(), []string{"large"}) {
		t.Fatalf("aux claim after removing turbo: %+v %v", c, ok)
	}
	if Check(e.hub, Claim{Repo: tok.Repo, Revision: tok.Revision, Aux: true}).State != Installed {
		t.Fatal("aux files deleted")
	}
	// The last user: everything goes.
	res, err = s.DeleteModel(ctx, tok, "large", ThisApp)
	if err != nil || !res.Deleted || len(res.OwnModels) != 0 {
		t.Fatalf("large aux: %+v %v", res, err)
	}
	// AllApps deletes even when another model of the app uses it, and says so.
	snap(t, e.hub, tok.Repo, tok.Revision, map[string]string{"vocab.json": "{}"})
	must(s.Claim(ctx, Claim{Repo: tok.Repo, Revision: tok.Revision, Model: "x", Aux: true}, Claim{Repo: tok.Repo, Revision: tok.Revision, Model: "y", Aux: true}))
	res, err = s.DeleteModel(ctx, tok, "x", AllApps)
	if err != nil || !res.Deleted || !slices.Equal(res.OwnModels, []string{"y"}) {
		t.Fatalf("all apps: %+v %v", res, err)
	}
	if cf, _ := s.Claims(); len(cf.Claims) != 1 { // repoB's claim
		t.Errorf("claims left %+v", cf.Claims)
	}
}

func TestReleaseLegacyClaims(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	s := e.store("dev.hehehai.dicta")
	snap(t, e.hub, repoA, revA, asrFiles())
	// A claims file from v0.1 (model, no models).
	os.WriteFile(e.fam.ClaimsFile(s.BundleID), []byte(`{"schema":"actnow.models.claims/1","app":"dicta","bundleId":"dev.hehehai.dicta","root":`+strconv.Quote(e.hub)+`,"claims":[{"repo":"`+repoA+`","revision":"`+revA+`","model":"old","files":null,"aux":false,"since":"2026-10-20T08:00:00Z"}]}`), 0o600)
	if err := s.Claim(ctx, Claim{Repo: repoA, Revision: revA, Model: "new"}); err != nil {
		t.Fatal(err)
	}
	cf, _ := s.Claims()
	if c, _ := cf.Find(Ref{repoA, revA}); c.Model != "old" || !slices.Equal(c.Users(), []string{"new", "old"}) {
		t.Fatalf("%+v", c)
	}
	if rest, err := s.Release(Ref{repoA, revA}, "unknown"); err != nil || len(rest) != 2 {
		t.Errorf("releasing a model that does not use it: %v %v", rest, err)
	}
	if rest, err := s.Release(Ref{repoA, revA}, "old"); err != nil || !slices.Equal(rest, []string{"new"}) {
		t.Errorf("%v %v", rest, err)
	}
	if rest, err := s.Release(Ref{repoA, revA}, "new"); err != nil || len(rest) != 0 {
		t.Errorf("%v %v", rest, err)
	}
	if cf, _ := s.Claims(); len(cf.Claims) != 0 {
		t.Errorf("claim left %+v", cf.Claims)
	}
	// A claim naming no model goes at once.
	s.Claim(ctx, Claim{Repo: repoA, Revision: revA})
	if rest, _ := s.Release(Ref{repoA, revA}, "whatever"); len(rest) != 0 {
		t.Error(rest)
	}
	if cf, _ := s.Claims(); len(cf.Claims) != 0 {
		t.Errorf("claim left %+v", cf.Claims)
	}
}

func TestDowngrade(t *testing.T) {
	hub := t.TempDir()
	ctx := context.Background()
	l, err := LockRepo(ctx, hub, repoA, family.Exclusive, Owner{BundleID: "a", Op: OpFinish})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := TryLockRepo(hub, repoA, family.Shared, Owner{BundleID: "b"}); !errors.Is(err, family.ErrLocked) {
		t.Fatal("shared while exclusive")
	}
	if err := l.Downgrade(ctx, OpDownload); err != nil || l.Mode() != family.Shared {
		t.Fatal(err, l.Mode())
	}
	other, err := TryLockRepo(hub, repoA, family.Shared, Owner{BundleID: "b"})
	if err != nil {
		t.Fatal("shared after downgrade:", err)
	}
	if _, err := TryLockRepo(hub, repoA, family.Exclusive, Owner{BundleID: "c"}); !errors.Is(err, family.ErrLocked) {
		t.Fatal("exclusive while shared")
	}
	other.Release()
	if err := l.Upgrade(ctx, OpFinish); err != nil || l.Mode() != family.Exclusive {
		t.Fatal(err)
	}
	l.Release()
	if err := l.Downgrade(ctx, OpDownload); err == nil {
		t.Error("downgrade of a released lock")
	}
}
