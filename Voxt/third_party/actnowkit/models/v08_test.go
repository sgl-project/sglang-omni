package models

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func inode(t *testing.T, p string) uint64 {
	t.Helper()
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Sys().(*syscall.Stat_t).Ino
}

// A3: a truncated blob is not "installed".
func TestCheckSizes(t *testing.T) {
	e := newEnv(t)
	snap(t, e.hub, repoA, revA, asrFiles())
	c := Claim{Repo: repoA, Revision: revA}
	if st := Check(e.hub, c); st.State != Installed {
		t.Fatalf("before %+v", st)
	}
	weights, _ := filepath.EvalSymlinks(filepath.Join(SnapshotDir(e.hub, Ref{repoA, revA}), "model.safetensors"))
	// Without recorded sizes, only the catalogue's size tells.
	os.Truncate(weights, 100)
	if st := Check(e.hub, c); st.State != Installed {
		t.Fatalf("no sizes known: %+v", st)
	}
	cs := c
	cs.Sizes = map[string]int64{"model.safetensors": 1000}
	if st := Check(e.hub, cs); st.State != Partial || len(st.Damaged) != 1 || st.Damaged[0] != "model.safetensors" {
		t.Fatalf("catalogue size: %+v", st)
	}
	// Recorded blob sizes (the downloader's) tell without a catalogue.
	if err := RecordBlobSizes(e.hub, repoA, map[string]int64{filepath.Base(weights): 1000, "not a blob/../x": 3}); err != nil {
		t.Fatal(err)
	}
	if st := Check(e.hub, c); st.State != Partial || len(st.Damaged) != 1 {
		t.Fatalf("recorded size: %+v", st)
	}
	if got := ReadBlobSizes(e.hub, repoA); len(got) != 1 {
		t.Fatalf("sizes %v", got)
	}
	os.WriteFile(weights, []byte(strings.Repeat("w", 1000)), 0o644)
	if st := Check(e.hub, c); st.State != Installed {
		t.Fatalf("repaired %+v", st)
	}
}

// A4: claims lost with the family folder make deletions unsure; Heal
// writes them back.
func TestUnclaimedAndHeal(t *testing.T) {
	ctx := context.Background()
	e := newEnv(t)
	snap(t, e.hub, repoA, revA, asrFiles())
	as := e.store("dev.hehehai.audiosmith")
	dicta := e.store("dev.hehehai.dicta")
	if err := dicta.Claim(ctx, Claim{Repo: repoA, Revision: revA, Model: "asr"}); err != nil {
		t.Fatal(err)
	}
	st := e.apps["dev.hehehai.dicta"]
	st.ModelsRoot = e.hub
	e.apps["dev.hehehai.dicta"] = st
	// The claims folder goes (a cleaner).
	os.RemoveAll(e.fam.ClaimsDir())
	d, err := as.Decide(Ref{repoA, revA})
	if err != nil || !d.Unsure || len(d.Unclaimed) != 1 || d.Unclaimed[0].Name != "Dicta" || !d.InUse() {
		t.Fatalf("decision %+v %v", d, err)
	}
	res, err := as.Delete(ctx, Ref{repoA, revA}, ThisApp)
	if err != nil || res.Deleted {
		t.Fatalf("deleted %+v %v", res, err)
	}
	// An app without models in this hub does not count.
	st.ModelsRoot = "/elsewhere"
	e.apps["dev.hehehai.dicta"] = st
	if d, _ := as.Decide(Ref{repoA, revA}); d.Unsure {
		t.Fatalf("other root %+v", d)
	}
	st.ModelsRoot = e.hub
	e.apps["dev.hehehai.dicta"] = st
	// Dicta becomes active: Heal claims its installed models again.
	healed, err := dicta.Heal(ctx, Claim{Repo: repoA, Revision: revA, Model: "asr"}, Claim{Repo: repoB, Revision: revB, Model: "llm"})
	if err != nil || !healed {
		t.Fatalf("heal %v %v", healed, err)
	}
	c, _ := dicta.Claims()
	if len(c.Claims) != 1 || c.Claims[0].Origin != OriginMigrated {
		t.Fatalf("claims %+v", c)
	}
	if again, _ := dicta.Heal(ctx); again {
		t.Fatal("healed twice")
	}
	d, _ = as.Decide(Ref{repoA, revA})
	if d.Unsure || len(d.Users) != 1 {
		t.Fatalf("after heal %+v", d)
	}
	// An app with no models writes an empty file: not "lost".
	act := e.store("dev.hehehai.act")
	s := e.apps["dev.hehehai.act"]
	s.ModelsRoot = e.hub
	e.apps["dev.hehehai.act"] = s
	if d, _ := as.Decide(Ref{repoA, revA}); !d.Unsure {
		t.Fatal("act without claims file should make it unsure")
	}
	if ok, err := act.Heal(ctx); !ok || err != nil {
		t.Fatal(ok, err)
	}
	if d, _ := as.Decide(Ref{repoA, revA}); d.Unsure {
		t.Fatalf("act healed %+v", d)
	}
}

// Q1: models come over from a Hugging Face cache by hard links, once.
func TestImport(t *testing.T) {
	ctx := context.Background()
	e := newEnv(t)
	old := filepath.Join(filepath.Dir(e.hub), "hf", "hub")
	snap(t, old, repoA, revA, asrFiles())
	snap(t, old, repoB, revB, map[string]string{"model.safetensors": "gemma", "config.json": "{}"})
	snap(t, old, "someone/own-model", revA2, map[string]string{"model.safetensors": "mine"})
	// Dicta's claims file still names the old library.
	old0 := &Store{Family: e.fam, Hub: old, BundleID: "dev.hehehai.dicta", AppID: "dicta"}
	if err := old0.Claim(ctx, Claim{Repo: repoB, Revision: revB, Model: "gemma"}); err != nil {
		t.Fatal(err)
	}
	dicta := e.store("dev.hehehai.dicta")
	want := []Claim{{Repo: repoA, Revision: revA, Model: "asr"}, {Repo: "mlx-community/absent", Revision: revA2, Model: "x"}}
	res, err := dicta.Import(ctx, []string{old, e.hub, "/does/not/exist"}, false, want...)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, r := range res {
		got[r.Ref.Repo] = r.Action
	}
	if got[repoA] != ImportImported || got[repoB] != ImportImported || got["mlx-community/absent"] != ImportAbsent || len(got) != 3 {
		t.Fatalf("results %v", res)
	}
	for _, r := range []Ref{{repoA, revA}, {repoB, revB}} {
		if st := Check(e.hub, Claim{Repo: r.Repo, Revision: r.Revision}); st.State != Installed {
			t.Fatalf("%s %+v", r, st)
		}
	}
	// Hard links: same inode, the source untouched.
	a := filepath.Join(SnapshotDir(e.hub, Ref{repoA, revA}), "model.safetensors")
	b := filepath.Join(SnapshotDir(old, Ref{repoA, revA}), "model.safetensors")
	if inode(t, a) != inode(t, b) {
		t.Fatal("not hard-linked")
	}
	if l, _ := os.Readlink(a); !strings.HasPrefix(l, "../../blobs/") {
		t.Fatalf("link %q", l)
	}
	if data, _ := os.ReadFile(filepath.Join(RepoDir(e.hub, repoA), "refs", "main")); string(data) != revA {
		t.Fatalf("refs/main %q", data)
	}
	if len(ReadBlobSizes(e.hub, repoA)) != 3 {
		t.Fatal("sizes not recorded")
	}
	if _, err := os.Stat(RepoDir(e.hub, "someone/own-model")); !os.IsNotExist(err) {
		t.Fatal("the user's own model was imported")
	}
	// Only its previous claim is claimed again; want is left to the app.
	c, _ := dicta.Claims()
	if c.Root != e.hub || len(c.Claims) != 1 || c.Claims[0].Repo != repoB || c.Claims[0].Origin != OriginImported {
		t.Fatalf("claims %+v", c)
	}
	if err := dicta.Claim(ctx, Claim{Repo: repoA, Revision: revA, Model: "asr", Origin: OriginMigrated}); err != nil {
		t.Fatal(err)
	}
	// Deleting it frees nothing while the cache keeps it, and it does not
	// come back at the next launch.
	d, _ := dicta.Decide(Ref{repoA, revA})
	if d.FreeBytes != 0 {
		t.Fatalf("free %d", d.FreeBytes)
	}
	r, err := dicta.Delete(ctx, Ref{repoA, revA}, ThisApp)
	if err != nil || !r.Deleted || r.Freed != 0 {
		t.Fatalf("delete %+v %v", r, err)
	}
	if _, err := os.Stat(b); err != nil {
		t.Fatal("the source lost its file")
	}
	res, err = dicta.Import(ctx, []string{old}, false, want...)
	if err != nil || len(res) != 0 {
		t.Fatalf("second import %v %v", res, err)
	}
	// Another app finds them there: present, nothing linked, not claimed.
	as := e.store("dev.hehehai.audiosmith")
	res, _ = as.Import(ctx, []string{old}, false, Claim{Repo: repoB, Revision: revB, Model: "gemma"})
	if len(res) != 1 || res[0].Action != ImportPresent || res[0].Linked != 0 {
		t.Fatalf("audiosmith %v", res)
	}
	if d, _ := dicta.Decide(Ref{repoB, revB}); len(d.Users) != 0 || !d.Mine {
		t.Fatalf("users %+v", d)
	}
}

// Q2: Time Machine marks follow models.backup.
func TestApplyBackup(t *testing.T) {
	marks := map[string]bool{}
	setBackupExcluded = func(p string, ex bool) error { marks[p] = ex; return nil }
	isBackupExcluded = func(p string) (bool, error) { return marks[p], nil }
	t.Cleanup(func() { setBackupExcluded, isBackupExcluded = platformSetBackupExcluded, platformIsBackupExcluded })
	ctx := context.Background()
	e := newEnv(t)
	e.hub = DefaultHubIn(e.fam)
	snap(t, e.hub, repoA, revA, asrFiles())
	if err := e.store("dev.hehehai.dicta").Claim(ctx, Claim{Repo: repoA, Revision: revA}); err != nil {
		t.Fatal(err)
	}
	if err := ApplyBackup(e.fam, e.hub, false); err != nil {
		t.Fatal(err)
	}
	if !marks[e.hub] || !marks[RepoDir(e.hub, repoA)] {
		t.Fatalf("marks %v", marks)
	}
	if err := ApplyBackup(e.fam, e.hub, true); err != nil {
		t.Fatal(err)
	}
	if marks[e.hub] || marks[RepoDir(e.hub, repoA)] {
		t.Fatalf("cleared %v", marks)
	}
	// A folder the user chose: only claimed repos, never the folder.
	other := filepath.Join(t.TempDir(), "hf")
	snap(t, other, repoA, revA, asrFiles())
	s := &Store{Family: e.fam, Hub: other, BundleID: "dev.hehehai.audiosmith"}
	s.Claim(ctx, Claim{Repo: repoA, Revision: revA})
	ApplyBackup(e.fam, other, false)
	if marks[other] || !marks[RepoDir(other, repoA)] {
		t.Fatalf("custom %v", marks)
	}
}
