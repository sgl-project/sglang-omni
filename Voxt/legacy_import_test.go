package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The previous Voxt's databases in testdata/legacy:
//
//	real  what old Voxt 1.16.3 wrote after three hot words were added in its
//	      window: schema and rows still only in the WAL
//	rich  replacement terms, a disabled term, CJK, 12 aliases, a 90-character
//	      term and a case duplicate
//	hist  one history entry of each kind, and an empty one
const legacyFixtures = "testdata/legacy"

// copyFixture copies a fixture's voxt.sqlite (and its -wal and -shm when it
// has them) into a temp dir laid out like the previous Voxt's Data/Library,
// and returns that dir.
func copyFixture(t *testing.T, name string) string {
	t.Helper()
	lib := t.TempDir()
	dst := filepath.Join(lib, "Application Support", "Voxt")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"voxt.sqlite", "voxt.sqlite-wal", "voxt.sqlite-shm"} {
		b, err := os.ReadFile(filepath.Join(legacyFixtures, name, n))
		if os.IsNotExist(err) && n != "voxt.sqlite" {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, n), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return lib
}

// The terms of old Voxt's real database, still only in its WAL.
func TestReadLegacyTermsReal(t *testing.T) {
	terms, skipped, err := readLegacyTerms(context.Background(), legacyDatabase(copyFixture(t, "real")))
	if err != nil || skipped != 0 {
		t.Fatalf("read: %v, skipped %d", err, skipped)
	}
	var got []string
	for _, tm := range terms {
		got = append(got, tm.Term)
		if !tm.Enabled {
			t.Errorf("%s not enabled", tm.Term)
		}
	}
	if strings.Join(got, ",") != "SGLang-Omni,Qwen3-ASR,Kennington Oval" {
		t.Errorf("terms %v", got)
	}
}

// Replacement terms become aliases (the first 10, in order), a disabled term
// stays disabled, and a term too long for the dictionary is counted, not read.
func TestReadLegacyTermsRich(t *testing.T) {
	terms, skipped, err := readLegacyTerms(context.Background(), legacyDatabase(copyFixture(t, "rich")))
	if err != nil {
		t.Fatal(err)
	}
	by := map[string]int{}
	for i, tm := range terms {
		by[tm.Term] = i
	}
	if skipped != 1 {
		t.Errorf("skipped %d, want 1 (the 90-character term)", skipped)
	}
	if i, ok := by["Kubernetes"]; !ok || len(terms[i].Aliases) != 2 {
		t.Errorf("Kubernetes with 2 aliases: %v", terms)
	}
	if i, ok := by["TensorRT"]; !ok || terms[i].Enabled {
		t.Errorf("TensorRT should be imported disabled")
	}
	if i, ok := by["张三丰"]; !ok || len(terms[i].Aliases) != 1 {
		t.Errorf("CJK term with its alias")
	}
	if i, ok := by["ManyAliases"]; !ok || len(terms[i].Aliases) != 10 || terms[i].Aliases[9] != "alias number 10" {
		t.Errorf("ManyAliases keeps its first 10 aliases in order: %v", terms[by["ManyAliases"]].Aliases)
	}
}

// History: dictation, translation and rewrite come over with stable ids;
// an empty entry is skipped; transcript rows go to meetings, not history.
func TestReadLegacyHistory(t *testing.T) {
	entries, skipped, err := readLegacyHistory(context.Background(), legacyDatabase(copyFixture(t, "hist")))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 || skipped != 1 {
		t.Fatalf("got %d entries, %d skipped: %+v", len(entries), skipped, entries)
	}
	want := map[string]string{"legacy-a1": "transcribe", "legacy-a2": "translate", "legacy-a3": "rewrite"}
	for _, e := range entries {
		if string(e.Kind) != want[e.ID] || e.Result == "" || e.CreatedAt.IsZero() || e.AudioMS == 0 {
			t.Errorf("%+v", e)
		}
	}
}

// A picked folder at any of the three levels finds the database; a wrong
// folder finds none.
func TestLegacyDatabaseLevels(t *testing.T) {
	lib := t.TempDir()
	db := filepath.Join(lib, "Application Support", "Voxt", "voxt.sqlite")
	os.MkdirAll(filepath.Dir(db), 0o755)
	os.WriteFile(db, []byte("x"), 0o644)
	for _, p := range []string{lib, filepath.Join(lib, "Application Support"), filepath.Join(lib, "Application Support", "Voxt")} {
		if legacyDatabase(p) != db {
			t.Errorf("from %s: %q", p, legacyDatabase(p))
		}
	}
	if legacyDatabase(t.TempDir()) != "" {
		t.Error("an empty folder found a database")
	}
}

// A model the previous Voxt downloaded (flat, model-storage/<owner>_<name>)
// is recognised file by file, refused when a byte differs, and once linked
// into a library it counts as installed. The model is made up: two small
// files, one of them "LFS" with its SHA-256.
func TestReuseLegacyModelFiles(t *testing.T) {
	dir := t.TempDir()
	files := map[string][]byte{"config.json": []byte(`{"a":1}`), "model.safetensors": []byte(strings.Repeat("w", 4096))}
	mod := models.Model{ID: "test.model", Repo: "owner/name", Revision: "0123456789abcdef0123456789abcdef01234567"}
	for _, p := range []string{"config.json", "model.safetensors"} {
		os.WriteFile(filepath.Join(dir, p), files[p], 0o644)
		sum := sha256.Sum256(files[p])
		mod.Files = append(mod.Files, models.File{Path: p, Size: int64(len(files[p])), SHA256: hex.EncodeToString(sum[:]), LFS: p == "model.safetensors"})
	}
	if !legacyModelMatches(dir, mod) {
		t.Fatal("the files do not match")
	}
	hub := t.TempDir()
	if _, err := linkModelFiles(dir, models.SnapshotDir(hub, mod, mod.Revision), mod); err != nil {
		t.Fatal(err)
	}
	if models.Installed(hub, mod) == "" {
		t.Fatal("linked model not installed")
	}
	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte(strings.Repeat("w", 4095)+"x"), 0o644)
	if legacyModelMatches(dir, mod) {
		t.Fatal("a changed file still matches")
	}
}

// The previous Voxt's dictionary export (Dictionary > Export, version 3):
// terms with their replacement terms as aliases; a term too long is skipped.
func TestParseLegacyExport(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(legacyFixtures, "export", "Voxt-Dictionary.json"))
	if err != nil {
		t.Fatal(err)
	}
	terms, skipped, err := parseLegacyExport(data)
	if err != nil || len(terms) != 3 || skipped != 1 {
		t.Fatalf("terms %v, skipped %d, err %v", terms, skipped, err)
	}
	if terms[0].Term != "SGLang-Omni" || len(terms[0].Aliases) != 2 || terms[2].Aliases[0] != "张三峰" {
		t.Errorf("%+v", terms)
	}
	if _, _, err := parseLegacyExport([]byte(`{"hello": 1}`)); err == nil {
		t.Error("a JSON file that is not an export was accepted")
	}
}

// The import reads the previous Voxt's database directly only when it opens
// and reads; a database that is found but cannot be opened (the container's
// EPERM, here a file without permissions) sends the user to the open panel.
func TestLegacyReadable(t *testing.T) {
	lib := copyFixture(t, "real")
	if !legacyReadable(lib) {
		t.Fatal("a readable database is not readable")
	}
	if legacyReadable(t.TempDir()) {
		t.Error("a folder without a database is readable")
	}
	db := legacyDatabase(lib)
	if err := os.Chmod(db, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(db, 0o644) })
	if legacyReadable(lib) {
		t.Error("a database that cannot be opened is readable")
	}
}

// Who set up a version without the import (the guide's card never showed)
// gets one card that offers it; dismissed, it does not come back on the next
// start. A fresh install gets the guide's card instead.
func TestOfferLegacyOnce(t *testing.T) {
	home := t.TempDir()
	if err := os.MkdirAll(filepath.Join(home, "Library", "Containers", legacyBundleID), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VOXTNEW_LEGACY_HOME", home)
	dir := testDir(t)
	ctx := context.Background()
	offered := func(a *app) bool {
		list, _ := a.notes.List(ctx)
		for _, n := range list {
			if n.ID == legacyOffered && n.Link == linkImportLegacy {
				return true
			}
		}
		return false
	}
	a, _, _ := newTestAppIn(t, dir)
	a.offerLegacy()
	if offered(a) {
		t.Fatal("offered on a fresh install, where the guide has its card")
	}
	a.prefs.Setup.Done = true
	a.offerLegacy()
	if !offered(a) {
		t.Fatal("no card for who set up without the guide's card")
	}
	if err := a.notes.Dismiss(ctx, legacyOffered); err != nil { // the card's ×, saved
		t.Fatal(err)
	}
	a.quit()
	b, _, _ := newTestAppIn(t, dir)
	b.prefs.Setup.Done = true
	b.offerLegacy()
	if offered(b) {
		t.Error("the dismissed card came back on the next start")
	}
}

// The previous Voxt's language models stay where they are: Voxt no longer
// runs language models on this Mac, so no retired model is offered for
// reuse, while the speech models are.
func TestLegacyReusableSkipsRetired(t *testing.T) {
	speech := 0
	for _, mod := range models.Catalog() {
		if mod.Retired && legacyReusable(mod) {
			t.Errorf("%s is retired but would come over", mod.ID)
		}
		if mod.Kind == models.ASR && legacyReusable(mod) {
			speech++
		}
	}
	if speech == 0 {
		t.Error("no speech model would come over")
	}
}

// An import ends the offer: the sidebar card goes, and does not come back on
// a later start (a user who imported from the setup guide, then skipped the
// rest of it, is not offered the import again).
func TestImportEndsTheOffer(t *testing.T) {
	home := t.TempDir()
	if err := os.MkdirAll(filepath.Join(home, "Library", "Containers", legacyBundleID), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VOXTNEW_LEGACY_HOME", home)
	ctx := context.Background()
	offered := func(a *app) bool {
		list, _ := a.notes.List(ctx)
		for _, n := range list {
			if n.ID == legacyOffered {
				return true
			}
		}
		return false
	}
	dir := testDir(t)
	a, _, _ := newTestAppIn(t, dir)
	a.importLegacyFrom(copyFixture(t, "real")) // from the setup guide, before Setup is done
	if !a.ob.legacyDone {
		t.Fatalf("the import did not succeed: %q", a.ob.legacy.text())
	}
	if _, ok, _ := a.lib.store.Documents().Get(ctx, legacyDockSettled); !ok { // legacyDockAtStart leaves the switch alone
		t.Error("the import did not record that it read Show in Dock")
	}
	a.quit()
	b, _, _ := newTestAppIn(t, dir)
	b.prefs.Setup.Done = true
	b.offerLegacy()
	if offered(b) {
		t.Error("offered again after an import")
	}
}

// The previous Voxt kept speech models in <root>/mlx-audio/<owner>_<name>,
// under model-storage or under the folder the user chose (its preference
// modelStorageRootPath).
func TestLegacyModelRoots(t *testing.T) {
	mod := models.Model{Repo: "mlx-community/Qwen3-ASR-0.6B-4bit"}
	if got, want := legacyModelDir("/r", mod), filepath.Join("/r", "mlx-audio", "mlx-community_Qwen3-ASR-0.6B-4bit"); got != want {
		t.Errorf("model dir %q, want %q", got, want)
	}
	lib := t.TempDir()
	voxtDir := filepath.Join(lib, "Application Support", "Voxt")
	if got := legacyModelRoots(voxtDir); len(got) != 1 || got[0] != filepath.Join(voxtDir, "model-storage") {
		t.Errorf("no preferences: roots %v", got)
	}
	if _, err := os.Stat("/usr/bin/plutil"); err != nil {
		t.Skip("no plutil: the chosen folder is read on macOS only")
	}
	chosen := t.TempDir()
	prefs := filepath.Join(lib, "Preferences", legacyBundleID+".plist")
	os.MkdirAll(filepath.Dir(prefs), 0o755)
	plist := `<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>modelStorageRootPath</key><string>` + chosen + `</string></dict></plist>`
	if err := os.WriteFile(prefs, []byte(plist), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := legacyModelRoots(voxtDir); len(got) != 2 || got[1] != chosen {
		t.Errorf("with a chosen folder: roots %v, want it second", got)
	}
}

// The upgrade test's import check (VOXTNEW_SMOKE_LEGACY=1,
// .github/workflows/voxt-upgrade-test.yaml): --smoke reads the previous
// Voxt's container without the open panel, imports it into its own data
// folder and prints what that folder then holds; an empty container fails.
func TestSmokeLegacy(t *testing.T) {
	t.Setenv("VOXTNEW_SMOKE_LEGACY", "1")
	for _, c := range []struct{ fixture, want string }{
		{"real", "smoke: legacy import: 3 terms, 0 history entries"},
		{"hist", "smoke: legacy import: 1 terms, 3 history entries"},
		{"", "is not readable without asking"},
	} {
		home := t.TempDir()
		container := filepath.Join(home, "Library", "Containers", legacyBundleID, "Data")
		if err := os.MkdirAll(container, 0o755); err != nil {
			t.Fatal(err)
		}
		lib := t.TempDir()
		if c.fixture != "" {
			lib = copyFixture(t, c.fixture)
		}
		if err := os.Symlink(lib, filepath.Join(container, "Library")); err != nil {
			t.Fatal(err)
		}
		t.Setenv("VOXTNEW_LEGACY_HOME", home)
		out, err := os.CreateTemp(t.TempDir(), "stdout")
		if err != nil {
			t.Fatal(err)
		}
		stdout := os.Stdout
		os.Stdout = out
		code := runSmoke(t.TempDir())
		os.Stdout = stdout
		printed, _ := os.ReadFile(out.Name())
		if !strings.Contains(string(printed), c.want) || (code == 0) != (c.fixture != "") {
			t.Errorf("%q: exit %d, printed:\n%s\nwant a line %q", c.fixture, code, printed, c.want)
		}
	}
}

// TestImportReportGroups: the bug was the import's result as one long
// paragraph. It reads in groups: what came over, what was left out and
// why (the history's meetings and empty entries), and what is left to do,
// each with Complete setup. An import from outside the setup guide shows
// them in a dialog; the guide's card shows them in place, and a to-do's
// Complete setup opens the setup's dialog with the way back to the guide.
func TestImportReportGroups(t *testing.T) {
	home := t.TempDir()
	if err := os.MkdirAll(filepath.Join(home, "Library", "Containers", legacyBundleID), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VOXTNEW_LEGACY_HOME", home)
	a, _, tt := newTestApp(t)
	a.importLegacyFrom(copyFixture(t, "hist"))
	if !a.ob.legacyOpen || !has(tt, "Imported from the previous Voxt") || !hasExact(tt, "Imported") || !has(tt, "History: 3 entries") ||
		!hasExact(tt, "Left out") || !has(tt, "1 history entry: empty") || hasExact(tt, "To do") {
		t.Fatalf("dialog: %q", tt.Texts())
	}
	click(t, tt, "Done")
	a.ob.legacy.todo = []legacyTodo{{text: setupNeedsText(local.LLM, settings.Provider{ID: "deepseek"}, providers.MissingKey), kind: local.LLM, id: "deepseek"}}
	a.openOnboarding()
	if !has(tt, "History: 3 entries") || !hasExact(tt, "To do") || !has(tt, "DeepSeek needs its API key. Paste it again to use it.") {
		t.Fatalf("the guide's card: %q", tt.Texts())
	}
	// With Linux's taller lines the button sits at the card's bottom edge:
	// scroll the card to it first.
	todo, _ := tt.Find("To do")
	tt.Scroll(todo.X+todo.W/2, todo.Y, 0, 200)
	frames(tt)
	if err := tt.ClickControl("Complete setup of DeepSeek"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if a.ob.open || a.shownPath() != remotePath || !a.provDlg.open || a.provDlg.id != "deepseek" || a.ob.awayTo != remotePath {
		t.Errorf("Complete setup: guide %v, %s, dialog %v %q", a.ob.open, a.shownPath(), a.provDlg.open, a.provDlg.id)
	}
}
