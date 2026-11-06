package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
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
// a meeting transcript and an empty entry are skipped.
func TestReadLegacyHistory(t *testing.T) {
	entries, skipped, err := readLegacyHistory(context.Background(), legacyDatabase(copyFixture(t, "hist")))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 || skipped != 2 {
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
