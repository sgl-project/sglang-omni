package models

import (
	"encoding/json"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
)

func TestCatalog(t *testing.T) {
	seen := map[string]bool{}
	defaults := map[Kind]int{}
	hexRE := regexp.MustCompile(`^[0-9a-f]{64}$`)
	for _, m := range Catalog() {
		if seen[m.ID] {
			t.Errorf("%s twice", m.ID)
		}
		seen[m.ID] = true
		if m.Default {
			defaults[m.Kind]++
		}
		if !fammodels.IsCommit(m.Revision) || !fammodels.ValidRepo(m.Repo) || m.Size() <= 0 || m.Family == "" {
			t.Errorf("%s: pin %q %q, family %q, %d bytes", m.ID, m.Repo, m.Revision, m.Family, m.Size())
		}
		if m.Borrowed != (len(m.Files) == 0) {
			t.Errorf("%s: borrowed %v with %d files: VoxtNew's own models list their files", m.ID, m.Borrowed, len(m.Files))
		}
		if m.Borrowed && (m.Default || len(m.Needs) > 0) {
			t.Errorf("%s: a borrowed model is never the default", m.ID)
		}
		weights := false
		for _, f := range m.Files {
			if !hexRE.MatchString(f.SHA256) || f.Size <= 0 || f.Path == "" {
				t.Errorf("%s: file %+v", m.ID, f)
			}
			weights = weights || f.LFS && (strings.HasSuffix(f.Path, ".safetensors") || strings.HasSuffix(f.Path, ".gguf"))
			if strings.HasPrefix(f.Path, "optiq/") {
				t.Errorf("%s: %s (the vision tower and MTP head are not used)", m.ID, f.Path)
			}
		}
		if !m.Borrowed && !weights {
			t.Errorf("%s: no weights file in LFS", m.ID)
		}
		for _, n := range append(slices.Clone(m.Needs), m.Alternatives...) {
			if o, ok := Lookup(n); !ok || n == m.ID {
				t.Errorf("%s names %s, not in the catalogue", m.ID, n)
			} else if slices.Contains(m.Alternatives, n) && o.Kind != m.Kind {
				t.Errorf("%s's alternative %s is a %s", m.ID, n, o.Kind)
			}
		}
	}
	for _, k := range []Kind{ASR, LLM, VAD} {
		if defaults[k] != 1 {
			t.Errorf("%d defaults of %s", defaults[k], k)
		}
	}
	for _, id := range []string{ASRAccurate, ASRFast, VADSilero, LLMBalanced, LLMFast, LLMTranslate, LLMQwen3, LLMGemma4} {
		if !seen[id] {
			t.Errorf("%s missing", id)
		}
	}
	if m, _ := Lookup(ASRAccurate); !m.Default || m.Repo != "mlx-community/Qwen3-ASR-1.7B-8bit" {
		t.Errorf("accurate: %+v", m)
	}
	if _, ok := Lookup("nope"); ok {
		t.Error("Lookup(nope)")
	}
}

// TestFamilyPins: VoxtNew's models are the family pins table's rows of
// voxtnew (same commit, family and quantisation, the files it lists); the
// borrowed ones are rows of the apps that install them, and the language
// models' alternatives are the family's Qwen3 and Gemma 4 E4B.
func TestFamilyPins(t *testing.T) {
	mine := map[string]fammodels.Pin{}
	for _, p := range fammodels.PinsOfApp("voxtnew") {
		mine[p.Repo] = p
	}
	for _, m := range Catalog() {
		p, ok := fammodels.PinOf(m.Repo)
		if !ok {
			t.Errorf("%s: %s is not in the family pins table", m.ID, m.Repo)
			continue
		}
		if p.Revision != m.Revision || p.Family != m.Family || p.Quant != m.Quant || p.Role != m.Kind.Role() {
			t.Errorf("%s: pinned %s %s %s %s, the family %s %s %s %s", m.ID, m.Revision, m.Family, m.Quant, m.Kind.Role(), p.Revision, p.Family, p.Quant, p.Role)
		}
		if _, listed := mine[m.Repo]; listed == m.Borrowed {
			t.Errorf("%s: borrowed %v, the pins table lists voxtnew %v", m.ID, m.Borrowed, listed)
		}
		if m.Borrowed && p.Bytes != m.Bytes {
			t.Errorf("%s: %d bytes, the pin %d", m.ID, m.Bytes, p.Bytes)
		}
		if len(p.Files) > 0 {
			var paths []string
			for _, f := range m.Files {
				paths = append(paths, f.Path)
			}
			if !slices.Equal(paths, p.Files) {
				t.Errorf("%s: files %v, the pin %v", m.ID, paths, p.Files)
			}
		}
		delete(mine, m.Repo)
	}
	delete(mine, "mlx-community/whisper-large-v3-turbo") // P1: not in the catalogue yet
	for repo := range mine {
		t.Errorf("the pins table has %s for voxtnew, the catalogue does not", repo)
	}
	family := func(f string) []string {
		var out []string
		for _, p := range fammodels.PinsOfFamily(f) {
			out = append(out, p.Repo)
		}
		return out
	}
	q, _ := Lookup(LLMQwen3)
	g, _ := Lookup(LLMGemma4)
	if !slices.Contains(family("qwen3"), q.Repo) || !slices.Contains(family("gemma4"), g.Repo) || !strings.Contains(g.Repo, "e4b") {
		t.Errorf("borrowed %s %s, the family's %v %v", q.Repo, g.Repo, family("qwen3"), family("gemma4"))
	}
}

// TestOmniPins: the omni models are pinned as the native runtime's CI
// pins them.
func TestOmniPins(t *testing.T) {
	b, err := os.ReadFile("../../../sglang_omni_mlx/native/ci/models.json")
	if err != nil {
		t.Skip("sglang_omni_mlx/native/ci/models.json not there:", err)
	}
	var pins map[string]struct {
		Revision string            `json:"revision"`
		Files    map[string]string `json:"files"`
	}
	if err := json.Unmarshal(b, &pins); err != nil {
		t.Fatal(err)
	}
	for _, m := range Catalog() {
		if m.Runtime != Omni {
			continue
		}
		p, ok := pins[m.Repo]
		if !ok || p.Revision != m.Revision || len(p.Files) != len(m.Files) {
			t.Errorf("%s: pinned %+v", m.ID, p)
			continue
		}
		for _, f := range m.Files {
			if p.Files[f.Path] != f.SHA256 {
				t.Errorf("%s: %s is %s in models.json", m.ID, f.Path, p.Files[f.Path])
			}
		}
	}
}

// TestEntry: a model is downloaded as hfhub's entry of its pinned commit
// with only the files it lists, the LFS ones carrying their SHA-256 (the
// Hub names the others by git id: a SHA-256 there would fail the plan),
// and claimed with its id, family, role and files.
func TestEntry(t *testing.T) {
	acc, _ := Lookup(ASRAccurate)
	e := acc.Entry("")
	if e.ID != ASRAccurate || e.Repo != acc.Repo || e.Revision != acc.Revision || e.Family != "qwen3-asr" || e.Role != fammodels.RoleASR || len(e.Files) != len(acc.Files) {
		t.Fatalf("%+v", e)
	}
	for i, f := range e.Files {
		want := ""
		if acc.Files[i].LFS {
			want = acc.Files[i].SHA256
		}
		if f.Path != acc.Files[i].Path || f.Size != acc.Files[i].Size || f.SHA256 != want {
			t.Errorf("file %+v, catalogue %+v", f, acc.Files[i])
		}
	}
	tr, _ := Lookup(LLMTranslate)
	if e := tr.Entry(""); len(e.Files) != 1 || e.Files[0].Path != "Hy-MT2-1.8B-Q4_K_M.gguf" || e.Files[0].SHA256 == "" {
		t.Errorf("translate %+v", e)
	}
	b, _ := Lookup(LLMBalanced)
	if e := b.Entry("abc"); e.Revision != "abc" || slices.ContainsFunc(e.Files, func(f hfhub.File) bool { return strings.HasPrefix(f.Path, "optiq/") }) {
		t.Errorf("balanced %+v", e)
	}
	c := tr.Claim("", fammodels.OriginReused)
	if c.Repo != tr.Repo || c.Revision != tr.Revision || c.Model != LLMTranslate || c.Role != fammodels.RoleLLM ||
		c.Family != "hy-mt2" || !slices.Equal(c.Files, []string{"Hy-MT2-1.8B-Q4_K_M.gguf"}) || c.Origin != fammodels.OriginReused {
		t.Errorf("claim %+v", c)
	}
	if q, _ := Lookup(LLMQwen3); q.Entry("").Files != nil || q.Claim("", "").Files != nil {
		t.Error("a borrowed model is the whole snapshot")
	}
	if got, want := SnapshotDir("/h", acc, ""), "/h/models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"+acc.Revision; got != want {
		t.Errorf("snapshot %s, want %s", got, want)
	}
}
