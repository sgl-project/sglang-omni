package models

import (
	"encoding/json"
	"os"
	"path/filepath"
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
			weights = weights || f.LFS && slices.ContainsFunc([]string{".safetensors", ".gguf", ".npz"}, func(ext string) bool { return strings.HasSuffix(f.Path, ext) })
			if strings.HasPrefix(f.Path, "optiq/") {
				t.Errorf("%s: %s (the vision tower and MTP head are not used)", m.ID, f.Path)
			}
		}
		if !m.Borrowed && m.Kind != Aux && !weights {
			t.Errorf("%s: no weights file in LFS", m.ID)
		}
		if m.Kind == Aux && (m.Default || len(m.Needs) > 0 || m.Runtime != Omni) {
			t.Errorf("%s: an aux is the files of an omni model, needing nothing", m.ID)
		}
		for _, n := range append(slices.Clone(m.Needs), m.Alternatives...) {
			if o, ok := Lookup(n); !ok || n == m.ID {
				t.Errorf("%s names %s, not in the catalogue", m.ID, n)
			} else if slices.Contains(m.Alternatives, n) && o.Kind != m.Kind {
				t.Errorf("%s's alternative %s is a %s", m.ID, n, o.Kind)
			}
		}
		if m.Draft != "" {
			if d, ok := Lookup(m.Draft); !ok || d.Kind != Draft || d.Runtime != m.Runtime || m.Kind != ASR {
				t.Errorf("%s's draft %s: %+v", m.ID, m.Draft, d)
			}
		}
		if m.Kind == Draft && (m.Default || len(m.Needs) > 0 || len(m.Alternatives) > 0 || m.Draft != "" || m.Borrowed) {
			t.Errorf("%s: a draft is never the default and names no other model", m.ID)
		}
	}
	owners := map[string]int{} // a draft belongs to one speech model
	for _, m := range Catalog() {
		if m.Draft != "" {
			owners[m.Draft]++
		}
	}
	for _, m := range Catalog() {
		if m.Kind == Draft && owners[m.ID] != 1 {
			t.Errorf("%s is the draft of %d speech models", m.ID, owners[m.ID])
		}
	}
	for _, k := range []Kind{ASR, LLM, VAD} {
		if defaults[k] != 1 {
			t.Errorf("%d defaults of %s", defaults[k], k)
		}
	}
	for _, id := range []string{ASRAccurate, ASRFast, VADSilero, LLMBalanced, LLMFast, LLMTranslate, LLMQwen3, LLMGemma4,
		ASRQwen6bit, ASRWhisperTurbo, ASRWhisperLarge, ASRWhisperSmall, ASRParakeet, ASRSenseVoice, ASRCohere, ASRNemotron, ASRMoss, AuxWhisperLarge, AuxWhisperSmall,
		ASRAccurateDraft, ASRFastDraft} {
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
// voxtnew (same commit, family and quantisation, the files it lists, the
// aux repos its tokenizer comes from); the borrowed ones are rows of the
// apps that install them, and the language models' alternatives are the
// family's Qwen3 and Gemma 4 E4B. Voxt 1.x's other speech models and the
// Whisper tokenizers may be missing from the table (no other ActNow app
// installs them); nothing reads it at run time.
func TestFamilyPins(t *testing.T) {
	// The drafts are proposed to the family pins table too; until an
	// actnowkit release lists them, TestOmniPins alone checks their pins.
	unpinned := []string{ASRQwen6bit, ASRWhisperLarge, ASRWhisperSmall, ASRParakeet, ASRSenseVoice, ASRCohere, ASRNemotron, ASRMoss, AuxWhisperLarge, AuxWhisperSmall,
		ASRAccurateDraft, ASRFastDraft}
	mine := map[string]fammodels.Pin{}
	for _, p := range fammodels.PinsOfApp("voxtnew") {
		mine[p.Repo] = p
	}
	for _, m := range Catalog() {
		p, ok := fammodels.PinOf(m.Repo)
		if !ok {
			if !slices.Contains(unpinned, m.ID) {
				t.Errorf("%s: %s is not in the family pins table", m.ID, m.Repo)
			}
			continue
		}
		if p.Revision != m.Revision || p.Family != m.Family || p.Quant != m.Quant || p.Role != m.Kind.Role() {
			t.Errorf("%s: pinned %s %s %s %s, the family %s %s %s %s", m.ID, m.Revision, m.Family, m.Quant, m.Kind.Role(), p.Revision, p.Family, p.Quant, p.Role)
		}
		if _, listed := mine[m.Repo]; listed == m.Borrowed && m.Kind != Aux {
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
		var aux []fammodels.Ref
		for _, n := range m.Needs {
			if o, _ := Lookup(n); o.Kind == Aux {
				aux = append(aux, fammodels.Ref{Repo: o.Repo, Revision: o.Revision})
			}
		}
		if len(p.Aux) > 0 && !slices.Equal(aux, p.Aux) {
			t.Errorf("%s: aux %v, the pin %v", m.ID, aux, p.Aux)
		}
		delete(mine, m.Repo)
	}
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

// omniPin is a repo's row in the native runtime's CI pins.
type omniPin struct {
	Revision  string            `json:"revision"`
	Files     map[string]string `json:"files"`
	Tokenizer *struct {
		Repo     string            `json:"repo"`
		Revision string            `json:"revision"`
		Files    map[string]string `json:"files"`
	} `json:"tokenizer"`
}

// TestOmniPins: the omni models are pinned as the native runtime's CI
// pins them (models.json, model_pins.json), with every file it checks;
// a Whisper model's tokenizer is its pin's tokenizer.
func TestOmniPins(t *testing.T) {
	pins := map[string]omniPin{}
	for _, name := range []string{"models.json", "model_pins.json"} {
		b, err := os.ReadFile("../../../sglang_omni_mlx/native/ci/" + name)
		if err != nil {
			t.Skip("sglang_omni_mlx/native/ci not there:", err)
		}
		if err := json.Unmarshal(b, &pins); err != nil {
			t.Fatal(name, err)
		}
	}
	same := func(id, where string, files []File, rev, wantRev string, want map[string]string) {
		if rev != wantRev || len(files) != len(want) {
			t.Errorf("%s: %s pins %s with %d files, the catalogue %s with %d", id, where, wantRev, len(want), rev, len(files))
			return
		}
		for _, f := range files {
			if want[f.Path] != f.SHA256 {
				t.Errorf("%s: %s is %s in %s", id, f.Path, want[f.Path], where)
			}
		}
	}
	for _, m := range Catalog() {
		if m.Runtime != Omni || m.Kind == Aux {
			continue
		}
		p, ok := pins[m.Repo]
		if !ok {
			t.Errorf("%s: %s is not pinned by the native CI", m.ID, m.Repo)
			continue
		}
		same(m.ID, "the native CI", m.Files, m.Revision, p.Revision, p.Files)
		var aux []Model
		for _, n := range m.Needs {
			if o, _ := Lookup(n); o.Kind == Aux {
				aux = append(aux, o)
			}
		}
		switch {
		case p.Tokenizer == nil && len(aux) == 0:
		case p.Tokenizer == nil || len(aux) != 1 || aux[0].Repo != p.Tokenizer.Repo:
			t.Errorf("%s: tokenizer %v, the native CI's %+v", m.ID, aux, p.Tokenizer)
		default:
			same(aux[0].ID, "the native CI's tokenizer", aux[0].Files, aux[0].Revision, p.Tokenizer.Revision, p.Tokenizer.Files)
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

// TestSameVolume: two folders of one volume are; /dev (devfs) and a
// folder that does not exist are not.
func TestSameVolume(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	if !SameVolume(a, b) {
		t.Error("two temporary folders are not on one volume")
	}
	if SameVolume(a, "/dev") || SameVolume(a, filepath.Join(b, "missing")) {
		t.Error("/dev, or a missing folder, counted as on the same volume")
	}
}
