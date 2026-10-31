package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// seedTerms puts sample terms in the dictionary.
func seedTerms(t *testing.T, a *app) {
	t.Helper()
	t0 := time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)
	for i, tm := range []store.Term{
		{ID: "t1", Term: "VoxtNew", Aliases: []string{"aura link", "奥拉林克"}, Note: "Our app", Enabled: true},
		{ID: "t2", Term: "Kubernetes", Aliases: []string{"cooper netties", "k8s"}, Enabled: true},
		{ID: "t3", Term: "豆包", Aliases: []string{"逗包"}, Note: "火山引擎的大模型", Enabled: true},
		{ID: "t4", Term: "MyGo", Enabled: false, Note: "UI toolkit"},
		{ID: "t5", Term: "DeepSeek", Aliases: []string{"deep seek"}, Enabled: true},
	} {
		tm.CreatedAt, tm.UpdatedAt = t0.Add(time.Duration(i)*time.Minute), t0.Add(time.Duration(i)*time.Minute)
		if err := a.lib.store.Dictionary().Put(context.Background(), tm); err != nil {
			t.Fatal(err)
		}
	}
}

func dictTester(t *testing.T) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	a.navigate("/dictionary")
	frames(tt)
	return a, tt
}

func terms(t *testing.T, a *app) []store.Term {
	t.Helper()
	ts, err := a.lib.store.Dictionary().List(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	return ts
}

func TestDictionaryAddEditDelete(t *testing.T) {
	a, tt := dictTester(t)
	if !has(tt, "Names and terms you use") {
		t.Fatalf("no empty state: %q", firstTexts(tt))
	}
	click(t, tt, "Add term")
	click(t, tt, "Save")
	if !has(tt, "Type the term as it must be written.") {
		t.Errorf("empty term saved: %q", firstTexts(tt))
	}
	click(t, tt, "Term as written")
	tt.Type("VoxtNew")
	click(t, tt, "Aliases of the term")
	tt.Type("aura link，奥拉林克, aura link")
	click(t, tt, "Save")
	ts := terms(t, a)
	if len(ts) != 1 || ts[0].Term != "VoxtNew" || strings.Join(ts[0].Aliases, "|") != "aura link|奥拉林克" || !ts[0].Enabled || a.dv.editOpen {
		t.Fatalf("terms %+v", ts)
	}
	if !has(tt, "VoxtNew") || !tt.HasText("heard as aura link, 奥拉林克") || !has(tt, "1 term") {
		t.Errorf("list %q", firstTexts(tt))
	}
	// The same spelling again is refused.
	click(t, tt, "Add term")
	click(t, tt, "Term as written")
	tt.Type("voxtnew")
	click(t, tt, "Save")
	if !has(tt, "This term is in the dictionary already.") || len(terms(t, a)) != 1 {
		t.Errorf("duplicate: %q", firstTexts(tt))
	}
	click(t, tt, "Cancel")
	// Edit: a note; switch it off from the list.
	click(t, tt, "Edit VoxtNew")
	click(t, tt, "Note on the term")
	tt.Type("our app")
	click(t, tt, "Save")
	if ts := terms(t, a); ts[0].Note != "our app" || ts[0].ID == "" {
		t.Errorf("edited %+v", ts)
	}
	click(t, tt, "Use VoxtNew")
	if ts := terms(t, a); ts[0].Enabled {
		t.Errorf("still on %+v", ts)
	}
	click(t, tt, "Delete VoxtNew")
	if len(terms(t, a)) != 0 || !has(tt, "Names and terms you use") {
		t.Errorf("not deleted: %q", firstTexts(tt))
	}
}

func TestDictionarySearchAndSwitches(t *testing.T) {
	a, tt := dictTester(t)
	seedTerms(t, a)
	a.loadDict()
	frames(tt)
	if !has(tt, "5 terms") || !has(tt, "豆包") {
		t.Fatalf("list %q", firstTexts(tt))
	}
	click(t, tt, "Search the dictionary")
	tt.Type("k8s")
	frames(tt)
	if len(a.dv.list) != 1 || a.dv.list[0].Term != "Kubernetes" {
		t.Errorf("search %+v", a.dv.list)
	}
	a.dv.query = "nothing"
	a.loadDict()
	if !has(tt, "No term matches “nothing”") {
		t.Errorf("no empty search state %q", firstTexts(tt))
	}
	click(t, tt, "Speech hints")
	click(t, tt, "Glossary for the language model")
	if d := a.lib.settings.Get().Dictionary; d.ASRHints || d.Glossary || !d.Enabled {
		t.Errorf("settings %+v", d)
	}
	click(t, tt, "Use the dictionary")
	if a.lib.settings.Get().Dictionary.Enabled || !has(tt, "The dictionary is off: its terms go nowhere.") {
		t.Error("dictionary still on")
	}
}

func TestDictionaryImportExport(t *testing.T) {
	a, tt := dictTester(t)
	seedTerms(t, a)
	dir := t.TempDir()
	in := filepath.Join(dir, "terms.csv")
	os.WriteFile(in, []byte("term,aliases,note\nvoxtnew,VoxtNew App;aura-link,\nGemini,gemeni,Google\n,,\n"), 0o644)
	saved := chooseImportFile
	t.Cleanup(func() { chooseImportFile = saved })
	chooseImportFile = func(a *app, then func(string)) { then(in) }
	a.importTerms()
	frames(tt)
	ts := terms(t, a)
	if len(ts) != 6 {
		t.Fatalf("terms %+v", ts)
	}
	if !strings.Contains(strings.Join(a.toasts, "\n"), "Imported: 1 new, 1 updated, 1 rows skipped.") && !has(tt, "Imported: 1 new, 1 updated, 1 rows skipped.") {
		t.Errorf("toasts %q", a.toasts)
	}
	for _, tm := range ts {
		if tm.Term == "VoxtNew" && strings.Join(tm.Aliases, "|") != "aura link|奥拉林克|VoxtNew App|aura-link" {
			t.Errorf("merged %+v", tm)
		}
	}
	// Export both formats.
	savedOut := chooseExportFile
	t.Cleanup(func() { chooseExportFile = savedOut })
	for _, format := range []string{"csv", "txt"} {
		out := filepath.Join(dir, "out."+format)
		chooseExportFile = func(a *app, name string, then func(string)) {
			if name != "voxtnew-dictionary."+format {
				t.Errorf("name %q", name)
			}
			then(out)
		}
		a.exportTerms(format)
		data, err := os.ReadFile(out)
		if err != nil || !strings.Contains(string(data), "Gemini") || !strings.Contains(string(data), "豆包") {
			t.Errorf("%s export %q %v", format, data, err)
		}
	}
	// A file that is no list.
	bad := filepath.Join(dir, "bad.csv")
	os.WriteFile(bad, []byte{0xff, 0xfe, 0, 1}, 0o644)
	chooseImportFile = func(a *app, then func(string)) { then(bad) }
	a.toasts = nil
	a.importTerms()
	if len(a.toasts) == 0 || !strings.Contains(a.toasts[0], "not a CSV or text list") {
		t.Errorf("bad import %q", a.toasts)
	}
}

// dictShots renders the Dictionary page with terms and its dialog
// (TestSnapshots).
func dictShots(t *testing.T, a *app, save func(string)) {
	seedTerms(t, a)
	a.navigate("/dictionary")
	a.loadDict()
	save("page-dictionary-terms")
	a.openTerm(a.dv.list[0])
	save("dictionary-edit")
	a.dv.editOpen = false
}
