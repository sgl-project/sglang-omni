package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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
	if !has(tt, "Type the term as it should be written.") {
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

// TestDictionaryGuide: the guide says where a term applies, in both
// languages, with the numbers the composer uses.
func TestDictionaryGuide(t *testing.T) {
	_, tt := dictTester(t)
	title := "How the dictionary works"
	if !has(tt, title) || !slices.Contains(tt.AccessLabels(), title) {
		t.Fatalf("no guide: %q", firstTexts(tt))
	}
	if has(tt, "Aliases are not sent") {
		t.Errorf("the guide starts open: %q", firstTexts(tt))
	}
	click(t, tt, title)
	for _, want := range []string{
		"is used in three places",
		"as words to expect. Aliases are not sent",
		fmt.Sprintf("With more than %d terms", compose.GlossaryAllUpTo),
		fmt.Sprintf("are sent, up to %d.", compose.MaxGlossaryTerms),
		"even without a language model",
	} {
		if !has(tt, want) {
			t.Errorf("no %q in %q", want, firstTexts(tt))
		}
	}
	i18n.SetLocale("zh-Hans")
	for _, want := range []string{
		"词典的用法",
		"启用的词条用在三个地方",
		"别名不发送",
		fmt.Sprintf("超过 %d 个时，只发送文本里出现的词条", compose.GlossaryAllUpTo),
		fmt.Sprintf("最多 %d 个", compose.MaxGlossaryTerms),
		"没有语言模型也有效",
	} {
		if !has(tt, want) {
			t.Errorf("no %q in %q", want, firstTexts(tt))
		}
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
	click(t, tt, "Search dictionary")
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
	if a.lib.settings.Get().Dictionary.Enabled || !has(tt, "The dictionary is off, so its terms are not used.") {
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
		chooseExportFile = func(a *app, _, name string, then func(string)) {
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
func dictShots(t *testing.T, a *app, tt *ui.Tester, save func(string)) {
	seedTerms(t, a)
	a.navigate("/dictionary")
	a.loadDict()
	save("page-dictionary-terms")
	a.suggestions = []learn.Suggestion{
		{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: learn.FromEdit},
		{Term: "Ghostty", Source: learn.FromHistory},
	}
	save("page-dictionary-suggestions")
	a.suggestions = nil
	a.dv.scan = scanState{open: true, model: fakeLLMModel, counted: true, count: 24}
	save("dictionary-scan")
	a.dv.scan = scanState{open: true, running: true, read: 12, total: 24, found: 3, model: fakeLLMModel}
	save("dictionary-scan-running")
	a.dv.scan = scanState{}
	a.openTerm(a.dv.list[0])
	save("dictionary-edit")
	a.dv.editOpen = false
	// The Personal dictionary view: empty, the reporter's table and its
	// example, 200 rows.
	a.dv.tab = dictNotesTab
	save("page-dictionary-notes-empty")
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = reporterNotes(t) })
	save("page-dictionary-notes")
	scrollToText(tt, i18n.T("How it is used"))
	save("page-dictionary-notes-example")
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = longNotes(200) })
	frames(tt)
	scrollToText(tt, "| Term199 |")
	save("page-dictionary-notes-long")
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = "" })
	a.dv.tab = dictTermsTab
}

// reporterNotes is issue #3815's personal dictionary.
func reporterNotes(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/dictionary.md")
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(b))
}

// longNotes is a table of n rows.
func longNotes(n int) string {
	var b strings.Builder
	b.WriteString("| Standard | Heard as | When |\n|---|---|---|\n")
	for i := range n {
		fmt.Fprintf(&b, "| Term%03d | term %d | row %d |\n", i, i, i)
	}
	return b.String()
}

// TestDictionaryNotes: the Personal dictionary view saves the Markdown
// as typed, its switch keeps it out of the prompt, and How it is used
// shows the request a session sends with the settings, in both
// languages; a 200 row table stays whole.
func TestDictionaryNotes(t *testing.T) {
	a, tt := dictTester(t)
	seedTerms(t, a)
	a.loadDict()
	frames(tt)
	clickSegment(t, tt, "Dictionary view", dictNotesTab, 2)
	if !has(tt, "Add to the language model prompt") || !has(tt, "How it is used") || has(tt, "Search dictionary") {
		t.Fatalf("not the notes view: %q", firstTexts(tt))
	}
	if !has(tt, "(not sent: empty or switched off)") {
		t.Errorf("an empty dictionary is sent: %q", firstTexts(tt))
	}
	notes := reporterNotes(t)
	click(t, tt, "Personal dictionary in Markdown")
	tt.Type(notes)
	frames(tt)
	click(t, tt, "How it is used") // the field loses the focus: saved
	d := a.lib.settings.Get().Dictionary
	if d.Notes != notes || d.NotesOff {
		t.Fatalf("saved %+v", d)
	}
	// The example is the request a session builds (session.PromptDictionary).
	out, err := compose.Build(a.notesExampleInput(hotkey.Transcribe))
	if err != nil {
		t.Fatal(err)
	}
	sys := out.Messages()[0].Content
	if !strings.Contains(sys, "`罗斯` → Roth") || !strings.Contains(sys, "- VoxtNew") || strings.Contains(sys, "MyGo") ||
		strings.Index(sys, "</dictionary>") > strings.Index(sys, "- VoxtNew") || !has(tt, sys) || !has(tt, out.Dictionary) {
		t.Errorf("the example's system prompt:\n%s", sys)
	}
	if !has(tt, trySample()) || !has(tt, "+ Speech recognition text") || !has(tt, "= What the language model receives") {
		t.Errorf("the example's parts: %q", tt.Texts())
	}
	clickSegment(t, tt, "Example for", 2, 3)
	instr, sel := rewriteSample()
	if !has(tt, sel) || !has(tt, instr) || !has(tt, "+ Your input (the selected text)") {
		t.Errorf("Rewrite's example: %q", tt.Texts())
	}
	// Off: kept, not sent.
	clickControl(t, tt, "Add to the language model prompt")
	if d := a.lib.settings.Get().Dictionary; !d.NotesOff || d.Notes != notes || !has(tt, "(not sent: empty or switched off)") {
		t.Errorf("switched off: %+v", d)
	}
	if out, _ := compose.Build(a.notesExampleInput(hotkey.Rewrite)); strings.Contains(out.System, "罗斯") {
		t.Errorf("sent while off")
	}
	clickControl(t, tt, "Add to the language model prompt")
	// Shown again from the settings, as after a restart.
	a.dv.notes = speechContextState{}
	if !has(tt, "| 401(k); Roth |") || a.dv.notes.text != notes {
		t.Errorf("not read back: %q", a.dv.notes.text)
	}
	// 200 rows: whole in the prompt, and the page still draws quickly.
	long := longNotes(200)
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = long })
	start := time.Now()
	for range 10 {
		tt.Frame()
	}
	if took := time.Since(start); took > 3*time.Second {
		t.Errorf("10 frames took %v", took)
	}
	if out, _ := compose.Build(a.notesExampleInput(hotkey.Translate)); !strings.Contains(out.System, "| Term199 | term 199 | row 199 |") {
		t.Errorf("the long table is cut")
	}
	i18n.SetLocale("zh-Hans")
	for _, want := range []string{"个人词典", "加到语言模型的提示词里", "怎么使用", "语言模型收到的内容"} {
		if !has(tt, want) {
			t.Errorf("no %q in %q", want, firstTexts(tt))
		}
	}
}

// TestDictionaryNotesExampleCached: How it is used builds its request
// once, not on every frame, and builds it again whenever what it shows
// depends on changes: the notes as typed, the example's feature, a term
// read back, the dictionary's switches, the language.
func TestDictionaryNotesExampleCached(t *testing.T) {
	a, tt := dictTester(t)
	a.loadDict()
	frames(tt)
	clickSegment(t, tt, "Dictionary view", dictNotesTab, 2)
	system := func() string {
		t.Helper()
		frames(tt)
		if len(a.dv.example.msgs) == 0 {
			t.Fatal("no example")
		}
		return a.dv.example.msgs[0].Content
	}
	// An empty dictionary: nothing to send, and frames reuse the build.
	if !has(tt, "(not sent: empty or switched off)") {
		t.Fatalf("empty: %q", firstTexts(tt))
	}
	built := &a.dv.example.msgs[0]
	for range 10 {
		tt.Frame()
	}
	if &a.dv.example.msgs[0] != built {
		t.Fatal("the example is built again on a frame that changes nothing")
	}
	// Terms read back join the glossary.
	seedTerms(t, a)
	a.loadDict()
	if s := system(); !strings.Contains(s, "- VoxtNew") || !has(tt, s) {
		t.Fatalf("new terms not in the example:\n%s", s)
	}
	// Notes as typed, Chinese and English mixed, before they are saved.
	click(t, tt, "Personal dictionary in Markdown")
	tt.Type("| 罗斯 Roth | wrath |")
	if s := system(); !strings.Contains(s, "| 罗斯 Roth | wrath |") {
		t.Fatalf("typed notes not in the example:\n%s", s)
	}
	// Another feature.
	clickSegment(t, tt, "Example for", 2, 3)
	if _, sel := rewriteSample(); !has(tt, sel) {
		t.Fatalf("Rewrite's example: %q", firstTexts(tt))
	}
	// The glossary switched off, then the whole dictionary.
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Glossary = false })
	if s := system(); strings.Contains(s, "- VoxtNew") || !strings.Contains(s, "罗斯 Roth") {
		t.Fatalf("glossary off:\n%s", s)
	}
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Enabled = false })
	if s := system(); strings.Contains(s, "罗斯 Roth") {
		t.Fatalf("dictionary off:\n%s", s)
	}
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Enabled, s.Dictionary.Glossary = true, true })
	// The language: the sample text is in the new one.
	en := system()
	i18n.SetLocale("zh-Hans")
	if zh := system(); zh == en {
		t.Fatal("the example keeps the old language")
	}
	i18n.SetLocale("en")
}
