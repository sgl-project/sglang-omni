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

func TestDictionarySuggestionCaseChangeRebuildsLabel(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.suggestions = []learn.Suggestion{{Term: "Ghostty", Source: learn.FromHistory}}
	first := a.dictionaryPageModel()
	a.suggestions[0].Term = "ghostty"
	second := a.dictionaryPageModel()
	if len(first.Suggestions) != 1 || len(second.Suggestions) != 1 || first.Suggestions[0].Term != "Ghostty" || second.Suggestions[0].Term != "ghostty" {
		t.Fatalf("case-only suggestion change kept stale labels %+v and %+v", first.Suggestions, second.Suggestions)
	}
}

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

func chooseDictionaryMenuItem(t *testing.T, tester *ui.Tester, label string) {
	t.Helper()
	clickControl(t, tester, i18n.T("More actions"))
	if err := tester.ChooseMenuItem(label); err != nil {
		t.Fatal(err)
	}
	frames(tester)
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
	if !has(tt, "No terms yet") {
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
	if len(ts) != 1 || ts[0].Term != "VoxtNew" || strings.Join(ts[0].Aliases, "|") != "aura link|奥拉林克" || !ts[0].Enabled || a.dictUI.editOpen {
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
	tt.Move(1, 1) // off the row the switch's click left the pointer on
	frames(tt)
	if slices.Contains(tt.AccessLabels(), "Delete VoxtNew") {
		t.Errorf("Delete shows without the pointer on the row: %q", tt.AccessLabels())
	}
	hoverText(t, tt, "VoxtNew")
	click(t, tt, "Delete VoxtNew")
	if len(terms(t, a)) != 0 || !has(tt, "No terms yet") {
		t.Errorf("not deleted: %q", firstTexts(tt))
	}
}

// TestDictionarySettingsTips: the settings dialog's ⓘ say where a term
// goes, in both languages, with the numbers the composer uses.
func TestDictionarySettingsTips(t *testing.T) {
	_, tt := dictTester(t)
	if has(tt, "Speech hints") {
		t.Fatalf("the switches show on the page: %q", firstTexts(tt))
	}
	click(t, tt, "Dictionary settings")
	tips := func(wants ...string) {
		t.Helper()
		labels := strings.Join(tt.AccessLabels(), "\n")
		for _, want := range wants {
			if !strings.Contains(labels, want) {
				t.Errorf("no %q in %q", want, tt.AccessLabels())
			}
		}
	}
	tips("Terms go to the speech model as words to expect.",
		fmt.Sprintf("With more than %d terms", compose.GlossaryAllUpTo),
		fmt.Sprintf("only those in the text, up to %d.", compose.MaxGlossaryTerms),
		"If you correct a word, it suggests your spelling.")
	i18n.SetLocale("zh-Hans")
	defer i18n.SetLocale("en")
	frames(tt)
	tips("让语音识别更容易认出这些词",
		fmt.Sprintf("超过 %d 个时，只用文中出现的", compose.GlossaryAllUpTo),
		fmt.Sprintf("最多 %d 个", compose.MaxGlossaryTerms))
}

func TestDictionaryControlExplanations(t *testing.T) {
	english := []string{
		"Terms go to the speech model as words to expect.",
		fmt.Sprintf("The language model writes terms as spelled here. With more than %d terms, it uses only those in the text, up to %d.", compose.GlossaryAllUpTo, compose.MaxGlossaryTerms),
		fmt.Sprintf("For %d seconds after inserting, Voxt reads the text around it. If you correct a word, it suggests your spelling. Nothing it reads is saved or sent.", learn.WatchSeconds),
	}
	chinese := []string{
		"让语音识别更容易认出这些词。",
		fmt.Sprintf("语言模型会按这里的写法输出词条。词条超过 %d 个时，只用文中出现的，最多 %d 个。", compose.GlossaryAllUpTo, compose.MaxGlossaryTerms),
		fmt.Sprintf("插入后的 %d 秒内，Voxt 会读取插入处附近的文字。你改了某个词，它会建议加入词典。读到的内容不保存，也不发送。", learn.WatchSeconds),
	}
	for _, test := range []struct {
		name          string
		locale        string
		dictionaryOff bool
		want          []string
	}{
		{name: "English", locale: "en", want: english},
		{name: "Chinese", locale: "zh-Hans", want: chinese},
		{name: "available before enabling the dictionary", locale: "en", dictionaryOff: true, want: english},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, tt := dictTester(t)
			i18n.SetLocale(test.locale)
			frames(tt)
			if test.dictionaryOff {
				chooseDictionaryMenuItem(t, tt, i18n.T("Use the dictionary"))
			}
			helpLabel := ""
			for _, label := range tt.AccessLabels() {
				if strings.Contains(label, test.want[0]) {
					helpLabel = label
					break
				}
			}
			if helpLabel == "" {
				t.Fatalf("no Speech hints help in %q", tt.AccessLabels())
			}
			clickControl(t, tt, helpLabel)
			for _, want := range test.want {
				if !has(tt, want) {
					t.Errorf("no %q in %q", want, firstTexts(tt))
				}
			}
		})
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
	a.dictUI.query = "nothing"
	a.loadDict()
	if !has(tt, "No term matches “nothing”") {
		t.Errorf("no empty search state %q", firstTexts(tt))
	}
	for _, label := range []string{"Speech hints", "Glossary for the language model"} {
		chooseDictionaryMenuItem(t, tt, label)
	}
	if d := a.lib.settings.Get().Dictionary; d.ASRHints || d.Glossary || !d.Enabled {
		t.Errorf("settings %+v", d)
	}
	chooseDictionaryMenuItem(t, tt, "Use the dictionary")
	if a.lib.settings.Get().Dictionary.Enabled || !has(tt, "The dictionary is off, so its terms are not used.") {
		t.Error("dictionary still on")
	}
	clickControl(t, tt, "Turn on the dictionary")
	if !a.lib.settings.Get().Dictionary.Enabled || has(tt, "The dictionary is off, so its terms are not used.") {
		t.Error("Turn on left the dictionary off")
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
	if err := a.lib.store.Dictionary().CreateProfile(context.Background(), store.DictionaryProfile{ID: "work", Name: "Work"}); err != nil {
		t.Fatal(err)
	}
	a.loadDictionaryProfiles()
	save("page-dictionary-profiles")
	if err := a.lib.store.Dictionary().DeleteProfile(context.Background(), "work"); err != nil {
		t.Fatal(err)
	}
	a.loadDictionaryProfiles()
	save("page-dictionary-terms")
	a.suggestions = []learn.Suggestion{
		{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: learn.FromEdit},
		{Term: "Ghostty", Source: learn.FromHistory},
	}
	save("page-dictionary-suggestions")
	a.suggestions = nil
	a.dv.scan = scanState{model: fakeLLMModel, counted: true, count: 24}
	a.dictUI.scanOpen = true
	save("dictionary-scan")
	a.dv.scan = scanState{running: true, read: 12, total: 24, found: 3, model: fakeLLMModel}
	save("dictionary-scan-running")
	a.dv.scan = scanState{}
	a.dictUI.scanOpen = false
	a.openTerm(a.dv.list[0])
	save("dictionary-edit")
	a.dictUI.editOpen = false
	// The Personal dictionary view: empty, the reporter's table.
	a.dictUI.tab = dictNotesTab
	save("page-dictionary-notes-empty")
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = reporterNotes(t) })
	save("page-dictionary-notes")
	a.dictUI.notesQuery, a.dictUI.notesFind = "401", finder{cur: 2}
	save("page-dictionary-notes-find")
	a.dictUI.notesQuery, a.dictUI.notesFind = "", finder{}
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = longNotes(200) })
	frames(tt)
	scrollToText(tt, "| Term199 |")
	save("page-dictionary-notes-long")
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = "" })
	a.dictUI.tab = dictTermsTab
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
// as typed, its switch keeps it out of the prompt a session sends, in
// both languages; a 200 row table stays whole. The view has no example
// of its own: the feature pages show the whole prompt (issue #4032).
func TestDictionaryNotes(t *testing.T) {
	a, tt := dictTester(t)
	seedTerms(t, a)
	a.loadDict()
	frames(tt)
	clickSegment(t, tt, "Dictionary view", dictNotesTab, 2)
	if !has(tt, "Add to the language model prompt") || has(tt, "Search dictionary") {
		t.Fatalf("not the notes view: %q", firstTexts(tt))
	}
	if has(tt, "How it is used") || has(tt, "Example for") || has(tt, "whole request") {
		t.Errorf("the view still has its own example or pointer to one: %q", firstTexts(tt))
	}
	notes := reporterNotes(t)
	click(t, tt, "Personal dictionary in Markdown")
	tt.Type(notes)
	frames(tt)
	click(t, tt, "Search personal dictionary") // the field loses the focus: saved
	d := a.lib.settings.Get().Dictionary
	if d.Notes != notes || d.NotesOff {
		t.Fatalf("saved %+v", d)
	}
	// The request is the one a session builds (session.PromptDictionary).
	out, err := a.composedPreview(hotkey.Transcribe)
	if err != nil {
		t.Fatal(err)
	}
	if sys := out.Messages()[0].Content; !strings.Contains(sys, "`罗斯` → Roth") || strings.Contains(sys, "dictionary>") {
		t.Errorf("the system prompt:\n%s", sys)
	}
	// Off: kept, not sent.
	clickControl(t, tt, "Add to the language model prompt")
	if d := a.lib.settings.Get().Dictionary; !d.NotesOff || d.Notes != notes {
		t.Errorf("switched off: %+v", d)
	}
	if out, _ := a.composedPreview(hotkey.Rewrite); strings.Contains(out.System, "罗斯") {
		t.Errorf("sent while off")
	}
	clickControl(t, tt, "Add to the language model prompt")
	// Shown again from the settings, as after a restart.
	a.dictUI.notes = speechContextState{}
	frames(tt)
	if a.dictUI.notes.text != notes {
		t.Errorf("not read back: %q", a.dictUI.notes.text)
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
	if out, _ := a.composedPreview(hotkey.Translate); !strings.Contains(out.System, "| Term199 | term 199 | row 199 |") {
		t.Errorf("the long table is cut")
	}
	i18n.SetLocale("zh-Hans")
	for _, want := range []string{"个人词典", "加到语言模型的提示词里"} {
		if !has(tt, want) {
			t.Errorf("no %q in %q", want, firstTexts(tt))
		}
	}
}

func TestDictionaryPageModelAppliesCompletedRead(t *testing.T) {
	a, _, _ := newTestApp(t)
	seedTerms(t, a)
	posts := make(chan func(), 8)
	a.postQueue = func(fn func()) { posts <- fn }
	a.navigate("/dictionary")
	model := a.dictionaryPageModel()
	if !model.Loading || len(model.Terms) != 0 {
		t.Fatalf("initial model: loading %t, rows %d", model.Loading, len(model.Terms))
	}
	drainPosts(posts)
	model = a.dictionaryPageModel()
	if model.Loading || len(model.Terms) != 5 {
		t.Fatalf("completed model: loading %t, rows %d", model.Loading, len(model.Terms))
	}
}

// hoverText moves the pointer onto the text s, for the controls that show
// under it.
func hoverText(t *testing.T, tt *ui.Tester, s string) {
	t.Helper()
	frames(tt)
	r, ok := tt.Find(s)
	if !ok {
		t.Fatalf("no %q to hover: %q", s, tt.Texts())
	}
	tt.Move(r.X+r.W/2, r.Y+r.H/2)
	frames(tt)
}
