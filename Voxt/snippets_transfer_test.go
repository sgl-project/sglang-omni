package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// snippetsTester is the Snippets page, read.
func snippetsTester(t *testing.T) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	frames(tt)
	click(t, tt, "Snippets")
	for deadline := time.Now().Add(5 * time.Second); !a.snip.loaded; frames(tt) {
		if time.Now().After(deadline) {
			t.Fatal("the snippets were not read")
		}
	}
	return a, tt
}

func shownTriggers(a *app) []string {
	var out []string
	for _, r := range a.snippetsPageModel().Rows {
		out = append(out, r.Trigger)
	}
	return out
}

// TestSnippetsImport: Import… opens a dialog that shows what the file
// looks like; choosing a Wispr Flow export there reads it, skips what it
// cannot use, closes the dialog and says so in the middle of the page for
// a moment; importing it again changes
// nothing. A file that is not snippets, or has none Voxt can use, says why
// in the dialog, which stays open.
func TestSnippetsImport(t *testing.T) {
	a, tt := snippetsTester(t)
	dir := t.TempDir()
	in := filepath.Join(dir, "wispr.json")
	os.WriteFile(in, []byte(`[{"name":"my email address","text":"ruiling@example.com"},
		{"name":"meeting link","text":"https://zoom.us/j/123"},
		{"name":"organize thoughts prompt","text":"Organize these thoughts."},
		{"name":"no text"}]`), 0o644)
	saved := chooseSnippetsFile
	t.Cleanup(func() { chooseSnippetsFile = saved })
	chooseSnippetsFile = func(a *app, then func(string)) { then(in) }

	click(t, tt, "Import…")
	if !has(tt, "Choose a .json file") || hasSubstring(tt, `"text": "zoom.us/j/123"`) {
		t.Fatalf("the dialog, its format folded: %q", tt.Texts())
	}
	click(t, tt, "File format")
	if !hasSubstring(tt, `"trigger": "my meeting link", "text": "zoom.us/j/123"`) {
		t.Fatalf("the format opened: %q", tt.Texts())
	}
	click(t, tt, "Choose a .json file")
	got := storedSnippets(t, a, tt, 3)
	if got[0].Trigger != "my email address" || got[1].Text != "https://zoom.us/j/123" {
		t.Errorf("stored %+v", got)
	}
	want := "Imported snippets: 3 new, 0 updated, 1 skipped."
	if !has(tt, want) || len(a.toasts) > 0 {
		t.Errorf("the notice: shown %v, toasts %q", has(tt, want), a.toasts)
	}
	if a.snipUI.importOpen {
		t.Error("the dialog stayed open after an import")
	}
	tt.Advance(snippetsImportNoticeFor + time.Second/2)
	frames(tt)
	if !has(tt, want) {
		t.Errorf("the notice of an import that skipped some went after %v", snippetsImportNoticeFor)
	}
	tt.Advance(snippetsImportNoticeSkippedFor - snippetsImportNoticeFor)
	frames(tt)
	if has(tt, want) || a.snipUI.notice != "" {
		t.Errorf("the notice stayed after %v", snippetsImportNoticeSkippedFor)
	}
	click(t, tt, "Import…")
	click(t, tt, "Choose a .json file")
	frames(tt)
	if want := "Imported snippets: 0 new, 0 updated, 1 skipped."; len(a.snip.list.Snippets) != 3 || !has(tt, want) {
		t.Errorf("a second import: %+v, toasts %q", a.snip.list.Snippets, a.toasts)
	}

	// Files that are not snippets, on a page with no toast before.
	a, tt = snippetsTester(t)
	for name, data := range map[string]string{
		"terms.json": `{"terms":[]}`,
		"words.json": `[{"phrase":"my email","replacement":"a@example.com"}]`,
	} {
		bad := filepath.Join(dir, name)
		os.WriteFile(bad, []byte(data), 0o644)
		chooseSnippetsFile = func(a *app, then func(string)) { then(bad) }
		click(t, tt, "Import…")
		click(t, tt, "Choose a .json file")
		want := map[string]string{
			"terms.json": "Could not import terms.json: it is not a JSON list of snippets.",
			"words.json": "Found no snippets Voxt can use in words.json.",
		}[name]
		for deadline := time.Now().Add(5 * time.Second); !has(tt, want); frames(tt) {
			if time.Now().After(deadline) {
				t.Fatalf("%s: texts %q", name, tt.Texts())
			}
		}
		if !a.snipUI.importOpen || !hasSubstring(tt, `"trigger": "my work email"`) || len(a.toasts) > 0 || len(a.snip.list.Snippets) != 0 {
			t.Errorf("%s: open %v, format shown %v, toasts %q, added %+v", name, a.snipUI.importOpen, a.snipUI.importExample, a.toasts, a.snip.list.Snippets)
		}
		click(t, tt, "Cancel")
	}
}

// TestSnippetsImportExample: the example under "File format", saved as
// a file in either language, imports through the dialog. The snippets
// already there stay; the one with the example's trigger takes its text.
func TestSnippetsImportExample(t *testing.T) {
	previous := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previous) })
	saved := chooseSnippetsFile
	t.Cleanup(func() { chooseSnippetsFile = saved })
	for _, locale := range []string{"en", "zh-Hans"} {
		a, tt := snippetsTester(t)
		i18n.SetLocale(locale) // after the app, which sets its own
		frames(tt)
		email := i18n.T("my work email")
		if (locale == "zh-Hans") != (email == "我的工作邮箱") {
			t.Fatalf("%s: the example's trigger is %q", locale, email)
		}
		a.snip.list = snippet.List{Snippets: []snippet.Snippet{
			{ID: "s1", Trigger: "sign off", Text: "Best, Alex"},
			{ID: "s2", Trigger: email, Text: "old@company.com"},
		}}
		a.saveSnippets()
		storedSnippets(t, a, tt, 2)
		in := filepath.Join(t.TempDir(), "example.json")
		os.WriteFile(in, []byte(snippetsImportExample()), 0o644)
		chooseSnippetsFile = func(a *app, then func(string)) { then(in) }

		click(t, tt, i18n.T("Import…"))
		click(t, tt, i18n.T("Choose a .json file"))
		got := storedSnippets(t, a, tt, 3)
		if got[0].Text != "Best, Alex" || got[1].Trigger != email || got[1].Text != "name@company.com" ||
			got[2].Trigger != i18n.T("my meeting link") || got[2].Text != "zoom.us/j/123" {
			t.Errorf("%s: stored %+v", locale, got)
		}
		want := i18n.T("Imported snippets: %d new, %d updated.", 1, 1)
		if !has(tt, want) {
			t.Errorf("%s: no notice %q; texts %q", locale, want, tt.Texts())
		}
		tt.Advance(snippetsImportNoticeFor + time.Second/2)
		frames(tt)
		if has(tt, want) {
			t.Errorf("%s: the notice stayed after %v", locale, snippetsImportNoticeFor)
		}
	}
}

// TestSnippetsSearchSortEdit: the newest first, then by trigger; the
// search finds a trigger or a text, and says when nothing matches; Edit
// shows under the pointer and opens the dialog.
func TestSnippetsSearchSortEdit(t *testing.T) {
	a, tt := snippetsTester(t)
	for _, s := range [][2]string{{"my email", "a@example.com"}, {"Zoom link", "https://zoom.us/j/1"}, {"bye", "See you"}} {
		if msg := a.saveSnippet("", s[0], s[1]); msg != "" {
			t.Fatal(msg)
		}
	}
	storedSnippets(t, a, tt, 3)
	if got := shownTriggers(a); !slices.Equal(got, []string{"bye", "Zoom link", "my email"}) {
		t.Errorf("the newest first: %q", got)
	}
	click(t, tt, "Sort")
	if err := tt.ChooseMenuItem("By trigger (A–Z)"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got := shownTriggers(a); !slices.Equal(got, []string{"bye", "my email", "Zoom link"}) {
		t.Errorf("by trigger: %q", got)
	}

	click(t, tt, "Search snippets")
	tt.Type("ZOOM.US")
	frames(tt)
	if got := shownTriggers(a); !slices.Equal(got, []string{"Zoom link"}) || has(tt, "a@example.com") {
		t.Errorf("searched by text: %q", got)
	}
	tt.Type("/nothing")
	frames(tt)
	if !has(tt, "No snippet matches “ZOOM.US/nothing”") {
		t.Fatalf("no match: %q", firstTexts(tt))
	}
	click(t, tt, "Clear search")
	if len(shownTriggers(a)) != 3 {
		t.Errorf("after clearing: %q", shownTriggers(a))
	}

	r, ok := tt.Find("a@example.com")
	if !ok {
		t.Fatalf("no row: %q", firstTexts(tt))
	}
	tt.Move(r.X+r.W/2, r.Y+r.H/2)
	frames(tt)
	clickControl(t, tt, "Edit my email")
	if !a.snipUI.editOpen || a.snipUI.edit.trigger != "my email" || a.snipUI.edit.id == "" {
		t.Errorf("Edit opened %+v", a.snipUI)
	}
}

// hasSubstring reports whether a text the view shows contains s.
func hasSubstring(tt *ui.Tester, s string) bool {
	frames(tt)
	for _, text := range tt.Texts() {
		if strings.Contains(text, s) {
			return true
		}
	}
	return false
}
