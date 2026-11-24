package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
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

// TestSnippetsImport: Import reads a Wispr Flow export, skips what it
// cannot use and says so; importing it again changes nothing, a file
// that is not snippets says why.
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
	got := storedSnippets(t, a, tt, 3)
	if got[0].Trigger != "my email address" || got[1].Text != "https://zoom.us/j/123" {
		t.Errorf("stored %+v", got)
	}
	if want := "Imported snippets: 3 new, 0 updated, 1 skipped."; !slices.Contains(a.toasts, want) && !has(tt, want) {
		t.Errorf("toasts %q", a.toasts)
	}
	a.toasts = nil
	click(t, tt, "Import…")
	frames(tt)
	if len(a.snip.list.Snippets) != 3 {
		t.Errorf("a second import: %+v", a.snip.list.Snippets)
	}

	// A file that is not snippets, on a page with no toast before.
	a, tt = snippetsTester(t)
	bad := filepath.Join(dir, "terms.json")
	os.WriteFile(bad, []byte(`{"terms":[]}`), 0o644)
	chooseSnippetsFile = func(a *app, then func(string)) { then(bad) }
	click(t, tt, "Import…")
	want := "Could not import terms.json: it is not a JSON list of snippets."
	for deadline := time.Now().Add(5 * time.Second); !slices.Contains(a.toasts, want) && !has(tt, want); frames(tt) {
		if time.Now().After(deadline) {
			t.Fatalf("toasts %q; texts %q", a.toasts, tt.Texts())
		}
	}
	if len(a.snip.list.Snippets) != 0 {
		t.Errorf("a bad file added %+v", a.snip.list.Snippets)
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
