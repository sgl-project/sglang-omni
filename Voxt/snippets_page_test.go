package main

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// storedSnippets waits for the snippets document to hold n snippets.
func storedSnippets(t *testing.T, a *app, tt *ui.Tester, n int) []snippet.Snippet {
	t.Helper()
	var l snippet.List
	var err error
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		frames(tt) // the save's goroutine posts back to the frames
		if l, err = readSnippets(context.Background(), a.lib.store.Documents()); err == nil && len(l.Snippets) == n {
			return l.Snippets
		}
	}
	t.Fatalf("stored %+v (%v), want %d", l.Snippets, err, n)
	return nil
}

// TestSnippetsPage: the sidebar opens the page; an empty page's example
// opens the dialog with its trigger; Save stores the snippet and lists
// it; a taken trigger keeps the dialog open with why; the dialog deletes.
func TestSnippetsPage(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	frames(tt)
	click(t, tt, "Snippets") // the sidebar's
	for deadline := time.Now().Add(5 * time.Second); !a.snip.loaded; frames(tt) {
		if time.Now().After(deadline) {
			t.Fatal("the snippets were not read")
		}
	}
	if a.shownPath() != "/snippets" || !has(tt, "No snippets yet") {
		t.Fatalf("at %s: %q", a.shownPath(), firstTexts(tt))
	}
	clickControl(t, tt, "my work email")
	if !a.snipUI.editOpen || a.snipUI.edit.trigger != "my work email" || a.snipUI.edit.id != "" {
		t.Fatalf("dialog %+v", a.snipUI)
	}
	click(t, tt, "Text of the snippet")
	tt.Type("name@example.com")
	click(t, tt, "Save")
	got := storedSnippets(t, a, tt, 1)
	if got[0].Trigger != "my work email" || got[0].Text != "name@example.com" || a.snipUI.editOpen {
		t.Fatalf("stored %+v, open %v", got, a.snipUI.editOpen)
	}
	if !hasExact(tt, "my work email") || !hasExact(tt, "name@example.com") {
		t.Errorf("list: %q", firstTexts(tt))
	}

	click(t, tt, "Add snippet")
	click(t, tt, "Trigger to say")
	tt.Type("My Work Email!")
	click(t, tt, "Text of the snippet")
	tt.Type("other@example.com")
	click(t, tt, "Save")
	if !a.snipUI.editOpen || !has(tt, "Another snippet has this trigger.") {
		t.Fatalf("a taken trigger: open %v, %q", a.snipUI.editOpen, firstTexts(tt))
	}
	click(t, tt, "Cancel")

	clickControl(t, tt, "my work email → name@example.com")
	click(t, tt, "Delete")
	if got := storedSnippets(t, a, tt, 0); len(got) != 0 || !has(tt, "No snippets yet") {
		t.Fatalf("after delete: %+v, %q", got, firstTexts(tt))
	}
}

// TestSnippetsUnreadable: a snippets document that cannot be read is
// left as it is: the page says so and offers no Add or Import, and a
// save or a delete changes nothing.
func TestSnippetsUnreadable(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	frames(tt)
	docs := a.lib.store.Documents()
	bad := []byte(`{"snippets":[{"id":"a","trigger":"my email","text":1}]}`)
	if err := docs.Put(context.Background(), snippet.Key, bad); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Snippets")
	for deadline := time.Now().Add(5 * time.Second); a.snip.err == ""; frames(tt) {
		if time.Now().After(deadline) {
			t.Fatal("no error")
		}
	}
	frames(tt)
	if !has(tt, "Could not read the snippets") || has(tt, "Add snippet") || has(tt, "Import…") || has(tt, "No snippets yet") {
		t.Errorf("page: %q", firstTexts(tt))
	}
	if msg := a.saveSnippet("", "my link", "https://example.com"); msg == "" {
		t.Error("saved over an unread list")
	}
	a.deleteSnippet("a")
	a.importSnippetsFile(filepath.Join(t.TempDir(), "none.json"))
	for range 20 {
		frames(tt)
	}
	if data, _, _ := docs.Get(context.Background(), snippet.Key); string(data) != string(bad) {
		t.Errorf("the document became %s", data)
	}
}
