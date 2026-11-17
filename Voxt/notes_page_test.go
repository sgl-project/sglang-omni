package main

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// TestNotesPage: the sidebar opens it; empty, it says how to record a
// note; it lists the notes newest first and deletes one through its
// button; Settings → Shortcuts' row opens it; the shortcut changes here.
func TestNotesPage(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	frames(tt)
	click(t, tt, "Notes") // the sidebar's
	if a.shownPath() != "/notes" {
		t.Fatalf("went to %s", a.shownPath())
	}
	if !has(tt, "No notes yet") || !has(tt, "Press Right ⌘ in any app, speak, and press it again.") {
		t.Errorf("empty: %q", firstTexts(tt))
	}
	repo := a.lib.store.Notes()
	t0 := time.Now().Add(-2 * time.Hour)
	for i, text := range []string{"buy milk", "call Ana"} {
		note := store.Note{ID: "n" + itoa(i), Text: text, CreatedAt: t0.Add(time.Duration(i) * time.Minute)}
		if i == 1 {
			note.AppName = "Slack"
		}
		if err := repo.Put(context.Background(), note); err != nil {
			t.Fatal(err)
		}
	}
	a.loadNotes()
	frames(tt)
	texts := tt.Texts()
	if i, j := slices.Index(texts, "call Ana"), slices.Index(texts, "buy milk"); i < 0 || j < 0 || i > j || !hasExact(tt, "2 notes") || !hasExact(tt, "2 hours ago") || !hasExact(tt, "1 hour ago · Slack") {
		t.Errorf("list: %q", texts)
	}
	for _, l := range []string{"Delete call Ana", "Delete buy milk", "Keys to record a note"} {
		if !slices.Contains(tt.AccessLabels(), l) {
			t.Errorf("no control %q: %q", l, tt.AccessLabels())
		}
	}
	if err := tt.ClickControl("Delete call Ana"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if list, _ := repo.List(context.Background()); len(list) != 1 || list[0].Text != "buy milk" || hasExact(tt, "call Ana") {
		t.Errorf("after delete: %+v, %q", list, tt.Texts())
	}

	a.navigate("/settings/shortcuts")
	click(t, tt, "Notes "+uitext.HotkeyText(settings.DefaultNote))
	if a.settingsOpen || a.shownPath() != "/notes" {
		t.Fatalf("Notes row: %v %s", a.settingsOpen, a.shownPath())
	}
	click(t, tt, "Keys to record a note")
	click(t, tt, "Off")
	if k := a.lib.settings.Get().Features.Note.Keys; k != "" {
		t.Fatalf("note keys %q", k)
	}
	click(t, tt, "Keys to record a note")
	click(t, tt, uitext.HotkeyText("RightShift"))
	if k := a.lib.settings.Get().Features.Note; k.Keys != "RightShift" || k.Trigger != settings.TriggerTap {
		t.Fatalf("note keys %+v", k)
	}
}

// TestNoteCopyEditAndCancel: Copy takes the whole text; Cancel keeps the
// stored note; Save keeps its time and source app and survives a restart.
func TestNoteCopyEditAndCancel(t *testing.T) {
	dir := testDir(t)
	a, _, tt := newTestAppIn(t, dir)
	tt.SetSize(windowWidth, 1200)
	original := "buy milk, coffee, and tea after the design review tomorrow morning"
	edited := "buy milk after the review"
	at := time.Now().Add(-time.Hour).Truncate(time.Millisecond)
	repo := a.lib.store.Notes()
	if err := repo.Put(context.Background(), store.Note{ID: "n1", Text: original, CreatedAt: at, AppName: "Slack"}); err != nil {
		t.Fatal(err)
	}
	a.navigate("/notes")
	frames(tt)
	click(t, tt, "Copy: "+noteExcerpt(original))
	if got := tt.Clipboard(); got != original {
		t.Fatalf("clipboard %q", got)
	}

	click(t, tt, "Edit "+noteExcerpt(original))
	tt.Key(ui.Cmd, ui.KeyA)
	tt.Type("discard this")
	click(t, tt, "Cancel")
	list, err := repo.List(context.Background())
	if err != nil || len(list) != 1 || list[0].Text != original || !has(tt, original) {
		t.Fatalf("cancel: %+v, %v, %q", list, err, tt.Texts())
	}

	click(t, tt, "Edit "+noteExcerpt(original))
	tt.Key(ui.Cmd, ui.KeyA)
	tt.Type(edited)
	click(t, tt, "Save")
	list, err = repo.List(context.Background())
	if err != nil || len(list) != 1 || list[0].Text != edited || !list[0].CreatedAt.Equal(at) || list[0].AppName != "Slack" {
		t.Fatalf("save: %+v, %v", list, err)
	}

	a.quit()
	a.lib.close()
	b, _, bt := newTestAppIn(t, dir)
	bt.SetSize(windowWidth, 1200)
	b.navigate("/notes")
	if !has(bt, edited) || has(bt, original) || !has(bt, "Slack") {
		t.Fatalf("after restart: %q", bt.Texts())
	}
}

// TestNoteNotices: a saved note says so in the overlay; one that could
// not be saved says the answer card shows it.
func TestNoteNotices(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.sessionEnded(session.Snapshot{ID: "s1", Kind: store.KindTranscribe, State: session.Done, Note: true, Result: "buy milk"})
	if a.notice == nil || a.notice.text != "Saved to Notes" || a.notice.warn {
		t.Errorf("saved: %+v", a.notice)
	}
	a.sessionEnded(session.Snapshot{ID: "s2", Kind: store.KindTranscribe, State: session.Done, Note: true, Result: "buy milk", Card: true,
		Problem: &session.Problem{Code: session.ProblemInternal, Err: errors.New("disk full"), Degraded: true}})
	if a.notice == nil || a.notice.text != "Could not save the note, so the card shows it." || !a.answer.open {
		t.Errorf("not saved: %+v, card %v", a.notice, a.answer.open)
	}
}

// TestNoteNoticesLanguageModel: a note whose language model failed is
// saved as recognised, and the overlay says it went to Notes, not where
// the cursor is.
func TestNoteNoticesLanguageModel(t *testing.T) {
	a, _, _ := newTestApp(t)
	for _, tc := range []struct {
		err  error
		want string
	}{
		{errors.New("bad gateway"), "The language model failed, so the recognised text was saved to Notes."},
		{context.DeadlineExceeded, "The language model did not answer in time, so the recognised text was saved to Notes."},
	} {
		a.sessionEnded(session.Snapshot{ID: "s3", Kind: store.KindTranscribe, State: session.Done, Note: true, Result: "buy milk",
			Problem: &session.Problem{Code: session.ProblemLLM, Err: tc.err, Degraded: true}})
		if a.notice == nil || a.notice.text != tc.want || !a.notice.warn {
			t.Errorf("%v: %+v", tc.err, a.notice)
		}
	}
}

// TestNotesPageAsksForAShortcut: with the note's shortcut off, or taken by
// another shortcut, the page asks for one and does not offer its keys.
func TestNotesPageAsksForAShortcut(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1200)
	a.navigate("/notes")
	a.updateSettings(func(s *settings.Settings) { s.Features.Note = settings.Hotkey{} })
	frames(tt)
	if !has(tt, "Set a shortcut to record notes.") {
		t.Errorf("off: %q", firstTexts(tt))
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Note = settings.Hotkey{Keys: settings.DefaultNote, Trigger: settings.TriggerTap}
		s.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: settings.DefaultNote, Trigger: settings.TriggerTap}}
	})
	frames(tt)
	if has(tt, "Press Right ⌘ in any app, speak, and press it again.") || !has(tt, "No notes yet") {
		t.Errorf("taken: %q", firstTexts(tt))
	}
}

// notesShots renders a note and its in-place editor (TestSnapshots).
func notesShots(t *testing.T, a *app, tt *ui.Tester, save func(string)) {
	note := store.Note{ID: "note-shot", Text: "Buy milk, coffee, and tea after the design review.", CreatedAt: time.Now().Add(-5 * time.Minute), AppName: "Slack"}
	if err := a.lib.store.Notes().Put(context.Background(), note); err != nil {
		t.Fatal(err)
	}
	a.navigate("/notes")
	a.loadNotes()
	save("page-notes-list")
	click(t, tt, i18n.T("Edit %s", noteExcerpt(note.Text)))
	save("page-notes-edit")
	click(t, tt, i18n.T("Cancel"))
}
