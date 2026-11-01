package main

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// More entries for the menu's cases (seedMenuHistory).
const (
	histSameRaw   = "01HIST0000000000000000SAM1" // done, the result is the recognised text
	histFailedRaw = "01HIST0000000000000000FRW1" // failed after recognition: no result
)

func seedMenuHistory(t *testing.T, a *app) {
	t.Helper()
	now := time.Now()
	for _, e := range []store.Entry{
		{
			ID: histSameRaw, Kind: store.KindTranscribe, CreatedAt: now.Add(-10 * time.Minute), Status: store.StatusDone,
			AppName: "Notes", RawText: "Buy milk and eggs.", Result: "Buy milk and eggs.", Delivery: "insert",
		},
		{
			ID: histFailedRaw, Kind: store.KindTranscribe, CreatedAt: now.Add(-20 * time.Minute), Status: store.StatusFailed,
			AppName: "Notes", RawText: "call the dentist on monday", Error: "llm.balanced: the language model stopped",
		},
	} {
		if err := a.lib.store.History().Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
}

// openMenu right-clicks the row showing text and returns its menu.
func openMenu(t *testing.T, tt *ui.Tester, text string) []string {
	t.Helper()
	frames(tt)
	tt.CloseMenu()
	if err := tt.RightClick(text); err != nil {
		t.Fatalf("right-click %q: %v (texts: %q)", text, err, firstTexts(tt))
	}
	frames(tt)
	m := tt.Menu()
	if m == nil {
		t.Fatalf("no menu on %q", text)
	}
	return m
}

// choose chooses a menu item and runs the frames after it.
func choose(t *testing.T, tt *ui.Tester, label string) {
	t.Helper()
	if err := tt.ChooseMenuItem(label); err != nil {
		t.Fatalf("choose %q: %v (menu %q)", label, err, tt.Menu())
	}
	frames(tt)
}

func TestEntryMenuItems(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	seedMenuHistory(t, a)
	a.navigate("/history")
	for _, c := range []struct {
		row  string
		want []string
	}{
		// Done, the recognised text differs from the result, a recording.
		{"Let's move the design review to Thursday afternoon.",
			[]string{"Copy", "Copy the recognised text", "Show details", "Show the recording in Finder", "Transcribe again", "-", "Delete"}},
		// Done, the result is the recognised text: one Copy.
		{"Buy milk and eggs.", []string{"Copy", "Show details", "-", "Delete"}},
		// Failed after recognition: Copy copies the recognised text.
		{"llm.balanced: the language model stopped", []string{"Copy", "Show details", "-", "Delete"}},
		// Failed with no text: Copy is there, disabled.
		{"asr.accurate: the speech server stopped answering (15s)", []string{"Copy", "Show details", "-", "Delete"}},
	} {
		if got := openMenu(t, tt, c.row); !slices.Equal(got, c.want) {
			t.Errorf("%s: menu %q, want %q", c.row, got, c.want)
		}
	}
	if err := tt.ChooseMenuItem("Copy"); err == nil {
		t.Error("Copy is enabled on an entry without text")
	}

	// A failed entry's Copy copies what was recognised.
	openMenu(t, tt, "llm.balanced: the language model stopped")
	choose(t, tt, "Copy")
	if got := tt.Clipboard(); got != "call the dentist on monday" {
		t.Errorf("clipboard = %q", got)
	}
}

func TestEntryMenuActions(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	var revealed []string
	was := revealInFinder
	revealInFinder = func(path string) { revealed = append(revealed, path) }
	t.Cleanup(func() { revealInFinder = was })
	a.navigate("/history")
	row := "Let's move the design review to Thursday afternoon."

	openMenu(t, tt, row)
	choose(t, tt, "Copy")
	if got := tt.Clipboard(); got != "Let's move the design review to Thursday afternoon.\nI'll send the new agenda tonight." {
		t.Errorf("clipboard = %q", got)
	}
	if !has(tt, "Copied") {
		t.Error("no Copied toast")
	}
	openMenu(t, tt, row)
	choose(t, tt, "Copy the recognised text")
	if got := tt.Clipboard(); got != "um so let's move the design review to thursday afternoon and I'll send the new agenda tonight" {
		t.Errorf("clipboard = %q", got)
	}
	if a.hist.sel != "" {
		t.Errorf("Copy selected %q", a.hist.sel)
	}
	openMenu(t, tt, row)
	choose(t, tt, "Show the recording in Finder")
	if want := filepath.Join(a.lib.store.History().AudioDir(), histInserted+".wav"); len(revealed) != 1 || revealed[0] != want {
		t.Errorf("revealed %v, want %s", revealed, want)
	}

	// Details opens the panel, and keeps it open when chosen again.
	for range 2 {
		openMenu(t, tt, row)
		choose(t, tt, "Show details")
		if a.hist.sel != histInserted || !has(tt, "Entry details") {
			t.Fatalf("Show details: selected %q", a.hist.sel)
		}
	}

	// Delete asks first; Cancel keeps it.
	openMenu(t, tt, "明天上午十点的站会改到线上，链接我稍后发到群里。")
	choose(t, tt, "Delete")
	if !a.hist.delAsk || a.hist.delID != histCard || !has(tt, "Delete this entry?") {
		t.Fatalf("no confirmation: %q", firstTexts(tt))
	}
	click(t, tt, "Cancel")
	if _, err := a.lib.store.History().Get(context.Background(), histCard); err != nil {
		t.Fatalf("Cancel deleted it: %v", err)
	}
	// Delete of another entry than the one in the panel: the panel stays.
	openMenu(t, tt, "明天上午十点的站会改到线上，链接我稍后发到群里。")
	choose(t, tt, "Delete")
	if err := tt.Click("Delete"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if _, err := a.lib.store.History().Get(context.Background(), histCard); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("still there: %v", err)
	}
	if a.hist.sel != histInserted || len(a.hist.list) != 3 || has(tt, "明天上午十点的站会改到线上，链接我稍后发到群里。") {
		t.Errorf("after Delete: selected %q, %v", a.hist.sel, historyIDs(a))
	}
	// The other observers heard of it: Home reads again when it shows,
	// the menu bar's recent results now.
	if !a.hm.stale {
		t.Error("Home not marked stale")
	}
	if slices.Contains(a.recent, "明天上午十点的站会改到线上，链接我稍后发到群里。") {
		t.Errorf("the menu bar still lists it: %q", a.recent)
	}
}

func TestHomeRecentMenu(t *testing.T) {
	a, tt := homeTester(t)
	seedHistory(t, a)
	a.historyChanged()
	frames(tt)
	e := a.hm.recent[0]
	title, _ := uitext.EntryTitle(e)
	want := []string{"Copy", "Copy the recognised text", "Show details", "Show the recording in Finder", "Transcribe again", "-", "Delete"}
	if got := openMenu(t, tt, title); !slices.Equal(got, want) {
		t.Errorf("menu %q, want %q", got, want)
	}
	choose(t, tt, "Copy")
	if got := tt.Clipboard(); got != e.Result {
		t.Errorf("clipboard = %q", got)
	}

	// Delete from Home: its dialog shows there, and the list reads again.
	openMenu(t, tt, title)
	choose(t, tt, "Delete")
	if a.router.Path() != "/home" || !has(tt, "Delete this entry?") {
		t.Fatalf("no confirmation on Home: %s %q", a.router.Path(), firstTexts(tt))
	}
	if err := tt.Click("Delete"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if _, err := a.lib.store.History().Get(context.Background(), e.ID); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("still there: %v", err)
	}
	if has(tt, title) || len(a.hm.recent) == 0 || a.hm.recent[0].ID == e.ID {
		t.Errorf("Home still lists it: %q", firstTexts(tt))
	}

	// Show details opens it in History, as a click does.
	next := a.hm.recent[0]
	nextTitle, _ := uitext.EntryTitle(next)
	openMenu(t, tt, nextTitle)
	choose(t, tt, "Show details")
	if a.router.Path() != "/history" || a.hist.sel != next.ID {
		t.Errorf("opened %s sel %q", a.router.Path(), a.hist.sel)
	}
}

func TestHistoryRowKeys(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	seedMenuHistory(t, a)
	a.navigate("/history")
	// A click focuses the row (and shows its details).
	click(t, tt, "Let's move the design review to Thursday afternoon.")
	tt.Key(ui.Cmd, ui.KeyC)
	frames(tt)
	if got := tt.Clipboard(); got != "Let's move the design review to Thursday afternoon.\nI'll send the new agenda tonight." {
		t.Errorf("⌘C copied %q", got)
	}
	// macOS: the Edit menu's Copy takes ⌘C and sends the command.
	tt.SetClipboard("before")
	tt.Command("copy")
	frames(tt)
	if got := tt.Clipboard(); got != "Let's move the design review to Thursday afternoon.\nI'll send the new agenda tonight." {
		t.Errorf("the Copy command copied %q", got)
	}
	// The row still takes clicks: a second one hides the details.
	click(t, tt, "Let's move the design review to Thursday afternoon.")
	if a.hist.sel != "" {
		t.Errorf("still selected: %q", a.hist.sel)
	}
	tt.Key(0, ui.KeyBackspace)
	frames(tt)
	if !a.hist.delAsk || a.hist.delID != histInserted {
		t.Errorf("⌫ asked %v for %q", a.hist.delAsk, a.hist.delID)
	}
	click(t, tt, "Cancel")
	// No text at all: ⌘C copies nothing.
	tt.SetClipboard("before")
	click(t, tt, "asr.accurate: the speech server stopped answering (15s)")
	tt.Key(ui.Cmd, ui.KeyC)
	tt.Command("copy")
	frames(tt)
	if tt.Clipboard() != "before" {
		t.Errorf("⌘C on an entry without text copied %q", tt.Clipboard())
	}
}
