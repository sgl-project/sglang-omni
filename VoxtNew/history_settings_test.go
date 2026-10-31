package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// seedAged puts transcriptions days old (one a day: "aged-<days>"), each
// with a recording.
func seedAged(t *testing.T, a *app, days ...int) {
	t.Helper()
	h := a.lib.store.History()
	for _, d := range days {
		id := fmt.Sprintf("aged-%d", d)
		e := store.Entry{ID: id, Kind: store.KindTranscribe, CreatedAt: time.Now().AddDate(0, 0, -d), Result: "said " + id, Audio: id + ".wav"}
		if err := os.WriteFile(filepath.Join(h.AudioDir(), e.Audio), []byte("RIFF "+id), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := h.Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
}

// entryIDs are every entry's ids, sorted.
func entryIDs(t *testing.T, a *app) []string {
	t.Helper()
	list, err := a.lib.store.History().List(context.Background(), store.HistoryQuery{})
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, e := range list {
		ids = append(ids, e.ID)
	}
	slices.Sort(ids)
	return ids
}

// openHistSettings shows History and opens the dialog with its button.
func openHistSettings(t *testing.T, a *app, tt *ui.Tester) {
	t.Helper()
	tt.SetSize(windowWidth, 1000)
	a.navigate("/history")
	click(t, tt, "History settings")
	if !a.histSet.open || !hasExact(tt, "Storage location") {
		t.Fatalf("the dialog did not open: %q", firstTexts(tt))
	}
}

// TestHistorySettingsDialog: the button opens it; Keep the recordings is
// Privacy's setting; the recordings' number and size show; Done closes.
func TestHistorySettingsDialog(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedAged(t, a, 1, 2)
	openHistSettings(t, a, tt)
	for _, s := range []string{"History settings", "Recordings", "Keep the recordings", "Saved recordings", "2 recordings · 22 B",
		"Export recordings", "Auto-cleanup", "Clean up automatically", "Keep entries for", "3 months", "Keep at most", "No limit"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, firstTexts(tt))
		}
	}
	click(t, tt, "Save each recording")
	if !a.prefs.Privacy.KeepAudio || !a.settings.Get().Privacy.KeepAudio {
		t.Error("Keep the recordings did not turn on")
	}
	click(t, tt, "Done")
	if a.histSet.open {
		t.Error("Done did not close it")
	}
	// Settings → Privacy shows the same value, and opens the dialog.
	a.navigate("/settings/privacy")
	frames(tt)
	if !a.settingsForm.keepAudio {
		t.Error("Privacy shows it off")
	}
	click(t, tt, "Open…")
	if !a.histSet.open || a.currentPage() != "history" || a.settingsOpen {
		t.Errorf("Privacy's link: open %v, page %s, settings %v", a.histSet.open, a.currentPage(), a.settingsOpen)
	}
}

// TestHistoryCleanupAsks: a shorter period that deletes entries asks
// with their number; Cancel keeps the setting, Delete saves it and
// deletes them with their recordings; one that deletes none saves at
// once; off saves at once, and on again asks.
func TestHistoryCleanupAsks(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedAged(t, a, 1, 10, 40, 60)
	openHistSettings(t, a, tt)
	click(t, tt, "3 months")
	click(t, tt, "1 week")
	if !a.histSet.pruneAsk || !has(tt, "Delete 3 entries now?") {
		t.Fatalf("no question: %q", firstTexts(tt))
	}
	click(t, tt, "Cancel")
	if a.prefs.History.KeepDays != 0 || len(entryIDs(t, a)) != 4 || !hasExact(tt, "3 months") {
		t.Errorf("cancelled: %+v %v", a.prefs.History, entryIDs(t, a))
	}
	click(t, tt, "3 months")
	click(t, tt, "30 days")
	if !has(tt, "Delete 2 entries now?") {
		t.Fatalf("no question: %q", firstTexts(tt))
	}
	click(t, tt, "Delete")
	if a.settings.Get().History.KeepDays != 30 {
		t.Errorf("saved %+v", a.settings.Get().History)
	}
	if got := entryIDs(t, a); !slices.Equal(got, []string{"aged-1", "aged-10"}) {
		t.Errorf("left %v", got)
	}
	if _, err := os.Stat(filepath.Join(a.lib.store.History().AudioDir(), "aged-40.wav")); !os.IsNotExist(err) {
		t.Errorf("the recording stayed: %v", err)
	}
	if len(a.hist.list) != 2 || !hasExact(tt, "30 days") {
		t.Errorf("the page: %d entries, %q", len(a.hist.list), firstTexts(tt))
	}
	// 100 a feature deletes none: saved without a question.
	click(t, tt, "No limit")
	click(t, tt, "100")
	if a.histSet.pruneAsk || a.settings.Get().History.KeepCount != 100 {
		t.Errorf("100: asked %v, %+v", a.histSet.pruneAsk, a.settings.Get().History)
	}
	click(t, tt, "Clean up the history")
	if !a.settings.Get().History.CleanupOff || has(tt, "Keep entries for") || !has(tt, "Entries stay until you delete them.") {
		t.Errorf("off: %+v %q", a.settings.Get().History, firstTexts(tt))
	}
	seedAged(t, a, 50)
	a.historyChanged() // off: nothing goes
	if len(entryIDs(t, a)) != 3 {
		t.Fatalf("cleaned while off: %v", entryIDs(t, a))
	}
	click(t, tt, "Clean up the history")
	if !has(tt, "Delete 1 entry now?") {
		t.Fatalf("on again: %q", firstTexts(tt))
	}
	tt.KeyDown(ui.KeyEscape)
	frames(tt)
	if a.histSet.pruneAsk || !a.settings.Get().History.CleanupOff || a.histSet.cleanOn {
		t.Errorf("escape: ask %v, %+v, switch %v", a.histSet.pruneAsk, a.settings.Get().History, a.histSet.cleanOn)
	}
}

// TestHistoryCleanupRuns: at the start, and after a new entry.
func TestHistoryCleanupRuns(t *testing.T) {
	dir := testDir(t)
	a, _, _ := newTestAppIn(t, dir)
	seedAged(t, a, 1, 100, 200)
	a.updateSettings(func(s *settings.Settings) { s.History.KeepDays = 365 })
	if got := entryIDs(t, a); len(got) != 3 {
		t.Fatalf("365 days deleted: %v", got)
	}
	a.updateSettings(func(s *settings.Settings) { s.History.KeepDays = 0 }) // 90 days
	if got := entryIDs(t, a); !slices.Equal(got, []string{"aged-1"}) {
		t.Fatalf("the setting's change: %v", got)
	}
	// A new old entry, then a session's: the clean-up follows.
	seedAged(t, a, 120)
	a.historyChanged()
	if got := entryIDs(t, a); len(got) != 1 {
		t.Errorf("after a new entry: %v", got)
	}
	// At the start.
	seedAged(t, a, 95)
	a.quit()
	a.lib.close()
	b, _, _ := newTestAppIn(t, dir)
	if got := entryIDs(t, b); !slices.Equal(got, []string{"aged-1"}) {
		t.Errorf("at the start: %v", got)
	}
}

// TestHistoryRecordingsFolder: Choose… moves the recordings and saves
// the folder; new recordings go there; Use the default moves them back;
// a folder missing at the start falls back to the default and says so.
func TestHistoryRecordingsFolder(t *testing.T) {
	dir := testDir(t)
	a, _, tt := newTestAppIn(t, dir)
	seedAged(t, a, 1, 2)
	h := a.lib.store.History()
	def := h.DefaultAudioDir()
	ext := filepath.Join(t.TempDir(), "Recordings")
	chosen := ext
	old := chooseFolder
	chooseFolder = func(_ *app, _, _ string, then func(string)) { then(chosen) }
	t.Cleanup(func() { chooseFolder = old })
	openHistSettings(t, a, tt)
	click(t, tt, "Choose…")
	if a.settings.Get().History.AudioDir != ext || h.AudioDir() != ext {
		t.Fatalf("saved %q, in use %q", a.settings.Get().History.AudioDir, h.AudioDir())
	}
	if n, _ := os.ReadDir(ext); len(n) != 2 {
		t.Errorf("moved %d", len(n))
	}
	if n, _ := os.ReadDir(def); len(n) != 0 {
		t.Errorf("left %d", len(n))
	}
	if !has(tt, "Moved 2 recordings to "+ext) && !slices.ContainsFunc(a.toasts, func(s string) bool { return s == "Moved 2 recordings to "+ext }) {
		t.Errorf("toast %q", a.toasts)
	}
	if !has(tt, "Use the default") {
		t.Fatalf("no Use the default: %q", firstTexts(tt))
	}
	// The drive goes away: the next start records into the default.
	a.quit()
	a.lib.close()
	os.Rename(ext, ext+"-away")
	b, _, tt := newTestAppIn(t, dir)
	tt.SetSize(windowWidth, 1000)
	if b.lib.store.History().AudioDir() != def || b.histSet.audioMissing != ext {
		t.Fatalf("missing: in use %s, missing %q", b.lib.store.History().AudioDir(), b.histSet.audioMissing)
	}
	b.navigate("/history")
	frames(tt)
	if !hasPrefix(tt, ext+" is not available") {
		t.Errorf("no notice: %q", firstTexts(tt))
	}
	// It comes back: opening the dialog finds it.
	os.Rename(ext+"-away", ext)
	click(t, tt, "History settings")
	if b.lib.store.History().AudioDir() != ext || b.histSet.audioMissing != "" {
		t.Errorf("back: %s %q", b.lib.store.History().AudioDir(), b.histSet.audioMissing)
	}
	click(t, tt, "Use the default")
	if b.settings.Get().History.AudioDir != "" || b.lib.store.History().AudioDir() != def {
		t.Errorf("default: %q %s", b.settings.Get().History.AudioDir, b.lib.store.History().AudioDir())
	}
	if n, _ := os.ReadDir(def); len(n) != 2 {
		t.Errorf("back in the default: %d", len(n))
	}
}

// TestHistoryExportAndDelete: Export… copies the recordings (all, or
// the tab's); Delete all… asks, then deletes them and keeps the entries.
func TestHistoryExportAndDelete(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedAged(t, a, 1, 2)
	out := t.TempDir()
	old := chooseFolder
	chooseFolder = func(_ *app, _, _ string, then func(string)) { then(out) }
	t.Cleanup(func() { chooseFolder = old })
	openHistSettings(t, a, tt)
	click(t, tt, "Export…")
	got, _ := os.ReadDir(filepath.Join(out, "transcription"))
	if len(got) != 2 {
		t.Fatalf("exported %v", got)
	}
	if !slices.Contains(a.toasts, "Exported 2 recordings") && !has(tt, "Exported 2 recordings") {
		t.Errorf("toast %q", a.toasts)
	}
	// The Translate tab's: none.
	a.hist.tab = 1
	frames(tt)
	click(t, tt, "All recordings")
	click(t, tt, "Translate only")
	out = t.TempDir()
	click(t, tt, "Export…")
	if got, _ := os.ReadDir(out); len(got) != 0 {
		t.Errorf("the tab's: %v", got)
	}
	click(t, tt, "Delete all…")
	if !has(tt, "Delete every recording?") {
		t.Fatalf("no question: %q", firstTexts(tt))
	}
	click(t, tt, "Delete the recordings")
	if n, _ := os.ReadDir(a.lib.store.History().AudioDir()); len(n) != 0 {
		t.Errorf("recordings left: %d", len(n))
	}
	if got := entryIDs(t, a); len(got) != 2 {
		t.Errorf("entries %v", got)
	}
	if !has(tt, "0 recordings · 0 B") {
		t.Errorf("usage: %q", firstTexts(tt))
	}
}

// historySettingsShots are the History settings dialog's snapshots: as
// it opens, and its question before a clean-up deletes entries.
func historySettingsShots(a *app, save func(string)) {
	a.navigate("/history")
	// The sample's recording (seedHistory), for the size.
	wav := filepath.Join(a.lib.store.History().AudioDir(), histInserted+".wav")
	os.WriteFile(wav, make([]byte, 236_844), 0o644)
	defer os.Remove(wav) // Settings → Storage's snapshots measure none
	a.openHistorySettingsDialog()
	save("history-settings")
	hs := &a.histSet
	hs.formFor, hs.daysSel = "asking", keepDaysLabels()[4]
	hs.pruneAsk, hs.pruneN = true, 12
	save("history-settings-ask")
	hs.pruneAsk, hs.open, hs.formFor = false, false, ""
}
