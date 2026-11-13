package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const accessTitle = "Let programs on this Mac read the history"

func openPrivacy(t *testing.T, a *app, tt *ui.Tester) {
	t.Helper()
	tt.SetSize(windowWidth, 1800)
	a.navigate("/settings/privacy")
	frames(tt)
	// The settings dialog keeps its height: bring the section into it.
	if r, ok := tt.Find("Other programs"); ok {
		tt.Scroll(float32(windowWidth)*0.6, 900, 0, r.Y-700)
		frames(tt)
	}
}

func socketGone(a *app) bool {
	_, err := os.Stat(a.historySocket())
	return errors.Is(err, os.ErrNotExist)
}

func seedRead(t *testing.T, a *app) {
	t.Helper()
	if err := a.lib.store.History().Put(context.Background(), store.Entry{ID: "r1", Kind: store.KindTranscribe, CreatedAt: time.Now(), Status: store.StatusDone, AppName: "Notes", RawText: "history text secret", Result: "history text secret"}); err != nil {
		t.Fatal(err)
	}
}

// waitFor pumps posts until ok holds.
func waitFor(t *testing.T, drain func(), ok func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		drain()
		if ok() {
			return
		}
	}
	t.Fatal("condition not reached")
}

// turnOnAccess flips the switch, answers the question and waits for the
// socket.
func turnOnAccess(t *testing.T, a *app, tt *ui.Tester, drain func()) {
	t.Helper()
	click(t, tt, "Allow reading")
	if !a.histAccess.ask {
		t.Fatalf("no question: %q", firstTexts(tt))
	}
	click(t, tt, "Turn on")
	drain()
	if a.histAccess.ask || !a.settings.Get().Privacy.HistoryAccess || socketGone(a) {
		t.Fatalf("not on: ask %v, setting %v, gone %v", a.histAccess.ask, a.settings.Get().Privacy.HistoryAccess, socketGone(a))
	}
}

// TestHistoryAccessOff: by default the setting is off and no socket
// exists, and the app opens that way too.
func TestHistoryAccessOff(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.openHistoryAccess()
	if a.prefs.Privacy.HistoryAccess || a.prefs.Privacy.HistorySelection {
		t.Errorf("defaults %+v", a.prefs.Privacy)
	}
	if !socketGone(a) {
		t.Error("a socket exists while off")
	}
	openPrivacy(t, a, tt)
	for _, s := range []string{"Other programs", accessTitle, "Also include the text you selected"} {
		if !has(tt, s) {
			t.Errorf("no %q", s)
		}
	}
	if has(tt, "Command line") {
		t.Error("the command shows while off")
	}
}

// TestHistoryAccessConfirm: the switch only turns on through the
// question; Cancel leaves everything off.
func TestHistoryAccessConfirm(t *testing.T) {
	a, _, tt := newTestApp(t)
	openPrivacy(t, a, tt)
	click(t, tt, "Allow reading")
	if !a.histAccess.ask || !has(tt, "Let programs on this Mac read the history?") {
		t.Fatalf("no question: %q", firstTexts(tt))
	}
	if a.settingsForm.historyAccess {
		t.Error("the switch shows on before the answer")
	}
	click(t, tt, "Cancel")
	if a.settings.Get().Privacy.HistoryAccess || !socketGone(a) || a.settingsForm.historyAccess {
		t.Errorf("cancelled: %+v gone %v", a.settings.Get().Privacy, socketGone(a))
	}
}

// TestHistoryAccessFlow is the whole path with the real client: on, a
// read shows in Last read, a new name raises one notice, the same name
// none; off, the socket goes and the client exits unavailable.
func TestHistoryAccessFlow(t *testing.T) {
	a, _, tt := newTestApp(t)
	drain := queuePosts(a)
	seedRead(t, a)
	openPrivacy(t, a, tt)
	turnOnAccess(t, a, tt, drain)
	fi, err := os.Stat(a.historySocket())
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		t.Log("skipping the socket mode assertion: Windows does not expose Unix permission bits")
	} else if fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", fi.Mode())
	}
	if !has(tt, "Command line") || !has(tt, "history export --day") {
		t.Errorf("no command: %q", tt.Texts())
	}
	for _, label := range []string{"Allow reading", "Include selected text", "Copy"} {
		if !slices.Contains(tt.AccessLabels(), label) {
			t.Errorf("no access label %q in %q", label, tt.AccessLabels())
		}
	}

	read := func(as string) {
		t.Helper()
		var got []historyapi.Row
		if _, err := historyapi.Export(a.historySocket(), historyapi.Request{Since: 1, As: as}, func(r historyapi.Row) error { got = append(got, r); return nil }); err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].Text != "history text secret" {
			t.Fatalf("rows %+v", got)
		}
	}
	read("daily-obsidian")
	waitFor(t, drain, func() bool { return len(a.histAccess.reads) == 1 && a.noticeCount("daily-obsidian") == 1 })
	drain()
	if line := readLine(a.histAccess.reads[0]); !strings.Contains(line, "1 entry") || !strings.Contains(line, "daily-obsidian") {
		t.Errorf("line %q", line)
	}
	if !has(tt, readLine(a.histAccess.reads[0])) {
		t.Errorf("Last read does not show: %q", tt.Texts())
	}
	read("daily-obsidian")
	waitFor(t, drain, func() bool { return len(a.histAccess.reads) == 2 })
	drain()
	if n := a.noticeCount("daily-obsidian"); n != 1 {
		t.Errorf("notices %d", n)
	}
	read("other")
	waitFor(t, drain, func() bool { return a.noticeCount("other") == 1 })

	// The log has no history text, even with Write text to the log on.
	a.updateSettings(func(s *settings.Settings) { s.Privacy.LogText = true })
	drain()
	read("third")
	waitFor(t, drain, func() bool { return len(a.histAccess.reads) == 4 })
	data, ok, err := a.lib.store.Documents().Get(context.Background(), historyReadsDoc)
	if err != nil || !ok {
		t.Fatalf("no document: %v", err)
	}
	if strings.Contains(string(data), "history text secret") {
		t.Errorf("the log has the text: %s", data)
	}

	// Read log dialog lists them.
	a.histAccess.listOpen = true
	if !has(tt, "Read log") {
		t.Errorf("no dialog: %q", firstTexts(tt))
	}
	a.histAccess.listOpen = false

	// Off: the socket goes and reading is unavailable.
	frames(tt)
	click(t, tt, "Allow reading")
	drain()
	if a.settings.Get().Privacy.HistoryAccess || !socketGone(a) {
		t.Fatalf("still on: %+v", a.settings.Get().Privacy)
	}
	if _, err := historyapi.FetchStatus(a.historySocket()); !errors.Is(err, historyapi.ErrUnavailable) {
		t.Errorf("after off: %v", err)
	}
	drain()
}

// TestHistoryAccessReopens: on at the start, the app serves again after
// a crash left a stale socket file.
func TestHistoryAccessReopens(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Privacy.HistoryAccess = true })
	if socketGone(a) {
		t.Fatal("no socket")
	}
	a.stopHistoryAccess()
	// A crash: the file stays and nothing listens.
	ln, err := historyapi.Listen(a.historySocket())
	if err != nil {
		t.Fatal(err)
	}
	if err := unlinkOnCloseOff(ln); err != nil {
		t.Fatal(err)
	}
	ln.Close()
	a.openHistoryAccess()
	if _, err := historyapi.FetchStatus(a.historySocket()); err != nil {
		t.Fatalf("after restart: %v", err)
	}
	a.closeHistoryAccess()
	if !socketGone(a) {
		t.Error("the socket outlived the app")
	}
}

// TestHistoryAccessQuit: quitting takes the socket with it.
func TestHistoryAccessQuit(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Privacy.HistoryAccess = true })
	if socketGone(a) {
		t.Fatal("no socket")
	}
	a.quit()
	if !socketGone(a) {
		t.Error("the socket outlived the app")
	}
}

// TestHistorySocketInDefaultFolder: the command looks in the default
// folder, so the socket stays there when the data folder has moved.
func TestHistorySocketInDefaultFolder(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.dataHome = "/default/home"
	if got := a.historySocket(); got != filepath.Join(a.dataHome, historyapi.SocketName) {
		t.Errorf("socket at %s", got)
	}
}

// TestSelectionNeedsItsSwitch: the selected text stays out until its
// own switch; it takes effect at once.
func TestSelectionNeedsItsSwitch(t *testing.T) {
	a, _, tt := newTestApp(t)
	drain := queuePosts(a)
	ctx := context.Background()
	if err := a.lib.store.History().Put(ctx, store.Entry{ID: "s1", Kind: store.KindTranslate, CreatedAt: time.Now(), Status: store.StatusDone, RawText: "heard", SourceText: "selected-secret", Result: "译文"}); err != nil {
		t.Fatal(err)
	}
	openPrivacy(t, a, tt)
	turnOnAccess(t, a, tt, drain)
	get := func() string {
		t.Helper()
		var sel string
		if _, err := historyapi.Export(a.historySocket(), historyapi.Request{Since: 1}, func(r historyapi.Row) error { sel = r.Selection; return nil }); err != nil {
			t.Fatal(err)
		}
		drain()
		return sel
	}
	if s := get(); s != "" {
		t.Errorf("selection went out: %q", s)
	}
	click(t, tt, "Include selected text")
	drain()
	if s := get(); s != "selected-secret" {
		t.Errorf("selection %q", s)
	}
}

// TestHistorySeenCap: the log and the names stay bounded; the newest
// survive; a name past the cap counts as new again.
func TestHistorySeenCap(t *testing.T) {
	var d historyReads
	for i := range historySeenKept + 5 {
		d.addRead(historyapi.Read{At: time.Now(), Program: "p" + strings.Repeat("x", i)})
	}
	if len(d.Reads) != historyReadsKept || len(d.Seen) != historySeenKept {
		t.Errorf("reads %d seen %d", len(d.Reads), len(d.Seen))
	}
	if isNew := d.addRead(historyapi.Read{Program: "p"}); !isNew {
		t.Error("an evicted name did not count as new")
	}
	if d.Seen[len(d.Seen)-1] != "p" {
		t.Errorf("newest not last: %v", d.Seen[len(d.Seen)-1])
	}
}

func TestHistoryReadDoc(t *testing.T) {
	data, err := json.Marshal(historyReads{Reads: []historyapi.Read{{Op: historyapi.OpExport, Count: 3, Program: "launchd", As: "daily"}}, Seen: []string{"daily (launchd)"}})
	if err != nil || strings.Contains(string(data), "text") {
		t.Fatalf("%s %v", data, err)
	}
}

// noticeCount is how many notifications name s.
func (a *app) noticeCount(s string) int {
	n := 0
	for _, x := range a.notices {
		if strings.Contains(x.Title, s) {
			n++
		}
	}
	return n
}

func unlinkOnCloseOff(ln interface{}) error {
	if u, ok := ln.(interface{ SetUnlinkOnClose(bool) }); ok {
		u.SetUnlinkOnClose(false)
		return nil
	}
	return errors.New("not a unix listener")
}

// historyAccessShots renders Settings → Privacy's section with the
// access on, and the confirmation (TestSnapshots).
func historyAccessShots(a *app, tt *ui.Tester, save func(string)) {
	orig := historyCommandPath
	historyCommandPath = func() string { return "/Applications/Voxt.app/Contents/MacOS/Voxt" }
	defer func() { historyCommandPath = orig }()
	a.navigate("/settings/privacy")
	a.updateSettings(func(s *settings.Settings) { s.Privacy.HistoryAccess = true })
	defer a.updateSettings(func(s *settings.Settings) { s.Privacy.HistoryAccess = false })
	a.histAccess.reads = []historyapi.Read{{At: time.Date(2026, 3, 4, 7, 0, 0, 0, time.Local), Op: historyapi.OpExport, Count: 23, Program: "launchd", As: "daily-obsidian"}}
	defer func() { a.histAccess.reads = nil }()
	frames(tt)
	w, h := snapshotSize()
	x, y := float32(w)*0.6, float32(h)*0.5
	r, _ := tt.Find(i18n.T("Other programs"))
	dy := r.Y - float32(h)*0.2
	tt.Scroll(x, y, 0, dy)
	defer tt.Scroll(x, y, 0, -dy) // the settings' own snapshots start at the top
	save("settings-privacy-access")
	a.histAccess.ask = true
	save("settings-privacy-access-ask")
	a.histAccess.ask = false
	frames(tt)
}
