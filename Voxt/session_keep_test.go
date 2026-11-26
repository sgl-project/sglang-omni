package main

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/egoist/mygo"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// keptApp starts the app over dir with sessions on and reads the saved
// sessions, as main does at start. opts with a rewrite script ends the
// first Rewrite in a card.
func keptApp(t *testing.T, dir, opts string, chat *fakeChat, said string, keepHistory bool) (*sessionApp, *fakeTranscriptions) {
	t.Helper()
	set, models, tr := fileSetup(t, said, chat)
	a := newSessionAppIn(t, dir, opts, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard, s.Features.SessionsOn, s.Privacy.KeepHistory = true, true, keepHistory
	}, models)
	a.loadSessions()
	a.pump("the saved sessions", func() bool { return a.sessionsLoaded })
	return a, tr
}

// rewriteScript is the helper's options for one Rewrite of chatA.
func rewriteScript(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "selection")
	selectNow(t, path, chatA)
	return helpertest.Opt("selection-file", path) + "," + script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
}

// savedDoc is the store's sessions document, "" when there is none.
func savedDoc(t *testing.T, a *sessionApp) string {
	t.Helper()
	data, ok, err := a.lib.store.Documents().Get(context.Background(), sessionsKey)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		return ""
	}
	return string(data)
}

// menuLabels is the menu bar's session list: "label*" is checked.
func menuLabels(a *sessionApp) ([]string, []*mygo.MenuItem) {
	items := a.sessionsMenu(func(f func()) func(*mygo.MenuItem, *mygo.Window) {
		return func(*mygo.MenuItem, *mygo.Window) { f() }
	})
	var labels []string
	for _, item := range items {
		label := item.Label
		if item.Checked {
			label += "*"
		}
		labels = append(labels, label)
	}
	return labels, items
}

// twoSessions leaves two sessions with a card each, session 1 current:
// "Docs status" with answer "one", and "My haiku" (renamed) with "two".
func twoSessions(t *testing.T, dir string) *sessionApp {
	t.Helper()
	chat := &fakeChat{texts: []string{"one", "Docs status", "two", "Haiku"}}
	a, tr := keptApp(t, dir, rewriteScript(t), chat, "summarize", true)
	a.pump("the card", func() bool { return a.sess.State.Ended() && a.answer.open })
	a.pump("the first title", func() bool { return a.slots[0].title == "Docs status" })
	a.newSession()
	rewriteSaid(t, a, tr, textEditApp, "write a haiku", "two")
	a.pump("the second title", func() bool { return a.slots[1].title == "Haiku" })
	a.nameSession(1, "My haiku")
	a.switchSession(0)
	a.pump("both turns in history", func() bool { return len(a.history()) == 2 })
	a.pump("the sessions on disk", func() bool {
		doc := savedDoc(t, a)
		return strings.Contains(doc, `"current":0`) && strings.Contains(doc, "My haiku")
	})
	return a
}

// TestSessionsSurviveRestart: the list, its order, the titles, the
// user's name and the current session come back, each session with its
// card from the history. Nothing comes on screen by itself; the menu
// bar's list shows the cards.
func TestSessionsSurviveRestart(t *testing.T) {
	dir := t.TempDir()
	t.Run("before restart", func(t *testing.T) { twoSessions(t, dir) })

	a, tr := keptApp(t, dir, "", &fakeChat{texts: []string{"three"}}, "shorter", true)
	if a.answer.open || a.emptyCard {
		t.Fatalf("a card came on screen at start: %+v", a.answer)
	}
	if len(a.slots) != 2 || a.slot != 0 || a.slots[0].answer.text != "one" || a.slots[1].answer.text != "two" {
		t.Fatalf("restored: current %d, %+v", a.slot, a.slots)
	}
	labels, items := menuLabels(a)
	if got := strings.Join(labels, " | "); got != "Sessions | 1 Docs status* | 2 My haiku | New session | Rename session" {
		t.Fatalf("menu: %q", got)
	}
	// The current session's card waits in it until it is asked for.
	items[1].Click(nil, nil)
	if !a.answer.open || a.answer.text != "one" {
		t.Fatalf("session 1 from the menu: %+v", a.answer)
	}
	if !a.boxHoldsCard(ovAnswer) {
		t.Fatal("the restored Rewrite answer is not in the Rewrite card")
	}
	items[2].Click(nil, nil)
	if a.slot != 1 || !a.answer.open || a.answer.text != "two" {
		t.Fatalf("session 2 from the menu: slot %d, %+v", a.slot, a.answer)
	}
	if got := chipLine(a); got != "1 Docs sta… | 2 My haiku*" {
		t.Fatalf("row: %q", got)
	}
	if labels, _ := menuLabels(a); labels[2] != "2 My haiku*" {
		t.Fatalf("menu after the switch: %q", labels)
	}
	// A follow-up continues the restored conversation; a new session
	// gets the next free number.
	rewriteSaid(t, a, tr, a.answer.target, "shorter", "three")
	if len(a.answer.turns) != 2 {
		t.Fatalf("follow-up: %d turns", len(a.answer.turns))
	}
	a.newSession()
	if a.slot != 2 {
		t.Fatalf("new session id %d, want 2", a.slot)
	}
}

// TestSessionsEndWithHistoryOff: with Keep a history off nothing is
// restored and the saved sessions are deleted, at start and at once
// when the switch goes off; the empty card says so.
func TestSessionsEndWithHistoryOff(t *testing.T) {
	dir := t.TempDir()
	t.Run("before restart", func(t *testing.T) { twoSessions(t, dir) })

	a, _ := keptApp(t, dir, "", &fakeChat{}, "", false)
	if len(a.sessions()) != 1 {
		t.Fatalf("history off restored %d sessions", len(a.sessions()))
	}
	a.pump("the saved sessions deleted", func() bool { return savedDoc(t, a) == "" })
	a.newSession()
	if text := a.cachedCard().Blocks[0].Text; !a.cachedCard().Empty || !strings.Contains(text, "History is off, so sessions end when Voxt quits.") {
		t.Fatalf("empty card with history off: %q", text)
	}
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = true })
	a.pump("saved with history on", func() bool { return strings.Contains(savedDoc(t, a), `"current":1`) })
	if text := a.cachedCard().Blocks[0].Text; strings.Contains(text, "History is off") {
		t.Fatalf("empty card with history on: %q", text)
	}
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = false })
	a.pump("deleted when history goes off", func() bool { return savedDoc(t, a) == "" })

	// The Sessions switch off deletes them too.
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = true })
	a.pump("saved again", func() bool { return savedDoc(t, a) != "" })
	a.setSessionsOn(false)
	a.pump("deleted with sessions off", func() bool { return savedDoc(t, a) == "" })
	if labels, _ := menuLabels(a); len(labels) != 0 {
		t.Fatalf("menu with sessions off: %q", labels)
	}
}

// TestClearHistoryEndsSessions: the other sessions' cards are history,
// so Clear history leaves only the current session, untitled.
func TestClearHistoryEndsSessions(t *testing.T) {
	a := twoSessions(t, t.TempDir())
	a.clearHistory()
	a.pump("one session", func() bool { return len(a.slots) == 1 })
	if p := a.slots[0]; p.id != 0 || p.title != "" || p.name != "" {
		t.Fatalf("left: %+v", p)
	}
	a.pump("the saved list", func() bool { return !strings.Contains(savedDoc(t, a), "My haiku") })
}

// TestSessionPillMark: the recording bar names the session only while
// there are two or more.
func TestSessionPillMark(t *testing.T) {
	if got := sessionMarked(2, "Mail"); got != "#2 · Mail" {
		t.Fatalf("marked: %q", got)
	}
	if got := sessionMarked(0, "Mail"); got != "Mail" {
		t.Fatalf("unmarked: %q", got)
	}
	a, _ := keptApp(t, t.TempDir(), "", &fakeChat{}, "", true)
	if n := a.sessionNumber(); n != 0 {
		t.Fatalf("one session: number %d", n)
	}
	a.newSession()
	if n := a.sessionNumber(); n != 2 {
		t.Fatalf("second of two: number %d", n)
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.SessionsOn = false })
	if n := a.sessionNumber(); n != 0 {
		t.Fatalf("sessions off: number %d", n)
	}
}

// TestSessionHistoryHints: Experimental warns that sessions end at quit
// while history is off, with a way to Privacy, whose Keep a history
// row then says what it covers.
func TestSessionHistoryHints(t *testing.T) {
	a, _, tt := newTestApp(t)
	const warn = "History is off, so sessions end when Voxt quits. Turn on Keep a history in Privacy to keep them."
	const covers = "When off, the menu bar and Dock menus do not list cards to reopen, and sessions end when Voxt quits. Clear history ends the other sessions too."
	a.navigate("/settings/experimental")
	a.updateSettings(func(s *settings.Settings) { s.Features.SessionsOn = true })
	if has(tt, warn) {
		t.Fatal("warned with history on")
	}
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = false })
	if !has(tt, warn) {
		t.Fatalf("no warning in %q", tt.Texts())
	}
	scrollToText(tt, "Open Privacy")
	click(t, tt, "Open Privacy")
	if a.shownPath() != "/settings/privacy" || !has(tt, covers) {
		t.Fatalf("Privacy: %s %q", a.shownPath(), tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.SessionsOn = false })
	if has(tt, covers) {
		t.Fatal("Privacy names sessions with the switch off")
	}
}

// TestDeletedEntryUntitlesSession: a session's automatic title comes
// from its first instruction, so it goes with that history entry; the
// name the user gave stays, and the menu calls the session New session.
func TestDeletedEntryUntitlesSession(t *testing.T) {
	a := twoSessions(t, t.TempDir())
	for _, entry := range a.history() {
		a.deleteHistoryEntry(entry.ID)
	}
	a.pump("the title gone", func() bool { return a.slots[0].title == "" })
	a.pump("the saved title gone", func() bool { return !strings.Contains(savedDoc(t, a), "Docs status") })
	if p := a.slots[1]; p.name != "My haiku" {
		t.Fatalf("the user's name: %+v", p)
	}
	if !a.answer.open {
		t.Fatal("the card on screen closed")
	}
	labels, _ := menuLabels(a)
	if got := strings.Join(labels, " | "); got != "Sessions | 1 New session* | 2 My haiku | New session | Rename session" {
		t.Fatalf("menu: %q", got)
	}
}
