package main

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// reopenItems are the menu's labels under Reopen card, top first.
func reopenItems(a *sessionApp) []string {
	var out []string
	in := false
	for _, item := range a.trayItems() {
		switch {
		case item.Label == "Reopen card":
			in = true
		case in && item.Click != nil && strings.Contains(item.Label, " · "):
			out = append(out, item.Label)
		case in:
			return out
		}
	}
	return out
}

func waitReopenItems(a *sessionApp, want ...string) {
	a.pump("the Reopen card menu", func() bool { return slices.Equal(reopenItems(a), want) })
}

func reopenEntry(id, instruction, result string, at time.Time) store.Entry {
	return store.Entry{ID: id, Conversation: id, Kind: store.KindRewrite, CreatedAt: at, Status: store.StatusDone,
		Delivery: "card", PromptText: instruction, Result: result, AppBundle: "com.tinyspeck.slackmacgap", AppName: "Slack"}
}

// The menu lists only unfinished conversations from the last day, five at most.
func TestReopenConversationsFilters(t *testing.T) {
	now := time.Date(2026, 11, 15, 12, 0, 0, 0, time.UTC)
	open := reopenEntry("open", "summarize", "one", now.Add(-time.Minute))
	inserted := reopenEntry("inserted", "decline", "two", now.Add(-2*time.Minute))
	inserted.Delivery = "insert"
	old := reopenEntry("old", "translate", "three", now.Add(-24*time.Hour))
	withoutID := reopenEntry("without", "older format", "four", now)
	withoutID.Conversation = ""
	failed := reopenEntry("failed", "failed", "five", now)
	failed.Status = store.StatusFailed
	got := reopenConversations([]store.Entry{open, inserted, old, withoutID, failed}, now)
	if len(got) != 1 || got[0].conversation != "open" || got[0].label != "Slack · summarize" {
		t.Fatalf("filtered conversations %+v", got)
	}

	var entries []store.Entry
	for i := range 6 {
		entries = append(entries, reopenEntry(fmt.Sprint(i), "ask "+fmt.Sprint(i), "answer", now.Add(-time.Duration(i)*time.Minute)))
	}
	got = reopenConversations(entries, now)
	if len(got) != reopenCardLimit || got[0].label != "Slack · ask 0" || got[4].label != "Slack · ask 4" {
		t.Fatalf("five newest conversations %+v", got)
	}
}

// A redo replaces its earlier answer instead of adding a turn; titles survive.
func TestCardFromHistoryRedo(t *testing.T) {
	now := time.Now()
	first := reopenEntry("first", "summarize", "old answer", now.Add(-3*time.Minute))
	first.Title = "Old"
	redo := reopenEntry("redo", "summarize", "new answer", now.Add(-2*time.Minute))
	redo.Conversation, redo.Replaces, redo.Title = "first", "first", "New"
	followUp := reopenEntry("follow-up", "shorter", "final answer", now.Add(-time.Minute))
	followUp.Conversation = "first"
	card, ok := cardFromHistory([]store.Entry{followUp, redo, first}, "first")
	if !ok || len(card.turns) != 2 || card.id != "follow-up" || card.text != "final answer" || card.turns[0].Answer.Content != "new answer" || card.turns[0].Answer.Title != "New" {
		t.Fatalf("restored %+v", card)
	}
}

// Typed text returns with its original role after a restart: typed-only
// text is the first turn's source, while selected and typed text stay apart.
func TestCardFromHistoryTyped(t *testing.T) {
	for _, tc := range []struct{ name, selected, typed string }{
		{"typed only", "", "a rough draft"},
		{"selected and typed", "the selected text", "and the tickets"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := reopenEntry("typed", "make it shorter", "A shorter text.", time.Now())
			entry.SourceText, entry.TypedText = tc.selected, tc.typed
			card, ok := cardFromHistory([]store.Entry{entry}, "typed")
			if !ok || card.input.Typed != tc.typed || len(card.turns) != 1 {
				t.Fatalf("restored %+v", card)
			}
			turn := card.turns[0]
			if tc.selected == "" {
				if turn.Selection != tc.typed || turn.Typed != "" {
					t.Errorf("typed-only turn %+v", turn)
				}
			} else if turn.Selection != tc.selected || turn.Typed != tc.typed {
				t.Errorf("selected and typed turn %+v", turn)
			}
		})
	}
}

func TestCardFromHistoryRestoresTextPreviews(t *testing.T) {
	for _, tc := range []struct {
		name, selection, typed string
		wantLabels             []string
	}{
		{name: "selection and typed", selection: "selected source", typed: "typed source", wantLabels: []string{"Selected text", "Text"}},
		{name: "typed only", typed: "typed source", wantLabels: []string{"Text"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := reopenEntry("rewrite", "polish", "Polished", time.Now())
			entry.SourceText, entry.TypedText = tc.selection, tc.typed
			card, ok := cardFromHistory([]store.Entry{entry}, "rewrite")
			if !ok || len(card.previews) != len(tc.wantLabels) {
				t.Fatalf("restored previews %+v", card.previews)
			}
			for index, label := range tc.wantLabels {
				preview := card.previews[index]
				if preview.Label != label || (index == 0 && tc.selection != "" && preview.Text != tc.selection) || (label == "Text" && preview.Text != tc.typed) {
					t.Errorf("preview %d %+v, want label %q", index, preview, label)
				}
			}
		})
	}
}

// A translation that became a Rewrite conversation is restored as its first turn.
func TestCardFromHistoryTranslation(t *testing.T) {
	now := time.Now()
	translation := reopenEntry("translate", "", "你好", now.Add(-2*time.Minute))
	translation.Kind, translation.RawText, translation.TargetLang = store.KindTranslate, "hello", "zh-Hans"
	rewrite := reopenEntry("rewrite", "shorter", "短", now.Add(-time.Minute))
	rewrite.Conversation = "translate"
	card, ok := cardFromHistory([]store.Entry{rewrite, translation}, "translate")
	wantInstruction := i18n.T("Translate into %s", uitext.TargetLanguageLabel("zh-Hans"))
	if !ok || card.kind != store.KindRewrite || len(card.turns) != 2 || card.turns[0].Instruction != wantInstruction || card.turns[0].Selection != "hello" || card.turns[0].Typed != "" || card.turns[0].Answer.Content != "你好" || card.turns[1].Instruction != "shorter" {
		t.Fatalf("restored %+v", card)
	}
}

func TestCardFromHistoryTypedText(t *testing.T) {
	for _, tc := range []struct {
		name, selection, typed, turnSelection, turnTyped string
	}{
		{name: "selection and typed", selection: "draft", typed: "typed context", turnSelection: "draft", turnTyped: "typed context"},
		{name: "typed only", typed: "typed context", turnSelection: "typed context"},
		{name: "older entry without typed text", selection: "draft", turnSelection: "draft"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := reopenEntry("rewrite", "polish", "Polished", time.Now())
			entry.SourceText, entry.TypedText = tc.selection, tc.typed
			card, ok := cardFromHistory([]store.Entry{entry}, "rewrite")
			if !ok || len(card.turns) != 1 {
				t.Fatalf("restored %+v", card)
			}
			if card.input.Raw != "polish" || card.input.Selection != tc.selection || card.input.Typed != tc.typed {
				t.Errorf("redo input %+v", card.input)
			}
			turn := card.turns[0]
			if turn.Instruction != "polish" || turn.Selection != tc.turnSelection || turn.Typed != tc.turnTyped || turn.Answer.Content != "Polished" {
				t.Errorf("restored turn %+v", turn)
			}
		})
	}
}

func TestReopenCardAfterRestartTypedInput(t *testing.T) {
	for _, tc := range []struct {
		name, selection, typed string
		followUp               bool
	}{
		{name: "typed only redo", typed: "typed context"},
		{name: "typed only follow-up", typed: "typed context", followUp: true},
		{name: "selection and typed redo", selection: "draft", typed: "typed context"},
		{name: "selection and typed follow-up", selection: "draft", typed: "typed context", followUp: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			var conversation, firstUser string
			t.Run("before restart", func(t *testing.T) {
				chat := &fakeChat{text: `{"title":"Polished","content":"Polished"}`}
				set, models, _ := fileSetup(t, "make it shorter", chat)
				a := newSessionAppIn(t, dir, helpertest.Opt("selection", tc.selection)+","+boxScript(t), func(s *settings.Settings) {
					set(s)
					s.Features.Rewrite.ShowCard = true
				}, models)
				a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
				a.postQueue(func() { a.box.text = tc.typed; a.boxSync() })
				a.pump("the answer", func() bool { return a.answer.open })
				reqs := chat.requests()
				if len(reqs) != 1 {
					t.Fatalf("%d chats", len(reqs))
				}
				firstUser = lastUserOf(reqs[0])
				conversation = a.answer.conv
				a.pump("the history entry", func() bool { return len(a.history()) == 1 })
				entry := a.history()[0]
				if entry.TypedText != tc.typed || entry.SourceText != tc.selection {
					t.Fatalf("saved typed input %q and selection %q, want %q and %q", entry.TypedText, entry.SourceText, tc.typed, tc.selection)
				}
			})

			chat := &fakeChat{text: `{"title":"Polished again","content":"Polished again"}`}
			set, models, tr := fileSetup(t, "more formal", chat)
			a := newSessionAppIn(t, dir, "", func(s *settings.Settings) {
				set(s)
				s.Features.Rewrite.ShowCard = true
			}, models)
			a.restoreAnswer(conversation)
			a.pump("the restored card", func() bool { return a.answer.open })
			if tc.followUp {
				a.recordFollowUp(t, tr, "more formal")
				messages := chatMessages(chat.requests()[0])
				if len(messages) != 4 || messages[1][1] != firstUser {
					t.Fatalf("follow-up changed the restored turn: %q\nwant first user:\n%s", messages, firstUser)
				}
			} else {
				input := a.answer.input
				a.redoAnswer(session.StartRequest{Kind: store.KindRewrite, Target: a.answer.target, Text: &input})
				a.pump("the restored redo", func() bool { return len(chat.requests()) == 1 && a.answer.redo == "" })
				if got := lastUserOf(chat.requests()[0]); got != firstUser {
					t.Fatalf("redo changed the input:\n%s\nwant:\n%s", got, firstUser)
				}
			}
		})
	}
}

// Closing a card leaves it in the menu; reopening restores every turn and its
// target's current process, and Rewrite sends those turns with the follow-up.
func TestReopenCardAndFollowUp(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "decline politely")
	slack := a.answer.target
	rewriteSaid(t, a, tr, slack, "shorter", "two")
	conversation := a.answer.conv
	a.closeAnswer()
	waitReopenItems(a, "Slack · decline politely")
	a.restoreAnswer(conversation)
	a.pump("the restored card", func() bool { return a.answer.open })
	card := a.answer
	if len(card.turns) != 2 || !card.conversation || card.text != "two" || card.target != slack || card.turns[0].Selection != chatA {
		t.Fatalf("restored %+v; target want %+v", card, slack)
	}
	waitReopenItems(a)
	rewriteSaid(t, a, tr, slack, "warmer", "three")
	messages := chatMessages(chat.requests()[2])
	if len(messages) != 6 || !strings.Contains(messages[1][1], "decline politely") || !strings.Contains(messages[3][1], "shorter") || !strings.Contains(messages[5][1], "warmer") {
		t.Fatalf("follow-up messages %q", messages)
	}
	a.pump("the follow-up in history", func() bool { return len(a.history()) == 3 })
	for _, entry := range a.history() {
		if entry.Conversation != conversation {
			t.Fatalf("%q belongs to %q, want %q", entry.PromptText, entry.Conversation, conversation)
		}
	}
}

// A new app over the same data directory lists and restores the conversation.
func TestReopenCardAfterRestart(t *testing.T) {
	dir := t.TempDir()
	var conversation string
	var slack proto.App
	t.Run("before restart", func(t *testing.T) {
		chat := &fakeChat{texts: []string{"one", "two"}}
		path := filepath.Join(t.TempDir(), "selection")
		selectNow(t, path, chatA)
		set, models, tr := fileSetup(t, "decline politely", chat)
		events := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
		a := newSessionAppIn(t, dir, helpertest.Opt("selection-file", path)+","+events, func(s *settings.Settings) { set(s); s.Features.Rewrite.ShowCard = true }, models)
		a.pump("the card", func() bool { return a.sess.State.Ended() && a.answer.open })
		slack = a.answer.target
		rewriteSaid(t, a, tr, slack, "shorter", "two")
		conversation = a.answer.conv
		a.pump("both turns in history", func() bool { return len(a.history()) == 2 })
	})

	chat := &fakeChat{texts: []string{"three"}}
	set, models, tr := fileSetup(t, "warmer", chat)
	a := newSessionAppIn(t, dir, "", func(s *settings.Settings) { set(s); s.Features.Rewrite.ShowCard = true }, models)
	if a.answer.open {
		t.Fatalf("the new app opened a card: %+v", a.answer)
	}
	a.loadTrayRecent()
	waitReopenItems(a, "Slack · decline politely")
	a.restoreAnswer(conversation)
	a.pump("the restored card", func() bool { return a.answer.open })
	if len(a.answer.turns) != 2 || a.answer.text != "two" || a.answer.target != slack {
		t.Fatalf("restored %+v; target want %+v", a.answer, slack)
	}
	rewriteSaid(t, a, tr, slack, "warmer", "three")
	messages := chatMessages(chat.requests()[0])
	if len(messages) != 6 || !strings.Contains(messages[1][1], "decline politely") || !strings.Contains(messages[3][1], "shorter") || !strings.Contains(messages[5][1], "warmer") {
		t.Fatalf("follow-up messages %q", messages)
	}
}

// Deleting an entry or clearing history removes its menu row and prevents reopening.
func TestReopenCardDeleteAndClear(t *testing.T) {
	for _, test := range []struct {
		name  string
		clear bool
	}{
		{name: "delete"},
		{name: "clear", clear: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, _, _ := rewriteSelecting(t, &fakeChat{text: "one"}, chatA, "summarize")
			conversation := a.answer.conv
			a.closeAnswer()
			waitReopenItems(a, "Slack · summarize")
			if test.clear {
				a.clearHistory()
			} else {
				a.deleteHistoryEntry(conversation)
			}
			a.pump("the history removal", func() bool { return len(a.history()) == 0 && len(reopenItems(a)) == 0 })
			a.restoreAnswer(conversation)
			a.pump("the ignored restore", func() bool { return len(a.queue) == 0 })
			if a.answer.open {
				t.Fatalf("reopened deleted history: %+v", a.answer)
			}
		})
	}
}

// Deleting one turn leaves the others available to restore.
func TestReopenCardWithoutDeletedTurn(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	rewriteSaid(t, a, tr, a.answer.target, "shorter", "two")
	conversation := a.answer.conv
	a.closeAnswer()
	waitReopenItems(a, "Slack · summarize")
	a.deleteHistoryEntry(conversation)
	a.pump("the deleted first turn", func() bool { return len(a.history()) == 1 && len(a.reopen) == 1 })
	a.restoreAnswer(conversation)
	a.pump("the restored remaining turn", func() bool { return a.answer.open })
	if len(a.answer.turns) != 1 || a.answer.turns[0].Instruction != "shorter" || a.answer.text != "two" {
		t.Fatalf("restored %+v", a.answer)
	}
}

// Inserting the current answer finishes the conversation and removes its row.
func TestReopenCardLeavesOutInserted(t *testing.T) {
	a, _, _ := rewriteSelecting(t, &fakeChat{text: "one"}, chatA, "summarize")
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if got := reopenItems(a); len(got) != 0 {
		t.Fatalf("the inserted conversation stayed in the menu: %q", got)
	}
	a.pump("the delivery in history", func() bool { return len(a.history()) == 1 && a.history()[0].Delivery == "insert" })
}

// Turning history off hides even conversations already stored, and new cards
// do not create history or reopen rows.
func TestReopenCardWithoutHistory(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	a.closeAnswer()
	waitReopenItems(a, "Slack · summarize")
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = false })
	waitReopenItems(a)
	rewriteSaid(t, a, tr, textEditApp, "another", "two")
	a.closeAnswer()
	if len(a.history()) != 1 || len(reopenItems(a)) != 0 {
		t.Fatalf("history %d, menu %q", len(a.history()), reopenItems(a))
	}
}

// With the menu bar icon off, the Dock menu offers the same conversation and
// clears when the menu bar icon comes back.
func TestReopenCardFromDockMenu(t *testing.T) {
	a, _, _ := rewriteSelecting(t, &fakeChat{text: "one"}, chatA, "summarize")
	a.closeAnswer()
	waitReopenItems(a, "Slack · summarize")
	var dock *mygo.Menu
	oldSetDockMenu := setDockMenu
	setDockMenu = func(menu *mygo.Menu) { dock = menu }
	t.Cleanup(func() { setDockMenu, a.hasMenu, a.tray = oldSetDockMenu, false, nil })
	a.hasMenu, a.prefs.General.MenuBar = true, false
	a.applyTray()
	a.pump("the Dock menu", func() bool {
		return dock != nil && findMenuItem(dock.Items(), "Slack · summarize") != nil
	})
	row := findMenuItem(dock.Items(), "Slack · summarize")
	row.Click(row, nil)
	a.pump("the card restored from the Dock", func() bool { return a.answer.open })
	if len(a.answer.turns) != 1 || a.answer.text != "one" {
		t.Fatalf("restored %+v", a.answer)
	}
	a.prefs.General.MenuBar = true
	a.tray = &mygo.Tray{} // already present: applyTray need not create a real one
	a.applyTray()
	if dock != nil {
		t.Fatal("the Dock fallback stayed after the menu bar icon returned")
	}
}
