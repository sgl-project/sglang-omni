package main

// Sessions over a restart (#4005). The list (order, titles, the current
// session) and which conversation each session's cards hold are saved
// in the store's "sessions" document; the cards themselves are rebuilt
// from the history, which stays the only copy of their text
// (session_restore.go). It follows Settings → Privacy's Keep a history:
// with history off, or sessions off, nothing is written and the
// document is deleted at once, so those sessions end when Voxt quits.
// An answer that was still being written is not resumed.

import (
	"context"
	"encoding/json"
	"slices"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const sessionsKey = "sessions"

type savedSession struct {
	ID    int    `json:"id"`
	Title string `json:"title,omitempty"`
	Name  string `json:"name,omitempty"`
	// Card and Under are the conversations of the session's card and of
	// the card waiting under it.
	Card  string `json:"card,omitempty"`
	Under string `json:"under,omitempty"`
}

type savedSessions struct {
	Current  int            `json:"current"`
	Sessions []savedSession `json:"sessions"`
}

// sessionsKept says whether the sessions are saved.
func (a *app) sessionsKept() bool {
	return a.prefs.Features.SessionsOn && a.prefs.Privacy.KeepHistory
}

// keepSessions saves the sessions when they changed, or deletes the
// saved ones when they are no longer kept. Before the saved ones were
// read (loadSessions) it does nothing: they must not be overwritten.
func (a *app) keepSessions() {
	if !a.sessionsLoaded || a.lib == nil || a.lib.store == nil {
		return
	}
	doc := ""
	if a.sessionsKept() {
		saved := savedSessions{Current: a.slot}
		for _, p := range a.sessions() {
			answer, under := p.answer, p.under
			if p.id == a.slot && (a.answer.open || !answer.open) {
				answer, under = a.answer, a.under
			}
			saved.Sessions = append(saved.Sessions, savedSession{ID: p.id, Title: p.title, Name: p.name, Card: answer.conv, Under: under.conv})
		}
		data, _ := json.Marshal(saved)
		doc = string(data)
	}
	if doc == a.sessionsSaved {
		return
	}
	a.sessionsSaved = doc
	seq := a.sessionsSeq.Add(1)
	documents := a.lib.store.Documents()
	a.async(func() {
		// One write at a time, and only the newest: an older one that
		// comes late must not undo it.
		a.sessionsWrite.Lock()
		defer a.sessionsWrite.Unlock()
		if seq != a.sessionsSeq.Load() {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if doc == "" {
			_ = documents.Delete(ctx, sessionsKey)
		} else {
			_ = documents.Put(ctx, sessionsKey, []byte(doc))
		}
	})
}

// loadSessions brings the saved sessions back at start, their cards
// from the history.
func (a *app) loadSessions() {
	kept := a.sessionsKept()
	documents, history := a.lib.store.Documents(), a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var saved savedSessions
		cards := map[string]answerCard{}
		if data, ok, err := documents.Get(ctx, sessionsKey); kept && err == nil && ok && json.Unmarshal(data, &saved) == nil {
			for _, s := range saved.Sessions {
				for _, conversation := range []string{s.Card, s.Under} {
					if conversation == "" {
						continue
					}
					entries, err := history.List(ctx, store.HistoryQuery{Conversation: conversation, Finished: true})
					if card, ok := cardFromHistory(entries, conversation); err == nil && ok {
						cards[conversation] = card
					}
				}
			}
		}
		a.post(func() {
			a.restoreSessions(saved, cards)
			// Whatever is on disk is replaced, or deleted when not kept.
			a.sessionsLoaded, a.sessionsSaved = true, "\x00"
			a.keepSessions()
		})
	})
}

// restoreSessions puts the saved sessions in place, unless the user is
// already at work. No card comes on screen by itself: a session's card
// waits in it until the session is asked for (showSession). A session
// whose card is gone from the history loses its automatic title too.
func (a *app) restoreSessions(saved savedSessions, cards map[string]answerCard) {
	if len(saved.Sessions) == 0 || !a.sessionsKept() || len(a.slots) > 1 || a.answer.open || a.sess.State.Live() || a.bgID != "" {
		return
	}
	var slots []parkedSlot
	for _, s := range saved.Sessions {
		if slices.ContainsFunc(slots, func(p parkedSlot) bool { return p.id == s.ID }) {
			continue
		}
		p := parkedSlot{id: s.ID, title: s.Title, name: s.Name, answer: cards[s.Card], under: cards[s.Under]}
		if !p.answer.open {
			p.answer, p.under = p.under, answerCard{}
		}
		if s.Card != "" && !p.answer.open {
			p.title = ""
		}
		slots = append(slots, p)
		a.slotSeq = max(a.slotSeq, s.ID)
	}
	a.slots, a.slot, a.emptyCard = slots, slots[0].id, false
	if a.slotIndex(saved.Current) >= 0 {
		a.slot = saved.Current
	}
	a.buildTrayMenu()
	// Insert needs each card's app as it runs now.
	helperCall(a, proto.MethodAppList, proto.AppListParams{}, 5*time.Second, func(result proto.AppListResult, _ error) {
		pid := func(card *answerCard) {
			for _, running := range result.Apps {
				if card.open && running.Running && running.BundleID == card.target.BundleID {
					card.target.PID = running.PID
					return
				}
			}
		}
		for i := range a.slots {
			pid(&a.slots[i].answer)
			pid(&a.slots[i].under)
		}
	})
}

// showSession brings session id on screen (the menu bar's session list,
// the next-session shortcut): a switch to it, or the current session's
// card, which waits in it after a restart, or its empty card.
func (a *app) showSession(id int) {
	n := a.slotIndex(id)
	if id != a.slot || n < 0 {
		a.switchSession(id)
		return
	}
	if a.answer.open || a.sess.State.Live() || a.box.open || a.bgVisible() {
		return
	}
	p := &a.slots[n]
	a.answer, a.under, a.followUp = p.answer, p.under, p.followUp
	p.answer, p.under, p.followUp = answerCard{}, answerCard{}, ""
	a.emptyCard = a.prefs.Features.SessionsOn && !a.answer.open
	a.boxForAnswer()
	a.buildTrayMenu()
	a.overlaySync()
}

// endSessions is Clear history's part here: the other sessions' cards
// are history too, so only the current session stays, untitled, with
// the card on screen as before.
func (a *app) endSessions() {
	if a.bgID != "" && a.bgSlot != a.slot && a.mgr != nil {
		a.mgr.Cancel(a.bgID)
	}
	a.slots, a.renaming = []parkedSlot{{id: a.slot}}, false
	a.buildTrayMenu()
	a.overlaySync()
}

// untitleSessions drops the automatic title of each session whose card
// is no longer in the history, after an entry was deleted: the title
// comes from that entry's instruction and is saved. The card on screen
// stays, as any open card does; a name the user gave stays too.
// shortcut: the automatic clean-up does not call this, the next start drops those titles (restoreSessions).
func (a *app) untitleSessions() {
	if !a.sessionsKept() {
		return
	}
	conversations := map[int]string{}
	for _, p := range a.sessions() {
		conversation := p.answer.conv
		if p.id == a.slot && a.answer.open {
			conversation = a.answer.conv
		}
		if p.title != "" && conversation != "" {
			conversations[p.id] = conversation
		}
	}
	history := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var gone []int
		for id, conversation := range conversations {
			if entries, err := history.List(ctx, store.HistoryQuery{Conversation: conversation}); err == nil && len(entries) == 0 {
				gone = append(gone, id)
			}
		}
		a.post(func() {
			for _, id := range gone {
				if n := a.slotIndex(id); n >= 0 {
					a.slots[n].title, a.slots[n].titled = "", ""
				}
			}
			if len(gone) > 0 {
				a.buildTrayMenu()
				a.overlaySync()
			}
		})
	})
}
