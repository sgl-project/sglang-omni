package main

// The sessions of the multi-session prototype (#4005): a list the user
// adds to and deletes from on the card's session row (overlay_session_row.go),
// or walks with the next-session shortcut. Rewrite, Translate and
// Transcribe run in the current session. Each session keeps its own
// answer card (and the card waiting under it, #4013) with its
// conversation; switching back restores them, so a covered Rewrite
// result is never lost. An answer still streaming when the user switches
// finishes in the background in its own session (wall 116): its card
// waits there until the session comes back. With Settings → Experimental's
// Sessions switch off there is one session and everything behaves as
// before.

import (
	"slices"
	"strconv"
	"strings"
	"unicode"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// parkedSlot is a session: its id, which never changes (its number on
// the card is its place in the list), its title, and its cards while
// another session is current.
type parkedSlot struct {
	id    int
	title string
	// name is the title the user gave it: it shows instead of title, and
	// nothing replaces it.
	name string
	// titled is the instruction a language model is asked a title for
	// ("" once it answered): only that title's answer may replace it.
	titled        string
	answer, under answerCard
	followUp      string
}

// sessions is the session list; there is always one.
func (a *app) sessions() []parkedSlot {
	if len(a.slots) == 0 {
		a.slots = []parkedSlot{{id: a.slot}}
	}
	return a.slots
}

// slotIndex is the place of session id in the list, -1 once deleted.
func (a *app) slotIndex(id int) int {
	return slices.IndexFunc(a.sessions(), func(p parkedSlot) bool { return p.id == id })
}

// switchSlot parks the current session's cards and restores session
// id's. While a session records or works it does nothing; a streaming
// answer finishes in the background in its own session. An open Rewrite
// box belongs to the session being left: it closes, as it does for
// another feature's session, and its recording ends with it; its
// answered card parks with the session. A session without a card shows
// the empty card, so that the session row stays in reach.
func (a *app) switchSlot(id int) {
	n := a.slotIndex(id)
	if id == a.slot || n < 0 || a.answer.inserting {
		return // an insert pastes: its callback must find its card in place
	}
	pin, pinned := a.cardWindowFrame()
	wasBox, boxTarget := a.box.open, a.box.target
	if a.box.open && a.sess.State.Live() && a.sess.ID == a.boxSessionID() && !a.answerBackgroundable() {
		a.closeBox() // which cancels its session
		a.sess = session.Snapshot{State: session.Idle, Rev: a.sess.Rev}
	}
	if a.sess.State.Live() {
		// Only a streaming answer can keep going meanwhile; anything
		// earlier needs the session's place (session_background.go).
		if !a.answerBackgroundable() || !a.backgroundAnswer() {
			return
		}
		// No new session starts here (startSessionID starts one right
		// after backgrounding, which covers this): the manager's
		// foreground snapshot stays the backgrounded session's until
		// then, so the app must not read it as live.
		a.sess = session.Snapshot{State: session.Idle, Rev: a.sess.Rev}
	}
	if a.box.open {
		a.closeBox()
	}
	if pinned {
		a.slotPin = &pin
	}
	from := &a.slots[a.slotIndex(a.slot)]
	if from.answer.open && a.answer.open && !a.under.open {
		a.under = from.answer // the card that waited here since the restart, never shown
	}
	if a.answer.open || !from.answer.open {
		from.answer, from.under, from.followUp = a.answer, a.under, a.followUp
	}
	to := &a.slots[n]
	a.slot = id
	a.answer, a.under, a.followUp = to.answer, to.under, to.followUp
	to.answer, to.under, to.followUp = answerCard{}, answerCard{}, ""
	a.notice, a.renaming = nil, false
	a.resurface()
	a.emptyCard = a.prefs.Features.SessionsOn && !a.answer.open && !a.bgVisible()
	a.boxForAnswer()
	if wasBox && !a.answer.open && !a.bgVisible() {
		a.boxIn(boxTarget) // a new session goes on in the Rewrite card
	}
	a.buildTrayMenu()
	a.syncHotkeys()
	a.overlaySync()
}

// switchSession is the user's own switch to session id (a session on
// the card, a session shortcut): it counts in Settings' use counts.
func (a *app) switchSession(id int) {
	from := a.slot
	a.switchSlot(id)
	if a.slot == from {
		return
	}
	back := a.answer.open
	a.updateSettings(func(s *settings.Settings) {
		s.Setup.SessionSwitchTotal++
		if back {
			s.Setup.SessionReturns++
		}
	})
}

// newSession adds an empty session at the end of the list and switches
// to it (the session row's "+").
func (a *app) newSession() {
	a.sessions()
	a.slotSeq++
	a.slots = append(a.slots, parkedSlot{id: a.slotSeq})
	if a.switchSession(a.slotSeq); a.slot != a.slotSeq {
		a.slots = a.slots[:len(a.slots)-1] // a recording or a paste holds the session
	}
}

// deleteSession removes session id with its cards (the session row's
// "×"). The current one leaves for its neighbour, the last one for a new
// empty session; an answer of its own still streaming is cancelled. It
// is not a switch of the user's: the use counts stay.
func (a *app) deleteSession(id int) {
	n := a.slotIndex(id)
	if n < 0 {
		return
	}
	if id == a.slot {
		last := len(a.slots) == 1
		if last {
			a.slotSeq++
			a.slots = append(a.slots, parkedSlot{id: a.slotSeq})
		}
		next := n + 1
		if next == len(a.slots) {
			next = n - 1
		}
		if a.switchSlot(a.slots[next].id); a.slot == id {
			if last {
				a.slots = a.slots[:1]
			}
			return // a recording or a paste holds the session
		}
	}
	if a.bgID != "" && a.bgSlot == id && a.mgr != nil {
		a.mgr.Cancel(a.bgID) // its end finds no session, and leaves no card (parkAnswer)
	}
	a.slots = slices.Delete(a.slots, n, n+1)
	a.renaming = false
	a.buildTrayMenu()
	a.overlaySync()
}

// sessionChips is the card's session row: nothing with sessions off,
// one chip per session otherwise, the only one too. The other sessions'
// titles are cut to sessionShortMax; the chip's tooltip has the whole one.
func (a *app) sessionChips() []sessionChipViewState {
	if !a.prefs.Features.SessionsOn {
		return nil
	}
	list := a.sessions() // one session too: the row's "+" adds the next
	chips := make([]sessionChipViewState, len(list))
	for i, p := range list {
		chip := sessionChipViewState{ID: p.id, Title: cmpOr(p.name, p.title), Current: p.id == a.slot, Renaming: a.renaming && p.id == a.renameID}
		if chip.Renaming {
			chip.Title = p.title // the field's placeholder: what an empty name gives
		}
		most := sessionShortMax
		if chip.Current {
			most = sessionTitleMax // a name the user typed can be longer
		}
		if short := cutTitle(chip.Title, most); !chip.Renaming && short != chip.Title {
			chip.Title, chip.Full = short, chip.Title
		}
		chips[i] = chip
	}
	return chips
}

// sessionNumber is the current session's number on the recording pill:
// 0 with sessions off or one session only.
func (a *app) sessionNumber() int {
	if !a.prefs.Features.SessionsOn || len(a.sessions()) < 2 {
		return 0
	}
	return a.slotIndex(a.slot) + 1
}

// sessionsMenu lists the sessions for the menu bar menu:
// "1 Title", the current one checked. Choosing one shows it.
func (a *app) sessionsMenu(act func(func()) func(*mygo.MenuItem, *mygo.Window)) []*mygo.MenuItem {
	if !a.prefs.Features.SessionsOn {
		return nil
	}
	items := []*mygo.MenuItem{{Label: i18n.T("Sessions"), Disabled: true}}
	for i, p := range a.sessions() {
		id := p.id
		label := strconv.Itoa(i+1) + " " + cmpOr(cmpOr(p.name, p.title), i18n.T("New session"))
		items = append(items, &mygo.MenuItem{Label: label, Type: mygo.MenuItemCheckbox, Checked: id == a.slot, Click: act(func() { a.showSession(id) })})
	}
	return append(items,
		&mygo.MenuItem{Label: i18n.T("New session"), Click: act(a.newSession)},
		&mygo.MenuItem{Label: i18n.T("Rename session"), Click: act(a.renameCurrentSession)})
}

// sessionTitleMax is how many characters of a title the current
// session's chip shows, sessionShortMax the other sessions' chips.
const sessionTitleMax, sessionShortMax = 18, 8

// sessionTitle is text as a chip's title: one line, cut short.
func sessionTitle(text string) string {
	return cutTitle(strings.Join(strings.Fields(text), " "), sessionTitleMax)
}

// cutTitle cuts text to most columns: a wide (CJK) character takes two.
func cutTitle(text string, most int) string {
	columns := 0
	for i, r := range text {
		if columns += runeColumns(r); columns > most {
			return strings.TrimSpace(text[:i]) + "…"
		}
	}
	return text
}

func runeColumns(r rune) int {
	if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) || r >= 0x3000 && r <= 0x303f || r >= 0xff00 && r <= 0xffef {
		return 2
	}
	return 1
}

// titleSession names the session an ended session ran in after its
// first instruction: what the user said or typed, else the text they
// selected. The head of it shows at once; a language model's summary
// replaces it when one answers (session_title.go).
func (a *app) titleSession(s session.Snapshot) {
	id := a.slot
	if s.ID == a.bgID {
		id = a.bgSlot
	}
	n := a.slotIndex(id)
	text := strings.TrimSpace(cmpOr(cmpOr(strings.TrimSpace(s.Raw), strings.TrimSpace(s.Typed)), s.Selection))
	if !a.prefs.Features.SessionsOn || n < 0 || a.slots[n].title != "" || text == "" {
		return
	}
	a.slots[n].title, a.slots[n].titled = sessionTitle(text), text
	a.askSessionTitle(id, text)
}

// parkAnswer gives an ended backgrounded answer its card in its own
// session while another one is current (#4005): the card, with its
// conversation, waits there until the session comes back. A redo of the
// session's card replaces its answer there; another open card of the
// session waits under the new one, as setAside would hold it. A session
// deleted meanwhile gets nothing.
func (a *app) parkAnswer(s session.Snapshot) {
	n := a.slotIndex(a.bgSlot)
	if n < 0 {
		return
	}
	p := &a.slots[n]
	answer, under, followUp := a.answer, a.under, a.followUp
	a.answer, a.under, a.followUp = p.answer, p.under, p.followUp
	switch {
	case a.answer.open && a.answer.redo == s.ID:
		a.endRedo(s)
	default:
		if a.answer.open && a.followUp == "" {
			a.setAside(false)
		}
		a.openAnswer(s)
	}
	p.answer, p.under, p.followUp = a.answer, a.under, a.followUp
	a.answer, a.under, a.followUp = answer, under, followUp
}

// boxForAnswer shows a restored Rewrite answer in the Rewrite card, the
// one card its conversation had (wall 122): the answer card would trade
// the input for Continue.
func (a *app) boxForAnswer() {
	if a.box.open || !a.answer.open || a.answer.kind != store.KindRewrite || a.sess.State.Live() {
		return
	}
	a.boxIn(a.answer.target)
}

// boxIn opens the Rewrite card for target with nothing running: an
// answer restored in it, or a new session's first instruction to come.
func (a *app) boxIn(target proto.App) {
	a.box = rewriteBox{open: true, shown: true, target: target, focusText: true, dragged: a.ov.cardDragged}
	a.boxShare.reset()
	a.boxSync()
}

// cardWindowFrame is the frame of the card window on screen, if any.
func (a *app) cardWindowFrame() (mygo.Rectangle, bool) {
	switch {
	case a.box.open && a.box.winShown:
		return a.box.frame, true
	case a.ov.shown && a.ov.mode == ovAnswer:
		return a.ov.frame, true
	}
	return mygo.Rectangle{}, false
}
