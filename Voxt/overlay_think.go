package main

// The answer card's thinking menu (#3879), and the working pill's while
// Transcribe's clean-up runs (#3940): how much the language model thinks
// for this one input. Its button lists the levels the card's
// service takes (its Thinking setting's, providers.Capabilities); one
// answers the card's input again with it, into the card, as the language
// menu translates it again. The service's setting stays the default. The
// list shows in the language menu's window (overlay_lang.go).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// thinkBudgets are the levels of a service that takes a budget, in
// tokens (Anthropic's least is 1024).
var thinkBudgets = []int{1024, 4096, 16384}

// thinkLevels are the thinking levels p takes, the service's setting
// (nil) first; none when it takes no thinking setting.
func thinkLevels(p settings.Provider) []*llm.Thinking {
	caps := providers.Capabilities(p.ID, p.Model)
	if len(caps.ThinkModes) == 0 {
		return nil
	}
	out := []*llm.Thinking{nil}
	for _, m := range caps.ThinkModes {
		switch m {
		case llm.ThinkOff, llm.ThinkOn:
			out = append(out, &llm.Thinking{Mode: m})
		case llm.ThinkEffort:
			for _, e := range caps.Efforts {
				out = append(out, &llm.Thinking{Mode: m, Effort: e})
			}
		case llm.ThinkBudget:
			for _, n := range thinkBudgets {
				out = append(out, &llm.Thinking{Mode: m, Budget: n})
			}
		}
	}
	return out
}

// llmThinkLevels are the levels of language model service id (none: it
// is not set up any more, or takes none).
func (a *app) llmThinkLevels(id string) []*llm.Thinking {
	for _, p := range a.prefs.Providers.LLM {
		if p.ID == id {
			return thinkLevels(p)
		}
	}
	return nil
}

// cardThinkLevels are the levels of the service that wrote the card's
// answer.
func (a *app) cardThinkLevels() []*llm.Thinking { return a.llmThinkLevels(a.answer.llmID) }

// pillThinks: the working pill in mode offers the thinking levels
// (#3940): Transcribe's clean-up runs (Use transcript shows) on a
// service that takes them.
func (a *app) pillThinks(mode overlayMode) bool {
	return canSkip(a.sess, mode) && a.sess.Kind == store.KindTranscribe && len(a.llmThinkLevels(a.sess.LLM.ID)) > 1
}

// pillThinkText is the pill's thinking capsule as it draws (cardCapsule's).
func pillThinkText(t *llm.Thinking) ovSized { return ovSized{askAgainText(t), 11, 600, 0} }

// menuThinkLevels are the thinking menu's levels: the pill's or the card's.
func (a *app) menuThinkLevels() []*llm.Thinking {
	if a.answer.pillMenu {
		return a.llmThinkLevels(a.sess.LLM.ID)
	}
	return a.cardThinkLevels()
}

// thinkLevelTitle is a level's row in the menu.
func thinkLevelTitle(t *llm.Thinking) string {
	switch {
	case t == nil:
		return i18n.T("As in settings")
	case t.Mode == llm.ThinkOff:
		return i18n.T("Off")
	case t.Mode == llm.ThinkOn:
		return i18n.T("On")
	case t.Mode == llm.ThinkEffort:
		return effortTitle(t.Effort)
	}
	return i18n.T("%d tokens", t.Budget)
}

// thinkText is level t as a message says it.
func thinkText(t *llm.Thinking) string {
	if t == nil {
		return i18n.T("Thinking")
	}
	return i18n.T("Thinking: %s", thinkLevelTitle(t))
}

// askAgainText is level t as the card button's tip and the pill's
// capsule say it: picking a level stops this answer and asks again
// (#3951).
func askAgainText(t *llm.Thinking) string {
	if t == nil {
		return i18n.T("Ask again: Thinking…")
	}
	return i18n.T("Ask again: Thinking (%s)", thinkLevelTitle(t))
}

// sameThinking says whether a and b are the same level.
func sameThinking(a, b *llm.Thinking) bool {
	return a == b || a != nil && b != nil && *a == *b
}

// toggleThinkMenu opens or closes the thinking menu (its button); it
// takes the language menu's place.
func (a *app) toggleThinkMenu() {
	a.answer.langOpen = !(a.answer.langOpen && a.answer.thinkMenu)
	a.answer.thinkMenu, a.answer.pillMenu = a.answer.langOpen, false
	a.syncLangMenu()
}

// togglePillThinkMenu opens or closes the working pill's thinking menu
// (its capsule).
func (a *app) togglePillThinkMenu() {
	open := !(a.answer.langOpen && a.answer.pillMenu)
	a.answer.langOpen, a.answer.thinkMenu, a.answer.pillMenu = open, open, open
	a.syncLangMenu()
}

// thinkMenuView is the thinking menu's rows.
func (a *app) thinkMenuView(c *ui.Context) {
	cur := a.cardNow().think
	if a.answer.pillMenu {
		cur = a.sess.Thinking
	}
	for i, t := range a.menuThinkLevels() {
		if menuRow(c.Key(i), thinkLevelTitle(t), sameThinking(t, cur)) {
			a.chooseThinking(t)
		}
	}
}

// chooseThinking is a level chosen in the menu: it closes, and the card's
// input is answered again with it, or the pill's text cleaned up again
// (session.Manager's Rethink); the same level: nothing.
func (a *app) chooseThinking(t *llm.Thinking) {
	pill := a.answer.pillMenu
	a.closeLangMenu()
	if pill {
		if a.mgr != nil && !sameThinking(t, a.sess.Thinking) {
			a.mgr.Rethink(a.sess.ID, t)
		}
		return
	}
	card := a.answer
	if !a.cardNow().canThink || sameThinking(t, card.think) {
		return
	}
	in := card.input
	a.redoAnswer(session.StartRequest{Kind: card.kind, Target: card.target, Text: &in, TargetLanguage: card.lang, Thinking: t})
}
