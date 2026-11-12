package main

// The answer card's thinking menu (#3879): how much the language model
// thinks for this one input. Its button lists the levels the card's
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

// cardThinkLevels are the levels of the service that wrote the card's
// answer (none: it is not set up any more, or takes none).
func (a *app) cardThinkLevels() []*llm.Thinking {
	for _, p := range a.prefs.Providers.LLM {
		if p.ID == a.answer.llmID {
			return thinkLevels(p)
		}
	}
	return nil
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

// thinkText is level t as the button's tip says it.
func thinkText(t *llm.Thinking) string {
	if t == nil {
		return i18n.T("Thinking")
	}
	return i18n.T("Thinking: %s", thinkLevelTitle(t))
}

// sameThinking says whether a and b are the same level.
func sameThinking(a, b *llm.Thinking) bool {
	return a == b || a != nil && b != nil && *a == *b
}

// toggleThinkMenu opens or closes the thinking menu (its button); it
// takes the language menu's place.
func (a *app) toggleThinkMenu() {
	a.answer.langOpen = !(a.answer.langOpen && a.answer.thinkMenu)
	a.answer.thinkMenu = a.answer.langOpen
	a.syncLangMenu()
}

// thinkMenuRows are the menu's rows, for its height.
func (a *app) thinkMenuRows() int { return len(a.cardThinkLevels()) }

// thinkMenuView is the thinking menu's rows.
func (a *app) thinkMenuView(c *ui.Context) {
	cur := a.cardNow().think
	for i, t := range a.cardThinkLevels() {
		if menuRow(c.Key(i), thinkLevelTitle(t), sameThinking(t, cur)) {
			a.chooseThinking(t)
		}
	}
}

// chooseThinking is a level chosen in the menu: it closes, and the card's
// input is answered again with it (the same level: nothing).
func (a *app) chooseThinking(t *llm.Thinking) {
	a.closeLangMenu()
	card := a.answer
	if !a.cardNow().canThink || sameThinking(t, card.think) {
		return
	}
	in := card.input
	a.redoAnswer(session.StartRequest{Kind: card.kind, Target: card.target, Text: &in, TargetLanguage: card.lang, Thinking: t})
}
