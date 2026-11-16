package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	sessionMenuLanguage     = "language"
	sessionMenuCardThinking = "card-thinking"
	sessionMenuPillThinking = "pill-thinking"
)

// sessionThinkingValue is the JSON-safe form of one optional thinking level.
// Its zero value selects the service's default.
type sessionThinkingValue struct {
	Mode   string `json:"mode,omitempty"`
	Effort string `json:"effort,omitempty"`
	Budget int    `json:"budget,omitempty"`
}

type sessionMenuOption struct {
	Language string               `json:"language,omitempty"`
	Thinking sessionThinkingValue `json:"thinking"`
	Label    string               `json:"label"`
	Selected bool                 `json:"selected"`
}

type sessionMenuViewState struct {
	Kind          string              `json:"kind"`
	Label         string              `json:"label"`
	Options       []sessionMenuOption `json:"options"`
	SelectedIndex int                 `json:"selected_index"`
}

type sessionMenuProjectionInput struct {
	Kind            string
	CurrentLanguage string
	CurrentThinking *llm.Thinking
	ThinkingLevels  []*llm.Thinking
	Generation      settings.Generation
}

func buildSessionMenuViewState(translator *i18n.Translator, input sessionMenuProjectionInput) sessionMenuViewState {
	state := sessionMenuViewState{Kind: input.Kind, SelectedIndex: -1}
	if input.Kind == sessionMenuLanguage {
		state.Label = translator.T("Translate into")
		state.Options = make([]sessionMenuOption, 0, len(compose.TargetLanguages))
		for index, code := range compose.TargetLanguages {
			selected := code == input.CurrentLanguage
			if selected {
				state.SelectedIndex = index
			}
			state.Options = append(state.Options, sessionMenuOption{Language: code,
				Label: compose.LanguageName(code, translator.Locale()), Selected: selected})
		}
		return state
	}
	state.Label = translator.T("Ask again with thinking level")
	state.Options = make([]sessionMenuOption, 0, len(input.ThinkingLevels))
	for index, level := range input.ThinkingLevels {
		selected := sameThinking(level, input.CurrentThinking)
		if selected {
			state.SelectedIndex = index
		}
		state.Options = append(state.Options, sessionMenuOption{Thinking: sessionThinkingValueOf(level),
			Label: thinkLevelTitleFor(translator, level, input.Generation), Selected: selected})
	}
	return state
}

func sessionThinkingValueOf(level *llm.Thinking) sessionThinkingValue {
	if level == nil {
		return sessionThinkingValue{}
	}
	return sessionThinkingValue{Mode: level.Mode, Effort: level.Effort, Budget: level.Budget}
}

func (value sessionThinkingValue) thinking() *llm.Thinking {
	if value == (sessionThinkingValue{}) {
		return nil
	}
	return &llm.Thinking{Mode: value.Mode, Effort: value.Effort, Budget: value.Budget}
}

// thinkBudgets are the levels of a service that takes a budget, in tokens.
var thinkBudgets = []int{1024, 4096, 16384}

func thinkLevels(provider settings.Provider) []*llm.Thinking {
	capabilities := providers.Capabilities(provider.ID, provider.Model)
	if len(capabilities.ThinkModes) == 0 {
		return nil
	}
	levels := []*llm.Thinking{nil}
	for _, mode := range capabilities.ThinkModes {
		switch mode {
		case llm.ThinkOff, llm.ThinkOn:
			levels = append(levels, &llm.Thinking{Mode: mode})
		case llm.ThinkEffort:
			for _, effort := range capabilities.Efforts {
				levels = append(levels, &llm.Thinking{Mode: mode, Effort: effort})
			}
		case llm.ThinkBudget:
			for _, budget := range thinkBudgets {
				levels = append(levels, &llm.Thinking{Mode: mode, Budget: budget})
			}
		}
	}
	return levels
}

func (a *app) llmThinkLevels(id string) []*llm.Thinking {
	if provider, ok := a.llmService(id); ok {
		return thinkLevels(provider)
	}
	return nil
}

func (a *app) llmService(id string) (settings.Provider, bool) {
	for _, provider := range a.prefs.Providers.LLM {
		if provider.ID == id {
			return provider, true
		}
	}
	return settings.Provider{}, false
}

func (a *app) llmGeneration(id string) settings.Generation {
	provider, _ := a.llmService(id)
	return provider.Generation
}

func (a *app) pillThinks(mode overlayMode) bool {
	snapshot := a.sess
	return mode == ovWorking && snapshot.State == session.Composing &&
		slices.Contains([]store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite}, snapshot.Kind) &&
		len(a.llmThinkLevels(snapshot.LLM.ID)) > 1
}

func (a *app) pillThinkText() ovSized {
	return ovSized{askAgainText(a.sess.Thinking, a.llmGeneration(a.sess.LLM.ID)), 11, 600, 0}
}

func settingLevel(generation settings.Generation) *llm.Thinking {
	switch {
	case generation.Thinking == llm.ThinkOff, generation.Thinking == llm.ThinkOn:
		return &llm.Thinking{Mode: generation.Thinking}
	case generation.Thinking == llm.ThinkEffort && generation.ThinkingEffort != "":
		return &llm.Thinking{Mode: generation.Thinking, Effort: generation.ThinkingEffort}
	case generation.Thinking == llm.ThinkBudget && generation.ThinkingBudget > 0:
		return &llm.Thinking{Mode: generation.Thinking, Budget: generation.ThinkingBudget}
	}
	return nil
}

func thinkLevelTitle(level *llm.Thinking, generation settings.Generation) string {
	return thinkLevelTitleFor(i18n.CurrentTranslator(), level, generation)
}

func thinkLevelTitleFor(translator *i18n.Translator, level *llm.Thinking, generation settings.Generation) string {
	switch {
	case level == nil:
		if defaultLevel := settingLevel(generation); defaultLevel != nil {
			return translator.T("%s (default)", thinkLevelTitleFor(translator, defaultLevel, generation))
		}
		return translator.T("Default")
	case level.Mode == llm.ThinkOff:
		return translator.T("Off")
	case level.Mode == llm.ThinkOn:
		return translator.T("On")
	case level.Mode == llm.ThinkEffort:
		return sessionEffortTitleFor(translator, level.Effort)
	}
	return translator.T("%d tokens", level.Budget)
}

func sessionEffortTitleFor(translator *i18n.Translator, effort string) string {
	switch effort {
	case "none":
		return translator.T("None")
	case "minimal":
		return translator.T("Minimal")
	case "low":
		return translator.T("Low")
	case "medium":
		return translator.T("Medium")
	case "high":
		return translator.T("High")
	case "xhigh":
		return translator.T("Extra high")
	case "max":
		return translator.T("Max")
	}
	return effort
}

func thinkText(level *llm.Thinking, generation settings.Generation) string {
	return i18n.T("Thinking: %s", thinkLevelTitle(level, generation))
}

func askAgainText(level *llm.Thinking, generation settings.Generation) string {
	return askAgainTextFor(i18n.CurrentTranslator(), level, generation)
}

func askAgainTextFor(translator *i18n.Translator, level *llm.Thinking, generation settings.Generation) string {
	if defaultLevel := settingLevel(generation); level == nil && defaultLevel != nil {
		return translator.T("Ask again: Thinking (%s, default)", thinkLevelTitleFor(translator, defaultLevel, generation))
	}
	return translator.T("Ask again: Thinking (%s)", thinkLevelTitleFor(translator, level, generation))
}

func sameThinking(first, second *llm.Thinking) bool {
	return first == second || first != nil && second != nil && *first == *second
}

func (a *app) toggleThinkMenu() {
	a.answer.ui.langOpen = !(a.answer.ui.langOpen && a.answer.ui.thinkMenu)
	a.answer.ui.thinkMenu, a.answer.ui.pillMenu = a.answer.ui.langOpen, false
	a.syncLangMenu()
}

func (a *app) togglePillThinkMenu() {
	open := !(a.answer.ui.langOpen && a.answer.ui.pillMenu)
	a.answer.ui.langOpen, a.answer.ui.thinkMenu, a.answer.ui.pillMenu = open, open, open
	a.syncLangMenu()
}

func (a *app) toggleLangMenu() {
	a.answer.ui.langOpen = !a.answer.ui.langOpen || a.answer.ui.thinkMenu
	a.answer.ui.thinkMenu, a.answer.ui.pillMenu = false, false
	a.syncLangMenu()
}

func (a *app) closeLangMenu() {
	if a.answer.ui.langOpen {
		a.answer.ui.langOpen, a.answer.ui.thinkMenu, a.answer.ui.pillMenu = false, false, false
		a.syncLangMenu()
	}
}

func (a *app) chooseLanguage(code string) {
	a.closeLangMenu()
	current := a.answer.lang
	if a.answer.redo != "" {
		current = a.answer.redoLang
	}
	if code != current {
		a.retryAnswer(code)
	}
}

func (a *app) chooseThinking(value sessionThinkingValue) {
	level := value.thinking()
	pill := a.answer.ui.pillMenu
	a.closeLangMenu()
	if pill {
		if a.mgr != nil && !sameThinking(level, a.sess.Thinking) {
			a.mgr.Rethink(a.sess.ID, level)
		}
		return
	}
	card := a.answer
	if a.ov.projection.view.Card == nil || !a.ov.projection.view.Card.CanThink || sameThinking(level, card.think) {
		return
	}
	input := card.input
	a.redoAnswer(session.StartRequest{Kind: card.kind, Target: card.target, Text: &input, TargetLanguage: card.lang, Thinking: level})
}

type sessionMenuProjectionKey struct {
	kind, locale, language, serviceID, model string
	thinking                                 sessionThinkingValue
	generationThinking, generationEffort     string
	generationBudget                         int
}

func (a *app) sessionMenuProjectionInput() (sessionMenuProjectionInput, settings.Provider) {
	input := sessionMenuProjectionInput{}
	provider := settings.Provider{}
	if !a.answer.ui.langOpen {
		return input, provider
	}
	switch {
	case a.answer.ui.pillMenu:
		input.Kind, input.CurrentThinking = sessionMenuPillThinking, a.sess.Thinking
		provider, _ = a.llmService(a.sess.LLM.ID)
	case a.answer.ui.thinkMenu:
		input.Kind, input.CurrentThinking = sessionMenuCardThinking, a.answer.think
		if a.answer.redo != "" {
			input.CurrentThinking = a.answer.redoThink
		}
		provider, _ = a.llmService(a.answer.llmID)
	default:
		input.Kind, input.CurrentLanguage = sessionMenuLanguage, a.answer.lang
		if a.answer.redo != "" {
			input.CurrentLanguage = a.answer.redoLang
		}
	}
	input.Generation = provider.Generation
	return input, provider
}

func sessionMenuKeyFor(input sessionMenuProjectionInput, provider settings.Provider, locale string) sessionMenuProjectionKey {
	return sessionMenuProjectionKey{kind: input.Kind, locale: locale, language: input.CurrentLanguage,
		serviceID: provider.ID, model: provider.Model, thinking: sessionThinkingValueOf(input.CurrentThinking),
		generationThinking: input.Generation.Thinking, generationEffort: input.Generation.ThinkingEffort,
		generationBudget: input.Generation.ThinkingBudget}
}

func (a *app) refreshSessionMenuProjection(projection *sessionOverlayProjection, next *sessionOverlayViewState, locale string) {
	if !a.answer.ui.langOpen {
		return
	}
	input, provider := a.sessionMenuProjectionInput()
	key := sessionMenuKeyFor(input, provider, locale)
	if !projection.menuReady || key != projection.menuKey {
		if input.Kind != sessionMenuLanguage {
			input.ThinkingLevels = thinkLevels(provider)
		}
		projection.menu = buildSessionMenuViewState(projection.translator, input)
		projection.menuKey, projection.menuReady = key, true
	}
	next.Menu = &projection.menu
}
