package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

var _ sessionOverlayActions = (*meetingDemoHost)(nil)

func (host *meetingDemoHost) resetSessionFixture(fixture meetingFixtureState) {
	host.sessionState = sessionOverlayViewState{}
	host.sessionSeed = sessionOverlayFixtureUI{}
	host.sessionUI = sessionOverlayLocalUI{}
	host.sessionCardUI = answerCardUI{}
	host.sessionLayout = sessionOverlayLayout{}
	host.sessionMemory = sessionOverlayLayoutMemory{}
	host.sessionMeasurements = map[ovSized]ovMeasured{}
	if fixture.Session == nil || fixture.SessionUI == nil {
		return
	}
	host.sessionState = cloneSessionOverlayViewState(*fixture.Session)
	if host.sessionState.Card != nil {
		host.sessionUI.cardPreviews = answerInputFixturePreviews(host.sessionState.Card.Inputs)
	}
	seed := *fixture.SessionUI
	seed.OpenQuotes = slices.Clone(seed.OpenQuotes)
	seed.LanguageMenu = cloneSessionMenuViewState(seed.LanguageMenu)
	seed.CardThinkingMenu = cloneSessionMenuViewState(seed.CardThinkingMenu)
	seed.PillThinkingMenu = cloneSessionMenuViewState(seed.PillThinkingMenu)
	host.sessionSeed = seed
	if len(seed.OpenQuotes) > 0 {
		host.sessionCardUI.quotesOpen = map[int]bool{}
		for _, turn := range seed.OpenQuotes {
			host.sessionCardUI.quotesOpen[turn] = true
		}
	}
	host.sessionCardUI.langOpen = seed.Menu != ""
	host.sessionCardUI.thinkMenu = seed.Menu == sessionMenuCardThinking
	host.sessionCardUI.pillMenu = seed.Menu == sessionMenuPillThinking
}

func (host *meetingDemoHost) sessionLocal() sessionOverlayLocal {
	return sessionOverlayLocal{Overlay: &host.sessionUI, Card: &host.sessionCardUI}
}

func (host *meetingDemoHost) rebuildSessionOverlayLayout() sessionOverlayLayout {
	overlay := host.sessionSeed.Overlay
	host.sessionLayout = sessionOverlayLayoutFor(&host.sessionState, sessionOverlayLayoutInput{
		Measure:  host.measureSessionOverlayText,
		WorkArea: ovTestArea,
		Overlay:  overlay,
		Memory:   host.sessionMemory,
		Measured: host.sessionMeasurements,
	})
	host.sessionMemory = sessionOverlayLayoutMemory{
		LiveID: host.sessionLayout.LiveID, LiveWidth: host.sessionLayout.LiveWidth, StreamHeight: host.sessionLayout.StreamHeight,
	}
	return host.sessionLayout
}

func (host *meetingDemoHost) measureSessionOverlayText(key ovSized) (ovMeasured, bool) {
	if measured, ok := host.sessionMeasurements[key]; ok {
		return measured, true
	}
	width, height := guessText(key)
	return ovMeasured{width, height}, false
}

func (host *meetingDemoHost) applySessionOverlayMeasurements(measurements []sessionOverlayMeasurement) {
	if len(measurements) == 0 {
		return
	}
	if len(host.sessionMeasurements) > 256 {
		host.sessionMeasurements = map[ovSized]ovMeasured{}
	}
	for _, measurement := range measurements {
		host.sessionMeasurements[measurement.Key] = ovMeasured{measurement.Width, measurement.Height}
	}
	host.rebuildSessionOverlayLayout()
}

func (host *meetingDemoHost) sessionMenuKind() string {
	switch {
	case !host.sessionCardUI.langOpen:
		return ""
	case host.sessionCardUI.pillMenu:
		return sessionMenuPillThinking
	case host.sessionCardUI.thinkMenu:
		return sessionMenuCardThinking
	default:
		return sessionMenuLanguage
	}
}

func (host *meetingDemoHost) setSessionMenu(kind string) {
	host.sessionCardUI.langOpen = kind != ""
	host.sessionCardUI.thinkMenu = kind == sessionMenuCardThinking
	host.sessionCardUI.pillMenu = kind == sessionMenuPillThinking
	host.sessionState.Menu = nil
	switch kind {
	case sessionMenuLanguage:
		host.sessionState.Menu = cloneSessionMenuViewState(host.sessionSeed.LanguageMenu)
	case sessionMenuCardThinking:
		host.sessionState.Menu = cloneSessionMenuViewState(host.sessionSeed.CardThinkingMenu)
	case sessionMenuPillThinking:
		host.sessionState.Menu = cloneSessionMenuViewState(host.sessionSeed.PillThinkingMenu)
	}
}

func (host *meetingDemoHost) closeSessionMenu() {
	host.setSessionMenu("")
}

func selectSessionMenuOption(menu *sessionMenuViewState, match func(sessionMenuOption) bool) {
	if menu == nil {
		return
	}
	menu.SelectedIndex = -1
	for index := range menu.Options {
		selected := match(menu.Options[index])
		menu.Options[index].Selected = selected
		if selected {
			menu.SelectedIndex = index
		}
	}
}

func (host *meetingDemoHost) skipLLM() { host.record("skipLLM") }

func (host *meetingDemoHost) stopSession() {
	host.closeSessionMenu()
	host.record("stopSession")
}

func (host *meetingDemoHost) stopLLM() {
	host.closeSessionMenu()
	host.record("stopLLM")
}

func (host *meetingDemoHost) followUpAnswer(hold string) bool {
	host.closeSessionMenu()
	if host.sessionState.Card == nil || !host.sessionState.Card.CanFollowUp {
		return false
	}
	host.record("followUpAnswer %q", hold)
	return true
}

func (host *meetingDemoHost) retryAnswer(language string) {
	host.closeSessionMenu()
	host.record("retryAnswer %q", language)
}

func (host *meetingDemoHost) insertAnswer() {
	host.closeSessionMenu()
	host.record("insertAnswer")
}

func (host *meetingDemoHost) copyAnswer() {
	host.closeSessionMenu()
	if host.sessionState.Card != nil {
		host.sessionState.Card.Copied = true
	}
	host.record("copyAnswer")
}

func (host *meetingDemoHost) closeAnswer() {
	host.closeSessionMenu()
	host.sessionState.Mode = ovHidden
	host.sessionState.Card = nil
	host.record("closeAnswer")
}

func (host *meetingDemoHost) noticeAction(action string) {
	host.closeSessionMenu()
	host.sessionState.Mode = ovHidden
	host.sessionState.Notice = nil
	host.record("noticeAction %s", action)
}

func (host *meetingDemoHost) toggleLangMenu() {
	kind := sessionMenuLanguage
	if host.sessionMenuKind() == kind {
		kind = ""
	}
	host.setSessionMenu(kind)
	host.record("toggleLangMenu")
}

func (host *meetingDemoHost) chooseLanguage(code string) {
	matches := func(option sessionMenuOption) bool { return option.Language == code }
	selectSessionMenuOption(host.sessionSeed.LanguageMenu, matches)
	selectSessionMenuOption(host.sessionState.Menu, matches)
	if card := host.sessionState.Card; card != nil {
		card.Language = code
		if host.sessionState.Menu != nil {
			for _, option := range host.sessionState.Menu.Options {
				if option.Language == code {
					card.LanguageLabel = option.Label
				}
			}
		}
	}
	host.closeSessionMenu()
	host.record("chooseLanguage %s", code)
}

func (host *meetingDemoHost) toggleThinkMenu() {
	kind := sessionMenuCardThinking
	if host.sessionMenuKind() == kind {
		kind = ""
	}
	host.setSessionMenu(kind)
	host.record("toggleThinkMenu")
}

func (host *meetingDemoHost) togglePillThinkMenu() {
	kind := sessionMenuPillThinking
	if host.sessionMenuKind() == kind {
		kind = ""
	}
	host.setSessionMenu(kind)
	host.record("togglePillThinkMenu")
}

func (host *meetingDemoHost) chooseThinking(value sessionThinkingValue) {
	kind := host.sessionMenuKind()
	matches := func(option sessionMenuOption) bool { return option.Thinking == value }
	selectSessionMenuOption(host.sessionState.Menu, matches)
	if kind == sessionMenuPillThinking {
		selectSessionMenuOption(host.sessionSeed.PillThinkingMenu, matches)
	} else {
		selectSessionMenuOption(host.sessionSeed.CardThinkingMenu, matches)
	}
	if host.sessionState.Menu != nil {
		for _, option := range host.sessionState.Menu.Options {
			if option.Thinking != value {
				continue
			}
			if kind == sessionMenuPillThinking {
				host.sessionState.Pill.ThinkingLabel = option.Label
			} else if host.sessionState.Card != nil {
				host.sessionState.Card.Thinking = value
				host.sessionState.Card.ThinkingLabel = option.Label
			}
		}
	}
	host.closeSessionMenu()
	host.record("chooseThinking %q %q %d", value.Mode, value.Effort, value.Budget)
}

func (host *meetingDemoHost) cycleOverlayCardSize() {
	host.closeSessionMenu()
	card := host.sessionState.Card
	if card == nil {
		return
	}
	card.SizeStep = (card.SizeStep + 1) % len(layout.CardSizes)
	host.record("cycleOverlayCardSize")
}

func (host *meetingDemoHost) toggleQuote(turn int) {
	if host.sessionCardUI.quotesOpen == nil {
		host.sessionCardUI.quotesOpen = map[int]bool{}
	}
	host.sessionCardUI.quotesOpen[turn] = !host.sessionCardUI.quotesOpen[turn]
	open := host.sessionCardUI.quotesOpen[turn]
	if card := host.sessionState.Card; card != nil {
		for index := range card.Blocks {
			updateSessionDemoQuote(&card.Blocks[index], turn, open)
		}
		for index := 0; index < card.LiveCount; index++ {
			updateSessionDemoQuote(&card.Live[index], turn, open)
		}
	}
	host.record("toggleQuote %d", turn)
}

func updateSessionDemoQuote(block *sessionCardBlock, turn int, open bool) {
	if block.Turn != turn {
		return
	}
	block.Open = open
	block.Quote, block.Cut = quoteText(block.FullQuote, open)
}

func (host *meetingDemoHost) copyBubble(block sessionCardBlock) {
	text := block.Text
	if block.Title != "" {
		text = block.Title + "\n\n" + text
	}
	host.sessionUI.bubbleCopied = block.Text
	host.sessionUI.bubbleSeq++
	host.record("copyBubble %q", text)
}

// The demo has no WebView. Record formatted requests and draw their native fallback.
func (host *meetingDemoHost) overlayTranscriptResult(_ *uiContext, request sessionOverlayPreviewRequest) bool {
	if transcriptMarkdown(request.text.text) != "" {
		host.record("overlayTranscriptResult %q", request.text.text)
	}
	return false
}
