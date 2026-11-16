package main

import "slices"

type meetingFixtureState struct {
	Name                  string                      `json:"name"`
	Kind                  string                      `json:"kind"`
	Overlay               *meetingOverlayViewState    `json:"overlay,omitempty"`
	Page                  *meetingsViewState          `json:"page,omitempty"`
	Start                 *meetingStartViewState      `json:"start,omitempty"`
	History               *historyViewState           `json:"history,omitempty"`
	HistorySettings       *historySettingsViewState   `json:"history_settings,omitempty"`
	HistoryAccess         *historyAccessViewState     `json:"history_access,omitempty"`
	Session               *sessionOverlayViewState    `json:"session,omitempty"`
	SessionUI             *sessionOverlayFixtureUI    `json:"session_ui,omitempty"`
	UpdatesSettings       *updatesViewState           `json:"updates_settings,omitempty"`
	UpdatesRelease        *releaseViewState           `json:"updates_release,omitempty"`
	Home                  *homeViewState              `json:"home,omitempty"`
	Onboarding            *onboardingViewState        `json:"onboarding,omitempty"`
	AppEnhancement        *appEnhancementViewState    `json:"appenhance,omitempty"`
	AppEnhancementDetails []appEnhancementGroupDetail `json:"appenhance_details,omitempty"`
	Dictionary            *dictionaryViewState        `json:"dictionary,omitempty"`
	Notes                 *notesViewState             `json:"notes,omitempty"`
	Remote                *remoteViewState            `json:"remote,omitempty"`
	RemoteCards           *remoteCardsViewState       `json:"remote_cards,omitempty"`
	RemoteDialog          *remoteDialogViewState      `json:"remote_dialog,omitempty"`
	RemoteDebug           *llmDebugViewState          `json:"remote_debug,omitempty"`
}

type sessionOverlayFixtureUI struct {
	Menu             string                    `json:"menu,omitempty"`
	OpenQuotes       []int                     `json:"open_quotes,omitempty"`
	Overlay          sessionOverlayLayoutStyle `json:"overlay"`
	LanguageMenu     *sessionMenuViewState     `json:"language_menu,omitempty"`
	CardThinkingMenu *sessionMenuViewState     `json:"card_thinking_menu,omitempty"`
	PillThinkingMenu *sessionMenuViewState     `json:"pill_thinking_menu,omitempty"`
}

func cloneSessionOverlayViewState(state sessionOverlayViewState) sessionOverlayViewState {
	clone := state
	if state.Notice != nil {
		notice := *state.Notice
		clone.Notice = &notice
	}
	if state.Card != nil {
		card := *state.Card
		card.Blocks = slices.Clone(state.Card.Blocks)
		clone.Card = &card
	}
	clone.Menu = cloneSessionMenuViewState(state.Menu)
	return clone
}

func cloneSessionMenuViewState(state *sessionMenuViewState) *sessionMenuViewState {
	if state == nil {
		return nil
	}
	clone := *state
	clone.Options = slices.Clone(state.Options)
	return &clone

}
