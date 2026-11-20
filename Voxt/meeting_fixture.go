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
	Models                *modelsViewState            `json:"models,omitempty"`
	Feature               *featureViewState           `json:"feature,omitempty"`
	Settings              *settingsDialogViewState    `json:"settings,omitempty"`
	MeetingSettings       *settingsMeetingViewState   `json:"meeting_settings,omitempty"`
	ShortcutPrompt        *shortcutPromptViewState    `json:"shortcut_prompt,omitempty"`
	KeyRecorder           *keyRecorderViewState       `json:"key_recorder,omitempty"`
	Shell                 *shellViewState             `json:"shell,omitempty"`
	ShellUI               *shellFixtureUI             `json:"shell_ui,omitempty"`
	Nav                   *navViewState               `json:"shell_nav,omitempty"`
	Notices               []noticeViewState           `json:"shell_notices,omitempty"`
	Logs                  *logsViewState              `json:"shell_logs,omitempty"`
	Palette               *paletteViewState           `json:"shell_palette,omitempty"`
	Sheet                 *sheetViewState             `json:"shell_sheet,omitempty"`
	Announcement          *announcementViewState      `json:"shell_announcement,omitempty"`
}

// shellFixtureUI seeds the shell views' local state in the demo: which
// of the shell's overlays is open.
type shellFixtureUI struct {
	PaletteOpen      bool   `json:"palette_open,omitempty"`
	PaletteQuery     string `json:"palette_query,omitempty"`
	PaletteSel       int    `json:"palette_sel,omitempty"`
	SheetOpen        bool   `json:"sheet_open,omitempty"`
	AnnouncementOpen bool   `json:"announcement_open,omitempty"`
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
		card.Inputs = slices.Clone(state.Card.Inputs)
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
