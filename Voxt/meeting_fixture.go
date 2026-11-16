package main

type meetingFixtureState struct {
	Name            string                    `json:"name"`
	Kind            string                    `json:"kind"`
	Overlay         *meetingOverlayViewState  `json:"overlay,omitempty"`
	Page            *meetingsViewState        `json:"page,omitempty"`
	Start           *meetingStartViewState    `json:"start,omitempty"`
	History         *historyViewState         `json:"history,omitempty"`
	HistorySettings *historySettingsViewState `json:"history_settings,omitempty"`
	HistoryAccess   *historyAccessViewState   `json:"history_access,omitempty"`
	Remote          *remoteViewState          `json:"remote,omitempty"`
	RemoteCards     *remoteCardsViewState     `json:"remote_cards,omitempty"`
	RemoteDialog    *remoteDialogViewState    `json:"remote_dialog,omitempty"`
	RemoteDebug     *llmDebugViewState        `json:"remote_debug,omitempty"`
}
