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
	AppEnhancement  *appEnhancementViewState  `json:"appenhance,omitempty"`
}
