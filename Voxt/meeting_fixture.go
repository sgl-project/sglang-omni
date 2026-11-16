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
	Settings        *settingsDialogViewState  `json:"settings,omitempty"`
	MeetingSettings *settingsMeetingViewState `json:"meeting_settings,omitempty"`
	ShortcutPrompt  *shortcutPromptViewState  `json:"shortcut_prompt,omitempty"`
	KeyRecorder     *keyRecorderViewState     `json:"key_recorder,omitempty"`
}
