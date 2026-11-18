package main

// The window shell's view state: the frame every page shares — theme,
// the sidebar, the notifications, the logs panel, the palette, the "?"
// sheet and the toasts — with the areas' dialogs beside it (the wiring
// renders those, shell.go's shellView takes them as a closure).

import "time"

// shellViewState is the shell's display data for one frame.
type shellViewState struct {
	// Opening is the first-use opening (#4173), over everything while it
	// plays; nil once it has.
	Opening *openingViewState `json:"opening,omitempty"`

	Theme string  `json:"theme"`
	Scale float64 `json:"scale"`
	// AccentPalette is the chosen colour ("#0d9384"); Transparent clears
	// the window's root fill (the transparent theme).
	AccentPalette string   `json:"accent_palette,omitempty"`
	Transparent   bool     `json:"transparent,omitempty"`
	Still         bool     `json:"still"` // no animations: tests and snapshots, Reduce motion
	Toasts        []string `json:"toasts,omitempty"`

	Nav     *navViewState     `json:"nav,omitempty"`
	Notices []noticeViewState `json:"notices,omitempty"`

	// Logs is the panel under the page (nil while closed or on /logs),
	// LogsHeight its height in DIPs at 100 %.
	Logs       *logsViewState `json:"logs,omitempty"`
	LogsHeight float32        `json:"logs_height"`

	Palette *paletteViewState `json:"palette,omitempty"`
	Sheet   *sheetViewState   `json:"sheet,omitempty"`

	SettingsOpen bool `json:"settings_open"` // the toasts move to the dialog's foot
}

// openingViewState is the first-use opening at one frame: how long it has
// played (opening.go draws each part's progress from it).
type openingViewState struct {
	Elapsed time.Duration `json:"elapsed"`
}
