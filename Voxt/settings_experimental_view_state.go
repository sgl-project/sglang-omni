package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type sessionSlotViewState struct {
	Slot     int                  `json:"slot"` // its index in Features.Sessions
	Title    string               `json:"title"`
	Recorder keyRecorderViewState `json:"recorder"`
	Problems []string             `json:"problems"`
}

type settingsExperimentalViewState struct {
	Sessions   []sessionSlotViewState `json:"sessions"`
	SessionsOn bool                   `json:"sessions_on,omitempty"`
	// SessionsUnsaved: sessions are on with Privacy's history off, so
	// they end when Voxt quits.
	SessionsUnsaved bool `json:"sessions_unsaved,omitempty"`
}

type settingsExperimentalStateInput struct {
	Features          settings.Features
	HelperRecordsKeys bool
	KeepHistory       bool
}

func buildSettingsExperimentalViewState(translator *i18n.Translator, input settingsExperimentalStateInput) settingsExperimentalViewState {
	state := settingsExperimentalViewState{
		SessionsOn:      input.Features.SessionsOn,
		SessionsUnsaved: input.Features.SessionsOn && !input.KeepHistory,
	}
	for i, id := range hotkey.Sessions {
		if id != hotkey.SessionNext {
			continue // the card's row and the menu bar go to a session; numbers shift as sessions come and go
		}
		recorder := buildKeyRecorderViewState(translator, input.HelperRecordsKeys, input.Features.Sessions[i].Keys)
		if input.Features.Sessions[i].Keys == "" {
			recorder.DisplayedKeys = uitext.HotkeyGlyphs("")
		}
		state.Sessions = append(state.Sessions, sessionSlotViewState{
			Slot:     i,
			Title:    uitext.FeatureTitleFor(translator, id),
			Recorder: recorder,
			Problems: uitext.HotkeyProblemsFor(translator, input.Features, id)[0],
		})
	}
	return state
}
