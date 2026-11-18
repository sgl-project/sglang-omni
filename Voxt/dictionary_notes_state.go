package main

import "github.com/sgl-project/sglang-omni/Voxt/internal/settings"

type dictionaryNotesViewState struct {
	Enabled  bool `json:"enabled"`
	NotesOff bool `json:"notes_off"`
}

func buildDictionaryNotesViewState(preferences settings.Settings) dictionaryNotesViewState {
	return dictionaryNotesViewState{Enabled: preferences.Dictionary.Enabled, NotesOff: preferences.Dictionary.NotesOff}
}
