package main

// The command palette's and the shortcut sheet's view state: the
// commands the filter keeps (titles and shortcuts translated), and the
// sheet's rows with the features' first shortcuts as key caps.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// paletteItemViewState is one command in the palette's list.
type paletteItemViewState struct {
	ID    string `json:"id"`
	Group string `json:"group"`
	Title string `json:"title"`
	Keys  string `json:"keys,omitempty"`
}

// paletteViewState is the palette's display data: the enabled commands
// matching the query, in the registry's order.
type paletteViewState struct {
	Items []paletteItemViewState `json:"items"`
}

// buildPaletteViewState keeps the enabled commands every word of the
// query matches.
func buildPaletteViewState(commands []command, query string) paletteViewState {
	state := paletteViewState{}
	for _, cmd := range commands {
		if !cmd.enabled || !cmd.matches(query) {
			continue
		}
		state.Items = append(state.Items, paletteItemViewState{ID: cmd.id, Group: cmd.group, Title: cmd.title, Keys: cmd.keys})
	}
	return state
}

// sheetRowViewState is one shortcut: its key cap and what it does.
type sheetRowViewState struct {
	Keys  string `json:"keys"`
	Title string `json:"title"`
}

// sheetGroupViewState is a section of the "?" sheet.
type sheetGroupViewState struct {
	Title string               `json:"title"`
	Rows  []sheetRowViewState `json:"rows"`
}

// sheetViewState is the "?" sheet's display data.
type sheetViewState struct {
	Groups []sheetGroupViewState `json:"groups"`
}

// buildSheetViewState lists the keyboard shortcuts: the features' first
// ones anywhere on the Mac, and the window's.
func buildSheetViewState(translator *i18n.Translator, features settings.Features) sheetViewState {
	first := func(hs []settings.Hotkey) string {
		if len(hs) == 0 {
			return "—"
		}
		return uitext.HotkeyGlyphs(hs[0].Keys)
	}
	return sheetViewState{Groups: []sheetGroupViewState{
		{translator.T("Anywhere on the Mac"), []sheetRowViewState{
			{first(features.Transcribe.Hotkeys), translator.T("Transcribe")},
			{first(features.Translate.Hotkeys), translator.T("Translate")},
			{first(features.Rewrite.Hotkeys), translator.T("Rewrite")},
			{uitext.HotkeyGlyphs(features.PasteLast.Keys), translator.T("Paste the last result")},
		}},
		{translator.T("Navigation"), []sheetRowViewState{
			{"⌘K", translator.T("Command palette")},
			{"⌘1  ⌘2", translator.T("Home, History")},
			{"⌘3 … ⌘6", translator.T("Transcribe, Translate, Rewrite, App enhancement")},
			{"⌘,", translator.T("Settings")},
			{"⌘[  ⌘]", translator.T("Back, forward")},
			{"⇧⌘L", translator.T("Show or hide the logs panel")},
			{"?", translator.T("Keyboard shortcuts")},
			{"Esc", translator.T("Close a dialog or popover")},
		}},
	}}
}
