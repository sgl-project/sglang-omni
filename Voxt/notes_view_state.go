package main

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type notesRowViewState struct {
	ID         string `json:"id"`
	Text       string `json:"text"`
	TimeText   string `json:"time_text"`
	DeleteText string `json:"delete_text"`
}

type notesKeyOptionViewState struct {
	Label string `json:"label"`
	Keys  string `json:"keys"`
}

type notesShortcutViewState struct {
	Keys          string                    `json:"keys"`
	CurrentText   string                    `json:"current_text"`
	Options       []notesKeyOptionViewState `json:"options"`
	Problems      []string                  `json:"problems"`
	ProblemCodes  []string                  `json:"problem_codes"`
	SystemCodes   []string                  `json:"system_codes"`
	UsedBy        []string                  `json:"used_by"`
	EmptyHint     string                    `json:"empty_hint"`
	SetHint       bool                      `json:"set_hint"`
	HelperRecords bool                      `json:"helper_records"`
	Selection     string                    `json:"selection"`
	Custom        bool                      `json:"custom"`
	Typing        bool                      `json:"typing"`
	Edited        bool                      `json:"edited"`
	Text          string                    `json:"text"`
	EditingText   string                    `json:"editing_text"`
	ParsedKeys    string                    `json:"parsed_keys"`
	ParseError    string                    `json:"parse_error"`
	ParseCode     string                    `json:"parse_code"`
	EditedHint    string                    `json:"edited_hint"`
	RecordDenied  bool                      `json:"record_denied"`
	Unknown       bool                      `json:"unknown"`
}

type notesViewState struct {
	Read       bool                   `json:"read"`
	Loading    bool                   `json:"loading"`
	Error      string                 `json:"error"`
	CountLabel string                 `json:"count_label"`
	Rows       []notesRowViewState    `json:"rows"`
	Shortcut   notesShortcutViewState `json:"shortcut"`
}

type notesShortcutStateInput struct {
	Selection    string
	Custom       bool
	Typing       bool
	Edited       bool
	Text         string
	RecordDenied bool
	Unknown      bool
	Err          error
}

type notesViewStateInput struct {
	Notes         []store.Note
	Read          bool
	Loading       bool
	Error         string
	Features      settings.Features
	Now           time.Time
	HelperRecords bool
	UsedBy        []string
	Editing       notesShortcutStateInput
}

func buildNotesViewState(translator *i18n.Translator, input notesViewStateInput) notesViewState {
	state := notesViewState{Read: input.Read, Loading: input.Loading, Error: input.Error}
	if len(input.Notes) > 0 {
		state.CountLabel = translator.N(len(input.Notes), "%d note", "%d notes", len(input.Notes))
	}
	state.Rows = make([]notesRowViewState, 0, len(input.Notes))
	for _, note := range input.Notes {
		state.Rows = append(state.Rows, notesRowViewState{
			ID: note.ID, Text: note.Text, TimeText: uitext.TimeAgoFor(translator, note.CreatedAt, input.Now),
			DeleteText: translator.T("Delete %s", noteExcerpt(note.Text)),
		})
	}
	state.Shortcut = buildNotesShortcutViewState(translator, input)
	return state
}

func buildNotesShortcutViewState(translator *i18n.Translator, input notesViewStateInput) notesShortcutViewState {
	current := input.Features.Note.Keys
	state := notesShortcutViewState{
		Keys: current, CurrentText: notesChordLabelFor(translator, current), HelperRecords: input.HelperRecords,
		Selection: input.Editing.Selection, Custom: input.Editing.Custom, Typing: input.Editing.Typing,
		Edited: input.Editing.Edited, Text: input.Editing.Text, EditingText: uitext.HotkeyTextFor(translator, input.Editing.Text),
		RecordDenied: input.Editing.RecordDenied, Unknown: input.Editing.Unknown, UsedBy: input.UsedBy,
		SetHint: current == "",
	}
	if !input.Editing.Custom {
		state.Selection = notesChordLabelFor(translator, current)
		state.Text, state.EditingText = current, uitext.HotkeyTextFor(translator, current)
		if current == "" {
			state.Selection = translator.T("Off")
		}
	}
	state.EmptyHint = translator.T("Choose a shortcut above, then press it in any app and speak. What you say is saved here instead of typed.")
	state.Problems = uitext.HotkeyProblemsFor(translator, input.Features, hotkey.Note)[0]
	for _, problem := range hotkey.Check(input.Features) {
		if problem.Feature == hotkey.Note || problem.Code == hotkey.ProblemConflict && problem.Other == hotkey.Note {
			state.ProblemCodes = append(state.ProblemCodes, problem.Code)
			if problem.System != "" {
				state.SystemCodes = append(state.SystemCodes, problem.System)
			}
		}
	}
	if current != "" && len(state.Problems) == 0 {
		state.EmptyHint = translator.T("Press %s in any app, speak, and press it again. What you say is saved here instead of typed.", uitext.HotkeyTextFor(translator, current))
	}
	state.Options = notesKeyOptionsFor(translator, current)
	editing := input.Editing
	err := editing.Err
	var chord hotkey.Chord
	if err == nil {
		chord, err = hotkey.Parse(editing.Text)
	}
	if err == nil {
		state.ParsedKeys = chord.String()
	}
	state.ParseError = uitext.ParseErrorTextFor(translator, err)
	switch {
	case errors.Is(err, hotkey.ErrEmpty):
		state.ParseCode = "empty"
	case errors.Is(err, hotkey.ErrNoModifier):
		state.ParseCode = "no_modifier"
	case errors.Is(err, hotkey.ErrTwoKeys):
		state.ParseCode = "two_keys"
	case errors.Is(err, hotkey.ErrFnSide):
		state.ParseCode = "fn_side"
	case err != nil:
		state.ParseCode = "unknown"
	}
	if editing.Edited && err == nil {
		state.EditedHint = translator.T("Press Return to use %s.", uitext.HotkeyTextFor(translator, editing.Text))
	}
	return state
}

func notesChordLabelFor(translator *i18n.Translator, keys string) string {
	if preset := presetOf(keys); preset != "" {
		return uitext.HotkeyTextFor(translator, preset)
	}
	return uitext.HotkeyTextFor(translator, keys)
}

func notesKeyOptionsFor(translator *i18n.Translator, current string) []notesKeyOptionViewState {
	options := []notesKeyOptionViewState{{Label: translator.T("Off")}}
	seen := map[string]bool{}
	for _, keys := range keyPresets {
		label := uitext.HotkeyTextFor(translator, keys)
		options = append(options, notesKeyOptionViewState{Label: label, Keys: keys})
		seen[label] = true
	}
	if current != "" && presetOf(current) == "" {
		label := uitext.HotkeyTextFor(translator, current)
		if !seen[label] {
			options = append(options, notesKeyOptionViewState{Label: label, Keys: current})
		}
	}
	return append(options, notesKeyOptionViewState{Label: translator.T("Custom…")})
}

func noteExcerpt(text string) string {
	if characters := []rune(text); len(characters) > 40 {
		return string(characters[:40]) + "…"
	}
	return text
}
