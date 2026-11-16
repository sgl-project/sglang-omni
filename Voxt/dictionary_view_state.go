package main

import (
	"errors"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const (
	dictTermsTab = iota
	dictNotesTab
)

type dictionaryTermViewState struct {
	ID          string `json:"id"`
	Term        string `json:"term"`
	AliasesText string `json:"aliases_text"`
	Note        string `json:"note"`
	Enabled     bool   `json:"enabled"`
}

type dictionarySuggestionViewState struct {
	Key        string `json:"key"`
	Term       string `json:"term"`
	HeardText  string `json:"heard_text"`
	SourceCode string `json:"source_code"`
	SourceText string `json:"source_text"`
}

type dictionaryTermEditViewState struct {
	Open       bool   `json:"open"`
	ID         string `json:"id"`
	Term       string `json:"term"`
	Aliases    string `json:"aliases"`
	Note       string `json:"note"`
	Enabled    bool   `json:"enabled"`
	Error      string `json:"error"`
	ErrorCode  string `json:"error_code"`
	Saving     bool   `json:"saving"`
	Original   string `json:"original"`
	Suggestion string `json:"suggestion"`
}

type dictionaryScanViewState struct {
	Stage        string `json:"stage"`
	ModelName    string `json:"model_name"`
	Body         string `json:"body"`
	Warning      string `json:"warning"`
	ErrorCode    string `json:"error_code"`
	CanStart     bool   `json:"can_start"`
	StopDisabled bool   `json:"stop_disabled"`
}

type dictionaryViewState struct {
	Read              bool                            `json:"read"`
	Loading           bool                            `json:"loading"`
	Error             string                          `json:"error"`
	Query             string                          `json:"query"`
	ListFor           string                          `json:"list_for"`
	CountLabel        string                          `json:"count_label"`
	Enabled           bool                            `json:"enabled"`
	ASRHints          bool                            `json:"asr_hints"`
	Glossary          bool                            `json:"glossary"`
	Learn             bool                            `json:"learn"`
	CanScan           bool                            `json:"can_scan"`
	LegacyImport      bool                            `json:"legacy_import"`
	Tab               int                             `json:"tab"`
	Terms             []dictionaryTermViewState       `json:"terms"`
	Suggestions       []dictionarySuggestionViewState `json:"suggestions"`
	SuggestionCount   int                             `json:"suggestion_count"`
	ScanOpen          bool                            `json:"scan_open"`
	Scan              dictionaryScanViewState         `json:"scan"`
	Edit              dictionaryTermEditViewState     `json:"edit"`
	NotesText         string                          `json:"notes_text"`
	NotesQuery        string                          `json:"notes_query"`
	NotesFindCurrent  int                             `json:"notes_find_current"`
	Notes             dictionaryNotesViewState        `json:"notes"`
	GlossaryAllUpTo   int                             `json:"glossary_all_up_to"`
	MaxGlossaryTerms  int                             `json:"max_glossary_terms"`
	LearnWatchSeconds int                             `json:"learn_watch_seconds"`
}

type dictionaryTermEditStateInput struct {
	Open       bool
	ID         string
	Term       string
	Aliases    string
	Note       string
	Enabled    bool
	Error      string
	Err        error
	Saving     bool
	Original   string
	Suggestion string
}

type dictionaryScanStateInput struct {
	Model    local.Model
	Counted  bool
	Count    int
	Running  bool
	Stopping bool
	Read     int
	Total    int
	Found    int
	Ended    bool
	Stopped  bool
	Err      error
}

type dictionaryViewStateInput struct {
	Terms        []store.Term
	TermRows     []dictionaryTermViewState
	TermsReady   bool
	Suggestions  []learn.Suggestion
	Settings     settings.Settings
	Installed    []local.Model
	Read         bool
	Loading      bool
	Error        string
	Query        string
	ListFor      string
	Total        int
	LegacyImport bool
	Tab          int
	ScanOpen     bool
	Scan         dictionaryScanStateInput
	Edit         dictionaryTermEditStateInput
	NotesText    string
	NotesQuery   string
	NotesFind    int
	Notes        *dictionaryNotesStateInput
}

func buildDictionaryViewState(translator *i18n.Translator, input dictionaryViewStateInput) dictionaryViewState {
	_, canScan := dictionaryScanModel(input.Settings, input.Installed)
	state := dictionaryViewState{
		Read: input.Read, Loading: input.Loading, Error: input.Error, Query: input.Query, ListFor: input.ListFor,
		Enabled: input.Settings.Dictionary.Enabled, ASRHints: input.Settings.Dictionary.ASRHints,
		Glossary: input.Settings.Dictionary.Glossary, Learn: input.Settings.Dictionary.Learn,
		CanScan: canScan, LegacyImport: input.LegacyImport, Tab: input.Tab, ScanOpen: input.ScanOpen,
		Scan: buildDictionaryScanViewState(translator, input.Scan), Edit: buildDictionaryTermEditViewState(translator, input.Edit),
		NotesText: input.NotesText, NotesQuery: input.NotesQuery, NotesFindCurrent: input.NotesFind,
		SuggestionCount: len(input.Suggestions), GlossaryAllUpTo: compose.GlossaryAllUpTo,
		MaxGlossaryTerms: compose.MaxGlossaryTerms, LearnWatchSeconds: learn.WatchSeconds,
	}
	if input.Total > 0 {
		state.CountLabel = translator.N(input.Total, "%d term", "%d terms", input.Total)
	}
	if input.TermsReady {
		state.Terms = input.TermRows
	} else {
		state.Terms = buildDictionaryTermViewStates(translator, input.Terms)
	}
	for _, suggestion := range input.Suggestions {
		source := translator.T("From your correction")
		if suggestion.Source == learn.FromHistory {
			source = translator.T("From your history")
		}
		state.Suggestions = append(state.Suggestions, dictionarySuggestionViewState{
			Key: suggestion.Key(), Term: suggestion.Term, HeardText: strings.Join(suggestion.Heard, translator.T(", ")),
			SourceCode: string(suggestion.Source), SourceText: source,
		})
	}
	if input.Notes != nil {
		state.Notes = buildDictionaryNotesViewState(translator, *input.Notes)
	}
	return state
}

func buildDictionaryTermViewStates(translator *i18n.Translator, terms []store.Term) []dictionaryTermViewState {
	rows := make([]dictionaryTermViewState, 0, len(terms))
	for _, term := range terms {
		rows = append(rows, dictionaryTermViewState{
			ID: term.ID, Term: term.Term, AliasesText: strings.Join(term.Aliases, translator.T(", ")), Note: term.Note, Enabled: term.Enabled,
		})
	}
	return rows
}

func buildDictionaryTermEditViewState(translator *i18n.Translator, input dictionaryTermEditStateInput) dictionaryTermEditViewState {
	state := dictionaryTermEditViewState{
		Open: input.Open, ID: input.ID, Term: input.Term, Aliases: input.Aliases, Note: input.Note,
		Enabled: input.Enabled, Error: input.Error, Saving: input.Saving, Original: input.Original, Suggestion: input.Suggestion,
	}
	if input.Err != nil {
		state.Error = uitext.TermErrorTextFor(translator, input.Err)
	}
	switch {
	case errors.Is(input.Err, dict.ErrEmpty):
		state.ErrorCode = "empty"
	case errors.Is(input.Err, dict.ErrLong):
		state.ErrorCode = "long"
	case errors.Is(input.Err, dict.ErrNewline):
		state.ErrorCode = "newline"
	case errors.Is(input.Err, dict.ErrFormat):
		state.ErrorCode = "format"
	case errors.Is(input.Err, core.ErrExists):
		state.ErrorCode = "exists"
	case input.Err != nil:
		state.ErrorCode = "generic"
	}
	return state
}

func dictionaryScanModel(preferences settings.Settings, installed []local.Model) (local.Model, bool) {
	id := preferences.Features.Transcribe.LLM
	model, ok := local.Choose(installed, id)
	return model, ok && model.ID == id
}

func buildDictionaryScanViewState(translator *i18n.Translator, input dictionaryScanStateInput) dictionaryScanViewState {
	state := dictionaryScanViewState{ModelName: input.Model.Name, StopDisabled: input.Stopping, ErrorCode: llm.Code(input.Err)}
	if input.Err != nil {
		state.Warning, _ = uitext.ModelErrorTextFor(translator, input.Err)
		if state.ErrorCode == "" {
			state.ErrorCode = "generic"
			switch {
			case errors.Is(input.Err, local.ErrNotInstalled):
				state.ErrorCode = "not_installed"
			case errors.Is(input.Err, local.ErrNoRuntime):
				state.ErrorCode = "no_runtime"
			}
		}
	}
	switch {
	case input.Running:
		state.Stage = "running"
		state.Body = translator.N(input.Found, "Read %d/%d · %d suggestion", "Read %d/%d · %d suggestions", input.Read, input.Total, input.Found)
	case input.Ended:
		state.Stage = "ended"
		state.Body = dictionaryScanFoundTextFor(translator, input.Found)
		if input.Stopped {
			state.Body = translator.N(input.Found, "Stopped, found %d suggestion", "Stopped, found %d suggestions", input.Found)
		}
	case input.Err != nil:
		state.Stage = "error"
		state.CanStart = input.Count > 0
	case input.Counted && input.Count == 0:
		state.Stage = "empty"
		state.Body = translator.T("No new Transcribe results to scan.")
	case input.Counted:
		state.Stage = "ready"
		state.Body = translator.N(input.Count, "Voxt will send %d Transcribe result to %s to find names and terms. Nothing is added until you confirm.",
			"Voxt will send %d Transcribe results to %s to find names and terms. Nothing is added until you confirm.", input.Count, input.Model.Name)
		state.CanStart = true
	default:
		state.Stage = "loading"
	}
	return state
}

func dictionaryScanFoundTextFor(translator *i18n.Translator, found int) string {
	return translator.N(found, "Found %d suggestion", "Found %d suggestions", found)
}
