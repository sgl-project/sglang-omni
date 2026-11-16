package main

import (
	"reflect"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func (a *app) saveNotes() {
	notes := &a.dictUI.notes
	notes.dirty = false
	notes.seen = notes.text
	if value := notes.text; value != a.prefs.Dictionary.Notes {
		a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = value })
	}
}

func (a *app) autosaveDictionaryNotes(changed, focused bool) dictionaryNotesViewState {
	notes := &a.dictUI.notes
	if changed {
		notes.dirty = true
		notes.seq++
		if seq := notes.seq; a.win != nil {
			time.AfterFunc(promptSaveDelay, func() {
				a.post(func() {
					if notes.dirty && notes.seq == seq {
						a.saveNotes()
					}
				})
			})
		}
	} else if notes.dirty && !focused {
		a.saveNotes()
	}
	return a.dictionaryNotesModel(a.dictionaryNotesInput(notesFeatures[a.dictUI.notesFeature]))
}

func (a *app) notesTerms() []store.Term {
	if a.prefs.Dictionary.Enabled && a.prefs.Dictionary.Glossary {
		return a.dv.all
	}
	return nil
}

func (a *app) dictionaryNotesInput(feature string) dictionaryNotesStateInput {
	return dictionaryNotesStateInput{Settings: a.prefs, Terms: a.notesTerms(), Feature: feature, NotesText: a.dictUI.notes.text}
}

func (a *app) notesExampleInput(feature string) compose.Input {
	_, full := dictionaryNotesComposeInputs(i18n.CurrentTranslator(), a.dictionaryNotesInput(feature))
	return full
}

func (a *app) dictionaryNotesModel(input dictionaryNotesStateInput) dictionaryNotesViewState {
	translator := i18n.CurrentTranslator()
	base, _ := dictionaryNotesComposeInputs(translator, input)
	example := &a.dv.example
	if example.built && len(input.Terms) == len(example.terms) &&
		(len(input.Terms) == 0 || &input.Terms[0] == &example.terms[0]) && reflect.DeepEqual(base, example.in) {
		return dictionaryNotesStateFromExample(input, example.example)
	}
	built := buildDictionaryNotesExample(translator, input)
	*example = notesExampleCache{in: base, terms: input.Terms, built: true, example: built}
	return dictionaryNotesStateFromExample(input, built)
}
