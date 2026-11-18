package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (host *meetingDemoHost) loadDict() { host.record("loadDict") }

func (host *meetingDemoHost) searchDictionary(query string) {
	host.dictionaryUI.query = query
	host.record("searchDictionary %q", query)
}

func (host *meetingDemoHost) setDictionaryEnabled(on bool) {
	host.record("setDictionaryEnabled %t", on)
}
func (host *meetingDemoHost) setDictionaryASRHints(on bool) {
	host.record("setDictionaryASRHints %t", on)
}
func (host *meetingDemoHost) setDictionaryGlossary(on bool) {
	host.record("setDictionaryGlossary %t", on)
}
func (host *meetingDemoHost) setDictionaryLearn(on bool) { host.record("setDictionaryLearn %t", on) }
func (host *meetingDemoHost) openDictionaryTranscribe()  { host.record("openDictionaryTranscribe") }
func (host *meetingDemoHost) importTerms()               { host.record("importTerms") }
func (host *meetingDemoHost) importLegacy()              { host.record("importLegacy") }
func (host *meetingDemoHost) exportTerms(format string)  { host.record("exportTerms %s", format) }

func (host *meetingDemoHost) openDictionaryTerm(id string) {
	host.dictionaryUI.editOpen = true
	host.dictionaryUI.edit = termEdit{enabled: true}
	if id != "" {
		for _, term := range host.current().state.Dictionary.Terms {
			if term.ID == id {
				host.dictionaryUI.edit = termEdit{id: id, term: term.Term, aliases: term.AliasesText, note: term.Note, enabled: term.Enabled, original: term.Term}
			}
		}
	}
	host.dictionaryUI.editWas = host.dictionaryUI.edit
	host.record("openDictionaryTerm %q", id)
}

func (host *meetingDemoHost) submitTerm() { host.record("submitTerm") }

func (host *meetingDemoHost) deleteDictionaryTerm(id, original string) {
	host.dictionaryUI.editOpen = false
	host.record("deleteDictionaryTerm %s %q", id, original)
}

func (host *meetingDemoHost) setDictionaryTermEnabled(id string, enabled bool) {
	host.record("setDictionaryTermEnabled %s %t", id, enabled)
}

func (host *meetingDemoHost) acceptDictionarySuggestion(key string) {
	host.record("acceptDictionarySuggestion %s", key)
}

func (host *meetingDemoHost) editDictionarySuggestion(key string) {
	for _, suggestion := range host.current().state.Dictionary.Suggestions {
		if suggestion.Key == key {
			host.dictionaryUI.editOpen = true
			host.dictionaryUI.edit = termEdit{term: suggestion.Term, enabled: true, suggestion: key}
			host.dictionaryUI.editWas = host.dictionaryUI.edit
		}
	}
	host.record("editDictionarySuggestion %s", key)
}

func (host *meetingDemoHost) dismissSuggestion(key string) { host.record("dismissSuggestion %s", key) }

func (host *meetingDemoHost) openDictionaryScan() {
	host.dictionaryUI.scanOpen = true
	host.record("openDictionaryScan")
}

func (host *meetingDemoHost) startScan()          { host.record("startScan") }
func (host *meetingDemoHost) stopDictionaryScan() { host.record("stopDictionaryScan") }
func (host *meetingDemoHost) saveNotes()          { host.record("saveNotes") }

func (host *meetingDemoHost) dictionaryNotesModel() dictionaryNotesViewState {
	dictionary := host.current().state.Dictionary
	preferences := settings.Defaults()
	preferences.Dictionary.Enabled = dictionary.Enabled
	preferences.Dictionary.Glossary = dictionary.Glossary
	preferences.Dictionary.NotesOff = dictionary.Notes.NotesOff
	preferences.Features.Transcribe.Enhance = !dictionary.Notes.EnhanceOff
	terms := make([]store.Term, 0, len(dictionary.Terms))
	for _, term := range dictionary.Terms {
		terms = append(terms, store.Term{ID: term.ID, Term: term.Term, Aliases: dict.SplitAliases(term.AliasesText), Note: term.Note, Enabled: term.Enabled})
	}
	return buildDictionaryNotesViewState(i18n.CurrentTranslator(), dictionaryNotesStateInput{
		Settings: preferences, Terms: terms, Feature: notesFeatures[host.dictionaryUI.notesFeature], NotesText: host.dictionaryUI.notes.text,
	})
}

func (host *meetingDemoHost) autosaveDictionaryNotes(changed, focused bool) dictionaryNotesViewState {
	notes := &host.dictionaryUI.notes
	if changed {
		notes.dirty = true
		notes.seq++
		host.record("autosaveDictionaryNotes %t %t", changed, focused)
	} else if notes.dirty && !focused {
		notes.dirty = false
		host.saveNotes()
	}
	return host.dictionaryNotesModel()
}

func (host *meetingDemoHost) setDictionaryNotesOff(off bool) {
	host.record("setDictionaryNotesOff %t", off)
}

func (host *meetingDemoHost) loadNotes() { host.record("loadNotes") }

func (host *meetingDemoHost) deleteDictionaryNote(id, text string) {
	host.record("deleteDictionaryNote %s %q", id, text)
}

func (host *meetingDemoHost) setNoteKeys(keys string) {
	host.notesUI.keys.text = keys
	host.record("setNoteKeys %q", keys)
}

func (host *meetingDemoHost) recordNoteKeys(keys string) notesShortcutViewState {
	chord, err := hotkey.Parse(keys)
	host.notesUI.keys.unknown = err != nil
	host.record("recordNoteKeys %q", keys)
	state := host.current().state.Notes.Shortcut
	state.Unknown = err != nil
	if err == nil {
		host.notesUI.keys.text, host.notesUI.keys.edited = chord.String(), false
		state.Text, state.EditingText, state.Edited = chord.String(), uitext.HotkeyText(chord.String()), false
		state.ParsedKeys = chord.String()
		state.ParseError, state.ParseCode, state.EditedHint = "", "", ""
	}
	return state
}

func (host *meetingDemoHost) updateNoteKeysDraft(text string) notesShortcutViewState {
	host.notesUI.keys.text, host.notesUI.keys.edited = text, true
	host.record("updateNoteKeysDraft %q", text)
	state := host.current().state.Notes.Shortcut
	parsed := buildNotesShortcutViewState(i18n.CurrentTranslator(), notesViewStateInput{
		Features: settings.Features{Note: settings.Hotkey{Keys: state.Keys}},
		Editing:  notesShortcutStateInput{Custom: state.Custom, Typing: state.Typing, Edited: true, Text: text},
	})
	state.Text, state.EditingText, state.Edited = parsed.Text, parsed.EditingText, true
	state.ParsedKeys, state.ParseError, state.ParseCode = parsed.ParsedKeys, parsed.ParseError, parsed.ParseCode
	state.EditedHint = parsed.EditedHint
	return state
}

func (host *meetingDemoHost) setNoteHotkeyFocus(focused bool) {
	if host.notesUI.keys.recFocused == focused {
		return
	}
	host.record("setNoteHotkeyFocus %t", focused)
}

func (host *meetingDemoHost) openNotePermissions() { host.record("openNotePermissions") }
