package main

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

func (host *meetingDemoHost) deleteDictionaryTerm(id, original, profile string) {
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

func (host *meetingDemoHost) autosaveDictionaryNotes(changed, focused bool) {
	notes := &host.dictionaryUI.notes
	if changed {
		notes.dirty = true
		notes.seq++
		host.record("autosaveDictionaryNotes %t %t", changed, focused)
	} else if notes.dirty && !focused {
		notes.dirty = false
		host.saveNotes()
	}
}

func (host *meetingDemoHost) setDictionaryNotesOff(off bool) {
	host.record("setDictionaryNotesOff %t", off)
}

func (host *meetingDemoHost) switchDictionaryProfile(id string) {
	host.record("switchDictionaryProfile %s", id)
}

func (host *meetingDemoHost) openDictionaryProfileEdit(id string) {
	edit := dictionaryProfileEdit{open: true}
	for _, profile := range host.current().state.Dictionary.Profiles {
		if id != "" && profile.ID == id {
			edit.id, edit.name = profile.ID, profile.Name
		}
	}
	host.dictionaryUI.profileEdit = edit
	host.record("openDictionaryProfileEdit %q", id)
}

func (host *meetingDemoHost) saveDictionaryProfile() {
	host.record("saveDictionaryProfile %q", host.dictionaryUI.profileEdit.name)
}

func (host *meetingDemoHost) deleteDictionaryProfile(id string) {
	host.record("deleteDictionaryProfile %s", id)
}

func (host *meetingDemoHost) loadNotes() { host.record("loadNotes") }

func (host *meetingDemoHost) deleteDictionaryNote(id, text string) {
	host.record("deleteDictionaryNote %s %q", id, text)
}

func (host *meetingDemoHost) copyNote(text string) { host.record("copyNote %q", text) }

func (host *meetingDemoHost) saveNoteEdit(id, text string) {
	host.record("saveNoteEdit %s %q", id, text)
}

func (host *meetingDemoHost) setNoteKeys(keys string) {
	host.notesUI.keys.text = keys
	host.record("setNoteKeys %q", keys)
}

func (host *meetingDemoHost) recordNoteKeys(keys string) notesShortcutViewState {
	host.notesUI.keys.text, host.notesUI.keys.edited = keys, false
	host.record("recordNoteKeys %q", keys)
	state := host.current().state.Notes.Shortcut
	state.Text, state.EditingText = keys, keys
	return state
}

func (host *meetingDemoHost) updateNoteKeysDraft(text string) notesShortcutViewState {
	host.notesUI.keys.text, host.notesUI.keys.edited = text, true
	host.record("updateNoteKeysDraft %q", text)
	state := host.current().state.Notes.Shortcut
	state.Text, state.EditingText, state.Edited = text, text, true
	state.ParsedKeys, state.ParseError = text, ""
	return state
}

func (host *meetingDemoHost) setNoteHotkeyFocus(focused bool) {
	if host.notesUI.keys.recFocused == focused {
		return
	}
	host.notesUI.keys.recFocused = focused
	host.record("setNoteHotkeyFocus %t", focused)
}

func (host *meetingDemoHost) openNotePermissions() { host.record("openNotePermissions") }
