package main

type dictionaryActions interface {
	loadDict()
	searchDictionary(query string)
	setDictionaryEnabled(on bool)
	setDictionaryASRHints(on bool)
	setDictionaryGlossary(on bool)
	setDictionaryLearn(on bool)
	openDictionaryTranscribe()
	importTerms()
	importLegacy()
	exportTerms(format string)
	openDictionaryTerm(id string)
	submitTerm()
	deleteDictionaryTerm(id, original, profile string)
	setDictionaryTermEnabled(id string, enabled bool)
	acceptDictionarySuggestion(key string)
	editDictionarySuggestion(key string)
	dismissSuggestion(key string)
	openDictionaryScan()
	startScan()
	stopDictionaryScan()
	autosaveDictionaryNotes(changed, focused bool)
	setDictionaryNotesOff(off bool)
	switchDictionaryProfile(id string)
	openDictionaryProfileEdit(id string)
	saveDictionaryProfile()
	deleteDictionaryProfile(id string)
}

type notesActions interface {
	loadNotes()
	deleteDictionaryNote(id, text string)
	copyNote(text string)
	saveNoteEdit(id, text string)
	setNoteKeys(keys string)
	recordNoteKeys(keys string) notesShortcutViewState
	updateNoteKeysDraft(text string) notesShortcutViewState
	setNoteHotkeyFocus(focused bool)
	openNotePermissions()
}
