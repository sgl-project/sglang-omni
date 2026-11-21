package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
)

func TestDictionaryDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		prepare func(*meetingDemoHost)
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "load", fixture: "en/dictionary-error", steps: historySteps("Try again"), want: []string{"loadDict"}},
		{name: "search", fixture: "en/dictionary-normal", steps: []historyDemoStep{{label: "Search dictionary", text: "vox"}}, want: []string{`searchDictionary "vox"`}},
		{name: "clear search", fixture: "en/dictionary-search-empty", steps: historySteps("Clear search"), want: []string{`searchDictionary ""`}},
		{name: "enabled", fixture: "en/dictionary-normal", steps: historySteps("Use the dictionary"), want: []string{"setDictionaryEnabled false"}},
		{name: "hints", fixture: "en/dictionary-normal", steps: historySteps("Speech hints"), want: []string{"setDictionaryASRHints false"}},
		{name: "glossary", fixture: "en/dictionary-normal", steps: historySteps("Glossary for the language model"), want: []string{"setDictionaryGlossary false"}},
		{name: "learn", fixture: "en/dictionary-normal", steps: historySteps("Learn from my corrections"), want: []string{"setDictionaryLearn true"}},
		{name: "scan model", fixture: "en/dictionary-scan-unavailable", steps: historySteps("Open Transcribe"), want: []string{"openDictionaryTranscribe"}},
		{name: "open scan", fixture: "en/dictionary-normal", steps: historySteps("Scan history…"), want: []string{"openDictionaryScan"}},
		{name: "import", fixture: "en/dictionary-normal", steps: []historyDemoStep{{label: "More actions"}, {label: "Import…", menu: true}}, want: []string{"importTerms"}},
		{name: "legacy import", fixture: "en/dictionary-normal", steps: []historyDemoStep{{label: "More actions"}, {label: "Import from Voxt 1.x…", menu: true}}, want: []string{"importLegacy"}},
		{name: "export csv", fixture: "en/dictionary-normal", steps: []historyDemoStep{{label: "More actions"}, {label: "Export as CSV…", menu: true}}, want: []string{"exportTerms csv"}},
		{name: "export text", fixture: "en/dictionary-normal", steps: []historyDemoStep{{label: "More actions"}, {label: "Export as text…", menu: true}}, want: []string{"exportTerms txt"}},
		{name: "empty import", fixture: "en/dictionary-empty", steps: historySteps("Import terms"), want: []string{"importTerms"}},
		{name: "add", fixture: "en/dictionary-normal", steps: historySteps("Add term"), want: []string{`openDictionaryTerm ""`}},
		{name: "edit", fixture: "en/dictionary-normal", steps: historySteps("Edit Voxt"), want: []string{`openDictionaryTerm "term-voxt"`}},
		{name: "delete", fixture: "en/dictionary-normal", steps: historySteps("Delete Voxt"), want: []string{`deleteDictionaryTerm term-voxt "Voxt"`}},
		{name: "toggle", fixture: "en/dictionary-normal", steps: historySteps("Use Voxt"), want: []string{"setDictionaryTermEnabled term-voxt false"}},
		{name: "save", fixture: "en/dictionary-edit-existing", steps: historySteps("Save"), want: []string{"submitTerm"}},
		{name: "submit return", fixture: "en/dictionary-edit-new", steps: []historyDemoStep{{label: "Term as written", text: "Voxt"}, {key: ui.KeyEnter}}, want: []string{"submitTerm"}},
		{name: "dialog delete", fixture: "en/dictionary-edit-existing", steps: historySteps("Delete"), want: []string{`deleteDictionaryTerm term-voxt "Voxt"`}},
		{name: "dialog cancel", fixture: "en/dictionary-edit-existing", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.dictionaryUI.editOpen {
				t.Fatal("term dialog stayed open")
			}
		}},
		{name: "suggestion add", fixture: "en/dictionary-normal", steps: historySteps("Add Claude Code to the dictionary"), want: []string{"acceptDictionarySuggestion claude code"}},
		{name: "suggestion edit", fixture: "en/dictionary-normal", steps: historySteps("Edit Claude Code first, then add it"), want: []string{"editDictionarySuggestion claude code"}},
		{name: "suggestion ignore", fixture: "en/dictionary-normal", steps: historySteps("Ignore Claude Code"), want: []string{"dismissSuggestion claude code"}},
		{name: "suggestions all", fixture: "en/dictionary-suggestions-all", steps: historySteps("All (6)"), check: func(t *testing.T, host *meetingDemoHost) {
			if !host.dictionaryUI.allSuggestions {
				t.Fatal("suggestions did not expand")
			}
		}},
		{name: "scan start", fixture: "en/dictionary-scan-ready", steps: historySteps("Start"), want: []string{"startScan"}},
		{name: "scan stop", fixture: "en/dictionary-scan-running", steps: historySteps("Stop"), want: []string{"stopDictionaryScan"}},
		{name: "scan done", fixture: "en/dictionary-scan-ended", steps: historySteps("Done"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.dictionaryUI.scanOpen {
				t.Fatal("scan dialog stayed open")
			}
		}},
		{name: "scan cancel", fixture: "en/dictionary-scan-ready", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.dictionaryUI.scanOpen {
				t.Fatal("scan dialog stayed open")
			}
		}},
		{name: "notes tab", fixture: "en/dictionary-normal", steps: []historyDemoStep{{label: "Personal dictionary", click: true}}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.dictionaryUI.tab != dictNotesTab {
				t.Fatal("personal dictionary tab did not open")
			}
		}},
		{name: "notes off", fixture: "en/dictionary-notes-populated", steps: historySteps("Add to the language model prompt"), want: []string{"setDictionaryNotesOff true"}},
		{name: "notes edit and blur", fixture: "en/dictionary-notes-empty", steps: []historyDemoStep{{label: "Personal dictionary in Markdown", text: "Roth"}, {label: "Search personal dictionary", click: true}}, want: []string{"autosaveDictionaryNotes true true", "saveNotes"}},
		{name: "guide", fixture: "en/dictionary-normal", steps: historySteps("How the dictionary works"), check: func(t *testing.T, host *meetingDemoHost) {
			if !host.dictionaryUI.guideOpen {
				t.Fatal("guide did not open")
			}
		}},
		{name: "switch profile", fixture: "en/dictionary-normal", prepare: withDemoProfiles, steps: []historyDemoStep{{label: "Work", click: true}}, want: []string{"switchDictionaryProfile work"}},
		{name: "new profile", fixture: "en/dictionary-normal", prepare: withDemoProfiles, steps: []historyDemoStep{{label: "New profile"}, {label: "Profile name", text: "Home"}, {label: "Save"}}, want: []string{`openDictionaryProfileEdit ""`, `saveDictionaryProfile "Home"`}},
		{name: "rename profile", fixture: "en/dictionary-normal", prepare: withDemoProfiles, steps: []historyDemoStep{{label: "Profile actions"}, {label: "Rename “Default”…", menu: true}}, want: []string{`openDictionaryProfileEdit "default"`}, check: func(t *testing.T, host *meetingDemoHost) {
			if edit := host.dictionaryUI.profileEdit; !edit.open || edit.id != "default" || edit.name != "Default" {
				t.Fatalf("rename dialog %+v", edit)
			}
		}},
		{name: "delete profile", fixture: "en/dictionary-normal", prepare: withDemoProfiles, steps: []historyDemoStep{{label: "Profile actions"}, {label: "Delete “Default”…", menu: true}, {label: "Delete"}}, want: []string{"deleteDictionaryProfile default"}},
		{name: "load notes", fixture: "en/notes-error", steps: historySteps("Try again"), want: []string{"loadNotes"}},
		{name: "copy note", fixture: "en/notes-normal", steps: historySteps("Copy: call Ana"), want: []string{`copyNote "call Ana"`}},
		{name: "edit note", fixture: "en/notes-normal", steps: historySteps("Edit call Ana", "Save"), want: []string{`saveNoteEdit note-2 "call Ana"`}},
		{name: "edit note cancel", fixture: "en/notes-normal", steps: historySteps("Edit call Ana", "Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.notesUI.editID != "" {
				t.Fatal("note editor stayed open")
			}
		}},
		{name: "delete note", fixture: "en/notes-normal", steps: historySteps("Delete call Ana"), want: []string{`deleteDictionaryNote note-2 "call Ana"`}},
		{name: "note keys off", fixture: "en/notes-normal", steps: []historyDemoStep{{label: "Keys to record a note", click: true}, {label: "Off", click: true}}, want: []string{`setNoteKeys ""`}},
		{name: "note keys preset", fixture: "en/notes-normal", steps: []historyDemoStep{{label: "Keys to record a note", click: true}, {label: "Right ⇧", click: true}}, want: []string{`setNoteKeys "RightShift"`}},
		{name: "note keys custom", fixture: "en/notes-normal", steps: []historyDemoStep{{label: "Keys to record a note", click: true}, {key: ui.KeyEnd}, {key: ui.KeyEnter}}, want: []string{"setNoteHotkeyFocus true"}, check: func(t *testing.T, host *meetingDemoHost) {
			if !host.notesUI.keys.custom {
				t.Fatal("custom key editor did not open")
			}
		}},
		{name: "note key press", fixture: "en/notes-shortcut-custom", steps: []historyDemoStep{{key: ui.KeyK}}, want: []string{`recordNoteKeys "K"`}},
		{name: "note key draft", fixture: "en/notes-shortcut-parse-empty", steps: []historyDemoStep{{label: "Custom keys to record a note", text: "Control+Option+K"}}, want: []string{`updateNoteKeysDraft "Control+Option+K"`}},
		{name: "note key save", fixture: "en/notes-shortcut-typing", steps: []historyDemoStep{{label: "Custom keys to record a note", text: "X"}, {key: ui.KeyEnter}}, want: []string{`setNoteKeys "Control+Option+K"`, `updateNoteKeysDraft "Control+Option+KX"`, `setNoteKeys "Control+Option+KX"`}},
		{name: "type keys", fixture: "en/notes-shortcut-custom", steps: historySteps("Type the keys instead"), want: []string{"setNoteHotkeyFocus false"}, check: func(t *testing.T, host *meetingDemoHost) {
			if !host.notesUI.keys.typing {
				t.Fatal("typed key editor did not open")
			}
		}},
		{name: "record keys", fixture: "en/notes-shortcut-typing", steps: historySteps("Press the keys instead"), want: []string{`setNoteKeys "Control+Option+K"`}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.notesUI.keys.typing {
				t.Fatal("key recorder did not reopen")
			}
		}},
		{name: "record permissions", fixture: "en/notes-shortcut-record-denied", steps: historySteps("Open Permissions"), want: []string{"setNoteHotkeyFocus false", "openNotePermissions"}},
		{name: "record focus", fixture: "en/notes-shortcut-custom", steps: []historyDemoStep{{label: "Notes", click: true}, {label: "Custom keys to record a note"}}, want: []string{"setNoteHotkeyFocus false", "setNoteHotkeyFocus true"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			if test.prepare != nil {
				test.prepare(host)
			}
			tester := newMeetingDemoTester(t, host)
			tester.SetSize(windowWidth, 900) // the keys menu's last item, Custom…, is below 520 DIPs with Linux fonts
			for _, step := range test.steps {
				step.label = platformText(step.label)
				var err error
				switch {
				case step.context:
					err = tester.RightClick(step.label)
				case step.menu:
					err = tester.ChooseMenuItem(step.label)
				case step.text != "":
					err = tester.Click(step.label)
					if err == nil {
						tester.Type(step.text)
					}
				case step.command != "":
					tester.Command(step.command)
				case step.key != 0:
					tester.Key(step.mods, step.key)
				case step.click:
					err = tester.Click(step.label)
				default:
					err = tester.ClickControl(step.label)
				}
				if err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

// withDemoProfiles gives the fixture two profiles, Default active.
func withDemoProfiles(host *meetingDemoHost) {
	host.current().state.Dictionary.Profiles = []dictionaryProfileViewState{{ID: "default", Name: "Default", Active: true}, {ID: "work", Name: "Work"}}
}
