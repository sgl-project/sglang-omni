package main

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func dictionaryFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*dictionaryViewStateInput)) {
		input := dictionaryFixtureInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "dictionary", dictionary: input})
	}
	add("dictionary-loading", func(input *dictionaryViewStateInput) { input.Read, input.Loading = false, true })
	add("dictionary-empty", func(input *dictionaryViewStateInput) { input.Terms, input.Total = nil, 0 })
	add("dictionary-normal", func(input *dictionaryViewStateInput) {})
	add("dictionary-error", func(input *dictionaryViewStateInput) { input.Error = "fixture read failed" })
	add("dictionary-search-empty", func(input *dictionaryViewStateInput) {
		input.Query, input.ListFor, input.Terms = "missing", "missing", nil
	})
	add("dictionary-search-match", func(input *dictionaryViewStateInput) {
		input.Query, input.ListFor = "cooper", "cooper"
		input.Terms = input.Terms[1:2]
	})
	add("dictionary-off", func(input *dictionaryViewStateInput) { input.Settings.Dictionary.Enabled = false })
	add("dictionary-scan-unavailable", func(input *dictionaryViewStateInput) { input.Installed = nil })
	add("dictionary-suggestions-all", func(input *dictionaryViewStateInput) {
		input.Suggestions = nil
		for index := range 6 {
			input.Suggestions = append(input.Suggestions, learn.Suggestion{Term: fmt.Sprintf("Term%d", index), Source: learn.FromHistory})
		}
	})
	add("dictionary-profiles", func(input *dictionaryViewStateInput) { input.Profiles = dictionaryFixtureProfiles() })
	add("dictionary-profile-new", func(input *dictionaryViewStateInput) {
		input.Profiles, input.ProfileEdit = dictionaryFixtureProfiles(), dictionaryProfileEdit{open: true}
	})
	add("dictionary-profile-rename", func(input *dictionaryViewStateInput) {
		input.Profiles, input.ProfileEdit = dictionaryFixtureProfiles(), dictionaryProfileEdit{open: true, id: "work", name: "Work"}
	})
	add("dictionary-profile-error", func(input *dictionaryViewStateInput) {
		input.Profiles = dictionaryFixtureProfiles()
		input.ProfileEdit = dictionaryProfileEdit{open: true, name: "Work", err: "A profile with this name already exists."}
	})
	add("dictionary-profile-delete", func(input *dictionaryViewStateInput) {
		input.Profiles, input.ProfileDeleteOpen, input.ProfileDeleteID = dictionaryFixtureProfiles(), true, "work"
	})
	add("dictionary-edit-new", func(input *dictionaryViewStateInput) {
		input.Edit = dictionaryTermEditStateInput{Open: true, Enabled: true}
	})
	add("dictionary-edit-existing", func(input *dictionaryViewStateInput) {
		term := input.Terms[0]
		input.Edit = dictionaryTermEditStateInput{Open: true, ID: term.ID, Term: term.Term, Aliases: "vox t", Note: term.Note, Enabled: term.Enabled, Original: term.Term}
	})
	for _, scenario := range []struct {
		name string
		err  error
	}{
		{"empty", dict.ErrEmpty}, {"long", dict.ErrLong}, {"newline", dict.ErrNewline}, {"format", dict.ErrFormat},
		{"exists", core.ErrExists}, {"generic", errors.New("fixture save failed")},
	} {
		add("dictionary-edit-error-"+scenario.name, func(input *dictionaryViewStateInput) {
			input.Edit = dictionaryTermEditStateInput{Open: true, ID: "term-voxt", Term: "Voxt", Enabled: true, Original: "Voxt", Err: scenario.err}
		})
	}
	add("dictionary-scan-loading", func(input *dictionaryViewStateInput) { input.ScanOpen = true })
	add("dictionary-scan-ready", func(input *dictionaryViewStateInput) {
		input.ScanOpen, input.Scan.Counted, input.Scan.Count = true, true, 12
	})
	add("dictionary-scan-empty", func(input *dictionaryViewStateInput) {
		input.ScanOpen, input.Scan.Counted, input.Scan.Count = true, true, 0
	})
	add("dictionary-scan-running", func(input *dictionaryViewStateInput) {
		input.ScanOpen, input.Scan.Running, input.Scan.Total, input.Scan.Read, input.Scan.Found = true, true, 24, 12, 3
	})
	add("dictionary-scan-stopping", func(input *dictionaryViewStateInput) {
		input.ScanOpen, input.Scan.Running, input.Scan.Stopping, input.Scan.Total = true, true, true, 24
	})
	add("dictionary-scan-ended", func(input *dictionaryViewStateInput) {
		input.ScanOpen, input.Scan.Ended, input.Scan.Found = true, true, 4
	})
	add("dictionary-scan-stopped", func(input *dictionaryViewStateInput) {
		input.ScanOpen, input.Scan.Ended, input.Scan.Stopped, input.Scan.Found = true, true, true, 2
	})
	for _, scenario := range []struct {
		name  string
		err   error
		ended bool
	}{
		{"generic", errors.New("fixture scan failed"), false},
		{"not-installed", local.ErrNotInstalled, true},
		{"no-runtime", local.ErrNoRuntime, true},
	} {
		add("dictionary-scan-error-"+scenario.name, func(input *dictionaryViewStateInput) {
			input.ScanOpen, input.Scan.Ended, input.Scan.Err = true, scenario.ended, scenario.err
		})
	}
	for _, code := range dictionaryLLMProblemCodes() {
		add("dictionary-scan-problem-"+strings.TrimPrefix(code, "llm."), func(input *dictionaryViewStateInput) {
			input.ScanOpen, input.Scan.Ended, input.Scan.Err = true, true, &llm.Error{Code: code, Err: errors.New("fixture failure")}
		})
	}

	addNotes := func(name, text string, mutate func(*dictionaryViewStateInput)) {
		input := dictionaryFixtureInput()
		input.Tab, input.NotesText = dictNotesTab, text
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "dictionary", dictionary: input})
	}
	addNotes("dictionary-notes-empty", "", func(input *dictionaryViewStateInput) {})
	addNotes("dictionary-notes-populated", "| Standard spelling | Correction mapping |\n|---|---|\n| Roth | wrath |", func(input *dictionaryViewStateInput) {})
	addNotes("dictionary-notes-find", "| Roth | wrath |\n| NVDA | — |", func(input *dictionaryViewStateInput) {
		input.NotesQuery, input.NotesFind = "Roth", 1
	})
	addNotes("dictionary-notes-off", "Roth", func(input *dictionaryViewStateInput) { input.Settings.Dictionary.NotesOff = true })
	addNotes("dictionary-notes-disabled", "Roth", func(input *dictionaryViewStateInput) { input.Settings.Dictionary.Enabled = false })
	addNotes("dictionary-notes-long", longNotes(200), func(input *dictionaryViewStateInput) {})
	return append(scenarios, notesFixtureScenarios()...)
}

func dictionaryFixtureInput() dictionaryViewStateInput {
	preferences := settings.Defaults()
	preferences.Features.Transcribe.LLM = "fixture"
	model := local.Model{Kind: local.LLM, ID: "fixture", Name: "Fixture model"}
	return dictionaryViewStateInput{
		Terms: []store.Term{
			{ID: "term-voxt", Term: "Voxt", Aliases: []string{"vox t"}, Note: "My app", Enabled: true},
			{ID: "term-kubernetes", Term: "Kubernetes", Aliases: []string{"cooper netties", "k8s"}, Enabled: false},
		},
		Suggestions: []learn.Suggestion{
			{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: learn.FromEdit},
			{Term: "Ghostty", Source: learn.FromHistory},
		},
		Settings: preferences, Installed: []local.Model{model}, Read: true, Total: 2, LegacyImport: true,
		Scan: dictionaryScanStateInput{Model: model},
	}
}

func dictionaryFixtureProfiles() []store.DictionaryProfile {
	return []store.DictionaryProfile{{ID: "default", Name: "Default"}, {ID: "work", Name: "Work", Active: true}}
}

func dictionaryLLMProblemCodes() []string {
	return []string{
		llm.CodeNoKey, llm.CodeAuth, llm.CodeRateLimited, llm.CodeServer, llm.CodeOverloaded,
		llm.CodeBadRequest, llm.CodeModelNotFound, llm.CodeContextLength, llm.CodeFiltered,
		llm.CodeTimeout, llm.CodeCancelled, llm.CodeNetwork, llm.CodeBadResponse, llm.CodeEmpty,
		llm.CodeInvalidJSON, llm.CodeSchema, llm.CodeNoRoute, llm.CodeUnknown, llm.CodeUnsupported,
		llm.CodeConfig, llm.CodeAddress, llm.CodeResponsesOnly, llm.CodeSignIn, llm.CodeUsageLimit, llm.CodeNotEligible,
	}
}

func notesFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*notesViewStateInput)) {
		input := notesFixtureInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "notes", notes: input})
	}
	add("notes-loading", func(input *notesViewStateInput) { input.Read, input.Loading = false, true })
	add("notes-empty", func(input *notesViewStateInput) { input.Notes = nil })
	add("notes-normal", func(input *notesViewStateInput) {})
	add("notes-error", func(input *notesViewStateInput) { input.Error = "fixture read failed" })
	add("notes-edit", func(input *notesViewStateInput) {
		input.Edit = notesEditViewState{ID: "note-2", Draft: "call Ana tomorrow"}
	})
	add("notes-edit-saving", func(input *notesViewStateInput) {
		input.Edit = notesEditViewState{ID: "note-2", Draft: "call Ana tomorrow", Saving: true}
	})
	add("notes-shortcut-off", func(input *notesViewStateInput) { input.Features.Note = settings.Hotkey{} })
	add("notes-shortcut-conflict", func(input *notesViewStateInput) {
		input.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: settings.DefaultNote, Trigger: settings.TriggerTap}}
	})
	add("notes-shortcut-invalid", func(input *notesViewStateInput) { input.Features.Note.Keys = "NotAKey" })
	add("notes-shortcut-bare-key", func(input *notesViewStateInput) { input.Features.Note.Keys = "K" })
	for _, scenario := range []struct {
		name string
		keys string
	}{
		{"spotlight", "Command+Space"},
		{"finder-search", "Option+Command+Space"},
		{"input-source", "Control+Space"},
		{"app-switcher", "Command+Tab"},
		{"window-switcher", "Command+`"},
		{"editing", "Command+C"},
	} {
		add("notes-shortcut-system-"+scenario.name, func(input *notesViewStateInput) { input.Features.Note.Keys = scenario.keys })
	}
	add("notes-shortcut-custom", func(input *notesViewStateInput) {
		input.Editing.Custom, input.Editing.Text = true, settings.DefaultNote
	})
	add("notes-shortcut-typing", func(input *notesViewStateInput) {
		input.Editing.Custom, input.Editing.Typing, input.Editing.Edited, input.Editing.Text = true, true, true, "Control+Option+K"
	})
	for _, scenario := range []struct {
		name string
		text string
	}{
		{"empty", ""}, {"two-keys", "Control+A+B"}, {"fn-side", "RightFn"}, {"unknown", "NotAKey"},
	} {
		add("notes-shortcut-parse-"+scenario.name, func(input *notesViewStateInput) {
			input.Editing.Custom, input.Editing.Typing, input.Editing.Edited, input.Editing.Text = true, true, true, scenario.text
		})
	}
	add("notes-shortcut-parse-no-modifier", func(input *notesViewStateInput) {
		input.Editing.Custom, input.Editing.Typing, input.Editing.Edited, input.Editing.Err = true, true, true, hotkey.ErrNoModifier
	})
	add("notes-shortcut-record-denied", func(input *notesViewStateInput) {
		input.Editing.Custom, input.Editing.RecordDenied, input.Editing.Text = true, true, settings.DefaultNote
	})
	add("notes-shortcut-unknown", func(input *notesViewStateInput) {
		input.Editing.Custom, input.Editing.Unknown, input.Editing.Text = true, true, settings.DefaultNote
	})
	add("notes-shortcut-helper", func(input *notesViewStateInput) { input.HelperRecords = true })
	add("notes-shortcut-used-by", func(input *notesViewStateInput) { input.UsedBy = []string{"Also used by Fixture App"} })
	return scenarios
}

func notesFixtureInput() notesViewStateInput {
	return notesViewStateInput{
		Notes: []store.Note{
			{ID: "note-2", Text: "call Ana", CreatedAt: meetingFixtureNow.Add(-time.Hour), AppName: "Slack"},
			{ID: "note-1", Text: "buy milk", CreatedAt: meetingFixtureNow.Add(-2 * time.Hour)},
		},
		Read: true, Features: settings.Defaults().Features, Now: meetingFixtureNow,
		Editing: notesShortcutStateInput{Text: settings.DefaultNote},
	}
}
