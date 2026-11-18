package main

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"strings"
	"testing"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestDictionaryFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		translator := i18n.NewTranslator(locale)
		fixtures := readMeetingFixtures(t, locale)
		sources, stages, editorErrors, scanErrors := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
		tabs, noteProblems, noteSystems, parseErrors := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
		stopping := false
		for _, fixture := range fixtures {
			if state := fixture.Dictionary; state != nil {
				tabs[strconv.Itoa(state.Tab)] = true
				for _, suggestion := range state.Suggestions {
					sources[suggestion.SourceCode] = true
				}
				if state.ScanOpen {
					stages[state.Scan.Stage] = true
					stopping = stopping || state.Scan.StopDisabled
					if state.Scan.ErrorCode != "" {
						scanErrors[state.Scan.ErrorCode] = true
					}
				}
				if state.Edit.ErrorCode != "" {
					editorErrors[state.Edit.ErrorCode] = true
				}
			}
			if state := fixture.Notes; state != nil {
				for _, code := range state.Shortcut.ProblemCodes {
					noteProblems[code] = true
				}
				for _, code := range state.Shortcut.SystemCodes {
					noteSystems[code] = true
				}
				if state.Shortcut.ParseCode != "" {
					parseErrors[state.Shortcut.ParseCode] = true
				}
				if !state.Shortcut.Custom {
					wantSelection := notesChordLabelFor(translator, state.Shortcut.Keys)
					if state.Shortcut.Keys == "" {
						wantSelection = translator.T("Off")
					}
					if state.Shortcut.Selection != wantSelection || state.Shortcut.Text != state.Shortcut.Keys {
						t.Errorf("%s: shortcut selection %q and text %q do not match keys %q", locale, state.Shortcut.Selection, state.Shortcut.Text, state.Shortcut.Keys)
					}
				}
			}
		}
		assertMeetingFixtureSet(t, locale+" dictionary suggestion sources", meetingTypedConstants(t, "internal/learn/learn.go", "Source"), sources)
		assertMeetingFixtureSet(t, locale+" dictionary scan stages", stringSet("loading", "ready", "empty", "running", "ended", "error"), stages)
		if !stopping {
			t.Errorf("%s: no stopping scan fixture", locale)
		}
		assertMeetingFixtureSet(t, locale+" dictionary tabs", stringSet("0", "1"), tabs)

		requiredEditorErrors := requiredDictionaryEditorErrorCodes(t)
		assertMeetingFixtureSet(t, locale+" dictionary editor errors", requiredEditorErrors, editorErrors)

		requiredScanErrors := meetingNamedConstants(t, "internal/llm/errors.go", "Code")
		for code := range stringSet("generic", "not_installed", "no_runtime") {
			requiredScanErrors[code] = true
		}
		assertMeetingFixtureSet(t, locale+" dictionary scan problem codes", requiredScanErrors, scanErrors)

		requiredNoteProblems := meetingNamedConstants(t, "internal/hotkey/check.go", "Problem")
		if len(hotkey.HotkeysOf(settings.Defaults().Features, hotkey.Note)) != 1 {
			t.Fatal("Notes now has multiple bindings: add its duplicate-problem fixture")
		}
		delete(requiredNoteProblems, hotkey.ProblemDuplicate) // Notes has one binding, so duplicate is unreachable.
		assertMeetingFixtureSet(t, locale+" notes problem codes", requiredNoteProblems, noteProblems)
		assertMeetingFixtureSet(t, locale+" notes system problem codes", meetingNamedConstants(t, "internal/hotkey/check.go", "System"), noteSystems)
		assertMeetingFixtureSet(t, locale+" notes parse errors", stringSet("empty", "two_keys", "fn_side", "unknown"), parseErrors)
	}
}

func TestDictionaryValidationErrorFixtures(t *testing.T) {
	actual := map[string]bool{}
	for _, scenario := range dictionaryFixtureScenarios() {
		if scenario.kind != "dictionary" {
			continue
		}
		state := buildDictionaryViewState(i18n.NewTranslator("en"), scenario.dictionary)
		if state.Edit.ErrorCode != "" {
			actual[state.Edit.ErrorCode] = true
		}
	}
	assertMeetingFixtureSet(t, "dictionary validation error scenarios", requiredDictionaryEditorErrorCodes(t), actual)
}

func requiredDictionaryEditorErrorCodes(t *testing.T) map[string]bool {
	t.Helper()
	codes := stringSet("exists", "generic")
	for name := range dictionaryDeclarationNames(t, "internal/dict/dict.go", "Err") {
		switch name {
		case "ErrNewline", "ErrFormat":
			// Clean removes newlines before submitTerm calls Check, and ErrFormat belongs to Import rather than the edit dialog.
			continue
		default:
			codes[dictionaryErrorCodeFromDeclaration(name)] = true
		}
	}
	return codes
}

func dictionaryErrorCodeFromDeclaration(declaration string) string {
	var code strings.Builder
	for index, character := range strings.TrimPrefix(declaration, "Err") {
		if unicode.IsUpper(character) && index > 0 {
			code.WriteByte('_')
		}
		code.WriteRune(unicode.ToLower(character))
	}
	return code.String()
}

func TestDictionaryErrorFixtureScenariosAreReachable(t *testing.T) {
	scenarios := dictionaryFixtureScenarios()
	byName := make(map[string]meetingFixtureScenario, len(scenarios))
	for _, scenario := range scenarios {
		byName[scenario.name] = scenario
		if strings.HasPrefix(scenario.name, "notes-shortcut-parse-") && scenario.notes.Editing.Err != nil {
			t.Errorf("%s injects %v instead of parsing its input", scenario.name, scenario.notes.Editing.Err)
		}
	}
	for _, test := range []struct {
		name     string
		code     string
		id       string
		term     string
		original string
		typedErr error
	}{
		{name: "empty", code: "empty", typedErr: dict.ErrEmpty},
		{name: "long", code: "long", term: strings.Repeat("R", dict.MaxTermChars+1), typedErr: dict.ErrLong},
		{name: "exists", code: "exists", term: "voxt", typedErr: core.ErrExists},
		{name: "generic", code: "generic", id: "term-voxt", term: "Voxt", original: "Voxt"},
	} {
		scenario, ok := byName["dictionary-edit-error-"+test.name]
		if !ok {
			t.Errorf("missing %s edit-error scenario", test.name)
			continue
		}
		edit := scenario.dictionary.Edit
		if !edit.Open || edit.ID != test.id || edit.Term != test.term || edit.Original != test.original || edit.Err == nil {
			t.Errorf("%s: unreachable edit input %+v", test.name, edit)
			continue
		}
		if test.typedErr != nil && !errors.Is(edit.Err, test.typedErr) {
			t.Errorf("%s: error %v does not wrap %v", test.name, edit.Err, test.typedErr)
		}
		state := buildDictionaryViewState(i18n.NewTranslator("en"), scenario.dictionary)
		if state.Edit.ErrorCode != test.code {
			t.Errorf("%s: error code %q, want %q", test.name, state.Edit.ErrorCode, test.code)
		}
	}
	for _, name := range []string{"dictionary-edit-error-newline", "dictionary-edit-error-format", "notes-shortcut-parse-no-modifier"} {
		if _, ok := byName[name]; ok {
			t.Errorf("%s remains in unreachable dialog coverage", name)
		}
	}
}

func stringSet(values ...string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}

func dictionaryDeclarationNames(t *testing.T, path, prefix string) map[string]bool {
	t.Helper()
	syntax, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.CONST && general.Tok != token.VAR {
			continue
		}
		for _, rawSpec := range general.Specs {
			spec := rawSpec.(*ast.ValueSpec)
			for _, name := range spec.Names {
				if strings.HasPrefix(name.Name, prefix) {
					names[name.Name] = true
				}
			}
		}
	}
	if len(names) == 0 {
		t.Fatalf("%s: no declarations found", path)
	}
	return names
}
