package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestHomeFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		checkHomeFixtureValues(t, locale, readMeetingFixtures(t, locale))
	}
}

func checkHomeFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for _, name := range []string{
		"home-loading", "home-empty", "home-normal", "home-error", "home-error-initial", "home-reloading",
		"home-history-off", "home-history-off-empty", "home-hidden-features", "home-speech-missing",
		"home-speech-on-disk", "home-speech-downloading", "home-speech-dictation-auto", "home-speech-dictation-explicit",
		"home-permissions-unchecked", "home-access-missing", "home-shortcut-missing", "home-shortcut-waiting", "home-shortcut-carbon",
	} {
		if _, ok := fixtures[name]; !ok {
			t.Errorf("%s: missing required home fixture %s", locale, name)
		}
	}
	kinds, fixes := map[string]bool{}, map[string]bool{}
	for _, fixture := range fixtures {
		home := fixture.Home
		if home == nil {
			continue
		}
		for _, kind := range home.Today.Kinds {
			kinds[kind.Kind] = true
		}
		for _, entry := range home.Recent {
			kinds[entry.Kind] = true
		}
		for _, readiness := range append(append([]homeReadinessViewState{}, home.Missing...), home.Ready...) {
			if readiness.Fix != "" {
				fixes[readiness.Fix] = true
			}
		}
	}
	assertMeetingFixtureSet(t, locale+" home kinds", meetingTypedConstants(t, "internal/store/store.go", "Kind"), kinds)
	assertMeetingFixtureSet(t, locale+" home readiness actions", meetingNamedConstants(t, "home_actions.go", "homeFix"), fixes)

	for name := range homePermissionConstantNames(t) {
		fixture := "home-microphone-" + strings.ToLower(name)
		if _, ok := fixtures[fixture]; !ok {
			t.Errorf("%s: missing fixture %s for microphone state %s", locale, fixture, name)
		}
	}
	helperStates, helperReasons := map[string]bool{}, map[string]bool{}
	problemCodes, triggers, retryStages := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, scenario := range meetingFixtureScenarios() {
		if scenario.kind != "home" {
			continue
		}
		helperStates[string(scenario.homeReadiness.Helper.State)] = true
		if reason := scenario.homeReadiness.Helper.Reason; reason != "" {
			helperReasons[string(reason)] = true
		}
		for _, problem := range hotkey.Check(scenario.home.Features) {
			if problem.Feature == hotkey.Transcribe && problem.Index == 0 {
				problemCodes[problem.Code] = true
			}
		}
		if hotkeys := hotkey.HotkeysOf(scenario.home.Features, hotkey.Transcribe); len(hotkeys) > 0 {
			trigger := hotkeys[0].Trigger
			if trigger == "" {
				trigger = settings.TriggerTap
			}
			triggers[trigger] = true
		}
		if stage := scenario.home.Retry.stage; stage != "" {
			retryStages[string(stage)] = true
		}
	}
	assertMeetingFixtureSet(t, locale+" home helper states", meetingTypedConstants(t, "internal/helper/helper.go", "State"), helperStates)
	assertMeetingFixtureSet(t, locale+" home helper reasons", meetingTypedConstants(t, "internal/helper/helper.go", "FailReason"), helperReasons)
	requiredProblems := meetingNamedConstants(t, "internal/hotkey/check.go", "Problem")
	// Check reports duplicates and conflicts on a later binding. Home reads only
	// Transcribe's first binding, so those two codes cannot reach this view.
	delete(requiredProblems, hotkey.ProblemDuplicate)
	delete(requiredProblems, hotkey.ProblemConflict)
	assertMeetingFixtureSet(t, locale+" home shortcut problems", requiredProblems, problemCodes)
	assertMeetingFixtureSet(t, locale+" home shortcut triggers", meetingNamedConstants(t, "internal/settings/settings.go", "Trigger"), triggers)
	assertMeetingFixtureSet(t, locale+" home retry stages", map[string]bool{"recognizing": true, "composing": true}, retryStages)
}

func homePermissionConstantNames(t *testing.T) map[string]bool {
	t.Helper()
	syntax, err := parser.ParseFile(token.NewFileSet(), "internal/permission/permission.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.CONST {
			continue
		}
		permissionStates := false
		for _, rawSpec := range general.Specs {
			spec := rawSpec.(*ast.ValueSpec)
			if identifier, ok := spec.Type.(*ast.Ident); ok && identifier.Name == "State" {
				permissionStates = true
			}
			if permissionStates {
				for _, name := range spec.Names {
					names[name.Name] = true
				}
			}
		}
	}
	if len(names) == 0 {
		t.Fatal("no permission states found")
	}
	return names
}
