package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestOnboardingFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		fixtures := readMeetingFixtures(t, locale)
		for _, name := range []string{
			"empty-closed", "step-welcome", "step-permissions", "step-transcribe", "step-translate", "step-rewrite", "step-more",
			"models-loading", "models-normal", "models-error", "models-downloading", "models-reuse", "models-vad-only", "models-alternative",
			"try-error", "try-language-model", "try-speech-wait", "try-blocked", "legacy-available", "legacy-imported", "legacy-error",
		} {
			if _, ok := fixtures[name]; !ok {
				t.Errorf("%s: missing required onboarding fixture %s", locale, name)
			}
		}
		steps := map[int]bool{}
		states := map[string]bool{}
		problems := map[string]bool{}
		permissions := map[string]bool{}
		for name, fixture := range fixtures {
			state := fixture.Onboarding
			if state == nil {
				continue
			}
			steps[state.Step] = true
			if state.Try.Card.State != "" {
				states[state.Try.Card.State] = true
			}
			if state.Try.Card.ProblemCode != "" {
				problems[state.Try.Card.ProblemCode] = true
				if name != "problem-"+state.Try.Card.ProblemCode {
					t.Errorf("%s/%s: problem code is %s", locale, name, state.Try.Card.ProblemCode)
				}
			}
			if state.Permissions.Microphone.Code != "" {
				permissions[state.Permissions.Microphone.Code] = true
			}
		}
		for step := 0; step < obSteps; step++ {
			if !steps[step] {
				t.Errorf("%s: missing fixture for onboarding step %d", locale, step)
			}
		}
		if len(steps) != obSteps {
			t.Errorf("%s: onboarding steps %v, want %d values", locale, steps, obSteps)
		}
		assertMeetingFixtureSet(t, locale+" onboarding session states", meetingTypedConstants(t, "internal/session/session.go", "State"), states)
		assertMeetingFixtureSet(t, locale+" onboarding problem codes", meetingNamedConstants(t, "internal/session/session.go", "Problem"), problems)
		assertMeetingFixtureSet(t, locale+" onboarding permission states", onboardingPermissionStateCodes(t), permissions)
		for _, preset := range append(append([]string{}, settings.Presets...), "") {
			name := preset
			if name == "" {
				name = "custom"
			}
			if _, ok := fixtures["more-preset-"+name]; !ok {
				t.Errorf("%s: missing fixture for shortcut preset %q", locale, preset)
			}
		}
	}
}

func onboardingPermissionStateCodes(t *testing.T) map[string]bool {
	t.Helper()
	syntax, err := parser.ParseFile(token.NewFileSet(), "internal/permission/permission.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	codes := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.CONST {
			continue
		}
		stateConstants := false
		for _, rawSpec := range general.Specs {
			spec := rawSpec.(*ast.ValueSpec)
			if spec.Type != nil {
				identifier, ok := spec.Type.(*ast.Ident)
				stateConstants = ok && identifier.Name == "State"
			}
			if !stateConstants {
				continue
			}
			for _, name := range spec.Names {
				codes[strings.ToLower(name.Name[:1])+name.Name[1:]] = true
			}
		}
	}
	if len(codes) == 0 {
		t.Fatal("internal/permission/permission.go: no State constants found")
	}
	return codes
}
