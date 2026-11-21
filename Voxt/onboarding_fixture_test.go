package main

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestOnboardingFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		fixtures := readMeetingFixtures(t, locale)
		for _, name := range []string{
			"empty-closed", "step-welcome", "step-permissions", "step-transcribe", "step-translate", "step-rewrite", "step-more",
			"models-loading", "models-normal", "models-error", "models-downloading", "models-reuse", "models-vad-only", "models-alternative",
			"try-error", "try-language-model", "try-speech-wait", "try-blocked", "try-highlights", "legacy-available", "legacy-imported", "legacy-error",
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
			if state.Try != nil {
				if state.Try.Card.State != "" {
					states[state.Try.Card.State] = true
				}
				if state.Try.Card.ProblemCode != "" {
					problems[state.Try.Card.ProblemCode] = true
					if name != "problem-"+state.Try.Card.ProblemCode {
						t.Errorf("%s/%s: problem code is %s", locale, name, state.Try.Card.ProblemCode)
					}
				}
			}
			if state.Permissions != nil && state.Permissions.Microphone.Code != "" {
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

func TestOnboardingViewStateIgnoresGlobalLocale(t *testing.T) {
	previousLocale := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	for _, step := range []int{obModels, obTranslate, obMore} {
		input := onboardingFixtureInput()
		input.Step = step
		input.Models.Found = []models.Found{{ID: models.ASRFast, Bytes: 1}}
		input.Models.Status[models.ASRAccurate] = models.Status{Err: errors.New("fixture download error")}
		input.Session = onboardingFixtureSession(session.Done)
		input.Session.Problem = &session.Problem{Code: session.ProblemNoASR, Err: errors.New("fixture problem")}
		input.Tries[step] = input.Session.ID
		input.LegacyAvailable = true
		input.Legacy.todo = []legacyTodo{{text: "Finish custom setup", kind: local.LLM, id: llm.Custom}}
		input.Conflicts = []shortcuts.Conflict{{Mine: shortcuts.Shortcut{ID: "transcribe"}, Other: shortcuts.Holder{Name: "AudioSmith"}, Theirs: shortcuts.Shortcut{ID: "transcribe"}}}
		translator := i18n.NewTranslator("en")
		localizeOnboardingViewStateInput(translator, &input)
		want := buildOnboardingViewState(translator, input)
		for _, globalLocale := range []string{"zh-Hans", "en"} {
			i18n.SetLocale(globalLocale)
			input.MoreKeys = nil
			localizeOnboardingViewStateInput(translator, &input)
			if got := buildOnboardingViewState(translator, input); !reflect.DeepEqual(got, want) {
				t.Errorf("step %d changed with global locale %s:\nwant %+v\ngot %+v", step, globalLocale, want, got)
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
