package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestRemoteFixtureCoverage(t *testing.T) {
	requiredScenarios := []string{
		"remote-page-empty", "remote-page-loading", "remote-page-normal", "remote-page-error",
		"remote-dialog-empty", "remote-dialog-loading", "remote-dialog-normal", "remote-dialog-error",
		"remote-debug-empty", "remote-debug-loading", "remote-debug-transcribe", "remote-debug-error",
		"remote-wait-custom", "remote-dialog-speech-realtime", "remote-dialog-speech-upload", "remote-cards-normal",
	}
	for _, locale := range i18n.Supported() {
		fixtures := readMeetingFixtures(t, locale)
		for _, name := range requiredScenarios {
			if _, ok := fixtures[name]; !ok {
				t.Errorf("%s: missing required remote fixture %s", locale, name)
			}
		}
		cardKinds, dialogKinds, missing := map[string]bool{}, map[string]bool{}, map[string]bool{}
		llmProblems, speechProblems, chatProblems := map[string]bool{}, map[string]bool{}, map[string]bool{}
		thinking, efforts, formats, verbositySet, apis, waits := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
		translator := i18n.NewTranslator(locale)
		for _, fixture := range fixtures {
			if fixture.Remote != nil {
				remote := fixture.Remote
				cardKinds[remote.Cards.Kind] = true
				for _, option := range remote.Wait.Options {
					if option.Title == remote.Wait.Selected {
						if option.Custom {
							waits[remote.Wait.Value] = true
						} else {
							waits[option.Value] = true
						}
					}
				}
				for _, group := range remote.Cards.Groups {
					for _, card := range group.Cards {
						if card.Missing != "" {
							missing[card.Missing] = true
						}
						if card.Test != nil && card.Test.Code != "" {
							llmProblems[card.Test.Code] = true
						}
					}
				}
			}
			if fixture.RemoteCards != nil {
				cardKinds[fixture.RemoteCards.Kind] = true
			}
			if fixture.RemoteDialog != nil {
				dialog := fixture.RemoteDialog
				dialogKinds[dialog.Kind] = true
				if dialog.Test.Code != "" {
					if dialog.Kind == "asr" {
						speechProblems[dialog.Test.Code] = true
					} else {
						llmProblems[dialog.Test.Code] = true
					}
				}
				if dialog.ChatGPT != nil && dialog.ChatGPT.Code != "" {
					chatProblems[dialog.ChatGPT.Code] = true
				}
				if dialog.Generation != nil {
					generation := dialog.Generation
					thinking[generation.Thinking] = true
					if generation.Thinking == llm.ThinkEffort {
						efforts[generation.Form.Effort] = true
					}
					formats[valueOf(generation.Form.Format, responseFormats, func(value string) string { return formatTitleFor(translator, value) })] = true
					verbositySet[valueOf(generation.Form.Verbosity, verbosities, func(value string) string { return verbosityTitleFor(translator, value) })] = true
					apis[valueOf(generation.Form.API, apiFormats, apiFormatTitle)] = true
				}
			}
		}
		if got, want := fixtures["remote-page-normal"].Remote.Cards.Groups[0].Cards[0].Used, translator.T("Used by %s", translator.T("Rewrite")); got != want {
			t.Errorf("%s: used-by text %q, want %q", locale, got, want)
		}
		if got, want := fixtures["remote-onboarding-back"].Remote.BackBar.Text, translator.T("The setup guide is paused at the %s step.", translator.T("Models")); got != want {
			t.Errorf("%s: onboarding text %q, want %q", locale, got, want)
		}
		keyDialog := fixtures["remote-dialog-key-spaced"].RemoteDialog
		if keyDialog.Key.Warning == "" || keyDialog.Form.TypedKeySpaces != 1 {
			t.Errorf("%s: key whitespace was not preserved in the fixture: %+v", locale, keyDialog.Key)
		}
		for _, problem := range []struct {
			name string
			kind asr.Kind
		}{{"remote-speech-problem-auth", asr.KindAuth}, {"remote-speech-problem-quota", asr.KindQuota}} {
			got := fixtures[problem.name].RemoteDialog.Test.LongText
			want, _ := uitext.ModelErrorTextFor(translator, &asr.Error{Kind: problem.kind, Provider: providers.StepFunASR})
			if got != want {
				t.Errorf("%s: %s text %q, want %q", locale, problem.name, got, want)
			}
		}
		if !fixtures["remote-cards-normal"].RemoteCards.Groups[0].Cards[0].Realtime || fixtures["remote-cards-upload"].RemoteCards.Groups[0].Cards[0].Realtime {
			t.Errorf("%s: speech card realtime states are wrong", locale)
		}
		assertMeetingFixtureSet(t, locale+" remote card kinds", meetingTypedConstants(t, "internal/local/local.go", "Kind"), cardKinds)
		assertMeetingFixtureSet(t, locale+" remote dialog kinds", meetingTypedConstants(t, "internal/local/local.go", "Kind"), dialogKinds)
		assertMeetingFixtureSet(t, locale+" remote missing codes", meetingNamedConstants(t, "internal/providers/providers.go", "Missing"), missing)
		assertMeetingFixtureSet(t, locale+" remote language problem codes", meetingNamedConstants(t, "internal/llm/errors.go", "Code"), llmProblems)
		assertMeetingFixtureSet(t, locale+" remote speech problem codes", meetingTypedConstants(t, "internal/asr/errors.go", "Kind"), speechProblems)
		assertMeetingFixtureSet(t, locale+" remote sign-in problem codes", meetingTypedConstants(t, "internal/codexauth/session.go", "Kind"), chatProblems)
		assertMeetingFixtureSet(t, locale+" remote thinking modes", meetingNamedConstants(t, "internal/llm/generation.go", "Think"), thinking)
		assertMeetingFixtureSet(t, locale+" remote efforts", remoteStringSet(providers.Capabilities(llm.Custom, "").Efforts), efforts)
		requiredFormats := remoteStringSet(responseFormats)
		assertMeetingFixtureSet(t, locale+" remote response formats", requiredFormats, formats)
		assertMeetingFixtureSet(t, locale+" remote verbosity", remoteStringSet(verbosities), verbositySet)
		assertMeetingFixtureSet(t, locale+" remote API formats", remoteStringSet(apiFormats), apis)
		requiredWaits := remoteStringSet(settings.WaitChoices)
		requiredWaits["45m"] = true
		assertMeetingFixtureSet(t, locale+" remote waits", requiredWaits, waits)
	}
}

func TestRemoteGenerationUsesExplicitLocale(t *testing.T) {
	previous := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previous) })
	translator := i18n.NewTranslator("zh-Hans")
	form := genForm{think: thinkTitleFor(translator, llm.ThinkOn), format: formatTitleFor(translator, llm.ResponseJSONSchema)}
	i18n.SetLocale("en")
	state := buildRemoteGenerationViewState(translator, form, remoteFixtureCaps())
	if state.Thinking != llm.ThinkOn {
		t.Fatalf("thinking %q with an explicit Chinese translator and English global locale", state.Thinking)
	}
}

func remoteStringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}
