package main

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func onboardingFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, step int, mutate func(*onboardingViewStateInput)) {
		input := onboardingFixtureInput()
		input.Step = step
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "onboarding", onboarding: input})
	}
	add("empty-closed", obWelcome, func(input *onboardingViewStateInput) { input.Open = false })
	add("step-welcome", obWelcome, func(input *onboardingViewStateInput) {})
	add("step-permissions", obPermissions, func(input *onboardingViewStateInput) { input.PermissionSpeech = permission.NotDetermined })
	add("models-loading", obModels, func(input *onboardingViewStateInput) {})
	add("models-normal", obModels, func(input *onboardingViewStateInput) {
		model := onboardingFixtureModel(models.ASRAccurate)
		input.Models.InstalledASR = []local.Model{model}
		input.Models.Status[model.ID] = models.Status{Installed: true, OnDisk: true, Mine: true}
		input.Models.Language, input.Models.LanguageReady = local.Model{Kind: local.LLM, Name: "Fixture language model"}, true
	})
	add("models-error", obModels, func(input *onboardingViewStateInput) {
		input.Models.Status[models.ASRAccurate] = models.Status{Err: errors.New("fixture download failed")}
	})
	add("models-downloading", obModels, func(input *onboardingViewStateInput) {
		size := onboardingFixtureModelSize(models.ASRAccurate)
		input.Models.Status[models.ASRAccurate] = models.Status{Downloading: true, Done: size / 3, Total: size}
	})
	add("models-reuse", obModels, func(input *onboardingViewStateInput) {
		for _, id := range input.Models.PackIDs {
			input.Models.Found = append(input.Models.Found, models.Found{ID: id, Bytes: onboardingFixtureModelSize(id), By: []string{"AudioSmith"}})
			input.Models.Status[id] = models.Status{OnDisk: true}
		}
	})
	add("models-vad-only", obModels, func(input *onboardingViewStateInput) {
		input.Models.Found = []models.Found{{ID: models.VADSilero, Bytes: onboardingFixtureModelSize(models.VADSilero), By: []string{"AudioSmith"}}}
		input.Models.Status[models.VADSilero] = models.Status{OnDisk: true}
	})
	add("models-alternative", obModels, func(input *onboardingViewStateInput) {
		input.Models.Found = []models.Found{{ID: models.ASRFast, Bytes: onboardingFixtureModelSize(models.ASRFast), By: []string{"AudioSmith"}}}
		input.Models.Status[models.ASRFast] = models.Status{OnDisk: true}
	})
	for _, state := range []permission.State{permission.Unknown, permission.Denied, permission.Restricted, permission.Granted} {
		name := map[permission.State]string{permission.Unknown: "unknown", permission.Denied: "denied", permission.Restricted: "restricted", permission.Granted: "granted"}[state]
		add("permission-"+name, obPermissions, func(input *onboardingViewStateInput) {
			input.PermissionMic, input.PermissionSpeech = state, state
			input.PermissionAccess = state == permission.Granted
		})
	}
	add("step-transcribe", obTranscribe, func(input *onboardingViewStateInput) {})
	add("step-translate", obTranslate, func(input *onboardingViewStateInput) {})
	add("step-rewrite", obRewrite, func(input *onboardingViewStateInput) {})
	add("step-more", obMore, func(input *onboardingViewStateInput) {})
	add("session-idle", obTranscribe, func(input *onboardingViewStateInput) {
		input.Session = session.Snapshot{}
	})
	for _, scenario := range tryFixtureActiveStateScenarios {
		add(scenario.onboardingName, obTranscribe, func(input *onboardingViewStateInput) {
			input.Session = onboardingFixtureSession(scenario.state)
			input.Tries[obTranscribe] = input.Session.ID
		})
	}
	for _, code := range historyFixtureProblemCodes() {
		if code == session.ProblemUnsupported {
			continue
		}
		step, state, degraded := onboardingFixtureProblemScenario(code)
		add("problem-"+code, step, func(input *onboardingViewStateInput) {
			input.Session = onboardingFixtureSession(state)
			input.Session.Problem = &session.Problem{Code: code, Err: errors.New("fixture problem"), Degraded: degraded}
			switch step {
			case obTranslate:
				input.Session.Kind, input.Session.TargetLanguage = store.KindTranslate, "en"
			case obRewrite:
				input.Session.Kind = store.KindRewrite
			}
			switch code {
			case session.ProblemNoSpeech:
				input.Session.Raw, input.Session.Result, input.Session.Draft, input.Session.LLM = "", "", "", session.Model{}
				input.Session.Timings.LLM, input.Session.Timings.Total = 0, 3800*time.Millisecond
			case session.ProblemTooShort:
				input.Session.Raw, input.Session.Result, input.Session.Draft = "", "", ""
				input.Session.ASR, input.Session.LLM, input.Session.Timings = session.Model{}, session.Model{}, session.Timings{}
			case session.ProblemEmpty:
				input.Session.Raw = "make this more formal"
				input.Session.Result, input.Session.Draft = "", ""
			}
			input.Tries[step] = input.Session.ID
		})
	}
	for _, preset := range append(append([]string{}, settings.Presets...), "") {
		name := preset
		if name == "" {
			name = "custom"
		}
		add("more-preset-"+name, obMore, func(input *onboardingViewStateInput) {
			if preset == "" {
				input.Settings.Features.Transcribe.Hotkeys[0].Keys = "F13"
			} else {
				settings.ApplyPreset(&input.Settings.Features, preset)
			}
		})
	}
	add("try-no-shortcut", obTranscribe, func(input *onboardingViewStateInput) {
		input.Settings.Features.Transcribe.Hotkeys = nil
	})
	add("try-accessibility-warning", obTranscribe, func(input *onboardingViewStateInput) { input.HotkeysInstalled = false })
	add("try-passed", obTranscribe, func(input *onboardingViewStateInput) {
		input.Passed[obTranscribe] = true
		input.Session = onboardingFixtureSession(session.Done)
		input.Tries[obTranscribe] = input.Session.ID
	})
	add("try-language-model", obTranslate, func(input *onboardingViewStateInput) {})
	scenarios[len(scenarios)-1].onboardingBlocker = "language"
	add("try-speech-wait", obTranscribe, func(input *onboardingViewStateInput) {})
	scenarios[len(scenarios)-1].onboardingBlocker = "speech-wait"
	add("try-blocked", obTranscribe, func(input *onboardingViewStateInput) {})
	scenarios[len(scenarios)-1].onboardingBlocker = "blocked"
	add("try-hint", obTranscribe, func(input *onboardingViewStateInput) {})
	scenarios[len(scenarios)-1].onboardingHint = true
	add("try-shortcut-conflict", obTranscribe, func(input *onboardingViewStateInput) {})
	scenarios[len(scenarios)-1].onboardingConflict = true
	add("more-shortcut-conflict", obMore, func(input *onboardingViewStateInput) {})
	scenarios[len(scenarios)-1].onboardingConflict = true
	add("legacy-available", obWelcome, func(input *onboardingViewStateInput) { input.LegacyAvailable = true })
	add("legacy-imported", obWelcome, func(input *onboardingViewStateInput) {
		input.LegacyAvailable, input.LegacyDone = true, true
		input.Legacy = legacyReport{
			imported: []string{"Fixture dictionary"}, skipped: []string{"Fixture unsupported setting"},
			todo: []legacyTodo{{text: "Finish the speech model setup", kind: local.ASR, id: models.ASRAccurate}},
		}
	})
	add("legacy-error", obWelcome, func(input *onboardingViewStateInput) {
		input.LegacyAvailable = true
		input.Legacy.failed = "fixture import failed"
	})
	return scenarios
}

func onboardingFixtureInput() onboardingViewStateInput {
	return onboardingViewStateInput{
		Open: true, Passed: map[int]bool{}, Tries: map[int]string{}, Settings: settings.Defaults(),
		PermissionMic: permission.NotDetermined, PermissionSpeech: permission.Unknown,
		Models: onboardingModelStateInput{
			SpeechID: models.ASRAccurate, PackIDs: []string{models.ASRAccurate, models.VADSilero}, Status: map[string]models.Status{},
		},
		HotkeysInstalled: true, Now: meetingFixtureNow,
	}
}

func onboardingFixtureModel(id string) local.Model {
	return local.Model{Kind: local.ASR, ID: id, Name: onboardingModelName(id), Stream: true}
}

func onboardingFixtureModelSize(id string) int64 {
	model, _ := models.Lookup(id)
	return model.Size()
}

func onboardingFixtureProblemScenario(code string) (step int, state session.State, degraded bool) {
	step, state = obTranscribe, session.Failed
	degraded = code == session.ProblemASR || code == session.ProblemLLM
	switch code {
	case session.ProblemNoSpeech:
		state = session.Done
	case session.ProblemTooShort:
		state = session.Cancelled
	case session.ProblemNoLLM:
		step = obTranslate
	case session.ProblemEmpty:
		step = obRewrite
	case session.ProblemTruncated:
		step, degraded = obTranslate, true
	}
	if degraded {
		state = session.Done
	}
	return step, state, degraded
}

func onboardingFixtureSession(state session.State) session.Snapshot {
	snapshot := session.Snapshot{
		Rev: 1, ID: "fixture-onboarding-try", Kind: store.KindTranscribe, State: state, Try: true,
		StartedAt: meetingFixtureNow.Add(-65 * time.Second), HeardAt: meetingFixtureNow.Add(-time.Minute),
		Levels: []float64{0.1, 0.4, 0.8}, Partial: "fixture live text", Draft: "fixture streaming answer", Raw: "fixture raw speech",
		Timings: session.Timings{Audio: 3 * time.Second, ASR: 800 * time.Millisecond, LLM: 1100 * time.Millisecond, Total: 5 * time.Second},
		ASR:     session.Model{Name: "Fixture speech model"}, LLM: session.Model{Name: "Fixture language model"},
	}
	if state == session.Done {
		snapshot.Result = "Fixture cleaned result"
	}
	return snapshot
}
