package main

import (
	"errors"
	"io/fs"

	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func modelsFixtureInput() modelsViewStateInput {
	return modelsViewStateInput{
		Catalog: models.Catalog(),
		Status: map[string]models.Status{
			models.ASRAccurate:  {Installed: true, OnDisk: true, Mine: true, Users: []string{"AudioSmith"}},
			models.VADSilero:    {Installed: true, OnDisk: true, Mine: true},
			models.LLMTranslate: {Installed: true, OnDisk: true, Mine: true},
		},
		Memory:   map[string]int64{models.ASRAccurate: 2_469_606_195},
		Settings: settings.Defaults(),
		InstalledASR: []local.Model{
			{Kind: local.ASR, ID: models.ASRAccurate, Name: "Qwen3-ASR 1.7B (8bit)", Stream: true},
			{Kind: local.ASR, ID: dictation.ModelID, Name: "System dictation", Stream: true},
		},
	}
}

func modelsFixtureModelInput(model models.Model) modelsViewStateInput {
	input := modelsViewStateInput{Catalog: []models.Model{model}, Status: map[string]models.Status{}, Settings: settings.Defaults()}
	status := models.Status{}
	if model.Retired {
		status.Installed, status.OnDisk = true, true
	}
	if model.Borrowed {
		status.OnDisk = true
		status.Users = []string{"AudioSmith"}
	}
	if model.Kind == models.Draft {
		for _, parent := range models.Catalog() {
			if parent.Draft == model.ID {
				input.Catalog = []models.Model{parent, model}
				input.Status[parent.ID] = models.Status{Installed: true, OnDisk: true}
			}
		}
		status.OnDisk = true
	}
	input.Status[model.ID] = status
	return input
}

func modelsFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*modelsViewStateInput)) {
		input := modelsFixtureInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "models", models: input})
	}
	addInput := func(name string, input modelsViewStateInput) {
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "models", models: input})
	}

	empty := modelsViewStateInput{Settings: settings.Defaults()}
	addInput("models-page-empty", empty)
	add("models-page-loading", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.ASRFast)
		input.Catalog = []models.Model{model}
		input.Status = map[string]models.Status{models.ASRFast: {Downloading: true, Done: model.Size() / 4, Total: model.Size()}}
		input.InstalledASR, input.Memory = nil, nil
	})
	add("models-page-normal", func(input *modelsViewStateInput) {})
	add("models-page-error", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.ASRFast)
		input.Catalog = []models.Model{model}
		input.Status = map[string]models.Status{models.ASRFast: {Err: hfhub.ErrDiskSpace}}
		input.InstalledASR, input.Memory = nil, nil
	})

	for _, model := range models.Catalog() {
		for _, value := range []struct {
			prefix  string
			name    string
			kind    models.Kind
			runtime models.Runtime
		}{
			{prefix: "kind", name: string(model.Kind), kind: model.Kind},
			{prefix: "runtime", name: string(model.Runtime), runtime: model.Runtime},
		} {
			seen := false
			for _, scenario := range scenarios {
				if scenario.name == "models-"+value.prefix+"-"+value.name {
					seen = true
					break
				}
			}
			if !seen {
				addInput("models-"+value.prefix+"-"+value.name, modelsFixtureModelInput(model))
			}
		}
	}

	fast, _ := models.Lookup(models.ASRFast)
	addInput("models-download-fresh", modelsFixtureModelInput(fast))
	add("models-download-progress", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.ASRFast)
		input.Status[models.ASRFast] = models.Status{Downloading: true, Done: model.Size() / 4, Total: model.Size()}
	})
	add("models-download-waiting", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.ASRFast)
		input.Status[models.ASRFast] = models.Status{Downloading: true, Done: model.Size() / 4, Total: model.Size(), WaitingFor: audioSmithID}
	})
	add("models-download-stopped", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.ASRFast)
		input.Status[models.ASRFast] = models.Status{Done: model.Size() / 4, Total: model.Size()}
	})
	add("models-download-elsewhere", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.ASRFast)
		input.Status[models.ASRFast] = models.Status{Done: model.Size() / 4, Total: model.Size(), Elsewhere: audioSmithID}
	})
	add("models-download-deleted", func(input *modelsViewStateInput) {
		input.Status[models.ASRFast] = models.Status{DeletedBy: dictaID}
	})
	add("models-download-on-disk", func(input *modelsViewStateInput) {
		input.Status[models.ASRFast] = models.Status{OnDisk: true, Users: []string{"AudioSmith"}}
	})
	add("models-download-loaded-unmeasured", func(input *modelsViewStateInput) {
		input.Memory[models.ASRAccurate] = 0
	})
	add("models-download-partial-retired", func(input *modelsViewStateInput) {
		model, _ := models.Lookup(models.LLMFast)
		input.Status[models.LLMFast] = models.Status{Done: model.Size() / 5, Total: model.Size()}
	})
	add("models-borrowed-on-disk", func(input *modelsViewStateInput) {
		input.Status[models.LLMQwen3] = models.Status{OnDisk: true, Users: []string{"Dicta"}}
	})
	add("models-borrowed-installed", func(input *modelsViewStateInput) {
		input.Status[models.LLMQwen3] = models.Status{Installed: true, OnDisk: true, Users: []string{"Dicta"}}
	})
	add("models-deciding-delete", func(input *modelsViewStateInput) {
		input.ModelDeciding = models.ASRAccurate
	})

	for _, scenario := range []struct {
		name string
		err  error
	}{
		{"disk-space", hfhub.ErrDiskSpace},
		{"checksum", hfhub.ErrChecksum},
		{"not-found", hfhub.ErrNotFound},
		{"gated", hfhub.ErrGated},
		{"offline", hfhub.ErrOffline},
		{"busy", &fammodels.BusyError{}},
		{"path", &fs.PathError{Op: "write", Path: "fixture", Err: errors.New("fixture write failed")}},
		{"generic", errors.New("fixture download failed")},
	} {
		add("models-error-"+scenario.name, func(input *modelsViewStateInput) {
			input.Status[models.ASRFast] = models.Status{Err: scenario.err}
		})
	}

	add("models-draft-on-missing", func(input *modelsViewStateInput) {
		input.Settings.Models.Faster = map[string]bool{models.ASRAccurate: true}
	})
	add("models-draft-off-installed", func(input *modelsViewStateInput) {
		input.Status[models.ASRAccurateDraft] = models.Status{Installed: true, OnDisk: true}
		input.Settings.Models.Faster = map[string]bool{models.ASRAccurate: false}
	})
	add("models-draft-default-installed", func(input *modelsViewStateInput) {
		input.Status[models.ASRAccurateDraft] = models.Status{Installed: true, OnDisk: true}
	})
	add("models-draft-alone", func(input *modelsViewStateInput) {
		delete(input.Status, models.ASRAccurate)
		input.Status[models.ASRAccurateDraft] = models.Status{OnDisk: true}
	})
	add("models-hidden-open", func(input *modelsViewStateInput) {
		input.Settings.Models.Hidden = []string{models.ASRFast}
		input.HiddenOpen = true
	})
	add("models-hidden-retired-legacy", func(input *modelsViewStateInput) {
		input.Settings.Models.Hidden = []string{models.LLMTranslate}
	})
	add("models-main-language-unsupported", func(input *modelsViewStateInput) {
		input.Settings.General.MainLanguage = "zz"
	})

	for _, value := range settings.KeepChoices {
		name := value
		if name == "" {
			name = "auto"
		}
		add("models-keep-"+name, func(input *modelsViewStateInput) {
			input.Settings.Models.KeepLoaded = value
			input.AdvancedOpen = true
		})
	}
	add("models-keep-unknown", func(input *modelsViewStateInput) {
		input.Settings.Models.KeepLoaded = "unknown"
		input.AdvancedOpen = true
	})

	add("models-delete-plain", func(input *modelsViewStateInput) {
		input.ModelAsk = models.ASRAccurate
	})
	add("models-delete-shared", func(input *modelsViewStateInput) {
		input.ModelShared = &sharedDelete{id: models.ASRAccurate, apps: []string{"AudioSmith", "Dicta"}, bytes: 2_469_606_195}
	})
	add("models-delete-all", func(input *modelsViewStateInput) {
		input.ModelShared = &sharedDelete{id: models.ASRAccurate, apps: []string{"AudioSmith", "Dicta"}, bytes: 2_469_606_195, forAll: true}
	})

	add("models-network-automatic", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
	})
	add("models-network-no-proxy", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.NetworkForm.proxyMode = proxyNone
	})
	add("models-network-custom-validation", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.NetworkForm.proxyMode = proxyCustom
		input.NetworkForm.proxyURL = "not-a-proxy"
	})
	scenarios[len(scenarios)-1].modelsProblem = "proxy"
	add("models-network-password-validation", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.NetworkForm.proxyMode = proxyCustom
		input.NetworkForm.proxyURL = "http://fixture"
		input.NetworkForm.proxyPassword = "bad\"password"
	})
	scenarios[len(scenarios)-1].modelsProblem = "password"
	for index, name := range []string{"official", "mirror"} {
		add("models-network-source-"+name, func(input *modelsViewStateInput) {
			input.NetworkOpen = true
			input.NetworkForm.source = index + 1
		})
	}
	add("models-network-source-custom", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.NetworkForm.source = len(networkSources)
		input.NetworkForm.sourceURL = "fixture-source"
	})
	add("models-network-follows", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.FamilyOthers = true
	})
	add("models-network-own", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.FamilyOthers = true
		input.Settings.Family.Follow = map[string]bool{actnow.KeyProxy: false}
	})
	add("models-network-conflict", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.FamilyOthers = true
		input.Settings.Family.Follow = map[string]bool{actnow.KeyProxy: false}
		input.Settings.Family.Conflicts = []string{actnow.KeyProxy}
	})
	add("models-network-readonly", func(input *modelsViewStateInput) {
		input.NetworkOpen = true
		input.FamilyReadOnly = true
	})

	assign := func(id string, kind local.Kind, selected [4]bool) modelAssignmentStateInput {
		input := modelAssignmentStateInput{Kind: kind, ID: id, Selected: selected, Settings: settings.Defaults()}
		switch kind {
		case local.ASR:
			input.Installed = []local.Model{
				{Kind: local.ASR, ID: models.ASRAccurate, Name: "Qwen3-ASR 1.7B (8bit)", Stream: true},
				{Kind: local.ASR, ID: dictation.ModelID, Name: "System dictation", Stream: true},
			}
		default:
			input.Installed = []local.Model{
				{Kind: local.LLM, ID: "translate-only", Name: "Translation only", Features: []string{hotkey.Translate}},
			}
		}
		return input
	}
	add("models-assign-asr", func(input *modelsViewStateInput) {
		assignment := assign(models.ASRAccurate, local.ASR, [4]bool{true, false, true, false})
		input.Assignment = &assignment
	})
	add("models-assign-llm", func(input *modelsViewStateInput) {
		assignment := assign("translate-only", local.LLM, [4]bool{false, true, false, false})
		input.Assignment = &assignment
	})
	add("models-assign-remote", func(input *modelsViewStateInput) {
		assignment := assign(models.ASRAccurate, local.ASR, [4]bool{true, true, true, true})
		assignment.Remote = true
		input.Assignment = &assignment
	})
	add("models-assign-unavailable", func(input *modelsViewStateInput) {
		assignment := assign("asr.gone", local.ASR, [4]bool{true, false, false, false})
		input.Assignment = &assignment
	})
	add("models-assign-cleanup-off", func(input *modelsViewStateInput) {
		assignment := assign("llm.all", local.LLM, [4]bool{true, false, false, false})
		assignment.Installed = []local.Model{{Kind: local.LLM, ID: "llm.all", Name: "Fixture language model"}}
		assignment.Settings.Features.Transcribe.Enhance = false
		input.Assignment = &assignment
	})

	return scenarios
}
