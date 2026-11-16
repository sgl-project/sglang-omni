package main

import (
	"fmt"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type settingsAboutRowViewState struct {
	Title string `json:"title"`
	Value string `json:"value"`
}

type settingsAboutCheckViewState struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Title  string `json:"title"`
	Detail string `json:"detail"`
}

type settingsAboutViewState struct {
	Rows         []settingsAboutRowViewState   `json:"rows"`
	SelfChecking bool                          `json:"self_checking"`
	Checks       []settingsAboutCheckViewState `json:"checks"`
}

type settingsAboutStateInput struct {
	Version       string
	HelperVersion string
	SystemInfo    *sysInfo
	SelfChecking  bool
	Checks        []selfCheck
}

func buildSettingsAboutViewState(translator *i18n.Translator, input settingsAboutStateInput) settingsAboutViewState {
	version := input.Version
	if version == "" {
		version = translator.T("development build")
	}
	state := settingsAboutViewState{
		Rows: []settingsAboutRowViewState{
			{Title: "Voxt", Value: version},
			{Title: translator.T("Helper"), Value: input.HelperVersion},
			{Title: translator.T("Helper protocol"), Value: fmt.Sprint(proto.Version)},
		},
		SelfChecking: input.SelfChecking,
	}
	if system := input.SystemInfo; system != nil {
		state.Rows = append(state.Rows,
			settingsAboutRowViewState{Title: translator.T("System"), Value: uitext.OrDash(system.OS)},
			settingsAboutRowViewState{Title: translator.T("Chip"), Value: uitext.OrDash(system.Chip)},
			settingsAboutRowViewState{Title: translator.T("Memory"), Value: translator.T("%d GB", system.MemoryGB())},
		)
	}
	for _, check := range input.Checks {
		state.Checks = append(state.Checks, settingsAboutCheckViewState{
			ID: check.id, Status: check.status, Title: check.title, Detail: check.detail,
		})
	}
	return state
}
