package main

// Serializable display state for the Language models page and its remote
// service cards. Builders take all locale, clock and domain values as input.

import (
	"net/url"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const (
	remoteStatusNeutral = "neutral"
	remoteStatusWarning = "warning"
	remoteStatusDanger  = "danger"
	remoteStatusSuccess = "success"
)

type remoteOptionViewState struct {
	Title  string `json:"title"`
	Value  string `json:"value"`
	Custom bool   `json:"custom,omitempty"`
}

type remoteBackBarViewState struct {
	Text string `json:"text"`
}

type remoteWaitViewState struct {
	Value          string                  `json:"value"`
	Options        []remoteOptionViewState `json:"options"`
	Selected       string                  `json:"selected"`
	Custom         bool                    `json:"custom"`
	Minutes        float64                 `json:"minutes"`
	CurrentMinutes float64                 `json:"current_minutes"`
	MaxMinutes     int                     `json:"max_minutes"`
	Token          string                  `json:"token"`
}

type remoteWaitStateInput struct {
	Value          string
	Locale         string
	CurrentMinutes float64
}

type remoteViewState struct {
	BackBar *remoteBackBarViewState `json:"back_bar,omitempty"`
	Wait    remoteWaitViewState     `json:"wait"`
	Cards   remoteCardsViewState    `json:"cards"`
	Dialog  *remoteDialogViewState  `json:"dialog,omitempty"`
}

type remotePageStateInput struct {
	BackBarText string
	Wait        remoteWaitStateInput
	Cards       remoteCardsStateInput
	Dialog      *remoteDialogStateInput
}

type remoteCardsViewState struct {
	Kind   string                    `json:"kind"`
	Groups []remoteServiceGroupState `json:"groups"`
}

type remoteServiceGroupState struct {
	Key     string                    `json:"key"`
	Title   string                    `json:"title"`
	Caption string                    `json:"caption,omitempty"`
	Cards   []remoteProviderCardState `json:"cards"`
}

type remoteProviderCardState struct {
	Kind         string               `json:"kind"`
	ID           string               `json:"id"`
	Account      string               `json:"account"`
	Name         string               `json:"name"`
	Model        string               `json:"model"`
	Used         string               `json:"used,omitempty"`
	Where        string               `json:"where"`
	Realtime     bool                 `json:"realtime,omitempty"`
	Configured   bool                 `json:"configured"`
	Missing      string               `json:"missing,omitempty"`
	Status       string               `json:"status"`
	StatusText   string               `json:"status_text"`
	KeyText      string               `json:"key_text,omitempty"`
	KeyStatus    string               `json:"key_status,omitempty"`
	Test         *remoteTestViewState `json:"test,omitempty"`
	TestDisabled bool                 `json:"test_disabled,omitempty"`
}

type remoteTestViewState struct {
	State       string `json:"state"`
	Code        string `json:"code,omitempty"`
	Text        string `json:"text,omitempty"`
	LongText    string `json:"long_text,omitempty"`
	Tooltip     string `json:"tooltip,omitempty"`
	LongTooltip string `json:"long_tooltip,omitempty"`
}

type remoteCardsStateInput struct {
	Kind     local.Kind
	Services []remoteServiceStateInput
}

type remoteServiceStateInput struct {
	Service    service
	Setup      settings.Provider
	Configured bool
	Missing    string
	Used       string
	Mask       string
	KeyRead    bool
	KeySpaced  bool
	Test       providerTest
	HasTest    bool
}

func buildRemotePageViewState(translator *i18n.Translator, input remotePageStateInput) remoteViewState {
	state := remoteViewState{
		Wait:  buildRemoteWaitViewState(translator, input.Wait),
		Cards: buildRemoteCardsViewState(translator, input.Cards),
	}
	if input.BackBarText != "" {
		state.BackBar = &remoteBackBarViewState{Text: input.BackBarText}
	}
	if input.Dialog != nil {
		dialog := buildRemoteDialogViewState(translator, *input.Dialog)
		state.Dialog = &dialog
	}
	return state
}

func buildRemoteWaitViewState(translator *i18n.Translator, input remoteWaitStateInput) remoteWaitViewState {
	labels := []string{translator.T("Automatic"), translator.T("1 minute"), translator.T("2 minutes"), translator.T("5 minutes"), translator.T("10 minutes"), translator.T("No limit")}
	state := remoteWaitViewState{Value: input.Value, MaxMinutes: settings.WaitMaxMinutes, Token: input.Value + "\x00" + input.Locale}
	for index, value := range settings.WaitChoices {
		state.Options = append(state.Options, remoteOptionViewState{Title: labels[index], Value: value})
	}
	custom := translator.T("Custom…")
	state.Options = append(state.Options, remoteOptionViewState{Title: custom, Custom: true})
	selected := state.Options[0]
	for _, option := range state.Options {
		if option.Value == input.Value && !option.Custom {
			selected = option
		}
	}
	state.CurrentMinutes = input.CurrentMinutes
	if state.CurrentMinutes > 0 {
		found := false
		for _, value := range settings.WaitChoices {
			found = found || value == input.Value
		}
		if !found {
			selected = state.Options[len(state.Options)-1]
			state.Custom = true
			state.Minutes = state.CurrentMinutes
		}
	}
	state.Selected = selected.Title
	return state
}

func buildRemoteCardsViewState(translator *i18n.Translator, input remoteCardsStateInput) remoteCardsViewState {
	state := remoteCardsViewState{Kind: string(input.Kind)}
	var mine, here, online []remoteProviderCardState
	for _, serviceInput := range input.Services {
		card := buildRemoteProviderCardState(translator, serviceInput)
		switch {
		case serviceInput.Configured:
			mine = append(mine, card)
		case onThisMac(serviceInput.Service.endpoint):
			here = append(here, card)
		default:
			online = append(online, card)
		}
	}
	if input.Kind == local.ASR {
		if len(mine) > 0 {
			state.Groups = append(state.Groups, remoteServiceGroupState{Key: "remote-asr-mine", Title: translator.T("Set up"), Cards: mine})
		}
		if rest := append(here, online...); len(rest) > 0 {
			state.Groups = append(state.Groups, remoteServiceGroupState{Key: "remote-asr-online", Title: translator.T("Online services"), Caption: translator.T("A service gets your recordings only for the features you choose it for."), Cards: rest})
		}
		return state
	}
	if len(mine) > 0 {
		state.Groups = append(state.Groups, remoteServiceGroupState{Key: "remote-mine", Title: translator.T("Set up"), Cards: mine})
	}
	if len(here) > 0 {
		state.Groups = append(state.Groups, remoteServiceGroupState{Key: "remote-here", Title: translator.T("On this Mac"), Caption: translator.T("Free, and the text stays on this Mac."), Cards: here})
	}
	if len(online) > 0 {
		state.Groups = append(state.Groups, remoteServiceGroupState{Key: "remote-online", Title: translator.T("Online services"), Cards: online})
	}
	return state
}

func buildRemoteProviderCardState(translator *i18n.Translator, input remoteServiceStateInput) remoteProviderCardState {
	sv := input.Service
	account := providers.Account(sv.kind, sv.id)
	card := remoteProviderCardState{
		Kind: string(sv.kind), ID: sv.id, Account: account, Name: sv.name, Used: input.Used,
		Configured: input.Configured, Missing: input.Missing, Status: remoteStatusSuccess, StatusText: translator.T("Ready"),
	}
	card.Model = setupModel(sv.kind, input.Setup)
	if card.Model == "" {
		card.Model = translator.T("Model not chosen")
	}
	card.Where = whereTextFor(translator, sv, input.Setup)
	card.Realtime = sv.kind == local.ASR && sv.speech.Streams(setupModel(sv.kind, input.Setup))
	if !input.Configured {
		card.Status, card.StatusText = remoteStatusNeutral, translator.T("Not set up")
	}
	if input.Configured {
		switch input.Missing {
		case providers.MissingKey:
			card.Status = remoteStatusWarning
			if sv.signIn {
				card.StatusText = translator.T("Sign-in needed")
			} else {
				card.StatusText = translator.T("API key needed")
			}
		case providers.MissingEndpoint:
			card.Status, card.StatusText = remoteStatusWarning, translator.T("No address")
		case providers.MissingAddress:
			card.Status, card.StatusText = remoteStatusWarning, translator.T("Address not allowed")
		case providers.MissingModel:
			card.Status, card.StatusText = remoteStatusWarning, translator.T("No model")
		default:
			if input.HasTest && !input.Test.running && input.Test.res.Err != nil {
				card.Status, card.StatusText = remoteStatusDanger, translator.T("Test failed")
			}
		}
	}
	card.KeyText, card.KeyStatus = input.Mask, remoteStatusNeutral
	switch {
	case input.Mask == "" && !input.KeyRead:
		card.KeyText = "…"
	case input.Missing != "":
		card.KeyText, card.KeyStatus = card.StatusText, remoteStatusWarning
	case input.KeySpaced:
		card.KeyText, card.KeyStatus = translator.T("Key has a space"), remoteStatusWarning
	case input.Mask == "":
		card.KeyText, card.KeyStatus = translator.T("No key"), "muted"
	}
	if input.Configured && input.Missing == "" && input.HasTest {
		test := buildRemoteTestViewState(translator, input.Test)
		card.Test = &test
		card.TestDisabled = input.Test.running
	}
	return card
}

func buildRemoteTestViewState(translator *i18n.Translator, test providerTest) remoteTestViewState {
	state := remoteTestViewState{State: "empty"}
	switch {
	case test.running:
		state.State = "running"
		state.LongText = translator.T("Testing…")
	case test.res.At.IsZero():
	case test.res.Err == nil:
		state.State = "success"
		state.Text = translator.T("%d ms", test.res.Latency.Milliseconds())
		state.LongText = translator.T("Connection OK · %d ms", test.res.Latency.Milliseconds())
		state.Tooltip, state.LongTooltip = state.Text, state.LongText
	default:
		state.State = "failure"
		state.Code = llm.Code(test.res.Err)
		if state.Code == "" {
			state.Code = string(asr.KindOf(test.res.Err))
		}
		state.LongText, _ = uitext.ModelErrorTextFor(translator, test.res.Err)
		if models := test.res.Models; len(models) > 0 {
			if len(models) > 8 {
				models = append(models[:8:8], "…")
			}
			state.LongText = translator.T("The server does not have this model. It has: %s", strings.Join(models, ", "))
		}
		if llm.Code(test.res.Err) == llm.CodeNoKey {
			state.LongText = translator.T("Paste the service's API key above, then test again.")
		}
		state.Text = translator.T("Failed")
		state.Tooltip = state.LongText + "\n\n" + test.detail
		state.LongTooltip = state.Tooltip
	}
	return state
}

// whereTextFor is a service's label: on this Mac (a loopback server) or remote.
func whereTextFor(translator *i18n.Translator, sv service, p settings.Provider) string {
	endpoint := strings.TrimSpace(p.Endpoint)
	if endpoint == "" {
		endpoint = sv.endpoint
	}
	if onThisMac(endpoint) {
		return translator.T("On this Mac")
	}
	return translator.T("Remote")
}

// onThisMac says whether a service's address is a server on this Mac.
func onThisMac(base string) bool {
	parsed, err := url.Parse(base)
	return err == nil && parsed.Host != "" && core.IsLoopback(parsed.Hostname())
}
