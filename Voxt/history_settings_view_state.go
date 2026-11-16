package main

import (
	"slices"
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type historyChoiceViewState struct {
	Label string `json:"label"`
	Value int    `json:"value"`
}

type historySettingsViewState struct {
	Open            bool                     `json:"open"`
	Directory       string                   `json:"directory"`
	MissingNotice   string                   `json:"missing_notice"`
	CanUseDefault   bool                     `json:"can_use_default"`
	UsageText       string                   `json:"usage_text"`
	CanDelete       bool                     `json:"can_delete"`
	CanExport       bool                     `json:"can_export"`
	RecordingScopes []string                 `json:"recording_scopes"`
	RecordingScope  string                   `json:"recording_scope"`
	TextScopes      []string                 `json:"text_scopes"`
	TextScope       string                   `json:"text_scope"`
	TextFormats     []string                 `json:"text_formats"`
	TextFormat      string                   `json:"text_format"`
	TextBusy        bool                     `json:"text_busy"`
	Busy            bool                     `json:"busy"`
	BusyLabel       string                   `json:"busy_label"`
	Progress        float64                  `json:"progress"`
	CanCancel       bool                     `json:"can_cancel"`
	CleanOn         bool                     `json:"clean_on"`
	DayOptions      []historyChoiceViewState `json:"day_options"`
	DaysSelection   string                   `json:"days_selection"`
	CountOptions    []historyChoiceViewState `json:"count_options"`
	CountSelection  string                   `json:"count_selection"`
	PruneAsk        bool                     `json:"prune_ask"`
	PrunePrompt     string                   `json:"prune_prompt"`
	PruneDetail     string                   `json:"prune_detail"`
	DeleteAudioAsk  bool                     `json:"delete_audio_ask"`
}

type historySettingsStateInput struct {
	Open            bool
	AudioDir        string
	DefaultAudioDir string
	HomeDir         string
	AudioMissing    string
	UsageN          int
	UsageBytes      int64
	UsageFor        string
	Busy            string
	Done            int
	Total           int
	CanCancel       bool
	Kind            store.Kind
	Query           string
	ExportSelection string
	TextScope       string
	TextFormat      string
	TextBusy        bool
	CleanOn         bool
	DaysSelection   string
	CountSelection  string
	PruneAsk        bool
	PruneN          int
	DeleteAudioAsk  bool
}

func historyKeepDayOptions(translator *i18n.Translator) []historyChoiceViewState {
	labels := []string{translator.T("1 year"), translator.T("6 months"), translator.T("3 months"), translator.T("30 days"), translator.T("1 week")}
	options := make([]historyChoiceViewState, len(settings.KeepDaysChoices))
	for index, value := range settings.KeepDaysChoices {
		options[index] = historyChoiceViewState{Label: labels[index], Value: value}
	}
	return options
}

func historyKeepCountOptions(translator *i18n.Translator) []historyChoiceViewState {
	options := make([]historyChoiceViewState, len(settings.KeepCountChoices))
	for index, value := range settings.KeepCountChoices {
		label := strconv.Itoa(value)
		if index == 0 {
			label = translator.T("No limit")
		}
		options[index] = historyChoiceViewState{Label: label, Value: value}
	}
	return options
}

func historyShownLabelFor(translator *i18n.Translator, kind store.Kind, query string) string {
	tab := historyKindName(translator, kind)
	if kind == "" {
		tab = translator.T("All")
	}
	query = strings.TrimSpace(query)
	if query == "" {
		if kind == "" {
			return ""
		}
		return translator.T("%s only", tab)
	}
	if runes := []rune(query); len(runes) > 16 {
		query = string(runes[:16]) + "…"
	}
	return translator.T("%s matching “%s”", tab, query)
}

func buildHistorySettingsViewState(translator *i18n.Translator, input historySettingsStateInput) historySettingsViewState {
	shown := input.AudioDir
	if shown == "" {
		shown = input.DefaultAudioDir
	}
	state := historySettingsViewState{
		Open: input.Open, Directory: uitext.ShortPathAt(shown, input.HomeDir), CanUseDefault: input.AudioDir != "",
		UsageText: translator.T("Measuring…"), RecordingScope: input.ExportSelection,
		TextScope: input.TextScope, TextFormat: input.TextFormat, TextBusy: input.TextBusy,
		Busy: input.Busy != "", CanCancel: input.CanCancel, CleanOn: input.CleanOn,
		DayOptions: historyKeepDayOptions(translator), DaysSelection: input.DaysSelection,
		CountOptions: historyKeepCountOptions(translator), CountSelection: input.CountSelection,
		PruneAsk: input.PruneAsk, DeleteAudioAsk: input.DeleteAudioAsk,
	}
	if input.AudioMissing != "" {
		state.MissingNotice = translator.T("%s is not available, so new recordings go to %s until it is back.",
			uitext.ShortPathAt(input.AudioMissing, input.HomeDir), uitext.ShortPathAt(input.DefaultAudioDir, input.HomeDir))
	}
	switch {
	case input.UsageFor != "done":
	case input.UsageN < 0:
		state.UsageText = translator.T("Could not measure them")
	default:
		state.UsageText = translator.N(input.UsageN, "%d recording · %s", "%d recordings · %s", input.UsageN, uitext.HumanBytes(input.UsageBytes))
	}
	state.CanDelete = !state.Busy && input.UsageN > 0
	state.CanExport = !state.Busy && input.UsageN != 0
	shownLabel := historyShownLabelFor(translator, input.Kind, input.Query)
	state.RecordingScopes = []string{translator.T("All recordings")}
	state.TextScopes = []string{translator.T("All entries")}
	if shownLabel != "" {
		state.RecordingScopes = append(state.RecordingScopes, shownLabel)
		state.TextScopes = append(state.TextScopes, shownLabel)
	}
	state.RecordingScope = historySelectedLabel(state.RecordingScope, state.RecordingScopes)
	state.TextScope = historySelectedLabel(state.TextScope, state.TextScopes)
	state.TextFormats = []string{"Markdown", "JSONL"}
	state.TextFormat = historySelectedLabel(state.TextFormat, state.TextFormats)
	if state.Busy {
		state.BusyLabel = translator.T("Exporting the recordings… %d of %d", input.Done, input.Total)
		if input.Busy == "move" {
			state.BusyLabel = translator.T("Moving the recordings… %d of %d", input.Done, input.Total)
		}
		if input.Total > 0 {
			state.Progress = float64(input.Done) / float64(input.Total)
		}
	}
	if state.PruneAsk {
		state.PrunePrompt = translator.N(input.PruneN, "Delete %d entry now?", "Delete %d entries now?", input.PruneN)
		state.PruneDetail = translator.N(input.PruneN,
			"This will delete %d entry and its recording now, and others as they pass the limit. This cannot be undone.",
			"This will delete %d entries and their recordings now, and others as they pass the limit. This cannot be undone.", input.PruneN)
	}
	return state
}

func historySelectedLabel(selected string, options []string) string {
	if slices.Contains(options, selected) {
		return selected
	}
	return options[0]
}

func historySelectedValue(label string, options []historyChoiceViewState) int {
	for _, option := range options {
		if option.Label == label {
			return option.Value
		}
	}
	return options[0].Value
}
