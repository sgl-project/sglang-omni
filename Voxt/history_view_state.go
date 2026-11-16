package main

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type historyMarkViewState struct {
	Label   string `json:"label"`
	Tooltip string `json:"tooltip"`
	Tone    string `json:"tone"`
}

type historyRetryViewState struct {
	Available bool   `json:"available"`
	Wanted    bool   `json:"wanted"`
	Active    bool   `json:"active"`
	Label     string `json:"label"`
	Disabled  bool   `json:"disabled"`
	Stage     string `json:"stage"`
	StageText string `json:"stage_text"`
	Tooltip   string `json:"tooltip"`
}

type historyRowViewState struct {
	ID          string                 `json:"id"`
	Kind        string                 `json:"kind"`
	Status      string                 `json:"status"`
	Delivery    string                 `json:"delivery"`
	RuleSource  string                 `json:"rule_source"`
	DSP         string                 `json:"dsp"`
	Title       string                 `json:"title"`
	FailedTitle bool                   `json:"failed_title"`
	Meta        string                 `json:"meta"`
	Selected    bool                   `json:"selected"`
	CopyText    string                 `json:"copy_text"`
	RawCopyText string                 `json:"raw_copy_text"`
	CanReveal   bool                   `json:"can_reveal"`
	Retry       historyRetryViewState  `json:"retry"`
	Marks       []historyMarkViewState `json:"marks"`
}

type historyTextBlockViewState struct {
	Key       string `json:"key"`
	Title     string `json:"title"`
	CopyLabel string `json:"copy_label"`
	Text      string `json:"text"`
}

type historyFactViewState struct {
	Label string `json:"label"`
	Value string `json:"value"`
	Empty bool   `json:"empty"`
}

type historyFactSectionViewState struct {
	Title string                 `json:"title"`
	Facts []historyFactViewState `json:"facts"`
}

type historyDetailViewState struct {
	ID                string                        `json:"id"`
	Kind              string                        `json:"kind"`
	Status            string                        `json:"status"`
	Delivery          string                        `json:"delivery"`
	RuleSource        string                        `json:"rule_source"`
	DSP               string                        `json:"dsp"`
	ErrorText         string                        `json:"error_text"`
	ErrorDetail       string                        `json:"error_detail"`
	ErrorWarning      bool                          `json:"error_warning"`
	CanReveal         bool                          `json:"can_reveal"`
	Retry             historyRetryViewState         `json:"retry"`
	RecordingKeptText string                        `json:"recording_kept_text"`
	TextBlocks        []historyTextBlockViewState   `json:"text_blocks"`
	Sections          []historyFactSectionViewState `json:"sections"`
}

type historyEmptyViewState struct {
	Search      bool   `json:"search"`
	Kind        string `json:"kind"`
	Title       string `json:"title"`
	Body        string `json:"body"`
	ActionLabel string `json:"action_label"`
}

type historyViewState struct {
	Read              bool                    `json:"read"`
	Loading           bool                    `json:"loading"`
	Error             string                  `json:"error"`
	Kind              string                  `json:"kind"`
	Query             string                  `json:"query"`
	Tabs              []string                `json:"tabs"`
	CountLabel        string                  `json:"count_label"`
	KeepHistory       bool                    `json:"keep_history"`
	SettingsAvailable bool                    `json:"settings_available"`
	FolderNotice      string                  `json:"folder_notice"`
	Rows              []historyRowViewState   `json:"rows"`
	Empty             *historyEmptyViewState  `json:"empty,omitempty"`
	Detail            *historyDetailViewState `json:"detail,omitempty"`
}

type historyDetailStateInput struct {
	Entry    store.Entry
	Retry    entryRetry
	ASRModel string
	LLMModel string
}

type historyViewStateInput struct {
	Entries           []store.Entry
	Kind              store.Kind
	Read              bool
	Loading           bool
	Error             string
	Query             string
	SelectedID        string
	Retry             entryRetry
	KeepHistory       bool
	SettingsAvailable bool
	AudioMissing      string
	DefaultAudioDir   string
	HomeDir           string
	Now               time.Time
	Detail            *historyDetailStateInput
}

func historyTabsFor(translator *i18n.Translator) []string {
	return []string{translator.T("Transcribe"), translator.T("Translate"), translator.T("Rewrite"), translator.T("All")}
}

func historyKindName(translator *i18n.Translator, kind store.Kind) string {
	for index, candidate := range store.Kinds {
		if candidate == kind {
			return historyTabsFor(translator)[index]
		}
	}
	return string(kind)
}

func buildHistoryViewState(translator *i18n.Translator, input historyViewStateInput) historyViewState {
	state := historyViewState{
		Read: input.Read, Loading: input.Loading, Error: input.Error, Kind: string(input.Kind), Query: input.Query,
		Tabs: historyTabsFor(translator), KeepHistory: input.KeepHistory, SettingsAvailable: input.SettingsAvailable,
	}
	if count := len(input.Entries); count > 0 {
		state.CountLabel = translator.N(count, "%d entry", "%d entries", count)
		if count >= historyLimit {
			state.CountLabel = translator.T("Newest %d", count)
		}
	}
	if input.AudioMissing != "" {
		state.FolderNotice = translator.T("%s is not available, so new recordings go to %s until it is back.",
			uitext.ShortPathAt(input.AudioMissing, input.HomeDir), uitext.ShortPathAt(input.DefaultAudioDir, input.HomeDir))
	}
	query := strings.TrimSpace(input.Query)
	for _, entry := range input.Entries {
		state.Rows = append(state.Rows, buildHistoryRowViewState(translator, entry, query, input.SelectedID, input.Kind == "", input.Retry, input.Now))
	}
	if input.Read && len(input.Entries) == 0 {
		state.Empty = buildHistoryEmptyViewState(translator, input.Kind, query)
	}
	if input.Detail != nil {
		state.Detail = buildHistoryDetailViewState(translator, *input.Detail, input.Now)
	}
	return state
}

func buildHistoryRowViewState(translator *i18n.Translator, entry store.Entry, query, selectedID string, showKind bool, retry entryRetry, now time.Time) historyRowViewState {
	title, failed := historyLineFor(translator, entry, query)
	meta := uitext.EntryMetaFor(translator, entry, now)
	if showKind {
		meta = historyKindName(translator, entry.Kind) + " · " + meta
	}
	state := historyRowViewState{
		ID: entry.ID, Kind: string(entry.Kind), Status: string(entry.Status), Delivery: entry.Delivery,
		RuleSource: entry.RuleSource, DSP: entry.DSP, Title: title, FailedTitle: failed, Meta: meta,
		Selected: selectedID == entry.ID, CopyText: historyEntryCopyText(entry), RawCopyText: historyEntryRawCopy(entry),
		CanReveal: entry.Audio != "", Retry: buildHistoryRetryViewState(translator, entry, retry),
	}
	if !state.Retry.Active && !(state.Retry.Available && state.Retry.Wanted) {
		state.Marks = historyEntryMarks(translator, entry)
	}
	return state
}

func historyLineFor(translator *i18n.Translator, entry store.Entry, query string) (line string, failed bool) {
	title, failed := uitext.EntryTitleFor(translator, entry)
	if query == "" || len(findFolded(title, query)) > 0 {
		return title, failed
	}
	for _, text := range []string{entry.Result, entry.RawText, entry.SourceText} {
		for line := range strings.SplitSeq(text, "\n") {
			if line = strings.TrimSpace(line); len(findFolded(line, query)) > 0 {
				return line, false
			}
		}
	}
	return title, failed
}

func historyEntryCopyText(entry store.Entry) string {
	if text := strings.TrimSpace(entry.Result); text != "" {
		return text
	}
	return strings.TrimSpace(entry.RawText)
}

func historyEntryRawCopy(entry store.Entry) string {
	if raw := strings.TrimSpace(entry.RawText); raw != historyEntryCopyText(entry) {
		return raw
	}
	return ""
}

func historyEntryMarks(translator *i18n.Translator, entry store.Entry) []historyMarkViewState {
	switch entry.Status {
	case store.StatusFailed:
		return []historyMarkViewState{{Label: translator.T("Failed"), Tooltip: translator.T("No recording was kept, so it cannot be retried"), Tone: "danger"}}
	case store.StatusCancelled:
		return []historyMarkViewState{{Label: translator.T("Cancelled")}}
	}
	switch entry.Delivery {
	case "card":
		return []historyMarkViewState{{Label: translator.T("Not inserted"), Tooltip: translator.T("Shown in the answer card, not inserted")}}
	case "copy":
		return []historyMarkViewState{{Label: translator.T("Copied"), Tooltip: translator.T("Copied from the answer card")}}
	}
	return nil
}

func historyRetryWanted(translator *i18n.Translator, entry store.Entry) bool {
	_, failed := uitext.EntryTitleFor(translator, entry)
	return failed || entry.Status == store.StatusFailed
}

func buildHistoryRetryViewState(translator *i18n.Translator, entry store.Entry, retry entryRetry) historyRetryViewState {
	wanted := historyRetryWanted(translator, entry)
	label := translator.T("Transcribe again")
	if wanted {
		label = translator.T("Retry")
	}
	state := historyRetryViewState{
		Available: entry.Audio != "", Wanted: wanted, Active: retry.id == entry.ID, Label: label, Disabled: retry.id != "",
	}
	if state.Active {
		state.Stage = string(retry.stage)
		state.StageText = translator.T("Recognising…")
		if retry.stage == "composing" {
			state.StageText = translator.T("Cleaning up…")
		}
	}
	if !wanted {
		state.Tooltip = translator.T("Recognise the recording again. Nothing is inserted.")
	}
	return state
}

func buildHistoryEmptyViewState(translator *i18n.Translator, kind store.Kind, query string) *historyEmptyViewState {
	if query != "" {
		if runes := []rune(query); len(runes) > 40 {
			query = string(runes[:40]) + "…"
		}
		return &historyEmptyViewState{Search: true, Title: translator.T("Nothing matches “%s”", query), ActionLabel: translator.T("Clear search")}
	}
	state := &historyEmptyViewState{Kind: string(kind)}
	switch kind {
	case "":
		state.Title = translator.T("No history yet")
		state.Body = translator.T("Transcriptions, translations and rewrites show up here.")
	case store.KindTranslate:
		state.Title = translator.T("No translations yet")
		state.Body = translator.T("Translations show up here with their source.")
		state.ActionLabel = translator.T("Set up Translate")
	case store.KindRewrite:
		state.Title = translator.T("No rewrites yet")
		state.Body = translator.T("Rewrites show up here with your instruction.")
		state.ActionLabel = translator.T("Set up Rewrite")
	default:
		state.Title = translator.T("No transcriptions yet")
		state.Body = translator.T("Each result shows up here.")
		state.ActionLabel = translator.T("Set up Transcribe")
	}
	return state
}

func buildHistoryDetailViewState(translator *i18n.Translator, input historyDetailStateInput, now time.Time) *historyDetailViewState {
	entry := input.Entry
	state := &historyDetailViewState{
		ID: entry.ID, Kind: string(entry.Kind), Status: string(entry.Status), Delivery: entry.Delivery,
		RuleSource: entry.RuleSource, DSP: entry.DSP, ErrorDetail: entry.Error, CanReveal: entry.Audio != "",
		Retry: buildHistoryRetryViewState(translator, entry, input.Retry),
	}
	if entry.Error != "" {
		state.ErrorText = translator.T("Failed: %s", uitext.FirstLine(entry.Error))
		state.ErrorWarning = entry.Status != store.StatusFailed
		if state.ErrorWarning {
			state.ErrorText = translator.T("Done, but with a problem: %s", uitext.FirstLine(entry.Error))
		}
	}
	if entry.Audio != "" && !entry.AudioUntil.IsZero() {
		state.RecordingKeptText = translator.T("Recording kept until %s because recognition failed.", translator.DateTimeAt(entry.AudioUntil, now))
		if entry.Status == store.StatusCancelled {
			state.RecordingKeptText = translator.T("Recording kept until %s so you can retry.", translator.DateTimeAt(entry.AudioUntil, now))
		}
	}
	addTextBlock := func(key, title, copyLabel, text string) {
		if text = strings.TrimSpace(text); text != "" {
			state.TextBlocks = append(state.TextBlocks, historyTextBlockViewState{Key: key, Title: title, CopyLabel: copyLabel, Text: text})
		}
	}
	addTextBlock("hist-result", translator.T("Result"), translator.T("Copy result"), entry.Result)
	if entry.PromptText != "" {
		addTextBlock("hist-prompt", translator.T("Instruction"), translator.T("Copy instruction"), entry.PromptText)
	}
	if entry.SourceText != "" {
		addTextBlock("hist-source", translator.T("Selected text"), translator.T("Copy selected text"), entry.SourceText)
	}
	addTextBlock("hist-raw", translator.T("Recognised text"), translator.T("Copy recognised text"), entry.RawText)
	appName := entry.AppName
	if entry.URLHost != "" {
		appName += " · " + entry.URLHost
	}
	details := []historyFactViewState{
		historyFact(translator.T("Time"), translator.DateTimeAt(entry.CreatedAt, now)),
		historyFact(translator.T("App"), appName),
	}
	if entry.Kind == store.KindTranslate {
		details = append(details, historyFact(translator.T("Target language"), uitext.TargetLangFor(translator, entry)))
	}
	details = append(details,
		historyFact(translator.T("Delivery"), uitext.DeliveryLabelFor(translator, entry)),
		historyFact(translator.T("Speech recognition"), input.ASRModel),
		historyFact(translator.T("Sound optimisation"), uitext.SoundTextFor(translator, entry)),
		historyFact(translator.T("Language model"), input.LLMModel),
		historyFact(translator.T("App rule"), uitext.RuleTextFor(translator, entry)),
	)
	state.Sections = []historyFactSectionViewState{
		{Title: translator.T("Details"), Facts: details},
		{Title: translator.T("Timing"), Facts: []historyFactViewState{
			historyFact(translator.T("Recording"), uitext.RecordingTimeFor(translator, entry)),
			historyFact(translator.T("Recognition"), uitext.StageTime(entry.ASRMS)),
			historyFact(translator.T("Language model"), uitext.StageTime(entry.LLMMS)),
			historyFact(translator.T("Total"), uitext.StageTime(entry.TotalMS)),
		}},
	}
	return state
}

func historyFact(label, value string) historyFactViewState {
	empty := value == ""
	if empty {
		value = "—"
	}
	return historyFactViewState{Label: label, Value: value, Empty: empty}
}
