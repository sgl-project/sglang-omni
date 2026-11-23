package main

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type historyAccessViewState struct {
	On           bool     `json:"on"`
	Selection    bool     `json:"selection"`
	Ask          bool     `json:"ask"`
	ListOpen     bool     `json:"list_open"`
	ShowLastRead bool     `json:"show_last_read"`
	LastRead     string   `json:"last_read"`
	Command      string   `json:"command"`
	ReadLines    []string `json:"read_lines"`
}

type historyAccessStateInput struct {
	On        bool
	Selection bool
	Ask       bool
	ListOpen  bool
	Exe       string
	Reads     []historyapi.Read
	Now       time.Time
}

func buildHistoryAccessViewState(translator *i18n.Translator, input historyAccessStateInput) historyAccessViewState {
	state := historyAccessViewState{
		On: input.On, Selection: input.Selection, Ask: input.Ask, ListOpen: input.ListOpen,
		ShowLastRead: input.On || len(input.Reads) > 0, LastRead: translator.T("No reads yet"),
	}
	if input.On {
		state.Command = input.Exe + ` history export --day "$(date +%F)"`
	}
	for _, read := range input.Reads {
		state.ReadLines = append(state.ReadLines, historyReadLineFor(translator, read, input.Now))
	}
	if len(state.ReadLines) > 0 {
		state.LastRead = state.ReadLines[0]
	}
	return state
}

func historyReaderLabelFor(translator *i18n.Translator, read historyapi.Read) string {
	if name := readerName(read); name != "" {
		return name
	}
	return translator.T("Unknown program")
}

func historyReadLineFor(translator *i18n.Translator, read historyapi.Read, now time.Time) string {
	what := translator.T("Status check")
	if read.Op == historyapi.OpExport {
		what = translator.N(read.Count, "%d entry", "%d entries", read.Count)
	}
	return strings.Join([]string{translator.DateTimeAt(read.At, now), what, historyReaderLabelFor(translator, read)}, " · ")
}
