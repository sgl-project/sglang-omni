package main

// The logs' view state: the lines the filters select, the newest first,
// with each line's time, level and the filters' labels already decided.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// The levels a log line shows, as codes (the view maps them to a
// colour).
const (
	logLevelDebug   = "debug"
	logLevelInfo    = "info"
	logLevelWarning = "warning"
	logLevelError   = "error"
)

// logLineViewState is one log line as the view shows it.
type logLineViewState struct {
	Seq     uint64 `json:"seq"`
	Time    string `json:"time"`
	Source  string `json:"source"`
	Message string `json:"message"`
	Level   string `json:"level"`
}

// logSourceViewState is one source filter: its code and its label.
type logSourceViewState struct {
	Code  string `json:"code"`
	Label string `json:"label"`
}

// logsViewState is the logs page's or panel's display data.
type logsViewState struct {
	Lines   []logLineViewState   `json:"lines"`
	Count   string               `json:"count"` // "500 lines"
	Sources []logSourceViewState `json:"sources"`
	Source  string               `json:"source"` // the current filter's label
	Query   string               `json:"query"`
	LogFile bool                 `json:"log_file"` // a log file exists to open
}

// logLevelCode is the code of a line's level.
func logLevelCode(level core.Severity) string {
	switch level {
	case core.SeverityError:
		return logLevelError
	case core.SeverityWarning:
		return logLevelWarning
	case core.SeverityDebug:
		return logLevelDebug
	}
	return logLevelInfo
}

// buildLogsViewState decides how the filtered lines show: their times
// and levels, the count and the filters' labels. The labels come from
// uitext.LogSourceLabel, which reads the active language: production
// and the fixture generator both set it before building.
func buildLogsViewState(translator *i18n.Translator, lines []core.LogEntry, source, query string, logFile bool) logsViewState {
	state := logsViewState{
		Count:   translator.N(len(lines), "%d line", "%d lines", len(lines)),
		Source:  uitext.LogSourceLabel(source),
		Query:   query,
		LogFile: logFile,
	}
	for _, code := range logSources {
		state.Sources = append(state.Sources, logSourceViewState{Code: code, Label: uitext.LogSourceLabel(code)})
	}
	if len(lines) > 0 {
		state.Lines = make([]logLineViewState, len(lines))
		for i, e := range lines {
			state.Lines[i] = logLineViewState{
				Seq:     e.Seq,
				Time:    e.Time.Format("15:04:05"),
				Source:  e.Source,
				Message: e.Message,
				Level:   logLevelCode(e.Level),
			}
		}
	}
	return state
}
