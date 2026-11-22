package main

// The logs' logic (shell_logs.go and shell_logs_order.go draw them): the
// lines the filters select, the copies, and the view state's cache.

import (
	"cmp"
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/hehehai/actnowkit/listsel"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/logs"
)

// logLines is how many lines the logs view shows at most.
const logLines = 500

// logSources are the filters of the logs view: "" every source.
var logSources = []string{"", "app", helper.LogSource}

// logsState is the logs' part of the app (main thread).
type logsState struct {
	logSel   logSelection // the lines selected (shell_logs_select.go)
	logsFind finder       // Return steps through the filter's matches (find_marks.go)
	logsUI   logsLocalUI  // the views' local state

	// logsView is the last built view state, logsViewKey its inputs
	// (logsModel).
	logsView    logsViewState
	logsViewKey logsCacheKey
}

// logsCacheKey is every input of a logsViewState: when it matches the
// last build's, logsModel reuses the state instead of rebuilding it
// every frame. Lines only ever join the log, with a new Seq, so the
// tail's ends and length name it.
type logsCacheKey struct {
	tailLen       int
	first, last   uint64
	source, query string
	locale        string
	logFile       bool
}

// visibleLogs are the lines the filters select, the newest first: the
// order the view shows (copies go back to the log's order, copyLogLines).
func (a *app) visibleLogs() []core.LogEntry {
	return filterLogs(a.logs.Tail(logLines*4, a.logsSource), a.logsQuery)
}

// filterLogs are the lines of tail (the newest last) matching q, the
// newest first, at most logLines of them.
func filterLogs(tail []core.LogEntry, query string) []core.LogEntry {
	q := strings.ToLower(strings.TrimSpace(query))
	var out []core.LogEntry
	for _, e := range slices.Backward(tail) {
		if q == "" || strings.Contains(strings.ToLower(e.Message), q) || strings.Contains(strings.ToLower(e.Source), q) {
			out = append(out, e)
			if len(out) == logLines {
				break
			}
		}
	}
	return out
}

// logsModel is the logs' view state, rebuilt when the log's tail, the
// filters or the language changed.
func (a *app) logsModel() *logsViewState {
	tail := a.logs.Tail(logLines*4, a.logsSource)
	key := logsCacheKey{
		tailLen: len(tail), source: a.logsSource, query: a.logsQuery,
		locale: i18n.Locale(), logFile: a.logs.Path() != "",
	}
	if len(tail) > 0 {
		key.first, key.last = tail[0].Seq, tail[len(tail)-1].Seq
	}
	if key != a.logsViewKey {
		a.logsView = buildLogsViewState(i18n.CurrentTranslator(), filterLogs(tail, a.logsQuery), a.logsSource, a.logsQuery, key.logFile)
		a.logsViewKey = key
	}
	return &a.logsView
}

// logsLocal returns the views' local state, wired to the app's on first
// use.
func (a *app) logsLocal() *logsLocalUI {
	if a.logsUI.sel == nil {
		a.logsUI.sel, a.logsUI.find = &a.logSel, &a.logsFind
		a.logsUI.query, a.logsUI.source = &a.logsQuery, &a.logsSource
	}
	return &a.logsUI
}

// copyLogLines copies the shown lines with these seqs as the log file
// has them, the oldest first whatever order they show in, says how
// many, and clears the selection.
func (a *app) copyLogLines(seqs []uint64) {
	if len(seqs) == 0 {
		return
	}
	shown := a.visibleLogs()
	var lines []core.LogEntry
	for _, e := range shown {
		if slices.Contains(seqs, e.Seq) {
			lines = append(lines, e)
		}
	}
	if len(lines) == 0 {
		return
	}
	lines = slices.SortedFunc(slices.Values(lines), func(x, y core.LogEntry) int { return cmp.Compare(x.Seq, y.Seq) })
	a.mainUI.WriteClipboard(listsel.Text(lines, logs.Format))
	a.toast(i18n.N(len(lines), "Copied %d line", "Copied %d lines", len(lines)))
	a.logSel.m.Clear()
}

// openLogFile opens the log file in the Finder.
func (a *app) openLogFile() {
	if path := a.logs.Path(); path != "" {
		mygo.Shell.OpenPath(path)
	}
}

// closeLogsPanel closes the panel under the page.
func (a *app) closeLogsPanel() { a.logsPanel = false }
