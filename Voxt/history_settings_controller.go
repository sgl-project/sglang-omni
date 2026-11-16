package main

import (
	"context"
	"slices"
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type histSettingsState struct {
	pruneTo      settings.History
	usageN       int
	usageBytes   int64
	usageFor     string
	usageSeq     int
	audioMissing string
	busy         string
	done, total  int
	cancel       context.CancelFunc
	pruning      bool
	pruneAgain   bool
	textBusy     bool
}

func (a *app) histSearch() string { return strings.TrimSpace(a.histUI.query) }

func (a *app) shownLabel() string {
	return historyShownLabelFor(i18n.CurrentTranslator(), a.histUI.kind(), a.histSearch())
}

func (a *app) openHistorySettingsDialog() {
	a.histSetUI.open, a.histSetUI.formFor, a.histSet.usageFor = true, "", ""
	if a.histSet.busy == "" {
		a.applyAudioDir()
	}
}

func (a *app) prepareHistorySettings() {
	if !a.histSetUI.open {
		return
	}
	if a.histSet.usageFor == "" {
		a.loadRecordingUsage()
	}
	a.syncHistForm()
}

func (a *app) syncHistForm() {
	local := &a.histSetUI
	history := a.prefs.History
	key := strconv.FormatBool(history.CleanupOff) + "/" + strconv.Itoa(history.Days()) + "/" + strconv.Itoa(history.Count()) + "/" + i18n.Locale()
	if local.formFor == key || local.formFor == "asking" {
		return
	}
	local.formFor, local.cleanOn = key, !history.CleanupOff
	dayOptions := historyKeepDayOptions(i18n.CurrentTranslator())
	local.daysSel = dayOptions[max(slices.Index(settings.KeepDaysChoices, history.Days()), 0)].Label
	countOptions := historyKeepCountOptions(i18n.CurrentTranslator())
	local.countSel = countOptions[max(slices.Index(settings.KeepCountChoices, history.Count()), 0)].Label
}

func (a *app) historySettingsModel() historySettingsViewState {
	return buildHistorySettingsViewState(i18n.CurrentTranslator(), historySettingsStateInput{
		Open: a.histSetUI.open, AudioDir: a.prefs.History.AudioDir, DefaultAudioDir: a.lib.store.History().DefaultAudioDir(),
		HomeDir: a.homeDir, AudioMissing: a.histSet.audioMissing, UsageN: a.histSet.usageN, UsageBytes: a.histSet.usageBytes,
		UsageFor: a.histSet.usageFor, Busy: a.histSet.busy, Done: a.histSet.done, Total: a.histSet.total,
		CanCancel: a.histSet.cancel != nil, Kind: a.histUI.kind(), Query: a.histUI.query,
		ExportSelection: a.histSetUI.exportSel, TextScope: a.histSetUI.textScope, TextFormat: a.histSetUI.textFormat,
		TextBusy: a.histSet.textBusy, CleanOn: a.histSetUI.cleanOn, DaysSelection: a.histSetUI.daysSel,
		CountSelection: a.histSetUI.countSel, PruneAsk: a.histSetUI.pruneAsk, PruneN: a.histSetUI.pruneN,
		DeleteAudioAsk: a.histSetUI.delAudioAsk,
	})
}

func (a *app) chooseHistoryRecordingFolder() {
	chooseFolder(a, i18n.T("Move the recordings to…"), i18n.T("Move here"), a.moveRecordings)
}

func (a *app) showHistoryRecordingFolder() {
	revealInFinder(a.lib.store.History().AudioDir())
}

func (a *app) useDefaultHistoryRecordingFolder() { a.moveRecordings("") }

func (a *app) exportHistoryRecordings(shown bool) {
	chooseFolder(a, i18n.T("Export the recordings to…"), i18n.T("Export here"), func(directory string) {
		a.exportRecordings(directory, shown)
	})
}

func (a *app) setHistoryCleanup(on bool) {
	next := a.prefs.History
	next.CleanupOff = !on
	if !on {
		a.saveCleanup(next)
		return
	}
	a.histSetUI.formFor = "asking"
	a.askCleanup(next)
}

func (a *app) setHistoryCleanupDays(keepDays int) {
	next := a.prefs.History
	next.KeepDays = keepDays
	a.histSetUI.formFor = "asking"
	a.askCleanup(next)
}

func (a *app) setHistoryCleanupCount(keepCount int) {
	next := a.prefs.History
	next.KeepCount = keepCount
	a.histSetUI.formFor = "asking"
	a.askCleanup(next)
}

func (a *app) savePendingHistoryCleanup() { a.saveCleanup(a.histSet.pruneTo) }

func (a *app) deleteHistoryRecordings() { a.deleteRecordings() }

func (a *app) cancelHistoryWork() {
	if a.histSet.cancel != nil {
		a.histSet.cancel()
	}
}

func (a *app) showHistorySettings() {
	a.navigate("/history")
	a.openHistorySettingsDialog()
}
