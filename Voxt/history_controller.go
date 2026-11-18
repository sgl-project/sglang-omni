package main

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const historyLimit = 500

type historyState struct {
	hist            historyView
	histUI          historyLocalUI
	histSet         histSettingsState
	histSetUI       historySettingsLocalUI
	homeDir         string
	defaultAudioDir string
}

type historyView struct {
	list    []store.Entry
	listFor string
	seq     int
	loading bool
	stale   bool
	err     string
	retry   entryRetry
	view    historyViewState
	viewKey historyViewCacheKey
}

type historyViewCacheKey struct {
	seq             int
	listFor         string
	loading         bool
	stale           bool
	err             string
	tab             int
	query           string
	selectedID      string
	retryID         string
	retryStage      string
	retryDraft      string
	keepHistory     bool
	audioMissing    string
	defaultAudioDir string
	homeDir         string
	locale          string
	location        string
	zoneOffset      int
	minute          int64
	asrModel        string
	llmModel        string
}

var revealInFinder = func(path string) { mygo.Shell.ShowItemInFolder(path) }

func (h *historyLocalUI) kind() store.Kind {
	if h.tab == len(store.Kinds) {
		return ""
	}
	return store.Kinds[min(max(h.tab, 0), len(store.Kinds)-1)]
}

func (h *historyLocalUI) wants() string {
	return string(h.kind()) + "\x00" + strings.TrimSpace(h.query)
}

func (h *historyView) selected(id string) *store.Entry {
	if id == "" {
		return nil
	}
	for index := range h.list {
		if h.list[index].ID == id {
			return &h.list[index]
		}
	}
	return nil
}

func (a *app) loadHistory() {
	h := &a.hist
	h.seq++
	h.loading, h.stale = true, false
	seq, kind, search, want := h.seq, a.histUI.kind(), strings.TrimSpace(a.histUI.query), a.histUI.wants()
	history := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := history.List(ctx, store.HistoryQuery{Kind: kind, Search: search, Limit: historyLimit})
		a.post(func() {
			if seq != a.hist.seq {
				return
			}
			h := &a.hist
			h.loading, h.listFor = false, want
			if err != nil {
				h.err = err.Error()
				return
			}
			h.list, h.err = list, ""
			if h.selected(a.histUI.sel) == nil {
				a.histUI.sel = ""
			}
		})
	})
}

func (a *app) historyChanged() {
	for _, observer := range a.onHistory {
		observer()
	}
}

func (a *app) historyReload() {
	if a.currentPage() == "history" {
		a.loadHistory()
		return
	}
	a.hist.stale = true
}

func (a *app) historyShown() { a.loadHistory() }

func (a *app) prepareHistoryPage() {
	if !a.hist.loading && (a.hist.listFor == "" || a.hist.stale) {
		a.loadHistory()
	}
}

func (a *app) historyPageModel(now time.Time) historyViewState {
	var detail *historyDetailStateInput
	asrModel, llmModel := "", ""
	a.histUI.transcriptKind, a.histUI.transcriptStreaming = "", false
	if selected := a.hist.selected(a.histUI.sel); selected != nil {
		asrModel = a.entryModel(local.ASR, selected.ASRProvider, selected.ASRModel)
		llmModel = a.entryModel(local.LLM, selected.LLMProvider, selected.LLMModel)
		detail = &historyDetailStateInput{Entry: *selected, Retry: a.hist.retry, ASRModel: asrModel, LLMModel: llmModel}
		a.histUI.transcriptKind = string(selected.Kind)
		a.histUI.transcriptStreaming = a.hist.retry.id == selected.ID && strings.TrimSpace(a.hist.retry.draft) != ""
	}
	_, zoneOffset := now.Zone()
	key := historyViewCacheKey{
		seq: a.hist.seq, listFor: a.hist.listFor, loading: a.hist.loading, stale: a.hist.stale, err: a.hist.err,
		tab: a.histUI.tab, query: a.histUI.query, selectedID: a.histUI.sel, retryID: a.hist.retry.id, retryStage: string(a.hist.retry.stage), retryDraft: a.hist.retry.draft,
		keepHistory: a.prefs.Privacy.KeepHistory, audioMissing: a.histSet.audioMissing, defaultAudioDir: a.defaultAudioDir,
		homeDir: a.homeDir, locale: i18n.Locale(), location: now.Location().String(), zoneOffset: zoneOffset,
		minute: now.Unix() / 60, asrModel: asrModel, llmModel: llmModel,
	}
	if key != a.hist.viewKey {
		a.hist.view = buildHistoryViewState(i18n.CurrentTranslator(), historyViewStateInput{
			Entries: a.hist.list, Kind: a.histUI.kind(), Read: a.hist.listFor != "",
			Error: a.hist.err, Query: a.histUI.query, SelectedID: a.histUI.sel, Retry: a.hist.retry,
			KeepHistory: a.prefs.Privacy.KeepHistory, SettingsAvailable: true, AudioMissing: a.histSet.audioMissing,
			DefaultAudioDir: a.defaultAudioDir, HomeDir: a.homeDir, Now: now, Detail: detail,
		})
		a.hist.viewKey = key
	}
	return a.hist.view
}

func (a *app) searchHistory(query string) {
	a.histUI.query = query
	a.loadHistory()
}

func (a *app) showHistoryTab(tab int) {
	a.histUI.tab, a.histUI.sel = tab, ""
	a.loadHistory()
}

func (a *app) openHistoryDetail(id string) { a.histUI.sel = id }
func (a *app) closeHistoryDetail()         { a.histUI.sel = "" }

func (a *app) openHistoryFeature(feature string) {
	a.navigate("/" + feature)
}

func (a *app) openHistoryPrivacy() { a.navigate("/settings/privacy") }

func (a *app) copyHistoryText(text string) {
	if text == "" {
		return
	}
	var err error
	if a.copyText != nil {
		err = a.copyText(text)
	} else {
		err = mygo.Clipboard.Write(transfer.TextData(text))
	}
	if err != nil {
		a.toast(i18n.T("Could not copy: %v", err))
		return
	}
	a.toast(i18n.T("Copied"))
}

func (a *app) historyEntryByID(id string) (store.Entry, bool) {
	visible := [][]store.Entry{a.hist.list, a.hm.recent}
	if a.currentPage() == "home" {
		visible = [][]store.Entry{a.hm.recent, a.hist.list}
	}
	for _, entries := range visible {
		for _, entry := range entries {
			if entry.ID == id {
				return entry, true
			}
		}
	}
	return store.Entry{}, false
}

func (a *app) showHistoryRecording(id string) {
	if entry, ok := a.historyEntryByID(id); ok && entry.Audio != "" {
		revealInFinder(a.lib.store.History().AudioPath(entry.Audio))
	}
}

func (a *app) retryHistoryEntry(id string) {
	if entry, ok := a.historyEntryByID(id); ok {
		a.retryEntry(entry)
	}
}

func (a *app) deleteHistoryEntry(id string) {
	if id == "" {
		return
	}
	history := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		err := history.Delete(ctx, id)
		a.post(func() {
			if err != nil && !errors.Is(err, core.ErrNotFound) {
				a.toast(i18n.T("Could not delete the entry: %v", err))
				return
			}
			if a.histUI.sel == id {
				a.histUI.sel = ""
			}
			a.toast(i18n.T("Entry deleted"))
			a.historyChanged()
		})
	})
}

func (a *app) openHistoryEntry(entry store.Entry) {
	for index, kind := range store.Kinds {
		if kind == entry.Kind && a.histUI.tab != index {
			a.histUI.tab, a.hist.listFor = index, ""
		}
	}
	a.histUI.query = ""
	a.histUI.sel = entry.ID
	a.navigate("/history")
}

func (a *app) entryModel(kind local.Kind, id, model string) string {
	if id == "" {
		return ""
	}
	name := uitext.ModelName(id)
	for _, installed := range a.installed(kind) {
		if installed.ID == id {
			name = installed.Name
		}
	}
	if model != "" && model != id {
		name += " · " + model
	}
	return name
}
