package main

// Other programs reading the history: socket lifecycle, policy and audit persistence.

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	historyReadsDoc  = "history:reads"
	historyReadsKept = 20
	historySeenKept  = 50
	historyDocsLink  = "https://github.com/sgl-project/sglang-omni/blob/main/Voxt/docs/history-access.md"
)

type historyAccessState struct {
	histAccess struct {
		exe   string
		ln    net.Listener
		stop  context.CancelFunc
		done  chan struct{}
		reads []historyapi.Read
	}
	histAccessUI historyAccessLocalUI
	histPolicy   atomic.Pointer[historyapi.Policy]
}

type historyReads struct {
	Reads []historyapi.Read `json:"reads"`
	Seen  []string          `json:"seen"`
}

func readerName(read historyapi.Read) string {
	switch {
	case read.As != "" && read.Program != "":
		return read.As + " (" + read.Program + ")"
	case read.As != "":
		return read.As
	}
	return read.Program
}

var historyCommandPath = func() string {
	executable, err := os.Executable()
	if err != nil {
		return "Voxt"
	}
	return executable
}

func (a *app) historySocket() string { return filepath.Join(a.home(), historyapi.SocketName) }

func (a *app) openHistoryAccess() {
	documents := a.lib.store.Documents()
	a.async(func() {
		document := loadHistoryReads(documents)
		a.post(func() { a.histAccess.reads = document.Reads })
	})
	a.syncHistoryAccess()
}

func (a *app) syncHistoryAccess() {
	privacy := a.prefs.Privacy
	days := a.prefs.History.Days()
	if a.prefs.History.CleanupOff {
		days = 0
	}
	a.histPolicy.Store(&historyapi.Policy{Selection: privacy.HistorySelection, KeepHistory: privacy.KeepHistory, KeepDays: days})
	access := &a.histAccess
	if privacy.HistoryAccess == (access.ln != nil) {
		return
	}
	if !privacy.HistoryAccess {
		a.stopHistoryAccess()
		return
	}
	access.exe = historyCommandPath()
	listener, err := historyapi.Listen(a.historySocket())
	if err != nil {
		log.Printf("history access: %v", err)
		a.toast(i18n.T("Could not open the connection for other programs: %v", err))
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	server := &historyapi.Server{
		History: a.lib.store.History(),
		Policy:  func() historyapi.Policy { return *a.histPolicy.Load() },
		Audit:   a.historyAudit(a.lib.store.Documents()),
		Peer:    macx.PeerProgram,
	}
	done, previous := make(chan struct{}), access.done
	go func() {
		defer close(done)
		if previous != nil {
			<-previous
		}
		if err := server.Serve(ctx, listener); err != nil {
			log.Printf("history access: %v", err)
		}
	}()
	access.ln, access.stop, access.done = listener, cancel, done
}

func (a *app) stopHistoryAccess() {
	access := &a.histAccess
	if access.ln == nil {
		return
	}
	access.stop()
	access.ln.Close()
	access.ln, access.stop = nil, nil
}

func (a *app) closeHistoryAccess() {
	done := a.histAccess.done
	a.stopHistoryAccess()
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
	}
}

func loadHistoryReads(documents store.Documents) historyReads {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var document historyReads
	data, ok, err := documents.Get(ctx, historyReadsDoc)
	if err != nil || !ok {
		return document
	}
	if err := json.Unmarshal(data, &document); err != nil {
		log.Printf("history access: the read log is damaged: %v", err)
		return historyReads{}
	}
	return document
}

func (d *historyReads) addRead(read historyapi.Read) (isNew bool) {
	d.Reads = slices.Insert(d.Reads, 0, read)
	if len(d.Reads) > historyReadsKept {
		d.Reads = d.Reads[:historyReadsKept]
	}
	name := readerName(read)
	index := slices.Index(d.Seen, name)
	isNew = index < 0
	if index >= 0 {
		d.Seen = slices.Delete(d.Seen, index, index+1)
	}
	d.Seen = append(d.Seen, name)
	if len(d.Seen) > historySeenKept {
		d.Seen = d.Seen[len(d.Seen)-historySeenKept:]
	}
	return isNew
}

func (a *app) historyAudit(documents store.Documents) func(historyapi.Read) {
	return func(read historyapi.Read) {
		document := loadHistoryReads(documents)
		isNew := document.addRead(read)
		data, err := json.Marshal(document)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err = documents.Put(ctx, historyReadsDoc, data)
			cancel()
		}
		if err != nil {
			log.Printf("history access: could not log a read: %v", err)
		}
		a.post(func() {
			a.histAccess.reads = document.Reads
			if isNew {
				a.notify(core.SeverityInfo, i18n.T("%s read the history", readerLabel(read)),
					i18n.T("Programs that read the history are listed in Settings → Privacy."), "voxtnew://settings/privacy")
			}
		})
	}
}

func readerLabel(read historyapi.Read) string {
	return historyReaderLabelFor(i18n.CurrentTranslator(), read)
}

func (a *app) historyAccessModel(now time.Time) historyAccessViewState {
	return buildHistoryAccessViewState(i18n.CurrentTranslator(), historyAccessStateInput{
		On: a.prefs.Privacy.HistoryAccess, Selection: a.histAccessUI.historySelection,
		Ask: a.histAccessUI.ask, ListOpen: a.histAccessUI.listOpen,
		Exe: a.histAccess.exe, Reads: a.histAccess.reads, Now: now,
	})
}

func (a *app) setHistoryAccess(on bool) {
	a.histAccessUI.historyAccess = on
	a.updateSettings(func(settings *settings.Settings) { settings.Privacy.HistoryAccess = on })
}

func (a *app) setHistorySelection(on bool) {
	a.histAccessUI.historySelection = on
	a.updateSettings(func(settings *settings.Settings) { settings.Privacy.HistorySelection = on })
}

func (a *app) copyHistoryAccessCommand() {
	a.copyHistoryText(a.histAccess.exe + ` history export --day "$(date +%F)"`)
}

func (a *app) openHistoryAccessDocs() {
	go mygo.Shell.OpenExternal(historyDocsLink)
}
