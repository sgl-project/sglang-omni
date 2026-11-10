package main

// Other programs reading the history (docs/history-access.md): one switch
// in Settings → Privacy, off by default. On, the app serves the socket of
// internal/historyapi in the default data folder; off, the file does not
// exist. Each read is logged without any of its text, and a program name
// not seen before raises a notice.

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	historyReadsDoc  = "history:reads"
	historyReadsKept = 20 // the log dialog's rows
	historySeenKept  = 50 // past it, a forgotten name raises one more notice
	historyDocsLink  = "https://github.com/sgl-project/sglang-omni/blob/main/Voxt/docs/history-access.md"
)

// historyAccessState is the area's state. Main thread only, except
// policy, which the server goroutine reads.
type historyAccessState struct {
	histAccess struct {
		ask, listOpen bool
		exe           string // the command's path (historyCommandPath)
		ln            net.Listener
		stop          context.CancelFunc
		done          chan struct{}
		reads         []historyapi.Read // newest first
	}
	histPolicy atomic.Pointer[historyapi.Policy]
}

// historyReads is the history:reads document.
type historyReads struct {
	Reads []historyapi.Read `json:"reads"`
	Seen  []string          `json:"seen"`
}

// readerName is how the log and the notice name a read's program.
func readerName(r historyapi.Read) string {
	switch {
	case r.As != "" && r.Program != "":
		return r.As + " (" + r.Program + ")"
	case r.As != "":
		return r.As
	}
	return r.Program
}

// historyCommandPath is the executable the Privacy page's command names
// (a variable: snapshots pin it).
var historyCommandPath = func() string {
	exe, err := os.Executable()
	if err != nil {
		return "Voxt"
	}
	return exe
}

// historySocket is the socket's path: the default data folder always, so
// the command finds it when the data folder has moved.
func (a *app) historySocket() string { return filepath.Join(a.home(), historyapi.SocketName) }

// openHistoryAccess runs once, when the app is ready.
func (a *app) openHistoryAccess() {
	docs := a.lib.store.Documents()
	a.async(func() {
		doc := loadHistoryReads(docs)
		a.post(func() { a.histAccess.reads = doc.Reads })
	})
	a.syncHistoryAccess()
}

// syncHistoryAccess makes the socket match the setting. It binds on the
// main thread so that a quick off and on cannot run out of order.
func (a *app) syncHistoryAccess() {
	p := a.prefs.Privacy
	days := a.prefs.History.Days()
	if a.prefs.History.CleanupOff {
		days = 0
	}
	a.histPolicy.Store(&historyapi.Policy{Selection: p.HistorySelection, KeepHistory: p.KeepHistory, KeepDays: days})
	ha := &a.histAccess
	if p.HistoryAccess == (ha.ln != nil) {
		return
	}
	if !p.HistoryAccess {
		a.stopHistoryAccess()
		return
	}
	ha.exe = historyCommandPath()
	ln, err := historyapi.Listen(a.historySocket())
	if err != nil {
		log.Printf("history access: %v", err)
		a.toast(i18n.T("Could not open the connection for other programs: %v", err))
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	srv := &historyapi.Server{
		History: a.lib.store.History(),
		Policy:  func() historyapi.Policy { return *a.histPolicy.Load() },
		Audit:   a.historyAudit(a.lib.store.Documents()),
		Peer:    macx.PeerProgram,
	}
	done, prev := make(chan struct{}), ha.done
	go func() {
		defer close(done)
		if prev != nil {
			<-prev // its last Audit first: one writer of the read log
		}
		if err := srv.Serve(ctx, ln); err != nil {
			log.Printf("history access: %v", err)
		}
	}()
	ha.ln, ha.stop, ha.done = ln, cancel, done
}

// stopHistoryAccess ends a read in progress and removes the socket.
func (a *app) stopHistoryAccess() {
	ha := &a.histAccess
	if ha.ln == nil {
		return
	}
	ha.stop()
	ha.ln.Close() // the file goes now, not when the goroutine notices
	ha.ln, ha.stop = nil, nil
}

// closeHistoryAccess stops it for quitting, so that no read is left on
// the database about to close.
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

// loadHistoryReads reads the log document; a damaged one starts over.
func loadHistoryReads(docs store.Documents) historyReads {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var doc historyReads
	data, ok, err := docs.Get(ctx, historyReadsDoc)
	if err != nil || !ok {
		return doc
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		log.Printf("history access: the read log is damaged: %v", err)
		return historyReads{}
	}
	return doc
}

// addRead logs r and reports whether its name is new; the newest name is
// the last of Seen.
func (d *historyReads) addRead(r historyapi.Read) (isNew bool) {
	d.Reads = slices.Insert(d.Reads, 0, r)
	if len(d.Reads) > historyReadsKept {
		d.Reads = d.Reads[:historyReadsKept]
	}
	name := readerName(r)
	i := slices.Index(d.Seen, name)
	isNew = i < 0
	if i >= 0 {
		d.Seen = slices.Delete(d.Seen, i, i+1)
	}
	d.Seen = append(d.Seen, name)
	if len(d.Seen) > historySeenKept {
		d.Seen = d.Seen[len(d.Seen)-historySeenKept:]
	}
	return isNew
}

// historyAudit is the server's Audit. It runs on the server's goroutine,
// one read at a time, so the document's updates cannot interleave.
func (a *app) historyAudit(docs store.Documents) func(historyapi.Read) {
	return func(r historyapi.Read) {
		doc := loadHistoryReads(docs)
		isNew := doc.addRead(r)
		data, err := json.Marshal(doc)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err = docs.Put(ctx, historyReadsDoc, data)
			cancel()
		}
		if err != nil {
			log.Printf("history access: could not log a read: %v", err)
		}
		a.post(func() {
			a.histAccess.reads = doc.Reads
			if isNew {
				a.notify(core.SeverityInfo, i18n.T("%s read the history", readerLabel(r)),
					i18n.T("Programs that read the history are listed in Settings → Privacy."), "voxtnew://settings/privacy")
			}
		})
	}
}

// readerLabel is readerName for the user.
func readerLabel(r historyapi.Read) string {
	if s := readerName(r); s != "" {
		return s
	}
	return i18n.T("Unknown program")
}

// readLine is a read's row: when, how many, who.
func readLine(r historyapi.Read) string {
	what := i18n.T("Status check")
	if r.Op == historyapi.OpExport {
		what = i18n.N(r.Count, "%d entry", "%d entries", r.Count)
	}
	return strings.Join([]string{i18n.DateTime(r.At), what, readerLabel(r)}, " · ")
}

// setHistoryAccess saves the switch.
func (a *app) setHistoryAccess(on bool) {
	a.settingsForm.historyAccess = on
	a.updateSettings(func(s *settings.Settings) { s.Privacy.HistoryAccess = on })
}

// historyAccessSection is Settings → Privacy's Other programs.
func (a *app) historyAccessSection(c *ui.Context) {
	ha := &a.histAccess
	f := &a.settingsForm
	on := a.prefs.Privacy.HistoryAccess
	sectionTitle(c, i18n.T("Other programs"))
	settingRow(c, "priv-access", i18n.T("Let programs on this Mac read the history"), i18n.T("Off, Voxt's read command does not work. This switch covers that command only: a program running under your account can still open Voxt's data files directly, on or off."), func() {
		if ui.Switch(c.Key("priv-access"), &f.historyAccess).Label(i18n.T("Allow reading")).Changed() {
			if f.historyAccess {
				f.historyAccess = false // only the question turns it on
				ha.ask = true
			} else {
				a.setHistoryAccess(false)
			}
		}
	})
	settingRow(c, "priv-access-sel", i18n.T("Also include the text you selected"), i18n.T("Off, the text you selected in other apps to translate or rewrite stays out of what programs read and of Export text."), func() {
		if ui.Switch(c.Key("priv-access-sel"), &f.historySelection).Label(i18n.T("Include selected text")).Disabled(!on).Changed() {
			v := f.historySelection
			a.updateSettings(func(s *settings.Settings) { s.Privacy.HistorySelection = v })
		}
	})
	if on || len(ha.reads) > 0 {
		last := i18n.T("No reads yet")
		if len(ha.reads) > 0 {
			last = readLine(ha.reads[0])
		}
		settingRowNote(c, "priv-access-last", i18n.T("Last read"), last, func() {
			if ui.Button(c.Key("priv-access-log"), i18n.T("Read log…")).Clicked() {
				ha.listOpen = true
			}
		})
	}
	if on {
		cmd := ha.exe + ` history export --day "$(date +%F)"`
		settingRowNote(c, "priv-access-cmd", i18n.T("Command line"), cmd, func() {
			if ui.Button(c.Key("priv-access-copy"), i18n.T("Copy")).Clicked() {
				c.Services().WriteClipboard(cmd)
				a.toast(i18n.T("Copied"))
			}
			if ghostButton(c.Key("priv-access-docs"), i18n.T("How to use it")).Clicked() {
				go mygo.Shell.OpenExternal(historyDocsLink)
			}
		})
	}
	a.historyAccessDialogs(c)
}

// historyAccessDialogs are the confirmation and the read log.
func (a *app) historyAccessDialogs(c *ui.Context) {
	t := c.Theme()
	ha := &a.histAccess
	if ha.ask {
		ui.Modal(c.Key("priv-access-ask"), &ha.ask, func() {
			ui.Text(c, i18n.T("Let programs on this Mac read the history?")).Bold()
			ui.Text(c, i18n.T("Any program you run in this account can then read the text Voxt transcribed, translated and rewrote, what it heard and the names of the apps, never the recordings or web addresses. Where a program sends that text afterwards, a cloud model included, is up to the program: Voxt cannot see or stop it.")).Width(px(t, 380))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("priv-access-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					ha.ask = false
				}
				if primaryButton(c.Key("priv-access-ok"), i18n.T("Turn on")).Clicked() {
					ha.ask = false
					a.setHistoryAccess(true)
				}
			})
		})
	}
	if ha.listOpen {
		dialogFrame(c, "priv-access-reads", &ha.listOpen, i18n.T("Read log"), i18n.T("The last reads by other programs: when, how many entries and which program. No text is kept here."), 480, func() {
			if len(ha.reads) == 0 {
				caption(c, i18n.T("No reads yet"))
			}
			for _, r := range ha.reads {
				caption(c, readLine(r))
			}
		}, func() {
			ui.Spacer(c)
			if primaryButton(c.Key("priv-access-reads-done"), i18n.T("Done")).Clicked() {
				ha.listOpen = false
			}
		})
	}
}
