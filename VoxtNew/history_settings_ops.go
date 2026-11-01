package main

// The History settings dialog's work (history_settings.go), off the main
// thread: the automatic clean-up (settings.History, the store's Prune),
// the recordings folder (where it is, missing at the start, moving it),
// their size, export and deleting them all (internal/recordings).
//
// The clean-up runs at the start, after each change of the history (a
// new entry: historyChanged's observers, events.go) and when its setting
// changes. Each entry it deletes goes with its recording, kept or kept
// for a while (a failed recognition's); the history's observers hear of
// it.

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/recordings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// chooseFolder asks for a folder (title, the button's label) and gives
// it to then on the main thread; tests replace it.
var chooseFolder = func(a *app, title, button string, then func(string)) {
	go func() {
		dirs, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: title,
			Directory: true, CreateDirectories: true, ButtonLabel: button})
		if err != nil || len(dirs) == 0 {
			return
		}
		a.post(func() { then(dirs[0]) })
	}()
}

// retention is what the history keeps under h at now.
func retention(h settings.History, now time.Time) store.Retention {
	return store.Retention{Before: h.Cutoff(now), PerKind: h.PerKind()}
}

// openHistorySettings applies the saved recordings folder and runs the
// clean-up once, as the app starts (newApp).
func (a *app) openHistorySettings() {
	a.applyAudioDir()
	a.pruneHistory()
}

// historySettingsChanged applies a change of settings.History
// (settingsChanged).
func (a *app) historySettingsChanged(old, new settings.Settings) {
	if old.History.AudioDir != new.History.AudioDir {
		a.applyAudioDir()
	}
	if !old.History.SameCleanup(new.History) {
		a.pruneHistory()
	}
}

// pruneHistory runs the clean-up in the background; one at a time, and
// once more when asked while it runs.
func (a *app) pruneHistory() {
	hs := &a.histSet
	if hs.pruning {
		hs.pruneAgain = true
		return
	}
	rt := retention(a.prefs.History, time.Now())
	if rt.Zero() {
		return
	}
	hs.pruning = true
	h := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		n, err := h.Prune(ctx, rt)
		a.post(func() {
			hs := &a.histSet
			hs.pruning = false
			if err != nil {
				log.Printf("history: clean-up: %v", err)
			}
			if n > 0 {
				log.Printf("history: the clean-up deleted %d entries", n)
				hs.usageFor = "" // the dialog measures again
				a.historyChanged()
			}
			if hs.pruneAgain {
				hs.pruneAgain = false
				a.pruneHistory()
			}
		})
	})
}

// askCleanup saves a new clean-up setting, after asking when it would
// delete entries now (the dialog's question shows their number).
func (a *app) askCleanup(next settings.History) {
	rt := retention(next, time.Now())
	h := a.lib.store.History()
	a.async(func() {
		n, err := h.CountPrune(context.Background(), rt)
		a.post(func() {
			hs := &a.histSet
			if err != nil {
				a.toast(i18n.T("Could not read the history: %v", err))
				hs.formFor = "" // the controls show the setting again
				return
			}
			if n > 0 {
				hs.pruneAsk, hs.pruneN, hs.pruneTo = true, n, next
				return
			}
			a.saveCleanup(next)
		})
	})
}

// saveCleanup saves the clean-up's part of h (the folder stays).
func (a *app) saveCleanup(h settings.History) {
	a.updateSettings(func(s *settings.Settings) {
		s.History.CleanupOff, s.History.KeepDays, s.History.KeepCount = h.CleanupOff, h.KeepDays, h.KeepCount
	})
	a.histSet.formFor = ""
}

// applyAudioDir makes the saved recordings folder the store's, or, when
// it is missing (a drive not connected), the default one until it is
// back: the dialog and the History page say so.
func (a *app) applyAudioDir() {
	dir := a.prefs.History.AudioDir
	h := a.lib.store.History()
	a.async(func() {
		use, missing := dir, ""
		if dir != "" {
			if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
				use, missing = "", dir
				log.Printf("history: the recordings folder %s is not available: recording into %s", dir, h.DefaultAudioDir())
			}
		}
		h.SetAudioDir(use)
		a.post(func() {
			if a.prefs.History.AudioDir == dir {
				a.histSet.audioMissing = missing
			}
		})
	})
}

// loadRecordingUsage measures the recordings for the dialog.
func (a *app) loadRecordingUsage() {
	hs := &a.histSet
	hs.usageSeq++
	seq := hs.usageSeq
	hs.usageFor = "loading"
	h := a.lib.store.History()
	a.async(func() {
		n, bytes, err := recordings.Usage(context.Background(), h)
		a.post(func() {
			hs := &a.histSet
			if seq != hs.usageSeq {
				return
			}
			hs.usageN, hs.usageBytes, hs.usageFor = n, bytes, "done"
			if err != nil {
				hs.usageN, hs.usageBytes = -1, 0
			}
		})
	})
}

// startWork marks the dialog busy with a cancellable job, and returns its
// context and a progress callback for its goroutine.
func (a *app) startWork(what string) (context.Context, func(done, total int)) {
	ctx, cancel := context.WithCancel(context.Background())
	hs := &a.histSet
	hs.busy, hs.done, hs.total, hs.cancel = what, 0, 0, cancel
	last := -1
	return ctx, func(done, total int) {
		if done == last {
			return
		}
		last = done
		a.post(func() { a.histSet.done, a.histSet.total = done, total })
	}
}

// endWork clears the dialog's job.
func (a *app) endWork() {
	hs := &a.histSet
	if hs.cancel != nil {
		hs.cancel()
	}
	hs.busy, hs.cancel = "", nil
	hs.usageFor = ""
	a.stReport = nil // Settings → Storage measures again
}

// moveRecordings moves the recordings to dir ("" = the default folder).
func (a *app) moveRecordings(dir string) {
	h := a.lib.store.History()
	def := h.DefaultAudioDir()
	if dir == "" {
		dir = def
	}
	saved := ""
	if filepath.Clean(dir) != filepath.Clean(def) {
		saved = dir
	}
	from := h.AudioDir()
	st := a.settings
	commit := func() error {
		h.SetAudioDir(saved)
		err := st.Update(context.Background(), func(s *settings.Settings) { s.History.AudioDir = saved })
		if err != nil {
			h.SetAudioDir(from)
		}
		return err
	}
	if filepath.Clean(from) == filepath.Clean(dir) {
		// Nothing to move (back to the default while the other is missing).
		h.SetAudioDir(saved)
		a.updateSettings(func(s *settings.Settings) { s.History.AudioDir = saved })
		return
	}
	if err := recordings.CheckTarget(from, dir); err != nil {
		a.toast(i18n.T("Cannot move the recordings there: %v", err))
		return
	}
	ctx, progress := a.startWork("move")
	a.async(func() {
		res, err := recordings.Move(ctx, h, dir, commit, progress)
		a.post(func() {
			canceled := ctx.Err() != nil
			a.endWork()
			switch {
			case err != nil && canceled:
				a.toast(i18n.T("The recordings were not moved"))
			case err != nil:
				a.toast(i18n.T("The recordings were not moved: %v", err))
			case res.Missing > 0:
				a.toast(i18n.N(res.Moved, "Moved %d recording to %s", "Moved %d recordings to %s", res.Moved, uitext.ShortPath(dir)) + " · " +
					i18n.N(res.Missing, "%d was not found", "%d were not found", res.Missing))
			default:
				a.toast(i18n.N(res.Moved, "Moved %d recording to %s", "Moved %d recordings to %s", res.Moved, uitext.ShortPath(dir)))
			}
		})
	})
}

// exportRecordings exports into dir the recordings of every entry, or of
// those the History page shows (its tab and search).
func (a *app) exportRecordings(dir string, shown bool) {
	h := a.lib.store.History()
	q := store.HistoryQuery{WithAudio: true}
	if shown {
		q.Kind, q.Search = a.hist.kind(), a.histSearch()
	}
	ctx, progress := a.startWork("export")
	a.async(func() {
		list, err := h.List(ctx, q)
		var res recordings.ExportResult
		if err == nil {
			res, err = recordings.Export(ctx, h, list, dir, progress)
		}
		a.post(func() {
			canceled := ctx.Err() != nil
			a.endWork()
			msg := i18n.N(res.Exported, "Exported %d recording", "Exported %d recordings", res.Exported)
			switch {
			case err != nil && !canceled:
				a.toast(i18n.T("Could not export the recordings: %v", err))
				return
			case canceled:
				msg = i18n.T("Export stopped") + " · " + msg
			}
			if res.Skipped > 0 {
				msg += " · " + i18n.N(res.Skipped, "%d was not found", "%d were not found", res.Skipped)
			}
			if res.Failed > 0 {
				msg += " · " + i18n.N(res.Failed, "%d could not be copied", "%d could not be copied", res.Failed)
			}
			a.toast(msg)
		})
	})
}

// deleteRecordings deletes every recording; the entries stay.
func (a *app) deleteRecordings() {
	h := a.lib.store.History()
	a.async(func() {
		n, err := h.DeleteAudio(context.Background())
		a.post(func() {
			a.histSet.usageFor = ""
			a.stReport = nil
			if err != nil {
				a.toast(i18n.T("Could not delete the recordings: %v", err))
			} else {
				a.toast(i18n.N(n, "Deleted %d recording", "Deleted %d recordings", n))
			}
			if n > 0 {
				a.historyChanged()
			}
		})
	})
}
