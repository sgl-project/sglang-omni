package main

// Settings → Storage (AudioSmith's, for VoxtNew's files): what takes
// room, opening the folders, moving the data folder (internal/datadir) and
// the reset tiers: the look, all settings, everything
// (settings_storage_erase.go). The local models are not in the data
// folder: they are in the ActNow family's library ($HUB, by default
// ~/.cache/huggingface/hub), which every family app shares; the page
// shows how much of it is VoxtNew's alone and how much it shares.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/datadir"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/recordings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// storageState is the Storage page's state.
type storageState struct {
	dataHome string // the default data folder (location.json lives there); "" = the one in use
	logDir   string

	stReport  *storageReport
	stLoading bool
	stLoc     datadir.Location

	moving     bool
	moveDone   int64
	moveTotal  int64
	moveTo     string
	moveErr    string
	moveCancel context.CancelFunc

	resetAsk  string // "settings" | "everything": the confirmation shown
	resetText string // typed confirmation
	// eraseKept are the models other apps use, which erasing keeps
	// (eraseKeptBytes their files); erasing runs.
	eraseKept      []string
	eraseKeptBytes int64
	erasing        bool
}

// storageReport is what takes room.
type storageReport struct {
	cats []storageCat
	data int64 // the data folder, all of it
	free int64 // free space on the data folder's disk
	// models is the model library's folder; modelsMine and modelsShared
	// the files of the models VoxtNew claims, only its own and shared with
	// other ActNow apps (blobs counted once).
	models                   string
	modelsMine, modelsShared int64
}

// storageCat is a category of files.
type storageCat struct {
	id, title string
	bytes     int64
	dir       string // to open; "" = none
}

// Test seams: relaunching the app, choosing a folder.
var (
	appRelaunch   = func() { mygo.App.Relaunch() }
	chooseMoveDir = func(a *app, then func(string)) {
		go func() {
			dirs, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Move the data folder to…"),
				Directory: true, CreateDirectories: true, ButtonLabel: i18n.T("Move here")})
			if err != nil || len(dirs) == 0 {
				return
			}
			a.post(func() { then(dirs[0]) })
		}()
	}
)

func (a *app) dataDir() string { return a.lib.store.Dir() }

// home is the default data folder.
func (a *app) home() string {
	if a.dataHome != "" {
		return a.dataHome
	}
	return a.dataDir()
}

// diskFree is the space free for the user on path's disk (a variable:
// snapshots pin it, snapshot_env_test.go).
var diskFree = func(path string) int64 {
	var st syscall.Statfs_t
	for p := path; p != "" && p != "/" && p != "."; p = filepath.Dir(p) {
		if syscall.Statfs(p, &st) == nil {
			return int64(st.Bavail) * int64(st.Bsize)
		}
	}
	if syscall.Statfs("/", &st) == nil {
		return int64(st.Bavail) * int64(st.Bsize)
	}
	return 0
}

// loadStorage measures the folders in the background.
func (a *app) loadStorage() {
	if a.stLoading {
		return
	}
	a.stLoading = true
	hist := a.lib.store.History()
	data, logs, audio := a.dataDir(), a.logDir, hist.AudioDir()
	m := a.models
	family := a.fam.Dir().Root
	a.stLoc = datadir.ReadLocation(a.home())
	a.async(func() {
		ctx := context.Background()
		r := &storageReport{models: m.Hub()}
		if u, err := m.Usage(); err == nil {
			r.modelsMine, r.modelsShared = u.Mine, u.Shared
		}
		rec, _ := datadir.TreeSize(ctx, audio, nil)
		if audio != hist.DefaultAudioDir() {
			// A folder the user chose may hold other files: only the
			// recordings count (internal/recordings).
			_, rec, _ = recordings.Usage(ctx, hist)
		}
		r.cats = append(r.cats, storageCat{id: "audio", title: i18n.T("Recordings kept with the history"), bytes: rec, dir: audio})
		var db int64
		if ents, err := os.ReadDir(data); err == nil {
			for _, e := range ents {
				if strings.HasPrefix(e.Name(), sqlitestore.FileName) {
					if fi, err := e.Info(); err == nil {
						db += fi.Size()
					}
				}
			}
		}
		r.cats = append(r.cats, storageCat{id: "database", title: i18n.T("Library database"), bytes: db})
		r.data, _ = datadir.TreeSize(ctx, data, outside(data, r.models, family))
		inData := rec // the recordings, when they are in the data folder
		if rel, err := filepath.Rel(data, audio); err != nil || strings.HasPrefix(rel, "..") {
			inData = 0
		}
		r.cats = append(r.cats, storageCat{id: "other", title: i18n.T("Other"), bytes: max(r.data-inData-db, 0), dir: data})
		if logs != "" {
			lb, _ := datadir.TreeSize(ctx, logs, nil)
			r.cats = append(r.cats, storageCat{id: "logs", title: i18n.T("Logs"), bytes: lb, dir: logs})
		}
		r.free = diskFree(data)
		a.post(func() { a.stReport, a.stLoading = r, false })
	})
}

// outside skips, in a tree size of data, the folders that are not its
// though they are in it (the model library and the family directory are
// elsewhere but in tests).
func outside(data string, dirs ...string) func(rel string) bool {
	var rels []string
	for _, d := range dirs {
		if rel, err := filepath.Rel(data, d); err == nil && rel != "." && !strings.HasPrefix(rel, "..") {
			rels = append(rels, rel)
		}
	}
	return func(rel string) bool { return slices.Contains(rels, rel) }
}

// startMove copies the data folder to dir/VoxtNew (or dir, when it is
// called VoxtNew or empty), then restarts to finish (internal/datadir).
func (a *app) startMove(dir string) {
	from := a.dataDir()
	to := dir
	if filepath.Base(dir) != "VoxtNew" {
		if ents, err := os.ReadDir(dir); err != nil || len(ents) > 0 {
			to = filepath.Join(dir, "VoxtNew")
		}
	}
	if filepath.Clean(to) == filepath.Clean(a.home()) && from != a.home() {
		// Back to the default folder: it keeps location.json.
	} else if err := datadir.CheckMoveTarget(from, to); err != nil {
		a.toast(i18n.T("Cannot move there: %v", err))
		return
	}
	need, _ := datadir.TreeSize(context.Background(), from, nil)
	if free := diskFree(filepath.Dir(to)); free > 0 && free < need+need/20+(256<<20) {
		a.toast(i18n.T("Not enough space there: %s needed, %s free.", uitext.HumanBytes(need), uitext.HumanBytes(free)))
		return
	}
	_, existed := os.Stat(to)
	ctx, cancel := context.WithCancel(context.Background())
	a.moving, a.moveTo, a.moveErr, a.moveDone, a.moveTotal, a.moveCancel = true, to, "", 0, need, cancel
	home := a.home()
	a.async(func() {
		skip := func(rel string) bool { return datadir.SkipData(rel, true) }
		err := datadir.CopyTree(ctx, from, to, skip, func(done, total int64) {
			a.post(func() { a.moveDone, a.moveTotal = done, total })
		})
		if err == nil {
			err = datadir.WriteJSON(filepath.Join(home, datadir.MoveFile), datadir.PendingMove{From: from, To: to})
		}
		a.post(func() {
			a.moveCancel = nil
			if err != nil {
				a.moving = false
				if existed != nil && to != home { // created by the move: roll back
					os.RemoveAll(to)
				} else {
					datadir.RemoveExcept(to)
				}
				if ctx.Err() == nil {
					a.moveErr = i18n.T("The data was not moved: %v", err)
				}
				return
			}
			appRelaunch() // the start finishes the move (datadir.Resolve)
		})
	})
}

// removePrevious deletes the folder the data was moved from.
func (a *app) removePrevious() {
	home, loc := a.home(), a.stLoc
	if loc.Previous == "" {
		return
	}
	a.async(func() {
		var err error
		if filepath.Clean(loc.Previous) == filepath.Clean(home) {
			err = datadir.RemoveExcept(home)
		} else {
			err = os.RemoveAll(loc.Previous)
		}
		loc.Previous = ""
		if err == nil {
			err = datadir.WriteJSON(filepath.Join(home, datadir.LocationFile), loc)
		}
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not remove it: %v", err))
			}
			a.stLoc = loc
			a.stReport = nil
		})
	})
}

// dismissMoveFailure forgets the last failed move.
func (a *app) dismissMoveFailure() {
	loc := a.stLoc
	loc.Failed = ""
	a.stLoc = loc
	home := a.home()
	a.async(func() { datadir.WriteJSON(filepath.Join(home, datadir.LocationFile), loc) })
}

// resetLook restores the appearance and the language.
func (a *app) resetLook() {
	d := settings.Defaults()
	a.updateSettings(func(s *settings.Settings) { s.Appearance, s.General.Language = d.Appearance, d.General.Language })
	a.sideW, a.navOpen = 0, nil
	a.toast(i18n.T("The look is back to its defaults"))
}

// resetSettings restores every setting (the setup stays done).
func (a *app) resetSettings() {
	a.updateSettings(func(s *settings.Settings) {
		*s = settings.Defaults()
		s.Setup.Done = true
	})
	a.sideW, a.navOpen = 0, nil
	a.toast(i18n.T("Every setting is back to its default"))
}

// storageSettings is Settings → Storage.
func (a *app) storageSettings(c *ui.Context) {
	t := c.Theme()
	if a.stReport == nil && !a.stLoading {
		a.loadStorage()
	}
	r := a.stReport
	muted := func(s string) { ui.Text(c, s).FontSize(px(t, fsCaption)).TextColor(t.TextMuted) }
	open := func(key, dir string) {
		if dir != "" && a.win != nil && ui.Button(c.Key(key), i18n.T("Open")).Clicked() {
			mygo.Shell.OpenPath(dir)
		}
	}

	sectionTitle(c, i18n.T("Data folder"))
	settingRow(c, "st-data", uitext.ShortPath(a.dataDir()), i18n.T("Your history and its recordings, your dictionary and the database."), func() {
		open("st-open-data", a.dataDir())
		if ui.Button(c.Key("st-move"), i18n.T("Move…")).Disabled(a.moving).Clicked() {
			chooseMoveDir(a, a.startMove)
		}
	})
	if dir := a.prefs.History.AudioDir; dir != "" {
		// Recordings kept elsewhere (history_settings.go).
		settingRowNote(c, "st-audio", i18n.T("Recordings folder"), uitext.ShortPath(dir), func() {
			if ui.Button(c.Key("st-audio-settings"), i18n.T("History settings…")).Clicked() {
				a.showHistorySettings()
			}
		})
	}
	if l := a.stLoc; l.Previous != "" {
		settingRowNote(c, "st-prev", i18n.T("The previous data folder is still there"), uitext.ShortPath(l.Previous), func() {
			if ui.Button(c.Key("st-prev-rm"), i18n.T("Remove it")).Clicked() {
				a.removePrevious()
			}
		})
	}
	if l := a.stLoc; l.Failed != "" {
		settingRowNote(c, "st-failed", i18n.T("The last move did not happen"), l.Failed, func() {
			if ui.Button(c.Key("st-failed-ok"), i18n.T("Dismiss")).Clicked() {
				a.dismissMoveFailure()
			}
		})
	}
	if a.moveErr != "" {
		ui.Text(c, a.moveErr).FontSize(px(t, fsCaption)).TextColor(t.Danger)
	}

	a.modelFolderRows(c, r, open) // settings_storage_erase.go

	sectionTitle(c, i18n.T("What takes room"))
	if r == nil {
		ui.Row(c).Gap(px(t, sp2)).Children(func() {
			ui.Spinner(c)
			muted(i18n.T("Measuring…"))
		})
	} else {
		muted(i18n.T("Data folder: %s · free on its disk: %s", uitext.HumanBytes(r.data), uitext.HumanBytes(r.free)))
		ui.Grid(c.Key("st-cats")).ColumnTracks(ui.Fr(1), ui.Fixed(px(t, 90)), ui.Fixed(px(t, 80))).GapX(px(t, sp3)).GapY(px(t, 6)).AlignItems(ui.Center).Children(func() {
			for _, cat := range r.cats {
				ui.Text(c, cat.title)
				ui.Text(c, uitext.HumanBytes(cat.bytes)).TextColor(t.TextMuted).TextAlign(ui.End)
				ui.Row(c.Key("cat-" + cat.id)).Children(func() { open("open", cat.dir) })
			}
		})
	}

	sectionTitle(c, i18n.T("Reset"))
	settingRow(c, "st-reset-look", i18n.T("Reset the look"), i18n.T("Theme, size, sidebar and language."), func() {
		if ui.Button(c.Key("st-reset-look"), i18n.T("Reset")).Clicked() {
			a.resetLook()
		}
	})
	settingRow(c, "st-reset-settings", i18n.T("Reset all settings"), i18n.T("Every setting goes back to its default. Your history and dictionary stay."), func() {
		if ui.Button(c.Key("st-reset-settings"), i18n.T("Reset…")).Clicked() {
			a.resetAsk = "settings"
		}
	})
	settingRow(c, "st-erase", i18n.T("Erase everything"), i18n.T("Deletes your history, recordings, dictionary, app enhancement rules, the local models no other ActNow app uses, settings and logs, then restarts VoxtNew as new."), func() {
		if ui.Button(c.Key("st-erase"), i18n.T("Erase…")).Disabled(a.erasing).Clicked() {
			a.askErase()
		}
	})
	a.storageDialogs(c)
}

// storageDialogs are the confirmations and the move's progress.
func (a *app) storageDialogs(c *ui.Context) {
	t := c.Theme()
	open := a.resetAsk == "settings"
	ui.Modal(c.Key("st-ask-settings"), &open, func() {
		ui.Text(c, i18n.T("Reset all settings?")).Bold()
		ui.Text(c, i18n.T("Every setting goes back to its default. Your history and dictionary stay.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("st-ask-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				a.resetAsk = ""
			}
			if ui.Button(c.Key("st-ask-ok"), i18n.T("Reset the settings")).Clicked() {
				a.resetAsk = ""
				a.resetSettings()
			}
		})
	})
	if !open && a.resetAsk == "settings" {
		a.resetAsk = ""
	}
	a.eraseDialog(c) // settings_storage_erase.go
	moving := a.moving
	ui.DialogBase(c.Key("st-moving"), &moving, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.3))
		panel.Width(px(t, 440)).Radius(px(t, radiusCard)).Background(t.Background).Border(1, t.Border).Padding(px(t, sp5)).Label(i18n.T("Moving the data folder"))
		ui.Column(c).Gap(px(t, 10)).Children(func() {
			ui.Text(c, i18n.T("Moving the data folder")).Bold()
			ui.Text(c, i18n.T("Copying to %s. VoxtNew restarts when the copy is checked, and finishes the move as it starts.", uitext.ShortPath(a.moveTo))).FontSize(px(t, fsCaption))
			frac := 0.0
			if a.moveTotal > 0 {
				frac = float64(a.moveDone) / float64(a.moveTotal)
			}
			ui.Progress(c, frac)
			ui.Text(c, fmt.Sprintf("%s / %s", uitext.HumanBytes(a.moveDone), uitext.HumanBytes(a.moveTotal))).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
			ui.Row(c).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("st-move-cancel"), i18n.T("Cancel")).Disabled(a.moveCancel == nil).Clicked() && a.moveCancel != nil {
					a.moveCancel()
				}
			})
		})
	})
}
