package main

// Settings → Storage (AudioSmith's, for VoxtNew's files): what takes
// room, opening the folders, moving the data folder (internal/datadir) and
// the reset tiers: the look, all settings, everything
// (settings_storage_erase_controller.go). The local models are not in the data
// folder: they are in the model library ($HUB, by default
// ~/Library/Application Support/SGLang Omni/models, models.HubIn, or the
// family's models.root, which the family apps share); the page shows how
// much of it is VoxtNew's alone and how much it shares.

import (
	"cmp"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"

	"github.com/egoist/mygo"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/recordings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// storageState is the Storage page's controller and local UI state.
type storageState struct {
	settingsStorageLocalUI

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

	// eraseKept are the models other apps use, which erasing keeps
	// (eraseKeptBytes their files); erasing runs.
	eraseKept      []string
	eraseKeptBytes int64
	erasing        bool

	// The model folder's change (settings_storage_erase_controller.go).
	modelMoving     bool
	modelMoveTo     string
	modelMoveCancel context.CancelFunc
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
	// shared: another app keeps its models in the folder too (its claims
	// name it).
	shared bool
}

// storageCat is a category of files.
type storageCat struct {
	id, title string
	bytes     int64
	dir       string // to open; "" = none
}

// Test seams: relaunching the app, choosing a folder.
var (
	appRelaunch = func() { mygo.App.Relaunch() }
	chooseDir   = func(a *app, title, button string, then func(string)) {
		go func() {
			dirs, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: title,
				Directory: true, CreateDirectories: true, ButtonLabel: button})
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
	if a.stReport != nil || a.stLoading {
		return
	}
	a.stLoading = true
	hist := a.lib.store.History()
	data, logs, audio := a.dataDir(), a.logDir, hist.AudioDir()
	m := a.models
	fam, bundleID := a.fam.Dir(), a.fam.BundleID()
	family := fam.Root
	translator := i18n.CurrentTranslator()
	a.stLoc = datadir.ReadLocation(a.home())
	a.async(func() {
		ctx := context.Background()
		r := &storageReport{models: m.Hub()}
		if u, err := m.Usage(); err == nil {
			r.modelsMine, r.modelsShared = u.Mine, u.Shared
		}
		all, _ := fammodels.ReadAllClaims(fam)
		for id, cs := range all {
			// An empty root is the family's default library.
			if id != bundleID && fammodels.SameHub(cmp.Or(cs.Root, fammodels.DefaultHubIn(fam)), r.models) {
				r.shared = true
			}
		}
		rec, _ := datadir.TreeSize(ctx, audio, nil)
		if audio != hist.DefaultAudioDir() {
			// A folder the user chose may hold other files: only the
			// recordings count (internal/recordings).
			_, rec, _ = recordings.Usage(ctx, hist)
		}
		r.cats = append(r.cats, storageCat{id: "audio", title: translator.T("Recordings kept with the history"), bytes: rec, dir: audio})
		meetDir := filepath.Join(data, "meetings")
		if meet, err := datadir.TreeSize(ctx, meetDir, nil); err == nil && meet > 0 {
			r.cats = append(r.cats, storageCat{id: "meetings", title: translator.T("Meeting recordings"), bytes: meet, dir: meetDir})
		}
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
		r.cats = append(r.cats, storageCat{id: "database", title: translator.T("Library database"), bytes: db})
		r.data, _ = datadir.TreeSize(ctx, data, outside(data, r.models, family))
		inData := rec // the recordings, when they are in the data folder
		if rel, err := filepath.Rel(data, audio); err != nil || strings.HasPrefix(rel, "..") {
			inData = 0
		}
		r.cats = append(r.cats, storageCat{id: "other", title: translator.T("Other"), bytes: max(r.data-inData-db, 0), dir: data})
		if logs != "" {
			lb, _ := datadir.TreeSize(ctx, logs, nil)
			r.cats = append(r.cats, storageCat{id: "logs", title: translator.T("Logs"), bytes: lb, dir: logs})
		}
		r.free = diskFree(data)
		a.post(func() { a.stReport, a.stLoading = r, false })
	})
}

// storageInvalidated measures Storage again when its section is open.
func (a *app) storageInvalidated() {
	a.stReport = nil
	if a.settingsOpen && a.setRouter.Path() == "/storage" {
		a.loadStorage()
	}
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
			a.storageInvalidated()
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
	a.invalidateAppearancePreviewSaves()
	a.updateSettings(func(s *settings.Settings) { s.Appearance, s.General.Language = d.Appearance, d.General.Language })
	a.navUI.sideW, a.navUI.sections = 0, nil
	a.toast(i18n.T("The look is back to its defaults"))
}

// resetSettings restores every setting (the setup stays done).
func (a *app) resetSettings() {
	a.invalidateAppearancePreviewSaves()
	a.updateSettings(func(s *settings.Settings) {
		*s = settings.Defaults()
		s.Setup.Done = true
	})
	a.navUI.sideW, a.navUI.sections = 0, nil
	a.toast(i18n.T("Every setting is back to its default"))
}

func (a *app) storageSettingsModel() settingsStorageViewState {
	canChangeModels, modelBackup := false, false
	if a.fam != nil {
		canChangeModels, modelBackup = !a.fam.ReadOnly(), a.fam.Backup()
	}
	return buildSettingsStorageViewState(i18n.CurrentTranslator(), settingsStorageStateInput{
		HomeDir: a.homeDir, DataDirectory: a.dataDir(), RecordingsDirectory: a.prefs.History.AudioDir,
		PreviousDirectory: a.stLoc.Previous, FailedMove: a.stLoc.Failed, CanOpen: a.win != nil,
		Report: a.stReport, MoveError: a.moveErr, Moving: a.moving,
		MoveDone: a.moveDone, MoveTotal: a.moveTotal, MoveTo: a.moveTo, CanCancelMove: a.moveCancel != nil,
		ResetAsk: a.resetAsk, ResetText: a.resetText, Erasing: a.erasing,
		KeptModelsText: keptModelsText(a.eraseKept, a.eraseKeptBytes), ModelDirectory: a.models.Hub(),
		FamilyOthers: a.familyOthers(), CanChangeModelFolder: canChangeModels, ModelBackup: modelBackup,
		ModelMoving: a.modelMoving, ModelMoveTo: a.modelMoveTo, CanCancelModelMove: a.modelMoveCancel != nil,
	})
}

func (a *app) openStorageFolder(directory string) {
	if a.win != nil {
		mygo.Shell.OpenPath(directory)
	}
}

func (a *app) chooseStorageDataFolder() {
	chooseDir(a, i18n.T("Move the data folder to…"), i18n.T("Move here"), a.startMove)
}

func (a *app) cancelStorageDataMove() {
	if a.moveCancel != nil {
		a.moveCancel()
	}
}
