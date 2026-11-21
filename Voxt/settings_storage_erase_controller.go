package main

// Settings → Storage's model folder, choosing another one (family spec
// §7.1: the family's models.root), and its Erase everything (family spec
// §7.6 step 3). The models are in the ActNow family's library, not in
// VoxtNew's data folder, and not VoxtNew's alone: erasing removes from it
// each model VoxtNew claims, deleting the files only of those no other
// family app uses, and says which stay; then VoxtNew's files in the
// family directory (its claims, manifest and runtime file) go. Only then
// does VoxtNew restart, and the start erases the data folder and the logs
// (internal/datadir).

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"

	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// modelsDirName is the library's folder made in a folder the user chose
// that holds other files.
const modelsDirName = "Voxt Models"

// modelsTarget is the library's folder for dir, the folder the user
// chose: dir itself when it is empty or already a library (it holds
// models--… folders, as a Hugging Face cache does), else a folder in it.
func modelsTarget(dir string) string {
	ents, _ := os.ReadDir(dir)
	empty := !slices.ContainsFunc(ents, func(e os.DirEntry) bool { return !strings.HasPrefix(e.Name(), ".") })
	if empty || slices.ContainsFunc(ents, func(e os.DirEntry) bool { return strings.HasPrefix(e.Name(), "models--") }) {
		return dir
	}
	return filepath.Join(dir, modelsDirName)
}

// startModelMove makes dir's library the model folder of every app that
// shares it (the family's models.root): the models are brought there
// (models.Manager.MoveTo), the setting written, and VoxtNew restarts to
// use it, as moving the data folder does. The old folder stays as it was:
// another app, or an older version, may still read it.
func (a *app) startModelMove(dir string) {
	from, to := a.models.Hub(), modelsTarget(dir)
	if fammodels.SameHub(from, to) {
		a.toast(i18n.T("The models are already there"))
		return
	}
	if rel, err := filepath.Rel(from, to); err == nil && !strings.HasPrefix(rel, "..") {
		a.toast(i18n.T("Choose a folder outside the model folder."))
		return
	}
	// Erase everything deletes these folders whole, and the model folder
	// holds the other apps' models too.
	for _, d := range []string{a.home(), a.dataDir(), a.stLoc.Previous, a.logDir} {
		if rel, err := filepath.Rel(d, to); d != "" && err == nil && !strings.HasPrefix(rel, "..") {
			a.toast(i18n.T("Choose a folder outside Voxt's data and logs."))
			return
		}
	}
	if slices.ContainsFunc(a.models.Catalog(), func(m models.Model) bool { return a.models.Status(m.ID).Downloading }) {
		a.toast(i18n.T("Wait for the downloads to finish, or cancel them, then try again."))
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.modelMoving, a.modelMoveTo, a.modelMoveCancel = true, to, cancel
	fam, m, prefs := a.fam, a.models, a.prefs
	a.async(func() {
		err := m.MoveTo(ctx, to, fam.Backup(), func() error {
			return fam.Set(ctx, prefs, actnow.KeyModelsRoot, to)
		})
		a.post(func() {
			a.modelMoveCancel = nil
			if err != nil {
				a.modelMoving = false
				if ctx.Err() == nil {
					log.Printf("models: moving the library to %s: %v", to, err)
					a.toast(i18n.T("The models were not moved: %v", err))
				}
				return
			}
			appRelaunch() // the start opens the library in its new folder
		})
	})
}

// askErase opens Erase everything's confirmation, and weighs which of
// VoxtNew's models other apps use (they stay).
func (a *app) askErase() {
	a.resetAsk, a.resetText = "everything", ""
	a.eraseKept, a.eraseKeptBytes = nil, 0
	m, ids := a.models, a.claimedModels()
	a.async(func() {
		var kept []string
		var bytes int64
		for _, id := range ids {
			if d, err := m.Decide(id); err == nil && d.Mine && d.InUse() {
				kept = append(kept, id)
				bytes += d.Status.Bytes
			}
		}
		a.post(func() { a.eraseKept, a.eraseKeptBytes = kept, bytes })
	})
}

// claimedModels are the models VoxtNew claims, and those it began to
// download (what erasing removes from the library).
func (a *app) claimedModels() []string {
	var ids []string
	for _, m := range a.models.Catalog() {
		st := a.modelStatus[m.ID]
		if st.Mine || (!st.OnDisk && st.Done > 0 && st.Elsewhere == "") {
			ids = append(ids, m.ID)
		}
	}
	return ids
}

// eraseAll removes VoxtNew's models from the family's library and its
// files from the family directory, then restarts VoxtNew: the start
// erases the data and the logs (internal/datadir).
func (a *app) eraseAll() {
	home, logs := a.home(), a.logDir
	ids, fam := a.claimedModels(), a.fam
	del := a.models.Delete
	if a.localLib != nil {
		del = a.localLib.Delete // the runtimes let the models go first
	}
	a.resetAsk, a.erasing = "", true
	h := a.lib.store.History()
	a.async(func() {
		if h.AudioDir() != h.DefaultAudioDir() {
			// Recordings outside the data folder (history_settings.go):
			// the start's erase does not reach them. One that stays
			// stops the erase, before anything else is gone.
			if _, err := h.DeleteAudio(context.Background()); err != nil {
				log.Printf("erase: recordings: %v", err)
				a.post(func() {
					a.erasing = false
					a.toast(i18n.T("Could not erase: %v", err))
				})
				return
			}
		}
		eraseFamily(fam, del, ids)
		err := datadir.WriteJSON(filepath.Join(home, datadir.ResetFile), datadir.PendingReset{Everything: true, Also: []string{logs}})
		a.post(func() {
			a.erasing = false
			if err != nil {
				a.toast(i18n.T("Could not erase: %v", err))
				return
			}
			appRelaunch()
		})
	})
}

// eraseFamily removes the models ids from VoxtNew (their files only when
// no other app uses them: models in use elsewhere stay, and so does one
// whose lock another app holds), then VoxtNew's claims, manifest and
// runtime file. Off the main thread.
func eraseFamily(fam *actnow.Family, del func(context.Context, string, fammodels.Scope) ([]fammodels.Result, error), ids []string) {
	ctx := context.Background()
	for _, id := range ids {
		if _, err := del(ctx, id, fammodels.ThisApp); err != nil {
			log.Printf("erase: model %s: %v", id, err)
		}
	}
	if fam.ReadOnly() {
		return
	}
	if err := fam.Store().Remove(); err != nil {
		log.Printf("erase: claims: %v", err)
	}
	if err := registry.Remove(fam.Dir(), fam.BundleID()); err != nil {
		log.Printf("erase: manifest: %v", err)
	}
	if err := famruntime.Remove(fam.Dir(), fam.BundleID()); err != nil {
		log.Printf("erase: runtime file: %v", err)
	}
}

func (a *app) chooseStorageModelFolder() {
	chooseDir(a, i18n.T("Keep the models in…"), i18n.T("Use this folder"), a.startModelMove)
}

func (a *app) cancelStorageModelMove() {
	if a.modelMoveCancel != nil {
		a.modelMoveCancel()
	}
}

func (a *app) setStorageModelBackup(on bool) {
	fam := a.fam
	a.async(func() {
		if err := fam.SetBackup(context.Background(), on); err != nil {
			log.Printf("family: Time Machine: %v", err)
		}
	})
}
