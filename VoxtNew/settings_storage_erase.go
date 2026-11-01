package main

// Settings → Storage's model folder and its Erase everything (family spec
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
	"path/filepath"
	"strings"

	"github.com/egoist/mygo/ui"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/datadir"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// modelFolderRows are the model library's folder, shared by the family,
// and the room VoxtNew's models take in it (r nil: still measuring). With
// no other ActNow app installed and nothing shared, the family goes
// unsaid: one row of VoxtNew's models.
func (a *app) modelFolderRows(c *ui.Context, r *storageReport, open func(key, dir string)) {
	t := c.Theme()
	hub := a.models.Hub()
	others := a.familyOthers()
	note := ""
	if others {
		note = i18n.T("Shared by all ActNow apps")
	}
	sectionTitle(c, i18n.T("Model folder"))
	settingRowNote(c, "st-models", uitext.ShortPath(hub), note, func() {
		open("st-open-models", hub)
	})
	if fam := a.fam; fam != nil && !fam.ReadOnly() {
		// Time Machine (family spec §7.12): the family's models.backup,
		// off by default.
		on := fam.Backup()
		hint := i18n.T("Models can be downloaded again, so Time Machine leaves them out.")
		if others {
			hint = i18n.T("Models can be downloaded again, so Time Machine leaves them out. Applies to all ActNow apps.")
		}
		settingRow(c, "st-models-backup", i18n.T("Back up models with Time Machine"), hint, func() {
			if ui.Switch(c.Key("st-models-backup-switch"), &on).Changed() {
				a.async(func() {
					if err := fam.SetBackup(context.Background(), on); err != nil {
						log.Printf("family: Time Machine: %v", err)
					}
				})
			}
		})
	}
	if r == nil {
		return
	}
	type useRow struct {
		title string
		bytes int64
	}
	rows := []useRow{{i18n.T("Used by VoxtNew"), r.modelsMine}}
	if others || r.modelsShared > 0 {
		rows = []useRow{{i18n.T("Only VoxtNew"), r.modelsMine}, {i18n.T("Shared with other ActNow apps"), r.modelsShared}}
	}
	ui.Grid(c.Key("st-models-use")).ColumnTracks(ui.Fr(1), ui.Fixed(px(t, 90)), ui.Fixed(px(t, 80))).GapX(px(t, sp3)).GapY(px(t, 6)).AlignItems(ui.Center).Children(func() {
		for _, row := range rows {
			ui.Text(c, row.title)
			ui.Text(c, uitext.HumanBytes(row.bytes)).TextColor(t.TextMuted).TextAlign(ui.End)
			ui.Box(c)
		}
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
	if lib, ok := a.local.(*local.Library); ok {
		del = lib.Delete // the runtimes let the models go first
	}
	a.resetAsk, a.erasing = "", true
	h := a.lib.store.History()
	a.async(func() {
		eraseFamily(fam, del, ids)
		if h.AudioDir() != h.DefaultAudioDir() {
			// Recordings outside the data folder (history_settings.go):
			// the start's erase does not reach them.
			if _, err := h.DeleteAudio(context.Background()); err != nil {
				log.Printf("erase: recordings: %v", err)
			}
		}
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

// eraseDialog confirms Erase everything.
func (a *app) eraseDialog(c *ui.Context) {
	t := c.Theme()
	erase := a.resetAsk == "everything"
	ui.Modal(c.Key("st-ask-erase"), &erase, func() {
		ui.Text(c, i18n.T("Erase everything?")).Bold()
		ui.Text(c, i18n.T("This deletes your history and its recordings, your dictionary, your app enhancement rules, the local models no other ActNow app uses, every setting and the logs. It cannot be undone. VoxtNew restarts as on its first day, and the models download again.")).Width(px(t, 420))
		if kept := keptModelsText(a.eraseKept, a.eraseKeptBytes); kept != "" { // models_delete.go
			ui.Text(c, kept).Width(px(t, 420))
		}
		ui.Text(c, i18n.T("Type DELETE to confirm.")).FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
		ui.TextInput(c.Key("st-erase-text"), &a.resetText).Placeholder("DELETE").Width(px(t, 200)).Label(i18n.T("Type DELETE to confirm."))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("st-erase-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				a.resetAsk = ""
			}
			if recordButton(c.Key("st-erase-ok"), i18n.T("Erase and restart")).Disabled(strings.TrimSpace(a.resetText) != "DELETE" || a.erasing).Clicked() {
				a.eraseAll()
			}
		})
	})
	if !erase && a.resetAsk == "everything" {
		a.resetAsk = ""
	}
}
