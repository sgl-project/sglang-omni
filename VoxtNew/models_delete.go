package main

// Deleting a model from the Local models page by the ActNow family's
// rules (family spec §7.6, §7.10.3): VoxtNew first weighs what deleting
// would do (models.Manager.Decide, off the main thread). When no other
// app uses the model, the usual confirmation deletes it; when one does,
// "Remove from VoxtNew" (the default) drops VoxtNew's claim and keeps
// the files for it, and "Delete for all apps…" asks once more, then
// deletes the files whoever uses them (the others see a tombstone). The
// model's runtime lets it go first (local.Library.Delete).

import (
	"context"
	"strings"

	"github.com/egoist/mygo/ui"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// sharedDelete is the deletion of a model other apps use, as confirmed.
type sharedDelete struct {
	id    string
	apps  []string // the other apps' names
	bytes int64    // its files
	// forAll: the second step, "Delete for all apps…" confirming.
	forAll bool
}

// askDeleteModel weighs deleting model id, then asks: the plain
// confirmation, or the family's when another app uses it.
func (a *app) askDeleteModel(id string) {
	if a.modelDeciding != "" {
		return
	}
	a.modelDeciding = id
	m := a.models
	a.async(func() {
		d, err := m.Decide(id)
		a.post(func() {
			a.modelDeciding = ""
			switch {
			case err != nil:
				a.toast(i18n.T("Could not delete the model: %v", err))
			case d.InUse():
				a.modelShared = &sharedDelete{id: id, apps: d.UserNames(), bytes: d.Status.Bytes}
			default:
				a.modelAsk, a.modelAskFree = id, d.FreeBytes
			}
		})
	})
}

// deleteModel removes model id from VoxtNew (scope: its files only when
// no other ActNow app uses them, or for all apps), after its runtime let
// it go (local.Library.Delete), and says what happened when the files
// stayed.
func (a *app) deleteModel(id string, scope fammodels.Scope) {
	del := a.models.Delete
	if lib, ok := a.local.(*local.Library); ok {
		del = lib.Delete
	}
	a.async(func() {
		res, err := del(context.Background(), id, scope)
		a.post(func() {
			if msg := deleteOutcome(id, res, err); msg != "" {
				a.toast(msg)
			}
			a.modelsChanged()
		})
	})
}

// deleteOutcome is what to tell after deleting model id ("" nothing): a
// failure (another app holds the model), or why its files stayed.
func deleteOutcome(id string, res []fammodels.Result, err error) string {
	name := uitext.ModelName(id)
	if bid, busy := models.Busy(err); busy {
		if bid == "" {
			return i18n.T("Another app is downloading or using this model. Try again later.")
		}
		return i18n.T("%s is downloading or using this model. Try again later.", ownerName(bid))
	}
	if err != nil {
		return i18n.T("Could not delete the model: %v", err)
	}
	if len(res) == 0 || res[0].Deleted {
		return ""
	}
	r := res[0]
	if len(r.OwnModels) > 0 {
		var names []string
		for _, m := range r.OwnModels {
			names = append(names, uitext.ModelName(m))
		}
		return i18n.T("%s was removed. Its files stay: %s uses them too.", name, uitext.Names(names))
	}
	if apps := r.Decision.UserNames(); len(apps) > 0 {
		return i18n.T("%s was removed from VoxtNew. Its files stay on this Mac for %s.", name, uitext.Names(apps))
	}
	return ""
}

// useModel claims a model another app installed for VoxtNew ("Use it").
func (a *app) useModel(id string) {
	m := a.models
	a.async(func() {
		err := m.Use(context.Background(), id)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not use the model: %v", err))
			}
			a.modelsChanged()
		})
	})
}

// modelDeleteDialogs are the confirmations of a deletion.
func (a *app) modelDeleteDialogs(c *ui.Context) {
	a.modelDeleteDialog(c)
	a.modelSharedDialog(c)
}

// modelDeleteDialog confirms deleting a model no other app uses.
func (a *app) modelDeleteDialog(c *ui.Context) {
	t := c.Theme()
	open := a.modelAsk != ""
	id := a.modelAsk
	ui.Modal(c.Key("model-ask-delete"), &open, func() {
		m, _ := models.Lookup(id)
		free := a.modelAskFree
		if free <= 0 {
			free = m.Size()
		}
		ui.Text(c, i18n.T("Delete %s?", uitext.ModelName(id))).Bold()
		ui.Text(c, i18n.T("This frees %s. You can download it again later.", uitext.HumanBytes(free))).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("model-ask-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				a.modelAsk = ""
			}
			if recordButton(c.Key("model-ask-ok"), i18n.T("Delete")).Label(i18n.T("Delete %s", uitext.ModelName(id))).Clicked() {
				a.modelAsk = ""
				a.deleteModel(id, fammodels.ThisApp)
			}
		})
	})
	if !open && a.modelAsk != "" {
		a.modelAsk = ""
	}
}

// modelSharedDialog confirms deleting a model other apps use too (spec
// §7.10.3): Remove from VoxtNew (the default) keeps its files; Delete for
// all apps… asks again, then deletes them.
func (a *app) modelSharedDialog(c *ui.Context) {
	t := c.Theme()
	sd := a.modelShared
	open := sd != nil
	ui.Modal(c.Key("model-ask-shared"), &open, func() {
		if sd == nil {
			return
		}
		name, apps, size := uitext.ModelName(sd.id), uitext.Names(sd.apps), uitext.HumanBytes(sd.bytes)
		if sd.forAll {
			ui.Text(c, i18n.T("Delete \"%s\" for all apps?", name)).Bold()
			ui.Text(c, i18n.T("%s will need to download it again (%s).", apps, size)).Width(px(t, 400))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("model-all-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					a.modelShared = nil
				}
				if recordButton(c.Key("model-all-ok"), i18n.T("Delete")).Label(i18n.T("Delete %s for all apps", name)).Clicked() {
					a.modelShared = nil
					a.deleteModel(sd.id, fammodels.AllApps)
				}
			})
			return
		}
		ui.Text(c, i18n.T("Remove \"%s\" from VoxtNew?", name)).Bold()
		ui.Text(c, i18n.N(len(sd.apps), "%s also uses this model, so its files (%s) stay on this Mac.",
			"%s also use this model, so its files (%s) stay on this Mac.", apps, size)).Width(px(t, 400))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Wrap().Children(func() {
			if ui.Button(c.Key("model-shared-cancel"), i18n.T("Cancel")).Clicked() {
				a.modelShared = nil
			}
			if ui.Button(c.Key("model-shared-all"), i18n.T("Delete for all apps…")).Clicked() {
				next := *sd
				next.forAll = true
				a.modelShared = &next
			}
			if primaryButton(c.Key("model-shared-remove"), i18n.T("Remove from VoxtNew")).AutoFocus().Clicked() {
				a.modelShared = nil
				a.deleteModel(sd.id, fammodels.ThisApp)
			}
		})
	})
	if !open && a.modelShared != nil {
		a.modelShared = nil
	}
}

// keptModelsText is the erase confirmation's line on the models other
// apps use, which stay (spec §7.6 step 3; "" none).
func keptModelsText(ids []string, bytes int64) string {
	if len(ids) == 0 {
		return ""
	}
	var names []string
	for _, id := range ids {
		names = append(names, uitext.ModelName(id))
	}
	return i18n.T("These models are also used by other ActNow apps and will be kept: %s (%s)",
		strings.Join(names, i18n.X("list", ", ")), uitext.HumanBytes(bytes))
}
