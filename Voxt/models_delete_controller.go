package main

import (
	"context"
	"strings"

	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
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
	if a.localLib != nil {
		del = a.localLib.Delete
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

func (a *app) discardModelDownload(id string) {
	a.deleteModel(id, fammodels.ThisApp)
}

func (a *app) confirmModelDelete(id string, allApps bool) {
	scope := fammodels.ThisApp
	if allApps {
		scope = fammodels.AllApps
	}
	a.deleteModel(id, scope)
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
		return i18n.T("%s was removed. Its files stay because %s uses them too.", name, uitext.Names(names))
	}
	if apps := r.Decision.UserNames(); len(apps) > 0 {
		return i18n.T("%s was removed from Voxt. Its files stay on this Mac for %s.", name, uitext.Names(apps))
	}
	return ""
}

// useModel claims a model another app installed for VoxtNew ("Use it"),
// with what it needs that is on disk, then downloads what it needs that
// is not (a speech model's voice detection), as the setup guide's obUse.
func (a *app) useModel(id string) {
	m := a.models
	ids, fetch := []string{id}, []string(nil)
	mod, _ := models.Lookup(id)
	for _, need := range mod.Needs {
		switch st := a.modelStatus[need]; {
		case st.Installed, st.Downloading:
		case st.OnDisk:
			ids = append(ids, need)
		default:
			fetch = append(fetch, need)
		}
	}
	a.async(func() {
		err := m.Use(context.Background(), ids...)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not use the model: %v", err))
			}
			a.modelsChanged()
			for _, need := range fetch {
				a.downloadModel(need)
			}
		})
	})
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
	return i18n.T("Other apps also use these models, so they are kept: %s (%s)",
		strings.Join(names, i18n.X("list", ", ")), uitext.HumanBytes(bytes))
}
