package main

// The Speech models page (/models; Local models until 2.1): the speech
// models of this Mac, each a card that says in plain words what it is good
// at and which languages it knows, its size,
// whether it is installed or downloading, with Download, Cancel and
// Delete, and, while it is loaded, the memory it uses (models_memory.go).
// The models live in the ActNow family's library, so a card also says
// which other family apps share the model, or download it, or deleted it
// (models_cards.go); deleting follows the family's rules
// (models_delete.go); Network… sets how downloads reach the Hub
// (models_network.go). The remote speech services follow, as the
// Language models page's cards (remote_page.go), and the language models
// Voxt 2.0 ran on this Mac close it while they are on disk.
//
// The models on disk and their downloads are internal/models' (a.models);
// what is installed, as the sessions open it, internal/local's (a.local,
// models_runtimes.go), with System dictation (internal/dictation) after
// the speech models while the helper can recognise. Views read a.modelStatus, the copy of the
// statuses modelsChanged takes on the main thread, and a.installed
// (memory only). A card in the sidebar says when Transcribe lacks a model
// (checkModels, shell_notices.go).

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// modelsPath is the Speech models page's path.
const modelsPath = "/models"

// modelsState is the local models and the Speech models page's state.
type modelsState struct {
	local local.Services
	// localLib is the Library under local (nil when tests swap in fakes):
	// a deletion unloads the model through it first.
	localLib *local.Library
	// dictation lists System dictation while the helper can recognise
	// (helperChanged).
	dictation *dictation.Services
	models    *models.Manager
	// modelStatus is each catalogue model's status, as modelsChanged
	// last read it.
	modelStatus map[string]models.Status
	// modelAsk is the model whose deletion is being confirmed when no
	// other app uses it, modelAskFree what that frees (models_delete.go).
	modelAsk     string
	modelAskFree int64
	// modelShared is the deletion being confirmed of a model other
	// ActNow apps use too (family spec §7.10.3).
	modelShared *sharedDelete
	// modelDeciding is the model whose deletion is being weighed (the
	// family's files read off the main thread).
	modelDeciding string
	// netOpen: the Network dialog is open; netForm is its fields
	// (models_network.go).
	netOpen bool
	netForm networkForm
	netWas  networkForm // the form as opened: a stray click or Escape keeps a changed one open
	// modelAdvanced: the Advanced section is open; modelKeepSel is its
	// select's label, modelKeepFor the setting and language it shows
	// (models_keep.go).
	modelAdvanced bool
	modelKeepSel  string
	modelKeepFor  string
	// modelHiddenOpen: the Hidden models section is open (models_cards.go).
	modelHiddenOpen bool
	// speechContext is the Speech context field (models_context.go).
	speechContext speechContextState
}

// openModels reads the models on disk and sets up the local models,
// then the sidebar's card.
func (a *app) openModels() {
	if a.models == nil {
		a.models = models.Open(a.managerOptions(func() { a.post(a.modelsChanged) }))
	}
	if a.local == nil {
		base := newLocalServices(a)
		a.localLib, _ = base.(*local.Library)
		a.dictation = dictation.NewServices(base, uitext.ModelName)
		a.local = a.withRemote(a.dictation) // remote_link.go
	}
	a.readModelStatus()
	a.checkModels() // shell_notices.go
}

// readModelStatus copies the statuses (memory only), and says which
// models were installed since the last copy.
func (a *app) readModelStatus() (installed []string) {
	next := map[string]models.Status{}
	for _, m := range models.Catalog() {
		s := a.models.Status(m.ID)
		if s.Installed && a.modelStatus != nil && !a.modelStatus[m.ID].Installed {
			installed = append(installed, m.ID)
		}
		next[m.ID] = s
	}
	a.modelStatus = next
	return installed
}

// modelsChanged takes the models' statuses after a change (a download's
// progress, its end, a deletion).
func (a *app) modelsChanged() {
	before := a.modelStatus
	for _, id := range a.readModelStatus() {
		if m, _ := models.Lookup(id); m.Kind != models.VAD {
			a.toast(i18n.T("%s is installed.", uitext.ModelName(id)))
		}
	}
	for id, s := range a.modelStatus {
		if before[id].Installed != s.Installed {
			a.stReport = nil // Settings → Storage measures again
		}
	}
	a.checkModels()
	if a.currentPage() == modelsPath[1:] {
		a.readModelMemory() // models_memory.go: a deleted model was unloaded
	}
}

// downloadModel starts downloading model id (and the models it needs).
func (a *app) downloadModel(id string) {
	m := a.models
	a.async(func() {
		m.Download(id) // a failure is the status's
		a.post(a.modelsChanged)
	})
}

// cancelModel stops model id's download.
func (a *app) cancelModel(id string) {
	m := a.models
	a.async(func() { m.Cancel(id) })
}

// installed is kind's installed models, the recommended one first
// (memory only).
func (a *app) installed(kind local.Kind) []local.Model {
	if a.local == nil {
		return nil
	}
	return a.local.Installed(kind)
}

// featureModel is the model a feature uses of kind, and whether one is
// installed that serves it.
func (a *app) featureModel(kind local.Kind, feature string) (local.Model, bool) {
	f := featureOf(a.prefs, feature)
	want := f.ASR
	if kind == local.LLM {
		want = f.LLM
	}
	return local.Choose(local.ForFeature(a.installed(kind), feature), want)
}

// usedBy says which features use model id of kind now, as Voxt 1.x's
// model list did ("Used by Transcribe · Rewrite"), "" for none. A
// language model serves Transcribe only while its clean-up is on.
func (a *app) usedBy(kind local.Kind, id string) string {
	var names []string
	for _, f := range []string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite} {
		if kind == local.LLM && f == hotkey.Transcribe && !a.prefs.Features.Transcribe.Enhance {
			continue
		}
		if m, ok := a.featureModel(kind, f); ok && m.ID == id {
			names = append(names, uitext.FeatureTitle(f))
		}
	}
	if len(names) == 0 {
		return ""
	}
	return i18n.T("Used by %s", strings.Join(names, " · "))
}

// featureOf is a feature's part of the settings (hotkey.Transcribe…).
func featureOf(s settings.Settings, feature string) settings.Feature {
	switch feature {
	case hotkey.Translate:
		return s.Features.Translate.Feature
	case hotkey.Rewrite:
		return s.Features.Rewrite.Feature
	}
	return s.Features.Transcribe.Feature
}

// setFeatureModel sets the model of kind a feature uses ("": the first
// installed).
func setFeatureModel(s *settings.Settings, kind local.Kind, feature, id string) {
	f := &s.Features.Transcribe.Feature
	switch feature {
	case hotkey.Translate:
		f = &s.Features.Translate.Feature
	case hotkey.Rewrite:
		f = &s.Features.Rewrite.Feature
	}
	if kind == local.ASR {
		f.ASR = id
	} else {
		f.LLM = id
	}
}
