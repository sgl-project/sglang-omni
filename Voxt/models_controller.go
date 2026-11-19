package main

import (
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type modelsLocalUI struct {
	modelAsk        string
	modelAskFree    int64
	modelShared     *sharedDelete
	netOpen         bool
	netForm         networkForm
	netWas          networkForm
	modelAdvanced   bool
	modelKeepSel    string
	modelKeepFor    string
	modelHiddenOpen bool
	speechContext   speechContextState
	assign          modelAssignmentLocalUI
}

// modelsState is the local models and the state of Settings → Speech models.
type modelsState struct {
	modelsLocalUI
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
	// modelDeciding is the model whose deletion is being weighed (the
	// family's files read off the main thread).
	modelDeciding string
	// The Language models section's wait widgets (llm_wait.go).
	llmWaitLocalUI
	modelView    modelsViewState
	modelViewKey string
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
		if m, _ := models.Lookup(id); m.Kind == models.ASR || m.Kind == models.LLM {
			a.toast(i18n.T("%s is installed.", uitext.ModelName(id)))
		}
	}
	for id, s := range a.modelStatus {
		if before[id].Installed != s.Installed {
			a.storageInvalidated() // Settings → Storage measures again
		}
	}
	a.checkModels()
	if a.shownPath() == modelsPath {
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
	return chooseFeatureModel(a.prefs, a.installed(kind), kind, feature)
}

func chooseFeatureModel(prefs settings.Settings, installed []local.Model, kind local.Kind, feature string) (local.Model, bool) {
	f := featureOf(prefs, feature)
	want := f.ASR
	if kind == local.LLM {
		want = f.LLM
	}
	return local.Choose(local.ForFeature(installed, feature), want)
}

// usedBy says which features use model id of kind now, as Voxt 1.x's
// model list did ("Used by Transcribe · Rewrite"), "" for none. A
// language model serves Transcribe only while its clean-up is on.
func (a *app) usedBy(kind local.Kind, id string) string {
	return modelsUsedBy(i18n.CurrentTranslator(), a.prefs, a.installed(local.ASR), a.installed(local.LLM), kind, id)
}

// featurePart is a feature's part of the settings (hotkey.Transcribe…),
// the one place that maps a feature to it.
func featurePart(s *settings.Settings, feature string) *settings.Feature {
	switch feature {
	case hotkey.Translate:
		return &s.Features.Translate.Feature
	case hotkey.Rewrite:
		return &s.Features.Rewrite.Feature
	case hotkey.Meeting:
		return &s.Features.Meeting.Feature
	}
	return &s.Features.Transcribe.Feature
}

func featureOf(s settings.Settings, feature string) settings.Feature {
	return *featurePart(&s, feature)
}

// setFeatureModel sets the model of kind a feature uses ("": the first
// installed).
func setFeatureModel(s *settings.Settings, kind local.Kind, feature, id string) {
	if kind == local.ASR {
		featurePart(s, feature).ASR = id
	} else {
		featurePart(s, feature).LLM = id
	}
}

// modelHidden says whether the user hid speech model id from the
// features' menus: a model in use never is.
func (a *app) modelHidden(id string) bool {
	return slices.Contains(a.prefs.Models.Hidden, id) && a.usedBy(local.ASR, id) == ""
}

func (a *app) modelShown(model models.Model) bool {
	return modelShown(model, a.modelStatus[model.ID])
}

func (a *app) setModelHidden(id string, hidden bool) {
	a.updateSettings(func(s *settings.Settings) {
		if hidden {
			s.Models.Hidden = append(slices.Clone(s.Models.Hidden), id)
			return
		}
		s.Models.Hidden = slices.DeleteFunc(slices.Clone(s.Models.Hidden), func(existing string) bool { return existing == id })
	})
}

func (a *app) setModelFaster(id string, on bool) {
	a.updateSettings(func(s *settings.Settings) {
		faster := maps.Clone(s.Models.Faster)
		if faster == nil {
			faster = map[string]bool{}
		}
		faster[id] = on
		s.Models.Faster = faster
	})
}

// keepIdle is the idle time of the speech models' servers that m sets.
func keepIdle(m settings.Models) time.Duration {
	if d, auto := m.KeepFor(); !auto {
		return d
	}
	return omni.DefaultIdleTimeout
}

// keepSetting is the saved choice (goroutine-safe: the runtimes are made
// on sessions' goroutines).
func (a *app) keepSetting() settings.Models {
	if a.settings == nil {
		return settings.Models{}
	}
	return a.settings.Get().Models
}

// applyKeepLoaded gives the running servers m's idle time
// (settingsChanged).
func (a *app) applyKeepLoaded(m settings.Models) {
	idle := keepIdle(m)
	a.async(func() {
		a.omniMu.Lock()
		rt := a.omniRT
		a.omniMu.Unlock()
		if rt != nil {
			rt.SetIdleTimeout(idle)
		}
	})
}

func keepLabels() []string {
	return keepLabelsFor(i18n.CurrentTranslator())
}

func keepLabel(value string) string {
	return keepLabelFor(i18n.CurrentTranslator(), value)
}

func (a *app) setModelsKeepLoaded(value string) {
	a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = value })
}

func (a *app) modelsPageModel() modelsViewState {
	input := a.modelsViewStateInput()
	if key := modelsViewCacheKey(input, i18n.Locale()); key != a.modelViewKey {
		a.modelView = buildModelsViewState(i18n.CurrentTranslator(), input)
		a.modelViewKey = key
	}
	return a.modelView
}

func (a *app) modelsViewStateInput() modelsViewStateInput {
	readOnly := false
	familyOthers := false
	if a.fam != nil {
		readOnly = a.fam.ReadOnly()
		familyOthers = a.familyOthers()
	}
	return modelsViewStateInput{
		Catalog: models.Catalog(), Status: a.modelStatus, Memory: a.modelMem,
		Settings: a.prefs, InstalledASR: a.installed(local.ASR), InstalledLLM: a.installed(local.LLM),
		ModelAsk: a.modelAsk, ModelAskFree: a.modelAskFree, ModelShared: a.modelShared, ModelDeciding: a.modelDeciding,
		NetworkOpen: a.netOpen, NetworkForm: a.netForm, FamilyOthers: familyOthers, FamilyReadOnly: readOnly,
		AdvancedOpen: a.modelAdvanced, HiddenOpen: a.modelHiddenOpen,
		Assignment: a.assignmentStateInput(),
	}
}

// assignmentStateInput is the Apply to dialog's state, nil while it is
// closed.
func (a *app) assignmentStateInput() *modelAssignmentStateInput {
	assign := a.modelsLocalUI.assign
	if !assign.open {
		return nil
	}
	_, remote := a.setupOf(assign.kind, assign.id)
	return &modelAssignmentStateInput{
		Kind: assign.kind, ID: assign.id, Installed: a.installed(assign.kind), Settings: a.prefs,
		Selected: assign.selected, Remote: remote,
	}
}

type speechContextState struct {
	text, seen string
	dirty      bool
	seq        int
}

// saveSpeechContext saves the field as typed.
func (a *app) saveSpeechContext() {
	sc := &a.speechContext
	sc.dirty = false
	v := strings.TrimSpace(sc.text)
	sc.seen = v
	if v != a.prefs.Dictionary.SpeechContext {
		a.updateSettings(func(s *settings.Settings) { s.Dictionary.SpeechContext = v })
	}
}

func (a *app) autosaveState(changed, focused bool, st *speechContextState, save func()) {
	if changed {
		st.dirty = true
		st.seq++
		if seq := st.seq; a.win != nil {
			time.AfterFunc(promptSaveDelay, func() {
				a.post(func() {
					if st.dirty && st.seq == seq {
						save()
					}
				})
			})
		}
	} else if st.dirty && !focused {
		save() // it lost the focus
	}
}

func (a *app) autosaveSpeechContext(changed, focused bool) {
	a.autosaveState(changed, focused, &a.speechContext, a.saveSpeechContext)
}
