package main

// Applying an installed model to several features at once, from the
// Speech models cards and the Language models cards. The dialog's
// drawing is in models_assignment_dialog.go.

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// assignmentFeatures are the features the dialog offers, in its order;
// modelAssignmentLocalUI.selected indexes them.
var assignmentFeatures = [4]string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite, hotkey.Meeting}

// modelAssignmentLocalUI is the dialog's open state and its checks
// (local widget state, never saved).
type modelAssignmentLocalUI struct {
	open     bool
	kind     local.Kind
	id       string
	selected [4]bool
}

// check is a feature row's check, for the dialog's checkbox.
func (d *modelAssignmentLocalUI) check(feature string) *bool {
	return &d.selected[slices.Index(assignmentFeatures[:], feature)]
}

type modelAssignmentRowViewState struct {
	Key      string `json:"key"`
	Title    string `json:"title"`
	Label    string `json:"label"`
	Current  string `json:"current"`
	Note     string `json:"note,omitempty"`
	Disabled bool   `json:"disabled,omitempty"`
}

type modelAssignmentViewState struct {
	Open     bool                          `json:"open"`
	Kind     string                        `json:"kind"`
	ID       string                        `json:"id"`
	Title    string                        `json:"title"`
	Subtitle string                        `json:"subtitle"`
	Caption  string                        `json:"caption"`
	Notices  []string                      `json:"notices,omitempty"`
	Rows     []modelAssignmentRowViewState `json:"rows"`
	Checked  [4]bool                       `json:"checked"`
	Apply    string                        `json:"apply"`
	CanApply bool                          `json:"can_apply"`
}

type modelAssignmentStateInput struct {
	Kind      local.Kind
	ID        string
	Installed []local.Model
	Settings  settings.Settings
	Selected  [4]bool
	Remote    bool
}

func buildModelAssignmentViewState(translator *i18n.Translator, input modelAssignmentStateInput) modelAssignmentViewState {
	model, available := assignmentModelAt(input.Installed, input.ID)
	state := modelAssignmentViewState{
		Open: true, Kind: string(input.Kind), ID: input.ID, Title: translator.T("Apply to features"), Apply: translator.T("Apply"),
		// A removed model is named by its id: assigning the automatic
		// candidate instead would change the model without saying so.
		Subtitle: input.ID,
		Caption:  translator.T("Only checked features change their model. Other settings and feature switches stay unchanged."),
		Checked:  input.Selected,
	}
	if available {
		state.Subtitle = model.Name
	} else {
		state.Notices = append(state.Notices, translator.T("This model is no longer available. Close this dialog and set it up again."))
	}
	if input.Remote && input.Kind == local.ASR {
		state.Notices = append(state.Notices, translator.T("Recordings from the selected features will be sent to this speech service."))
	}
	for index, feature := range assignmentFeatures {
		row := assignmentRowViewState(translator, input, model, available, feature)
		state.CanApply = state.CanApply || input.Selected[index] && !row.Disabled
		state.Rows = append(state.Rows, row)
	}
	return state
}

func assignmentRowViewState(translator *i18n.Translator, input modelAssignmentStateInput, model local.Model, available bool, feature string) modelAssignmentRowViewState {
	row := modelAssignmentRowViewState{
		Key: feature, Title: assignmentFeatureTitle(translator, input.Kind, feature),
		Label: translator.T("Apply model to %s", assignmentFeatureTitle(translator, input.Kind, feature)),
	}
	row.Disabled = !available || !model.For(feature)
	row.Current = translator.T("Current: %s", assignmentCurrentLabel(translator, input, feature))
	switch {
	case available && !model.For(feature):
		row.Note = translator.T("This model does not support this feature.")
	case feature == hotkey.Transcribe && input.Kind == local.LLM && !input.Settings.Features.Transcribe.Enhance:
		row.Note = translator.T("Clean-up is off. This model will be used when you turn it on.")
	default:
	}
	return row
}

// assignmentCurrentLabel is the model a feature uses now: its name, the
// automatic choice, or the id of a model that is gone.
func assignmentCurrentLabel(translator *i18n.Translator, input modelAssignmentStateInput, feature string) string {
	ready := local.ForFeature(input.Installed, feature)
	_, automatic := modelOptionsFor(translator, ready, ready)
	current := assignmentModelID(&input.Settings, input.Kind, feature)
	switch model, found := assignmentModelAt(ready, current); {
	case current == "":
		return automatic
	case found:
		return model.Name
	default:
		return translator.T("Unavailable: %s", current)
	}
}

func assignmentFeatureTitle(translator *i18n.Translator, kind local.Kind, feature string) string {
	if feature == hotkey.Transcribe && kind == local.LLM {
		return translator.T("Transcribe clean-up")
	}
	return uitext.FeatureTitleFor(translator, feature)
}

func assignmentModelID(s *settings.Settings, kind local.Kind, feature string) string {
	if kind == local.ASR {
		return featurePart(s, feature).ASR
	}
	return featurePart(s, feature).LLM
}

// assignmentModelAt finds an installed model by id alone: unlike
// local.Choose, a removed model never falls back to the recommended one.
func assignmentModelAt(installed []local.Model, id string) (local.Model, bool) {
	index := slices.IndexFunc(installed, func(model local.Model) bool { return model.ID == id })
	if index < 0 {
		return local.Model{}, false
	}
	return installed[index], true
}

func (a *app) openModelAssignment(kind, id string) {
	modelKind := remoteKind(kind)
	model, ok := assignmentModelAt(a.installed(modelKind), id)
	if !ok {
		return
	}
	assign := &a.modelsLocalUI.assign
	assign.open, assign.kind, assign.id = true, modelKind, id
	explicit := false
	for index, feature := range assignmentFeatures {
		assign.selected[index] = model.For(feature) && assignmentModelID(&a.prefs, modelKind, feature) == id
		explicit = explicit || assign.selected[index]
	}
	if explicit {
		return
	}
	// Nothing named this model: offer every feature it serves.
	for index, feature := range assignmentFeatures {
		assign.selected[index] = model.For(feature)
	}
}

// applyModelAssignment writes only the checked features, after checking
// again that the model is still installed and still serves them.
func (a *app) applyModelAssignment() {
	assign := &a.modelsLocalUI.assign
	modelKind := assign.kind
	model, ok := assignmentModelAt(a.installed(modelKind), assign.id)
	if !ok {
		return
	}
	var features []string
	for index, feature := range assignmentFeatures {
		if assign.selected[index] && model.For(feature) {
			features = append(features, feature)
		}
	}
	if len(features) == 0 {
		return
	}
	id := assign.id
	a.updateSettings(func(s *settings.Settings) {
		for _, feature := range features {
			setFeatureModel(s, modelKind, feature, id)
		}
	})
	assign.open = false
	a.toast(i18n.T("Model applied to selected features."))
}
