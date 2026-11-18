package main

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type modelAssignment struct {
	open     bool
	kind     local.Kind
	id       string
	selected [4]bool
}

var assignmentFeatures = [4]string{hotkey.Transcribe, hotkey.Translate, hotkey.Rewrite, hotkey.Meeting}

func assignmentFeature(s *settings.Settings, feature string) *settings.Feature {
	switch feature {
	case hotkey.Translate:
		return &s.Features.Translate.Feature
	case hotkey.Rewrite:
		return &s.Features.Rewrite.Feature
	case hotkey.Meeting:
		return &s.Features.Meeting.Feature
	default:
		return &s.Features.Transcribe.Feature
	}
}

func assignmentID(s settings.Settings, kind local.Kind, feature string) string {
	f := assignmentFeature(&s, feature)
	if kind == local.ASR {
		return f.ASR
	}
	return f.LLM
}

// Unlike Choose, assignment requires an exact match: a removed model must
// never silently assign the automatic candidate instead.
func (a *app) assignmentModel(kind local.Kind, id string) (local.Model, bool) {
	for _, m := range a.installed(kind) {
		if m.ID == id {
			return m, true
		}
	}
	return local.Model{}, false
}

func (a *app) openModelAssignment(kind local.Kind, id string) {
	m, ok := a.assignmentModel(kind, id)
	if !ok {
		return
	}
	d := modelAssignment{open: true, kind: kind, id: id}
	any := false
	for i, feature := range assignmentFeatures {
		d.selected[i] = m.For(feature) && assignmentID(a.prefs, kind, feature) == id
		any = any || d.selected[i]
	}
	if !any {
		for i, feature := range assignmentFeatures {
			d.selected[i] = m.For(feature)
		}
	}
	a.assignment = d
}

func (a *app) modelAssignmentButton(c *ui.Context, kind local.Kind, id string) {
	if _, ok := a.assignmentModel(kind, id); !ok {
		return
	}
	if ghostButton(c.Key("assign-"+string(kind)+"-"+id), i18n.T("Apply to")).Clicked() {
		a.openModelAssignment(kind, id)
	}
}

func (a *app) applyModelAssignment() {
	d := a.assignment
	m, ok := a.assignmentModel(d.kind, d.id)
	if !ok {
		return
	}
	selected := d.selected
	for i, feature := range assignmentFeatures {
		selected[i] = selected[i] && m.For(feature)
	}
	if !slices.Contains(selected[:], true) {
		return
	}
	a.updateSettings(func(s *settings.Settings) {
		for i, feature := range assignmentFeatures {
			if !selected[i] {
				continue
			}
			f := assignmentFeature(s, feature)
			if d.kind == local.ASR {
				f.ASR = d.id
			} else {
				f.LLM = d.id
			}
		}
	})
	a.assignment = modelAssignment{}
	a.toast(i18n.T("Model applied to selected features."))
}

func (a *app) modelAssignmentDialog(c *ui.Context) {
	d := &a.assignment
	if !d.open {
		return
	}
	m, available := a.assignmentModel(d.kind, d.id)
	canApply := false
	for i, feature := range assignmentFeatures {
		canApply = canApply || available && m.For(feature) && d.selected[i]
	}
	dialogFrame(c, "model-assignment", &d.open, i18n.T("Apply to features"), m.Name, 520, func() {
		caption(c, i18n.T("Only checked features change their model. Other settings and feature switches stay unchanged."))
		if !available {
			caption(c, i18n.T("This model is no longer available. Close this dialog and set it up again."))
		}
		if _, remote := a.setupOf(d.kind, d.id); remote && d.kind == local.ASR {
			caption(c, i18n.T("Recordings from the selected features will be sent to this speech service."))
		}
		for i, feature := range assignmentFeatures {
			title := uitext.FeatureTitle(feature)
			if feature == hotkey.Transcribe && d.kind == local.LLM {
				title = i18n.T("Transcribe clean-up")
			}
			ui.Column(c.Key("assignment-row-" + feature)).Gap(px(c.Theme(), sp1)).Children(func() {
				ui.Checkbox(c.Key("selected"), &d.selected[i], title).Label(i18n.T("Apply model to %s", title)).Disabled(!available || !m.For(feature))
				current := assignmentID(a.prefs, d.kind, feature)
				ready := a.installed(d.kind)
				if d.kind == local.LLM && feature != hotkey.Meeting {
					ready = local.ForFeature(ready, feature)
				}
				_, auto := modelOptions(ready, ready)
				label := modelLabel(ready, current, auto)
				if current != "" && !slices.ContainsFunc(ready, func(m local.Model) bool { return m.ID == current }) {
					label = i18n.T("Unavailable: %s", current)
				}
				caption(c, i18n.T("Current: %s", label))
				if available && !m.For(feature) {
					caption(c, i18n.T("This model does not support this feature."))
				}
				if feature == hotkey.Transcribe && d.kind == local.LLM && !a.prefs.Features.Transcribe.Enhance {
					caption(c, i18n.T("Clean-up is off. This model will be used when you turn it on."))
				}
				if feature == hotkey.Meeting && d.kind == local.LLM {
					caption(c, i18n.T("Used for meeting summaries, live translation and follow-up questions."))
				}
			})
		}
	}, func() {
		if ghostButton(c.Key("assignment-cancel"), i18n.T("Cancel")).Clicked() {
			d.open = false
		}
		if primaryButton(c.Key("assignment-apply"), i18n.T("Apply")).Disabled(!canApply).Clicked() && canApply {
			a.applyModelAssignment()
		}
	})
}
