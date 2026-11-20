package main

// Deletion confirmations for the Speech models page. The controller
// decides the family's delete scope; these views only confirm the intent.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func modelDeleteDialogs(c *ui.Context, state *modelsDeleteViewState, actions modelsActions, local *modelsLocalUI) {
	modelDeleteDialog(c, state.Plain, actions, local)
	modelSharedDialog(c, state.Shared, actions, local)
}

func modelDeleteDialog(c *ui.Context, state *modelPlainDeleteViewState, actions modelsActions, local *modelsLocalUI) {
	if local.modelAsk == "" || state == nil {
		return
	}
	t := c.Theme()
	open := true
	ui.Modal(c.Key("model-ask-delete"), &open, func() {
		ui.Text(c, state.Prompt).Bold()
		ui.Text(c, state.Detail).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("model-ask-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
				local.modelAsk = ""
			}
			if recordButton(c.Key("model-ask-ok"), i18n.T("Delete")).Label(i18n.T("Delete %s", state.Name)).Clicked() {
				id := local.modelAsk
				local.modelAsk = ""
				actions.confirmModelDelete(id, false)
			}
		})
	})
	if !open {
		local.modelAsk = ""
	}
}

func modelSharedDialog(c *ui.Context, state *modelSharedDeleteViewState, actions modelsActions, local *modelsLocalUI) {
	if local.modelShared == nil || state == nil {
		return
	}
	t := c.Theme()
	open := true
	ui.Modal(c.Key("model-ask-shared"), &open, func() {
		if local.modelShared.forAll {
			ui.Text(c, state.AllPrompt).Bold()
			ui.Text(c, state.AllDetail).Width(px(t, 400))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("model-all-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					local.modelShared = nil
				}
				if recordButton(c.Key("model-all-ok"), i18n.T("Delete")).Label(i18n.T("Delete %s for all apps", state.Name)).Clicked() {
					id := local.modelShared.id
					local.modelShared = nil
					actions.confirmModelDelete(id, true)
				}
			})
			return
		}
		ui.Text(c, state.RemovePrompt).Bold()
		ui.Text(c, state.RemoveDetail).Width(px(t, 400))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Wrap().Children(func() {
			if ui.Button(c.Key("model-shared-cancel"), i18n.T("Cancel")).Clicked() {
				local.modelShared = nil
			}
			if ui.Button(c.Key("model-shared-all"), i18n.T("Delete for all apps…")).Clicked() {
				next := *local.modelShared
				next.forAll = true
				local.modelShared = &next
			}
			if primaryButton(c.Key("model-shared-remove"), i18n.T("Remove from Voxt")).AutoFocus().Clicked() {
				id := local.modelShared.id
				local.modelShared = nil
				actions.confirmModelDelete(id, false)
			}
		})
	})
	if !open {
		local.modelShared = nil
	}
}
