package main

// The Apply to dialog: a model's four feature rows, each with the model
// it uses now, and the intents that apply the chosen ones.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func modelAssignmentDialog(c *ui.Context, state *modelAssignmentViewState, actions modelsActions, local *modelAssignmentLocalUI) {
	if state == nil || !local.open {
		return
	}
	t := c.Theme()
	dialogFrame(c, "assignment-dlg", &local.open, state.Title, state.Subtitle, 520, func() {
		caption(c, state.Caption)
		for _, notice := range state.Notices {
			caption(c, notice).TextColor(tok(t).Warning)
		}
		for _, row := range state.Rows {
			ui.Column(c.Key("assignment-row-" + row.Key)).Gap(px(t, sp1)).Children(func() {
				ui.Checkbox(c.Key("assignment-check-"+row.Key), local.check(row.Key), row.Title).Label(row.Label).Disabled(row.Disabled)
				caption(c, row.Current)
				if row.Note != "" {
					caption(c, row.Note)
				}
			})
		}
	}, func() {
		if ghostButton(c.Key("assignment-cancel"), i18n.T("Cancel")).Clicked() {
			local.open = false
		}
		if primaryButton(c.Key("assignment-apply"), state.Apply).Disabled(!state.CanApply).Clicked() && state.CanApply {
			actions.applyModelAssignment()
		}
	})
}
