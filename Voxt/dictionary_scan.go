package main

// The history scan dialog renders controller-owned progress and results.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func scanDialog(c *ui.Context, state *dictionaryScanViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	if !local.scanOpen {
		return
	}
	dialogFrame(c, "scan-dlg", &local.scanOpen, i18n.T("Scan history"), "", 460, func() {
		if state.Body != "" {
			ui.Text(c, state.Body)
		}
		if state.Warning != "" {
			warnLine(c, state.Warning)
		}
	}, func() {
		ui.Spacer(c)
		switch state.Stage {
		case "running":
			if primaryButton(c.Key("scan-stop"), i18n.T("Stop")).Disabled(state.StopDisabled).Clicked() {
				actions.stopDictionaryScan()
			}
		case "ended":
			if primaryButton(c.Key("scan-done"), i18n.T("Done")).Clicked() {
				local.scanOpen = false
			}
		default:
			if ui.Button(c.Key("scan-cancel"), i18n.T("Cancel")).Clicked() {
				local.scanOpen = false
			}
			if primaryButton(c.Key("scan-start"), i18n.T("Start")).Disabled(!state.CanStart).Clicked() {
				actions.startScan()
			}
		}
	})
}
