package main

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

const cardInputH = 48

func (a *app) cardInput(c *ui.Context) {
	ui.Row(c).Height(cardInputH).Shrink(0).Padding(8, 20, 4, 0).Children(func() {
		ui.TextInput(c.Key("ov-card-input"), &a.ov.instruction).
			Placeholder(i18n.T("Say or type your rewrite request…")).Label(i18n.T("Rewrite request")).
			Grow(1).MinWidth(0).OnSubmit(a.submitCardInput)
	})
}

func (a *app) submitCardInput() {
	if text := strings.TrimSpace(a.ov.instruction); text != "" && a.mgr != nil {
		a.mgr.SubmitInstruction(a.sess.ID, text)
	} else {
		a.stopSession()
	}
}
