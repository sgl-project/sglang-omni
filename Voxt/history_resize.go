package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// historyDetailDivider keeps resizing within the detail and list minimum widths.
func historyDetailDivider(c *ui.Context, actions historyActions, local *historyLocalUI, width, limit float32) float32 {
	t := c.Theme()
	lo, scale := px(t, historyPanelW), px(t, 1)
	next, held := width, false
	ui.Box(c.Key("hist-divider")).Width(1).Shrink(0).Background(tok(t).Line).Children(func() {
		if limit <= lo {
			return
		}
		edge := ui.Box(c.Key("resize")).Absolute().Left(-3).Top(0).Bottom(0).Width(7).
			Focusable().Cursor(ui.CursorResizeEW).Role(ui.RoleSplitter).
			Label(i18n.T("Resize entry details")).Tooltip(i18n.T("Drag to widen. Double-click to reset.")).
			Range(float64(lo/scale), float64(limit/scale), float64(width/scale)).Step(10)
		dx, _, dragging := edge.Dragged()
		held = dragging
		if held {
			edge.Focus()
			next = width - dx
		}
		if edge.Shortcut(0, ui.KeyLeft) {
			next += px(t, 10)
		}
		if edge.Shortcut(0, ui.KeyRight) {
			next -= px(t, 10)
		}
		if edge.DoubleClicked() {
			next = lo
		}
	})
	next = min(max(next, lo), limit)
	if next != width {
		local.panelWidth = next / scale
	}
	if local.panelResizing && !held || next != width && !held {
		saved := float64(local.panelWidth)
		if saved <= historyPanelW {
			saved = 0
		}
		actions.saveHistoryDetailWidth(saved)
	}
	local.panelResizing = held
	return next
}
