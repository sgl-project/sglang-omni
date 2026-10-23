package main

// The kit's side panels (ui_kit.go): an inline column beside the page when
// it fits, else a drawer under the header.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// ---- Side panels ----

// pageWidth is the width of the page beside the sidebar (or the rail).
func (a *app) pageWidth(c *ui.Context) float32 {
	t := c.Theme()
	w, _ := c.Size()
	side := a.sideW
	if a.railMode() {
		side = px(t, railWidth)
	}
	return w - side
}

// panelFits says whether a side panel w wide fits beside the page while
// the page's main column keeps 480 DIPs; when it does not, the panel
// shows as a drawer over the page (sidePanel).
func (a *app) panelFits(c *ui.Context, w float32) bool {
	return a.pageWidth(c)-w >= px(c.Theme(), 480)
}

// sidePanel shows fn in a side panel w wide, at the right of the page:
// beside the main column when inline (a hairline, then a scrolling
// column; call it inside the page's body row), or else as a drawer over
// the page, under its header, which a click outside or Escape closes
// (close may be nil: then only the panel's own controls close it).
func sidePanel(c *ui.Context, key string, w float32, inline bool, label string, close func(), fn func()) {
	t := c.Theme()
	k := tok(t)
	if inline {
		ui.Box(c).Width(1).Background(k.Line).Shrink(0)
		ui.Scroll(c.Key(key)).Width(w).Shrink(0).Label(label).Children(fn)
		return
	}
	ui.Overlay(c, func() {
		_, h := c.Size()
		panel := ui.Scroll(c.Key(key+"-drawer")).Absolute().Right(0).Top(titleBarHeight).Width(w).
			Height(h-titleBarHeight).Background(k.Bg).BorderWidth(0, 0, 0, 1).BorderColor(k.Line).
			Shadow(-8, 0, 24, 0, ui.RGBA(0, 0, 0, 0.12)).Label(label)
		panel.Children(fn)
		if close != nil && (panel.PressedOutside() || panel.OverlayShortcut(0, ui.KeyEscape)) {
			close()
		}
	})
}

// panelHeader is a side panel's title row: the title (15, semibold) and
// a close icon button.
func panelHeader(c *ui.Context, title string, close func()) {
	t := c.Theme()
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		sectionHeading(c, title).Grow(1)
		if iconButton(c.Key("panel-close"), icons.Close, i18n.T("Close")).Clicked() {
			close()
		}
	})
}

// panelState is a page's side panel (an inspector, a list, a step's
// settings): shown beside the main column when both fit, else hidden and
// opened by its toggle as a drawer over the page.
type panelState struct {
	open, set bool // the toggle's choice, once it was used
}

// layout says whether the panel, width wide, fits beside a main column
// of at least minMain (inline), and whether it shows.
func (sp *panelState) layout(a *app, c *ui.Context, width, minMain float32) (inline, shown bool) {
	inline = a.pageWidth(c)-width >= minMain
	shown = inline
	if sp.set {
		shown = sp.open
	}
	return inline, shown
}

// toggle is the panel's header button (an inspector's toggle): ic on the pressed face while the panel shows.
func (sp *panelState) toggle(c *ui.Context, ic icons.Icon, shown bool, showTip, hideTip string) {
	tip := showTip
	if shown {
		tip = hideTip
	}
	if panelToggle(c, ic, shown, tip).Clicked() {
		sp.open, sp.set = !shown, true
	}
}

// view builds the panel when it shows: sidePanel on the
// right (beside the main column, or a drawer), or the same on the left
// (call it where the panel goes in the page's row).
func (sp *panelState) view(c *ui.Context, key, label string, inline, shown, left bool, width float32, build func()) {
	if !shown {
		return
	}
	hide := func() { sp.open, sp.set = false, true }
	if !left {
		sidePanel(c, key, width, inline, label, hide, build)
		return
	}
	t := c.Theme()
	k := tok(t)
	if inline {
		ui.Scroll(c.Key(key)).Width(width).Shrink(0).Label(label).Children(build)
		ui.Box(c).Width(1).Background(k.Line).Shrink(0)
		return
	}
	ui.Overlay(c, func() {
		w, h := c.Size()
		panel := ui.Scroll(c.Key(key+"-drawer")).Absolute().Left(0).Top(titleBarHeight).Width(min(width, w-px(t, 48))).
			Height(h-titleBarHeight).Background(k.Bg).BorderWidth(0, 1, 0, 0).BorderColor(k.Line).
			Shadow(8, 0, 24, 0, ui.RGBA(0, 0, 0, 0.12)).Label(label)
		panel.Children(build)
		if panel.PressedOutside() || panel.OverlayShortcut(0, ui.KeyEscape) {
			hide()
		}
	})
}
