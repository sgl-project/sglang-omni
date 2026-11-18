package main

// The window shell's views (from AudioSmith): the frame of the window —
// the sidebar and the router of pages, the overlays every page shares
// (command palette, the shortcut sheet, the settings dialog beside the
// shell, wired in shell_controller.go), and the theme; no status bar:
// notifications are cards at the foot of the sidebar. Its parts live in
// shell_*.go:
//
//	shell_nav.go       the sidebar and the icon rail
//	shell_commands.go  the ⌘K palette, the ? sheet, the window's keys
//	shell_notices.go   notification cards
//	shell_logs.go      logs page and panel
//	pages.go           the pages of the sidebar (the router's wiring)

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// Aliases of the MyGo types the shell's logic files name (the shell's
// state, the logs' selection, the pages' wiring): a file that names only
// the aliases does not import MyGo, so the boundary check reads it as
// the logic file it is (the settings demo's settingsDemoRouter does the
// same).
type (
	uiContext    = ui.Context
	uiRoute      = ui.Route
	uiRouter     = ui.Router
	uiServices   = ui.Services
	uiRect       = ui.Rect
	uiInputKind  = ui.InputKind
	uiInputEvent = ui.InputEvent
	uiModifiers  = ui.Modifiers
)

// The main window's size, DIPs: by default and at the least. A size the
// window remembers (StateKey "main") wins over the default, so a window
// left larger by an older version keeps its size.
const (
	windowWidth     = 800
	windowHeight    = 520
	windowMinWidth  = 680
	windowMinHeight = 440
)

const (
	titleBarHeight = 52
	sidebarWidth   = 200
)

// baseFontSize is the platform's text size at 100 %: px scales sizes
// designed for it with the UI scale (settings.Appearance.Scale).
var baseFontSize = ui.LightTheme().FontSize

// px scales a size in DIPs designed at 100 % to the theme in use:
// ui.Text(c, s).FontSize(px(t, 12)).
func px(t *ui.Theme, n float32) float32 { return n * t.FontSize / baseFontSize }

// section is the first part of a path: "/settings/models" → "/settings".
func section(path string) string {
	if i := strings.IndexByte(path[1:], '/'); i >= 0 {
		return path[:i+1]
	}
	return path
}

// shellContent is the main window's content: the shell over the app's
// wiring (shell_controller.go's openWindow).
func shellContent(view func(c *ui.Context)) *ui.Content { return ui.View(view) }

// newShellRouter is the shell's router of pages (fade between them) and
// the settings dialog's of sections.
func newShellRouter(path string, fade bool) *uiRouter {
	r := ui.NewRouter(path)
	if fade {
		r.Transition = ui.TransitionFade
	}
	return r
}

// shellLocalUI is the shell views' interface-local state, wired to the
// app's (shell_controller.go's shellLocal); the fixture demo substitutes
// its own.
type shellLocalUI struct {
	nav              *navLocalUI
	logs             *logsLocalUI
	palette          *paletteLocalUI
	sheetOpen        *bool
	announcementOpen *bool
}

// shellView is the frame: the sidebar and the page, the logs panel, the
// areas' dialogs (wired by the caller), then the palette, the sheet and
// the toasts over everything.
func shellView(c *ui.Context, state *shellViewState, actions shellActions, local *shellLocalUI, page func(), dialogs func()) {
	applyTheme(c, state.Theme, state.Scale, state.AccentPalette)
	if state.Transparent {
		// Local widget themes reset the root fill. Clear it after the
		// build so only the sidebar and page supply the window's tint.
		defer c.Root().Background(ui.Transparent)
	}
	t := c.Theme()
	if state.Still {
		// Tests and snapshots show every change at once, as does the
		// user's Reduce motion.
		c.DisableAnimations()
	}
	if state.Opening != nil {
		openingView(c, state.Opening)
		return
	}
	for _, m := range state.Toasts {
		c.Toast(m)
	}
	ui.Column(c).Fill().AlignItems(ui.Stretch).Children(func() {
		sidebarFrame(c, state.Nav, state.Notices, actions, actions, local.nav, func() { // shell_nav.go: the sidebar or the icon rail
			page()
			if state.Logs != nil {
				// At most half the window: the page keeps its header and
				// its first controls (at 680×440 too).
				_, wh := c.Size()
				ph := min(px(t, state.LogsHeight), max(px(t, 120), wh/2-px(t, sp6)))
				ui.Divider(c)
				ui.Column(c.Key("logs-panel")).Height(ph).Shrink(0).Children(func() {
					logsView(c, state.Logs, actions, local.logs, true)
				})
			}
		})
	})
	dialogs()
	paletteView(c, state.Palette, actions, local.palette)
	shortcutSheetView(c, state.Sheet, local.sheetOpen)
	toastsView(c, state, local) // last: over dialogs too
}

// toastsView gives the toasts their place and look: small ink toasts in
// the middle of the page's header (between its title and its actions,
// where it is empty), centred over the page; more stack under it. At the
// window's bottom (MyGo's place) they covered the bars and cards there.
// While the Settings dialog shows, they go to the foot of the dialog
// instead.
func toastsView(c *ui.Context, state *shellViewState, local *shellLocalUI) {
	t := c.Theme()
	k := tok(t)
	ui.ToastViewportBase(c, func(vp ui.Element, toasts []ui.Toast) {
		w, _ := c.Size()
		page := w - local.nav.sideW
		if local.nav.rail {
			page = w - px(t, railWidth)
		}
		vp.AlignItems(ui.Center).Gap(px(t, sp2))
		if state.SettingsOpen {
			// Over the Settings dialog's foot: its title row is at the top.
			vp.Justify(ui.End).Padding(0, px(t, sp4), px(t, sp6+sp8))
		} else {
			vp.Justify(ui.Start).Padding(max(px(t, 4), (titleBarHeight-px(t, 36))/2), px(t, sp4), 0, w-page+px(t, sp4))
		}
		for _, ts := range toasts {
			p := ui.ToastBase(c, ts)
			box := p.Root.Row().AlignItems(ui.Center).Gap(px(t, sp3)).Padding(px(t, 9), px(t, 14)).Radius(px(t, radiusControl)).
				MaxWidth(min(px(t, 480), page-px(t, sp8))).Background(k.Ink).TextColor(k.OnInk).Shadow(0, 6, 20, 0, ui.RGBA(0, 0, 0, 0.22))
			// It fades out at the end.
			const fade = 200 * time.Millisecond
			switch left := p.Left(); {
			case left > 0 && left < fade:
				box.Opacity(float32(left) / float32(fade))
				c.AnimationFrame()
			case left > 0:
				c.After(left - fade)
			}
			box.Children(func() {
				ui.Text(c, ts.Title).MinWidth(0).Shrink(1)
				if ts.Action == "" {
					return
				}
				b := p.ActionButton().Padding(px(t, 3), px(t, 8)).Radius(px(t, 6)).TextColor(k.OnInk).FontWeight(600).Label(ts.Action).Shrink(0)
				if b.Hovered() {
					b.Background(k.OnInk.Alpha(0.14))
				}
				b.Children(func() { ui.Text(c, ts.Action).SingleLine() })
			})
		}
	})
}

// dialogDoneButton is a dialogFrame footer: a Done button at the right,
// closing the dialog.
func dialogDoneButton(c *ui.Context, key string, open *bool) {
	ui.Spacer(c)
	if primaryButton(c.Key(key), i18n.T("Done")).Clicked() {
		*open = false
	}
}

// notFoundPage is the router's fallback: no page at the path.
func notFoundPage(c *ui.Context, path string, actions navActions) {
	pageHeader(c, i18n.T("Not found"), nil)
	emptyState(c, icons.Magnifer, i18n.T("There is no page at %s.", path), "", func() {
		if ui.Button(c.Key("go-home"), i18n.T("Go to Home")).Clicked() {
			actions.navigate("/home")
		}
	})
}
