//go:build linux && (amd64 || arm64)

package linux

import (
	"strings"
	"unsafe"

	"github.com/egoist/mygo/internal/bridge"
	"github.com/egoist/mygo/internal/platform"
)

// A window with a hidden title bar has no decorations, as a frameless one,
// and the outer pixels of its page resize it. GTK's own title buttons go
// over the page, in an overlay: a header bar for each side of the desktop's
// button layout (gtk-decoration-layout) that has buttons, with its
// background cleared. They look and act as the buttons of the desktop's
// header bars do, and only those the layout names show: GNOME, KDE and Xfce
// keep it in step with their window managers, and users of tiling window
// managers empty it to have none.
//
// A Wayland compositor that decorates windows itself gets no buttons at all.
// KWin, Hyprland and Sway say so with the default mode of
// org_kde_kwin_server_decoration_manager, and GTK then asks them to decorate
// a window without decorations too. The compositor's title bar has the
// buttons, and a tiling compositor, which draws none, wants none, whatever
// the button layout says: GTK reads it from GNOME's settings on Wayland, and
// their default shows close.

// controlsCSS clears the bars around the buttons. With TitleBarHeight, the
// bars take that height instead of the theme's.
const controlsCSS = `headerbar.mygo-window-controls { background: none; border: none; box-shadow: none; }
headerbar.mygo-window-controls.mygo-sized { min-height: 0; }`

// windowControls are the title buttons of a window with a hidden title bar.
type windowControls struct {
	overlay ptr    // GtkOverlay of the web view and the bars
	bars    [2]ptr // the header bars at the start and the end, or 0
	room    platform.TitleBar
}

var controlsStyled bool

// newControls puts an overlay, whose bars hold the title buttons, where the
// web view goes. It comes before the web view, whose first page is told the
// room the buttons take.
func (w *window) newControls() {
	if !controlsStyled {
		provider := gtkCssProviderNew()
		gtkCssProviderLoadFromData(provider, cs(controlsCSS), len(controlsCSS), nil)
		gtkStyleContextAddProviderForScreen(gtkWidgetGetScreen(w.win), provider, 600) // GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
		gObjectUnref(provider)
		controlsStyled = true
	}
	w.controls = &windowControls{overlay: gtkOverlayNew()}
	// In the window before the bars: a header bar shows window buttons
	// only inside one.
	gtkBoxPackStart(w.box, w.controls.overlay, true, true, 0)
	w.layoutControls()
}

// layoutControls gives the bars the buttons of the desktop's layout: GTK
// shows each only when the window can do it (maximize when it is
// resizable, close when it is deletable).
func (w *window) layoutControls() {
	c := w.controls
	start, end := buttonLayout()
	if w.b.compositorDecorates {
		start, end = "", ""
	}
	for side, buttons := range [2]string{start, end} {
		bar := c.bars[side]
		if buttons == "" {
			if bar != 0 {
				gtkWidgetDestroy(bar)
				c.bars[side] = 0
			}
			continue
		}
		if bar == 0 {
			bar = w.newControlsBar(side)
			c.bars[side] = bar
		}
		layout := buttons + ":"
		if side == 1 {
			layout = ":" + buttons
		}
		// Also rebuilds the buttons for the window's state.
		gtkHeaderBarSetDecorationLayout(bar, cs(layout))
		gtkWidgetShowAll(bar)
		gtkWidgetSetVisible(bar, w.state&stateFullscreen == 0)
	}
	w.measureControls()
}

func (w *window) newControlsBar(side int) ptr {
	bar := gtkHeaderBarNew()
	gtkHeaderBarSetShowCloseButton(bar, true) // the window buttons
	gtkHeaderBarSetHasSubtitle(bar, false)
	style := gtkWidgetGetStyleContext(bar)
	gtkStyleContextAddClass(style, cs("mygo-window-controls"))
	if h := w.opts.TitleBarHeight; h > 0 {
		gtkStyleContextAddClass(style, cs("mygo-sized"))
		gtkWidgetSetSizeRequest(bar, -1, int32(h))
	}
	const alignStart, alignEnd = 1, 2
	align := int32(alignEnd)
	if side == 0 {
		align = alignStart
	}
	gtkWidgetSetHalign(bar, align)
	gtkWidgetSetValign(bar, alignStart)
	gtkOverlayAddOverlay(w.controls.overlay, bar)
	connect(bar, "size-allocate", cbControlsAllocated, ptr(w.id))
	return bar
}

// measureControls takes note of the room the bars take, from their
// allocations once GTK laid them out and from their natural sizes before,
// and tells the core when it changed.
func (w *window) measureControls() {
	c := w.controls
	room := platform.TitleBar{Height: w.opts.TitleBarHeight}
	for side, bar := range c.bars {
		if bar == 0 {
			continue
		}
		var a gdkRectangle
		gtkWidgetGetAllocation(bar, &a)
		width, height := a.Width, a.Height
		if width <= 1 { // not allocated yet
			var minimum int32
			gtkWidgetGetPreferredWidth(bar, &minimum, &width)
			gtkWidgetGetPreferredHeight(bar, &minimum, &height)
		}
		if side == 0 {
			room.Left = int(width)
		} else {
			room.Right = int(width)
		}
		if w.opts.TitleBarHeight == 0 {
			room.Height = max(room.Height, int(height))
		}
	}
	if room != c.room {
		c.room = room
		w.h.TitleBarChanged()
	}
}

// fullScreenChanged hides the buttons in full screen, and shows them after.
func (w *window) fullScreenChanged() {
	for _, bar := range w.controls.bars {
		if bar != 0 {
			gtkWidgetSetVisible(bar, w.state&stateFullscreen == 0)
		}
	}
	w.h.TitleBarChanged()
}

func (w *window) TitleBar() platform.TitleBar {
	if w.controls == nil || w.state&stateFullscreen != 0 {
		return platform.TitleBar{}
	}
	return w.controls.room
}

// titleBarScript tells the first page the room of the title buttons, at
// document start.
func (w *window) titleBarScript() string { return bridge.TitleBarScript(w.TitleBar(), w.opts.Zoom) }

// buttonLayout returns the window buttons that gtk-decoration-layout puts at
// the start and at the end of a title bar, such as "close,minimize" and
// "minimize,maximize,close", without the icon and menu it may name.
func buttonLayout() (start, end string) {
	var p ptr
	gObjectGetPtr(gtkSettingsGetDefault(), cs("gtk-decoration-layout"), unsafe.Pointer(&p), 0)
	before, after, _ := strings.Cut(takeStr(p), ":")
	return windowButtons(before), windowButtons(after)
}

func windowButtons(side string) string {
	var kept []string
	for b := range strings.SplitSeq(side, ",") {
		switch b = strings.TrimSpace(b); b {
		case "minimize", "maximize", "close":
			kept = append(kept, b)
		}
	}
	return strings.Join(kept, ",")
}
