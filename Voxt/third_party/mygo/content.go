package mygo

import (
	"bytes"
	"errors"
	"image"
	"image/png"
	"math"
	"slices"

	"github.com/egoist/mygo/i18n"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/surface"
	"github.com/egoist/mygo/transfer"
)

// Content is what a window shows in place of a web page: a user interface
// MyGo draws itself, on the GPU where it can. Package ui provides it:
//
//	mygo.NewWindow(mygo.WindowOptions{Title: "Counter", Content: ui.View(counter.View)})
type Content interface {
	// AttachContent connects the content to its window. NewWindow calls it
	// on the main thread; only MyGo's own packages implement it.
	AttachContent(conn *surface.Conn)
}

// attachContent connects the window's Content to its surface. Main thread
// only.
func (w *Window) attachContent() {
	s := w.native.Surface()
	if s == nil {
		panic("mygo: the backend " + backend().Name() + " created no surface for the window's Content")
	}
	w.conn = &surface.Conn{
		Surface:        s,
		Window:         w,
		Clipboard:      backend().Clipboard(),
		StartDataDrag:  w.startDataDrag,
		CancelDataDrag: w.cancelDataDrag,
		StartDrag: func() {
			if w.native != nil {
				w.native.StartDrag()
			}
		},
		TitleBarDoubleClicked: func() {
			if w.native != nil {
				w.native.TitleBarDoubleClicked()
			}
		},
		Escape:        w.escape,
		IsDark:        func() bool { return backend().Theme().IsDark() },
		Preferences:   func() platform.Preferences { return backend().Theme().Preferences() },
		Locale:        func() string { return App.Locale() },
		RightToLeft:   appRightToLeft,
		Region:        func() i18n.Region { return App.Region() },
		UIFont:        func() string { return backend().Theme().UIFont() },
		FontRendering: func() platform.FontRendering { return backend().Theme().FontRendering() },
		TitleBar: func() platform.TitleBar {
			if !w.hiddenTitleBar || w.native == nil {
				return platform.TitleBar{}
			}
			return w.native.TitleBar()
		},
		OpenURL: func(url string, done func(error)) {
			// The system may take a while, as Windows' shell does: not
			// in the frame.
			go func() {
				err := Shell.OpenExternal(url)
				if done != nil {
					postMain(func() {
						if w.native != nil {
							done(err)
						}
					})
				}
			}()
		},
		DevTools:   w.devTools,
		Invalidate: w.Invalidate,
		Ready:      w.readyToShow,
		Post: func(fn func()) {
			postMain(func() {
				if w.native != nil {
					fn()
				}
			})
		},
		PlaceWebViews: w.placeWebViews,
		PopupMenu: func(m *platform.Menu, x, y float64, chosen func(int)) {
			pos := &platform.Point{X: int(math.Round(x)), Y: int(math.Round(y))}
			menu := NewMenu(contentMenu(m, chosen))
			// The menu waits for the user in a loop of its own: not in the
			// event that asked for it, which the surface is handling.
			postMain(func() {
				if w.native != nil {
					menu.popup(w, pos, m.RightToLeft)
				}
			})
		},
	}
	w.content.AttachContent(w.conn)
}

// contentMenu returns the items of a context menu of the Content, whose
// Click passes the ID of the item to chosen.
func contentMenu(m *platform.Menu, chosen func(int)) []*MenuItem {
	items := make([]*MenuItem, 0, len(m.Items))
	for _, p := range m.Items {
		it := &MenuItem{Label: p.Label, Accelerator: p.Accelerator, Disabled: !p.Enabled, Hidden: !p.Visible, Checked: p.Checked}
		switch p.Type {
		case platform.MenuItemSeparator:
			it.Type = MenuItemSeparator
		case platform.MenuItemCheckbox:
			it.Type = MenuItemCheckbox
		case platform.MenuItemRadio:
			it.Type = MenuItemRadio
		case platform.MenuItemSubmenu:
			it.Type = MenuItemSubmenu
			it.Submenu = []*MenuItem{}
			if p.Submenu != nil {
				it.Submenu = contentMenu(p.Submenu, chosen)
			}
		}
		id := p.ID
		it.Click = func(*MenuItem, *Window) { chosen(id) }
		items = append(items, it)
	}
	return items
}

// page runs fn with the native window, unless the window shows Content
// and so has no page.
func (w *Window) page(fn func(n platform.Window)) {
	if w.content != nil {
		return
	}
	w.do(fn)
}

// pageGet is page for functions returning a value.
func pageGet[T any](w *Window, fn func(n platform.Window) T) T {
	if w.content != nil {
		var zero T
		return zero
	}
	return get(w, fn)
}

// Invalidate redraws the window's Content, as after a change of the state
// its user interface shows. It is safe from any goroutine; calls made
// before the next frame make one frame. It does nothing for a web page.
func (w *Window) Invalidate() {
	if w.content == nil || !w.invalidating.CompareAndSwap(false, true) {
		return
	}
	postMain(func() {
		w.invalidating.Store(false)
		w.contentChanged()
	})
}

// Update runs fn on the main thread, where the window's Content builds its
// user interface, then redraws the Content. Goroutines change the state
// the interface shows through it:
//
//	go func() {
//		items := load()
//		win.Update(func() { app.items = items })
//	}()
//
// It returns without waiting for fn.
func (w *Window) Update(fn func()) {
	postMain(func() {
		fn()
		w.contentChanged()
	})
}

// contentChanged asks the Content for a frame built anew, on the main
// thread, after the app changed its state.
func (w *Window) contentChanged() {
	switch {
	case w.conn == nil || w.native == nil:
	case w.conn.Changed != nil:
		w.conn.Changed()
	default:
		w.conn.Surface.RequestFrame()
	}
}

// captureContent renders the Content into a PNG image.
func (w *Window) captureContent() ([]byte, error) {
	var (
		width, height int
		pix           []byte
	)
	ok := false
	onMain(func() {
		if w.conn != nil && w.conn.Capture != nil {
			width, height, pix = w.conn.Capture()
			ok = true
		}
	})
	if !ok {
		return nil, errDestroyed
	}
	if width == 0 || height == 0 {
		return nil, errors.New("mygo: the window's content has no size")
	}
	img := &image.RGBA{Pix: pix, Stride: 4 * width, Rect: image.Rect(0, 0, width, height)}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// SurfaceEvent passes the surface's events to the Content. Files the
// Content does not take go to OnFileDrop, when it has listeners.
func (h *windowHandler) SurfaceEvent(ev platform.SurfaceEvent) bool {
	resolveDataDrag(ev.Drag)
	c := h.w.conn
	if c == nil || c.Event == nil {
		return false
	}
	if c.Event(ev) {
		return true
	}
	if d := ev.Drag; d != nil {
		switch ev.Kind {
		case platform.DataDragLeave:
			c.Event(platform.SurfaceEvent{Kind: platform.FileDragLeave})
		case platform.DataDragOver, platform.DataDrop:
			if d.HasFiles || slices.Contains(d.Offer.Formats, transfer.FileList) {
				d.Operation = transfer.Negotiate(d.Offer.Operations, transfer.Copy, d.Offer.Suggested)
				if d.Operation == transfer.None {
					return false
				}
				fileEvent := platform.SurfaceEvent{Kind: platform.FileDragOver, X: ev.X, Y: ev.Y}
				if ev.Kind == platform.DataDrop {
					fileEvent.Kind = platform.FileDrop
					paths := ev.Files
					if len(paths) == 0 {
						paths, _ = d.Data.Files()
					}
					if len(paths) == 0 {
						d.Operation = transfer.None
						return false
					}
					fileEvent.Files = paths
				}
				d.Formats = []transfer.Format{transfer.FileList}
				if h.SurfaceEvent(fileEvent) {
					return true
				}
			}
		}
		d.Operation = transfer.None
	}
	switch ev.Kind {
	case platform.FileDragOver:
		return h.w.onFileDrop.len() > 0
	case platform.FileDrop:
		if h.w.onFileDrop.len() == 0 {
			return false
		}
		fire1(&h.w.onFileDrop, &FileDropEvent{Paths: ev.Files, X: int(ev.X), Y: int(ev.Y)})
		return true
	}
	return false
}

// contentCommand performs an edit command (an Edit menu role: "copy",
// "paste", …) in the window's native UI. Main thread only.
func (w *Window) contentCommand(cmd string) {
	(&windowHandler{w}).SurfaceEvent(platform.SurfaceEvent{Kind: platform.SurfaceCommand, Text: cmd})
}

// detachContent tells the Content its window closed. Main thread only.
func (w *Window) detachContent() {
	if c := w.conn; c != nil {
		w.conn = nil
		if c.Detach != nil {
			c.Detach()
		}
	}
}

// contentThemeChanged tells the Contents of the windows that the system
// appearance changed. Main thread only.
// contentLocaleChanged tells the windows of native UI that the language
// the app shows or the regional formats changed, which they draw anew.
// Main thread only.
func contentLocaleChanged() {
	for _, w := range Windows() {
		if c := w.conn; c != nil && c.LocaleChanged != nil {
			c.LocaleChanged()
		}
	}
}

func contentThemeChanged() {
	for _, w := range Windows() {
		if c := w.conn; c != nil && c.ThemeChanged != nil {
			c.ThemeChanged()
		}
	}
}
