//go:build linux && (amd64 || arm64)

package linux

import (
	"image"
	"log"
	"math"
	"os"
	"sync"
	"unicode/utf8"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/gpu/gl"
	"github.com/egoist/mygo/internal/platform"
)

// The surface of a window that shows content MyGo draws itself is a
// GtkGLArea in place of the web view: frames are drawn with OpenGL in its
// render signal, into the framebuffer of the context GTK makes current,
// and GTK shows them. Without OpenGL, frames drawn in memory are painted
// with cairo in the draw signal, as on a GtkDrawingArea, which the surface
// is with MYGO_GPU=0 and where OpenGL would not draw on a GPU (gpuGL).
// GTK's frame clock paces both; keys go through a GtkIMContext while a
// text input has the focus.
//
// Frames drawn in memory are drawn in the frame clock's update phase, by a
// tick callback, before GTK paints: the area then asks GTK to repaint what
// they changed, which GTK paints from them, keeping the rest of its buffer
// and telling the compositor only that changed. A draw GTK asks for by
// itself, as when the window is resized or uncovered, draws a frame there.

var (
	surfaceOnce                 sync.Once
	gtkDrawingAreaNew           func() ptr
	gtkWidgetSetCanFocus        func(w ptr, v bool)
	gtkWidgetAddEvents          func(w ptr, mask int32)
	gtkWidgetQueueDraw          func(w ptr)
	gtkWidgetQueueDrawArea      func(w ptr, x, y, width, height int32)
	gtkWidgetGetHasWindow       func(w ptr) bool
	gtkWidgetAddTickCallback    func(w, callback, data, notify ptr) uint32
	cairoClipExtents            func(cr ptr, x1, y1, x2, y2 *float64)
	gtkWidgetGetAllocatedWidth  func(w ptr) int32
	gtkWidgetGetAllocatedHeight func(w ptr) int32
	gtkWidgetGetScaleFactor     func(w ptr) int32
	gtkWidgetGetDisplay         func(w ptr) ptr
	gtkWidgetHasFocus           func(w ptr) bool
	gtkIMMulticontextNew        func() ptr
	gtkIMContextSetClientWindow func(im, win ptr)
	gtkIMContextFilterKeypress  func(im, event ptr) bool
	gtkIMContextFocusIn         func(im ptr)
	gtkIMContextFocusOut        func(im ptr)
	gtkIMContextReset           func(im ptr)
	gtkIMContextSetCursorLoc    func(im ptr, r *gdkRectangle)
	gtkIMContextGetPreedit      func(im ptr, str *ptr, attrs *ptr, cursor *int32)
	gdkKeyvalToUnicode          func(keyval uint32) uint32
	cairoImageSurfaceForData    func(data *byte, format, width, height, stride int32) ptr
	cairoSurfaceSetDeviceScale  func(s ptr, x, y float64)
	cairoSetSourceSurface       func(cr, s ptr, x, y float64)
	cairoSetOperator            func(cr ptr, op int32)
	cairoPaint                  func(cr ptr)
	gtkGLAreaNew                func() ptr
	gtkGLAreaSetRequiredVersion func(a ptr, major, minor int32)
	gtkGLAreaSetHasAlpha        func(a ptr, alpha bool)
	gtkGLAreaGetError           func(a ptr) ptr
	gtkGLAreaSetError           func(a, gerr ptr)
	gdkWindowPeekChildren       func(w ptr) ptr
	gdkWindowGetUserData        func(w ptr, data *ptr)

	cbSurfaceDraw, cbSurfaceSize, cbSurfaceRealize, cbSurfaceButton ptr
	cbSurfaceMotion, cbSurfaceLeave, cbSurfaceScroll, cbSurfaceKey  ptr
	cbSurfaceFocusIn, cbSurfaceFocusOut, cbSurfaceScale, cbIMCommit ptr
	cbIMPreedit, cbIMPreeditEnd, cbSurfaceRender, cbAreaContext     ptr
	cbSurfaceUnrealize, cbSurfaceTick                               ptr
	surfaceCursors                                                  = map[platform.Cursor]ptr{}
)

func loadSurface() {
	surfaceOnce.Do(func() {
		t, d, c := libGTK, libGDK, libCairo
		mustBind(t, &gtkDrawingAreaNew, "gtk_drawing_area_new")
		mustBind(t, &gtkWidgetSetCanFocus, "gtk_widget_set_can_focus")
		mustBind(t, &gtkWidgetAddEvents, "gtk_widget_add_events")
		mustBind(t, &gtkWidgetQueueDraw, "gtk_widget_queue_draw")
		mustBind(t, &gtkWidgetQueueDrawArea, "gtk_widget_queue_draw_area")
		mustBind(t, &gtkWidgetGetHasWindow, "gtk_widget_get_has_window")
		mustBind(t, &gtkWidgetAddTickCallback, "gtk_widget_add_tick_callback")
		mustBind(c, &cairoClipExtents, "cairo_clip_extents")
		mustBind(t, &gtkWidgetGetAllocatedWidth, "gtk_widget_get_allocated_width")
		mustBind(t, &gtkWidgetGetAllocatedHeight, "gtk_widget_get_allocated_height")
		mustBind(t, &gtkWidgetGetScaleFactor, "gtk_widget_get_scale_factor")
		mustBind(t, &gtkWidgetGetDisplay, "gtk_widget_get_display")
		mustBind(t, &gtkWidgetHasFocus, "gtk_widget_has_focus")
		mustBind(t, &gtkIMMulticontextNew, "gtk_im_multicontext_new")
		mustBind(t, &gtkIMContextSetClientWindow, "gtk_im_context_set_client_window")
		mustBind(t, &gtkIMContextFilterKeypress, "gtk_im_context_filter_keypress")
		mustBind(t, &gtkIMContextFocusIn, "gtk_im_context_focus_in")
		mustBind(t, &gtkIMContextFocusOut, "gtk_im_context_focus_out")
		mustBind(t, &gtkIMContextReset, "gtk_im_context_reset")
		mustBind(t, &gtkIMContextSetCursorLoc, "gtk_im_context_set_cursor_location")
		mustBind(t, &gtkIMContextGetPreedit, "gtk_im_context_get_preedit_string")
		mustBind(d, &gdkKeyvalToUnicode, "gdk_keyval_to_unicode")
		mustBind(c, &cairoImageSurfaceForData, "cairo_image_surface_create_for_data")
		mustBind(c, &cairoSurfaceSetDeviceScale, "cairo_surface_set_device_scale")
		mustBind(c, &cairoSetSourceSurface, "cairo_set_source_surface")
		mustBind(c, &cairoSetOperator, "cairo_set_operator")
		mustBind(c, &cairoPaint, "cairo_paint")
		mustBind(d, &gdkWindowPeekChildren, "gdk_window_peek_children")
		mustBind(d, &gdkWindowGetUserData, "gdk_window_get_user_data")
		// GtkGLArea came in GTK 3.16.
		if !bind(t, &gtkGLAreaNew, "gtk_gl_area_new") || !bind(t, &gtkGLAreaSetRequiredVersion, "gtk_gl_area_set_required_version") ||
			!bind(t, &gtkGLAreaSetHasAlpha, "gtk_gl_area_set_has_alpha") || !bind(t, &gtkGLAreaGetError, "gtk_gl_area_get_error") ||
			!bind(t, &gtkGLAreaSetError, "gtk_gl_area_set_error") {
			gtkGLAreaNew = nil
		}
	})
}

type surface struct {
	clientComposition platform.InputComposition
	w                 *window
	area              ptr // GtkGLArea, or GtkDrawingArea
	im                ptr // GtkIMContext
	cr                ptr // the cairo context of the draw signal in progress
	cursor            platform.Cursor
	// gl tells that the area is a GtkGLArea, rendering that its render
	// signal is in progress, rendered that it ran.
	gl, rendering, rendered bool
	// present shows the frames the content draws in memory in a GtkGLArea,
	// as when its GL renderer fails; inMemory tells that it did, failed
	// that it failed too.
	present                 gl.Presenter
	inMemory, presentFailed bool
	// tick is the tick callback drawing frames in memory, while one is
	// asked for (RequestFrame); again tells that another was asked for
	// while it drew one, and ticking that it draws one.
	tick           uint32
	again, ticking bool
	// staged is the frame the tick callback drew, which the draw signal
	// paints in the same frame of the frame clock: the host's memory,
	// valid until it draws or frees another (see Idle).
	staged struct {
		pix                   []byte
		stride, width, height int
	}

	textInput bool
	caret     platform.RectF
	input     platform.TextInputState
	// lastKey is a copy of the last key press, which a context menu the
	// key opens shows for.
	lastKey ptr
	// pressing is the key press being handled, which the Tab moving the
	// focus to a web view passes on to it (TabInto).
	pressing     ptr
	lastPointer  ptr
	clicks       platform.ClickCounter
	dragContext  ptr
	dragRequest  *platform.DragRequest
	dragError    error
	dragCanceled bool
	dataDrop     *gtkDataDrop
	// layers is the overlay of the area over web, the layout of the web
	// views under it, and placed where the content shows them
	// (webview.go), once the window has some.
	layers, web ptr
	placed      []platform.WebViewPlacement
}

// GDK event masks of the drawing area.
const surfaceEvents = 1<<2 | 1<<8 | 1<<9 | 1<<10 | 1<<11 | 1<<12 | 1<<13 | 1<<14 | 1<<21 | 1<<23

func (w *window) createSurface() {
	loadSurface()
	data := ptr(w.id)
	s := &surface{w: w}
	s.im = gtkIMMulticontextNew()
	s.newArea(gtkGLAreaNew != nil && os.Getenv("MYGO_GPU") != "0" && gpuGL())
	connect(s.im, "commit", cbIMCommit, data)
	connect(s.im, "preedit-changed", cbIMPreedit, data)
	connect(s.im, "preedit-end", cbIMPreeditEnd, data)
	s.connectSystem(data)
	w.surface = s
}

// newArea creates the surface's widget, a GtkGLArea or a GtkDrawingArea
// that assistive technology sees the content of, and connects its input.
func (s *surface) newArea(gl bool) {
	data := ptr(s.w.id)
	s.area = newSurfaceArea(gl)
	if gl {
		gtkGLAreaSetRequiredVersion(s.area, 3, 3)
		// Only transparent windows show what is behind them. With an alpha
		// channel GTK keeps the area in a texture it blends, rather than in
		// a renderbuffer it copies, which took a small window 50 MB more of
		// the GPU's memory, and an animation half as much CPU again.
		gtkGLAreaSetHasAlpha(s.area, s.w.opts.Transparent)
		connect(s.area, "create-context", cbAreaContext, data)
		connect(s.area, "render", cbSurfaceRender, data)
	}
	s.gl = gl
	gtkWidgetSetCanFocus(s.area, true)
	gtkWidgetAddEvents(s.area, surfaceEvents)
	// The edges of a frameless window resize it, as over a page.
	connect(s.area, "button-press-event", cbButtonPress, data)
	if s.w.undecorated() {
		connect(s.area, "motion-notify-event", cbMotion, data)
	}
	connect(s.area, "draw", cbSurfaceDraw, data)
	connect(s.area, "size-allocate", cbSurfaceSize, data)
	connect(s.area, "realize", cbSurfaceRealize, data)
	connect(s.area, "unrealize", cbSurfaceUnrealize, data)
	connect(s.area, "notify::scale-factor", cbSurfaceScale, data)
	connect(s.area, "button-press-event", cbSurfaceButton, data)
	connect(s.area, "button-release-event", cbSurfaceButton, data)
	connect(s.area, "motion-notify-event", cbSurfaceMotion, data)
	connect(s.area, "leave-notify-event", cbSurfaceLeave, data)
	connect(s.area, "scroll-event", cbSurfaceScroll, data)
	connect(s.area, "key-press-event", cbSurfaceKey, data)
	connect(s.area, "key-release-event", cbSurfaceKey, data)
	connect(s.area, "focus-in-event", cbSurfaceFocusIn, data)
	connect(s.area, "focus-out-event", cbSurfaceFocusOut, data)
}

// contentWidget returns the widget showing the window's content: the web
// view or the surface, in the overlay over its web views once it has
// some.
func (w *window) contentWidget() ptr {
	if s := w.surface; s != nil {
		if s.layers != 0 {
			return s.layers
		}
		return s.area
	}
	return w.web
}

// contentWindow returns the GdkWindow that receives the input of the
// window's content.
func (w *window) contentWindow() ptr {
	if w.surface != nil {
		return w.surface.eventWindow()
	}
	return gtkWidgetGetWindow(w.web)
}

func (s *surface) destroy() {
	s.cancelPendingDrop()
	s.CancelDataDrag()
	if s.lastPointer != 0 {
		gdkEventFree(s.lastPointer)
		s.lastPointer = 0
	}
	gtkIMContextSetClientWindow(s.im, 0)
	gObjectUnref(s.im)
	s.im = 0
	if s.lastKey != 0 {
		gdkEventFree(s.lastKey)
		s.lastKey = 0
	}
}

// popupTrigger returns the last press of a button or, in a surface, of a
// key in the window's content: the event a context menu shows for, whose
// time and serial GTK grabs the pointer with.
func (w *window) popupTrigger() ptr {
	ev := w.press.event
	if s := w.surface; s != nil && s.lastKey != 0 {
		// GdkEventButton and GdkEventKey: time 20.
		if ev == 0 || field[uint32](s.lastKey, 20)-field[uint32](ev, 20) < 1<<31 {
			ev = s.lastKey
		}
	}
	return ev
}

// Surface returns the surface of a window created with
// WindowOptions.Surface.
func (w *window) Surface() platform.Surface {
	if w.surface == nil {
		return nil
	}
	return w.surface
}

func (s *surface) Native() platform.SurfaceNative {
	n := platform.SurfaceNative{Widget: s.area}
	if s.rendering {
		n.GLArea = s.area
	}
	return n
}

// eventWindow returns the GdkWindow that receives the surface's input: the
// drawing area's own, or the input-only window a GtkGLArea, which draws in
// its parent's window, adds when realized.
func (s *surface) eventWindow() ptr {
	win := gtkWidgetGetWindow(s.area)
	if win == 0 || !s.gl {
		return win
	}
	// GList: data 0, next 8.
	for l := gdkWindowPeekChildren(win); l != 0; l = field[ptr](l, 8) {
		child := field[ptr](l, 0)
		var owner ptr
		gdkWindowGetUserData(child, &owner)
		if owner == s.area {
			return child
		}
	}
	return win
}

func (s *surface) Size() (float64, float64, float64) {
	return float64(gtkWidgetGetAllocatedWidth(s.area)), float64(gtkWidgetGetAllocatedHeight(s.area)), float64(max(gtkWidgetGetScaleFactor(s.area), 1))
}

func (s *surface) RequestFrame() {
	switch {
	case s.w.closed:
	case s.drawsGL():
		gtkWidgetQueueDraw(s.area)
	default:
		s.again = true
		if s.tick == 0 {
			s.tick = gtkWidgetAddTickCallback(s.area, cbSurfaceTick, ptr(s.w.id), 0)
		}
	}
}

// drawsGL tells that the area draws its frames with OpenGL, in its render
// signal: a GtkGLArea with a context.
func (s *surface) drawsGL() bool { return s.gl && gtkGLAreaGetError(s.area) == 0 }

// PresentDamage shows a frame drawn in memory that changed the last one
// only within damage. Drawn by the tick callback, it is painted where GTK
// repaints what changed; in the draw signal or the render signal, it is
// painted whole, and what it changed outside GTK's clip is repainted in
// the next frame.
func (s *surface) PresentDamage(pix []byte, stride, width, height int, damage []image.Rectangle) {
	scale := max(int(gtkWidgetGetScaleFactor(s.area)), 1)
	// GTK takes the area to repaint in the coordinates of the widget's
	// GdkWindow, whatever its documentation says: a GtkGLArea has none of
	// its own, and its parent's starts above and left of it, by the title
	// bar GTK draws, its shadow, or the menu bar.
	var at gdkRectangle
	if !gtkWidgetGetHasWindow(s.area) {
		gtkWidgetGetAllocation(s.area, &at)
	}
	queue := func(d image.Rectangle) {
		x0, y0 := d.Min.X/scale, d.Min.Y/scale
		gtkWidgetQueueDrawArea(s.area, at.X+int32(x0), at.Y+int32(y0), int32((d.Max.X+scale-1)/scale-x0), int32((d.Max.Y+scale-1)/scale-y0))
	}
	if !s.ticking {
		s.PresentPixels(pix, stride, width, height)
		if s.cr != 0 {
			var x0, y0, x1, y1 float64
			cairoClipExtents(s.cr, &x0, &y0, &x1, &y1)
			clip := image.Rect(int(math.Floor(x0))*scale, int(math.Floor(y0))*scale, int(math.Ceil(x1))*scale, int(math.Ceil(y1))*scale)
			for _, d := range damage {
				if !d.In(clip) {
					queue(d)
				}
			}
		}
		return
	}
	s.staged.pix, s.staged.stride, s.staged.width, s.staged.height = pix, stride, width, height
	for _, d := range damage {
		queue(d)
	}
}

// mallocTrim is glibc's malloc_trim, which other C libraries lack.
var mallocTrim, _ = purego.Dlsym(purego.RTLD_DEFAULT, "malloc_trim")

// Idle gives the system back the memory that C code freed while drawing
// frames, which glibc keeps for later: GTK paints a window it composites
// with OpenGL into an image as large as the window each frame, and glibc
// raises its threshold for giving large blocks back as they are freed.
func (s *surface) Idle() {
	s.staged.pix = nil // the host freed it
	if mallocTrim != 0 {
		purego.SyscallN(mallocTrim, 0)
	}
}

// RefreshRate returns the refresh rate of the monitor showing the area,
// which GDK knows in millihertz.
func (s *surface) RefreshRate() float64 {
	win := gtkWidgetGetWindow(s.area)
	if s.w.closed || win == 0 {
		return 0
	}
	m := gdkDisplayGetMonitorAtWin(gdkWindowGetDisplay(win), win)
	if m == 0 {
		return 0
	}
	return float64(gdkMonitorGetRefreshRate(m)) / 1000
}

func (s *surface) PresentPixels(pix []byte, stride, width, height int) {
	if s.cr == 0 {
		if s.rendering {
			// The content draws in memory after all, as when its GL
			// renderer fails: GTK shows only what OpenGL draws.
			s.inMemory = true
			if err := s.present.Present(pix, stride, width, height); err != nil && !s.presentFailed {
				s.presentFailed = true
				log.Printf("mygo: native UI shows nothing: %v", err)
			}
			return
		}
		// Frames are drawn in the draw signal; outside of it, ask for one.
		gtkWidgetQueueDraw(s.area)
		return
	}
	if len(pix) < stride*height || width == 0 || height == 0 {
		return
	}
	const formatARGB32, operatorSource, operatorOver = 0, 1, 2 // premultiplied, native endian: BGRA
	img := cairoImageSurfaceForData(&pix[0], formatARGB32, int32(width), int32(height), int32(stride))
	scale := float64(max(gtkWidgetGetScaleFactor(s.area), 1))
	cairoSurfaceSetDeviceScale(img, scale, scale)
	cairoSetSourceSurface(s.cr, img, 0, 0)
	op := int32(operatorSource)
	if s.layers != 0 {
		op = operatorOver // over the web views, which holes show
	}
	cairoSetOperator(s.cr, op)
	cairoPaint(s.cr)
	cairoSurfaceDestroy(img)
}

var cursorNames = map[platform.Cursor]string{
	platform.CursorDefault: "default", platform.CursorPointer: "pointer", platform.CursorText: "text",
	platform.CursorMove: "move", platform.CursorResizeEW: "ew-resize", platform.CursorResizeNS: "ns-resize",
	platform.CursorResizeNWSE: "nwse-resize", platform.CursorResizeNESW: "nesw-resize",
	platform.CursorNotAllowed: "not-allowed", platform.CursorCrosshair: "crosshair",
	platform.CursorGrab: "grab", platform.CursorGrabbing: "grabbing",
	platform.CursorResizeN: "n-resize", platform.CursorResizeE: "e-resize", platform.CursorResizeS: "s-resize",
	platform.CursorResizeW: "w-resize", platform.CursorResizeColumn: "col-resize", platform.CursorResizeRow: "row-resize",
	platform.CursorVerticalText: "vertical-text", platform.CursorCopy: "copy", platform.CursorAlias: "alias",
	platform.CursorContextMenu: "context-menu", platform.CursorNone: "none",
}

func (s *surface) SetCursor(c platform.Cursor) {
	s.cursor = c
	win := s.eventWindow()
	if win == 0 || s.w.cursor.on {
		return // a resize cursor of the edges shows
	}
	cur, ok := surfaceCursors[c]
	if !ok {
		name := cursorNames[c]
		if name == "" {
			name = "default"
		}
		cur = gdkCursorNewFromName(gtkWidgetGetDisplay(s.area), cs(name))
		surfaceCursors[c] = cur
	}
	gdkWindowSetCursor(win, cur)
}

func (s *surface) SetTextInput(t platform.TextInputState) {
	active, caret := t.Active, t.Caret
	if s.textInput && (!active || s.input.Client != t.Client || t.NoInputMethod) {
		s.clientComposition.Reset()
		gtkIMContextReset(s.im)
	}
	// Without input methods, the input context loses the focus, as a
	// Wayland compositor's input method grabs the keys while it has it,
	// and keys type the layout's characters (typed), as winit's do.
	if was, raw := s.rawInput(), active && t.NoInputMethod; raw != was {
		if raw {
			gtkIMContextFocusOut(s.im)
		} else if gtkWidgetHasFocus(s.area) {
			gtkIMContextFocusIn(s.im)
		}
	}
	s.input = t
	s.textInput, s.caret = active, caret
	if active {
		r := gdkRectangle{X: int32(caret.X), Y: int32(caret.Y), Width: max(int32(caret.W), 1), Height: int32(caret.H + 0.5)}
		gtkIMContextSetCursorLoc(s.im, &r)
	}
}

// rawInput reports whether the focused element takes text with input
// methods off (NoInputMethod).
func (s *surface) rawInput() bool { return s.textInput && s.input.NoInputMethod }

// typed sends the text a key press types with input methods off: the
// character of its keyval, unless a modifier makes it a shortcut, as
// GtkIMContextSimple has it.
func (s *surface) typed(event ptr) {
	if gdkMods(field[uint32](event, 24))&(platform.ModCtrl|platform.ModAlt|platform.ModSuper) != 0 {
		return
	}
	r := rune(gdkKeyvalToUnicode(field[uint32](event, 28)))
	if r < 0x20 || r == 0x7F {
		return
	}
	s.commit(string(r))
}

// commit sends text typed or committed by an input method.
func (s *surface) commit(text string) {
	if c := s.input.Client; c != nil {
		s.clientComposition.Replace(c, nil, text)
		return
	}
	if text != "" {
		s.send(platform.SurfaceEvent{Kind: platform.TextInput, Text: text})
	}
}

func (s *surface) send(ev platform.SurfaceEvent) bool {
	if ev.Kind == platform.PointerDown || ev.Kind == platform.SurfaceBlur {
		s.clientComposition.Reset()
	}
	if s.w.closed {
		return false
	}
	return s.w.h.SurfaceEvent(ev)
}

func gdkMods(state uint32) platform.Modifiers {
	var m platform.Modifiers
	if state&1 != 0 {
		m |= platform.ModShift
	}
	if state&(1<<2) != 0 {
		m |= platform.ModCtrl
	}
	if state&(1<<3) != 0 {
		m |= platform.ModAlt
	}
	if state&(1<<26|1<<28) != 0 {
		m |= platform.ModSuper
	}
	return m
}

// modifierKeyval is the modifier a GDK keyval presses: Shift_L/R, Control_L/R,
// Alt_L/R and Meta_L/R, Super_L/R and Hyper_L/R; 0 for other keys.
func modifierKeyval(keyval uint32) platform.Modifiers {
	switch keyval {
	case 0xffe1, 0xffe2:
		return platform.ModShift
	case 0xffe3, 0xffe4:
		return platform.ModCtrl
	case 0xffe9, 0xffea, 0xffe7, 0xffe8:
		return platform.ModAlt
	case 0xffeb, 0xffec, 0xffed, 0xffee:
		return platform.ModSuper
	}
	return 0
}

var gdkKeys = map[uint32]platform.Key{
	0xff0d: platform.KeyEnter, 0xff8d: platform.KeyEnter, 0xff1b: platform.KeyEscape, 0xff08: platform.KeyBackspace,
	0xff09: platform.KeyTab, 0xfe20: platform.KeyTab, 0x20: platform.KeySpace, 0xffff: platform.KeyDelete,
	0xff9f: platform.KeyDelete, 0xff63: platform.KeyInsert, 0xff50: platform.KeyHome, 0xff95: platform.KeyHome,
	0xff57: platform.KeyEnd, 0xff9c: platform.KeyEnd, 0xff55: platform.KeyPageUp, 0xff9a: platform.KeyPageUp,
	0xff56: platform.KeyPageDown, 0xff9b: platform.KeyPageDown, 0xff51: platform.KeyLeft, 0xff96: platform.KeyLeft,
	0xff52: platform.KeyUp, 0xff97: platform.KeyUp, 0xff53: platform.KeyRight, 0xff98: platform.KeyRight,
	0xff54: platform.KeyDown, 0xff99: platform.KeyDown, 0xff67: platform.KeyContextMenu,
	0x1008ff26: platform.KeyBack, 0x1008ff27: platform.KeyForward, // XF86Back, XF86Forward
}

func keyvalKey(keyval uint32) platform.Key {
	if k, ok := gdkKeys[keyval]; ok {
		return k
	}
	if keyval >= 0xffbe && keyval <= 0xffc9 {
		return platform.KeyF1 + platform.Key(keyval-0xffbe)
	}
	if r := gdkKeyvalToUnicode(keyval); r != 0 {
		return platform.KeyForRune(rune(r))
	}
	return platform.KeyUnknown
}

// eventPhysicalKey reads GdkEventKey.hardware_keycode on the supported
// 64-bit GTK ABI. X11 and Wayland expose the evdev code plus 8 here.
func eventPhysicalKey(event ptr) platform.Key {
	if code := field[uint16](event, 48); code >= 8 {
		return platform.KeyForScancode(code - 8)
	}
	return platform.KeyUnknown
}

func (b *Backend) surfaceOf(data ptr) *surface {
	if w := b.window(data); w != nil {
		return w.surface
	}
	return nil
}

func initSurfaceCallbacks() {
	b := func() *Backend { return theBackend }
	cbSurfaceDraw = purego.NewCallback(func(widget, cr, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return true
		}
		// A GtkGLArea draws in its render signal, unless it has no context:
		// then, as a drawing area, in this one.
		if s.drawsGL() {
			return false
		}
		s.cr = cr
		// The frame the tick callback drew, unless GTK resized the area
		// since; else GTK asked for this draw by itself, and the content
		// draws a frame now.
		st := &s.staged
		w, h, scale := s.Size()
		if st.pix != nil && st.width == int(math.Ceil(w*scale)) && st.height == int(math.Ceil(h*scale)) {
			s.PresentPixels(st.pix, st.stride, st.width, st.height)
		} else {
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceFrame})
		}
		st.pix = nil
		s.cr = 0
		return true
	})
	// The tick callback draws the frames asked for in memory, before GTK
	// paints, as long as they are asked for.
	cbSurfaceTick = purego.NewCallback(func(widget, clock, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		if s.drawsGL() {
			// The area has its context now: frames come from its render
			// signal.
			s.tick = 0
			gtkWidgetQueueDraw(s.area)
			return false
		}
		s.again, s.ticking, s.staged.pix = false, true, nil
		s.send(platform.SurfaceEvent{Kind: platform.SurfaceFrame})
		s.ticking = false
		if !s.again {
			s.tick = 0
		}
		return s.again
	})
	// The GtkGLArea's context: OpenGL 3.3, else OpenGL ES 3.0. One GDK
	// cannot make leaves the area its error, so that it draws with cairo,
	// without the context GTK's own handler would make.
	cbAreaContext = purego.NewCallback(func(area, data ptr) ptr {
		ctx, gerr := glContext(gtkWidgetGetWindow(area))
		if gerr != 0 {
			gtkGLAreaSetError(area, gerr)
			gErrorFree(gerr)
		}
		return ctx
	})
	cbSurfaceRender = purego.NewCallback(func(area, context, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		s.rendering, s.rendered, s.inMemory = true, true, false
		s.send(platform.SurfaceEvent{Kind: platform.SurfaceFrame})
		s.rendering = false
		return true
	})
	cbSurfaceSize = purego.NewCallback(func(widget, allocation, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceResize})
		}
	})
	cbSurfaceScale = purego.NewCallback(func(widget, pspec, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceResize})
			gtkWidgetQueueDraw(s.area)
		}
	})
	// Input methods let go of the GdkWindow before it goes, as GtkEntry
	// has them: fcitx5's disconnects from it.
	cbSurfaceUnrealize = purego.NewCallback(func(widget, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			gtkIMContextSetClientWindow(s.im, 0)
		}
	})
	cbSurfaceRealize = purego.NewCallback(func(widget, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			gtkIMContextSetClientWindow(s.im, s.eventWindow())
			if s.cursor != platform.CursorDefault {
				s.SetCursor(s.cursor) // on the area that took another's place
			}
		}
	})
	// GdkEventButton: type 0, time 20, x 24, y 32, state 48, button 52.
	cbSurfaceButton = purego.NewCallback(func(widget, event, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		typ := field[int32](event, 0)
		if typ != 4 && typ != 7 { // GDK_BUTTON_PRESS, GDK_BUTTON_RELEASE: GTK's own double click events add nothing
			return true
		}
		n, mods := field[uint32](event, 52), gdkMods(field[uint32](event, 48))
		if n == 8 || n == 9 {
			// The side buttons go back and forward, as keys do.
			kind, k := platform.KeyPressed, platform.KeyBack
			if typ == 7 {
				kind = platform.KeyReleased
			}
			if n == 9 {
				k = platform.KeyForward
			}
			s.send(platform.SurfaceEvent{Kind: kind, Key: k, Mods: mods})
			return true
		}
		button, ok := map[uint32]int{1: 0, 2: 2, 3: 1}[n]
		if !ok {
			return true
		}
		kind := platform.PointerDown
		if typ == 7 {
			kind = platform.PointerUp
		} else if !gtkWidgetHasFocus(s.area) {
			gtkWidgetGrabFocus(s.area)
		}
		if kind == platform.PointerDown && button == 0 {
			if s.lastPointer != 0 {
				gdkEventFree(s.lastPointer)
			}
			s.lastPointer = gdkEventCopy(event)
		}
		x, y := field[float64](event, 24), field[float64](event, 32)
		clicks := 0
		if kind == platform.PointerDown {
			clicks = s.clicks.Press(button, x, y, field[uint32](event, 20), doubleClick())
		}
		s.send(platform.SurfaceEvent{Kind: kind, X: x, Y: y, Button: button, Clicks: clicks, Mods: mods})
		if kind == platform.PointerUp && button == 0 && s.dragRequest == nil && s.lastPointer != 0 {
			gdkEventFree(s.lastPointer)
			s.lastPointer = 0
		}
		return true
	})
	// GdkEventMotion: x 24, y 32, state 48.
	cbSurfaceMotion = purego.NewCallback(func(widget, event, data ptr) bool {
		if s := b().surfaceOf(data); s != nil {
			if field[uint32](event, 48)&(1<<8) != 0 {
				if s.lastPointer != 0 {
					gdkEventFree(s.lastPointer)
				}
				s.lastPointer = gdkEventCopy(event)
			}
			s.send(platform.SurfaceEvent{Kind: platform.PointerMove, X: field[float64](event, 24), Y: field[float64](event, 32), Mods: gdkMods(field[uint32](event, 48))})
		}
		return false
	})
	// GdkEventCrossing: mode 72; only normal crossings leave.
	cbSurfaceLeave = purego.NewCallback(func(widget, event, data ptr) bool {
		if s := b().surfaceOf(data); s != nil && field[int32](event, 72) == 0 {
			s.send(platform.SurfaceEvent{Kind: platform.PointerLeave})
		}
		return false
	})
	// GdkEventScroll: x 24, y 32, state 40, direction 44, delta_x 72, delta_y 80.
	cbSurfaceScroll = purego.NewCallback(func(widget, event, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		ev := platform.SurfaceEvent{Kind: platform.PointerScroll, X: field[float64](event, 24), Y: field[float64](event, 32), Mods: gdkMods(field[uint32](event, 40))}
		const step = 50
		switch field[int32](event, 44) {
		case 0:
			ev.DY = -step
		case 1:
			ev.DY = step
		case 2:
			ev.DX = -step
		case 3:
			ev.DX = step
		case 4: // smooth
			dx, dy := field[float64](event, 72), field[float64](event, 80)
			ev.DX, ev.DY = dx*step, dy*step
			ev.Precise = dx != float64(int64(dx)) || dy != float64(int64(dy))
		}
		s.send(ev)
		return true
	})
	// GdkEventKey: type 0, state 24, keyval 28, hardware_keycode 48.
	cbSurfaceKey = purego.NewCallback(func(widget, event, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		kind := platform.KeyPressed
		if field[int32](event, 0) == 9 { // GDK_KEY_RELEASE
			kind = platform.KeyReleased
		} else {
			s.clientComposition.Reset()
			if s.lastKey != 0 {
				gdkEventFree(s.lastKey)
			}
			s.lastKey = gdkEventCopy(event)
		}
		if s.rawInput() {
			if kind == platform.KeyPressed {
				defer s.typed(event)
			}
		} else if s.textInput && gtkIMContextFilterKeypress(s.im, event) {
			return true
		}
		// A modifier key on its own: the state is the one before the
		// event, so the key's own bit goes in or out.
		if bit := modifierKeyval(field[uint32](event, 28)); bit != 0 {
			state := gdkMods(field[uint32](event, 24))
			if kind == platform.KeyPressed {
				state |= bit
			} else {
				state &^= bit
			}
			s.send(platform.SurfaceEvent{Kind: platform.ModifiersChanged, Mods: state})
			return false
		}
		k := keyvalKey(field[uint32](event, 28))
		mods := gdkMods(field[uint32](event, 24))
		physical := eventPhysicalKey(event)
		// A key the layout types no US key for, as S on a Cyrillic layout,
		// still reaches shortcuts by its position.
		// A release is sent whatever the modifiers held now, so that a press
		// sent as a chord is let go of even after Ctrl was.
		if k == platform.KeyUnknown {
			switch {
			case kind == platform.KeyReleased && physical != platform.KeyUnknown:
				s.send(platform.SurfaceEvent{Kind: kind, PhysicalKey: physical, Mods: mods})
				return false
			case !platform.KeyByPosition(k, physical, mods):
				return false
			}
		}
		if kind == platform.KeyPressed {
			s.pressing = event
			defer func() { s.pressing = 0 }()
		}
		s.send(platform.SurfaceEvent{Kind: kind, Key: k, PhysicalKey: physical, Mods: mods})
		return true
	})
	cbSurfaceFocusIn = purego.NewCallback(func(widget, event, data ptr) bool {
		if s := b().surfaceOf(data); s != nil {
			if !s.rawInput() {
				gtkIMContextFocusIn(s.im)
			}
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceFocus})
		}
		return false
	})
	cbSurfaceFocusOut = purego.NewCallback(func(widget, event, data ptr) bool {
		if s := b().surfaceOf(data); s != nil {
			gtkIMContextFocusOut(s.im)
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceBlur})
		}
		return false
	})
	cbIMCommit = purego.NewCallback(func(im, str, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			s.commit(goStr(str))
		}
	})
	cbIMPreedit = purego.NewCallback(func(im, data ptr) {
		s := b().surfaceOf(data)
		if s == nil {
			return
		}
		var str ptr
		var cursor int32
		gtkIMContextGetPreedit(s.im, &str, nil, &cursor)
		text := takeStr(str)
		if c := s.input.Client; c != nil {
			s.clientComposition.Reset()
			caret := platform.UTF16Len(string([]rune(text)[:min(int(cursor), len([]rune(text)))]))
			c.SetMarkedText(nil, text, platform.TextRange{Start: caret, End: caret})
			return
		}
		s.send(platform.SurfaceEvent{Kind: platform.TextComposition, Text: text, Caret: min(int(cursor), utf8.RuneCountInString(text))})
	})
	cbIMPreeditEnd = purego.NewCallback(func(im, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			if c := s.input.Client; c != nil {
				s.clientComposition.End(c)
				return
			}
			s.send(platform.SurfaceEvent{Kind: platform.TextComposition})
		}
	})
}
