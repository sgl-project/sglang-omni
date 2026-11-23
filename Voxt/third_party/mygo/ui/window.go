package ui

import (
	"log"
	"os"
	"slices"
	"time"

	"github.com/egoist/mygo/i18n"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/raster"
	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/surface"
	"github.com/egoist/mygo/internal/text"
	"github.com/egoist/mygo/transfer"
)

// Content is a user interface for a window, the value of
// mygo.WindowOptions.Content. Create it with View.
type Content struct {
	view func(*context)
	// unsmoothed is FontSmoothing(false).
	unsmoothed bool
}

// FontSmoothing turns font smoothing off (or back on) for the text of the
// windows showing the content. On macOS, Core Text's font smoothing, on
// while the user leaves it on, emboldens glyphs, the more the lighter the
// text; AppKit applies it to text over an opaque background. Text over a
// window's material or a transparent window, as a floating HUD's, can
// turn it off to draw as thin as text styled -webkit-font-smoothing:
// antialiased (Flat glyphs). Thickened glyphs (Font.Thicken) stay thick.
// Other platforms, which do not smooth fonts this way, ignore it. Set it
// before the content is attached to a window.
func (v *Content) FontSmoothing(on bool) *Content {
	v.unsmoothed = !on
	return v
}

// View returns the content of a window whose user interface view builds,
// for mygo.WindowOptions.Content:
//
//	mygo.NewWindow(mygo.WindowOptions{Title: "Counter", Content: ui.View(app.View)})
//
// view runs on the main thread whenever the window needs a frame: after
// input, after Context.Invalidate or Window.Invalidate, and while
// something animates. A Content can serve several windows, each with its
// own state.
func coreView(view func(c *context)) *Content { return &Content{view: view} }

// RegisterFont adds a TrueType or OpenType font, or collection, that text
// can use with Font(family). An empty family keeps the font's own name.
func RegisterFont(data []byte, family string) error {
	return text.Shared().RegisterFont(data, family)
}

// A FontFamily is a font family text can use, as FontFamilies lists it.
type FontFamily struct {
	// Name is the family as Font and Theme take it.
	Name string
	// Monospaced is whether its regular font gives every character the
	// same advance, as a terminal's or a code editor's font setting
	// offers.
	Monospaced bool
	// Registered is whether the app registered it with RegisterFont, under
	// Name: the name it gave, or the font's own when it gave none.
	Registered bool
}

// FontFamilies lists the font families text can use, as a font setting
// offers them: the app's registered ones first, then the system's, each
// once, sorted without regard to case. A font registered under a name the
// app gave is listed under that name alone. Browsers do not tell what
// fonts they have: there the list holds the registered families alone,
// and a family is Monospaced once its font has loaded.
//
// The first call asks the system for its families, which takes a while
// with many fonts installed (tens of milliseconds on a Mac with hundreds),
// without holding up the windows' text; later calls reuse that list, until
// RegisterFont, so fonts installed while the app runs show after a
// restart. It is safe from any goroutine.
func FontFamilies() []FontFamily {
	families := text.Shared().FontFamilies()
	out := make([]FontFamily, len(families))
	for i, f := range families {
		out[i] = FontFamily(f)
	}
	return out
}

// AttachContent connects the content to a window; package mygo calls it.
func (v *Content) AttachContent(conn *surface.Conn) {
	h := &windowHost{conn: conn}
	rt := newRuntime(v.view, h)
	rt.unsmoothed = v.unsmoothed
	h.rt = rt
	conn.Event = h.event
	conn.ThemeChanged = func() {
		// The interface font is part of the appearance.
		h.uiFont()
		rt.themeChanged()
	}
	conn.TitleBarChanged = rt.requestFrame
	conn.Changed = rt.changed
	conn.LocaleChanged = func() {
		h.textLanguage()
		rt.changed()
	}
	conn.Capture = h.capture
	conn.Detach = h.detach
	rt.insp.enabled = conn.DevTools
	conn.ToggleDevTools = func() {
		if rt.insp.enabled {
			rt.toggleInspector()
		}
	}
	h.uiFont()
	h.textLanguage()
	// Load the fonts while the window shows up.
	go text.Shared().Preload()
	conn.Surface.RequestFrame()
}

// windowHost presents frames on a window's surface, with a GPU renderer
// when the platform has one and in memory otherwise.
type windowHost struct {
	conn *surface.Conn
	rt   *engine
	gpu  gpuRenderer
	// gpuTried tells that the first frame tried the GPU, onGPU that its
	// renderer drew on a GPU, degraded that one drawing in software took
	// its place, and gpuSince when the renderer was made. After a failure,
	// retryAt is when a frame makes another, and backoff how long the next
	// failure waits.
	gpuTried bool
	onGPU    bool
	degraded bool
	gpuSince time.Time
	retryAt  time.Time
	backoff  time.Duration
	soft     raster.Renderer
	last     *scene.Scene
	// lastFrame is when the last frame was presented.
	lastFrame time.Time
	// framing tells that the surface asked for the frame being drawn.
	framing bool
	// told is what accelerated last told the frame being painted, asked
	// that it did.
	told, asked bool
	// path is how the last frame was drawn, for MYGO_FRAME_STATS.
	path string
	// wide tells that the GPU renderer draws the colors of scenes outside
	// the sRGB gamut (see wideHold), until wideUntil.
	wide      bool
	wideUntil time.Time
	// idleTimer runs idle once frames stop, at idleAt when idleArmed.
	idleTimer *time.Timer
	idleAt    time.Time
	idleArmed bool
	// afterT and repaintT are the timers of After and Painter.After.
	afterT, repaintT *time.Timer
	detached         bool
	// shown tells that a frame was presented, which makes the window
	// ready to show.
	shown bool
	// placed is where the surface shows the window's web views, views
	// the web views.
	placed []platform.WebViewPlacement
	views  []NativeWebView
}

func (h *windowHost) framePath() string {
	if h.path == "drawn on the GPU" && h.gpu != nil {
		// The stats logger calls this only while MYGO_FRAME_STATS is enabled.
		if f, ok := h.gpu.(interface{ FrameInfo() string }); ok {
			return h.path + ": " + f.FrameInfo()
		}
	}
	return h.path
}

func (h *windowHost) rightToLeft() bool { return h.conn.RightToLeft != nil && h.conn.RightToLeft() }

// newGPU makes the GPU renderer of a surface; tests replace it.
var newGPU = newGPURenderer

// event handles an event of the surface, noting when it asks for a frame:
// only then may renderers draw (OpenGL's context is current only then).
func (h *windowHost) event(ev platform.SurfaceEvent) bool {
	if ev.Kind == platform.SurfaceRenew {
		// The next frame makes the renderer again, for the surface's new
		// native objects.
		if h.gpu != nil {
			h.gpu.Release()
			h.gpu = nil
		}
		h.gpuTried, h.retryAt = false, time.Time{}
		h.conn.Surface.RequestFrame()
		return false
	}
	if ev.Kind == platform.WebViewTabOut {
		if i := slices.IndexFunc(h.placed, func(p platform.WebViewPlacement) bool { return p.WebView == ev.WebView }); i >= 0 {
			h.rt.webViewTabOut(h.views[i], ev.Mods&platform.ModShift != 0)
		}
		return false
	}
	if ev.Kind == platform.SurfaceFrame {
		h.framing = true
		defer func() { h.framing = false }()
	}
	return h.rt.event(ev)
}

// gpuRenderer draws scenes into a surface on the GPU.
type gpuRenderer interface {
	Render(s *scene.Scene) error
	Release()
}

// wideRenderer is a GPU renderer that can draw the colors of scenes outside
// the sRGB gamut (Metal's).
type wideRenderer interface {
	// SetWide has the frames drawn from now on show those colors, into
	// drawables of a wide gamut, or their nearest sRGB ones, and reports
	// whether they do.
	SetWide(on bool) bool
}

// A window whose screen shows more than sRGB, as a Mac's Display P3
// screen, draws the colors of ui.Oklch outside the sRGB gamut once a frame
// has some, on the GPU, whose drawables then hold float16 components, and
// goes on for wideHold after the last frame with some: a blinking caret or
// a color in transition does not switch the drawables back and forth. The
// CPU draws sRGB colors: frames meanwhile are the GPU's, and those after
// it may draw on the CPU again. A window on an sRGB screen draws the
// nearest sRGB colors, as it would show anyway, with the CPU's frames.
const wideHold = 2 * time.Second

// frameIdle is how long after the last frame the host frees the frame
// drawn in memory, as large as the window, and the surface gives back what
// its frames took (platform.IdleSurface). The next frame draws whole.
const frameIdle = 2 * time.Second

func (h *windowHost) refreshRate() float32 { return float32(h.conn.Surface.RefreshRate()) }

func (h *windowHost) occluded() bool {
	s, ok := h.conn.Surface.(platform.OccludableSurface)
	return ok && s.Occluded()
}

func (h *windowHost) size() (float32, float32, float32) {
	w, ht, s := h.conn.Surface.Size()
	if s <= 0 {
		s = 1
	}
	return float32(w), float32(ht), float32(s)
}

func (h *windowHost) present(s *scene.Scene) {
	h.last, h.path = s, ""
	defer h.recheck()
	if s.Width <= 0 || s.Height <= 0 {
		return
	}
	if !h.framing {
		// A frame built outside the surface's, for a capture, is shown by
		// the surface's next.
		h.conn.Surface.RequestFrame()
		return
	}
	due := !h.retryAt.IsZero() && !time.Now().Before(h.retryAt)
	switch {
	case !h.gpuTried || h.gpu == nil && due:
		h.makeGPU()
	case due && h.degraded:
		// Drawing in software since the GPU went away: is it back?
		h.gpu.Release()
		h.gpu = nil
		h.makeGPU()
	}
	now := time.Now()
	h.lastFrame = now
	h.armIdle(frameIdle)
	h.useWide(s, now)
	defer h.ready()
	if h.render(s) {
		h.path = "drawn on the GPU"
		if h.degraded {
			h.path = "drawn by the GPU renderer in software"
		}
		if behind(h.gpu) {
			h.path = "skipped to catch up with the display"
			h.conn.Surface.RequestFrame()
		}
		return
	}
	h.path = "drawn in memory"
	damage := h.soft.Render(s)
	m := &h.soft.Image
	if d, ok := h.conn.Surface.(platform.DamageSurface); ok {
		d.PresentDamage(m.Pix, m.Stride, m.W, m.H, damage)
	} else {
		h.conn.Surface.PresentPixels(m.Pix, m.Stride, m.W, m.H)
	}
}

// ready tells the window, after the first frame, that it is ready to show:
// once the frame returns, as its listeners may show the window.
func (h *windowHost) ready() {
	if h.shown {
		return
	}
	h.shown = true
	if h.conn.Ready != nil && h.conn.Post != nil {
		h.conn.Post(h.conn.Ready)
	}
}

// armIdle has idle run d from now, unless it runs sooner already.
func (h *windowHost) armIdle(d time.Duration) {
	if h.conn.Post == nil {
		return
	}
	at := time.Now().Add(d)
	if h.idleArmed && !h.idleAt.After(at) {
		return
	}
	h.idleAt, h.idleArmed = at, true
	if h.idleTimer == nil {
		h.idleTimer = time.AfterFunc(d, func() { h.conn.Post(h.idle) })
	} else {
		h.idleTimer.Reset(d)
	}
}

// idle runs on the main thread after frames stop: it frees the frame drawn
// in memory, and has the surface give back memory, once the window is idle
// for frameIdle. A frame since rearms it.
func (h *windowHost) idle() {
	h.idleArmed = false
	if h.detached {
		return
	}
	since := time.Since(h.lastFrame)
	if since < frameIdle {
		h.armIdle(frameIdle - since)
		return
	}
	h.soft.Release()
	if s, ok := h.conn.Surface.(platform.IdleSurface); ok {
		s.Idle()
	}
}

// useWide has the GPU renderer draw the colors of s outside the sRGB gamut
// when the window shows them (see wideHold), and reports whether it does.
func (h *windowHost) useWide(s *scene.Scene, now time.Time) bool {
	w, ok := h.gpu.(wideRenderer)
	if !ok {
		return false
	}
	if len(s.Wide) > 0 {
		h.wideUntil = now.Add(wideHold)
	}
	if on := now.Before(h.wideUntil) && h.screenWide(); on != h.wide {
		h.wide = w.SetWide(on)
	}
	return h.wide
}

// screenWide reports whether the window's screen shows more than sRGB.
func (h *windowHost) screenWide() bool {
	w, ok := h.conn.Surface.(platform.WideGamutSurface)
	return ok && w.WideGamut()
}

// render draws s with the GPU renderer and reports whether it did. One
// that fails gives way to another made at once, in case the GPU is back
// already, or another, or Windows' software rasterizer, can draw; when
// that fails too, frames are drawn in memory until a later try.
func (h *windowHost) render(s *scene.Scene) bool {
	for try := 0; h.gpu != nil; try++ {
		err := h.gpu.Render(s)
		if err == nil {
			return true
		}
		log.Printf("mygo: drawing without the GPU: %v", err)
		h.gpu.Release()
		h.gpu = nil
		if try > 0 {
			h.retryLater()
			return false
		}
		if time.Since(h.gpuSince) > time.Minute {
			h.backoff = 0 // it worked for a while: the waits start over
		}
		h.makeGPU()
	}
	return false
}

// makeGPU makes the surface's GPU renderer: for the first frame, after one
// failed, and when the wait after a failure is over. Where the first frame
// cannot have one, the window draws in memory for good, and where it has a
// GPU, a renderer that draws in software in its place tries for the GPU
// again from time to time.
func (h *windowHost) makeGPU() {
	retry := h.gpuTried
	h.gpuTried, h.retryAt = true, time.Time{}
	n := h.conn.Surface.Native()
	if os.Getenv("MYGO_GPU") == "0" || n == (platform.SurfaceNative{}) {
		return
	}
	r, err := newGPU(n)
	switch {
	case err != nil:
		log.Printf("mygo: drawing without the GPU: %v", err)
		if retry {
			h.retryLater()
		}
		return
	case r == nil:
		return
	case !retry:
		h.onGPU = !software(r)
	case h.onGPU && software(r):
		if !h.degraded {
			log.Print("mygo: drawing in software until the GPU is back")
		}
		h.degraded = true
		h.retryLater()
	default:
		if !software(r) {
			log.Print("mygo: drawing with the GPU again")
		}
		h.degraded = false
	}
	h.gpu, h.gpuSince = r, time.Now()
	h.wide = false // the renderer draws sRGB until told otherwise
}

// accelerated reports whether the frame draws on a GPU (Painter.Accelerated).
// The first frame makes its GPU renderer as it paints rather than as it
// presents, to tell.
func (h *windowHost) accelerated() bool {
	if h.framing && !h.gpuTried {
		h.makeGPU()
	}
	h.told, h.asked = h.gpu != nil && !software(h.gpu), true
	return h.told
}

// recheck paints the window again when the frame presented draws on a GPU
// and accelerated told its painters otherwise, or the other way round: the
// GPU went away or came back as it presented.
func (h *windowHost) recheck() {
	if h.asked && h.told != (h.gpu != nil && !software(h.gpu)) && h.conn.Invalidate != nil {
		h.invalidate()
	}
	h.asked = false
}

// software reports whether a renderer draws on the CPU, as Direct3D's WARP.
func software(r gpuRenderer) bool {
	s, ok := r.(interface{ Software() bool })
	return ok && s.Software()
}

// behind reports whether r drew nothing of the last frame, to catch up
// with the display (Metal's): the window draws another at the next
// refresh.
func behind(r gpuRenderer) bool {
	b, ok := r.(interface{ Behind() bool })
	return ok && b.Behind()
}

// retryLater has a frame make a GPU renderer again after a wait, which
// doubles with each failure up to half a minute: a driver that resets, a
// GPU unplugged or a machine waking up may take a while. A window that
// draws nothing meanwhile is not woken for it.
func (h *windowHost) retryLater() {
	h.backoff = min(max(2*h.backoff, time.Second), 30*time.Second)
	h.retryAt = time.Now().Add(h.backoff)
}

// capture renders the last frame in memory.
func (h *windowHost) capture() (int, int, []byte) {
	if h.last == nil {
		h.rt.runFrame()
	}
	s := h.last
	if s == nil {
		return 0, 0, nil
	}
	var img raster.Image
	img.Resize(s.Width, s.Height)
	raster.Render(&img, s)
	return img.W, img.H, img.RGBA()
}

func (h *windowHost) detach() {
	h.detached = true
	if h.idleTimer != nil {
		h.idleTimer.Stop()
	}
	h.rt.close()
	h.soft.Release()
	if h.gpu != nil {
		h.gpu.Release()
		h.gpu = nil
	}
}

func (h *windowHost) requestFrame() { h.conn.Surface.RequestFrame() }

func (h *windowHost) afterTimer(at time.Time) {
	switch {
	case at.IsZero():
		if h.afterT != nil {
			h.afterT.Stop()
		}
	case h.afterT != nil:
		h.afterT.Reset(max(time.Until(at), time.Millisecond))
	default:
		h.afterT = time.AfterFunc(max(time.Until(at), time.Millisecond), h.invalidate)
	}
}

func (h *windowHost) repaintTimer(at time.Time) {
	if at.IsZero() {
		if h.repaintT != nil {
			h.repaintT.Stop()
		}
		return
	}
	d := max(time.Until(at), time.Millisecond)
	if h.repaintT == nil {
		h.repaintT = time.AfterFunc(d, func() { h.post(h.rt.repaintNow) })
	} else {
		h.repaintT.Reset(d)
	}
}

func (h *windowHost) startDataDrag(data transfer.Data, local any, options transfer.DragOptions, x, y float32) error {
	return h.conn.StartDataDrag(data, local, options, float64(x), float64(y))
}
func (h *windowHost) cancelDataDrag()                    { h.conn.CancelDataDrag() }
func (h *windowHost) setDropFormats(f []transfer.Format) { h.conn.Surface.SetDropFormats(f) }

func (h *windowHost) post(fn func()) {
	if h.conn.Post != nil {
		h.conn.Post(fn)
	} else {
		h.conn.Invalidate()
	}
}

// uiFont gives system-ui the desktop's interface font, and text the
// desktop's settings for rasterizing it, where the text system does not
// know them.
func (h *windowHost) uiFont() {
	if h.conn.UIFont != nil {
		text.Shared().SetUIFamily(h.conn.UIFont())
	}
	if h.conn.FontRendering != nil {
		r := h.conn.FontRendering()
		text.Shared().SetFontRendering(r.Antialias, r.Hinting, r.Subpixels)
	}
}

func (h *windowHost) setCursor(c Cursor) { h.conn.Surface.SetCursor(platform.Cursor(c)) }

func (h *windowHost) setTextInput(t platform.TextInputState) { h.conn.Surface.SetTextInput(t) }

func (h *windowHost) updateAccessibility(tree *platform.AccessTree) {
	h.conn.Surface.UpdateAccessibility(tree)
}

func (h *windowHost) readClipboard() string {
	if h.conn.Clipboard == nil {
		return ""
	}
	return h.conn.Clipboard.ReadText()
}

func (h *windowHost) clipboardFormats() []transfer.Format {
	if h.conn.Clipboard == nil {
		return nil
	}
	return h.conn.Clipboard.Formats()
}

func (h *windowHost) writeClipboard(s string) {
	if h.conn.Clipboard != nil {
		h.conn.Clipboard.WriteText(s)
	}
}

func (h *windowHost) readClipboardData(formats []transfer.Format) transfer.Data {
	if h.conn.Clipboard == nil {
		return transfer.Data{}
	}
	d, err := h.conn.Clipboard.ReadData(formats)
	if err != nil && len(d.Items()) == 0 {
		return transfer.Data{}
	}
	return d
}

func (h *windowHost) writeClipboardData(d transfer.Data) {
	if h.conn.Clipboard != nil {
		h.conn.Clipboard.WriteData(d, nil)
	}
}

func (h *windowHost) startDrag() {
	if h.conn.StartDrag != nil {
		h.conn.StartDrag()
	}
}

func (h *windowHost) titleBarDoubleClicked() {
	if h.conn.TitleBarDoubleClicked != nil {
		h.conn.TitleBarDoubleClicked()
	}
}

func (h *windowHost) escape() {
	if h.conn.Escape != nil {
		h.conn.Escape()
	}
}

func (h *windowHost) isDark() bool { return h.conn.IsDark != nil && h.conn.IsDark() }

func (h *windowHost) preferences() platform.Preferences {
	if h.conn.Preferences == nil {
		return platform.Preferences{}
	}
	return h.conn.Preferences()
}

func (h *windowHost) locale() string {
	if h.conn.Locale == nil {
		return ""
	}
	return h.conn.Locale()
}

func (h *windowHost) region() i18n.Region {
	if h.conn.Region == nil {
		return i18n.Region{}
	}
	return h.conn.Region()
}

// textLanguage makes the language the app shows that of the text the
// text engine lays out without one, as Painter.Shape does.
func (h *windowHost) textLanguage() {
	text.Shared().SetLanguage(h.locale())
}

func (h *windowHost) titleBar() TitleBar {
	if h.conn.TitleBar == nil {
		return TitleBar{}
	}
	t := h.conn.TitleBar()
	return TitleBar{Height: float32(t.Height), Left: float32(t.Left), Right: float32(t.Right)}
}

func (h *windowHost) vibrancy() bool {
	s, ok := h.conn.Surface.(platform.MaterialSurface)
	return ok && s.ShowsMaterial()
}

func (h *windowHost) invalidate() { h.conn.Invalidate() }

func (h *windowHost) openURL(u string, done func(error)) {
	if h.conn.OpenURL != nil {
		h.conn.OpenURL(u, done)
	}
}

// placeWebViews gives the surface where the frame shows the window's web
// views, unless it is where the last frame showed them.
func (h *windowHost) placeWebViews(views []placedWebView) {
	list := make([]platform.WebViewPlacement, 0, len(views))
	natives := make([]NativeWebView, 0, len(views))
	for _, v := range views {
		if n := v.v.SurfaceWebView(h.conn); n != nil {
			list = append(list, platform.WebViewPlacement{WebView: n, Frame: v.frame, Clip: v.clip, Covers: v.covers})
			natives = append(natives, v.v)
		}
	}
	if slices.EqualFunc(list, h.placed, func(a, b platform.WebViewPlacement) bool {
		return a.WebView == b.WebView && a.Frame == b.Frame && a.Clip == b.Clip && slices.Equal(a.Covers, b.Covers)
	}) {
		return
	}
	h.placed, h.views = list, natives
	if h.conn.PlaceWebViews != nil {
		h.conn.PlaceWebViews(list)
	} else {
		h.conn.Surface.PlaceWebViews(list)
	}
}

func (h *windowHost) platformWebView(v NativeWebView) platform.WebView {
	return v.SurfaceWebView(h.conn)
}

func (h *windowHost) tabIntoWebView(v NativeWebView, back bool) {
	if n := v.SurfaceWebView(h.conn); n != nil {
		n.TabInto(back)
	}
}

func (h *windowHost) popupMenu(m *platform.Menu, x, y float32, chosen func(int)) {
	if h.conn.PopupMenu != nil {
		h.conn.PopupMenu(m, float64(x), float64(y), chosen)
	}
}
