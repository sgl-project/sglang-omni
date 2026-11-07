//go:build js

// Package web is the backend for browsers (GOOS=js, GOARCH=wasm): windows
// showing content MyGo draws itself (package ui) on canvases of the page,
// drawn with WebGL 2 by the GL renderer (package gl), or in memory where
// the browser has no WebGL 2. A window fills the page's element with the
// ID "mygo", or else the page; windows opened later cover earlier ones.
// Windows of web pages, menus, trays and the like have nothing in a
// browser to stand for them, and report so.
//
// Go's main goroutine, which runs the event loop (Run), is the main thread.
// JavaScript calls Go's functions (js.FuncOf) on goroutines of their own:
// the backend has the main goroutine handle their events and waits for
// it, so that they can still prevent the browser's defaults.
package web

import (
	"bytes"
	"errors"
	"runtime"
	"strconv"
	"sync"
	"syscall/js"
	"time"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

var errUnsupported = errors.New("mygo: browsers do not support this")

// Backend is the browser backend.
type Backend struct {
	h      platform.AppHandler
	mainID uint64
	tasks  chan func()
	signal chan struct{}
	wake   chan struct{}
	quit   chan struct{}

	mu      sync.Mutex
	windows []*window
	// clip is the text copied, which pastes read: browsers give the
	// clipboard's text only to paste events (see surface.paste). clipData
	// is all that was copied or pasted, in its text formats, as HTML and
	// a rich text's own format.
	clip     string
	clipData transfer.Data
	// source is the appearance the app asked for: "light", "dark" or
	// "system".
	source string
	mac    bool
	funcs  []js.Func
}

// New returns the browser backend.
func New() *Backend {
	return &Backend{
		tasks:  make(chan func()),
		signal: make(chan struct{}, 1),
		wake:   make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}
}

func goroutineID() uint64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	b := bytes.TrimPrefix(buf[:n], []byte("goroutine "))
	b = b[:max(bytes.IndexByte(b, ' '), 0)]
	id, _ := strconv.ParseUint(string(b), 10, 64)
	return id
}

func (b *Backend) Name() string { return "js/webgl2" }

func (b *Backend) Init(h platform.AppHandler, _ platform.AppOptions) error {
	if js.Global().Get("document").IsUndefined() {
		return errors.New("mygo: no document to draw in (a worker?)")
	}
	b.h = h
	b.mainID = goroutineID()
	installFiles()
	nav := js.Global().Get("navigator")
	platformName := nav.Get("platform")
	b.mac = !platformName.IsUndefined() && len(platformName.String()) >= 3 && platformName.String()[:3] == "Mac"
	changed := b.funcOf(func(js.Value, []js.Value) any {
		b.sync(func() { b.h.ThemeChanged() })
		return nil
	})
	for _, q := range []string{"(prefers-color-scheme: dark)", "(prefers-reduced-motion: reduce)", "(prefers-reduced-transparency: reduce)", "(prefers-contrast: more)"} {
		js.Global().Call("matchMedia", q).Call("addEventListener", "change", changed)
	}
	// The languages the user reads: the browser's.
	js.Global().Call("addEventListener", "languagechange", b.funcOf(func(js.Value, []js.Value) any {
		b.sync(func() { b.h.LocaleChanged() })
		return nil
	}))
	// The page's own choice of appearance (themeAttr), as a page framing
	// the app sets it, since frames take the media's.
	js.Global().Get("MutationObserver").New(changed).Call("observe", js.Global().Get("document").Get("documentElement"),
		map[string]any{"attributes": true, "attributeFilter": []any{themeAttr}})
	return nil
}

// funcOf wraps fn for JavaScript, for the life of the app.
func (b *Backend) funcOf(fn func(this js.Value, args []js.Value) any) js.Func {
	f := js.FuncOf(fn)
	b.funcs = append(b.funcs, f)
	return f
}

func (b *Backend) Run() error {
	b.h.Ready()
	for {
		select {
		case <-b.quit:
			b.h.Terminating()
			return nil
		case <-b.signal:
			b.h.Dispatch()
		case t := <-b.tasks:
			t()
		case <-b.wake:
		}
	}
}

func (b *Backend) Quit() {
	select {
	case <-b.quit:
	default:
		close(b.quit)
	}
}

func (b *Backend) IsMainThread() bool { return b.mainID == 0 || goroutineID() == b.mainID }

func (b *Backend) Signal() {
	select {
	case b.signal <- struct{}{}:
	default:
	}
}

func (b *Backend) Step() {
	select {
	case <-b.signal:
		b.h.Dispatch()
	case t := <-b.tasks:
		t()
	case <-b.wake:
	case <-time.After(50 * time.Millisecond):
	}
}

func (b *Backend) Wake() {
	select {
	case b.wake <- struct{}{}:
	default:
	}
}

// sync runs fn on the main goroutine and waits for it: JavaScript's
// callbacks handle their events through it.
func (b *Backend) sync(fn func()) {
	if b.IsMainThread() {
		fn()
		return
	}
	done := make(chan struct{})
	select {
	case b.tasks <- func() { defer close(done); fn() }:
		<-done
	case <-b.quit:
	}
}

func (b *Backend) NewWindow(o *platform.WindowOptions, h platform.WindowHandler) (platform.Window, error) {
	if !o.Surface {
		return nil, errors.New("mygo: browsers show windows of native UI (WindowOptions.Content) only")
	}
	w, err := newWindow(b, o, h)
	if err != nil {
		return nil, err
	}
	b.mu.Lock()
	b.windows = append(b.windows, w)
	b.mu.Unlock()
	return w, nil
}

func (b *Backend) removeWindow(w *window) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, x := range b.windows {
		if x == w {
			b.windows = append(b.windows[:i], b.windows[i+1:]...)
			return
		}
	}
}

func (*Backend) SetApplicationMenu(*platform.Menu)                   {}
func (*Backend) UpdateMenuItem(*platform.MenuItem)                   {}
func (b *Backend) App() platform.AppController                       { return app{} }
func (b *Backend) Dialogs() platform.Dialogs                         { return dialogs{} }
func (b *Backend) Clipboard() platform.Clipboard                     { return clipboard{b} }
func (b *Backend) Shell() platform.Shell                             { return shell{} }
func (b *Backend) Screen() platform.Screen                           { return screen{} }
func (b *Backend) Theme() platform.Theme                             { return theme{b} }
func (b *Backend) Power() platform.Power                             { return power{} }
func (*Backend) NewTray(platform.TrayHandler) (platform.Tray, error) { return nil, errUnsupported }
func (*Backend) RegisterHotkey(int, string) error                    { return errUnsupported }
func (*Backend) UnregisterHotkey(int)                                {}
func (*Backend) NotificationsSupported() bool                        { return false }
func (*Backend) WebViewAvailable() bool                              { return false }
func (*Backend) RemoveNotification(string)                           {}
func (*Backend) RemoveAllNotifications()                             {}

func (*Backend) ShowNotification(_ *platform.Notification, done func(error)) {
	done(errUnsupported)
}

type app struct{}

func (app) SetActivationPolicy(string)                        {}
func (app) Activate()                                         {}
func (app) Hide()                                             {}
func (app) Unhide()                                           {}
func (app) IsHidden() bool                                    { return js.Global().Get("document").Get("hidden").Bool() }
func (app) SetBadge(string)                                   {}
func (app) Badge() string                                     { return "" }
func (app) Bounce(bool) int                                   { return 0 }
func (app) CancelBounce(int)                                  {}
func (app) SetDockIcon([]byte) error                          { return errUnsupported }
func (app) ShowAboutPanel(platform.AboutPanelOptions)         {}
func (app) Package() (platform.PackageInfo, bool)             { return platform.PackageInfo{}, false }
func (app) RegisterURLScheme(string, string, string) error    { return errUnsupported }
func (app) UnregisterURLScheme(string, string, string) error  { return errUnsupported }
func (app) IsURLSchemeRegistered(string, string, string) bool { return false }
func (app) SetOpenAtLogin(bool, string, string, string) error { return errUnsupported }
func (app) OpenAtLogin(string, string, string) bool           { return false }
func (app) OpenedAtLogin() bool                               { return false }
func (app) SetDockMenu(*platform.Menu)                        {}
func (app) ClearBrowsingData(done func(error))                { done(errUnsupported) }

// PreferredLanguages returns the browser's languages, navigator.languages.
func (app) PreferredLanguages() []string {
	nav := js.Global().Get("navigator")
	var out []string
	if list := nav.Get("languages"); list.Type() == js.TypeObject {
		for i := range list.Length() {
			if l := list.Index(i); l.Type() == js.TypeString {
				out = append(out, l.String())
			}
		}
	}
	if len(out) == 0 {
		if l := nav.Get("language"); l.Type() == js.TypeString {
			out = append(out, l.String())
		}
	}
	return out
}

// Region returns the formats of the browser's locale, as Intl resolves
// it: its first day of the week where the browser tells it
// (Intl.Locale's getWeekInfo, or weekInfo before), and its hour cycle.
func (app) Region() platform.Region {
	r := platform.Region{FirstWeekday: -1}
	intl := js.Global().Get("Intl")
	if intl.Type() != js.TypeObject {
		return r
	}
	opts := intl.Get("DateTimeFormat").New(js.Undefined(), map[string]any{"hour": "numeric"}).Call("resolvedOptions")
	if l := opts.Get("locale"); l.Type() == js.TypeString {
		r.Locale = l.String()
	}
	switch opts.Get("hourCycle").String() {
	case "h11", "h12":
		r.HourCycle = 12
	case "h23", "h24":
		r.HourCycle = 24
	}
	if r.Locale == "" || intl.Get("Locale").Type() != js.TypeFunction {
		return r
	}
	func() {
		defer func() { recover() }() // a browser that throws on either
		locale := intl.Get("Locale").New(r.Locale)
		info := js.Undefined()
		if locale.Get("getWeekInfo").Type() == js.TypeFunction {
			info = locale.Call("getWeekInfo")
		} else {
			info = locale.Get("weekInfo")
		}
		if info.Type() == js.TypeObject && info.Get("firstDay").Type() == js.TypeNumber {
			r.FirstWeekday = info.Get("firstDay").Int() % 7 // 7 is Sunday
		}
	}()
	return r
}

// localeKey is the key of localStorage the app's language is kept under.
const localeKey = "mygo:locale"

// SavedLocale reads the app's language from the page's localStorage.
func (app) SavedLocale() (tag string, kept bool) {
	defer func() { recover() }() // storage the page may not use
	if v := js.Global().Get("localStorage").Call("getItem", localeKey); v.Type() == js.TypeString {
		tag = v.String()
	}
	return tag, true
}

// SaveLocale keeps the app's language in the page's localStorage.
func (app) SaveLocale(tag string) bool {
	defer func() { recover() }()
	storage := js.Global().Get("localStorage")
	if tag == "" {
		storage.Call("removeItem", localeKey)
	} else {
		storage.Call("setItem", localeKey, tag)
	}
	return true
}

// SetLanguages makes the app's language the page's (<html lang>), which
// browsers draw the text of canvases and read the page aloud in, and its
// direction the page's (<html dir>).
func (app) SetLanguages(tags []string, rtl bool) {
	root := js.Global().Get("document").Get("documentElement")
	if len(tags) > 0 {
		root.Set("lang", tags[0])
	}
	dir := "ltr"
	if rtl {
		dir = "rtl"
	}
	root.Set("dir", dir)
}

type dialogs struct{}

func (dialogs) ShowOpenDialog(_ platform.Window, _ *platform.OpenDialogOptions, cb func([]string, error)) {
	cb(nil, errUnsupported)
}

func (dialogs) ShowSaveDialog(_ platform.Window, _ *platform.SaveDialogOptions, cb func(string, error)) {
	cb("", errUnsupported)
}

// ShowMessageBox asks with the browser's own dialogs: an alert for one
// button, a confirm for more, whose OK is the first button and Cancel the
// cancel button.
func (dialogs) ShowMessageBox(_ platform.Window, o *platform.MessageBoxOptions, cb func(platform.MessageBoxResult, error)) {
	text := o.Message
	if o.Detail != "" {
		text += "\n\n" + o.Detail
	}
	if len(o.Buttons) <= 1 {
		js.Global().Call("alert", text)
		cb(platform.MessageBoxResult{}, nil)
		return
	}
	res := o.CancelID
	if js.Global().Call("confirm", text).Bool() {
		res = 0
		if res == o.CancelID {
			res = 1
		}
	}
	cb(platform.MessageBoxResult{Response: res, CheckboxChecked: o.CheckboxChecked}, nil)
}

// clipboard keeps the text copied, and writes it to the system's
// clipboard as well, which browsers allow while the user acts (a key
// copying). Reading the system's clipboard takes a paste event, which
// gives the surface its text (surface.paste).
type clipboard struct{ b *Backend }

func (c clipboard) ReadText() string {
	c.b.mu.Lock()
	defer c.b.mu.Unlock()
	return c.b.clip
}

func (c clipboard) WriteText(text string) {
	c.b.mu.Lock()
	c.b.clip = text
	c.b.clipData = transfer.TextData(text)
	c.b.mu.Unlock()
	if cb := js.Global().Get("navigator").Get("clipboard"); !cb.IsUndefined() {
		p := cb.Call("writeText", text)
		p.Call("catch", js.Global().Get("Function").New("")) // without the user acting
	}
}

func (c clipboard) ReadData(formats []transfer.Format) (transfer.Data, error) {
	c.b.mu.Lock()
	d := c.b.clipData
	c.b.mu.Unlock()
	var reps []transfer.Representation
	for _, f := range formats {
		if b, err := d.Read(f); err == nil {
			reps = append(reps, transfer.Bytes(f, b))
		}
	}
	if formats == nil {
		return d, nil
	}
	if len(reps) == 0 {
		return transfer.Data{}, nil
	}
	return transfer.New(transfer.NewItem(reps...)), nil
}

// WriteData keeps data's text formats, which the copy event under way
// gives the browser (surface.copy), and writes its plain text to the
// system's clipboard, which browsers allow while the user acts.
func (c clipboard) WriteData(d transfer.Data, released func()) error {
	t, err := d.Read(transfer.Text)
	if err != nil {
		return errUnsupported
	}
	d, err = d.Materialize(d.Formats())
	if err != nil {
		return err
	}
	c.WriteText(string(t))
	c.b.mu.Lock()
	c.b.clipData = d
	c.b.mu.Unlock()
	if released != nil {
		released()
	}
	return nil
}

func (c clipboard) Formats() []transfer.Format {
	c.b.mu.Lock()
	defer c.b.mu.Unlock()
	if c.b.clip == "" && len(c.b.clipData.Items()) == 0 {
		return nil
	}
	return c.b.clipData.Formats()
}

func (clipboard) Flush() error                 { return nil }
func (clipboard) Close()                       {}
func (clipboard) ReadHTML() string             { return "" }
func (clipboard) WriteHTML(string)             {}
func (clipboard) ReadImage() []byte            { return nil }
func (clipboard) WriteImage([]byte) error      { return errUnsupported }
func (c clipboard) Clear()                     { c.WriteText("") }
func (c clipboard) AvailableFormats() []string { return []string{"text/plain"} }

type shell struct{}

func (shell) OpenExternal(url string) error {
	js.Global().Call("open", url, "_blank", "noopener")
	return nil
}
func (shell) OpenPath(string) error                { return errUnsupported }
func (shell) ShowItemInFolder(string)              {}
func (shell) TrashItem(string) error               { return errUnsupported }
func (shell) Beep()                                {}
func (shell) FileIcon(string, int) ([]byte, error) { return nil, errUnsupported }
func (shell) OpenApplication(string) error         { return errUnsupported }

func (*Backend) Applications() ([]platform.Application, error) { return nil, errUnsupported }

type screen struct{}

func (screen) Displays() []platform.Display {
	s := js.Global().Get("screen")
	return []platform.Display{{
		ID:          1,
		Bounds:      platform.Rect{Width: s.Get("width").Int(), Height: s.Get("height").Int()},
		WorkArea:    platform.Rect{Width: s.Get("availWidth").Int(), Height: s.Get("availHeight").Int()},
		ScaleFactor: devicePixelRatio(),
		Primary:     true,
	}}
}

func (screen) CursorPoint() platform.Point { return platform.Point{} }

func devicePixelRatio() float64 {
	if r := js.Global().Get("devicePixelRatio"); r.Type() == js.TypeNumber && r.Float() > 0 {
		return r.Float()
	}
	return 1
}

func matches(query string) bool {
	return js.Global().Call("matchMedia", query).Get("matches").Bool()
}

type theme struct{ b *Backend }

// themeAttr is the attribute of the page's root element that chooses the
// appearance, "dark" or "light", over the browser's: a page framing the
// app sets it to its own, which the frame's prefers-color-scheme does not
// follow in Chrome.
const themeAttr = "data-mygo-theme"

func (t theme) IsDark() bool {
	t.b.mu.Lock()
	source := t.b.source
	t.b.mu.Unlock()
	switch source {
	case "dark":
		return true
	case "light":
		return false
	}
	switch js.Global().Get("document").Get("documentElement").Call("getAttribute", themeAttr).String() {
	case "dark":
		return true
	case "light":
		return false
	}
	return matches("(prefers-color-scheme: dark)")
}

func (t theme) SetSource(source string) {
	t.b.mu.Lock()
	t.b.source = source
	t.b.mu.Unlock()
	t.b.h.ThemeChanged()
}

func (theme) UIFont() string                        { return "" }
func (theme) FontRendering() platform.FontRendering { return platform.FontRendering{} }

func (theme) Preferences() platform.Preferences {
	return platform.Preferences{
		ReduceMotion:       matches("(prefers-reduced-motion: reduce)"),
		ReduceTransparency: matches("(prefers-reduced-transparency: reduce)"),
		HighContrast:       matches("(prefers-contrast: more)"),
		TextScale:          1,
	}
}

type power struct{}

func (power) Watch()                        {}
func (power) KeepAwake(bool, string) func() { return func() {} }
func (power) OnBattery() bool               { return false }
func (power) IdleTime() time.Duration       { return 0 }
