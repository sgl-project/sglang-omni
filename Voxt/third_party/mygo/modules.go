package mygo

import (
	"bytes"
	"cmp"
	"encoding/json/v2"
	"errors"
	"fmt"
	"image/color"
	"image/png"
	"math"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/egoist/mygo/internal/accelerator"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

var errLoopStopped = errors.New("mygo: the event loop has stopped")

// ShellModule integrates with the desktop environment.
type ShellModule struct{}

// Shell opens URLs and files with the default applications.
var Shell ShellModule

// OpenExternal opens a URL with the default application, e.g. a web page
// in the default browser.
func (ShellModule) OpenExternal(url string) error {
	needsApp("Shell.OpenExternal")
	return onMainValue(func() error { return backend().Shell().OpenExternal(url) })
}

// OpenPath opens a file or directory with the default application.
func (ShellModule) OpenPath(path string) error {
	needsApp("Shell.OpenPath")
	return onMainValue(func() error { return backend().Shell().OpenPath(path) })
}

// ShowItemInFolder reveals a file in the file manager.
func (ShellModule) ShowItemInFolder(path string) {
	needsApp("Shell.ShowItemInFolder")
	onMain(func() { backend().Shell().ShowItemInFolder(path) })
}

// TrashItem moves a file or directory to the trash.
func (ShellModule) TrashItem(path string) error {
	needsApp("Shell.TrashItem")
	return onMainValue(func() error { return backend().Shell().TrashItem(path) })
}

// FileIcon returns the icon the file manager shows for a file, a folder or
// an application, as a PNG image of size by size pixels: pass the size in
// DIPs times the display's scale factor for a sharp icon. On Windows path
// may also name a shell item, such as an app of the Start menu by its
// shell:AppsFolder\<AppUserModelID> path; on Linux, the icon of a desktop
// entry (.desktop) is its application's, from the icon theme. It fails
// in browsers.
func (ShellModule) FileIcon(path string, size int) ([]byte, error) {
	needsApp("Shell.FileIcon")
	if size <= 0 {
		return nil, errors.New("mygo: Shell.FileIcon: size must be positive")
	}
	var png []byte
	err := onMainValue(func() (err error) {
		png, err = backend().Shell().FileIcon(path, size)
		return err
	})
	return png, err
}

// InstalledApp is an application installed on the computer, which
// Shell.InstalledApps lists.
type InstalledApp struct {
	// Name is the name the system shows for it, in the user's language.
	Name string
	// ID identifies it: its bundle identifier on macOS (or its path for a
	// bundle without one), its AppUserModelID on Windows (for a desktop
	// app without one, its path under a known folder), and its desktop
	// file ID on Linux, such as org.gnome.Nautilus.desktop.
	ID string
	// Path is where it is: its .app bundle on macOS, its item of the
	// Applications folder on Windows (shell:AppsFolder\<ID>), and its
	// desktop entry on Linux. Shell.FileIcon draws its icon.
	Path string
}

// InstalledApps lists the applications installed on the computer, sorted
// by name, as a launcher shows them:
//
//   - macOS: the .app bundles in /Applications, /System/Applications,
//     ~/Applications and CoreServices' Applications, and in their
//     folders (Utilities), plus Finder.
//   - Windows: the Start menu's All apps, desktop and packaged apps
//     alike (the shell's Applications folder).
//   - Linux: the applications of the desktop entries menus show.
//
// It reads the disk: call it from a goroutine of its own, which it does not
// block the main thread from. Browsers return an error.
func (ShellModule) InstalledApps() ([]InstalledApp, error) {
	needsApp("Shell.InstalledApps")
	if !isMainThread() {
		App.waitReady()
	}
	found, err := backend().Applications()
	if err != nil {
		return nil, err
	}
	apps := make([]InstalledApp, len(found))
	for i, a := range found {
		apps[i] = InstalledApp(a)
	}
	slices.SortStableFunc(apps, func(a, b InstalledApp) int {
		return cmp.Or(strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name)), strings.Compare(a.Path, b.Path))
	})
	return apps, nil
}

// LaunchApp starts an application of InstalledApps, or brings it to the
// front if it runs.
func (ShellModule) LaunchApp(app InstalledApp) error {
	needsApp("Shell.LaunchApp")
	if app.Path == "" {
		return errors.New("mygo: Shell.LaunchApp: the app has no path")
	}
	return onMainValue(func() error { return backend().Shell().OpenApplication(app.Path) })
}

// Beep plays the system alert sound.
func (ShellModule) Beep() {
	needsApp("Shell.Beep")
	onMain(func() { backend().Shell().Beep() })
}

// ClipboardModule reads and writes the system clipboard.
type ClipboardModule struct{}

// Clipboard is the system clipboard.
var Clipboard ClipboardModule

// ReadText returns the plain text on the clipboard.
func (ClipboardModule) ReadText() string {
	needsApp("Clipboard.ReadText")
	return clipboardNativeValue(func() string { return backend().Clipboard().ReadText() })
}

// WriteText puts plain text on the clipboard.
func (c ClipboardModule) WriteText(text string) {
	needsApp("Clipboard.WriteText")
	_ = c.Write(transfer.TextData(text))
}

// ReadHTML returns the HTML on the clipboard.
func (ClipboardModule) ReadHTML() string {
	needsApp("Clipboard.ReadHTML")
	return clipboardNativeValue(func() string { return backend().Clipboard().ReadHTML() })
}

// WriteHTML puts HTML on the clipboard.
func (c ClipboardModule) WriteHTML(markup string) {
	needsApp("Clipboard.WriteHTML")
	_ = c.Write(transfer.New(transfer.NewItem(transfer.Bytes(transfer.HTML, []byte(markup)), transfer.Bytes(transfer.Text, []byte(markup)))))
}

// ReadImage returns the image on the clipboard as PNG, or nil.
func (ClipboardModule) ReadImage() []byte {
	needsApp("Clipboard.ReadImage")
	return clipboardNativeValue(func() []byte { return backend().Clipboard().ReadImage() })
}

// WriteImage puts a PNG image on the clipboard.
func (c ClipboardModule) WriteImage(data []byte) error {
	needsApp("Clipboard.WriteImage")
	if _, err := png.Decode(bytes.NewReader(data)); err != nil {
		return err
	}
	return c.Write(transfer.New(transfer.NewItem(transfer.Bytes(transfer.PNG, data))))
}

// Clear empties the clipboard.
func (ClipboardModule) Clear() {
	needsApp("Clipboard.Clear")
	onMain(func() {
		if !clipboardProviding && !clipboardStopped {
			withClipboardOperation(func() { backend().Clipboard().Clear() })
		}
	})
}

// AvailableFormats lists the formats (MIME types or platform types) on the
// clipboard.
func (ClipboardModule) AvailableFormats() []string {
	needsApp("Clipboard.AvailableFormats")
	return clipboardNativeValue(func() []string { return backend().Clipboard().AvailableFormats() })
}

// Display describes a monitor.
type Display struct {
	ID    int64
	Label string
	// Bounds is the area of the display in screen coordinates.
	Bounds Rectangle
	// WorkArea excludes the menu bar, Dock and taskbars.
	WorkArea    Rectangle
	ScaleFactor float64
	Rotation    int
	Internal    bool
	Primary     bool
}

// ScreenModule queries displays.
type ScreenModule struct {
	onChanged listeners[func()]
}

// Screen describes the connected displays.
var Screen = &ScreenModule{}

// Displays returns all displays; the primary one comes first.
func (s *ScreenModule) Displays() []Display {
	needsApp("Screen.Displays")
	return onMainValue(func() []Display {
		var out []Display
		for _, d := range backend().Screen().Displays() {
			out = append(out, Display{
				ID:          d.ID,
				Label:       d.Label,
				Bounds:      Rectangle(d.Bounds),
				WorkArea:    Rectangle(d.WorkArea),
				ScaleFactor: d.ScaleFactor,
				Rotation:    d.Rotation,
				Internal:    d.Internal,
				Primary:     d.Primary,
			})
		}
		return out
	})
}

// PrimaryDisplay returns the display with the menu bar or taskbar.
func (s *ScreenModule) PrimaryDisplay() Display {
	needsApp("Screen.PrimaryDisplay")
	ds := s.Displays()
	for _, d := range ds {
		if d.Primary {
			return d
		}
	}
	if len(ds) > 0 {
		return ds[0]
	}
	return Display{}
}

// CursorScreenPoint returns the mouse position in screen coordinates.
func (s *ScreenModule) CursorScreenPoint() Point {
	needsApp("Screen.CursorScreenPoint")
	return onMainValue(func() Point { return Point(backend().Screen().CursorPoint()) })
}

// DisplayNearestPoint returns the display closest to p.
func (s *ScreenModule) DisplayNearestPoint(p Point) Display {
	needsApp("Screen.DisplayNearestPoint")
	best, bestDist := Display{}, math.Inf(1)
	for _, d := range s.Displays() {
		b := d.Bounds
		dx := math.Max(math.Max(float64(b.X-p.X), 0), float64(p.X-(b.X+b.Width)))
		dy := math.Max(math.Max(float64(b.Y-p.Y), 0), float64(p.Y-(b.Y+b.Height)))
		if dist := dx*dx + dy*dy; dist < bestDist {
			best, bestDist = d, dist
		}
	}
	return best
}

// DisplayMatching returns the display that overlaps r the most.
func (s *ScreenModule) DisplayMatching(r Rectangle) Display {
	needsApp("Screen.DisplayMatching")
	best, bestArea := Display{}, -1
	for _, d := range s.Displays() {
		b := d.Bounds
		w := min(b.X+b.Width, r.X+r.Width) - max(b.X, r.X)
		h := min(b.Y+b.Height, r.Y+r.Height) - max(b.Y, r.Y)
		area := max(w, 0) * max(h, 0)
		if area > bestArea {
			best, bestArea = d, area
		}
	}
	return best
}

// OnDisplaysChanged is called when displays are added, removed or change
// resolution.
func (s *ScreenModule) OnDisplaysChanged(fn func()) (off func()) { return s.onChanged.add(fn, false) }

func (s *ScreenModule) changed() { fire(&s.onChanged) }

// ThemeSource selects the appearance of the app.
type ThemeSource string

// Theme sources.
const (
	ThemeSystem ThemeSource = "system"
	ThemeLight  ThemeSource = "light"
	ThemeDark   ThemeSource = "dark"
)

// ThemePreferences are the accent color and the accessibility settings of
// the desktop, which an app and its pages should follow. Pages get them
// as JSON, as mygo-runtime's ThemePreferences, which is also what they
// encode to: {"accent": "#rrggbb" or null, "reduceMotion": …,
// "reduceTransparency": …, "highContrast": …, "textScale": …}.
type ThemePreferences struct {
	// Accent is the color of the desktop's controls, 8-bit sRGB with
	// straight alpha, or the zero color (A 0) where the desktop has none,
	// as Linux desktops whose settings portal has no accent-color and
	// browsers. Package ui's default theme already follows it; elsewhere
	// ui.RGB(Accent.R, Accent.G, Accent.B) makes it a ui.Color.
	Accent color.NRGBA
	// ReduceMotion reports that the desktop asks for less motion: macOS's
	// Reduce Motion, Windows's animation effects and GNOME's animations
	// turned off.
	ReduceMotion bool
	// ReduceTransparency reports that the desktop asks for opaque
	// surfaces in place of translucent ones, such as vibrancy: macOS's
	// Reduce Transparency, Windows's transparency effects turned off, and
	// prefers-reduced-transparency in browsers. Linux desktops have no
	// such setting, so it is false there.
	ReduceTransparency bool
	// HighContrast reports that the desktop asks for more contrast:
	// macOS's Increase Contrast, Windows's contrast themes, the settings
	// portal's higher contrast or a HighContrast GTK theme on Linux.
	HighContrast bool
	// TextScale is how many times larger than usual text should be, as
	// Windows's and GNOME's text size settings say; 1 is usual, and what
	// macOS, which has no such setting, reports.
	TextScale float64
}

// ThemeModule reads and overrides the light/dark appearance.
type ThemeModule struct {
	mu        sync.Mutex
	source    ThemeSource
	onUpdated listeners[func()]

	// The preferences read last, until the next change. Main thread only.
	prefs   ThemePreferences
	prefsOK bool
}

// Theme is the system appearance. Pages follow it through the
// prefers-color-scheme media query, and the accent color and accessibility
// settings through the --mygo-accent and --mygo-text-scale CSS variables
// and preferences and onPreferences of mygo-runtime.
var Theme = &ThemeModule{}

// IsDark reports whether the app currently uses a dark appearance.
func (t *ThemeModule) IsDark() bool {
	needsApp("Theme.IsDark")
	return onMainValue(func() bool { return backend().Theme().IsDark() })
}

// Preferences returns the accent color and the accessibility settings of
// the desktop, all read at once. It is safe from any goroutine; like the
// other calls that need the running app, it panics on the main goroutine
// before App.Run, and waits for Run elsewhere.
func (t *ThemeModule) Preferences() ThemePreferences {
	needsApp("Theme.Preferences")
	prefs := onMainValue(t.preferences)
	// After the app has quit, onMain drops the call and the zero value is
	// left; 0 is not a text scale.
	if prefs.TextScale <= 0 {
		prefs.TextScale = 1
	}
	return prefs
}

// preferences returns the desktop's preferences, read once per change. The
// read can call out to the desktop (the settings portal on Linux), so it is
// kept until the next change, as package ui keeps it. Main thread only.
func (t *ThemeModule) preferences() ThemePreferences {
	if !t.prefsOK {
		t.prefs = themePreferences(backend().Theme().Preferences())
		t.prefsOK = true
	}
	return t.prefs
}

// themePreferences is the public view of a backend's preferences. Policy
// lives here, not in the backends.
func themePreferences(p platform.Preferences) ThemePreferences {
	prefs := ThemePreferences{
		Accent:             color.NRGBA{R: p.Accent.R, G: p.Accent.G, B: p.Accent.B, A: p.Accent.A},
		ReduceMotion:       p.ReduceMotion,
		ReduceTransparency: p.ReduceTransparency,
		HighContrast:       p.HighContrast,
		TextScale:          p.TextScale,
	}
	if prefs.TextScale <= 0 {
		prefs.TextScale = 1 // the usual size, as package ui normalizes it
	}
	return prefs
}

// themePreferencesJSON is ThemePreferences as pages see it, mygo-runtime's
// ThemePreferences: the accent as a CSS color, null where there is none.
type themePreferencesJSON struct {
	Accent             *string `json:"accent"`
	ReduceMotion       bool    `json:"reduceMotion"`
	ReduceTransparency bool    `json:"reduceTransparency"`
	HighContrast       bool    `json:"highContrast"`
	TextScale          float64 `json:"textScale"`
}

// MarshalJSON encodes the preferences as pages get them, mygo-runtime's
// ThemePreferences, with the accent as "#rrggbb" ("#rrggbbaa" when it is
// not opaque), or null where the desktop has none.
func (p ThemePreferences) MarshalJSON() ([]byte, error) {
	v := themePreferencesJSON{
		ReduceMotion:       p.ReduceMotion,
		ReduceTransparency: p.ReduceTransparency,
		HighContrast:       p.HighContrast,
		TextScale:          p.TextScale,
	}
	if c := p.Accent; c.A != 0 {
		s := fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
		if c.A != 255 {
			s += fmt.Sprintf("%02x", c.A)
		}
		v.Accent = &s
	}
	return json.Marshal(v)
}

// UnmarshalJSON decodes what MarshalJSON encodes.
func (p *ThemePreferences) UnmarshalJSON(b []byte) error {
	var v themePreferencesJSON
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*p = ThemePreferences{
		ReduceMotion:       v.ReduceMotion,
		ReduceTransparency: v.ReduceTransparency,
		HighContrast:       v.HighContrast,
		TextScale:          v.TextScale,
	}
	if v.Accent != nil {
		c, err := parseColor(*v.Accent)
		if err != nil {
			return err
		}
		p.Accent = color.NRGBA{R: c.R, G: c.G, B: c.B, A: c.A}
	}
	return nil
}

// Source returns the appearance override.
func (t *ThemeModule) Source() ThemeSource {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.source == "" {
		return ThemeSystem
	}
	return t.source
}

// SetSource forces a light or dark appearance, or follows the system. It
// may be called before App.Run, e.g. with a saved preference, to start the
// app in that appearance.
func (t *ThemeModule) SetSource(s ThemeSource) {
	if s == "" {
		s = ThemeSystem
	}
	t.mu.Lock()
	t.source = s
	t.mu.Unlock()
	onMain(func() {
		if !App.initialized {
			return // Run applies it
		}
		t.apply()
		// Not every backend reports its own change.
		updateBackgrounds()
		contentThemeChanged()
	})
}

// apply gives the backend the source set last, if any: calls made on
// several goroutines may reach the main thread in another order. Main
// thread only.
func (t *ThemeModule) apply() {
	t.mu.Lock()
	s := t.source
	t.mu.Unlock()
	if s != "" {
		backend().Theme().SetSource(string(s))
	}
}

// OnUpdated is called when the effective appearance, or the desktop's
// preferences (Preferences), change.
func (t *ThemeModule) OnUpdated(fn func()) (off func()) { return t.onUpdated.add(fn, false) }

// changed forgets the preferences read before and tells the listeners that
// the system appearance changed. Main thread only.
func (t *ThemeModule) changed() {
	t.prefs, t.prefsOK = ThemePreferences{}, false
	fire(&t.onUpdated)
}

// GlobalShortcutModule registers system wide keyboard shortcuts.
type GlobalShortcutModule struct {
	mu      sync.Mutex
	next    int
	byKey   map[string]int
	byID    map[int]func()
	keyByID map[int]string
}

// GlobalShortcut registers keyboard shortcuts that work while the app is in
// the background.
var GlobalShortcut = &GlobalShortcutModule{}

// Register calls fn on the main thread whenever accelerator (e.g.
// "CmdOrCtrl+Shift+Space") is pressed, even when the app is not focused.
//
// On Wayland the desktop binds the shortcut, through the XDG desktop
// portal, shortly after Register returns: it may ask the user to confirm
// it, lets them pick other keys, and needs the app installed with its
// desktop entry. The window fn shows or focuses first gets the activation
// token of the key press, which lets it take the focus.
//
// Register claims the accelerator before the desktop binds it, so that
// calls arriving at once cannot bind it twice: IsRegistered reports it
// while the desktop binds it, and a second Register of the same
// accelerator fails.
func (g *GlobalShortcutModule) Register(acc string, fn func()) error {
	needsApp("GlobalShortcut.Register")
	a, err := accelerator.Parse(acc, runtime.GOOS)
	if err != nil {
		return err
	}
	key := a.String()
	g.mu.Lock()
	if g.byKey == nil {
		g.byKey, g.byID, g.keyByID = map[string]int{}, map[int]func(){}, map[int]string{}
	}
	if _, ok := g.byKey[key]; ok {
		g.mu.Unlock()
		return errors.New("mygo: shortcut " + key + " is already registered")
	}
	g.next++
	id := g.next
	g.byKey[key] = id
	g.mu.Unlock()
	if err := onMainValue(func() error { return backend().RegisterHotkey(id, acc) }); err != nil {
		g.mu.Lock()
		if g.byKey[key] == id {
			delete(g.byKey, key)
		}
		g.mu.Unlock()
		return err
	}
	g.mu.Lock()
	if g.byKey[key] != id {
		// Unregistered, or registered again, while the desktop bound it:
		// give the desktop back the shortcut this call made.
		g.mu.Unlock()
		onMain(func() { backend().UnregisterHotkey(id) })
		return errors.New("mygo: shortcut " + key + " was unregistered while it was being registered")
	}
	g.byID[id], g.keyByID[id] = fn, key
	g.mu.Unlock()
	return nil
}

// IsRegistered reports whether accelerator is registered by this app.
func (g *GlobalShortcutModule) IsRegistered(acc string) bool {
	a, err := accelerator.Parse(acc, runtime.GOOS)
	if err != nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	_, ok := g.byKey[a.String()]
	return ok
}

// Unregister removes a shortcut.
func (g *GlobalShortcutModule) Unregister(acc string) {
	a, err := accelerator.Parse(acc, runtime.GOOS)
	if err != nil {
		return
	}
	g.mu.Lock()
	id, ok := g.byKey[a.String()]
	if ok {
		delete(g.byKey, a.String())
		delete(g.byID, id)
		delete(g.keyByID, id)
	}
	g.mu.Unlock()
	if ok {
		onMain(func() { backend().UnregisterHotkey(id) })
	}
}

// UnregisterAll removes all shortcuts registered by this app.
func (g *GlobalShortcutModule) UnregisterAll() {
	g.mu.Lock()
	var ids []int
	for id := range g.byID {
		ids = append(ids, id)
	}
	g.byKey, g.byID, g.keyByID = map[string]int{}, map[int]func(){}, map[int]string{}
	g.mu.Unlock()
	onMain(func() {
		for _, id := range ids {
			backend().UnregisterHotkey(id)
		}
	})
}

func (g *GlobalShortcutModule) pressed(id int) {
	g.mu.Lock()
	fn := g.byID[id]
	g.mu.Unlock()
	if fn != nil {
		fn()
	}
}

// TrayOptions configures NewTray.
type TrayOptions struct {
	// Icon is a PNG image, ideally 16x16 points (32x32 pixels for Retina).
	Icon []byte
	// IconIsTemplate lets macOS tint a black-and-transparent icon to match
	// the menu bar.
	IconIsTemplate bool
	// Title is shown next to the icon (macOS).
	Title   string
	ToolTip string
	// Menu is shown when the icon is clicked.
	Menu *Menu
}

// Tray is an icon in the menu bar (macOS) or notification area.
type Tray struct {
	native  platform.Tray
	menu    *Menu
	onClick listeners[func()]
	onRight listeners[func()]
}

type trayHandler struct{ t *Tray }

func (h trayHandler) Clicked()      { fire(&h.t.onClick) }
func (h trayHandler) RightClicked() { fire(&h.t.onRight) }

// NewTray adds an icon to the menu bar or notification area. It must be
// called after the application is ready.
func NewTray(opts TrayOptions) (*Tray, error) {
	if !isMainThread() {
		App.waitReady()
	} else if !App.IsReady() {
		panic("mygo: NewTray called before the application is ready; create trays in App.WhenReady")
	}
	t := &Tray{menu: opts.Menu}
	var err error
	onMain(func() {
		t.native, err = backend().NewTray(trayHandler{t})
		if err != nil {
			return
		}
		if len(opts.Icon) > 0 {
			if err = t.native.SetImage(opts.Icon, opts.IconIsTemplate); err != nil {
				return
			}
		}
		if opts.Title != "" {
			t.native.SetTitle(opts.Title)
		}
		if opts.ToolTip != "" {
			t.native.SetToolTip(opts.ToolTip)
		}
		if opts.Menu != nil {
			t.native.SetMenu(opts.Menu.snapshot())
		}
	})
	if err != nil {
		return nil, err
	}
	if t.native == nil {
		return nil, errLoopStopped
	}
	return t, nil
}

func (t *Tray) do(fn func(n platform.Tray)) {
	onMain(func() {
		if t.native != nil {
			fn(t.native)
		}
	})
}

// SetIcon replaces the icon with a PNG image.
func (t *Tray) SetIcon(png []byte, template bool) error {
	var err error
	t.do(func(n platform.Tray) { err = n.SetImage(png, template) })
	return err
}

// SetTitle sets the text next to the icon (macOS).
func (t *Tray) SetTitle(title string) { t.do(func(n platform.Tray) { n.SetTitle(title) }) }

// SetToolTip sets the hover text.
func (t *Tray) SetToolTip(tip string) { t.do(func(n platform.Tray) { n.SetToolTip(tip) }) }

// SetMenu sets the menu shown when the icon is clicked; nil removes it so
// clicks reach OnClick.
func (t *Tray) SetMenu(m *Menu) {
	t.menu = m
	snap := m.snapshot()
	t.do(func(n platform.Tray) { n.SetMenu(snap) })
}

// PopUpMenu shows the tray menu programmatically.
func (t *Tray) PopUpMenu() {
	snap := t.menu.snapshot()
	t.do(func(n platform.Tray) { n.PopUpMenu(snap) })
}

// Bounds returns the position of the icon on screen.
func (t *Tray) Bounds() Rectangle {
	return onMainValue(func() Rectangle {
		if t.native == nil {
			return Rectangle{}
		}
		return Rectangle(t.native.Bounds())
	})
}

// Destroy removes the icon.
func (t *Tray) Destroy() {
	onMain(func() {
		if t.native != nil {
			t.native.Destroy()
			t.native = nil
		}
	})
}

// OnClick is called when the icon is clicked and it has no menu.
func (t *Tray) OnClick(fn func()) (off func()) { return t.onClick.add(fn, false) }

// OnRightClick is called when the icon is right-clicked.
func (t *Tray) OnRightClick(fn func()) (off func()) { return t.onRight.add(fn, false) }

// NotificationOptions configures a desktop notification.
type NotificationOptions struct {
	// ID identifies the notification: App.OnNotificationClick receives
	// it, also after the app has quit and been launched again, and showing
	// a notification with the ID of one still shown replaces that one.
	// Empty gives it an ID of its own.
	ID       string
	Title    string
	Subtitle string
	Body     string
	// Silent suppresses the notification sound.
	Silent bool
	// Group gathers the notifications that share it in one stack of
	// Notification Center (macOS), such as the messages of a conversation.
	// Empty leaves them in the app's.
	Group string
}

// Notification is a desktop notification.
type Notification struct {
	id      string
	opts    NotificationOptions
	onClick listeners[func()]
}

var notifications struct {
	sync.Mutex
	byID map[string]*Notification
}

// NotificationsSupported reports whether desktop notifications can be
// shown. On macOS they require a packaged app, which `mygo dev` and
// `mygo build` produce.
func NotificationsSupported() bool {
	needsApp("NotificationsSupported")
	return onMainValue(func() bool { return backend().NotificationsSupported() })
}

// WebViewAvailable reports whether the system webview is there, and so
// whether a window showing a web page can work: WKWebView on macOS,
// WebKitGTK on Linux, the WebView2 Runtime on Windows. A window showing
// native UI (WindowOptions.Content) needs none, so an app whose windows all
// do runs without one, and this reports false in browsers, which show
// native UI only.
//
// Ask before creating the first window showing a page, as NewWindow panics
// when the platform has no webview. It may be called from any goroutine,
// also in main before App.Run:
//
//	mygo.App.WhenReady(func() {
//		if mygo.WebViewAvailable() {
//			mygo.NewWindow(mygo.WindowOptions{URL: "/"})
//		} else {
//			mygo.NewWindow(mygo.WindowOptions{Content: ui.View(offlineView)})
//		}
//	})
//
// It looks for the runtime without starting it: on Windows for the
// WebView2 Runtime or a WebView2Loader.dll next to the executable, on
// Linux by loading WebKitGTK. A runtime that is there but broken still
// reports true; the window then reports calls that need its page as
// failed. Once a window showing a page started the runtime, it reports
// whether that worked.
func WebViewAvailable() bool { return backend().WebViewAvailable() }

// NewNotification creates a notification; call Show to display it.
func NewNotification(opts NotificationOptions) *Notification {
	id := opts.ID
	if id == "" {
		// Unique across runs too, as notifications outlive them.
		id = "mygo-" + randomText()
	}
	return &Notification{id: id, opts: opts}
}

// ID returns the notification's ID: NotificationOptions.ID, or the one it
// was given.
func (n *Notification) ID() string { return n.id }

// ErrNotificationsDenied is returned by Notification.Show when the user has
// not allowed the app to show notifications (macOS).
var ErrNotificationsDenied = platform.ErrNotificationsDenied

// Show displays the notification, and returns once the system has it. On
// macOS the first notification asks the user whether to allow them, and
// Show waits for the answer; it returns ErrNotificationsDenied when they
// are not allowed.
func (n *Notification) Show() error {
	needsApp("Notification.Show")
	notifications.Lock()
	if notifications.byID == nil {
		notifications.byID = map[string]*Notification{}
	}
	_, shown := notifications.byID[n.id]
	notifications.byID[n.id] = n
	notifications.Unlock()
	ch := make(chan error, 1)
	onMain(func() {
		backend().ShowNotification(&platform.Notification{
			ID:       n.id,
			Title:    n.opts.Title,
			Subtitle: n.opts.Subtitle,
			Body:     n.opts.Body,
			Silent:   n.opts.Silent,
			Group:    n.opts.Group,
		}, func(err error) { deliver(ch, err) })
	})
	err := await(ch)
	if err != nil && !shown {
		// Nothing will be clicked.
		notifications.Lock()
		delete(notifications.byID, n.id)
		notifications.Unlock()
	}
	return err
}

// Close removes the notification.
func (n *Notification) Close() {
	notifications.Lock()
	delete(notifications.byID, n.id)
	notifications.Unlock()
	onMain(func() { backend().RemoveNotification(n.id) })
}

// ClearNotifications removes the app's notifications, on macOS those of
// earlier runs left in Notification Center too. An app calls it as it
// comes to the front, so that what the user has seen in the app does not
// wait there:
//
//	mygo.App.OnDidBecomeActive(func() { mygo.ClearNotifications() })
func ClearNotifications() {
	notifications.Lock()
	clear(notifications.byID)
	notifications.Unlock()
	onMain(func() { backend().RemoveAllNotifications() })
}

// OnClick is called when the user clicks the notification.
func (n *Notification) OnClick(fn func()) (off func()) { return n.onClick.add(fn, false) }

func notificationClicked(id string) {
	notifications.Lock()
	n := notifications.byID[id]
	notifications.Unlock()
	if n != nil {
		fire(&n.onClick)
	}
	fire1(&App.onNotificationClick, id)
}
