package mygo

// Vibrancy is a translucent material that blurs what is behind a window,
// shown behind a transparent page (see WindowOptions.Vibrancy).
//
// macOS has all of them (NSVisualEffectView materials). Windows 11 has the
// Mica, Acrylic and Tabbed system backdrops; the other materials use the
// closest one: Acrylic for transient UI (menus, popovers, HUDs, sheets,
// tooltips, selection, full-screen UI), Mica otherwise. On macOS, Mica and
// Tabbed use the under-window material and Acrylic the HUD one. Linux has
// none.
type Vibrancy string

// Materials for window areas.
const (
	VibrancyNone        Vibrancy = ""
	VibrancyWindow      Vibrancy = "window"
	VibrancyContent     Vibrancy = "content"
	VibrancySidebar     Vibrancy = "sidebar"
	VibrancyHeader      Vibrancy = "header"
	VibrancyTitlebar    Vibrancy = "titlebar"
	VibrancyUnderWindow Vibrancy = "under-window"
	VibrancyUnderPage   Vibrancy = "under-page"
)

// Materials for transient UI.
const (
	VibrancyMenu         Vibrancy = "menu"
	VibrancyPopover      Vibrancy = "popover"
	VibrancyHUD          Vibrancy = "hud"
	VibrancySheet        Vibrancy = "sheet"
	VibrancyTooltip      Vibrancy = "tooltip"
	VibrancySelection    Vibrancy = "selection"
	VibrancyFullScreenUI Vibrancy = "fullscreen-ui"
)

// Windows 11 system backdrops.
const (
	// VibrancyMica tints the window with the desktop wallpaper.
	VibrancyMica Vibrancy = "mica"
	// VibrancyAcrylic blurs what is behind the window.
	VibrancyAcrylic Vibrancy = "acrylic"
	// VibrancyTabbed is Mica for windows with tabs in the title bar.
	VibrancyTabbed Vibrancy = "tabbed"
)
