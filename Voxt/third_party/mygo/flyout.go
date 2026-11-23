package mygo

import (
	"fmt"

	"github.com/egoist/mygo/internal/platform"
)

// Placement is where a flyout goes from its anchor: the side, and how it
// lines up with the anchor along that side, as in "bottom-start", below
// the anchor with their left edges lined up.
type Placement string

// Placements.
const (
	PlacementBottomStart Placement = "bottom-start"
	PlacementBottom      Placement = "bottom"
	PlacementBottomEnd   Placement = "bottom-end"
	PlacementTopStart    Placement = "top-start"
	PlacementTop         Placement = "top"
	PlacementTopEnd      Placement = "top-end"
	PlacementRightStart  Placement = "right-start"
	PlacementRight       Placement = "right"
	PlacementRightEnd    Placement = "right-end"
	PlacementLeftStart   Placement = "left-start"
	PlacementLeft        Placement = "left"
	PlacementLeftEnd     Placement = "left-end"
)

// placements are the sides and alignments of the placements.
var placements = map[Placement]struct {
	side  platform.Side
	align platform.Align
}{
	"":                   {platform.SideBottom, platform.AlignStart},
	PlacementBottomStart: {platform.SideBottom, platform.AlignStart},
	PlacementBottom:      {platform.SideBottom, platform.AlignCenter},
	PlacementBottomEnd:   {platform.SideBottom, platform.AlignEnd},
	PlacementTopStart:    {platform.SideTop, platform.AlignStart},
	PlacementTop:         {platform.SideTop, platform.AlignCenter},
	PlacementTopEnd:      {platform.SideTop, platform.AlignEnd},
	PlacementRightStart:  {platform.SideRight, platform.AlignStart},
	PlacementRight:       {platform.SideRight, platform.AlignCenter},
	PlacementRightEnd:    {platform.SideRight, platform.AlignEnd},
	PlacementLeftStart:   {platform.SideLeft, platform.AlignStart},
	PlacementLeft:        {platform.SideLeft, platform.AlignCenter},
	PlacementLeftEnd:     {platform.SideLeft, platform.AlignEnd},
}

// FlyoutOptions configures NewFlyout.
type FlyoutOptions struct {
	// Parent owns the flyout, which stays above it, follows it as it moves
	// and closes with it. A flyout has a Parent or a Tray.
	Parent *Window
	// Tray anchors the flyout to a tray icon instead, as the panel of a
	// menu bar app: it goes next to the icon, on the side Placement
	// names (PlacementBottom below an icon of the macOS menu bar, which
	// flips above one of a taskbar at the bottom of the screen), and
	// floats above the windows of other apps. Anchor does not apply.
	// Linux's tray icons tell neither clicks nor where they are: a menu
	// item of the icon opens it there, in the middle of the primary
	// display's work area.
	Tray *Tray
	// Anchor is the rectangle the flyout is placed against, in DIPs
	// relative to the parent's content area: the box of the element
	// that opens it, as ui.Element.Bounds gives it, or a point (a
	// rectangle of zero size), such as the pointer's.
	Anchor Rectangle
	// Placement is where the flyout goes from Anchor (default
	// PlacementBottomStart). Where the display's work area has no room
	// for it there, the flyout flips to the other side of the anchor, or
	// to its other end, if that has room, then slides back into the work
	// area, then shrinks to fit it.
	Placement Placement
	// Gap is the distance between Anchor and the flyout, in DIPs.
	Gap int
	// Width and Height of the flyout in DIPs (default 200x200).
	Width, Height int
	// Focusable makes the flyout take the keyboard as it shows, and close
	// as a menu does when the user presses outside it, another window
	// takes the keyboard or the app is deactivated: OnClose listeners
	// hear it, and may keep it open, except on Wayland, whose compositor
	// dismisses it. On Linux and Windows the press only closes it, as it
	// closes their menus; on macOS it goes on to what is under the
	// pointer, but on Anchor, as AppKit's popovers: so a click on the
	// button that opened it closes it, and does not open it again.
	// Menus, lists to choose from and popovers are focusable, and handle
	// their own keys.
	//
	// Without it, the flyout never takes the keyboard, which stays in
	// the parent, and stays until it is closed: tooltips, hover cards,
	// or the suggestions of a text field in the parent, which closes
	// them as it sees fit, as when it loses the keyboard.
	//
	// Either closes, as the user closing it, on an Escape its content
	// does not handle: no element or shortcut of native UI takes it, no
	// handler of the page calls preventDefault.
	Focusable bool
	// Hidden creates the flyout without showing it.
	Hidden bool
	// Shadow gives the flyout the system's window shadow (macOS,
	// Windows). On macOS it follows the shape of what the content draws,
	// and outlines it with a hairline: the content draws no border there.
	// On Windows 11 the system rounds the flyout's corners as it rounds
	// its menus, and outlines it: the content draws neither a border nor
	// rounded corners there.
	Shadow bool
	// Popover gives the flyout the look of the system's popovers on macOS:
	// it shows in an NSPopover, which draws its material, rounded corners,
	// shadow and an arrow pointing at the middle of Anchor, and animates
	// as it opens and closes. AppKit places it on the side Placement
	// names, or the opposite one where the screen has no room: the rest
	// of Placement, and Gap, do not apply. As AppKit's popovers do, it has
	// the keyboard while it shows, Focusable or not, which makes it close
	// as the user clicks elsewhere. Its content draws no background there
	// (ui.Context.Vibrancy reports the material): elsewhere, where Popover
	// changes nothing, it draws its own.
	Popover bool

	// URL is loaded right after the flyout is created, as
	// WindowOptions.URL.
	URL string
	// Page configures the flyout's web page.
	Page PageOptions
	// Content makes the flyout show native UI instead of a web page, as
	// WindowOptions.Content.
	Content Content
}

// NewFlyout creates a flyout: a window without a frame, a background, a
// shadow or a taskbar button, owned by its parent and placed next to a
// rectangle of it, which may extend beyond the parent but stays in the
// work area of a display. Its page or native UI draws all of it: give the
// page a background, or the native UI's root one. Like NewWindow, it must
// be called after the application is ready.
func NewFlyout(opts FlyoutOptions) *Window {
	if (opts.Parent == nil) == (opts.Tray == nil) {
		panic("mygo: NewFlyout needs a Parent or a Tray")
	}
	fo, err := opts.platform()
	if err != nil {
		panic(err)
	}
	return createWindow("NewFlyout", WindowOptions{
		URL:               opts.URL,
		Width:             or(opts.Width, 200),
		Height:            or(opts.Height, 200),
		UseContentSize:    true,
		Frameless:         true,
		Transparent:       true,
		DisableResize:     true,
		DisableMinimize:   true,
		DisableMaximize:   true,
		DisableFullScreen: true,
		DisableShadow:     !opts.Shadow,
		SkipTaskbar:       true,
		Parent:            opts.Parent,
		Page:              opts.Page,
		Content:           opts.Content,
		Hidden:            opts.Hidden,
	}, nil, &flyoutOptions{fo, opts.Tray})
}

// flyoutOptions carries what newWindow needs of a flyout: its tray's
// native icon is read on the main thread.
type flyoutOptions struct {
	flyout *platform.Flyout
	tray   *Tray
}

func (o *FlyoutOptions) platform() (*platform.Flyout, error) {
	p, ok := placements[o.Placement]
	if !ok {
		return nil, fmt.Errorf("mygo: unknown flyout placement %q", o.Placement)
	}
	return &platform.Flyout{
		Anchor:    platform.Rect(o.Anchor),
		Side:      p.side,
		Align:     p.align,
		Gap:       o.Gap,
		Focusable: o.Focusable,
		Popover:   o.Popover,
	}, nil
}

// IsFlyout reports whether NewFlyout created the window.
func (w *Window) IsFlyout() bool { return w.flyout != nil }

// SetAnchor places a flyout against another rectangle of its parent, in
// DIPs relative to the parent's content area, as FlyoutOptions.Anchor. It
// does nothing for other windows.
func (w *Window) SetAnchor(anchor Rectangle) {
	onMain(func() {
		if w.flyout == nil || w.native == nil {
			return
		}
		w.flyout.Anchor = platform.Rect(anchor)
		w.placeFlyout()
	})
}

// placeFlyout places a flyout where it goes now. Main thread only.
func (w *Window) placeFlyout() {
	if w.flyout != nil && w.native != nil {
		w.native.PlaceFlyout(*w.flyout, w.flyoutSize)
	}
}

// placeFlyouts places the flyouts of a window that moved or resized
// again, which keeps them next to their anchors and in the work area.
// Main thread only.
func (w *Window) placeFlyouts() {
	for _, x := range Windows() {
		if x.parent == w && x.flyout != nil {
			x.placeFlyout()
		}
	}
}

// setFlyoutSize resizes a flyout, which SetSize and SetContentSize do
// alike, and places it again: it reports false for other windows. Main
// thread only.
func (w *Window) setFlyoutSize(width, height int) bool {
	if w.flyout == nil {
		return false
	}
	w.flyoutSize = platform.Size{Width: max(width, 1), Height: max(height, 1)}
	w.placeFlyout()
	return true
}

// escape closes a flyout or a panel on an Escape its page or native UI
// did not handle, as the user closing it: menus, popovers and launchers
// close so. Its content may be building a frame: the window closes after
// it. Main thread only.
func (w *Window) escape() {
	if w.flyout != nil || w.panel {
		postMain(func() { w.close() })
	}
}
