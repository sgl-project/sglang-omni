//go:build darwin

package darwin

import (
	"bytes"
	"image"
	pngenc "image/png"
	"runtime"
	"structs"
	"sync"
	"time"
	"unicode/utf16"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// The surface of a window that shows content MyGo draws itself is a
// layer-backed view in place of the web view, inside the same content view,
// so that vibrancy, the traffic lights and a hidden title bar behave as
// with a page. It turns AppKit's events into surface events, is the text
// input client of input methods, and paces frames with a display link.

// nsRange is NSRange, for the text input methods.
type nsRange struct {
	_                structs.HostLayout
	Location, Length uint
}

// cgRect is an NSRect returned by a method implemented in Go.
type cgRect struct {
	_          structs.HostLayout
	X, Y, W, H float64
}

const nsNotFound = uint(1<<63 - 1)

var (
	surfaceOnce        sync.Once
	msgConvertRectView func(obj id, sel objc.SEL, r NSRect, view id) NSRect
	msgInitTracking    func(obj id, sel objc.SEL, r NSRect, options uint, owner, info id) id
	msgTimer           func(cls id, sel objc.SEL, interval float64, target id, selector objc.SEL, info id, repeats bool) id
	cgColorSpaceSRGB   uintptr
	cgImageCreate      uintptr
	cgImageRelease     uintptr
	cgProviderCreate   uintptr
	cgProviderRelease  uintptr
	cfDataCreate       uintptr
	// Read the keyboard type from each event, so an external ISO keyboard
	// does not inherit the built-in keyboard's ANSI mapping.
	cgEventGetIntegerValueField func(event uintptr, field uint32) int64
	kbGetLayoutType             func(keyboardType int16) uint32
	lmGetKbdType                func() uint8

	// caPass counts the passes of the main run loop, past Core Animation's
	// commit of each: frames drawn in the same pass go into the same
	// transaction.
	caPass uint64 = 1
)

const (
	cfRunLoopBeforeWaiting = 1 << 5
	cfRunLoopExit          = 1 << 7
	// caCommitOrder is the order of Core Animation's observer committing
	// the transaction of each pass of the run loop.
	caCommitOrder = 2000000
)

func loadSurface() {
	surfaceOnce.Do(func() {
		stret := msgSendAddr
		if runtime.GOARCH == "amd64" {
			stret = mustDlsym(libObjC, "objc_msgSend_stret")
		}
		purego.RegisterFunc(&msgConvertRectView, stret)
		purego.RegisterFunc(&msgInitTracking, msgSendAddr)
		purego.RegisterFunc(&msgTimer, msgSendAddr)
		cgImageCreate = mustDlsym(libCG, "CGImageCreate")
		cgImageRelease = mustDlsym(libCG, "CGImageRelease")
		cgProviderCreate = mustDlsym(libCG, "CGDataProviderCreateWithCFData")
		cgProviderRelease = mustDlsym(libCG, "CGDataProviderRelease")
		cfDataCreate = mustDlsym(libCF, "CFDataCreate")
		purego.RegisterLibFunc(&cgEventGetIntegerValueField, libCG, "CGEventGetIntegerValueField")
		purego.RegisterLibFunc(&kbGetLayoutType, libCarbon, "KBGetLayoutType")
		purego.RegisterLibFunc(&lmGetKbdType, libCarbon, "LMGetKbdType")
		p := mustDlsym(libCG, "kCGColorSpaceSRGB")
		name := **(**uintptr)(unsafe.Pointer(&p))
		cgColorSpaceSRGB, _, _ = purego.SyscallN(mustDlsym(libCG, "CGColorSpaceCreateWithName"), name)
		// An observer after Core Animation's counts the transactions it
		// committed.
		var observerCreate func(alloc uintptr, activities uint, repeats bool, order int, callout, ctx uintptr) uintptr
		var addObserver func(rl, observer, mode uintptr)
		purego.RegisterLibFunc(&observerCreate, libCF, "CFRunLoopObserverCreate")
		purego.RegisterLibFunc(&addObserver, libCF, "CFRunLoopAddObserver")
		passed := purego.NewCallback(func(observer, activity, info uintptr) { caPass++ })
		if o := observerCreate(0, cfRunLoopBeforeWaiting|cfRunLoopExit, true, caCommitOrder+1, passed, 0); o != 0 {
			addObserver(cfRunLoopGetMain(), o, kCFRunLoopCommonModes)
		}
	})
}

type surface struct {
	w        *window
	view     id
	tracking id
	link     id // CADisplayLink (macOS 14), nil before
	// The display link runs while frames follow each other (linkRunning),
	// and due is whether its next tick draws one. Without a display link,
	// a timer paces frames at the display's rate: timing is true while
	// one is due. lastFrame is when the last frame began, and framePass
	// the pass of the run loop that drew it (caPass); resized tells that
	// the view changed size or scale since.
	linkRunning bool
	due         bool
	timing      bool
	resized     bool
	lastFrame   time.Time
	framePass   uint64
	cursor      platform.Cursor
	inside      bool
	ctrlClick   bool // the primary button is down for a Control-click

	// input is the state of the text input with the keyboard; marked is
	// the input method's composition and markedSel its selection in it.
	input     platform.TextInputState
	marked    string
	markedSel nsRange
	keyDown   bool // in interpretKeyEvents: text typed with the key

	// Accessibility: whether the content describes its frames, the last
	// tree, the elements of its nodes and those at the top (an NSArray).
	accessOn      bool
	access        platform.AccessTree
	elements      map[uint64]*accessElement
	topLevel      id
	updating      bool
	dataDrag      *macDataSource
	dragOperation transfer.Operation
	// placed are where the content shows the window's web views, under
	// it (webview.go).
	placed []platform.WebViewPlacement
}

func (w *window) createSurface(content NSRect) {
	loadSurface()
	s := &surface{w: w}
	s.view = msgInitRect(send(class("MyGoSurfaceView"), "alloc"), sel("initWithFrame:"), NSRect{Size: content.Size})
	send(s.view, "setWantsLayer:", 1)
	send(s.view, "setLayerContentsRedrawPolicy:", 2) // on setNeedsDisplay
	send(s.view, "setAutoresizingMask:", nsViewWidthHeightSizable)
	layer := send(s.view, "layer")
	send(layer, "setContentsGravity:", uintptr(nsString("resize")))
	// Tracking for hover, inside the visible part of the view.
	options := uint(0x01 | 0x02 | 0x04 | 0x40 | 0x200) // entered/exited, moved, cursor update, active app, visible rect
	if w.opts.Flyout == nil && !w.opts.Focusable {
		// A non-activating window (WindowOptions.NonActivating) works
		// while another app is active: hover it all the same.
		options = options&^0x40 | 0x80 // NSTrackingActiveAlways
	}
	s.tracking = msgInitTracking(send(class("NSTrackingArea"), "alloc"), sel("initWithRect:options:owner:userInfo:"), NSRect{}, options, s.view, 0)
	send(s.view, "addTrackingArea:", uintptr(s.tracking))
	if respondsTo(s.view, "displayLinkWithTarget:selector:") {
		s.link = retain(send(s.view, "displayLinkWithTarget:selector:", uintptr(s.view), uintptr(sel("mygoTick:"))))
		send(s.link, "setPaused:", 1)
		send(s.link, "addToRunLoop:forMode:", uintptr(send(class("NSRunLoop"), "mainRunLoop")), kCFRunLoopCommonModes)
	}
	// Files dragged from other apps.
	send(s.view, "registerForDraggedTypes:", uintptr(nsArray(nsString("public.file-url"))))
	w.surface = s
	w.b.bySurface[s.view] = s
}

func (s *surface) destroy() {
	s.CancelDataDrag()
	if s.link != 0 {
		send(s.link, "invalidate")
		release(s.link)
		s.link = 0
	}
	s.destroyAccess()
	send(s.view, "removeTrackingArea:", uintptr(s.tracking))
	release(s.tracking)
	delete(s.w.b.bySurface, s.view)
	release(s.view)
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
	return platform.SurfaceNative{View: uintptr(s.view), Layer: uintptr(send(s.view, "layer"))}
}

// ShowsMaterial reports whether the window has a material, whose effect
// view, behind the surface, shows where its frames are transparent.
func (s *surface) ShowsMaterial() bool { return s.w.effect != 0 || s.w.popover != 0 }

func (s *surface) scale() float64 {
	if f := msgFloat(s.w.win, sel("backingScaleFactor")); f > 0 {
		return f
	}
	return 1
}

func (s *surface) Size() (float64, float64, float64) {
	b := msgRect(s.view, sel("bounds"))
	return b.Size.Width, b.Size.Height, s.scale()
}

func (s *surface) RequestFrame() {
	if s.w.closed {
		return
	}
	if s.link != 0 {
		switch {
		case s.linkRunning:
			s.due = true
		case time.Since(s.lastFrame) > 2*s.refresh() && s.visible():
			// After a pause, the frame draws at once, at the end of this
			// pass of the run loop, as AppKit's views do. A window out of
			// sight waits for the link, which ticks once it shows: AppKit
			// draws the views of hidden windows too.
			send(s.view, "setNeedsDisplay:", 1)
		default:
			// Frames that follow each other draw at the display's
			// refreshes; the link keeps running while they come, rather
			// than pausing after each, which costs a call to the window
			// server.
			s.due, s.linkRunning = true, true
			send(s.link, "setPaused:", 0)
		}
		return
	}
	if s.timing {
		return
	}
	// One refresh after the last frame, or at once after a pause.
	delay := s.refresh() - time.Since(s.lastFrame)
	if delay <= 0 {
		send(s.view, "setNeedsDisplay:", 1)
		return
	}
	s.timing = true
	withPool(func() {
		t := msgTimer(class("NSTimer"), sel("timerWithTimeInterval:target:selector:userInfo:repeats:"), delay.Seconds(), s.view, sel("mygoTimer:"), 0, false)
		send(send(class("NSRunLoop"), "mainRunLoop"), "addTimer:forMode:", uintptr(t), kCFRunLoopCommonModes)
	})
}

// presentsWithTransaction reports whether the GPU renderer's CAMetalLayer
// presents with Core Animation's transaction (internal/gpu/metal), as it
// does the frames that change it: the frame drawn last then takes a
// drawable until the transaction commits.
func (s *surface) presentsWithTransaction() bool {
	sublayers := send(send(s.view, "layer"), "sublayers")
	metalLayer := uintptr(class("CAMetalLayer"))
	for i := range sendInt(sublayers, "count") {
		if l := send(sublayers, "objectAtIndex:", uintptr(i)); sendBool(l, "isKindOfClass:", metalLayer) {
			return sendBool(l, "presentsWithTransaction")
		}
	}
	return false
}

// Occluded reports whether nothing of the window shows on screen: hidden,
// minimized, on another space or covered by other windows, which still
// get display link ticks.
func (s *surface) Occluded() bool { return !s.visible() }

// visible reports whether some of the window shows on screen.
func (s *surface) visible() bool {
	const occlusionStateVisible = 1 << 1
	return sendInt(s.w.win, "occlusionState")&occlusionStateVisible != 0
}

// refresh returns the time between two refreshes of the window's display.
func (s *surface) refresh() time.Duration {
	fps := 60
	if r := int(s.RefreshRate()); r > 0 {
		fps = max(r, 30)
	}
	return time.Second / time.Duration(fps)
}

// RefreshRate returns the most frames a second the window's screen shows,
// as ProMotion displays vary it.
func (s *surface) RefreshRate() float64 {
	if screen := send(s.w.win, "screen"); screen != 0 && respondsTo(screen, "maximumFramesPerSecond") {
		return float64(sendInt(screen, "maximumFramesPerSecond"))
	}
	return 0
}

// WideGamut reports whether the window's screen shows Display P3's colors,
// outside the sRGB gamut, as the screens of recent Macs do.
func (s *surface) WideGamut() bool {
	const displayGamutP3 = 2 // NSDisplayGamutP3
	screen := send(s.w.win, "screen")
	return screen != 0 && respondsTo(screen, "canRepresentDisplayGamut:") && byte(send(screen, "canRepresentDisplayGamut:", displayGamutP3)) != 0
}

func (s *surface) PresentPixels(pix []byte, stride, width, height int) {
	if len(pix) < stride*height || width == 0 || height == 0 {
		return
	}
	data, _, _ := purego.SyscallN(cfDataCreate, 0, uintptr(unsafe.Pointer(&pix[0])), uintptr(stride*height))
	provider, _, _ := purego.SyscallN(cgProviderCreate, data)
	const bitmapInfo = 2 | 2<<12 // premultiplied alpha first, 32-bit little endian: BGRA
	img, _, _ := purego.SyscallN(cgImageCreate, uintptr(width), uintptr(height), 8, 32, uintptr(stride), cgColorSpaceSRGB, bitmapInfo, provider, 0, 0, 0)
	layer := send(s.view, "layer")
	msgSetFloat(layer, sel("setContentsScale:"), s.scale())
	send(layer, "setContents:", img)
	purego.SyscallN(cgImageRelease, img)
	purego.SyscallN(cgProviderRelease, provider)
	cfRelease(data)
}

var cursorSelectors = map[platform.Cursor]string{
	platform.CursorDefault:    "arrowCursor",
	platform.CursorPointer:    "pointingHandCursor",
	platform.CursorText:       "IBeamCursor",
	platform.CursorMove:       "openHandCursor",
	platform.CursorResizeEW:   "resizeLeftRightCursor",
	platform.CursorResizeNS:   "resizeUpDownCursor",
	platform.CursorNotAllowed: "operationNotAllowedCursor",
	platform.CursorCrosshair:  "crosshairCursor",
	platform.CursorGrab:       "openHandCursor",
	platform.CursorGrabbing:   "closedHandCursor",
	platform.CursorResizeN:    "resizeUpCursor",
	platform.CursorResizeE:    "resizeRightCursor",
	platform.CursorResizeS:    "resizeDownCursor",
	platform.CursorResizeW:    "resizeLeftCursor",
	// macOS 15 has cursors for columns and rows.
	platform.CursorResizeColumn: "columnResizeCursor",
	platform.CursorResizeRow:    "rowResizeCursor",
	platform.CursorVerticalText: "IBeamCursorForVerticalLayout",
	platform.CursorCopy:         "dragCopyCursor",
	platform.CursorAlias:        "dragLinkCursor",
	platform.CursorContextMenu:  "contextualMenuCursor",
}

// cursorFallbacks are the cursors of older macOS for those it lacks.
var cursorFallbacks = map[platform.Cursor]string{
	platform.CursorResizeColumn: "resizeLeftRightCursor",
	platform.CursorResizeRow:    "resizeUpDownCursor",
}

// applyCursor sets the content's cursor, unless the pointer is over a
// web view, whose page sets its own.
func (s *surface) applyCursor() {
	if len(s.placed) > 0 {
		p := msgPoint(class("NSEvent"), sel("mouseLocation"))
		inWindow := msgRectToRect(s.w.win, sel("convertRectFromScreen:"), NSRect{Origin: p})
		local := msgPointFromView(s.view, sel("convertPoint:fromView:"), inWindow.Origin, 0)
		if s.webViewAt(local.X, local.Y) != nil {
			return
		}
	}
	send(nsCursor(s.cursor), "set")
}

// nsCursor returns the NSCursor of c. The diagonal resize cursors are the
// frame resize cursors of macOS 15, else AppKit's own older ones.
func nsCursor(c platform.Cursor) id {
	cls := class("NSCursor")
	if c == platform.CursorResizeNWSE || c == platform.CursorResizeNESW {
		corner, older := uintptr(1|2), "_windowResizeNorthWestSouthEastCursor" // top left
		if c == platform.CursorResizeNESW {
			corner, older = 1|8, "_windowResizeNorthEastSouthWestCursor" // top right
		}
		if respondsTo(cls, "frameResizeCursorFromPosition:inDirections:") {
			if cur := send(cls, "frameResizeCursorFromPosition:inDirections:", corner, 1|2); cur != 0 { // inward and outward
				return cur
			}
		}
		if respondsTo(cls, older) {
			if cur := send(cls, older); cur != 0 {
				return cur
			}
		}
	}
	if c == platform.CursorNone {
		return noCursor()
	}
	name, ok := cursorSelectors[c]
	if !ok {
		name = "arrowCursor"
	}
	if !respondsTo(cls, name) {
		if name, ok = cursorFallbacks[c]; !ok {
			name = "arrowCursor"
		}
	}
	return send(cls, name)
}

var hiddenCursor id

// noCursor returns a cursor of a transparent image, which hides the
// pointer while it is over the view, unlike NSCursor's hide, which hides
// it everywhere until unhidden.
func noCursor() id {
	if hiddenCursor == 0 {
		// A PNG of one transparent pixel.
		var png bytes.Buffer
		if err := pngenc.Encode(&png, image.NewNRGBA(image.Rect(0, 0, 1, 1))); err != nil {
			return send(class("NSCursor"), "arrowCursor")
		}
		b := png.Bytes()
		withPool(func() {
			data := send(class("NSData"), "dataWithBytes:length:", uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)))
			img := send(send(class("NSImage"), "alloc"), "initWithData:", uintptr(data))
			hiddenCursor = msgInitIDPoint(send(class("NSCursor"), "alloc"), sel("initWithImage:hotSpot:"), img, NSPoint{})
			send(img, "release")
		})
	}
	return hiddenCursor
}

func (s *surface) SetCursor(c platform.Cursor) {
	s.cursor = c
	if s.inside {
		s.applyCursor()
	}
}

func (s *surface) SetTextInput(t platform.TextInputState) {
	if s.input.Client != t.Client && s.input.Active {
		s.marked = ""
		send(send(s.view, "inputContext"), "discardMarkedText")
	}
	if s.input.Active && t.Active && t.NoInputMethod && !s.input.NoInputMethod {
		// Input methods turned off give up what they composed.
		s.marked = ""
		send(send(s.view, "inputContext"), "discardMarkedText")
	}
	if s.input.Active && !t.Active && s.marked != "" {
		s.marked = ""
		send(send(s.view, "inputContext"), "discardMarkedText")
	}
	moved := t.Active != s.input.Active || t.Caret != s.input.Caret
	s.input = t
	if t.Active && moved {
		send(send(s.view, "inputContext"), "invalidateCharacterCoordinates")
	}
}

// surfaceKeyChanged tells the content of a window that became or ceased
// to be the key window that its surface gained or lost the keyboard, as
// focus events do on the other platforms: AppKit tells views only when
// the first responder changes.
func (w *window) surfaceKeyChanged(key bool) {
	s := w.surface
	if s == nil || send(w.win, "firstResponder") != s.view {
		return
	}
	kind := platform.SurfaceBlur
	if key {
		kind = platform.SurfaceFocus
	}
	s.send(platform.SurfaceEvent{Kind: kind})
}

func (s *surface) send(ev platform.SurfaceEvent) bool {
	if s.w.closed {
		return false
	}
	return s.w.h.SurfaceEvent(ev)
}

func eventMods(ev id) platform.Modifiers {
	flags := uint(send(ev, "modifierFlags"))
	var m platform.Modifiers
	if flags&(1<<17) != 0 {
		m |= platform.ModShift
	}
	if flags&(1<<18) != 0 {
		m |= platform.ModCtrl
	}
	if flags&(1<<19) != 0 {
		m |= platform.ModAlt
	}
	if flags&(1<<20) != 0 {
		m |= platform.ModSuper
	}
	return m
}

// location returns where a mouse event happened, in the view's flipped
// coordinates.
func (s *surface) location(ev id) (float64, float64) {
	p := msgPointFromView(s.view, sel("convertPoint:fromView:"), msgPoint(ev, sel("locationInWindow")), 0)
	return p.X, p.Y
}

// sideButton sends a mouse's back or forward button pressed or released
// as KeyBack or KeyForward.
func (s *surface) sideButton(kind platform.SurfaceEventKind, back bool, mods platform.Modifiers) {
	k := platform.KeyForward
	if back {
		k = platform.KeyBack
	}
	switch kind {
	case platform.PointerDown:
		s.send(platform.SurfaceEvent{Kind: platform.KeyPressed, Key: k, Mods: mods})
	case platform.PointerUp:
		s.send(platform.SurfaceEvent{Kind: platform.KeyReleased, Key: k, Mods: mods})
	}
}

func (s *surface) mouse(kind platform.SurfaceEventKind, ev id, button int) {
	x, y := s.location(ev)
	s.send(platform.SurfaceEvent{Kind: kind, X: x, Y: y, Button: button, Clicks: sendInt(ev, "clickCount"), Mods: eventMods(ev)})
}

// macKeys maps virtual key codes that do not type a character.
var macKeys = map[uint16]platform.Key{
	0x24: platform.KeyEnter, 0x4C: platform.KeyEnter, 0x30: platform.KeyTab, 0x31: platform.KeySpace,
	0x33: platform.KeyBackspace, 0x35: platform.KeyEscape, 0x75: platform.KeyDelete, 0x72: platform.KeyInsert,
	0x73: platform.KeyHome, 0x77: platform.KeyEnd, 0x74: platform.KeyPageUp, 0x79: platform.KeyPageDown,
	0x7B: platform.KeyLeft, 0x7C: platform.KeyRight, 0x7D: platform.KeyDown, 0x7E: platform.KeyUp,
	0x7A: platform.KeyF1, 0x78: platform.KeyF2, 0x63: platform.KeyF3, 0x76: platform.KeyF4,
	0x60: platform.KeyF5, 0x61: platform.KeyF6, 0x62: platform.KeyF7, 0x64: platform.KeyF8,
	0x65: platform.KeyF9, 0x6D: platform.KeyF10, 0x67: platform.KeyF11, 0x6F: platform.KeyF12,
	0x6E: platform.KeyContextMenu,
}

func eventKey(ev id) platform.Key {
	code := uint16(send(ev, "keyCode"))
	if k, ok := macKeys[code]; ok {
		return k
	}
	chars := goString(send(ev, "charactersIgnoringModifiers"))
	for _, r := range chars {
		return platform.KeyForRune(r)
	}
	return platform.KeyUnknown
}

// ansiKeys maps virtual key codes (Carbon's kVK_ANSI_*) to the key at that
// position of an ANSI US keyboard.
var ansiKeys = map[uint16]platform.Key{
	0x00: platform.KeyA, 0x01: platform.KeyS, 0x02: platform.KeyD, 0x03: platform.KeyF,
	0x04: platform.KeyH, 0x05: platform.KeyG, 0x06: platform.KeyZ, 0x07: platform.KeyX,
	0x08: platform.KeyC, 0x09: platform.KeyV, 0x0B: platform.KeyB, 0x0C: platform.KeyQ,
	0x0D: platform.KeyW, 0x0E: platform.KeyE, 0x0F: platform.KeyR, 0x10: platform.KeyY,
	0x11: platform.KeyT, 0x1F: platform.KeyO, 0x20: platform.KeyU, 0x22: platform.KeyI,
	0x23: platform.KeyP, 0x25: platform.KeyL, 0x26: platform.KeyJ, 0x28: platform.KeyK,
	0x2D: platform.KeyN, 0x2E: platform.KeyM,
	0x12: platform.Key1, 0x13: platform.Key2, 0x14: platform.Key3, 0x15: platform.Key4,
	0x16: platform.Key6, 0x17: platform.Key5, 0x19: platform.Key9, 0x1A: platform.Key7,
	0x1C: platform.Key8, 0x1D: platform.Key0,
	0x18: platform.KeyEqual, 0x1B: platform.KeyMinus, 0x1E: platform.KeyBracketRight,
	0x21: platform.KeyBracketLeft, 0x27: platform.KeyQuote, 0x29: platform.KeySemicolon,
	0x2A: platform.KeyBackslash, 0x2B: platform.KeyComma, 0x2C: platform.KeySlash,
	0x2F: platform.KeyPeriod, 0x32: platform.KeyBackquote,
}

// keyAtCode returns the key a virtual key code stands for by position:
// the non-text keys of macKeys, then the ANSI US letter, digit and
// punctuation keys.
func keyAtCode(code uint16) platform.Key {
	if k, ok := macKeys[code]; ok {
		return k
	}
	return ansiKeys[code]
}

const (
	macISOSectionCode = 0x0A
	macANSIGraveCode  = 0x32
	macISOLayoutType  = 0x49534F20 // Carbon's kKeyboardISO ('ISO ')
)

func keyAtCodeForLayout(code uint16, layoutType uint32) platform.Key {
	if layoutType == macISOLayoutType {
		// On ISO keyboards the key left of 1 is kVK_ISO_Section, while
		// kVK_ANSI_Grave is the extra key beside the left Shift.
		switch code {
		case macISOSectionCode:
			return platform.KeyBackquote
		case macANSIGraveCode:
			return platform.KeyUnknown // No key for ISO's extra key in platform.Key.
		}
	}
	return keyAtCode(code)
}

// physicalKey returns the ANSI US key at the event's physical position,
// independent of the current input source. Only the grave key differs on ISO.
func physicalKey(ev id) platform.Key {
	code := uint16(send(ev, "keyCode"))
	if code != macISOSectionCode && code != macANSIGraveCode {
		return keyAtCode(code)
	}
	var keyboardType int16
	if cgEvent := send(ev, "CGEvent"); cgEvent != 0 {
		keyboardType = int16(cgEventGetIntegerValueField(uintptr(cgEvent), 10)) // kCGKeyboardEventKeyboardType
	}
	if keyboardType == 0 { // Synthetic NSEvents can lack a keyboard type.
		keyboardType = int16(lmGetKbdType())
	}
	return keyAtCodeForLayout(code, kbGetLayoutType(keyboardType))
}

// stringOf returns the text of an NSString or NSAttributedString.
func stringOf(obj id) string {
	if obj == 0 {
		return ""
	}
	if respondsTo(obj, "string") {
		obj = send(obj, "string")
	}
	return goString(obj)
}

// utf16Len returns the length of s in UTF-16 code units, as NSRanges count.
func utf16Len(s string) int { return len(utf16.Encode([]rune(s))) }

// runesBefore converts a UTF-16 offset in s into a rune offset.
func runesBefore(s string, units int) int {
	u := utf16.Encode([]rune(s))
	units = max(0, min(units, len(u)))
	return len(utf16.Decode(u[:units]))
}

func (b *Backend) surfaceOf(view id) *surface {
	s := b.bySurface[view]
	if s == nil || s.w.closed {
		return nil
	}
	return s
}

func registerSurfaceClass() {
	registerDataDragClasses()
	b := func() *Backend { return theBackend }
	mouse := func(kind platform.SurfaceEventKind, button int) func(id, objc.SEL, id) {
		return func(self id, _ objc.SEL, ev id) {
			s := b().surfaceOf(self)
			if s == nil {
				return
			}
			btn := button
			switch {
			case button == 2:
				n := sendInt(ev, "buttonNumber")
				if n == 3 || n == 4 {
					// The side buttons go back and forward, as keys do.
					s.sideButton(kind, n == 3, eventMods(ev))
					return
				}
				btn = min(n, 2)
			case button == 0 && kind == platform.PointerDown:
				// A Control-click is a secondary click, until its release.
				s.ctrlClick = eventMods(ev)&platform.ModCtrl != 0
			}
			if button == 0 && s.ctrlClick {
				btn = 1
			}
			if kind == platform.PointerDown {
				if btn == 0 {
					release(s.w.lastMouseDown)
					s.w.lastMouseDown = retain(ev)
				}
				send(s.w.win, "makeFirstResponder:", uintptr(self))
			}
			s.mouse(kind, ev, btn)
			if kind == platform.PointerUp && button == 0 {
				s.ctrlClick = false
			}
		}
	}
	command := func(name string) objc.MethodDef {
		return method(name+":", func(self id, _ objc.SEL, sender id) {
			if s := b().surfaceOf(self); s != nil {
				s.send(platform.SurfaceEvent{Kind: platform.SurfaceCommand, Text: name})
			}
		})
	}
	dragged := func(self id, _ objc.SEL, info id) uint {
		s := b().surfaceOf(self)
		if s == nil {
			return 0
		}
		return s.nativeDataOperation(info)
	}
	methods := []objc.MethodDef{
		// Files dragged from other apps.
		method("draggingEntered:", dragged),
		method("draggingUpdated:", dragged),
		method("draggingExited:", func(self id, _ objc.SEL, info id) {
			if s := b().surfaceOf(self); s != nil {
				s.nativeDataEvent(info, platform.DataDragLeave)
			}
		}),
		method("prepareForDragOperation:", func(self id, _ objc.SEL, info id) bool { return true }),
		method("performDragOperation:", func(self id, _ objc.SEL, info id) bool {
			s := b().surfaceOf(self)
			if s == nil {
				return false
			}
			return s.nativeDataEvent(info, platform.DataDrop)
		}),
	}
	classDef("MyGoSurfaceView", "NSView", []string{"NSTextInputClient"}, append(append(methods, accessViewMethods()...), []objc.MethodDef{
		method("isFlipped", func(self id, _ objc.SEL) bool { return true }),
		// The pointer goes through to the web views under the surface
		// where the content shows them (webview.go).
		method("hitTest:", func(self id, cmd objc.SEL, p NSPoint) id {
			if s := b().surfaceOf(self); s != nil && len(s.placed) > 0 {
				local := msgPointFromView(self, sel("convertPoint:fromView:"), p, send(self, "superview"))
				if s.webViewAt(local.X, local.Y) != nil {
					return 0
				}
			}
			return sendSuperPoint(self, "MyGoSurfaceView", cmd, p)
		}),
		method("acceptsFirstResponder", func(self id, _ objc.SEL) bool { return true }),
		method("acceptsFirstMouse:", func(self id, _ objc.SEL, ev id) bool { return true }),
		method("mouseDownCanMoveWindow", func(self id, _ objc.SEL) bool { return false }),
		method("wantsUpdateLayer", func(self id, _ objc.SEL) bool { return true }),
		method("updateLayer", func(self id, _ objc.SEL) {
			s := b().surfaceOf(self)
			if s == nil {
				return
			}
			if s.framePass == caPass && send(self, "inLiveResize") == 0 && (!s.resized || s.presentsWithTransaction()) {
				// AppKit displays the view again before Core Animation
				// commits the frame drawn in this pass, as while the
				// trackpad scrolls. The next refresh draws it: as one just
				// drew, not at once; and after a frame presented with the
				// transaction, the frame would wait a second for a
				// drawable, the layer's two being taken until the commit.
				// A frame for a new size after one presented at once, as
				// those of the display link's ticks are, draws at once, or
				// the layer would stretch the last one meanwhile. A window
				// that resizes, as zooming animates it, displays the view
				// for each of its sizes within one pass of the run loop,
				// committing each: those frames draw at once too.
				s.lastFrame = time.Now()
				s.RequestFrame()
				return
			}
			s.resized = false
			s.framePass = caPass
			s.lastFrame = time.Now()
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceFrame})
		}),
		method("mygoTick:", func(self id, _ objc.SEL, link id) {
			s := b().surfaceOf(self)
			if s == nil || !s.due {
				// No frame was asked for since the last refresh.
				send(link, "setPaused:", 1)
				if s != nil {
					s.linkRunning = false
				}
				return
			}
			s.due = false
			if sendBool(send(self, "layer"), "needsDisplay") {
				// AppKit displays the view in this pass, as after a
				// resize (the view's needsDisplay tells nothing): the
				// frame shows with the window's changes then.
				return
			}
			// The frame draws now, rather than when AppKit displays the
			// view, which would cost a pass of its display cycle and a
			// commit of Core Animation's transaction for each frame: the
			// renderer presents frames that change nothing else as soon
			// as drawn.
			s.resized = false
			s.framePass = caPass
			s.lastFrame = time.Now()
			s.send(platform.SurfaceEvent{Kind: platform.SurfaceFrame})
		}),
		method("mygoTimer:", func(self id, _ objc.SEL, timer id) {
			// A closed window's view lives until its timer fires.
			if s := b().surfaceOf(self); s != nil {
				s.timing = false
				send(self, "setNeedsDisplay:", 1)
			}
		}),
		method("setFrameSize:", func(self id, cmd objc.SEL, size NSSize) {
			sendSuperSize(self, "MyGoSurfaceView", cmd, size)
			if s := b().surfaceOf(self); s != nil {
				s.resized = true
				s.send(platform.SurfaceEvent{Kind: platform.SurfaceResize})
				send(self, "setNeedsDisplay:", 1)
			}
		}),
		method("viewDidChangeBackingProperties", func(self id, _ objc.SEL) {
			if s := b().surfaceOf(self); s != nil {
				s.resized = true
				s.send(platform.SurfaceEvent{Kind: platform.SurfaceResize})
				send(self, "setNeedsDisplay:", 1)
			}
		}),
		method("becomeFirstResponder", func(self id, _ objc.SEL) bool {
			if s := b().surfaceOf(self); s != nil {
				s.send(platform.SurfaceEvent{Kind: platform.SurfaceFocus})
			}
			return true
		}),
		method("resignFirstResponder", func(self id, _ objc.SEL) bool {
			if s := b().surfaceOf(self); s != nil {
				s.send(platform.SurfaceEvent{Kind: platform.SurfaceBlur})
			}
			return true
		}),
		method("mouseDown:", mouse(platform.PointerDown, 0)),
		method("mouseUp:", mouse(platform.PointerUp, 0)),
		method("mouseDragged:", mouse(platform.PointerMove, 0)),
		method("rightMouseDown:", mouse(platform.PointerDown, 1)),
		method("rightMouseUp:", mouse(platform.PointerUp, 1)),
		method("rightMouseDragged:", mouse(platform.PointerMove, 1)),
		method("otherMouseDown:", mouse(platform.PointerDown, 2)),
		method("otherMouseUp:", mouse(platform.PointerUp, 2)),
		method("otherMouseDragged:", mouse(platform.PointerMove, 2)),
		method("mouseMoved:", mouse(platform.PointerMove, 0)),
		method("mouseEntered:", func(self id, _ objc.SEL, ev id) {
			if s := b().surfaceOf(self); s != nil {
				s.inside = true
				s.applyCursor()
			}
		}),
		method("mouseExited:", func(self id, _ objc.SEL, ev id) {
			if s := b().surfaceOf(self); s != nil {
				s.inside = false
				s.send(platform.SurfaceEvent{Kind: platform.PointerLeave})
			}
		}),
		method("cursorUpdate:", func(self id, _ objc.SEL, ev id) {
			if s := b().surfaceOf(self); s != nil {
				s.applyCursor()
			}
		}),
		method("scrollWheel:", func(self id, _ objc.SEL, ev id) {
			s := b().surfaceOf(self)
			if s == nil {
				return
			}
			x, y := s.location(ev)
			dx, dy := msgFloat(ev, sel("scrollingDeltaX")), msgFloat(ev, sel("scrollingDeltaY"))
			precise := sendBool(ev, "hasPreciseScrollingDeltas")
			if !precise {
				// A mouse wheel scrolls by lines: 40 DIPs, as in browsers.
				dx, dy = dx*40, dy*40
			}
			s.send(platform.SurfaceEvent{Kind: platform.PointerScroll, X: x, Y: y, DX: -dx, DY: -dy, Precise: precise, Mods: eventMods(ev)})
		}),
		method("keyDown:", func(self id, _ objc.SEL, ev id) {
			s := b().surfaceOf(self)
			if s == nil {
				return
			}
			mods := eventMods(ev)
			text := s.input.Active && mods&(platform.ModSuper|platform.ModCtrl) == 0
			ime := text && !s.input.NoInputMethod
			// A key typed while the input method composes is the input
			// method's alone, as Enter choosing a candidate or Escape
			// giving the composition up, as GTK's and IMM32's filtering
			// keeps them on Linux and Windows.
			if !ime || !s.hasMarkedText() {
				s.send(platform.SurfaceEvent{Kind: platform.KeyPressed, Key: eventKey(ev), PhysicalKey: physicalKey(ev), Mods: mods, Repeat: sendBool(ev, "isARepeat")})
			}
			// Input methods see the key while a text input has the focus;
			// they answer with insertText: or setMarkedText:.
			if ime {
				s.keyDown = true
				send(self, "interpretKeyEvents:", uintptr(nsArray(ev)))
				s.keyDown = false
			} else if text {
				// Input methods off: the key types the characters of the
				// layout, as winit's keys do without them.
				if t := typedChars(stringOf(send(ev, "characters"))); t != "" {
					s.insertText(t, nsRange{Location: nsNotFound})
				}
			}
		}),
		method("keyUp:", func(self id, _ objc.SEL, ev id) {
			if s := b().surfaceOf(self); s != nil {
				s.send(platform.SurfaceEvent{Kind: platform.KeyReleased, Key: eventKey(ev), PhysicalKey: physicalKey(ev), Mods: eventMods(ev)})
			}
		}),
		method("flagsChanged:", func(self id, _ objc.SEL, ev id) {
			if s := b().surfaceOf(self); s != nil {
				s.send(platform.SurfaceEvent{Kind: platform.ModifiersChanged, Mods: eventMods(ev)})
			}
		}),
		command("copy"), command("cut"), command("paste"), command("selectAll"), command("undo"), command("redo"), command("delete"),
		method("pasteAsPlainText:", func(self id, _ objc.SEL, sender id) {
			if s := b().surfaceOf(self); s != nil {
				s.send(platform.SurfaceEvent{Kind: platform.SurfaceCommand, Text: "pasteAndMatchStyle"})
			}
		}),
		method("validateMenuItem:", func(self id, _ objc.SEL, item id) bool { return true }),

		// NSTextInputClient
		method("hasMarkedText", func(self id, _ objc.SEL) bool {
			s := b().surfaceOf(self)
			return s != nil && s.hasMarkedText()
		}),
		method("markedRange", func(self id, _ objc.SEL) nsRange {
			if s := b().surfaceOf(self); s != nil {
				return s.markedRange()
			}
			return nsRange{Location: nsNotFound}
		}),
		method("selectedRange", func(self id, _ objc.SEL) nsRange {
			if s := b().surfaceOf(self); s != nil {
				return s.selectedRange()
			}
			return nsRange{Location: nsNotFound}
		}),
		method("setMarkedText:selectedRange:replacementRange:", func(self id, _ objc.SEL, text id, selected, replacement nsRange) {
			if s := b().surfaceOf(self); s != nil {
				s.setMarkedText(stringOf(text), selected, replacement)
			}
		}),
		method("unmarkText", func(self id, _ objc.SEL) {
			if s := b().surfaceOf(self); s != nil {
				s.unmarkText()
			}
		}),
		method("validAttributesForMarkedText", func(self id, _ objc.SEL) id { return nsArray() }),
		method("attributedSubstringForProposedRange:actualRange:", func(self id, _ objc.SEL, r nsRange, actual *nsRange) id {
			if s := b().surfaceOf(self); s != nil {
				return s.substring(r, actual)
			}
			return 0
		}),
		method("insertText:replacementRange:", func(self id, _ objc.SEL, text id, replacement nsRange) {
			if s := b().surfaceOf(self); s != nil {
				s.insertText(stringOf(text), replacement)
			}
		}),
		method("characterIndexForPoint:", func(self id, _ objc.SEL, p NSPoint) uint {
			s := b().surfaceOf(self)
			if s != nil && s.input.Client != nil {
				window := msgRectToRect(s.w.win, sel("convertRectFromScreen:"), NSRect{Origin: p})
				local := msgConvertRectView(self, sel("convertRect:fromView:"), window, 0)
				if index, ok := s.input.Client.IndexForPoint(local.Origin.X, local.Origin.Y); ok {
					return uint(index)
				}
			}
			return nsNotFound
		}),
		method("firstRectForCharacterRange:actualRange:", func(self id, _ objc.SEL, r nsRange, actual *nsRange) cgRect {
			s := b().surfaceOf(self)
			if s == nil {
				return cgRect{}
			}
			c := s.input.Caret
			if client := s.input.Client; client != nil {
				rangeWanted := clientRange(r)
				if rangeWanted == nil {
					return cgRect{}
				}
				bounds, used, ok := client.BoundsForRange(*rangeWanted)
				if !ok {
					return cgRect{}
				}
				c = bounds
				if actual != nil {
					*actual = nsRange{Location: uint(used.Start), Length: uint(max(0, used.End-used.Start))}
				}
			}
			inWindow := msgConvertRectView(self, sel("convertRect:toView:"), NSRect{Origin: NSPoint{c.X, c.Y}, Size: NSSize{max(c.W, 1), c.H}}, 0)
			screen := msgRectToRect(s.w.win, sel("convertRectToScreen:"), inWindow)
			return cgRect{X: screen.Origin.X, Y: screen.Origin.Y, W: screen.Size.Width, H: screen.Size.Height}
		}),
		method("doCommandBySelector:", func(self id, _ objc.SEL, cmd objc.SEL) {}),
	}...))
}

// dragPoint returns where a drag is, in the view's coordinates.
func (s *surface) dragPoint(info id) (float64, float64) {
	p := msgPointFromView(s.view, sel("convertPoint:fromView:"), msgPoint(info, sel("draggingLocation")), 0)
	return p.X, p.Y
}
