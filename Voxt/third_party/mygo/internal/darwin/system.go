//go:build darwin

package darwin

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// Clipboard (NSPasteboard).

type clipboard struct{}

const (
	utString = "public.utf8-plain-text"
	utHTML   = "public.html"
	utPNG    = "public.png"
	utTIFF   = "public.tiff"
)

func pasteboard() id { return send(class("NSPasteboard"), "generalPasteboard") }

func (clipboard) ReadText() string {
	var s string
	withPool(func() { s = goString(send(pasteboard(), "stringForType:", uintptr(nsString(utString)))) })
	return s
}

func (c clipboard) WriteText(text string) { _ = c.WriteData(transfer.TextData(text), nil) }

func (clipboard) ReadHTML() string {
	var s string
	withPool(func() { s = goString(send(pasteboard(), "stringForType:", uintptr(nsString(utHTML)))) })
	return s
}

func (c clipboard) WriteHTML(markup string) {
	_ = c.WriteData(transfer.New(transfer.NewItem(transfer.Bytes(transfer.HTML, []byte(markup)), transfer.Bytes(transfer.Text, []byte(markup)))), nil)
}

func (clipboard) ReadImage() []byte {
	var out []byte
	withPool(func() {
		pb := pasteboard()
		if data := send(pb, "dataForType:", uintptr(nsString(utPNG))); data != 0 {
			out = goBytes(data)
			return
		}
		if data := send(pb, "dataForType:", uintptr(nsString(utTIFF))); data != 0 {
			rep := send(class("NSBitmapImageRep"), "imageRepWithData:", uintptr(data))
			if rep != 0 {
				out = goBytes(send(rep, "representationUsingType:properties:", 4, uintptr(send(class("NSDictionary"), "dictionary"))))
			}
		}
	})
	return out
}

func (c clipboard) WriteImage(png []byte) error {
	var err error
	withPool(func() {
		img := autorelease(send(send(class("NSImage"), "alloc"), "initWithData:", uintptr(nsData(png))))
		if img == 0 {
			err = errInvalidImage
		}
	})
	if err != nil {
		return err
	}
	return c.WriteData(transfer.New(transfer.NewItem(transfer.Bytes(transfer.PNG, png))), nil)
}
func (clipboard) Clear() {
	withPool(func() {
		send(pasteboard(), "clearContents")
		if s := macClipboard; s != nil {
			s.release()
		}
	})
}

func (clipboard) AvailableFormats() []string {
	var out []string
	withPool(func() {
		for _, t := range arrayItems(send(pasteboard(), "types")) {
			out = append(out, goString(t))
		}
	})
	return out
}

// Shell (NSWorkspace).

type shell struct{}

func workspace() id { return send(class("NSWorkspace"), "sharedWorkspace") }

func (shell) OpenExternal(url string) error {
	var err error
	withPool(func() {
		u := nsURL(url)
		if u == 0 {
			err = fmt.Errorf("mygo: invalid URL %q", url)
			return
		}
		if !sendBool(workspace(), "openURL:", uintptr(u)) {
			err = fmt.Errorf("mygo: no application can open %q", url)
		}
	})
	return err
}

func (shell) OpenPath(path string) error {
	var err error
	withPool(func() {
		if !sendBool(workspace(), "openURL:", uintptr(fileURL(path))) {
			err = fmt.Errorf("mygo: cannot open %q", path)
		}
	})
	return err
}

func (shell) ShowItemInFolder(path string) {
	withPool(func() {
		send(workspace(), "activateFileViewerSelectingURLs:", uintptr(nsArray(fileURL(path))))
	})
}

func (shell) TrashItem(path string) error {
	var err error
	withPool(func() {
		var nserr id
		fm := send(class("NSFileManager"), "defaultManager")
		if !sendBool(fm, "trashItemAtURL:resultingItemURL:error:", uintptr(fileURL(path)), 0, uintptr(unsafe.Pointer(&nserr))) {
			err = nsError(nserr)
			if err == nil {
				err = errors.New("mygo: cannot move to trash")
			}
		}
	})
	return err
}

func (shell) Beep() { nsBeep() }

// Screen (NSScreen).

type screen struct{}

func (screen) Displays() []platform.Display {
	var out []platform.Display
	withPool(func() {
		screens := arrayItems(send(class("NSScreen"), "screens"))
		if len(screens) == 0 {
			return
		}
		key := nsString("NSScreenNumber")
		for i, s := range screens {
			num := send(send(s, "deviceDescription"), "objectForKey:", uintptr(key))
			displayID := uint32(send(num, "unsignedIntValue"))
			d := platform.Display{
				ID:          int64(displayID),
				Bounds:      rectFromMac(msgRect(s, sel("frame"))),
				WorkArea:    rectFromMac(msgRect(s, sel("visibleFrame"))),
				ScaleFactor: msgFloat(s, sel("backingScaleFactor")),
				Rotation:    int(math.Round(cgDisplayRotation(displayID))),
				Internal:    cgDisplayIsBuiltin(displayID),
				Primary:     i == 0,
			}
			if respondsTo(s, "localizedName") {
				d.Label = goString(send(s, "localizedName"))
			}
			out = append(out, d)
		}
	})
	return out
}

func (screen) CursorPoint() platform.Point {
	p := msgPoint(class("NSEvent"), sel("mouseLocation"))
	return platform.Point{X: int(math.Round(p.X)), Y: int(math.Round(primaryScreenHeight() - p.Y))}
}

// Theme (NSAppearance).

type theme struct{ b *Backend }

// UIFont is Core Text's to find: the system font.
func (theme) UIFont() string { return "" }

func (theme) FontRendering() platform.FontRendering { return platform.FontRendering{} }

// Preferences reads the accent color and the display settings of
// accessibility.
func (t theme) Preferences() platform.Preferences {
	var p platform.Preferences
	withPool(func() {
		ws := workspace()
		p.ReduceMotion = sendBool(ws, "accessibilityDisplayShouldReduceMotion")
		p.ReduceTransparency = sendBool(ws, "accessibilityDisplayShouldReduceTransparency")
		p.HighContrast = sendBool(ws, "accessibilityDisplayShouldIncreaseContrast")
		// The accent, as the app's appearance draws it.
		var c id
		appearance := class("NSAppearance")
		prev := send(appearance, "currentAppearance")
		send(appearance, "setCurrentAppearance:", uintptr(send(t.b.app, "effectiveAppearance")))
		if accent := send(class("NSColor"), "controlAccentColor"); accent != 0 {
			c = send(accent, "colorUsingColorSpace:", uintptr(send(class("NSColorSpace"), "sRGBColorSpace")))
		}
		send(appearance, "setCurrentAppearance:", uintptr(prev))
		if c != 0 {
			var r, g, b, a float64
			send(c, "getRed:green:blue:alpha:", uintptr(unsafe.Pointer(&r)), uintptr(unsafe.Pointer(&g)), uintptr(unsafe.Pointer(&b)), uintptr(unsafe.Pointer(&a)))
			byteOf := func(v float64) uint8 { return uint8(max(0, min(v, 1))*255 + 0.5) }
			p.Accent = platform.Color{R: byteOf(r), G: byteOf(g), B: byteOf(b), A: 255}
		}
	})
	return p
}

func (t theme) IsDark() bool {
	var dark bool
	withPool(func() {
		appearance := send(t.b.app, "effectiveAppearance")
		names := nsArray(nsString("NSAppearanceNameAqua"), nsString("NSAppearanceNameDarkAqua"))
		dark = goString(send(appearance, "bestMatchFromAppearancesWithNames:", uintptr(names))) == "NSAppearanceNameDarkAqua"
	})
	return dark
}

func (t theme) SetSource(source string) {
	withPool(func() {
		var appearance id
		switch source {
		case "light":
			appearance = send(class("NSAppearance"), "appearanceNamed:", uintptr(nsString("NSAppearanceNameAqua")))
		case "dark":
			appearance = send(class("NSAppearance"), "appearanceNamed:", uintptr(nsString("NSAppearanceNameDarkAqua")))
		}
		send(t.b.app, "setAppearance:", uintptr(appearance))
	})
}
