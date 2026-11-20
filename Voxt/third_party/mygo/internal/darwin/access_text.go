//go:build darwin

package darwin

import (
	"unicode/utf16"

	"github.com/egoist/mygo/internal/platform"
)

// attributedAccessString returns the attributed string of the UTF-16
// range r of a text field's value, with the formats of its runs in the
// keys of NSAccessibility (AXFont, AXForegroundColor, AXUnderline, …),
// autoreleased.
func attributedAccessString(n *platform.AccessNode, r nsRange) id {
	runes := []rune(n.Value)
	u := utf16.Encode(runes)
	lo := int(min(r.Location, uint(len(u))))
	hi := int(min(uint(lo)+r.Length, uint(len(u))))
	str := nsString(string(utf16.Decode(u[lo:hi])))
	as := autorelease(send(send(class("NSMutableAttributedString"), "alloc"), "initWithString:", uintptr(str)))
	if hi == lo {
		return as
	}
	// The UTF-16 offset of each rune.
	units := make([]int, len(runes)+1)
	for i, c := range runes {
		units[i+1] = units[i] + utf16.RuneLen(c)
	}
	add := func(key string, value id, a, z int) {
		a, z = max(a, lo), min(z, hi)
		if value == 0 || a >= z {
			return
		}
		send(as, "addAttribute:value:range:", uintptr(nsString(key)), uintptr(value), uintptr(a-lo), uintptr(z-a))
	}
	add("AXFont", accessFont(platform.AccessTextRun{}), lo, hi)
	for _, run := range n.TextRuns {
		a, z := units[max(0, min(run.Start, len(runes)))], units[max(0, min(run.End, len(runes)))]
		if z <= lo || a >= hi {
			continue
		}
		add("AXFont", accessFont(run), a, z)
		if run.Color[3] > 0 {
			add("AXForegroundColor", accessColor(run.Color), a, z)
		}
		if run.Background[3] > 0 {
			add("AXBackgroundColor", accessColor(run.Background), a, z)
		}
		if run.Underline {
			add("AXUnderline", nsNumberInt(1), a, z)
		}
		if run.Strikethrough {
			add("AXStrikethrough", nsBool(true), a, z)
		}
		if run.Link != "" {
			if url := send(class("NSURL"), "URLWithString:", uintptr(nsString(run.Link))); url != 0 {
				add("AXLink", url, a, z)
			}
		}
	}
	return as
}

// accessFont returns the AXFont of a run: the dictionary of its font's
// name, family and size, the font as AppKit finds it for the run's family,
// weight and italics.
func accessFont(run platform.AccessTextRun) id {
	size := float64(run.Size)
	if size <= 0 {
		size = 14
	}
	var font id
	if run.Family == "monospace" {
		font = msgFloat2ID(class("NSFont"), sel("monospacedSystemFontOfSize:weight:"), size, nsFontWeight(run.Weight))
	} else if run.Family != "" && run.Family != "system-ui" {
		var traits uint
		if run.Weight >= 600 {
			traits |= 2 // NSBoldFontMask
		}
		if run.Italic {
			traits |= 1 // NSItalicFontMask
		}
		font = msgFontFamily(send(class("NSFontManager"), "sharedFontManager"), sel("fontWithFamily:traits:weight:size:"),
			nsString(run.Family), traits, 5, size)
	}
	if font == 0 {
		font = msgFloat2ID(class("NSFont"), sel("systemFontOfSize:weight:"), size, nsFontWeight(run.Weight))
		if run.Italic && font != 0 {
			font = send(send(class("NSFontManager"), "sharedFontManager"), "convertFont:toHaveTrait:", uintptr(font), 1)
		}
	}
	d := send(class("NSMutableDictionary"), "dictionary")
	if font != 0 {
		name := send(font, "fontName")
		send(d, "setObject:forKey:", uintptr(name), uintptr(nsString("AXFontName")))
		send(d, "setObject:forKey:", uintptr(send(font, "familyName")), uintptr(nsString("AXFontFamily")))
		send(d, "setObject:forKey:", uintptr(send(font, "displayName")), uintptr(nsString("AXVisibleName")))
	}
	send(d, "setObject:forKey:", uintptr(msgFloatID(class("NSNumber"), sel("numberWithDouble:"), size)), uintptr(nsString("AXFontSize")))
	return d
}

// nsFontWeight returns the NSFontWeight of a CSS weight.
func nsFontWeight(w int) float64 {
	switch {
	case w <= 0:
		return 0
	case w < 200:
		return -0.8
	case w < 300:
		return -0.6
	case w < 400:
		return -0.4
	case w < 500:
		return 0
	case w < 600:
		return 0.23
	case w < 700:
		return 0.3
	case w < 800:
		return 0.4
	case w < 900:
		return 0.56
	}
	return 0.62
}

// accessColor returns the CGColor of an RGBA color, which the NSColor
// autoreleased with it keeps.
func accessColor(c [4]uint8) id {
	return send(nsColor(platform.Color{R: c[0], G: c[1], B: c[2], A: c[3]}), "CGColor")
}
