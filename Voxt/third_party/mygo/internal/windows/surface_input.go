//go:build windows && (amd64 || arm64)

package windows

import (
	"unicode/utf16"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// Input methods ask for the text around the caret with WM_IME_REQUEST:
// IMR_DOCUMENTFEED to convert in context, and IMR_RECONVERTSTRING to
// convert again what was typed, whose range IMR_CONFIRMRECONVERTSTRING
// then settles. The composition that follows replaces that range.

const (
	wmImeRequest              = 0x0288
	imrReconvertString        = 0x0004
	imrConfirmReconvertString = 0x0005
	imrDocumentFeed           = 0x0007
)

// reconvertString is RECONVERTSTRING, which the string follows in memory.
// Lengths count UTF-16 units; offsets, bytes from the string's start.
type reconvertString struct {
	Size, Version, StrLen, StrOffset, CompStrLen, CompStrOffset, TargetStrLen, TargetStrOffset uint32
}

// imeRequest answers WM_IME_REQUEST.
func (s *surface) imeRequest(wp, lp uintptr) uintptr {
	if !s.input.Active {
		return 0
	}
	units := utf16Units(s.input.Text)
	var context platform.TextContext
	start, end := unitOffset(s.input.Text, s.input.Start), unitOffset(s.input.Text, s.input.End)
	if c := s.input.Client; c != nil {
		context = platform.ClientTextContext(c)
		units = utf16Units(context.Text)
		start, end = context.Start, context.End
	}
	switch wp {
	case imrDocumentFeed, imrReconvertString:
		head := uint32(unsafe.Sizeof(reconvertString{}))
		size := head + uint32(len(units))*2
		if lp == 0 {
			return uintptr(size)
		}
		rs := (*reconvertString)(native(lp))
		if rs.Size < size {
			return 0
		}
		*rs = reconvertString{Size: size, StrLen: uint32(len(units)), StrOffset: head,
			CompStrLen: uint32(end - start), CompStrOffset: uint32(start) * 2,
			TargetStrLen: uint32(end - start), TargetStrOffset: uint32(start) * 2}
		copy(unsafe.Slice((*uint16)(unsafe.Add(native(lp), head)), len(units)), units)
		return uintptr(size)
	case imrConfirmReconvertString:
		rs := (*reconvertString)(native(lp))
		from := int(rs.CompStrOffset / 2)
		to := from + int(rs.CompStrLen)
		if from < 0 || to > len(units) {
			return 0
		}
		if s.input.Client != nil {
			r := context.Range(platform.TextRange{Start: from, End: to})
			s.reconvert = &[2]int{r.Start, r.End}
		} else {
			s.reconvert = &[2]int{runeOffset(units, from), runeOffset(units, to)}
		}
		return 1
	}
	return 0
}

// composed sends what an input method composed or committed, in place of
// the text it reconverts, if any.
func (s *surface) composed(ev platform.SurfaceEvent) {
	if r := s.reconvert; r != nil {
		ev.Replace, ev.From, ev.To = true, r[0], r[1]
		s.reconvert = nil
	}
	if c := s.input.Client; c != nil {
		var r *platform.TextRange
		if ev.Replace {
			rangeWanted := platform.TextRange{Start: ev.From, End: ev.To}
			r = &rangeWanted
		}
		if ev.Kind == platform.TextInput {
			s.clientComposition.Replace(c, r, ev.Text)
		} else {
			s.clientComposition.Reset()
			caret := platform.UTF16Len(string([]rune(ev.Text)[:min(ev.Caret, len([]rune(ev.Text)))]))
			c.SetMarkedText(r, ev.Text, platform.TextRange{Start: caret, End: caret})
		}
		return
	}
	s.send(ev)
}

// unitOffset returns the UTF-16 offset of rune i of s.
func unitOffset(s string, i int) int {
	n := 0
	for _, r := range s {
		if i == 0 {
			break
		}
		n += utf16.RuneLen(r)
		i--
	}
	return n
}

// runeOffset returns the rune offset of UTF-16 unit i.
func runeOffset(units []uint16, i int) int {
	return len(utf16.Decode(units[:i]))
}
