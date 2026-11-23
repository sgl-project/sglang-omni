//go:build windows && (amd64 || arm64)

package windows

import (
	"math"
	"runtime"
	"strconv"
	"syscall"
	"time"
	"unicode/utf16"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// Hooks for tests of native UI with input methods, file drops and
// assistive technology.

func surfaceByHandle(hwnd uintptr) *surface {
	if w := theBackend.windows[hwnd]; w != nil {
		return w.surface
	}
	return nil
}

// TestDocumentFeed returns the text around the caret that input methods
// get from a window showing native UI with IMR_DOCUMENTFEED, and the
// selection in it, in runes.
func TestDocumentFeed(hwnd uintptr) (text string, start, length int, ok bool) {
	s := surfaceByHandle(hwnd)
	if s == nil {
		return "", 0, 0, false
	}
	size, _, _ := procSendMessageW.Call(s.hwnd, wmImeRequest, imrDocumentFeed, 0)
	if size == 0 {
		return "", 0, 0, true
	}
	buf := make([]uint32, (size+3)/4)
	rs := (*reconvertString)(unsafe.Pointer(&buf[0]))
	rs.Size = uint32(size)
	procSendMessageW.Call(s.hwnd, wmImeRequest, imrDocumentFeed, uintptr(unsafe.Pointer(rs)))
	units := unsafe.Slice((*uint16)(unsafe.Add(unsafe.Pointer(rs), rs.StrOffset)), rs.StrLen)
	from := int(rs.TargetStrOffset / 2)
	to := from + int(rs.TargetStrLen)
	return string(utf16.Decode(units)), runeOffset(units, from), runeOffset(units, to) - runeOffset(units, from), true
}

// TestComposeOver does what an input method that converts typed text again
// does: unless from is -1, it settles the reconversion of length runes from
// rune from of the text around the caret; then it composes text with the
// caret at caret, or commits it.
func TestComposeOver(hwnd uintptr, text string, caret int, commit bool, from, length int) bool {
	s := surfaceByHandle(hwnd)
	if s == nil {
		return false
	}
	if from >= 0 {
		doc, _, _, _ := TestDocumentFeed(hwnd)
		units := utf16Units(doc)
		head := uint32(unsafe.Sizeof(reconvertString{}))
		buf := make([]uint32, (int(head)+len(units)*2+3)/4)
		rs := (*reconvertString)(unsafe.Pointer(&buf[0]))
		a, b := unitOffset(doc, from), unitOffset(doc, from+length)
		*rs = reconvertString{Size: head + uint32(len(units))*2, StrLen: uint32(len(units)), StrOffset: head,
			CompStrLen: uint32(b - a), CompStrOffset: uint32(a) * 2, TargetStrLen: uint32(b - a), TargetStrOffset: uint32(a) * 2}
		copy(unsafe.Slice((*uint16)(unsafe.Add(unsafe.Pointer(rs), head)), len(units)), units)
		if r, _, _ := procSendMessageW.Call(s.hwnd, wmImeRequest, imrConfirmReconvertString, uintptr(unsafe.Pointer(rs))); r == 0 {
			return false
		}
	}
	// Input methods compose in contexts out of the test's reach: send what
	// WM_IME_COMPOSITION would.
	if commit {
		s.composed(platform.SurfaceEvent{Kind: platform.TextInput, Text: text})
	} else {
		s.composed(platform.SurfaceEvent{Kind: platform.TextComposition, Text: text, Caret: caret})
	}
	return true
}

// testDataObject is an IDataObject holding files, as Explorer's are.
type testDataObject struct {
	vtbl  *[12]uintptr
	refs  int32
	paths []string
}

var (
	testDataVtbl    [12]uintptr
	testDataObjects = map[uintptr]*testDataObject{}
)

func newTestDataObject(paths []string) uintptr {
	if testDataVtbl[0] == 0 {
		const eNotImpl, dvFormatEtc = 0x80004001, 0x80040064
		data := func(this uintptr) *testDataObject { return testDataObjects[this] }
		isHDrop := func(f uintptr) bool {
			fe := (*formatEtc)(native(f))
			return fe.Format == cfHDrop && fe.Tymed&tymedHGlobal != 0
		}
		notImpl := syscall.NewCallback(func(this, a, b, c uintptr) uintptr { return eNotImpl })
		testDataVtbl = [12]uintptr{
			syscall.NewCallback(func(this, riid, out uintptr) uintptr {
				*(*uintptr)(native(out)) = this
				data(this).refs++
				return sOK
			}),
			syscall.NewCallback(func(this uintptr) uintptr { d := data(this); d.refs++; return uintptr(d.refs) }),
			syscall.NewCallback(func(this uintptr) uintptr {
				d := data(this)
				if d.refs--; d.refs == 0 {
					delete(testDataObjects, this)
				}
				return uintptr(d.refs)
			}),
			// GetData: a DROPFILES followed by the paths, as an HDROP.
			syscall.NewCallback(func(this, format, medium uintptr) uintptr {
				if !isHDrop(format) {
					return dvFormatEtc
				}
				var names []uint16
				for _, p := range data(this).paths {
					names = append(names, utf16z(p)...)
				}
				names = append(names, 0)
				const headSize, gmemZeroInit = 20, 0x40 // DROPFILES: pFiles, pt, fNC, fWide
				h, _, _ := procGlobalAlloc.Call(gmemMoveable|gmemZeroInit, uintptr(headSize+2*len(names)))
				p, _, _ := procGlobalLock.Call(h)
				*(*uint32)(native(p)) = headSize
				*(*uint32)(unsafe.Add(native(p), 16)) = 1 // fWide
				copy(unsafe.Slice((*uint16)(unsafe.Add(native(p), headSize)), len(names)), names)
				procGlobalUnlock.Call(h)
				*(*stgMedium)(native(medium)) = stgMedium{Tymed: tymedHGlobal, Handle: h}
				return sOK
			}),
			notImpl,
			syscall.NewCallback(func(this, format uintptr) uintptr { // QueryGetData
				if isHDrop(format) {
					return sOK
				}
				return dvFormatEtc
			}),
			notImpl, notImpl, notImpl, notImpl, notImpl, notImpl,
		}
	}
	d := &testDataObject{vtbl: &testDataVtbl, refs: 1, paths: paths}
	p := uintptr(unsafe.Pointer(d))
	testDataObjects[p] = d
	return p
}

// TestDropFiles drags files from Explorer to (x, y), in DIPs, of a window
// showing native UI and drops them there, through its drop target, and
// reports whether the content took them over there and when dropped.
func TestDropFiles(hwnd uintptr, x, y float64, paths []string) (over, dropped bool) {
	s := surfaceByHandle(hwnd)
	if s == nil || s.dropTarget == 0 {
		return false, false
	}
	data := newTestDataObject(paths)
	defer release(data)
	scale := float64(s.dpi()) / 96
	pt := point{int32(x * scale), int32(y * scale)}
	procClientToScreen.Call(s.hwnd, uintptr(unsafe.Pointer(&pt)))
	pl := uintptr(uint32(pt.X)) | uintptr(uint32(pt.Y))<<32 // a POINTL, passed by value
	const all = dropEffectCopy | 2 | dropEffectLink         // and DROPEFFECT_MOVE, as Explorer allows
	effect := uint32(all)
	comCall(s.dropTarget, 3, data, 0, pl, uintptr(unsafe.Pointer(&effect))) // DragEnter
	effect = all
	comCall(s.dropTarget, 4, 0, pl, uintptr(unsafe.Pointer(&effect))) // DragOver
	over = effect == dropEffectCopy
	effect = all
	comCall(s.dropTarget, 6, data, 0, pl, uintptr(unsafe.Pointer(&effect))) // Drop
	return over, effect == dropEffectCopy
}

// TestDropData uses the production IDataObject and OLE destination with
// no process-local token, exercising serialized data and delayed rendering.
func TestDropData(hwnd uintptr, x, y float64, d transfer.Data, ops transfer.Operation) (operation transfer.Operation, dropped bool) {
	s := surfaceByHandle(hwnd)
	if s == nil {
		return
	}
	data := newDragData(platform.DragRequest{Data: d.Snapshot()})
	defer release(data)
	scale := float64(s.dpi()) / 96
	pt := point{int32(x * scale), int32(y * scale)}
	procClientToScreen.Call(s.hwnd, uintptr(unsafe.Pointer(&pt)))
	pl := uintptr(uint32(pt.X)) | uintptr(uint32(pt.Y))<<32
	effect := uint32(ops)
	comCall(s.dropTarget, 3, data, 0, pl, uintptr(unsafe.Pointer(&effect)))
	operation = transfer.Operation(effect)
	effect = uint32(ops)
	comCall(s.dropTarget, 6, data, 0, pl, uintptr(unsafe.Pointer(&effect)))
	dropped = effect != 0
	return
}

// TestDragFiles drags files from another application to (x, y), in DIPs,
// of a window showing native UI and drops them there, as OLE does: through
// the window's drop target, with the production IDataObject, asking the
// target again while the window runs until it takes them, as the page of
// a web view answers later, for a few seconds. It reports whether the
// drop took them.
func TestDragFiles(hwnd uintptr, x, y float64, paths []string) (dropped bool) {
	s := surfaceByHandle(hwnd)
	files, err := transfer.FileData(paths...)
	if s == nil || err != nil {
		return false
	}
	data := newDragData(platform.DragRequest{Data: files.Snapshot()})
	defer release(data)
	scale := float64(s.dpi()) / 96
	pt := point{int32(x * scale), int32(y * scale)}
	procClientToScreen.Call(s.hwnd, uintptr(unsafe.Pointer(&pt)))
	pl := uintptr(uint32(pt.X)) | uintptr(uint32(pt.Y))<<32
	const all = dropEffectCopy | dropEffectMove | dropEffectLink
	effect := uint32(all)
	comCall(s.dropTarget, 3, data, 0, pl, uintptr(unsafe.Pointer(&effect))) // DragEnter
	for end := time.Now().Add(5 * time.Second); effect == 0 && time.Now().Before(end); {
		theBackend.Step()
		effect = all
		comCall(s.dropTarget, 4, 0, pl, uintptr(unsafe.Pointer(&effect))) // DragOver
	}
	if effect == 0 {
		comCall(s.dropTarget, 5) // DragLeave
		return false
	}
	effect = all
	comCall(s.dropTarget, 6, data, 0, pl, uintptr(unsafe.Pointer(&effect))) // Drop
	return effect != 0
}

// TestAccessNode is an element of native UI as UI Automation reads it: the
// name of its control type, its name, and its value.
type TestAccessNode struct {
	Role, Label, Value string
	simple             uintptr
}

var uiaControlTypeNames = map[int32]string{50000: "Button", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit",
	50005: "Hyperlink", 50006: "Image", 50008: "List", 50012: "ProgressBar", 50013: "RadioButton", 50015: "Slider",
	50020: "Text", 50022: "ToolTip", 50026: "Group", 50033: "Pane", 50007: "ListItem", 50029: "DataItem", 50036: "Table"}

// uiaPattern returns an element's provider of a pattern, or 0.
func uiaPattern(simple uintptr, pattern int) uintptr {
	var p uintptr
	comCall(simple, 4, uintptr(pattern), uintptr(unsafe.Pointer(&p))) // GetPatternProvider
	return p
}

func uiaProperty(simple uintptr, property int) variant {
	var v variant
	comCall(simple, 5, uintptr(property), uintptr(unsafe.Pointer(&v))) // GetPropertyValue
	return v
}

func takeBSTR(b uintptr) string {
	s := wstr(b)
	procSysFreeString.Call(b)
	return s
}

// TestAccessibility reads the elements of a window showing native UI as UI
// Automation's clients do, in order, and reports false when it cannot.
func TestAccessibility(hwnd uintptr) ([]TestAccessNode, bool) {
	s := surfaceByHandle(hwnd)
	if s == nil {
		return nil, false
	}
	var nodes []TestAccessNode
	var walk func(fragment uintptr)
	walk = func(fragment uintptr) {
		var child uintptr
		comCall(fragment, 3, 3, uintptr(unsafe.Pointer(&child))) // Navigate(FirstChild)
		for child != 0 {
			simple := queryInterface(child, &uiaIIDs[ifaceSimple])
			n := TestAccessNode{simple: simple}
			n.Role = uiaControlTypeNames[int32(uiaProperty(simple, uiaControlTypeProperty).Val)]
			if v := uiaProperty(simple, uiaNameProperty); v.VT == vtBSTR {
				n.Label = takeBSTR(uintptr(v.Val))
			}
			if p := uiaPattern(simple, 10015); p != 0 { // Toggle
				var state int32
				comCall(p, 4, uintptr(unsafe.Pointer(&state)))
				n.Value = strconv.Itoa(int(state))
				release(p)
			} else if p := uiaPattern(simple, 10002); p != 0 { // Value
				var b uintptr
				comCall(p, 4, uintptr(unsafe.Pointer(&b)))
				n.Value = takeBSTR(b)
				release(p)
			} else if p := uiaPattern(simple, 10003); p != 0 { // RangeValue
				var v float64
				comCall(p, 4, uintptr(unsafe.Pointer(&v)))
				n.Value = strconv.FormatFloat(v, 'g', -1, 64)
				release(p)
			} else if n.Role == "Text" {
				n.Value = n.Label
			}
			nodes = append(nodes, n)
			walk(child)
			var next uintptr
			comCall(child, 3, 1, uintptr(unsafe.Pointer(&next))) // Navigate(NextSibling)
			release(simple)
			release(child)
			child = next
		}
	}
	walk(s.accessRoot().ptr(ifaceFragment))
	return nodes, true
}

// TestAccessibilityPerform acts on the element labeled label as UI
// Automation's clients do: "press", "increment", "decrement", "focus", or
// "value" to set its text to value. It reports whether the element can.
func TestAccessibilityPerform(hwnd uintptr, label, action, value string) bool {
	nodes, _ := TestAccessibility(hwnd)
	for _, n := range nodes {
		if n.Label != label {
			continue
		}
		call := func(pattern, method int, args ...uintptr) bool {
			p := uiaPattern(n.simple, pattern)
			if p == 0 {
				return false
			}
			defer release(p)
			return comCall(p, method, args...) == sOK
		}
		switch action {
		case "press":
			// Invoke, Toggle, SelectionItem's Select, or ExpandCollapse's
			// Expand.
			return call(10000, 3) || call(10015, 3) || call(10010, 3) || call(10005, 3)
		case "increment", "decrement":
			p := uiaPattern(n.simple, 10003)
			if p == 0 {
				return false
			}
			defer release(p)
			var v float64
			comCall(p, 4, uintptr(unsafe.Pointer(&v)))
			if action == "increment" {
				v++
			} else {
				v--
			}
			if runtime.GOARCH == "amd64" {
				// Go's calls set the XMM registers too, where SetValue
				// takes its double.
				return comCall(p, 3, uintptr(math.Float64bits(v))) == sOK
			}
			return uiaOf(p).setRangeValue(v) == sOK
		case "focus":
			f := queryInterface(n.simple, &uiaIIDs[ifaceFragment])
			defer release(f)
			return comCall(f, 7) == sOK // SetFocus
		case "value":
			text := u16(value)
			defer runtime.KeepAlive(text)
			return call(10002, 3, uintptr(unsafe.Pointer(text)))
		}
		return false
	}
	return false
}

// TestSurfacePixel returns the color the screen shows at (x, y), in DIPs,
// in a window showing native UI, once what was drawn shows: there, in a
// window without a redirection bitmap, the frames over the material.
func TestSurfacePixel(hwnd uintptr, x, y float64) (r, g, b uint8, ok bool) {
	s := surfaceByHandle(hwnd)
	if s == nil {
		return 0, 0, 0, false
	}
	dwmapi.NewProc("DwmFlush").Call()
	scale := float64(s.dpi()) / 96
	pt := point{int32(math.Round(x * scale)), int32(math.Round(y * scale))}
	procClientToScreen.Call(s.hwnd, uintptr(unsafe.Pointer(&pt)))
	screen, _, _ := procGetDC.Call(0)
	defer procReleaseDC.Call(0, screen)
	px, _, _ := gdi32.NewProc("GetPixel").Call(screen, uintptr(pt.X), uintptr(pt.Y))
	if px == 0xFFFFFFFF { // CLR_INVALID
		return 0, 0, 0, false
	}
	return uint8(px), uint8(px >> 8), uint8(px >> 16), true
}
