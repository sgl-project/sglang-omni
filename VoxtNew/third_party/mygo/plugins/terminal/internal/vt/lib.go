// Package vt binds libghostty-vt, the terminal emulator of Ghostty, which
// it loads at run time through purego: no cgo. It mirrors the structs of
// the C API it passes, checks them against the layouts the library
// describes (ghostty_type_json) as it loads, and reads packed cells by the
// bit positions the library reports, so that a library of another version
// fails to load instead of corrupting memory.
//
// Nothing here is safe for concurrent use: callers serialize the calls on a
// Terminal and on what is made from it.
package vt

import (
	"encoding/json/v2"
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

var lib struct {
	mu     sync.Mutex
	loaded bool
	err    error
}

// Symbols called with integer and pointer arguments only, through call.
var (
	fnTerminalNew, fnTerminalFree, fnTerminalReset, fnTerminalResize, fnTerminalSet, fnTerminalGet,
	fnTerminalVTWrite, fnTerminalCompress, fnTerminalCompressionActivity uintptr

	fnRenderStateNew, fnRenderStateFree, fnRenderStateBeginUpdate, fnRenderStateEndUpdate, fnRenderStateClean,
	fnRenderStateGet, fnRowIteratorNew, fnRowIteratorFree, fnRowIteratorNext, fnRowGet,
	fnRowCellsNew, fnRowCellsFree, fnRowCellsSelect, fnRowCellsGet uintptr

	fnKeyEncoderNew, fnKeyEncoderFree, fnKeyEncoderSetopt, fnKeyEncoderSetoptFromTerminal, fnKeyEncoderEncode,
	fnKeyEventNew, fnKeyEventFree, fnKeyEventSetAction, fnKeyEventSetKey, fnKeyEventSetMods,
	fnKeyEventSetConsumedMods, fnKeyEventSetComposing, fnKeyEventSetUTF8, fnKeyEventSetUnshifted uintptr

	fnMouseEncoderNew, fnMouseEncoderFree, fnMouseEncoderSetopt, fnMouseEncoderSetoptFromTerminal,
	fnMouseEncoderReset, fnMouseEncoderEncode, fnMouseEventNew, fnMouseEventFree, fnMouseEventSetAction,
	fnMouseEventSetButton, fnMouseEventClearButton, fnMouseEventSetMods uintptr

	fnFocusEncode, fnPasteEncode, fnPasteIsSafe, fnTerminalPaste, fnFree uintptr

	fnGestureNew, fnGestureFree, fnGestureReset, fnGestureGet, fnGestureEventNew, fnGestureEventFree,
	fnGestureEventSet, fnGestureEvent, fnSelectAll uintptr

	fnGridRefHyperlinkURI, fnGridRefGraphemes, fnGridRefCell, fnTypeJSON, fnFormatterFormatAlloc, fnFormatterFree uintptr

	// Bound with registerFunc below: they take structs or floats.
	fnTerminalScrollViewport, fnTerminalGridRef, fnMouseEventSetPosition, fnSelectionFormatAlloc, fnFormatterTerminalNew uintptr
)

// Functions taking structs or floats by value, which purego passes as the
// platform's C ABI does.
var (
	terminalScrollViewport func(t uintptr, v scrollViewport)
	terminalGridRef        func(t uintptr, p point, out *GridRef) int32
	mouseEventSetPosition  func(ev uintptr, pos mousePosition)
	selectionFormatAlloc   func(t uintptr, alloc uintptr, opts selectionFormatOptions, out *uintptr, n *uintptr) int32
	formatterTerminalNew   func(alloc uintptr, out *uintptr, t uintptr, opts formatterOptions) int32
)

var symbols = []struct {
	name string
	addr *uintptr
}{
	{"ghostty_terminal_new", &fnTerminalNew},
	{"ghostty_terminal_free", &fnTerminalFree},
	{"ghostty_terminal_reset", &fnTerminalReset},
	{"ghostty_terminal_resize", &fnTerminalResize},
	{"ghostty_terminal_set", &fnTerminalSet},
	{"ghostty_terminal_get", &fnTerminalGet},
	{"ghostty_terminal_vt_write", &fnTerminalVTWrite},
	{"ghostty_terminal_compress", &fnTerminalCompress},
	{"ghostty_terminal_compression_activity", &fnTerminalCompressionActivity},
	{"ghostty_terminal_scroll_viewport", &fnTerminalScrollViewport},
	{"ghostty_terminal_grid_ref", &fnTerminalGridRef},
	{"ghostty_render_state_new", &fnRenderStateNew},
	{"ghostty_render_state_free", &fnRenderStateFree},
	{"ghostty_render_state_begin_update", &fnRenderStateBeginUpdate},
	{"ghostty_render_state_end_update", &fnRenderStateEndUpdate},
	{"ghostty_render_state_clean", &fnRenderStateClean},
	{"ghostty_render_state_get", &fnRenderStateGet},
	{"ghostty_render_state_row_iterator_new", &fnRowIteratorNew},
	{"ghostty_render_state_row_iterator_free", &fnRowIteratorFree},
	{"ghostty_render_state_row_iterator_next", &fnRowIteratorNext},
	{"ghostty_render_state_row_get", &fnRowGet},
	{"ghostty_render_state_row_cells_new", &fnRowCellsNew},
	{"ghostty_render_state_row_cells_free", &fnRowCellsFree},
	{"ghostty_render_state_row_cells_select", &fnRowCellsSelect},
	{"ghostty_render_state_row_cells_get", &fnRowCellsGet},
	{"ghostty_key_encoder_new", &fnKeyEncoderNew},
	{"ghostty_key_encoder_free", &fnKeyEncoderFree},
	{"ghostty_key_encoder_setopt", &fnKeyEncoderSetopt},
	{"ghostty_key_encoder_setopt_from_terminal", &fnKeyEncoderSetoptFromTerminal},
	{"ghostty_key_encoder_encode", &fnKeyEncoderEncode},
	{"ghostty_key_event_new", &fnKeyEventNew},
	{"ghostty_key_event_free", &fnKeyEventFree},
	{"ghostty_key_event_set_action", &fnKeyEventSetAction},
	{"ghostty_key_event_set_key", &fnKeyEventSetKey},
	{"ghostty_key_event_set_mods", &fnKeyEventSetMods},
	{"ghostty_key_event_set_consumed_mods", &fnKeyEventSetConsumedMods},
	{"ghostty_key_event_set_composing", &fnKeyEventSetComposing},
	{"ghostty_key_event_set_utf8", &fnKeyEventSetUTF8},
	{"ghostty_key_event_set_unshifted_codepoint", &fnKeyEventSetUnshifted},
	{"ghostty_mouse_encoder_new", &fnMouseEncoderNew},
	{"ghostty_mouse_encoder_free", &fnMouseEncoderFree},
	{"ghostty_mouse_encoder_setopt", &fnMouseEncoderSetopt},
	{"ghostty_mouse_encoder_setopt_from_terminal", &fnMouseEncoderSetoptFromTerminal},
	{"ghostty_mouse_encoder_reset", &fnMouseEncoderReset},
	{"ghostty_mouse_encoder_encode", &fnMouseEncoderEncode},
	{"ghostty_mouse_event_new", &fnMouseEventNew},
	{"ghostty_mouse_event_free", &fnMouseEventFree},
	{"ghostty_mouse_event_set_action", &fnMouseEventSetAction},
	{"ghostty_mouse_event_set_button", &fnMouseEventSetButton},
	{"ghostty_mouse_event_clear_button", &fnMouseEventClearButton},
	{"ghostty_mouse_event_set_mods", &fnMouseEventSetMods},
	{"ghostty_mouse_event_set_position", &fnMouseEventSetPosition},
	{"ghostty_focus_encode", &fnFocusEncode},
	{"ghostty_paste_encode", &fnPasteEncode},
	{"ghostty_terminal_paste", &fnTerminalPaste},
	{"ghostty_paste_is_safe", &fnPasteIsSafe},
	{"ghostty_free", &fnFree},
	{"ghostty_selection_gesture_new", &fnGestureNew},
	{"ghostty_selection_gesture_free", &fnGestureFree},
	{"ghostty_selection_gesture_reset", &fnGestureReset},
	{"ghostty_selection_gesture_get", &fnGestureGet},
	{"ghostty_selection_gesture_event_new", &fnGestureEventNew},
	{"ghostty_selection_gesture_event_free", &fnGestureEventFree},
	{"ghostty_selection_gesture_event_set", &fnGestureEventSet},
	{"ghostty_selection_gesture_event", &fnGestureEvent},
	{"ghostty_terminal_select_all", &fnSelectAll},
	{"ghostty_terminal_selection_format_alloc", &fnSelectionFormatAlloc},
	{"ghostty_grid_ref_hyperlink_uri", &fnGridRefHyperlinkURI},
	{"ghostty_grid_ref_graphemes", &fnGridRefGraphemes},
	{"ghostty_grid_ref_cell", &fnGridRefCell},
	{"ghostty_type_json", &fnTypeJSON},
	{"ghostty_formatter_terminal_new", &fnFormatterTerminalNew},
	{"ghostty_formatter_format_alloc", &fnFormatterFormatAlloc},
	{"ghostty_formatter_free", &fnFormatterFree},
}

// Load loads the library at path. Only the first call loads one: later
// calls report how it went, whatever their path.
func Load(path string) error {
	lib.mu.Lock()
	defer lib.mu.Unlock()
	if lib.loaded || lib.err != nil {
		return lib.err
	}
	lib.err = load(path)
	lib.loaded = lib.err == nil
	return lib.err
}

func load(path string) error {
	h, err := open(path)
	if err != nil {
		return fmt.Errorf("loading %s: %w", path, err)
	}
	for _, s := range symbols {
		if *s.addr, err = sym(h, s.name); err != nil {
			return fmt.Errorf("%s is not a libghostty-vt this package binds: %w", path, err)
		}
	}
	if err := checkLayouts(); err != nil {
		return fmt.Errorf("%s is not a libghostty-vt this package binds: %w", path, err)
	}
	registerFunc(&terminalScrollViewport, fnTerminalScrollViewport)
	registerFunc(&terminalGridRef, fnTerminalGridRef)
	registerFunc(&mouseEventSetPosition, fnMouseEventSetPosition)
	registerFunc(&selectionFormatAlloc, fnSelectionFormatAlloc)
	registerFunc(&formatterTerminalNew, fnFormatterTerminalNew)
	makeCallbacks()
	return nil
}

// Loaded reports whether Load succeeded.
func Loaded() bool {
	lib.mu.Lock()
	defer lib.mu.Unlock()
	return lib.loaded
}

// Result is a GhosttyResult other than GHOSTTY_SUCCESS.
type Result int32

const (
	success     Result = 0
	OutOfMemory Result = -1
	Invalid     Result = -2
	OutOfSpace  Result = -3
	NoValue     Result = -4
	IOError     Result = -5
	Exceeded    Result = -6
	Rejected    Result = -7
)

func (r Result) Error() string {
	switch r {
	case OutOfMemory:
		return "libghostty-vt: out of memory"
	case Invalid:
		return "libghostty-vt: invalid value"
	case OutOfSpace:
		return "libghostty-vt: out of space"
	case NoValue:
		return "libghostty-vt: no value"
	case IOError:
		return "libghostty-vt: I/O error"
	case Exceeded:
		return "libghostty-vt: limit exceeded"
	case Rejected:
		return "libghostty-vt: rejected"
	}
	return fmt.Sprintf("libghostty-vt: error %d", int32(r))
}

// result converts what a function returning GhosttyResult returned.
func result(r uintptr) error {
	if v := Result(int32(r)); v != success {
		return v
	}
	return nil
}

// ok reports whether a function returning GhosttyResult succeeded.
func ok(r uintptr) bool { return int32(r) == 0 }

// cbool reads a C bool returned in a register, whose upper bits are
// undefined.
func cbool(r uintptr) bool { return byte(r) != 0 }

func b2u(b bool) uintptr {
	if b {
		return 1
	}
	return 0
}

// mem converts an address the library gave into a pointer.
func mem(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

// goString copies a string the library owns.
func goString(p, n uintptr) string {
	if p == 0 || n == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(mem(p)), n))
}

// cStringAt copies a NUL-terminated string the library owns.
func cStringAt(p uintptr) string {
	n := uintptr(0)
	for *(*byte)(unsafe.Add(mem(p), n)) != 0 {
		n++
	}
	return goString(p, n)
}

// cell is the layout of the packed GhosttyCell, from the manifest.
var cell struct {
	tag, codepoint, styleID, wide, hyperlink bitField
	// paletteIndex, r, g and b are the content of cells holding only a
	// background color.
	paletteIndex, r, g, b bitField
}

type bitField struct{ lsb, width uint }

func (f bitField) of(v uint64) uint64 { return v >> f.lsb & (1<<f.width - 1) }

// manifest is what ghostty_type_json describes.
type manifest struct {
	ABI struct {
		PointerSize int `json:"pointer_size"`
	} `json:"abi"`
	Types map[string]typeInfo `json:"types"`
}

type typeInfo struct {
	Kind   string               `json:"kind"`
	Size   uintptr              `json:"size"`
	Fields map[string]fieldInfo `json:"fields"`
	Bits   map[string]bitInfo   `json:"bits"`
	Values map[string]int64     `json:"values"`
}

// enums are values of enums this package hardcodes, which the manifest
// must agree with.
var enums = []struct {
	typ, name string
	value     int64
}{
	{"GhosttyKey", "A", int64(KeyA)},
	{"GhosttyKey", "ARROW_UP", int64(KeyArrowUp)},
	{"GhosttyKey", "ESCAPE", int64(KeyEscape)},
	{"GhosttyKey", "F1", int64(KeyF1)},
	{"GhosttyKeyAction", "REPEAT", int64(KeyRepeat)},
	{"GhosttyMouseAction", "MOTION", int64(MouseMotion)},
	{"GhosttyMouseButton", "FIVE", WheelDown},
	{"GhosttyRenderStateCursorVisualStyle", "BLOCK_HOLLOW", int64(CursorBlockHollow)},
	{"GhosttyRenderStateDirty", "FULL", int64(Full)},
	{"GhosttyCellWide", "SPACER_HEAD", int64(SpacerHead)},
	{"GhosttyStyleColorTag", "RGB", colorRGB},
	{"GhosttySgrUnderline", "DASHED", UnderlineDashed},
	{"GhosttyTerminalScrollViewportTag", "ROW", 3},
	{"GhosttyPointTag", "VIEWPORT", 1},
	{"GhosttyClipboardWriteResult", "UNSUPPORTED", 2},
	{"GhosttySelectionGestureEventType", "DRAG", gestureDrag},
}

type fieldInfo struct {
	Offset uintptr `json:"offset"`
}

type bitInfo struct {
	LSB   uint                `json:"lsb"`
	Width uint                `json:"width"`
	Arms  map[string]*bitArms `json:"arms"`
}

type bitArms struct {
	Bits map[string]bitInfo `json:"bits"`
}

// layout is a struct this package mirrors: its size and the offsets of
// its fields as Go lays them out.
type layout struct {
	size   uintptr
	fields map[string]uintptr
}

// checkLayouts compares the structs of this package with the layouts the
// library reports, and reads the bit positions of packed cells.
func checkLayouts() error {
	p := call(fnTypeJSON)
	if p == 0 {
		return errors.New("no type manifest")
	}
	var m manifest
	if err := json.Unmarshal([]byte(cStringAt(p)), &m); err != nil {
		return fmt.Errorf("type manifest: %w", err)
	}
	return m.check()
}

func (m *manifest) check() error {
	if m.ABI.PointerSize != int(unsafe.Sizeof(uintptr(0))) {
		return fmt.Errorf("built for %d-byte pointers", m.ABI.PointerSize)
	}
	for name, want := range layouts() {
		t, ok := m.Types[name]
		if !ok {
			return fmt.Errorf("no type %s", name)
		}
		if t.Size != want.size {
			return fmt.Errorf("%s is %d bytes, not %d", name, t.Size, want.size)
		}
		for field, off := range want.fields {
			f, ok := t.Fields[field]
			if !ok {
				return fmt.Errorf("%s has no field %s", name, field)
			}
			if f.Offset != off {
				return fmt.Errorf("%s.%s is at %d, not %d", name, field, f.Offset, off)
			}
		}
	}
	for _, e := range enums {
		if v, ok := m.Types[e.typ].Values[e.name]; !ok || v != e.value {
			return fmt.Errorf("%s %s is not %d", e.typ, e.name, e.value)
		}
	}
	c, ok := m.Types["GhosttyCell"]
	if !ok || c.Kind != "packed" || c.Size != 8 {
		return errors.New("GhosttyCell is not a packed 64-bit value")
	}
	var errs []error
	field := func(dst *bitField, name string) {
		b, ok := c.Bits[name]
		if !ok {
			errs = append(errs, fmt.Errorf("GhosttyCell has no %s", name))
		}
		*dst = bitField{b.LSB, b.Width}
	}
	arm := func(dst *bitField, tag, name string) {
		content := c.Bits["content"]
		a := content.Arms[tag]
		if a == nil {
			errs = append(errs, fmt.Errorf("GhosttyCell content has no %s", tag))
			return
		}
		b, ok := a.Bits[name]
		if !ok {
			errs = append(errs, fmt.Errorf("GhosttyCell content has no %s.%s", tag, name))
		}
		*dst = bitField{content.LSB + b.LSB, b.Width}
	}
	field(&cell.tag, "content_tag")
	field(&cell.styleID, "style_id")
	field(&cell.wide, "wide")
	field(&cell.hyperlink, "hyperlink")
	arm(&cell.codepoint, "CODEPOINT", "codepoint")
	arm(&cell.paletteIndex, "BG_COLOR_PALETTE", "index")
	arm(&cell.r, "BG_COLOR_RGB", "r")
	arm(&cell.g, "BG_COLOR_RGB", "g")
	arm(&cell.b, "BG_COLOR_RGB", "b")
	return errors.Join(errs...)
}
