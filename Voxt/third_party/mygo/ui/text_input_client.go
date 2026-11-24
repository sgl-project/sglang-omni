package ui

import (
	"reflect"

	"github.com/egoist/mygo/internal/platform"
)

// TextInputRange is a half-open range of UTF-16 code units. These are the
// units of native text services and can be converted at the storage boundary.
// They do not prescribe the client's buffer, grapheme or selection model.
type TextInputRange = platform.TextRange

// TextInputSelection is the primary selection reported to native input
// methods. Reversed places its caret at Range.Start. An editor can keep
// additional cursors or discontiguous ranges in its own state.
type TextInputSelection = platform.TextSelection

// Point is a position in DIPs relative to an element or text layout.
type Point struct{ X, Y float32 }

// TextInputClient is an application-owned text-input target. It provides
// native input methods with document queries, composition and geometry; it
// owns the text, selection, marked range, editing policy and undo history.
// Pass a pointer to your client state. Callbacks run on the UI thread, as HandleInput does, and must not block on
// work requiring that thread. Keep the client alive across frames.
//
// All ranges count UTF-16 code units of the current document, including its
// marked text. TextForRange clamps/adjusts requested ranges to valid text and
// returns the actual range. BoundsForRange returns element-relative geometry
// and the actual range represented, or false for unavailable/offscreen text.
// IndexForPoint takes an element-relative point and returns a UTF-16 offset.
//
// A nil replacement range means the current marked range, if present,
// otherwise the selection. SetMarkedText's selected range is relative to
// the new marked text. ReplaceText commits and clears preedit; UnmarkText
// clears marked status, retaining the document's text. Selection is
// queried again after every mutation, before the next native event.
type TextInputClient interface {
	TextForRange(TextInputRange) (string, TextInputRange)
	Selection() TextInputSelection
	MarkedRange() (TextInputRange, bool)
	ReplaceText(*TextInputRange, string)
	SetMarkedText(*TextInputRange, string, TextInputRange)
	UnmarkText()
	BoundsForRange(TextInputRange) (Rect, TextInputRange, bool)
	IndexForPoint(Point) (int, bool)
}

// HandleTextInput connects an element to the system's text-input services.
// Pair it with HandleInput for keys, pointer selection and Edit-menu commands,
// and Draw for rendering. It adds focus and the text cursor without creating
// a text buffer, formatting UI or undo history. A nil client disconnects it.
// Existing TextInput/TextArea widgets and TextCaret handlers are unchanged.
func (e *node) HandleTextInput(client TextInputClient) *node {
	if client != nil {
		v := reflect.ValueOf(client)
		if v.Kind() != reflect.Pointer {
			panic("ui: HandleTextInput requires a pointer client")
		}
		if v.IsNil() {
			client = nil
		}
	}
	e.textClient = client
	if client != nil {
		e.flags |= flagFocusable | flagHover
		e.Cursor(CursorText)
	}
	return e
}

// UTF16Len returns the number of native text offsets in s.
func UTF16Len(s string) int { return platform.UTF16Len(s) }

// UTF16ByteOffset converts a native offset to a UTF-8 byte offset, rounding
// inside a surrogate pair down to the rune's beginning and clamping to s.
func UTF16ByteOffset(s string, index int) int { return platform.UTF16ByteOffset(s, max(0, index)) }

// textInputAdapter is stable for an element's lifetime. A stale native
// reference cannot edit another element after focus, disposal or reuse.
type textInputAdapter struct {
	rt *engine
	id uint64
}

func (a *textInputAdapter) client() (TextInputClient, *state) {
	if a.rt.textInputClosed || !a.rt.windowFocus() || a.rt.focused != a.id {
		return nil, nil
	}
	s := a.rt.states[a.id]
	if s == nil || s.textAdapter != a || s.textClient == nil || s.flags&flagDisabled != 0 {
		return nil, nil
	}
	return s.textClient, s
}

func (a *textInputAdapter) release() {
	if s := a.rt.states[a.id]; s != nil && s.textAdapter == a && s.textClient != nil {
		if _, marked := s.textClient.MarkedRange(); marked {
			s.textClient.UnmarkText()
		}
	}
}
func (a *textInputAdapter) TextForRange(r platform.TextRange) (string, platform.TextRange) {
	if c, _ := a.client(); c != nil {
		return c.TextForRange(platform.NormalizeTextRange(r))
	}
	return "", platform.TextRange{}
}
func (a *textInputAdapter) Selection() platform.TextSelection {
	if c, _ := a.client(); c != nil {
		s := c.Selection()
		s.Range = platform.NormalizeTextRange(s.Range)
		return s
	}
	return platform.TextSelection{}
}
func (a *textInputAdapter) MarkedRange() (platform.TextRange, bool) {
	if c, _ := a.client(); c != nil {
		r, ok := c.MarkedRange()
		return platform.NormalizeTextRange(r), ok
	}
	return platform.TextRange{}, false
}
func (a *textInputAdapter) changed() { a.rt.requestFrame(); a.rt.updateTextInput() }
func (a *textInputAdapter) ReplaceText(r *platform.TextRange, s string) {
	if c, _ := a.client(); c != nil {
		c.ReplaceText(r, s)
		a.changed()
	}
}
func (a *textInputAdapter) SetMarkedText(r *platform.TextRange, s string, selected platform.TextRange) {
	if c, _ := a.client(); c != nil {
		c.SetMarkedText(r, s, selected)
		a.changed()
	}
}
func (a *textInputAdapter) UnmarkText() {
	if c, _ := a.client(); c != nil {
		c.UnmarkText()
		a.changed()
	}
}
func (a *textInputAdapter) BoundsForRange(r platform.TextRange) (platform.RectF, platform.TextRange, bool) {
	if c, s := a.client(); c != nil {
		b, actual, ok := c.BoundsForRange(r)
		b = transformRect(s.world, Rect{X: s.x + b.X, Y: s.y + b.Y, W: b.W, H: b.H})
		return platform.RectF{X: float64(b.X), Y: float64(b.Y), W: float64(b.W), H: float64(b.H)}, actual, ok
	}
	return platform.RectF{}, platform.TextRange{}, false
}
func (a *textInputAdapter) IndexForPoint(x, y float64) (int, bool) {
	if c, s := a.client(); c != nil {
		if _, ok := s.world.Inverse(); !ok {
			return 0, false
		}
		lx, ly := s.local(float32(x), float32(y))
		return c.IndexForPoint(Point{lx, ly})
	}
	return 0, false
}
