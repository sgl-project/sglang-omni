package ui

import time "time"

// InputKind is the kind of an InputEvent.
type InputKind uint8

// Kinds of input.
const (
	// InputKeyDown and InputKeyUp report a key with the modifiers held;
	// Repeat marks the auto-repeat of a key held down. A key that types
	// text comes before the text, an InputText event, on macOS and
	// Windows; on Linux, only the text comes while the element takes text
	// (TextCaret) with input methods on (InputMethod).
	InputKeyDown InputKind = iota + 1
	InputKeyUp
	// InputText is text typed, or committed by an input method.
	InputText
	// InputCompose is the composition of an input method, with its caret
	// at rune Caret; an empty Text ends it.
	InputCompose
	// InputCommand is an edit command of the menus, such as Copy of the
	// Edit menu: "copy", "cut", "paste", "pasteAndMatchStyle" (which
	// comes again as "paste" when the element doesn't take it),
	// "selectAll", "undo", "redo" or "delete", or one of the Format menu,
	// as Bold: "bold", "italic", "underline", "strikethrough",
	// "alignLeft", "alignCenter" or "alignRight".
	InputCommand
	// InputPointerDown and InputPointerUp report Button pressed or
	// released at X, Y, with Clicks the count of quick successive presses
	// (2 for a double click). Once the element takes an InputPointerDown,
	// the pointer's moves and its release come to it wherever they are,
	// and the release comes when the window loses the keyboard too.
	InputPointerDown
	InputPointerUp
	// InputPointerMove reports the pointer at X, Y, with Button held, or
	// -1 when none is.
	InputPointerMove
	// InputScroll scrolls by DX, DY DIPs at X, Y; Precise marks
	// touchpads, which scroll by pixels rather than by lines.
	InputScroll
	// InputSelect selects runes Start to End of the text an element
	// describes to assistive technology (Element.Access), as a screen
	// reader moves the selection of a text field of the app's own.
	InputSelect
)

// InputEvent is input an element takes as it comes (HandleInput).
type InputEvent struct {
	Kind InputKind
	Key  Key
	// PhysicalKey is Key by position on an ANSI US keyboard, whatever the
	// layout types: the S key is KeyS on a Cyrillic layout too, where Key
	// is KeyUnknown. KeyUnknown where the platform does not report it.
	PhysicalKey Key
	Mods        Modifiers
	Repeat      bool
	// Text of InputText, InputCompose and InputCommand, and Caret the
	// rune of an InputCompose's caret.
	Text  string
	Caret int
	// X and Y locate the pointer relative to the element's box.
	X, Y float32
	// Button is 0 for the primary button, 1 the secondary and 2 the
	// middle.
	Button int
	Clicks int
	// DX and DY are what InputScroll scrolls by, in local axes; positive DY moves the view
	// down the content.
	DX, DY  float32
	Precise bool
	// Start and End are the runes InputSelect selects.
	Start, End int
	// Time is when the event came, by the clock of the window's frames
	// (Context.Now), which a Tester's clock is in tests.
	Time time.Time
}

// HandleInput has fn take the element's input as it comes, on the main
// thread, before the next frame is built: keys, text and edit commands
// while the element has the keyboard focus, the pointer pressed on it,
// moving over it or scrolling over it. fn reports whether it took the
// event; one it leaves goes on as it would without fn, to shortcuts, Tab,
// context menus and scroll containers. Keys that the window or an element
// around the focus handles as a Shortcut go to the shortcut first. A frame
// follows the events fn takes.
//
// It is for widgets that need every key as it is pressed, such as a
// terminal; most widgets ask about their input as they are built instead
// (Clicked, Shortcut, Dragged).
func (e *node) HandleInput(fn func(ev InputEvent) bool) *node {
	e.inputFn = fn
	return e
}

// TextCaret has the element take text from the system's input methods
// while it has the keyboard focus, with their composition at r, the caret,
// relative to the element's box: candidate windows show there. Its
// InputText and Compose events (HandleInput) bring the text.
func (e *node) TextCaret(r Rect) *node {
	e.caret, e.takesText = r, true
	return e
}

// InputMethod sets whether the system's input methods compose text for the
// element while it has the keyboard focus, as they do by default for one
// that takes text (HandleTextInput, TextCaret). Turned off, keys type the
// characters of the keyboard's layout, which come as InputText, and
// nothing is composed: a mode that reads letters as commands, as Vim's
// normal mode, gets j as j with a Chinese input method on, as Neovide and
// winit's set_ime_allowed(false) have it. Browsers cannot turn input
// methods off, and compose still.
func (e *node) InputMethod(on bool) *node {
	e.noIME = !on
	return e
}

// handler returns the state of the innermost element of chain that
// handles its input.
func (rt *engine) handler(chain []uint64) *state {
	for _, id := range chain {
		if s := rt.states[id]; s != nil && s.input != nil && s.flags&flagDisabled == 0 {
			return s
		}
	}
	return nil
}

// deliver gives ev to the element of s, with the pointer's position
// relative to its box, and reports whether it took it.
func (rt *engine) deliver(s *state, ev InputEvent) bool {
	if s == nil || s.input == nil {
		return false
	}
	ev.X, ev.Y = s.local(rt.pointerX, rt.pointerY)
	ev.Time = rt.now()
	if ev.Kind == InputScroll {
		ev.DX, ev.DY = s.vector(rt.pointerX, rt.pointerY, ev.DX, ev.DY)
	}
	if !s.input(ev) {
		return false
	}
	rt.requestFrame()
	return true
}

// focusHandler returns the state of the focused element when it handles
// its input.
func (rt *engine) focusHandler() *state {
	if s := rt.states[rt.focused]; s != nil && s.input != nil && rt.windowFocus() {
		return s
	}
	return nil
}
