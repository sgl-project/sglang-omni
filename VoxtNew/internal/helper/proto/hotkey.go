package proto

// ---- hotkey.* ----

// HotkeyBinding is one shortcut the event tap watches: a modifier-only
// chord (Key nil: fn, fn+⇧, right ⌘…) or a key with modifiers (⌃⌘V).
// The helper's recognizer decides on the tap's thread, synchronously,
// whether to swallow each key event, as macOS requires.
type HotkeyBinding struct {
	ID string `json:"id"` // the app's name for it: "transcribe/0"
	// Modifiers maps each modifier family the binding needs (Mod*) to its
	// side (Side*); families not listed must be up.
	Modifiers map[string]string `json:"modifiers"`
	// Key is the virtual key code (kVK_*, 0 is A) of a key binding.
	Key      *int   `json:"key,omitempty"`
	Behavior string `json:"behavior"` // BehaviorTap | BehaviorDoubleTap | BehaviorLongPress
	// Swallow, for a key binding, keeps its key events from the app in
	// front (default true). Modifier-only chords are never swallowed, but
	// fn with HotkeyConfigureParams.SwallowFn.
	Swallow *bool `json:"swallow,omitempty"`
}

// Modifier families and sides.
const (
	ModFn      = "fn"
	ModShift   = "shift"
	ModControl = "control"
	ModOption  = "option"
	ModCommand = "command"

	SideAny   = "any"
	SideLeft  = "left"
	SideRight = "right"
)

// Hotkey behaviours and phases.
const (
	BehaviorTap       = "tap"
	BehaviorDoubleTap = "doubleTap"
	BehaviorLongPress = "longPress"

	PhaseTrigger = "trigger" // tap or doubleTap fired; a key binding's tap fires on key down
	PhaseDown    = "down"    // longPress: held long enough
	PhaseUp      = "up"      // longPress: released after down
)

// HotkeyEscape is the binding of the hotkey.event sent for Escape while
// Active (a recording runs); Escape is then swallowed.
const HotkeyEscape = "escape"

// HotkeyConfigureParams replaces the bindings; an empty list removes the
// event tap. Changing only the timings or Active keeps a chord in progress:
// the app flips Active as a session starts and ends.
type HotkeyConfigureParams struct {
	Bindings []HotkeyBinding `json:"bindings"`
	// DoubleTapMS is the double-tap window: a second press starting
	// within it of the first release makes a double tap (default: the
	// system's double-click interval, at most 800, else 500). LongPressMS
	// is the hold that fires a long press (300), TapMaxMS the longest
	// press that is still a tap (1500: real ⇧+fn presses took up to 983 ms).
	DoubleTapMS float64 `json:"doubleTapMs,omitempty"`
	LongPressMS float64 `json:"longPressMs,omitempty"`
	TapMaxMS    float64 `json:"tapMaxMs,omitempty"`
	// SwallowFn swallows fn's own events (and the 🌐 key's that follow)
	// while an fn chord is bound, so the 🌐 action does not run.
	SwallowFn bool `json:"swallowFn,omitempty"`
	// Active: a session records. A tap fires at once (no wait for a
	// double tap), and Escape is reported and swallowed.
	Active bool `json:"active,omitempty"`
}

// HotkeyStatus answers hotkey.configure and hotkey.status, and is
// hotkey.state's params. Without Accessibility the tap cannot be
// installed: the helper retries every 2 s and sends hotkey.state when it
// gets in; meanwhile the app falls back to MyGo's Carbon shortcuts for
// plain key combinations.
type HotkeyStatus struct {
	Installed     bool `json:"installed"`
	Accessibility bool `json:"accessibility"`
	Bindings      int  `json:"bindings"`
	Events        int  `json:"events"` // keyboard events seen
	// DisabledByTimeout counts the times macOS disabled the tap (the
	// helper enables it again); MissedEvents the modifier events found
	// missing (a release macOS did not deliver), after which the chord in
	// progress was dropped.
	DisabledByTimeout int     `json:"disabledByTimeout"`
	MissedEvents      int     `json:"missedEvents"`
	MaxCallbackMicros float64 `json:"maxCallbackMicros"` // slowest tap callback
	Error             string  `json:"error,omitempty"`
}

// HotkeyEventParams reports a binding that fired.
type HotkeyEventParams struct {
	Binding  string  `json:"binding"` // HotkeyBinding.ID, or HotkeyEscape
	Phase    string  `json:"phase"`
	Behavior string  `json:"behavior"`
	HeldMS   float64 `json:"heldMs,omitempty"` // how long the chord was held, when known
	T        float64 `json:"t"`                // when it was decided, ms of the helper's monotonic clock
	// Front is the app in front as the key went down: the start of the
	// session's context snapshot.
	Front *App `json:"front,omitempty"`
}
