package proto

// ---- textio.* ----

// FocusResult describes the focused element of the app in front, for
// deciding whether a result can be inserted there.
type FocusResult struct {
	PID int `json:"pid"`
	// App is the app the focused element belongs to (PID's), or the app
	// in front when nothing is focused; an older helper sends none.
	App     *App   `json:"app,omitempty"`
	Role    string `json:"role,omitempty"`    // "AXTextArea", "AXTextField", "AXWebArea"…
	Subrole string `json:"subrole,omitempty"` // "AXSecureTextField": a password field
	// Editable: its value can be set, it is a text role, or it is inside
	// a web page's editable region (AXEditableAncestor). Selection is
	// the selected range (UTF-16), when it has one.
	Editable    bool       `json:"editable"`
	Selection   *TextRange `json:"selection,omitempty"`
	ValueLength int        `json:"valueLength"` // UTF-16 units of its value (0: none or unknown)
	Error       string     `json:"error,omitempty"`
}

// TextRange is a range of UTF-16 units.
type TextRange struct {
	Location int `json:"location"`
	Length   int `json:"length"`
}

// SelectionParams reads the selected text of the app in front:
// Accessibility first (the selected text, the text of the selected range,
// WebKit's and Chromium's text markers), then, with AllowCopy, ⌘C, waiting
// up to CopyTimeoutMS (default 300) for the clipboard to change, which is
// then put back. The app decides AllowCopy (not where ⌘C copies a whole
// line on an empty selection: VS Code, Cursor…).
type SelectionParams struct {
	AllowCopy     bool `json:"allowCopy,omitempty"`
	CopyTimeoutMS int  `json:"copyTimeoutMs,omitempty"`
}

// SelectionResult is the text ("" for none) and how it was read.
type SelectionResult struct {
	Text  string   `json:"text,omitempty"`
	Via   string   `json:"via"` // Via*
	MS    float64  `json:"ms"`
	Steps []string `json:"steps,omitempty"` // what was tried, for the log
}

// How a selection was read.
const (
	ViaAX       = "ax"
	ViaAXRange  = "ax-range"
	ViaAXMarker = "ax-marker"
	ViaCopy     = "copy"
	ViaNone     = "none"
)

// PasteParams inserts Text where the cursor is: the clipboard is
// snapshotted (every item and type), Text written (marked transient for
// clipboard managers), ⌘V posted, then the keys in Press, and after
// RestoreDelayMS the snapshot written back unless someone wrote to the
// clipboard meanwhile. With VerifyMS the helper checks that the text
// went in (PasteResult.Check) before it presses keys or restores.
type PasteParams struct {
	Text string `json:"text"`
	// KeepClipboard leaves Text on the clipboard instead of restoring.
	KeepClipboard  bool `json:"keepClipboard,omitempty"`
	RestoreDelayMS int  `json:"restoreDelayMs,omitempty"` // default 500
	// TargetPID, when set, must be the app in front: otherwise nothing is
	// pasted (Posted false, Front says who is in front). The app activates
	// the target itself (macx.Activate) when it has to.
	TargetPID int `json:"targetPid,omitempty"`
	// Press are keys posted after ⌘V (app enhancement's auto key: Return,
	// ⌘Return…), PressDelayMS (default 120) after it.
	Press        []KeyPress `json:"press,omitempty"`
	PressDelayMS int        `json:"pressDelayMs,omitempty"`
	// VerifyMS, when set, has the helper read the focused element (its
	// value, its length and its selection) before ⌘V and again every
	// 25 ms after it, for up to VerifyMS, until it changed. Inserted:
	// Press follows and the clipboard is restored as soon as the change
	// showed. Unchanged: Press is skipped and the clipboard keeps Text
	// (Kept). Unknown (the element tells nothing): as without VerifyMS.
	VerifyMS int `json:"verifyMs,omitempty"`
}

// KeyPress is a key (kVK_*) with modifier families (Mod*, not fn).
type KeyPress struct {
	Key       int      `json:"key"`
	Modifiers []string `json:"modifiers,omitempty"`
}

// PasteResult says what happened.
type PasteResult struct {
	Posted bool `json:"posted"`
	// Restored: the old clipboard is back. ClipboardTaken: someone wrote
	// to the clipboard before the restore, which was then skipped.
	Restored       bool    `json:"restored"`
	ClipboardTaken bool    `json:"clipboardTaken"`
	MS             float64 `json:"ms"`
	Front          *App    `json:"front,omitempty"` // when not posted: the app in front
	// Check is the verification's outcome (PasteParams.VerifyMS; "" when
	// not asked): Check*. CheckMS is when the change showed, after ⌘V
	// (inserted), or how long the helper watched (unchanged).
	Check   string  `json:"check,omitempty"`
	CheckMS float64 `json:"checkMs,omitempty"`
	// Field is the focused element the check watched, for the log.
	Field *PasteField `json:"field,omitempty"`
	// SecureInput is the app that had secure event input on when ⌘V was
	// posted (a password field, a terminal's Secure Keyboard Entry: the
	// system may keep posted keys from other apps); PID 0 when the owner
	// is unknown. Absent: it was off.
	SecureInput *App `json:"secureInput,omitempty"`
	// PressSkipped: Press was not posted (the check found no change).
	// Kept: the clipboard was left holding Text (the check found no
	// change: the user can paste it).
	PressSkipped bool `json:"pressSkipped,omitempty"`
	Kept         bool `json:"kept,omitempty"`
}

// PasteField is the focused element a paste's check watched: its role
// and its state before ⌘V and at the end of the check (absent: not
// readable).
type PasteField struct {
	Role   string      `json:"role,omitempty"`
	Before *FieldState `json:"before,omitempty"`
	After  *FieldState `json:"after,omitempty"`
}

// FieldState is what the check compares: the value's length (UTF-16)
// and the selection; the value itself and the text marker are compared
// too, by the helper only (a selection replaced by text of the same
// length; a web area's caret).
type FieldState struct {
	Length    int        `json:"length"`
	Selection *TextRange `json:"selection,omitempty"`
	// Marker: the selected text marker range was read (WebKit's and
	// Chromium's caret; a web area has only that), compared by digest.
	Marker bool `json:"marker,omitempty"`
}

// The outcomes of a paste's check.
const (
	CheckInserted  = "inserted"  // the focused element changed after ⌘V
	CheckUnchanged = "unchanged" // it did not change within VerifyMS
	CheckUnknown   = "unknown"   // it has no value or selection to read, or another element took the focus
)

// ---- browser.url ----

// BrowserURLParams asks for the front tab's URL of a browser, read
// through Accessibility only (no AppleScript, no Automation consent): the
// web area that holds the focus, the focused window's AXDocument, then the
// first web area's AXURL. PID picks the process (the hotkey's front);
// else BundleID's first running one; neither: the app in front.
type BrowserURLParams struct {
	BundleID string `json:"bundleId,omitempty"`
	PID      int    `json:"pid,omitempty"`
}

// BrowserURLResult is the URL ("" when none was read) and how. A URL is
// used only to match app enhancement rules, never sent to a model.
type BrowserURLResult struct {
	BundleID string  `json:"bundleId"`
	URL      string  `json:"url,omitempty"`
	Via      string  `json:"via"` // URLVia*
	MS       float64 `json:"ms"`
	// Error says why none was read, for the log and the App enhancement
	// page ("no focused window", "no web area").
	Error string `json:"error,omitempty"`
	// Retried: the first read found no web area in a Chromium browser, so
	// the helper set AXManualAccessibility on it (once per process) and
	// read again after a short wait; URL is the second read's.
	Retried bool `json:"retried,omitempty"`
}

// How a URL was read.
const (
	URLViaAXFocus    = "ax-focus"    // the AXURL of the web area around the focused element
	URLViaAXDocument = "ax-document" // the focused window's AXDocument
	URLViaAXWebURL   = "ax-weburl"   // the AXURL of the window's first web area
	URLViaNone       = "none"
)
