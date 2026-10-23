// Package deliver puts a result where the user is typing when it is
// ready: one insert transaction through the helper's text I/O. It reads
// the app in front now and its focused element (textio.focus). When that
// element takes text, the result is pasted there, whichever app it is
// (textio.paste with that app's pid: the clipboard is snapshotted,
// written, ⌘V posted, keys pressed after it, the clipboard restored
// unless someone wrote to it meanwhile). Otherwise nothing is typed and
// the caller shows the answer card, which offers Insert and Copy.
//
// The start app (Request.Target, in front when the shortcut was pressed)
// still matters in three ways: an element there is given the benefit of
// the doubt (the user pressed the shortcut in it), a selection is
// replaced only there, and its auto key is pressed only there. When
// VoxtNew itself came to the front (a click on its window), the start
// app is activated again first (macx.Activate): nothing is ever typed
// into VoxtNew, nor into a password field.
//
// The paste is checked (Request.VerifyMS): the helper reads the focused
// element before ⌘V and after it. When it reads the same up to the end of
// the check, the text did not go in: nothing is reported inserted, the
// clipboard keeps the text, the auto key is not pressed and the caller
// shows the card (ReasonUnconfirmed; ReasonSecureInput when another app
// had secure event input on, which may keep posted keys away). An element
// that tells nothing (no value nor selection, a web area) is trusted as
// before.
//
// Policy is here, the mechanics in the helper: the restore delay, the
// keys after the paste, whether the clipboard keeps the text and how
// long the check waits are params of textio.paste.
//
// Imports: helper/proto.
package deliver

import (
	"context"
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// TextIO is the helper's text I/O (textio.focus, textio.paste).
type TextIO interface {
	Focus(ctx context.Context) (proto.FocusResult, error)
	Paste(ctx context.Context, p proto.PasteParams) (proto.PasteResult, error)
}

// Reason says why a result was not inserted.
type Reason string

const (
	ReasonNone Reason = ""
	// ReasonEmpty: there was no text.
	ReasonEmpty Reason = "empty"
	// ReasonNoTarget: a selection is to be replaced, but the app it was
	// selected in is not known.
	ReasonNoTarget Reason = "noTarget"
	// ReasonNotFront: the selection's app is not in front any more, or
	// VoxtNew is in front and the start app could not be brought back, or
	// the front changed just before the paste.
	ReasonNotFront Reason = "notFront"
	// ReasonNotEditable: the focused element of the app in front takes no
	// text (or nothing has the focus).
	ReasonNotEditable Reason = "notEditable"
	// ReasonSecure: the focused element is a password field.
	ReasonSecure Reason = "secure"
	// ReasonNoAccess: VoxtNew may not post keys (Accessibility).
	ReasonNoAccess Reason = "noAccess"
	// ReasonFailed: the helper failed (Result.Err).
	ReasonFailed Reason = "failed"
	// ReasonCard: the result goes to the answer card by choice (Translate's
	// "Show the result in a card"), or always (Rewrite, Voxt's); ReasonLong
	// was 0.1.0's: an answer longer than the card's limit. The session sets
	// these, not Insert.
	ReasonCard Reason = "card"
	ReasonLong Reason = "long" // 0.1.0's (the history may hold it)
	// ReasonUnconfirmed: ⌘V was posted, but the focused element did not
	// change (the paste's check): the text is on the clipboard.
	ReasonUnconfirmed Reason = "unconfirmed"
	// ReasonSecureInput: as ReasonUnconfirmed, while another app had
	// secure event input on (Result.Paste.SecureInput).
	ReasonSecureInput Reason = "secureInput"
)

// DefaultVerifyMS is how long a paste's check waits for the focused
// element to change: a web page's text area shows it within 50–200 ms;
// a busy page may read the clipboard late, and the clipboard keeps the
// text for it anyway when the check gives up.
const DefaultVerifyMS = 1000

// Request is one insert.
type Request struct {
	Text string
	// Target is the app the session started in (hotkey.event's front).
	Target proto.App
	// Press are keys posted after the paste (app enhancement's auto
	// key), PressDelayMS after it (0: the helper's 120). They are the
	// start app's group's: pressed only when the text goes into Target.
	Press        []proto.KeyPress
	PressDelayMS int
	// KeepClipboard leaves the text on the clipboard; RestoreDelayMS
	// waits before restoring it (0: the helper's 500).
	KeepClipboard  bool
	RestoreDelayMS int
	// Self is VoxtNew's pid. When VoxtNew is in front, Activate (nil:
	// none) gives the front back to Target before pasting.
	Self     int
	Activate func(pid int) bool
	// AnyEditable skips the check of Target's focused element (the answer
	// card's Insert: the user looked and chose); another app's must
	// still take text.
	AnyEditable bool
	// Replace: the text replaces a selection made in Target. It goes
	// only there, and only into an element Accessibility calls editable
	// (text selected on a web page must not be "replaced" by a paste
	// that goes nowhere).
	Replace bool
	// VerifyMS checks the paste for up to that long (DefaultVerifyMS;
	// 0: no check).
	VerifyMS int
}

// Result is what happened.
type Result struct {
	Inserted bool
	Reason   Reason
	// To is the app the text went into (Check: would go into); Front is
	// the app in front when it was not inserted.
	To        proto.App
	Front     proto.App
	Focus     proto.FocusResult
	Paste     proto.PasteResult
	Activated bool // the target was activated again first
	Err       error
	Elapsed   time.Duration
}

// Moved reports whether the text went (or would go) into another app
// than the start app: the user switched apps.
func (r Result) Moved(target proto.App) bool {
	return r.To.PID != 0 && r.To.PID != target.PID
}

// nonText are focused roles that never take text.
var nonText = map[string]bool{
	"AXButton": true, "AXCheckBox": true, "AXRadioButton": true, "AXPopUpButton": true, "AXMenuButton": true,
	"AXMenuItem": true, "AXMenuBarItem": true, "AXImage": true, "AXList": true, "AXOutline": true, "AXTable": true,
	"AXRow": true, "AXCell": true, "AXSlider": true, "AXTabGroup": true, "AXToolbar": true, "AXDisclosureTriangle": true,
	"AXIncrementor": true, "AXColorWell": true, "AXSplitter": true, "AXStaticText": true, "AXLink": true, "AXHeading": true,
}

// Writable says whether the start app's focused element takes text: an
// editable one, or one VoxtNew cannot tell (no Accessibility answer, a
// web area, a group: contenteditable editors look so), but not nothing
// at all, nor a button, a list or another control that never does.
func Writable(f proto.FocusResult) bool {
	if f.Editable || f.Error != "" {
		return true
	}
	if f.Role == "" {
		return false
	}
	return !nonText[f.Role]
}

// Secure says whether the focused element is a password field.
func Secure(f proto.FocusResult) bool { return f.Subrole == "AXSecureTextField" }

// Insert pastes r.Text where the user is typing when it can (see the
// package comment). Call it off the main thread: it waits for the helper.
func Insert(ctx context.Context, io TextIO, r Request) Result {
	start := time.Now()
	res := insert(ctx, io, r)
	res.Elapsed = time.Since(start)
	return res
}

// Check tells, without pasting or activating anything, whether Insert
// would paste now (Result.Reason is "" when it would). Sessions ask it
// before replacing a selection, to stream the answer into the card at
// once when it cannot replace it.
func Check(ctx context.Context, io TextIO, r Request) Result {
	start := time.Now()
	r.Activate = nil
	res := check(ctx, io, r)
	res.Elapsed = time.Since(start)
	return res
}

func insert(ctx context.Context, io TextIO, r Request) Result {
	res := check(ctx, io, r)
	if res.Reason != ReasonNone {
		return res
	}
	press := r.Press
	if res.Moved(r.Target) {
		press = nil // the start app's auto key, not this app's
	}
	p, err := io.Paste(ctx, proto.PasteParams{
		Text: r.Text, KeepClipboard: r.KeepClipboard, RestoreDelayMS: r.RestoreDelayMS,
		TargetPID: res.To.PID, Press: press, PressDelayMS: r.PressDelayMS, VerifyMS: r.VerifyMS,
	})
	if err != nil {
		return failed(res, err)
	}
	res.Paste = p
	if !p.Posted {
		res.Reason = ReasonNotFront
		if p.Front != nil {
			res.Front = *p.Front
		}
		return res
	}
	if p.Check == proto.CheckUnchanged {
		res.Reason = ReasonUnconfirmed
		if p.SecureInput != nil {
			res.Reason = ReasonSecureInput
		}
		res.Front = res.To
		return res
	}
	res.Inserted = true
	return res
}

// check is Insert up to the paste.
func check(ctx context.Context, io TextIO, r Request) Result {
	var res Result
	if r.Text == "" {
		res.Reason = ReasonEmpty
		return res
	}
	if r.Replace && r.Target.PID == 0 {
		res.Reason = ReasonNoTarget
		return res
	}
	focus, err := io.Focus(ctx)
	if err != nil {
		return failed(res, err)
	}
	self := r.Self != 0 && focus.PID == r.Self
	if self && r.Target.PID != 0 && r.Target.PID != r.Self && r.Activate != nil && r.Activate(r.Target.PID) {
		res.Activated = true
		if focus, err = io.Focus(ctx); err != nil {
			return failed(res, err)
		}
		self = focus.PID == r.Self
	}
	res.Focus = focus
	front := focusApp(focus, r.Target)
	inTarget := r.Target.PID != 0 && focus.PID == r.Target.PID
	var why Reason
	switch {
	case self, r.Replace && !inTarget:
		why = ReasonNotFront
	case Secure(focus):
		why = ReasonSecure
	case r.Replace:
		if !focus.Editable {
			why = ReasonNotEditable
		}
	case inTarget:
		if !r.AnyEditable && !Writable(focus) {
			why = ReasonNotEditable
		}
	case !focus.Editable || focus.Error != "":
		// Another app: the user did not press the shortcut there, so its
		// element must say it takes text (not a web page being read).
		why = ReasonNotEditable
	}
	if why != ReasonNone {
		res.Reason, res.Front = why, front
		return res
	}
	res.To = front
	return res
}

// focusApp is the app of the focused element: the helper's description,
// or the start app's when it is that one.
func focusApp(f proto.FocusResult, target proto.App) proto.App {
	if f.PID == target.PID && target.PID != 0 && (f.App == nil || f.App.Name == "") {
		return target
	}
	if f.App != nil && f.App.PID == f.PID {
		return *f.App
	}
	return proto.App{PID: f.PID}
}

// failed records a helper error: Accessibility refused, or another one.
func failed(res Result, err error) Result {
	res.Err = err
	res.Reason = ReasonFailed
	var pe *proto.Error
	if errors.As(err, &pe) && pe.Code == proto.CodePermissionDenied {
		res.Reason = ReasonNoAccess
	}
	return res
}
