package deliver

import (
	"cmp"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

type fakeIO struct {
	focus    []proto.FocusResult // one per call; the last repeats
	focusErr error
	paste    proto.PasteResult
	pasteErr error
	pasted   []proto.PasteParams
	calls    int
}

func (f *fakeIO) Focus(context.Context) (proto.FocusResult, error) {
	if f.focusErr != nil {
		return proto.FocusResult{}, f.focusErr
	}
	r := f.focus[min(f.calls, len(f.focus)-1)]
	f.calls++
	return r, nil
}

func (f *fakeIO) Paste(_ context.Context, p proto.PasteParams) (proto.PasteResult, error) {
	f.pasted = append(f.pasted, p)
	return f.paste, f.pasteErr
}

var (
	slack    = proto.App{PID: 812, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}
	textEdit = proto.App{PID: 530, BundleID: "com.apple.TextEdit", Name: "TextEdit"}
)

func TestInsert(t *testing.T) {
	textArea := proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}
	elsewhere := proto.FocusResult{PID: 530, App: &textEdit, Role: "AXTextArea", Editable: true}
	posted := proto.PasteResult{Posted: true, Restored: true}
	for _, tc := range []struct {
		name     string
		req      Request
		io       fakeIO
		inserted bool
		reason   Reason
		pastes   int
		activate bool
		to       int // the paste's targetPid (812 when 0)
	}{
		{name: "pastes", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{textArea}, paste: posted}, inserted: true, pastes: 1},
		{name: "empty", req: Request{Target: slack}, io: fakeIO{focus: []proto.FocusResult{textArea}}, reason: ReasonEmpty},
		{name: "start app unknown: where the user types", req: Request{Text: "hi"}, io: fakeIO{focus: []proto.FocusResult{elsewhere}, paste: posted}, inserted: true, pastes: 1, to: 530},
		{name: "switched to another app's text field", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{elsewhere}, paste: posted}, inserted: true, pastes: 1, to: 530},
		{name: "switched to an app with nothing focused", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 530, Error: "no focused element"}}}, reason: ReasonNotEditable},
		{name: "switched to a web page being read", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 530, Role: "AXWebArea"}}}, reason: ReasonNotEditable},
		{name: "switched, the card's Insert still needs a field", req: Request{Text: "hi", Target: slack, AnyEditable: true}, io: fakeIO{focus: []proto.FocusResult{{PID: 530, Role: "AXButton"}}}, reason: ReasonNotEditable},
		{name: "switched, the card's Insert", req: Request{Text: "hi", Target: slack, AnyEditable: true}, io: fakeIO{focus: []proto.FocusResult{elsewhere}, paste: posted}, inserted: true, pastes: 1, to: 530},
		{name: "a password field", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXTextField", Subrole: "AXSecureTextField", Editable: true}}}, reason: ReasonSecure},
		{name: "a password field, the user chose", req: Request{Text: "hi", Target: slack, AnyEditable: true}, io: fakeIO{focus: []proto.FocusResult{{PID: 530, Role: "AXTextField", Subrole: "AXSecureTextField", Editable: true}}}, reason: ReasonSecure},
		{name: "nothing focused", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 812}}}, reason: ReasonNotEditable},
		{name: "a button", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXButton"}}}, reason: ReasonNotEditable},
		{name: "a button, the user chose", req: Request{Text: "hi", Target: slack, AnyEditable: true}, io: fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXButton"}}, paste: posted}, inserted: true, pastes: 1},
		{name: "a web area may be an editor", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXWebArea"}}, paste: posted}, inserted: true, pastes: 1},
		{name: "a selection needs an editable element", req: Request{Text: "hi", Target: slack, Replace: true}, io: fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXWebArea"}}}, reason: ReasonNotEditable},
		{name: "an editable selection", req: Request{Text: "hi", Target: slack, Replace: true}, io: fakeIO{focus: []proto.FocusResult{textArea}, paste: posted}, inserted: true, pastes: 1},
		{name: "a selection, another app in front", req: Request{Text: "hi", Target: slack, Replace: true}, io: fakeIO{focus: []proto.FocusResult{elsewhere}}, reason: ReasonNotFront},
		{name: "a selection, start app unknown", req: Request{Text: "hi", Replace: true}, io: fakeIO{focus: []proto.FocusResult{elsewhere}}, reason: ReasonNoTarget},
		{name: "AX could not tell", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{{PID: 812, Error: "-25212"}}, paste: posted}, inserted: true, pastes: 1},
		{name: "front changed before the paste", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{textArea}, paste: proto.PasteResult{Front: &proto.App{PID: 530, Name: "TextEdit"}}}, reason: ReasonNotFront, pastes: 1},
		{name: "no accessibility", req: Request{Text: "hi", Target: slack}, io: fakeIO{focusErr: &proto.Error{Code: proto.CodePermissionDenied, Message: "no"}}, reason: ReasonNoAccess},
		{name: "paste refused", req: Request{Text: "hi", Target: slack}, io: fakeIO{focus: []proto.FocusResult{textArea}, pasteErr: &proto.Error{Code: proto.CodePermissionDenied}}, reason: ReasonNoAccess, pastes: 1},
		{name: "helper gone", req: Request{Text: "hi", Target: slack}, io: fakeIO{focusErr: errors.New("helper exited")}, reason: ReasonFailed},
		{name: "VoxtNew in front: activate the target", req: Request{Text: "hi", Target: slack, Self: 77, Activate: func(int) bool { return true }},
			io: fakeIO{focus: []proto.FocusResult{{PID: 77, Role: "AXTextField", Editable: true}, textArea}, paste: posted}, inserted: true, pastes: 1, activate: true},
		{name: "VoxtNew in front, activation failed: never into its own field", req: Request{Text: "hi", Target: slack, Self: 77, Activate: func(int) bool { return false }},
			io: fakeIO{focus: []proto.FocusResult{{PID: 77, Role: "AXTextField", Editable: true}}}, reason: ReasonNotFront},
		{name: "VoxtNew in front, no start app", req: Request{Text: "hi", Self: 77, Activate: func(int) bool { return true }},
			io: fakeIO{focus: []proto.FocusResult{{PID: 77, Role: "AXTextField", Editable: true}}}, reason: ReasonNotFront},
	} {
		io := tc.io
		res := Insert(context.Background(), &io, tc.req)
		if res.Inserted != tc.inserted || res.Reason != tc.reason || len(io.pasted) != tc.pastes || res.Activated != tc.activate {
			t.Errorf("%s: %+v (pastes %d)", tc.name, res, len(io.pasted))
		}
		to := cmp.Or(tc.to, 812)
		if len(io.pasted) > 0 && (io.pasted[0].TargetPID != to || io.pasted[0].Text != "hi") {
			t.Errorf("%s: pasted %+v", tc.name, io.pasted[0])
		}
		if tc.inserted && res.To.PID != to {
			t.Errorf("%s: went to %+v", tc.name, res.To)
		}
	}
}

// TestInsertElsewhere: text that goes into another app is recorded with
// that app's name, and the start app's auto key is not pressed there.
func TestInsertElsewhere(t *testing.T) {
	io := &fakeIO{focus: []proto.FocusResult{{PID: 530, App: &textEdit, Role: "AXTextArea", Editable: true}}, paste: proto.PasteResult{Posted: true}}
	res := Insert(context.Background(), io, Request{Text: "x", Target: slack, Press: []proto.KeyPress{{Key: 36}}})
	if !res.Inserted || res.To != textEdit || !res.Moved(slack) || len(io.pasted[0].Press) != 0 {
		t.Errorf("elsewhere %+v, pasted %+v", res, io.pasted)
	}
	io = &fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXTextArea", Editable: true}}, paste: proto.PasteResult{Posted: true}}
	res = Insert(context.Background(), io, Request{Text: "x", Target: slack, Press: []proto.KeyPress{{Key: 36}}})
	if res.To != slack || res.Moved(slack) || len(io.pasted[0].Press) != 1 {
		t.Errorf("same app %+v, pasted %+v", res, io.pasted)
	}
	io = &fakeIO{focus: []proto.FocusResult{{PID: 530, Error: "no focused element"}}}
	res = Insert(context.Background(), io, Request{Text: "x", Target: slack})
	if res.Front.PID != 530 || res.To.PID != 0 {
		t.Errorf("not inserted %+v", res)
	}
}

func TestInsertPassesPolicy(t *testing.T) {
	io := &fakeIO{focus: []proto.FocusResult{{PID: 812, Editable: true}}, paste: proto.PasteResult{Posted: true}}
	Insert(context.Background(), io, Request{Text: "x", Target: slack, Press: []proto.KeyPress{{Key: 36}}, PressDelayMS: 200, KeepClipboard: true, RestoreDelayMS: 800})
	p := io.pasted[0]
	if len(p.Press) != 1 || p.Press[0].Key != 36 || p.PressDelayMS != 200 || !p.KeepClipboard || p.RestoreDelayMS != 800 {
		t.Errorf("params %+v", p)
	}
}

func TestCheck(t *testing.T) {
	io := &fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXTextArea", Editable: true}}}
	if res := Check(context.Background(), io, Request{Text: "x", Target: slack}); res.Reason != ReasonNone || res.Inserted || len(io.pasted) != 0 {
		t.Errorf("check %+v, pasted %d", res, len(io.pasted))
	}
	activated := false
	io = &fakeIO{focus: []proto.FocusResult{{PID: 77}}}
	res := Check(context.Background(), io, Request{Text: "x", Target: slack, Self: 77, Activate: func(int) bool { activated = true; return true }})
	if res.Reason != ReasonNotFront || activated || len(io.pasted) != 0 {
		t.Errorf("check with VoxtNew in front %+v (activated %v)", res, activated)
	}
	io = &fakeIO{focus: []proto.FocusResult{{PID: 812, Role: "AXButton"}}}
	if res := Check(context.Background(), io, Request{Text: "x", Target: slack}); res.Reason != ReasonNotEditable {
		t.Errorf("check a button %+v", res)
	}
}

// TestInsertChecked: the paste's check decides. Inserted and unknown
// (an element that tells nothing, an older helper) are inserts as
// before; unchanged is not: the card, the reason, the secure input's
// owner when it was on. The check's time goes to the helper.
func TestInsertChecked(t *testing.T) {
	area := proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true, ValueLength: 12, Selection: &proto.TextRange{Location: 12}}
	terminal := &proto.App{PID: 1702, BundleID: "com.apple.Terminal", Name: "Terminal"}
	field := &proto.PasteField{Role: "AXTextArea", Before: &proto.FieldState{Length: 12}, After: &proto.FieldState{Length: 12}}
	for _, tc := range []struct {
		name     string
		paste    proto.PasteResult
		inserted bool
		reason   Reason
		diag     string
	}{
		{name: "inserted", paste: proto.PasteResult{Posted: true, Restored: true, Check: proto.CheckInserted, CheckMS: 96,
			Field: &proto.PasteField{Role: "AXTextArea", Before: &proto.FieldState{Length: 12}, After: &proto.FieldState{Length: 20}}},
			inserted: true, diag: "check inserted after 96 ms (AXTextArea: value 12 → value 20), clipboard restored"},
		{name: "unknown", paste: proto.PasteResult{Posted: true, Restored: true, Check: proto.CheckUnknown, Field: &proto.PasteField{Role: "AXWebArea"}},
			inserted: true, diag: "check unknown (AXWebArea: unreadable → unreadable)"},
		{name: "an older helper", paste: proto.PasteResult{Posted: true, Restored: true}, inserted: true, diag: "not checked"},
		{name: "unchanged", paste: proto.PasteResult{Posted: true, Check: proto.CheckUnchanged, CheckMS: 1000, Field: field, Kept: true, PressSkipped: true},
			reason: ReasonUnconfirmed, diag: "check unchanged after 1000 ms (AXTextArea: value 12 → value 12), the clipboard keeps the text, auto key skipped"},
		{name: "unchanged, secure input on", paste: proto.PasteResult{Posted: true, Check: proto.CheckUnchanged, CheckMS: 1000, Field: field, Kept: true, SecureInput: terminal},
			reason: ReasonSecureInput, diag: "secure input on (Terminal, pid 1702)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			io := &fakeIO{focus: []proto.FocusResult{area}, paste: tc.paste}
			res := Insert(context.Background(), io, Request{Text: "hi", Target: slack, VerifyMS: DefaultVerifyMS, Press: []proto.KeyPress{{Key: 36}}})
			if res.Inserted != tc.inserted || res.Reason != tc.reason {
				t.Errorf("inserted %v, reason %q; want %v, %q", res.Inserted, res.Reason, tc.inserted, tc.reason)
			}
			if len(io.pasted) != 1 || io.pasted[0].VerifyMS != DefaultVerifyMS {
				t.Fatalf("pasted %+v: the check's time goes to the helper", io.pasted)
			}
			if !tc.inserted && res.Front.PID != slack.PID {
				t.Errorf("front %+v: the app it was meant for", res.Front)
			}
			d := res.Diag()
			if !strings.Contains(d, tc.diag) || !strings.HasPrefix(d, "focus AXTextArea in Slack (pid 812, editable, value 12, selection 12+0); ⌘V posted") {
				t.Errorf("diag %q, want it to hold %q", d, tc.diag)
			}
		})
	}
}

func TestDiagNotPasted(t *testing.T) {
	res := Insert(context.Background(), &fakeIO{focus: []proto.FocusResult{{PID: 530, App: &textEdit, Error: "no focused element"}}}, Request{Text: "hi", Target: slack})
	if d := res.Diag(); d != "focus no role in TextEdit (pid 530, value 0, no focused element); not pasted (notEditable)" {
		t.Errorf("diag %q", d)
	}
}
