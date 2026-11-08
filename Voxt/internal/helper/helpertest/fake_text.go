package helpertest

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"
	"unicode/utf16"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// ---- hotkeys ----

func (f *fake) hotkeyStatus() proto.HotkeyStatus {
	f.mu.Lock()
	defer f.mu.Unlock()
	st := proto.HotkeyStatus{Installed: f.installed, Accessibility: f.trusted, Bindings: f.bindings}
	if (f.bindings > 0 || f.hkCancel) && !f.trusted {
		st.Error = "CGEvent.tapCreate failed: VoxtNew may not use Accessibility; retrying every 2 s"
	}
	return st
}

func (f *fake) hotkeys(m proto.Message) {
	var p proto.HotkeyConfigureParams
	if err := m.DecodeParams(&p); err != nil {
		f.fail(m.ID, proto.CodeBadRequest, "Bad bindings: %v", err)
		return
	}
	f.mu.Lock()
	f.bindings = len(p.Bindings)
	f.hkCancel = p.Cancel != nil
	f.installed = (f.bindings > 0 || f.hkCancel) && f.trusted
	f.hkActive = p.Active
	install := f.installed && !f.scripted
	if install {
		f.scripted = true
	}
	f.mu.Unlock()
	f.result(m.ID, f.hotkeyStatus())
	if install {
		go f.playScript(p.Bindings)
	}
}

// trust grants Accessibility (trust-after): a configured tap installs and
// says so.
func (f *fake) trust() {
	f.mu.Lock()
	f.trusted = true
	install := (f.bindings > 0 || f.hkCancel) && !f.installed
	f.installed = f.bindings > 0 || f.hkCancel
	play := install && !f.scripted
	if play {
		f.scripted = true
	}
	f.mu.Unlock()
	if install {
		f.notify(proto.MethodHotkeyState, f.hotkeyStatus())
	}
	if play {
		go f.playScript(nil)
	}
}

// playScript plays press= and events= once the tap is installed.
func (f *fake) playScript(bindings []proto.HotkeyBinding) {
	if id := f.opts["press"]; id != "" {
		front := f.front()
		f.notify(proto.MethodHotkeyEvent, proto.HotkeyEventParams{Binding: id, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap,
			HeldMS: 110, T: f.now(), Front: &front})
	}
	path := f.opts["events"]
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "events:", err)
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var s Script
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			fmt.Fprintln(os.Stderr, "events:", err)
			continue
		}
		time.Sleep(time.Duration(s.AfterMS) * time.Millisecond)
		if s.Method == proto.MethodAppActivated {
			var app proto.App
			if json.Unmarshal(s.Params, &app) == nil {
				f.mu.Lock()
				f.moved = &app
				f.mu.Unlock()
			}
		}
		f.send(proto.Message{Method: s.Method, Params: s.Params})
	}
}

// ---- textio ----

func (f *fake) textio(m proto.Message) {
	if !f.isTrusted() {
		f.failData(m.ID, proto.CodePermissionDenied, map[string]string{"permission": proto.PermissionAccessibility}, "VoxtNew may not use Accessibility")
		return
	}
	front := f.front()
	if a, ok := f.switched(); ok {
		front = a
	}
	switch m.Method {
	case proto.MethodTextFocus:
		res := proto.FocusResult{PID: front.PID, App: &front, Role: "AXTextArea", Editable: true, ValueLength: 40}
		switch {
		case f.has("readonly") && (f.opts["readonly"] == "" || f.opts["readonly"] == front.BundleID):
			res.Role, res.Editable, res.ValueLength = "AXStaticText", false, 0
		case f.opts["focus"] == "none":
			res = proto.FocusResult{PID: front.PID, App: &front, Error: "no focused element"}
		case f.opts["focus"] == "secure":
			res.Role, res.Subrole, res.ValueLength = "AXTextField", "AXSecureTextField", 0
		}
		if sel := f.opts["selection"]; sel != "" {
			res.Selection = &proto.TextRange{Location: 0, Length: len([]rune(sel))}
		}
		f.result(m.ID, res)
	case proto.MethodTextSelection:
		var p proto.SelectionParams
		m.DecodeParams(&p)
		res := proto.SelectionResult{Via: proto.ViaNone, MS: 1, Steps: []string{"focus=AXTextArea"}}
		if sel := f.opts["selection"]; sel != "" {
			res.Text, res.Via = sel, proto.ViaAX
		} else if sel := f.opts["copy-selection"]; sel != "" && p.AllowCopy {
			res.Text, res.Via, res.Steps = sel, proto.ViaCopy, append(res.Steps, "copy: changed")
		}
		f.result(m.ID, res)
	case proto.MethodTextPaste:
		var p proto.PasteParams
		m.DecodeParams(&p)
		if p.TargetPID != 0 && p.TargetPID != front.PID {
			f.result(m.ID, proto.PasteResult{MS: 0.1, Front: &front})
			return
		}
		appendLine(f.opts["paste-log"], p)
		f.result(m.ID, f.pasteResult(p))
	case proto.MethodTextRead:
		var p proto.ReadParams
		m.DecodeParams(&p)
		appendLine(f.opts["read-log"], p)
		if text, ok := f.opts["garble"]; ok {
			f.writeRaw(`{"id":"` + m.ID + `","result":{"text":"` + text)
			os.Exit(3)
		}
		f.result(m.ID, f.read(p, front))
	}
}

// appendLine appends v to path as a JSON line (nothing when path is "").
func appendLine(path string, v any) {
	if path == "" {
		return
	}
	line, _ := json.Marshal(v)
	if fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
		fh.Write(append(line, '\n'))
		fh.Close()
	}
}

// Field is a state of the focused field textio.read sees (read=): its
// value, its identity, or an error (proto.Read*).
type Field struct {
	Value   string `json:"value,omitempty"`
	Element int    `json:"element,omitempty"`
	Error   string `json:"error,omitempty"`
}

// read answers a textio.read: the field's next state from read=, and of
// its value the range asked, clamped to it and to MaxRead (UTF-16; the
// fake does not widen to whole characters).
func (f *fake) read(p proto.ReadParams, front proto.App) proto.ReadResult {
	switch {
	case p.PID != front.PID:
		return proto.ReadResult{Error: proto.ReadNotFront}
	case f.opts["focus"] == "none":
		return proto.ReadResult{Error: proto.ReadNoFocus}
	case f.opts["focus"] == "secure":
		return proto.ReadResult{Error: proto.ReadSecure}
	}
	var fields []Field
	if path := f.opts["read"]; path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "read:", err)
		}
		for _, line := range strings.Split(string(data), "\n") {
			var fl Field
			if strings.TrimSpace(line) != "" && json.Unmarshal([]byte(line), &fl) == nil {
				fields = append(fields, fl)
			}
		}
	}
	f.mu.Lock()
	n := f.reads
	f.reads++
	f.mu.Unlock()
	fl := Field{Element: 1}
	if len(fields) > 0 {
		fl = fields[min(n, len(fields)-1)]
	}
	if fl.Error != "" {
		return proto.ReadResult{Error: fl.Error}
	}
	u := utf16.Encode([]rune(fl.Value))
	start := min(max(0, p.Range.Location), len(u))
	end := min(max(start, p.Range.Location+p.Range.Length), len(u), start+proto.MaxRead)
	return proto.ReadResult{Text: string(utf16.Decode(u[start:end])), Range: proto.TextRange{Location: start, Length: end - start},
		Length: len(u), Element: fl.Element}
}

// pasteResult is a posted paste and, with VerifyMS, its check:
// paste-check= (inserted by default, unchanged, unknown); secure-input=
// names the app with secure input on.
func (f *fake) pasteResult(p proto.PasteParams) proto.PasteResult {
	res := proto.PasteResult{Posted: true, Restored: !p.KeepClipboard, MS: 2}
	if name := f.opts["secure-input"]; name != "" {
		res.SecureInput = &proto.App{PID: 1702, Name: name}
	}
	if p.VerifyMS <= 0 {
		return res
	}
	before := &proto.FieldState{Length: 40, Selection: &proto.TextRange{Location: 40}}
	switch check := f.opts["paste-check"]; check {
	case proto.CheckUnchanged:
		res.Check, res.CheckMS = check, float64(p.VerifyMS)
		res.Field = &proto.PasteField{Role: "AXTextArea", Before: before, After: before}
		res.Restored, res.Kept, res.PressSkipped = false, !p.KeepClipboard, len(p.Press) > 0
	case proto.CheckUnknown:
		res.Check, res.Field = check, &proto.PasteField{Role: "AXWebArea"}
	default:
		n := 40 + len([]rune(p.Text))
		res.Check, res.CheckMS = proto.CheckInserted, 1
		res.Field = &proto.PasteField{Role: "AXTextArea", Before: before, After: &proto.FieldState{Length: n, Selection: &proto.TextRange{Location: n}}}
	}
	return res
}

// switched is the app the user went to after the shortcut (switch=):
// textio.* see it in front, the shortcut saw front=.
func (f *fake) switched() (proto.App, bool) {
	id := f.opts["switch"]
	if id == "" {
		return proto.App{}, false
	}
	if a, ok := Apps[id]; ok {
		return a, true
	}
	return proto.App{PID: 701, BundleID: id, Name: id}, true
}

// listed reports whether app.configure named bundleID.
func (f *fake) listed(bundleID string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.manual, func(id string) bool { return strings.EqualFold(id, bundleID) })
}
