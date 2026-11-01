package deliver

import (
	"fmt"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// Diag describes an insert for the log, so that a result that did not
// arrive can be explained from the log alone: the focused element as
// the delivery saw it, the paste, its check (the element before and
// after), the clipboard and secure event input. No text, only lengths.
func (r Result) Diag() string {
	var b strings.Builder
	f := r.Focus
	app := f.PID
	name := ""
	switch {
	case f.App != nil:
		name = f.App.Name
	case r.To.PID == f.PID:
		name = r.To.Name
	case r.Front.PID == f.PID:
		name = r.Front.Name
	}
	role := f.Role
	if role == "" {
		role = "no role"
	}
	if f.Subrole != "" {
		role += "/" + f.Subrole
	}
	fmt.Fprintf(&b, "focus %s in %s (pid %d", role, orDash(name), app)
	if f.Editable {
		b.WriteString(", editable")
	}
	fmt.Fprintf(&b, ", value %d", f.ValueLength)
	if f.Selection != nil {
		fmt.Fprintf(&b, ", selection %s", rangeText(f.Selection))
	}
	if f.Error != "" {
		fmt.Fprintf(&b, ", %s", f.Error)
	}
	b.WriteString(")")
	if r.Activated {
		b.WriteString("; activated the start app first")
	}
	p := r.Paste
	switch {
	case r.Err != nil:
		fmt.Fprintf(&b, "; failed: %v", r.Err)
		return b.String()
	case !p.Posted && r.Reason != ReasonNone:
		fmt.Fprintf(&b, "; not pasted (%s)", r.Reason)
		return b.String()
	}
	b.WriteString("; ⌘V posted")
	switch p.Check {
	case "":
		b.WriteString(", not checked")
	case proto.CheckUnknown:
		b.WriteString(", check unknown")
	default:
		fmt.Fprintf(&b, ", check %s after %.0f ms", p.Check, p.CheckMS)
	}
	if fl := p.Field; fl != nil {
		fmt.Fprintf(&b, " (%s: %s → %s)", orDash(fl.Role), stateText(fl.Before), stateText(fl.After))
	}
	switch {
	case p.Kept:
		b.WriteString(", the clipboard keeps the text")
	case p.Restored:
		b.WriteString(", clipboard restored")
	case p.ClipboardTaken:
		b.WriteString(", clipboard taken by another app")
	}
	if p.PressSkipped {
		b.WriteString(", auto key skipped")
	}
	if s := p.SecureInput; s != nil {
		fmt.Fprintf(&b, "; secure input on (%s, pid %d)", orDash(s.Name), s.PID)
	}
	fmt.Fprintf(&b, "; %.0f ms", p.MS)
	return b.String()
}

func stateText(s *proto.FieldState) string {
	if s == nil {
		return "unreadable"
	}
	out := fmt.Sprintf("value %d", s.Length)
	if s.Selection != nil {
		out += ", selection " + rangeText(s.Selection)
	}
	if s.Marker {
		out += ", marker"
	}
	return out
}

func rangeText(r *proto.TextRange) string { return fmt.Sprintf("%d+%d", r.Location, r.Length) }

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
