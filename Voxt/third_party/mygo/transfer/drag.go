package transfer

import "image"

// Operation is a native drag effect, or a bitmask of allowed effects.
type Operation uint8

const (
	None Operation = 0
	Copy Operation = 1
	Move Operation = 2
)

// Allowed supplies the safe default (copy) and removes unsupported effects.
func (o Operation) Allowed() Operation {
	if o == None {
		return Copy
	}
	return o & (Copy | Move)
}

// Negotiate chooses one effect shared by source and destination. A source's
// modifier-key suggestion wins when allowed; otherwise copy precedes move.
func Negotiate(source, destination, suggested Operation) Operation {
	allowed := source & destination.Allowed()
	if suggested == Copy || suggested == Move {
		if allowed&suggested != 0 {
			return suggested
		}
	}
	if allowed&Copy != 0 {
		return Copy
	}
	if allowed&Move != 0 {
		return Move
	}
	return None
}

// DragOptions configures a native source. The default operation is copy.
type DragOptions struct {
	Operations Operation
	// Preview is copied at drag start. Hotspot is in image pixels; the
	// preview is displayed in DIPs (one pixel per DIP).
	Preview image.Image
	Hotspot image.Point
	// Done runs on the main thread exactly once, including cancellation,
	// startup failure and source-window destruction. Delete or remove the
	// original only after a successful Move; MyGo never deletes files.
	Done func(Result)
}

// Result reports the destination's final effect. None means no drop.
type Result struct {
	Operation Operation
	Canceled  bool
	Err       error
}

// Offer describes a drag during hover, without reading its bytes.
type Offer struct {
	Formats    []Format
	Operations Operation
	Suggested  Operation
}

// Preferred selects a format without requesting any representation bytes.
func (o Offer) Preferred(formats ...Format) (Format, bool) {
	return PreferredFormat(o.Formats, formats...)
}

// DropOptions describes a destination. At least one offered format must
// match Formats. Empty Formats accepts none; zero Operations permits copy.
type DropOptions struct {
	Formats    []Format
	Operations Operation
}

// Drop is a completed transfer. Its data is independent of the native
// session and can be read later, from any goroutine.
type Drop struct {
	Data      Data
	Operation Operation
}
