package hotkey

import "github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"

// Phase is where the session is, as routing sees it.
type Phase int

const (
	// Idle: no session, or one that ended (an answer card may show).
	Idle Phase = iota
	// Recording: the microphone is on.
	Recording
	// Busy: recognising, composing or delivering; only Escape acts.
	Busy
)

// Action is what a hotkey event does.
type Action int

const (
	None Action = iota
	// Start a session of Decision.Feature; Hold when a long press
	// started it (its release stops it).
	Start
	// Stop recording (the session goes on to recognition).
	Stop
	// Cancel the session.
	Cancel
	// Paste the last result.
	Paste
)

func (a Action) String() string {
	switch a {
	case Start:
		return "start"
	case Stop:
		return "stop"
	case Cancel:
		return "cancel"
	case Paste:
		return "paste"
	}
	return "none"
}

// Decision is Route's answer.
type Decision struct {
	Action  Action
	Feature string
	Hold    bool
}

// Route decides what a hotkey.event does (Voxt's rules, docs/
// HotkeyRules.zh-CN.md, simplified):
//
//   - Idle: a tap or double tap of a feature's binding starts it; a long
//     press starts it on "down" and its "up" stops it; PasteLast pastes.
//   - Recording: any feature's binding stops (the "common stop"): a tap
//     fires at once while active; the "up" of the long press that started
//     the session stops it, other "up"s do nothing. Escape cancels.
//   - Busy: Escape cancels; everything else waits.
//
// holdBinding is the binding whose long press started the session ("":
// none).
func Route(e proto.HotkeyEventParams, phase Phase, holdBinding string) Decision {
	if e.Binding == proto.HotkeyEscape {
		if phase != Idle {
			return Decision{Action: Cancel}
		}
		return Decision{}
	}
	feature, _, ok := ParseID(e.Binding)
	if !ok {
		return Decision{}
	}
	if feature == PasteLast {
		if phase == Idle && e.Phase == proto.PhaseTrigger {
			return Decision{Action: Paste}
		}
		return Decision{}
	}
	switch phase {
	case Idle:
		switch e.Phase {
		case proto.PhaseTrigger:
			return Decision{Action: Start, Feature: feature}
		case proto.PhaseDown:
			return Decision{Action: Start, Feature: feature, Hold: true}
		}
	case Recording:
		switch e.Phase {
		case proto.PhaseTrigger, proto.PhaseDown:
			return Decision{Action: Stop}
		case proto.PhaseUp:
			if e.Binding == holdBinding {
				return Decision{Action: Stop}
			}
		}
	}
	return Decision{}
}
