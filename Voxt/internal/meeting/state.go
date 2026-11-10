package meeting

import "github.com/sgl-project/sglang-omni/Voxt/internal/store"

// Active reports whether shortcuts should treat a meeting as in progress.
func (s State) Active() bool {
	switch s {
	case StateStarting, StateRecording, StatePaused, StateAsking, StateFinishing:
		return true
	default:
		return false
	}
}

func toStoreStatus(s State) store.MeetingStatus {
	switch s {
	case StateRecording, StateStarting:
		return store.MeetingRecording
	case StatePaused:
		return store.MeetingRecording
	case StateAsking:
		return store.MeetingAsking
	case StateFinishing:
		return store.MeetingFinishing
	case StateDone:
		return store.MeetingDone
	case StateInterrupted:
		return store.MeetingInterrupted
	case StateFailed:
		return store.MeetingFailed
	default:
		return store.MeetingRecording
	}
}
