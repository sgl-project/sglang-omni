package meeting

// Active reports whether shortcuts should treat a meeting as in progress.
func (s State) Active() bool {
	switch s {
	case StateStarting, StateRecording, StatePaused, StateAsking, StateFinishing:
		return true
	default:
		return false
	}
}
