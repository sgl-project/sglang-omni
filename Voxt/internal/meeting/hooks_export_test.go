package meeting

// SetModeStarted sets the function a mode change runs once its new tracks
// have started, before it takes the lock again (Manager.modeStarted).
func SetModeStarted(m *Manager, f func()) { m.modeStarted = f }

// WAVRuns is a meeting WAV's samples as value×ms runs ("1×100 0×50").
var WAVRuns = wavRuns
