package local

// Holder keeps local speech models loaded for the duration of a meeting
// (omni ASR and Silero VAD), so idle unload does not run during pause.
type Holder interface {
	Hold() (release func())
}
