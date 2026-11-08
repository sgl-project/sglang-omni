package meeting

import "github.com/sgl-project/sglang-omni/Voxt/internal/store"

// Sources lists which tracks mode uses.
func (m Mode) Sources() []Source {
	switch m {
	case ModeSubtitles:
		return []Source{SourceSystem}
	case ModeRecording:
		return []Source{SourceMic}
	default:
		return []Source{SourceMic, SourceSystem}
	}
}

// Diarize is true only in full meeting mode (speaker labels, C1-3b).
func (m Mode) Diarize() bool { return m == "" || m == ModeMeeting }

// KeepsAudio is true in recording-only mode (always keep WAVs).
func (m Mode) KeepsAudio() bool { return m == ModeRecording }

func toStoreMode(m Mode) store.MeetingMode {
	switch m {
	case ModeSubtitles:
		return store.MeetingModeSubtitles
	case ModeRecording:
		return store.MeetingModeRecording
	default:
		return store.MeetingModeMeeting
	}
}

func modeNeedsSource(mode Mode, src Source) bool {
	for _, s := range mode.Sources() {
		if s == src {
			return true
		}
	}
	return false
}

func fromStoreMode(m store.MeetingMode) Mode {
	switch m {
	case store.MeetingModeSubtitles:
		return ModeSubtitles
	case store.MeetingModeRecording:
		return ModeRecording
	default:
		return ModeMeeting
	}
}
