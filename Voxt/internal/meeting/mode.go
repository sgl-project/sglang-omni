package meeting

import "github.com/sgl-project/sglang-omni/Voxt/internal/store"

// Sources lists which tracks mode uses.
func (m Mode) Sources() []Source {
	switch m {
	case ModeSubtitles, ModeImport:
		return []Source{SourceSystem}
	case ModeRecording:
		return []Source{SourceMic}
	default:
		return []Source{SourceMic, SourceSystem}
	}
}

// Diarize is true in full meeting mode (speaker labels, C1-3b) and for an
// imported recording, whose one mixed track is all MTD labels.
func (m Mode) Diarize() bool { return m == "" || m == ModeMeeting || m == ModeImport }

// KeepsAudio is true in recording-only mode (always keep WAVs).
func (m Mode) KeepsAudio() bool { return m == ModeRecording }

func toStoreMode(m Mode) store.MeetingMode {
	switch m {
	case ModeSubtitles:
		return store.MeetingModeSubtitles
	case ModeRecording:
		return store.MeetingModeRecording
	case ModeImport:
		return store.MeetingModeImport
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
	case store.MeetingModeImport:
		return ModeImport
	default:
		return ModeMeeting
	}
}
