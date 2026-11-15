// Meeting is the user's settings for meeting capture (C1-1b+).
package settings

// Meeting holds shortcuts and options for recording a meeting. Zero values
// match Voxt 1.x defaults (mode "" = full meeting, no extra hotkeys).
type Meeting struct {
	Feature
	Mode string `json:"mode,omitempty"` // "" = meeting; subtitles | recording
	// NoFinalPass disables the post-meeting re-transcription pass (C1-3a).
	NoFinalPass bool `json:"noFinalPass,omitempty"`
	// NoAutoSummary disables generating a summary when a meeting ends.
	NoAutoSummary bool `json:"noAutoSummary,omitempty"`
	// Speakers turns on speaker diarisation on the system track (C1-3b).
	Speakers bool `json:"speakers,omitempty"`
	// TranslateTarget is the language code for live translation ("" = en).
	TranslateTarget string `json:"translateTarget,omitempty"`
	// Collapsed keeps the meeting overlay in its compact layout.
	Collapsed bool `json:"collapsed,omitempty"`
	// HideFromSharing hides the overlay from screen sharing.
	HideFromSharing bool `json:"hideFromSharing,omitempty"`
}
