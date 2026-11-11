//go:build !darwin && !windows

package main

// System Settings panes on platforms without a deep-link scheme.
const (
	privacyMicrophone    = ""
	privacyAccessibility = ""
	privacySpeech        = ""
	privacyAudioCapture  = ""
)
