//go:build !darwin

package main

// Privacy settings deep links exist only on macOS.
const (
	privacyMicrophone    = ""
	privacyAccessibility = ""
	privacySpeech        = ""
	privacyAudioCapture  = ""
)
