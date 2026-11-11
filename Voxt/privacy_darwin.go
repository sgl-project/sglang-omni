//go:build darwin

package main

// System Settings panes on macOS.
const (
	privacyMicrophone    = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
	privacyAccessibility = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
	privacySpeech        = "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition"
	privacyAudioCapture  = "x-apple.systempreferences:com.apple.preference.security?Privacy_AudioCapture"
)
