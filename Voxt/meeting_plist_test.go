package main

import (
	"os"
	"strings"
	"testing"
)

// TestMeetingPrivacyPlistKeys matches design-C1 §4.11: system-audio and
// microphone usage strings mention meetings; no screen-capture usage key.
func TestMeetingPrivacyPlistKeys(t *testing.T) {
	for _, file := range []string{"mygo.json", "helper/Support/Info.plist"} {
		b, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		s := string(b)
		if !strings.Contains(s, "NSAudioCaptureUsageDescription") {
			t.Fatalf("%s: missing NSAudioCaptureUsageDescription", file)
		}
		if !strings.Contains(s, "meeting") {
			t.Fatalf("%s: usage strings must mention meeting", file)
		}
		if strings.Contains(s, "NSScreenCaptureUsageDescription") {
			t.Fatalf("%s: must not request screen capture", file)
		}
		if !strings.Contains(s, "speech service") {
			t.Fatalf("%s: microphone prompt must name the speech service", file)
		}
	}
}
