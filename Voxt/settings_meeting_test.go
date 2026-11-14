package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestMeetingSettingsModeKey(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Features.Meeting.Mode = "subtitles"
	if a.meetingSettingsKey() != "subtitles|0" {
		t.Fatalf("mode key %q", a.meetingSettingsKey())
	}
}

func TestMeetingFeatureLabels(t *testing.T) {
	if uitext.FeaturePrefix(hotkey.Meeting) != "mt" {
		t.Fatal("prefix")
	}
	if uitext.FeatureTitle(hotkey.Meeting) == "" {
		t.Fatal("title")
	}
}
