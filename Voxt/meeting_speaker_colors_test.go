package main

import (
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

func TestMeetingSpeakerIdentitySurvivesNaming(t *testing.T) {
	snapshot := meeting.Snapshot{State: meeting.StateRecording, Mode: meeting.ModeMeeting, Segments: []meeting.Segment{
		{ID: "1", Source: meeting.SourceSystem, Speaker: "s1", Text: "First speaker.", End: time.Second},
		{ID: "2", Source: meeting.SourceSystem, Speaker: "s2", Text: "Second speaker.", Start: time.Second, End: 2 * time.Second},
	}}
	translator := i18n.NewTranslator("en")
	before := meetingOverlayTranscriptRows(translator, snapshot)
	snapshot.Speakers = map[string]string{"s1": "Alice", "s2": "Bob"}
	after := meetingOverlayTranscriptRows(translator, snapshot)
	for i := range before {
		if before[i].SpeakerID != after[i].SpeakerID || before[i].Key != after[i].Key {
			t.Fatalf("naming replaced identity: %+v %+v", before, after)
		}
		if before[i].Who == after[i].Who {
			t.Fatal("name was not updated")
		}
	}
	if after[0].Who != "Alice" || after[1].Who != "Bob" {
		t.Fatalf("names: %+v", after)
	}
	if speakerColor(after[0].SpeakerID, true) == speakerColor(after[1].SpeakerID, true) {
		t.Fatal("two speakers share a color")
	}
}

func TestMeetingSpeakerColorsReadable(t *testing.T) {
	for _, dark := range []bool{false, true} {
		seen := map[ui.Color]bool{}
		surface := defaultLightTokens.Bg
		if dark {
			surface = defaultDarkTokens.Bg
		}
		for _, id := range []string{"me", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"} {
			color := speakerColor(id, dark)
			if seen[color] {
				t.Fatalf("duplicate color: %s, dark=%t", id, dark)
			}
			seen[color] = true
			if ratio := contrastRatio(color, surface); ratio < 4.5 {
				t.Fatalf("%s dark=%t contrast %.2f", id, dark, ratio)
			}
			if dark && contrastRatio(color, ui.RGB(22, 22, 26)) < 4.5 {
				t.Fatalf("%s: unreadable overlay label", id)
			}
		}
	}
}
