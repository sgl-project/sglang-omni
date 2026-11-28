package meeting

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestSpeakerEvidenceRejectsSwitchesAndStaleNames(t *testing.T) {
	s := speakerTracker{names: map[string]string{}}
	s.seen = []speakerSeen{{at: 0, id: "Alice"}, {at: time.Second, id: "Bob"}, {at: 2 * time.Second, id: ""}}
	for _, span := range [][2]time.Duration{{0, 2 * time.Second}, {2 * time.Second, 3 * time.Second}, {4 * time.Second, 5 * time.Second}} {
		if got := s.forRange(span[0], span[1]); got != "" {
			t.Fatalf("ambiguous/stale name %q for %v", got, span)
		}
	}
	seg := Segment{Speaker: "s2", Start: time.Second, End: 2 * time.Second}
	s.nameSegment(&seg)
	if s.names["s2"] != "Bob" || seg.Speaker != "s2" {
		t.Fatalf("name changed voice ID: %+v %+v", seg, s.names)
	}
	s.seen = append(s.seen, speakerSeen{at: 3 * time.Second, id: "Carol"})
	conflict := Segment{Speaker: "s2", Start: 3 * time.Second, End: 4 * time.Second}
	s.nameSegment(&conflict)
	s.nameSegment(&seg)
	if s.names["s2"] != "" || !s.ambiguous["s2"] || conflict.Speaker != "s2" {
		t.Fatal("conflicting screen names silently relabeled a merged voice")
	}
}

func TestWindowSpeakersNeedsSpeakingEvidence(t *testing.T) {
	window := proto.MeetingWindowResult{Nodes: []proto.MeetingNode{
		{Parent: -1, Role: "AXWindow", Identifier: "zm.meeting.window.main"},
		{Parent: 0, Role: "AXVideoRender", Description: "Alice, computer audio connected"},
		{Parent: 0, Role: "AXVideoRender", Description: "Bob, muted"},
	}}
	if got := ScreenSpeakers(window, "zoom"); len(got.Active) != 0 {
		t.Fatalf("audio connection became speaking: %+v", got)
	}
	window.Tiles = []proto.MeetingScreenTile{{Name: "Alice", Speaking: true}, {Name: "Bob", Speaking: false}}
	if got := ScreenSpeakers(window, "zoom"); len(got.Active) != 1 || got.Active[0] != "Alice" {
		t.Fatalf("screen speaker: %+v", got)
	}
	window.Tiles = append(window.Tiles, proto.MeetingScreenTile{Name: "Bob", Speaking: true})
	if got := ScreenSpeakers(window, "zoom"); len(got.Active) != 2 {
		t.Fatalf("overlap evidence was lost: %+v", got)
	}
}

func TestFinalSpeakerLabelsKeepLiveIdentities(t *testing.T) {
	live := []store.MeetingSegment{
		{Source: store.SegmentSystem, Speaker: "s1", StartMS: 0, EndMS: 2000},
		{Source: store.SegmentSystem, Speaker: "s2", StartMS: 2000, EndMS: 4000},
	}
	final := []Segment{{Speaker: "s2", Start: 0, End: 2 * time.Second}, {Speaker: "s1", Start: 2 * time.Second, End: 4 * time.Second}, {Speaker: "s3", Start: 5 * time.Second, End: 6 * time.Second}}
	keepLiveSpeakers(final, SourceSystem, live)
	if final[0].Speaker != "s1" || final[1].Speaker != "s2" || final[2].Speaker == "s1" || final[2].Speaker == "s2" {
		t.Fatalf("final labels: %+v", final)
	}
}
