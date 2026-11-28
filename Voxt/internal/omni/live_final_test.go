package omni

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

func TestHTTPFinalCandidatesAreAdvisoryAndOptIn(t *testing.T) {
	for _, tc := range []struct {
		name               string
		enabled            bool
		withDictionary     bool
		malformedCandidate bool
		wantCandidates     int
		wantRequest        bool
	}{
		{name: "disabled"},
		{name: "enabled", enabled: true, wantCandidates: 1, wantRequest: true},
		{name: "malformed candidate", enabled: true, malformedCandidate: true, wantRequest: true},
		{name: "dictionary echo cleanup", enabled: true, withDictionary: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enabled {
				t.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", "1")
			} else {
				t.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", "")
			}
			f := newFake(t, omnitest.Opts{
				Candidate:          "provisional transcript",
				MalformedCandidate: tc.malformedCandidate,
				Text:               "authoritative transcript",
			}, nil)
			var hints []string
			if tc.withDictionary {
				hints = []string{"VoxtNew"}
			} else {
			}
			st, err := f.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{
				SampleRate:   asr.SampleRate,
				Hints:        hints,
				LatePartials: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := st.Send(tone(500*time.Millisecond, 0.3)); err != nil {
				t.Fatal(err)
			}
			if err := st.Finish(); err != nil {
				t.Fatal(err)
			}
			events := drain(t, st)
			candidateCount := 0
			for _, event := range events {
				if event.Kind == asr.EventCandidate && event.Text == "provisional transcript" {
					candidateCount++
				} else if event.Kind == asr.EventFinal && event.Text != "authoritative transcript" {
					t.Errorf("authoritative result %q", event.Text)
				}
			}
			if candidateCount != tc.wantCandidates {
				t.Errorf("got %d HTTP candidates, want %d: events=%+v finals=%+v", candidateCount, tc.wantCandidates, events, f.records("final"))
			}
			finals := f.records("final")
			if len(finals) != 1 {
				t.Fatalf("got %d HTTP Final requests", len(finals))
			}
			if included := finals[0].Fields["include_transcript_candidates"] == "true"; included != tc.wantRequest {
				t.Errorf("candidate request opt-in %v, want %v", included, tc.wantRequest)
			}
		})
	}
}
