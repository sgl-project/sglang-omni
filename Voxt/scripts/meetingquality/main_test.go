package main

import "testing"

func TestRequestWindowsUseTheResultCap(t *testing.T) {
	var result resultFile
	result.AudioDurationMs = 601943
	for i := 0; i < 6; i++ {
		audioMs := 120240
		if i == 5 {
			audioMs = 6708
		}
		result.Stages = append(result.Stages, struct {
			Name    string `json:"name"`
			Track   string `json:"track,omitempty"`
			Index   int    `json:"index"`
			AudioMs int    `json:"audio_ms"`
		}{Name: "diarization", Index: i, AudioMs: audioMs})
	}
	windows, err := requestWindows(result, 120)
	if err != nil {
		t.Fatal(err)
	}
	if len(windows) != 6 {
		t.Fatalf("got %d requests, want 6", len(windows))
	}
	for _, test := range []struct {
		seconds float64
		want    int
	}{{218, 2}, {438, 4}, {601, 6}} {
		got, err := requestAt(windows, test.seconds)
		if err != nil || got != test.want {
			t.Errorf("requestAt(%v) = %d, %v; want %d", test.seconds, got, err, test.want)
		}
	}
	if got := windows[1].NativeStart; got != 118.88 {
		t.Errorf("request 2 native start = %v, want 118.88", got)
	}
	if got := windows[3].NativeStart; got != 356.88 {
		t.Errorf("request 4 native start = %v, want 356.88", got)
	}
	if got := windows[3].TimelineStart; got != 357 {
		t.Errorf("request 4 timeline start = %v, want 357", got)
	}
	if got := windows[5].NativeEnd - windows[5].NativeStart; got < 6.707999 || got > 6.708001 {
		t.Errorf("request 6 native duration = %v, want 6.708", got)
	}
}

func TestRequestWindowsRejectsMultipleTracks(t *testing.T) {
	var result resultFile
	result.Stages = append(result.Stages,
		struct {
			Name    string `json:"name"`
			Track   string `json:"track,omitempty"`
			Index   int    `json:"index"`
			AudioMs int    `json:"audio_ms"`
		}{Name: "diarization", Track: "mic", AudioMs: 1000},
		struct {
			Name    string `json:"name"`
			Track   string `json:"track,omitempty"`
			Index   int    `json:"index"`
			AudioMs int    `json:"audio_ms"`
		}{Name: "diarization", Track: "system", Index: 1, AudioMs: 1000},
	)
	if _, err := requestWindows(result, 120); err == nil {
		t.Fatal("two diarization tracks were silently treated as one timeline")
	}
}

func TestLongClipIDsRejectsAnEmptyFixture(t *testing.T) {
	if _, err := longClipIDs(clipsFile{}); err == nil {
		t.Fatal("an empty long-clip fixture was accepted")
	}
}

func TestRequestAtRejectsAudioOutsideTheTimeline(t *testing.T) {
	var result resultFile
	result.AudioDurationMs = 601943
	for i := 0; i < 6; i++ {
		result.Stages = append(result.Stages, struct {
			Name    string `json:"name"`
			Track   string `json:"track,omitempty"`
			Index   int    `json:"index"`
			AudioMs int    `json:"audio_ms"`
		}{Name: "diarization", Index: i, AudioMs: 120240})
	}
	windows, err := requestWindows(result, 120)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestAt(windows, 720); err == nil {
		t.Fatal("audio outside the six-request timeline was unexpectedly accepted")
	}
}
