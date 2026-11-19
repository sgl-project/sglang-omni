package meeting

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// seqDiarizer answers with results per request and records the requests.
type seqDiarizer struct {
	mu       sync.Mutex
	results  []asr.Result
	requests []asr.FileRequest
}

func (f *seqDiarizer) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *seqDiarizer) Diarize(_ context.Context, req asr.FileRequest) (asr.Result, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, req)
	return f.results[len(f.requests)-1], nil
}

// A long track goes to MTD in chunks (a dictation slips in between), the
// later requests carry the last labeled lines (a voice keeps its number
// across the boundary), and the segments are offset to their chunk.
func TestDiarizeTrackChunksAndCarriesSpeakers(t *testing.T) {
	d := &seqDiarizer{results: []asr.Result{
		{Segments: []asr.SpeakerSegment{{Start: 0, End: time.Second, Speaker: "S01", Text: "hello team"}}},
		{Segments: []asr.SpeakerSegment{{Start: time.Second, End: 2 * time.Second, Speaker: "S01", Text: "still me"}}},
	}}
	rate := asr.SampleRate
	pcm := make([]byte, rate*200*2) // 200 s
	speech := []vad.Segment{{Start: 0, End: rate * 200}}
	segs, err := diarizeTrack(context.Background(), SourceSystem, d, asr.FileRequest{}, rate, pcm, speech, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.requests) != 2 {
		t.Fatalf("%d requests, want one per chunk", len(d.requests))
	}
	first, second := d.requests[0], d.requests[1]
	if first.Prompt != mossPrompt {
		t.Errorf("the first request's prompt %q", first.Prompt)
	}
	if !strings.Contains(second.Prompt, mossPrompt) || !strings.Contains(second.Prompt, "earlier part") || !strings.Contains(second.Prompt, "[S01] hello team") {
		t.Errorf("the second request's prompt carries nothing over: %q", second.Prompt)
	}
	if len(first.Audio) >= len(pcm) || len(second.Audio) >= len(pcm) || len(first.Audio)+len(second.Audio) < len(pcm) {
		t.Errorf("chunk sizes %d, %d of %d", len(first.Audio)/2/rate, len(second.Audio)/2/rate, len(pcm)/2/rate)
	}
	if len(segs) != 2 || segs[0].Text != "hello team" || segs[1].Text != "still me" {
		t.Fatalf("segments %+v", segs)
	}
	if segs[0].Start != 0 || segs[1].Start < 119*time.Second || segs[1].Start > 122*time.Second {
		t.Errorf("offsets %v, %v", segs[0].Start, segs[1].Start)
	}
	if segs[0].Speaker != "s1" || segs[1].Speaker != "s1" {
		t.Errorf("speakers %+v, %+v", segs[0].Speaker, segs[1].Speaker)
	}
}

// A hard cut starts the next chunk inside audio the previous chunk
// returned: its repeat of the lines there is dropped, not transcribed
// twice, and a line straddling the cut keeps only its new words.
func TestDiarizeTrackDropsOverlapRepeat(t *testing.T) {
	d := &seqDiarizer{results: []asr.Result{
		{Segments: []asr.SpeakerSegment{
			{Start: 0, End: time.Second, Speaker: "S01", Text: "hello team"},
			{Start: 119 * time.Second, End: 120 * time.Second, Speaker: "S01", Text: "ship it today"},
		}},
		{Segments: []asr.SpeakerSegment{
			{Start: 100 * time.Millisecond, End: 1100 * time.Millisecond, Speaker: "S01", Text: "ship it today"},
			{Start: 2 * time.Second, End: 3 * time.Second, Speaker: "S01", Text: "and tomorrow too"},
		}},
	}}
	rate := asr.SampleRate
	pcm := make([]byte, rate*200*2) // 200 s
	speech := []vad.Segment{{Start: 0, End: rate * 200}}
	segs, err := diarizeTrack(context.Background(), SourceSystem, d, asr.FileRequest{}, rate, pcm, speech, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 3 {
		t.Fatalf("segments %+v", segs)
	}
	if segs[1].Text != "ship it today" || segs[2].Text != "and tomorrow too" {
		t.Errorf("segments %+v", segs)
	}
}

// The model can number a voice again in a later chunk: the repeated line
// in the overlap anchors the new number to the earlier speaker instead of
// merging two voices into one, and the carry shows the earlier numbers.
func TestDiarizeTrackAnchorsRenumberedSpeaker(t *testing.T) {
	d := &seqDiarizer{results: []asr.Result{
		{Segments: []asr.SpeakerSegment{
			{Start: 0, End: time.Second, Speaker: "S01", Text: "hello team"},
			{Start: 119 * time.Second, End: 120 * time.Second, Speaker: "S02", Text: "good point let us ship it"},
		}},
		{Segments: []asr.SpeakerSegment{
			// the overlap re-transcribed, with the second voice renumbered S01
			{Start: 100 * time.Millisecond, End: 1100 * time.Millisecond, Speaker: "S01", Text: "good point let us ship it"},
			{Start: 2 * time.Second, End: 3 * time.Second, Speaker: "S01", Text: "next steps now"},
		}},
	}}
	rate := asr.SampleRate
	pcm := make([]byte, rate*200*2) // 200 s
	speech := []vad.Segment{{Start: 0, End: rate * 200}}
	segs, err := diarizeTrack(context.Background(), SourceSystem, d, asr.FileRequest{}, rate, pcm, speech, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 3 {
		t.Fatalf("segments %+v", segs)
	}
	if segs[1].Speaker != "s2" || segs[2].Speaker != "s2" || segs[2].Text != "next steps now" {
		t.Errorf("segments %+v", segs)
	}
	if !strings.Contains(d.requests[1].Prompt, "[S02] good point let us ship it") {
		t.Errorf("the second request's prompt: %q", d.requests[1].Prompt)
	}
}

// Pauses shorter than a chunk do not start a new MTD request: speech
// packs into chunks up to the cap, so a pause-rich track's request count
// follows its length, not its pause count.
func TestDiarizeTrackPacksPausesToTheCap(t *testing.T) {
	rate := asr.SampleRate
	pcm := make([]byte, rate*300*2) // 300 s
	var speech []vad.Segment
	for start := 0; start+10 <= 300; start += 12 {
		speech = append(speech, vad.Segment{Start: start * rate, End: (start + 10) * rate})
	}
	d := &seqDiarizer{}
	for len(d.results) < len(speech) {
		d.results = append(d.results, asr.Result{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "hi"}}})
	}
	if _, err := diarizeTrack(context.Background(), SourceSystem, d, asr.FileRequest{}, rate, pcm, speech, nil); err != nil {
		t.Fatal(err)
	}
	if len(d.requests) != 3 {
		t.Errorf("%d requests for 300 s of pause-rich speech, want 3 (one per %v, not one per pause)", len(d.requests), diarizeChunkMax)
	}
}

func TestDiarizeTrackHonorsProfilingChunkParams(t *testing.T) {
	rate := asr.SampleRate
	pcm := make([]byte, rate*300*2)
	speech := []vad.Segment{{Start: 0, End: rate * 300}}
	d := &seqDiarizer{results: []asr.Result{
		{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "one"}}},
		{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "two"}}},
		{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "three"}}},
		{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "four"}}},
	}}
	params := DefaultChunkParams()
	params.MaxDur, params.PauseMin = 90*time.Second, 90*time.Second
	if _, err := diarizeTrack(context.Background(), SourceSystem, d, asr.FileRequest{}, rate, pcm, speech, nil, params); err != nil {
		t.Fatal(err)
	}
	if len(d.requests) != 4 {
		t.Fatalf("%d requests with a 90-second profiling cap, want 4", len(d.requests))
	}
}

// Speaker numbers past S04 keep their number (an import can have more
// voices than a call's other side).
func TestMTDSpeakerNumbers(t *testing.T) {
	for in, want := range map[string]string{"S01": "s1", "S4": "s4", "S05": "s5", "S12": "s12", "S00": "", "Sx": "", "me": ""} {
		if got := mtdSpeaker(in); got != want {
			t.Errorf("mtdSpeaker(%q) = %q, want %q", in, got, want)
		}
	}
	if got := SpeakerLabel("s5", nil); got != "Speaker 5" {
		t.Errorf("SpeakerLabel(s5) = %q", got)
	}
	if got := SpeakerLabel("", nil); got != "Them" {
		t.Errorf("SpeakerLabel(\"\") = %q", got)
	}
}
