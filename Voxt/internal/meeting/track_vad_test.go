package meeting

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type trackVAD struct {
	speech                   bool
	opened, closed, analyzed int
}

func (v *trackVAD) Open(context.Context) (vad.ModelStream, error) { v.opened++; return v, nil }
func (v *trackVAD) Speech(context.Context, []byte, int) ([]vad.Segment, error) {
	v.analyzed++
	if v.speech {
		return []vad.Segment{{Start: 0, End: asr.SampleRate}}, nil
	}
	return nil, nil
}
func (v *trackVAD) Feed(pcm []byte) ([]float64, error) {
	p := 0.0
	if v.speech {
		p = 1
	}
	return []float64{p}, nil
}
func (*trackVAD) Chunk() int         { return 160 }
func (*trackVAD) Threshold() float64 { return .5 }
func (v *trackVAD) Close()           { v.closed++ }

// A loud signal without neural speech evidence must produce neither a live
// preview nor a saved line. The same words actually spoken remain allowed.
func TestTrackOpensSpeechDetectorAndGuardsPreview(t *testing.T) {
	for _, said := range []bool{false, true} {
		ctx := context.Background()
		v := &trackVAD{speech: said}
		streamer := &liveFakeStreamer{partial: "Subtitles by the Amara.org community"}
		m := NewManager(Config{ASR: func(context.Context, Source) (TrackASR, error) {
			return TrackASR{Caps: streamer.Caps(), Streamer: streamer, VAD: v}, nil
		}})
		m.timeline = newTimeline(time.Now, time.Now())
		tr, err := m.newTrackTranscriber(ctx, SourceMic, "test")
		if err != nil {
			t.Fatal(err)
		}
		var lines []Segment
		tr.live.OnCommit = func(s Segment) { lines = append(lines, s) }
		if err := tr.onPCM(ctx, speechPCM(time.Second, asr.SampleRate)); err != nil {
			t.Fatal(err)
		}
		if text := tr.live.LiveText(); (text != "") != said {
			t.Fatalf("speech=%v preview=%q", said, text)
		}
		if err := tr.finish(ctx); err != nil {
			t.Fatal(err)
		}
		if (len(lines) > 0) != said || v.opened != 1 || v.closed != 1 {
			t.Fatalf("speech=%v lines=%v opened=%d closed=%d", said, lines, v.opened, v.closed)
		}
	}
}

type vadCountFile struct{ calls int }

func (*vadCountFile) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *vadCountFile) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	f.calls++
	return asr.Result{}, nil
}

func TestFileTrackRejectsNoiseBeforeRecognition(t *testing.T) {
	for _, said := range []bool{false, true} {
		ctx := context.Background()
		v, file := &trackVAD{speech: said}, &vadCountFile{}
		m := NewManager(Config{ASR: func(context.Context, Source) (TrackASR, error) {
			return TrackASR{Caps: asr.Caps{File: true}, File: file, VAD: v}, nil
		}})
		m.timeline = newTimeline(time.Now, time.Now())
		tr, err := m.newTrackTranscriber(ctx, SourceSystem, "test")
		if err != nil {
			t.Fatal(err)
		}
		if err := tr.onPCM(ctx, speechPCM(3*time.Second, asr.SampleRate)); err != nil {
			t.Fatal(err)
		}
		if err := tr.finish(ctx); err != nil {
			t.Fatal(err)
		}
		if v.analyzed == 0 || (file.calls > 0) != said {
			t.Fatalf("speech=%v analyses=%d calls=%d", said, v.analyzed, file.calls)
		}
	}
}
