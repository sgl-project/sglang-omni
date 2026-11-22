package meeting

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

func TestBatchedSpeechRegionsJoinsWindows(t *testing.T) {
	rate := asr.SampleRate
	blob := tonePCMSamples(rateSamples(400 * time.Millisecond))
	var pcm []byte
	pcm = append(pcm, asr.Silence(9*time.Minute+30*time.Second)...)
	pcm = append(pcm, blob...)
	pcm = append(pcm, asr.Silence(30*time.Second)...)
	pcm = append(pcm, blob...)
	segs, err := speechRegions(context.Background(), pcm, rate, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) < 2 {
		t.Fatalf("want speech in both windows, got %d segments", len(segs))
	}
}

func tonePCMSamples(samples int) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		out[i*2] = 0x10
		out[i*2+1] = 0x7f
	}
	return out
}

type batchVAD struct{ requests []int }

func (v *batchVAD) Speech(_ context.Context, pcm []byte, _ int) ([]vad.Segment, error) {
	v.requests = append(v.requests, len(pcm)/2)
	return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
}

func (v *batchVAD) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

func TestSpeechRegionsBoundsModelRequests(t *testing.T) {
	model := &batchVAD{}
	samples := rateSamples(11 * time.Minute)
	segments, err := speechRegions(context.Background(), tonePCMSamples(samples), asr.SampleRate, model)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 2 {
		t.Fatalf("VAD requests = %d, want two bounded batches", model.requests)
	}
	limit := rateSamples(vadWindowBatch + vadOverlapBatch)
	for _, request := range model.requests {
		if request > limit {
			t.Errorf("VAD request holds %d samples, limit %d", request, limit)
		}
	}
	if len(segments) != 1 || segments[0].Start != 0 || segments[0].End != samples {
		t.Fatalf("speech segments = %+v, want the whole track", segments)
	}
}
