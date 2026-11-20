package meeting

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
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
