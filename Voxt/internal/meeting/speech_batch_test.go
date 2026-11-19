package meeting

import (
	"context"
	"errors"
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
	segs, err := speechRegions(context.Background(), pcm, rate, nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) < 2 {
		t.Fatalf("want speech in both windows, got %d segments", len(segs))
	}
}

type batchVAD struct {
	calls       int
	succeedFrom int
}

func (v *batchVAD) Speech(_ context.Context, pcm []byte, _ int) ([]vad.Segment, error) {
	v.calls++
	if v.succeedFrom > 0 && v.calls >= v.succeedFrom {
		return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
	}
	return nil, errors.New("neural VAD failed")
}

func (v *batchVAD) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

func TestSpeechRegionsNotifiesOnlyActualEnergyFallback(t *testing.T) {
	pcm := tonePCMSamples(asr.SampleRate / 10)
	t.Run("whole track retry succeeds", func(t *testing.T) {
		model := &batchVAD{succeedFrom: 2}
		fallbacks := 0
		segments, err := speechRegions(context.Background(), pcm, asr.SampleRate, model, func() { fallbacks++ }, "")
		if err != nil || len(segments) != 1 || model.calls != 2 || fallbacks != 0 {
			t.Fatalf("segments %+v, err %v, calls %d, fallbacks %d", segments, err, model.calls, fallbacks)
		}
	})
	t.Run("energy fallback", func(t *testing.T) {
		model := &batchVAD{}
		fallbacks := 0
		_, _ = speechRegions(context.Background(), pcm, asr.SampleRate, model, func() { fallbacks++ }, "")
		if fallbacks != 1 {
			t.Fatalf("fallbacks %d", fallbacks)
		}
	})
	t.Run("default energy path", func(t *testing.T) {
		fallbacks := 0
		_, _ = speechRegions(context.Background(), pcm, asr.SampleRate, nil, func() { fallbacks++ }, "")
		if fallbacks != 0 {
			t.Fatalf("nil model notified %d times", fallbacks)
		}
	})
}

func TestSpeechRegionsStrategy(t *testing.T) {
	blob := tonePCMSamples(asr.SampleRate / 10)
	var pcm []byte
	pcm = append(pcm, asr.Silence(9*time.Minute+30*time.Second)...)
	pcm = append(pcm, blob...)
	pcm = append(pcm, asr.Silence(30*time.Second)...)
	pcm = append(pcm, blob...)
	whole, err := speechRegions(context.Background(), pcm, asr.SampleRate, &batchVAD{succeedFrom: 1}, nil, "whole")
	if err != nil || len(whole) != 1 {
		t.Fatalf("whole segments %+v, err %v", whole, err)
	}
	batched, err := speechRegions(context.Background(), pcm, asr.SampleRate, &batchVAD{succeedFrom: 1}, nil, "batched")
	if err != nil || len(batched) < 2 {
		t.Fatalf("batched segments %+v, err %v", batched, err)
	}
	if _, err := speechRegions(context.Background(), pcm, asr.SampleRate, &batchVAD{}, nil, "whole"); err == nil {
		t.Fatal("whole-track neural failure was hidden by the batched fallback")
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
