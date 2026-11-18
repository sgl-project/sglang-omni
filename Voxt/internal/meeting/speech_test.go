package meeting

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type fakeModelStream struct {
	probs [][]float64
	call  int
	thr   float64
}

func (f *fakeModelStream) Feed(pcm []byte) ([]float64, error) {
	if f.call >= len(f.probs) {
		return nil, nil
	}
	p := f.probs[f.call]
	f.call++
	return p, nil
}

func (f *fakeModelStream) Chunk() int         { return vad.DefaultRate / 50 }
func (f *fakeModelStream) Threshold() float64 { return f.thr }
func (f *fakeModelStream) Close()             {}

func TestSpeechModelThreshold(t *testing.T) {
	ms := &fakeModelStream{
		thr: 0.5,
		probs: [][]float64{
			{0.9},
			{0.1, 0.1},
			{0.1},
		},
	}
	s := NewSpeech(cutterRate, ms)
	speak, _ := s.Push(synthPCM(10, true))
	if !speak {
		t.Fatal("expected speech on high probability")
	}
	speak, _ = s.Push(synthPCM(30, true))
	if speak {
		t.Fatal("expected no speech on low probability")
	}
	speak, silence := s.Push(synthPCM(100, false))
	if speak {
		t.Fatal("still no speech")
	}
	if silence < 90*time.Millisecond {
		t.Fatalf("silence %v too short", silence)
	}
}

func TestSpeechEnergyFallback(t *testing.T) {
	s := NewSpeech(cutterRate, nil)
	speak, _ := s.Push(synthPCM(300, true))
	if !speak {
		t.Fatal("loud PCM should register as speech")
	}
	var silence time.Duration
	for i := 0; i < 6; i++ {
		_, silence = s.Push(synthPCM(200, false))
		if silence >= 400*time.Millisecond {
			return
		}
	}
	t.Fatalf("silence %v after quiet", silence)
}
