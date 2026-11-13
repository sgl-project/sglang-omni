package session

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestLengthLimitStops: with Audio.MaxSeconds, a recording nobody
// stops ends at that length and is recognised as after Stop; the
// snapshot carries the limit for the pill's warning (#3840).
func TestLengthLimitStops(t *testing.T) {
	f := &fakeFile{text: "a long talk"}
	h := newHarness(t, f, nil, func(_ *Config, s *settings.Settings) { s.Audio.MaxSeconds = 1 })
	h.start()
	end := h.rec.wait(t)
	if end.State != Done || end.Raw != "a long talk" || end.MaxLen != time.Second {
		t.Fatalf("end %s %q, limit %v; want done, the text, 1s", end.State, end.Raw, end.MaxLen)
	}
	if a := end.Timings.Audio; a < time.Second || a > 1200*time.Millisecond {
		t.Errorf("recorded %v; want the 1 s limit", a)
	}
}

// uploads is a file speech model with an upload limit, as OpenAI's.
type uploads struct{ *fakeFile }

func (uploads) Caps() asr.Caps { return asr.Caps{File: true, MaxBytes: 25 << 20} }

// TestUploadLimit: the user's upload limit reaches the speech model's
// request, and the snapshot says how long a recording fits in one
// upload: the user's limit when it is the smaller, else the service's;
// none for a speech model without a limit.
func TestUploadLimit(t *testing.T) {
	for _, c := range []struct {
		name string
		prov asr.Provider
		mb   int
		fits time.Duration
	}{
		{"user's", uploads{&fakeFile{text: "hi"}}, 5, time.Duration(5<<20-44) * time.Second / 32000},
		{"service's", uploads{&fakeFile{text: "hi"}}, 0, time.Duration(25<<20-44) * time.Second / 32000},
		{"over the service's", uploads{&fakeFile{text: "hi"}}, 200, time.Duration(25<<20-44) * time.Second / 32000},
		{"no limit", &fakeFile{text: "hi"}, 5, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, c.prov, nil, func(_ *Config, s *settings.Settings) { s.Audio.UploadMB = c.mb })
			h.speak(h.start(), 10)
			end := h.rec.wait(t)
			f, _ := c.prov.(*fakeFile)
			if u, ok := c.prov.(uploads); ok {
				f = u.fakeFile
			}
			if end.State != Done || end.UploadLen != c.fits || f.most != c.mb<<20 {
				t.Errorf("end %s, fits %v, request MaxBytes %d; want done, %v, %d", end.State, end.UploadLen, f.most, c.fits, c.mb<<20)
			}
		})
	}
}
