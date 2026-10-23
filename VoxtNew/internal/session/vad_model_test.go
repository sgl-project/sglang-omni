package session

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// fakeVAD is a voice activity model that hears speech in the 100 ms
// frames speaks says (whatever their energy), or fails.
type fakeVAD struct {
	speaks               func(n int) bool
	failSpeech, failOpen bool
	failAfter            int // a stream fails after this many chunks (0: never)

	mu       sync.Mutex
	speeches int
	streams  int
	closed   int
}

const frameSamples = 1600

func (f *fakeVAD) Speech(_ context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
	f.mu.Lock()
	f.speeches++
	f.mu.Unlock()
	if f.failSpeech {
		return nil, errors.New("fake vad: failed")
	}
	var segs []vad.Segment
	for n := 0; (n+1)*frameSamples*2 <= len(pcm); n++ {
		if !f.speaks(n) {
			continue
		}
		if k := len(segs); k > 0 && segs[k-1].End == n*frameSamples {
			segs[k-1].End += frameSamples
		} else {
			segs = append(segs, vad.Segment{Start: n * frameSamples, End: (n + 1) * frameSamples})
		}
	}
	return segs, nil
}

func (f *fakeVAD) Open(ctx context.Context) (vad.ModelStream, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failOpen {
		return nil, errors.New("fake vad: no server")
	}
	f.streams++
	return &fakeVADStream{f: f, ctx: ctx}, nil
}

// fakeVADStream fails once Open's context ends, as a real one does.
type fakeVADStream struct {
	f      *fakeVAD
	ctx    context.Context
	pos    int // samples consumed in chunks
	rest   int
	chunks int
}

func (s *fakeVADStream) Chunk() int         { return 512 }
func (s *fakeVADStream) Threshold() float64 { return 0.5 }

func (s *fakeVADStream) Feed(pcm []byte) ([]float64, error) {
	if err := s.ctx.Err(); err != nil {
		return nil, err
	}
	s.rest += len(pcm) / 2
	var out []float64
	for s.rest >= 512 {
		if s.f.failAfter > 0 && s.chunks >= s.f.failAfter {
			return out, errors.New("fake vad: the stream broke")
		}
		p := 0.1
		if s.f.speaks(s.pos / frameSamples) {
			p = 0.9
		}
		out = append(out, p)
		s.pos += 512
		s.rest -= 512
		s.chunks++
	}
	return out, nil
}

func (s *fakeVADStream) Close() {
	s.f.mu.Lock()
	s.f.closed++
	s.f.mu.Unlock()
}

// laterSpeech: the model hears only the second burst of speech (the
// first, which energy takes for speech, is noise to it).
func laterSpeech(n int) bool { return n >= 50 && n < 55 }

// TestModelTrimsUpload: with a model, a file transcriber gets the speech
// the model heard (0.3 s before it, 0.5 s, 0.3 s after); without one, or
// when it fails, the energy detector's (about 2.5 s).
func TestModelTrimsUpload(t *testing.T) {
	for _, c := range []struct {
		name     string
		model    *fakeVAD
		min, max time.Duration
		log      string
	}{
		{"model", &fakeVAD{speaks: laterSpeech}, 1000 * time.Millisecond, 1200 * time.Millisecond, "voice activity model: 1 segments"},
		{"model fails", &fakeVAD{speaks: laterSpeech, failSpeech: true}, 2300 * time.Millisecond, 3200 * time.Millisecond, "using the energy detector"},
	} {
		t.Run(c.name, func(t *testing.T) {
			file := &fakeFile{text: "hello"}
			h := newHarness(t, file, nil, nil)
			h.vadModel = c.model
			h.audio.speaks = speech
			id := h.start()
			h.speak(id, speechFrames)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "hello" {
				t.Fatalf("end %+v", end)
			}
			if got := pcmDuration(file.bytes); got < c.min || got > c.max || end.Timings.Sent != got {
				t.Errorf("sent %v (snapshot %v), want %v…%v", got, end.Timings.Sent, c.min, c.max)
			}
			if c.model.speeches != 1 || !h.logged(c.log) {
				t.Errorf("%d calls, logs %q", c.model.speeches, h.logs)
			}
		})
	}
}

// TestModelHearsNoSpeech: loud audio the model takes for no speech is
// sent nowhere.
func TestModelHearsNoSpeech(t *testing.T) {
	file := &fakeFile{text: "never"}
	h := newHarness(t, file, nil, nil)
	h.vadModel = &fakeVAD{speaks: func(int) bool { return false }}
	id := h.start()
	h.speak(id, 20)
	end := h.rec.wait(t)
	if end.Problem == nil || end.Problem.Code != ProblemNoSpeech || file.calls != 0 {
		t.Fatalf("end %+v, %d uploads", end, file.calls)
	}
}

// TestModelGatesStream: a stream gets nothing before the model's onset
// (less the 300 ms before it); if the model is missing or breaks, the
// energy detector's onset counts.
func TestModelGatesStream(t *testing.T) {
	for _, c := range []struct {
		name  string
		model *fakeVAD
		from  time.Duration
		log   string
	}{
		{"model", &fakeVAD{speaks: laterSpeech}, 4700 * time.Millisecond, "held back"},
		{"no stream", &fakeVAD{speaks: laterSpeech, failOpen: true}, 1700 * time.Millisecond, "using the energy detector"},
		{"stream breaks", &fakeVAD{speaks: laterSpeech, failAfter: 20}, 1700 * time.Millisecond, "the energy detector went on"},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := &fakeStreamer{final: "hello"}
			h := newHarness(t, st, nil, nil)
			h.vadModel = c.model
			h.audio.speaks = speech
			id := h.start()
			h.speak(id, speechFrames)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "hello" {
				t.Fatalf("end %+v", end)
			}
			s := st.streams[0]
			s.mu.Lock()
			sent := pcmDuration(s.bytes)
			s.mu.Unlock()
			want := end.Timings.Audio - c.from
			if d := sent - want; d < -70*time.Millisecond || d > 70*time.Millisecond {
				t.Errorf("sent %v, want about %v of %v", sent, want, end.Timings.Audio)
			}
			if !h.logged(c.log) || c.name == "model" && h.logged("voice activity model") {
				t.Errorf("logs %q", h.logs)
			}
			c.model.mu.Lock()
			defer c.model.mu.Unlock()
			if c.model.closed != c.model.streams {
				t.Errorf("%d streams, %d closed", c.model.streams, c.model.closed)
			}
		})
	}
}

// TestSpeechLang: the speech model gets the main language as its hint
// (the settings', else the UI's), also when the user speaks other
// languages: without a hint Qwen3-ASR may take a mixed dictation that
// starts in English for English and translate the rest (docs/eval.md).
func TestSpeechLang(t *testing.T) {
	for _, c := range []struct {
		main, locale string
		others       []string
		want         string
	}{{"ja", "en", nil, "ja"}, {"", "zh-Hans", nil, "zh"}, {"zh", "en", []string{"en"}, "zh"}} {
		file := &fakeFile{text: "hi"}
		h := newHarness(t, file, nil, func(cfg *Config, s *settings.Settings) {
			s.General.MainLanguage, s.General.OtherLanguages = c.main, c.others
			cfg.Locale = func() string { return c.locale }
		})
		id := h.start()
		h.speak(id, 10)
		h.rec.wait(t)
		if file.lang != c.want {
			t.Errorf("main %q, locale %q, others %q: hint %q, want %q", c.main, c.locale, c.others, file.lang, c.want)
		}
	}
}
