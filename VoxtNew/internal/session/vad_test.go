package session

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// speech says, frame by frame (100 ms), a recording of 7.5 s: 2 s of a
// quiet room, a second of speech, 2 s of quiet, half a second of speech,
// and 2 s of quiet.
func speech(n int) bool { return n >= 20 && n < 30 || n >= 50 && n < 55 }

const speechFrames = 75

// TestNoSpeechNotSent: a recording without speech goes to no speech model
// (a file transcriber gets no request, a stream no audio and no Finish)
// and ends as "nothing was heard", which the overlay says.
func TestNoSpeechNotSent(t *testing.T) {
	t.Run("file", func(t *testing.T) {
		file := &fakeFile{text: "never"}
		llm := &fakeLLM{text: "never"}
		h := newHarness(t, file, llm, nil)
		h.audio.quiet = true
		id := h.start()
		h.speak(id, 15)
		end := h.rec.wait(t)
		if end.State != Done || end.Problem == nil || end.Problem.Code != ProblemNoSpeech || end.Inserted || end.Result != "" {
			t.Fatalf("end %+v", end)
		}
		if file.calls != 0 || len(llm.requests()) != 0 || len(h.text.pasted) != 0 || end.Timings.Sent != 0 || end.Timings.Audio < time.Second {
			t.Errorf("sent: %d calls, %d chats, %v of %v", file.calls, len(llm.requests()), end.Timings.Sent, end.Timings.Audio)
		}
		if got := statesString(h.states()); got != "recording recognizing done" {
			t.Errorf("states %s", got)
		}
		if e := h.hist.all(); len(e) != 1 || e[0].SentMS != 0 || e[0].AudioMS < 1000 {
			t.Errorf("history %+v", e)
		}
		if !h.logged("no speech") {
			t.Errorf("logs %q", h.logs)
		}
	})
	t.Run("stream", func(t *testing.T) {
		st := &fakeStreamer{final: "never"}
		llm := &fakeLLM{text: "never"}
		h := newHarness(t, st, llm, nil)
		h.audio.quiet = true
		id := h.start()
		h.speak(id, 15)
		end := h.rec.wait(t)
		if end.State != Done || end.Problem == nil || end.Problem.Code != ProblemNoSpeech || end.Result != "" || end.Partial != "" {
			t.Fatalf("end %+v", end)
		}
		s := st.streams[0]
		s.mu.Lock()
		sends, closed, done := s.sends, s.closed, s.done
		s.mu.Unlock()
		if sends != 0 || done || !closed || len(llm.requests()) != 0 {
			t.Errorf("stream: %d sends, finished %v, closed %v, %d chats", sends, done, closed, len(llm.requests()))
		}
	})
}

// TestTrimmedUpload: a file transcriber gets the speech without the silence
// around it and with the long pause shortened; the history keeps how much
// went. With the detection off, it gets the whole recording.
func TestTrimmedUpload(t *testing.T) {
	for _, off := range []bool{false, true} {
		file := &fakeFile{text: "hello aura link"}
		h := newHarness(t, file, nil, func(_ *Config, s *settings.Settings) { s.Audio.VAD.Off = off })
		h.audio.speaks = speech
		id := h.start()
		h.speak(id, speechFrames)
		end := h.rec.wait(t)
		if end.State != Done || end.Result != "hello VoxtNew" || !end.Inserted {
			t.Fatalf("off %v: end %+v", off, end)
		}
		got := pcmDuration(file.bytes)
		e := h.hist.all()
		if len(e) != 1 || e[0].SentMS != got.Milliseconds() || e[0].AudioMS != end.Timings.Audio.Milliseconds() || end.Timings.Sent != got {
			t.Errorf("off %v: history %+v, sent %v", off, e, end.Timings.Sent)
		}
		if off {
			if got != end.Timings.Audio {
				t.Errorf("off: sent %v of %v", got, end.Timings.Audio)
			}
			continue
		}
		// 0.3 s before, 1 s, the 2 s pause as 0.4 s, 0.5 s, 0.3 s after
		// (and the detector's frames of hangover): about 2.5 s of 7.5.
		if got < 2300*time.Millisecond || got > 3200*time.Millisecond || end.Timings.Audio < 7*time.Second {
			t.Errorf("sent %v of %v", got, end.Timings.Audio)
		}
		t.Logf("sent %v of %v", got, end.Timings.Audio)
	}
}

// TestStreamHeldUntilSpeech: a streaming speech model gets nothing of the
// leading silence but the 300 ms before the speech, then everything,
// the pause included.
func TestStreamHeldUntilSpeech(t *testing.T) {
	st := &fakeStreamer{final: "hello aura link"}
	h := newHarness(t, st, nil, nil)
	h.audio.speaks = speech
	id := h.start()
	h.speak(id, speechFrames)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "hello VoxtNew" {
		t.Fatalf("end %+v", end)
	}
	s := st.streams[0]
	s.mu.Lock()
	sent := pcmDuration(s.bytes)
	s.mu.Unlock()
	// From 1.7 s (2 s, less the 300 ms before it) to the end.
	want := end.Timings.Audio - 1700*time.Millisecond
	if d := sent - want; d < -60*time.Millisecond || d > 60*time.Millisecond || end.Timings.Sent != sent {
		t.Errorf("sent %v (snapshot %v), want about %v of %v", sent, end.Timings.Sent, want, end.Timings.Audio)
	}
	if !h.logged("held back") {
		t.Errorf("logs %q", h.logs)
	}
}

func (h *harness) logged(s string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, l := range h.logs {
		if strings.Contains(l, s) {
			return true
		}
	}
	return false
}
