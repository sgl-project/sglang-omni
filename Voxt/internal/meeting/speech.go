package meeting

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

const speechFrame = 10 * time.Millisecond

// Speech reports whether anyone is talking and how long it has been quiet (C1-2).
// With a Silero ModelStream it uses speech probability vs Threshold(); otherwise energy VAD.
type Speech struct {
	rate       int
	frameBytes int
	model      vad.ModelStream
	energy     *vad.Detector
	pending    []byte
	pos        time.Duration
	lastSpeech time.Duration
	hasSpeech  bool
}

// NewSpeech builds a tracker for PCM16 mono at rate Hz (0 → 16 kHz).
func NewSpeech(rate int, model vad.ModelStream) *Speech {
	if rate <= 0 {
		rate = vad.DefaultRate
	}
	fb := rate * int(speechFrame/time.Millisecond) / 1000 * 2
	s := &Speech{rate: rate, frameBytes: fb, model: model}
	if model == nil {
		s.energy = vad.NewDetector(rate, vad.Defaults())
	}
	return s
}

// Push ingests PCM and returns whether the latest frame is speech and silence since last speech.
func (s *Speech) Push(pcm []byte) (speaking bool, silence time.Duration) {
	if len(s.pending) > 0 {
		pcm = append(append([]byte(nil), s.pending...), pcm...)
		s.pending = nil
	}
	for off := 0; off+s.frameBytes <= len(pcm); off += s.frameBytes {
		frame := pcm[off : off+s.frameBytes]
		frameEnd := s.pos + speechFrame
		speaking = s.frameSpeaking(frame)
		if speaking {
			s.hasSpeech = true
			s.lastSpeech = frameEnd
		}
		s.pos = frameEnd
	}
	if tail := pcm[len(pcm)/s.frameBytes*s.frameBytes:]; len(tail) > 0 {
		s.pending = append(s.pending[:0], tail...)
	}
	silence = 0
	if s.hasSpeech {
		silence = s.pos - s.lastSpeech
		if !speaking {
			return false, silence
		}
	}
	return speaking, silence
}

func (s *Speech) frameSpeaking(pcm []byte) bool {
	if s.model != nil {
		probs, err := s.model.Feed(pcm)
		if err != nil {
			if s.energy == nil {
				s.energy = vad.NewDetector(s.rate, vad.Defaults())
			}
		} else {
			thr := s.model.Threshold()
			for _, p := range probs {
				if p >= thr {
					return true
				}
			}
			return false
		}
	}
	if s.energy == nil {
		s.energy = vad.NewDetector(s.rate, vad.Defaults())
	}
	s.energy.Write(pcm)
	return s.energy.Speaking()
}
