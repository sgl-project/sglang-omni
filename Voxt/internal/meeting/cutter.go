package meeting

import (
	"encoding/binary"
	"math"
	"time"
)

const (
	cutterRate         = 16000
	cutterFrame        = 10 * time.Millisecond
	cutterSilenceCut   = 450 * time.Millisecond
	cutterMinUtterance = 350 * time.Millisecond
	cutterMaxSegment   = 7 * time.Second
	cutterPreRoll      = 300 * time.Millisecond
	cutterQuietSearch  = time.Second
	cutterQuietWindow  = 100 * time.Millisecond
	cutterSpeechRMS    = 800.0
)

// Cutter splits a PCM16 mono stream into utterances for file-mode ASR (Voxt 1.x quality tier).
type Cutter struct {
	pos        time.Duration
	pending    []byte
	ring       []byte
	buf        []byte
	utterStart time.Duration
	speechAt   int // byte offset where voiced audio starts in buf
	voicedDur  time.Duration
	inUtter    bool
	silence    time.Duration
}

// Push ingests PCM16 mono at 16 kHz and returns any completed utterances.
func (c *Cutter) Push(pcm []byte) []Utterance {
	frameBytes := int(cutterRate*cutterFrame/time.Second) * 2
	if len(c.pending) > 0 {
		pcm = append(append([]byte(nil), c.pending...), pcm...)
		c.pending = nil
	}
	var out []Utterance
	for off := 0; off+frameBytes <= len(pcm); off += frameBytes {
		out = append(out, c.pushFrame(pcm[off:off+frameBytes])...)
	}
	if tail := pcm[len(pcm)/frameBytes*frameBytes:]; len(tail) > 0 {
		c.pending = append(c.pending[:0], tail...)
	}
	return out
}

func (c *Cutter) pushFrame(frame []byte) []Utterance {
	c.appendRing(frame)
	speech := frameRMS(frame) >= cutterSpeechRMS
	var out []Utterance
	if !c.inUtter {
		if !speech {
			c.pos += cutterFrame
			return nil
		}
		c.startUtterance()
	}
	c.buf = append(c.buf, frame...)
	if speech {
		c.silence = 0
		c.voicedDur += cutterFrame
	} else {
		c.silence += cutterFrame
	}
	end := c.pos + cutterFrame
	if c.silence >= cutterSilenceCut {
		if u, ok := c.finalize(end, true); ok {
			out = append(out, u)
		}
		c.resetUtter()
	} else if end-c.utterStart >= cutterMaxSegment {
		if u, ok := c.cutAtQuietest(end); ok {
			out = append(out, u)
		}
	}
	c.pos = end
	return out
}

// Flush returns a partial utterance still buffered (pause/end).
func (c *Cutter) Flush() []Utterance {
	if !c.inUtter || len(c.buf) == 0 {
		return nil
	}
	end := c.pos
	u := Utterance{Start: c.utterStart, End: end, PCM: append([]byte(nil), c.buf...)}
	c.resetUtter()
	return []Utterance{u}
}

func (c *Cutter) startUtterance() {
	c.inUtter = true
	c.utterStart = c.pos
	if len(c.ring) > 0 {
		c.buf = append(append([]byte(nil), c.ring...), c.buf...)
		if c.utterStart > cutterPreRoll {
			c.utterStart -= cutterPreRoll
		} else {
			c.utterStart = 0
		}
	}
	c.speechAt = len(c.buf)
	c.voicedDur = 0
	c.silence = 0
}

func (c *Cutter) appendRing(frame []byte) {
	maxRing := int(cutterRate*cutterPreRoll/time.Second) * 2
	c.ring = append(c.ring, frame...)
	if len(c.ring) > maxRing {
		c.ring = c.ring[len(c.ring)-maxRing:]
	}
}

func (c *Cutter) finalize(end time.Duration, minLen bool) (Utterance, bool) {
	if minLen && c.voicedDur < cutterMinUtterance {
		return Utterance{}, false
	}
	return Utterance{
		Start: c.utterStart,
		End:   end,
		PCM:   append([]byte(nil), c.buf...),
	}, true
}

func (c *Cutter) cutAtQuietest(end time.Duration) (Utterance, bool) {
	cut := quietestCutOffset(c.buf, cutterQuietSearch, cutterQuietWindow)
	if cut <= 0 {
		return c.finalize(end, true)
	}
	first := c.buf[:cut]
	u := Utterance{
		Start: c.utterStart,
		End:   c.utterStart + pcmDuration(first),
		PCM:   append([]byte(nil), first...),
	}
	c.buf = c.buf[cut:]
	c.utterStart = u.End
	c.speechAt = 0
	c.voicedDur = 0
	c.silence = 0
	return u, true
}

func (c *Cutter) resetUtter() {
	c.buf = nil
	c.speechAt = 0
	c.voicedDur = 0
	c.inUtter = false
	c.silence = 0
}

func pcmDuration(pcm []byte) time.Duration {
	samples := len(pcm) / 2
	if samples == 0 {
		return 0
	}
	return time.Duration(samples) * time.Second / cutterRate
}

func frameRMS(pcm []byte) float64 {
	n := len(pcm) / 2
	if n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		s := int16(binary.LittleEndian.Uint16(pcm[i*2:]))
		v := float64(s)
		sum += v * v
	}
	return math.Sqrt(sum / float64(n))
}

func quietestCutOffset(pcm []byte, search, window time.Duration) int {
	searchBytes := int(cutterRate*search/time.Second) * 2
	winBytes := int(cutterRate*window/time.Second) * 2
	frameBytes := int(cutterRate*cutterFrame/time.Second) * 2
	if len(pcm) < winBytes || searchBytes > len(pcm) {
		searchBytes = len(pcm)
	}
	tail := pcm[len(pcm)-searchBytes:]
	bestOff := 0
	best := math.MaxFloat64
	for off := 0; off+winBytes <= len(tail); off += frameBytes {
		rms := frameRMS(tail[off : off+winBytes])
		if rms < best {
			best = rms
			bestOff = off
		}
	}
	if best == math.MaxFloat64 {
		return 0
	}
	return len(pcm) - searchBytes + bestOff + winBytes/2
}
