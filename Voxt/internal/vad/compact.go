package vad

import (
	"encoding/binary"
	"time"
)

const fadeMS = 4 // the linear fade at each cut

// Span is a stretch of the original kept in the output, in samples, End
// exclusive. Spans are in order and do not overlap.
type Span struct{ Start, End int }

// Compacted is a recording with its silence trimmed.
type Compacted struct {
	PCM            []byte // what to send (the original slice itself when nothing is cut)
	Spans          []Span // the parts of the original in PCM, in order
	Original, Kept time.Duration
	// Changed is false when PCM is the original: there was no speech (see
	// Analysis.Speech) or compacting would save less than MinSavingMS.
	Changed bool
}

// Compact trims a's recording: it keeps LeadMS before the first speech and
// TailMS after the last, and shortens each pause between segments longer
// than LongGapMS to GapKeepMS. pcm must be the recording a describes
// (segments past its end are clipped); rate <= 0 is DefaultRate.
func Compact(pcm []byte, rate int, a Analysis, p Params) Compacted {
	if rate <= 0 {
		rate = DefaultRate
	}
	p = p.Resolved()
	n := len(pcm) / 2
	dur := func(samples int) time.Duration {
		return time.Duration(samples) * time.Second / time.Duration(rate)
	}
	orig := Compacted{PCM: pcm, Spans: []Span{{0, n}}, Original: dur(n), Kept: dur(n)}
	if n == 0 {
		orig.Spans = nil
		return orig
	}
	ms := func(v int) int { return v * rate / 1000 }
	spans := keptSpans(a.Segments, n, ms(p.LeadMS), ms(p.TailMS), ms(p.LongGapMS), ms(p.GapKeepMS))
	if len(spans) == 0 {
		return orig
	}
	kept := 0
	for _, s := range spans {
		kept += s.End - s.Start
	}
	if n-kept < ms(p.MinSavingMS) {
		return orig
	}
	out := make([]byte, 0, kept*2)
	fade := max(ms(fadeMS), 1)
	for _, s := range spans {
		at := len(out) / 2
		out = append(out, pcm[s.Start*2:s.End*2]...)
		f := min(fade, (s.End-s.Start)/2)
		if s.Start > 0 {
			ramp(out[at*2:], f, true)
		}
		if s.End < n {
			ramp(out[(at+s.End-s.Start-f)*2:], f, false)
		}
	}
	return Compacted{PCM: out, Spans: spans, Original: dur(n), Kept: dur(kept), Changed: true}
}

// keptSpans are the parts of a recording of n samples to keep around segs.
func keptSpans(segs []Segment, n, lead, tail, longGap, gapKeep int) []Span {
	var spans []Span
	for _, g := range segs {
		g.Start, g.End = max(g.Start, 0), min(g.End, n)
		if g.End <= g.Start {
			continue
		}
		if len(spans) == 0 {
			spans = append(spans, Span{max(g.Start-lead, 0), g.End})
			continue
		}
		last := &spans[len(spans)-1]
		if g.Start-last.End > longGap {
			last.End += gapKeep / 2
			spans = append(spans, Span{g.Start - (gapKeep - gapKeep/2), g.End})
			continue
		}
		last.End = max(last.End, g.End)
	}
	if len(spans) > 0 {
		last := &spans[len(spans)-1]
		last.End = min(last.End+tail, n)
	}
	return spans
}

// ramp fades the first f samples of b in (up) or out (!up), linearly.
func ramp(b []byte, f int, up bool) {
	for i := range f {
		g := (float64(i) + 0.5) / float64(f)
		if !up {
			g = 1 - g
		}
		v := int16(binary.LittleEndian.Uint16(b[i*2:]))
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(float64(v)*g)))
	}
}
