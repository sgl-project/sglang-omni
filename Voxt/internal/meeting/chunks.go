package meeting

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// Chunk is a slice of PCM16 samples to send through file-mode ASR.
type Chunk struct {
	Start, End int // half-open sample indices
}

// ChunkParams tunes PlanChunks (design C1 §4.7).
type ChunkParams struct {
	Rate          int
	MaxDur        time.Duration
	MinDur        time.Duration
	PauseMin      time.Duration
	Pad           time.Duration
	HardOverlap   time.Duration
	CapMaxDur     time.Duration // remote ASR cap; 0 uses MaxDur only
}

// DefaultChunkParams matches the approved design defaults.
func DefaultChunkParams() ChunkParams {
	return ChunkParams{
		Rate:        asr.SampleRate,
		MaxDur:      22 * time.Second,
		MinDur:      600 * time.Millisecond,
		PauseMin:    850 * time.Millisecond,
		Pad:         120 * time.Millisecond,
		HardOverlap: time.Second,
	}
}

func (p ChunkParams) rate() int {
	if p.Rate > 0 {
		return p.Rate
	}
	return asr.SampleRate
}

func (p ChunkParams) maxSamples() int {
	d := p.MaxDur
	if p.CapMaxDur > 0 && p.CapMaxDur < d {
		d = p.CapMaxDur
	}
	if d <= 0 {
		d = 22 * time.Second
	}
	return int(d.Seconds()*float64(p.rate()) + 0.999)
}

func (p ChunkParams) minSamples() int {
	d := p.MinDur
	if d <= 0 {
		d = 600 * time.Millisecond
	}
	return int(d.Seconds() * float64(p.rate()))
}

func (p ChunkParams) pauseSamples() int {
	d := p.PauseMin
	if d <= 0 {
		d = 850 * time.Millisecond
	}
	return int(d.Seconds() * float64(p.rate()))
}

func (p ChunkParams) padSamples() int {
	d := p.Pad
	if d <= 0 {
		d = 120 * time.Millisecond
	}
	return int(d.Seconds() * float64(p.rate()))
}

func (p ChunkParams) overlapSamples() int {
	d := p.HardOverlap
	if d <= 0 {
		d = time.Second
	}
	return int(d.Seconds() * float64(p.rate()))
}

// PlanChunks splits speech regions into ASR-sized chunks with padding and overlap.
func PlanChunks(speech []vad.Segment, totalSamples int, p ChunkParams) []Chunk {
	if len(speech) == 0 || totalSamples <= 0 {
		return nil
	}
	segs := append([]vad.Segment(nil), speech...)
	max := p.maxSamples()
	if max <= 0 {
		return nil
	}
	var out []Chunk
	for i := 0; i < len(segs); {
		spanStart := segs[i].Start
		spanEnd := segs[i].End
		j := i + 1
		for j < len(segs) && segs[j].Start-spanEnd <= p.pauseSamples() {
			spanEnd = segs[j].End
			j++
		}
		out = append(out, splitSpan(segs[i:j], spanStart, spanEnd, totalSamples, p)...)
		i = j
	}
	out = mergeShortChunks(out, p.minSamples())
	return out
}

func splitSpan(parts []vad.Segment, spanStart, spanEnd, total int, p ChunkParams) []Chunk {
	max := p.maxSamples()
	var out []Chunk
	cur := spanStart
	for cur < spanEnd {
		lim := cur + max
		if lim >= spanEnd {
			out = append(out, clampChunk(cur, spanEnd, total, p))
			break
		}
		cut, hard := pickCut(parts, cur, lim, p.pauseSamples())
		if cut <= cur {
			cut = lim
			hard = true
		}
		out = append(out, clampChunk(cur, cut, total, p))
		if hard {
			cur = cut - p.overlapSamples()
		} else {
			cur = cut
		}
		if cur < spanStart {
			cur = spanStart
		}
	}
	return out
}

func pickCut(parts []vad.Segment, from, lim, pauseMin int) (cut int, hard bool) {
	best := -1
	for k := 0; k+1 < len(parts); k++ {
		gapStart := parts[k].End
		gapEnd := parts[k+1].Start
		if gapEnd <= from || gapStart >= lim {
			continue
		}
		gap := gapEnd - gapStart
		if gap >= pauseMin && gapStart > best && gapStart <= lim {
			best = gapStart + gap/2
		}
	}
	if best > from && best <= lim {
		return best, false
	}
	return lim, true
}

func clampChunk(start, end, total int, p ChunkParams) Chunk {
	pad := p.padSamples()
	s := max(0, start-pad)
	e := min(total, end+pad)
	if e <= s {
		e = min(total, s+1)
	}
	return Chunk{Start: s, End: e}
}

func mergeShortChunks(in []Chunk, minSamples int) []Chunk {
	if len(in) <= 1 || minSamples <= 0 {
		return in
	}
	out := make([]Chunk, 0, len(in))
	for _, c := range in {
		if len(out) == 0 {
			out = append(out, c)
			continue
		}
		if c.End-c.Start >= minSamples {
			out = append(out, c)
			continue
		}
		prev := &out[len(out)-1]
		prev.End = max(prev.End, c.End)
	}
	if len(out) <= 1 {
		return out
	}
	// merge trailing short chunk into previous
	last := out[len(out)-1]
	if last.End-last.Start < minSamples {
		prev := &out[len(out)-2]
		prev.End = max(prev.End, last.End)
		return out[:len(out)-1]
	}
	return out
}

// DedupeOverlapText trims duplicated words where hard-cut chunks overlapped in time.
func DedupeOverlapText(prev, next string) string {
	if prev == "" || next == "" {
		return next
	}
	wordsP := splitWords(prev)
	wordsN := splitWords(next)
	maxK := min(len(wordsP), len(wordsN))
	for k := maxK; k > 0; k-- {
		if joinWords(wordsP[len(wordsP)-k:]) == joinWords(wordsN[:k]) {
			return joinWords(wordsN[k:])
		}
	}
	return next
}

func splitWords(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' {
			if cur != "" {
				out = append(out, cur)
				cur = ""
			}
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

func joinWords(w []string) string {
	if len(w) == 0 {
		return ""
	}
	s := w[0]
	for _, x := range w[1:] {
		s += " " + x
	}
	return s
}
