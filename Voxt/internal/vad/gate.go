package vad

// Gate holds a stream's leading silence (streaming ASR): nothing goes out
// until speech starts, then PreRollMS of what came before the onset and
// everything after it, pauses included (a live stream needs a
// continuous timeline). It opens anyway after MaxHoldMS of audio without
// speech, sending from PreRollMS before that point. While it holds, it
// keeps only the little audio it may still have to send.
//
// The detector keeps listening after the gate opens, so Heard turns true
// if speech comes after a MaxHold opening. A Gate is not safe for
// concurrent use.
type Gate struct {
	d       *Detector
	aux     *Detector   // Hear's, nil until it is called
	m       *modelOnset // UseModel's, nil for none
	preRoll int         // bytes
	maxHold int         // bytes
	keep    int         // bytes held at most while closed
	buf     []byte
	start   int // the stream offset of buf[0], in bytes (even)
	total   int // bytes pushed
	dropped int
	open    bool
}

// NewGate is a gate for PCM16 mono at rate Hz (rate <= 0: DefaultRate).
func NewGate(rate int, p Params) *Gate {
	d := NewDetector(rate, p)
	ms := func(v int) int { return v * d.rate / 1000 * 2 }
	pre := ms(d.p.PreRollMS)
	return &Gate{
		d:       d,
		preRoll: pre,
		maxHold: max(ms(d.p.MaxHoldMS), pre),
		// The onset is at most onsetN frames, plus a partial one, before
		// the detector notices it.
		keep: pre + (d.onsetN+2)*d.frame*2,
	}
}

// Push takes the next chunk and returns what to send now: nil while the
// gate holds; on opening the held audio from the pre-roll on; once open,
// pcm itself.
func (g *Gate) Push(pcm []byte) []byte {
	g.d.Write(pcm)
	if g.m != nil {
		g.m.feed(pcm)
	}
	g.total += len(pcm)
	if g.open {
		return pcm
	}
	g.buf = append(g.buf, pcm...)
	first := g.first()
	from := -1
	if first >= 0 {
		from = first*2 - g.preRoll
	}
	if g.total >= g.maxHold {
		if m := g.maxHold - g.preRoll; from < 0 || m < from {
			from = m
		}
	}
	if from >= 0 || first >= 0 {
		from = max(from, g.start)
		out := g.buf[from-g.start:]
		g.dropped += from - g.start
		g.buf, g.open = nil, true
		return out
	}
	if cut := (len(g.buf) - g.keep) &^ 1; cut > 0 {
		g.buf = g.buf[:copy(g.buf, g.buf[cut:])]
		g.start += cut
		g.dropped += cut
	}
	return nil
}

// Hear lets a second detector listen to another version of the stream
// (the recording before a filter, when Push gets it filtered): call it
// with each chunk just before Push gets that chunk's other version, of
// the same length. The gate opens when either detector hears speech, and
// Heard is either's: a filter can only make it open earlier.
func (g *Gate) Hear(pcm []byte) {
	if g.aux == nil {
		g.aux = NewDetector(g.d.rate, g.d.p)
	}
	g.aux.Write(pcm)
}

// UseModel lets a voice activity model decide where speech starts
// instead of the energy detectors (call it before the first Push): Push
// feeds it each chunk, and the gate opens on its onset. If the model
// fails, the gate goes on with the energy detectors, which listened all
// along (ModelErr says why).
func (g *Gate) UseModel(s ModelStream) {
	g.m = newModelOnset(s, g.d.rate, g.d.p)
	// The model's onset may lie a few chunks back.
	g.keep = max(g.keep, g.preRoll+(g.m.onsetN+2)*g.m.chunk*2)
}

// ModelErr is the model's failure, nil while it works (or without one).
func (g *Gate) ModelErr() error {
	if g.m == nil {
		return nil
	}
	return g.m.err
}

// first is the start of the first speech either detector heard; -1
// before any.
func (g *Gate) first() int {
	if g.m != nil && g.m.err == nil {
		return g.m.first
	}
	f := g.d.first
	if g.aux != nil && g.aux.first >= 0 && (f < 0 || g.aux.first < f) {
		f = g.aux.first
	}
	return f
}

// Open reports whether the gate has opened.
func (g *Gate) Open() bool { return g.open }

// Heard reports whether speech was detected; false when the gate opened by
// MaxHoldMS only and no speech came since.
func (g *Gate) Heard() bool { return g.first() >= 0 }

// Dropped is the number of bytes never sent: the leading silence (while the
// gate holds, those it already let go).
func (g *Gate) Dropped() int { return g.dropped }
