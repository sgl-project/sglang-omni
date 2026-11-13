package audio

import "math"

// voices are the voices playing, or fading out, under mu.
var voices []*voice

// rampFrames is how long volume changes, fades in and fades out take: 5 ms.
const rampFrames = sampleRate / 200

// A source gives a voice its frames. Its methods run under mu and never
// block.
type source interface {
	// read reads frames into dst, interleaved, and returns how many it
	// read: fewer than fit only at the end, where ended is true, or while
	// a stream decodes, where it is false. It goes on from the start when
	// looping.
	read(dst []float32) (n int, ended bool)
	seek(frame int64)
	// setLoop sets whether the source loops, from frame start to end
	// (0 for its end).
	setLoop(loop bool, start, end int64)
	close()
}

// A voice is a player's state: its source, where it is, and how it
// resamples to the output.
type voice struct {
	src    source
	rate   float64
	mono   bool
	length int64 // frames, or -1

	playing bool // as the API sees it
	active  bool // in voices, which it stays in while fading out
	snap    bool // starts at its start: no fade in
	fadeIn  bool // resumes in the middle: fades in from silence
	closed  bool
	orphan  bool // its Player is gone: close at the end

	volume, pan, speed float64
	loop               bool
	loopStart, loopEnd int64 // frames, loopEnd 0 for the end

	// The four source frames the output is interpolated between, the
	// second at idx, and how far the output is past it.
	hist  [4][2]float32
	idx   int64
	phase float64
	end   int64 // the frame after the last, once the source ended, or -1

	in    []float32 // frames read from the source and not yet used
	inPos int
	inLen int

	// The gains applied, a 2×2 matrix from the source's channels to the
	// output's (ll, rl, lr, rr), ramping to target.
	gain, target, step [4]float32
	rampLeft           int
}

// reset starts the voice over at frame f of its source, which the caller
// has moved there.
func (v *voice) reset(f int64) {
	v.hist = [4][2]float32{}
	// Three frames come in before the first output: the second is f.
	v.idx = f - 3
	v.phase = 3
	v.end = -1
	v.inPos, v.inLen = 0, 0
}

// ended reports whether the voice played to the end of its source.
func (v *voice) ended() bool { return v.end >= 0 && v.idx >= v.end }

// position returns the source frame the voice is at.
func (v *voice) position() float64 {
	p := max(float64(v.idx)+v.phase, 0)
	if v.length == 0 {
		return 0
	}
	if v.loop {
		end := float64(v.loopEnd)
		if end <= 0 {
			end = float64(v.length)
		}
		if start := float64(v.loopStart); end > start && p >= end {
			p = start + math.Mod(p-start, end-start)
		}
		return p
	}
	if v.length > 0 {
		p = min(p, float64(v.length))
	}
	return p
}

func (v *voice) close() {
	if v.closed {
		return
	}
	v.closed = true
	v.playing = false
	v.src.close()
}

// next shifts in the source's next frame, and reports false while a stream
// has none decoded yet.
func (v *voice) next() bool {
	ch := 2
	if v.mono {
		ch = 1
	}
	if v.inPos >= v.inLen {
		if v.end >= 0 {
			// Past the end, silence.
			v.shift(0, 0)
			return true
		}
		if v.in == nil {
			v.in = make([]float32, 256*ch)
		}
		n, ended := v.src.read(v.in)
		v.inPos, v.inLen = 0, n*ch
		if n == 0 {
			if !ended {
				return false
			}
			// The frame coming in would be at idx+3.
			v.end = v.idx + 3
			v.shift(0, 0)
			return true
		}
	}
	l := v.in[v.inPos]
	r := l
	if !v.mono {
		r = v.in[v.inPos+1]
	}
	v.inPos += ch
	v.shift(l, r)
	return true
}

func (v *voice) shift(l, r float32) {
	v.hist[0], v.hist[1], v.hist[2] = v.hist[1], v.hist[2], v.hist[3]
	v.hist[3] = [2]float32{l, r}
	v.idx++
}

// gains returns the matrix of the voice's volume and pan, and the master
// volume.
func (v *voice) gains(master float64) [4]float32 {
	g := v.volume * master
	if !v.playing {
		g = 0
	}
	if v.mono {
		// Equal power, the middle as decoded.
		x := (v.pan + 1) / 2 * math.Pi / 2
		return [4]float32{float32(g * math.Sqrt2 * math.Cos(x)), 0, 0, float32(g * math.Sqrt2 * math.Sin(x))}
	}
	// Web Audio's StereoPannerNode for stereo input.
	if v.pan <= 0 {
		x := (v.pan + 1) * math.Pi / 2
		return [4]float32{float32(g), 0, float32(g * math.Cos(x)), float32(g * math.Sin(x))}
	}
	x := v.pan * math.Pi / 2
	return [4]float32{float32(g * math.Cos(x)), float32(g * math.Sin(x)), 0, float32(g)}
}

// retarget ramps the gains to t from now.
func (v *voice) retarget(t [4]float32) {
	switch {
	case v.snap:
		v.snap, v.fadeIn = false, false
		v.gain, v.target, v.rampLeft = t, t, 0
		return
	case v.fadeIn:
		v.fadeIn = false
		v.gain = [4]float32{}
	case t == v.target:
		return
	}
	v.target = t
	for i := range t {
		v.step[i] = (t[i] - v.gain[i]) / rampFrames
	}
	v.rampLeft = rampFrames
}

// mix adds the voice's next frames to out and reports whether it is still
// active: false once at its end, or faded out.
func (v *voice) mix(out []float32, master float64) bool {
	v.retarget(v.gains(master))
	step := v.rate / sampleRate * v.speed
	for i := 0; i < len(out); i += 2 {
		for v.phase >= 1 {
			if !v.next() {
				// The stream is late: silence until it catches up.
				return true
			}
			v.phase--
		}
		if v.ended() {
			v.playing = false
			return false
		}
		if v.rampLeft > 0 {
			for k := range v.gain {
				v.gain[k] += v.step[k]
			}
			v.rampLeft--
			if v.rampLeft == 0 {
				v.gain = v.target
			}
		} else if !v.playing {
			// Faded out.
			return false
		}
		l, r := v.sample()
		out[i] += l*v.gain[0] + r*v.gain[2]
		out[i+1] += l*v.gain[1] + r*v.gain[3]
		v.phase += step
	}
	return true
}

// sample interpolates between the second and third frames with a
// Catmull-Rom spline.
func (v *voice) sample() (l, r float32) {
	t := float32(v.phase)
	h := &v.hist
	cr := func(p0, p1, p2, p3 float32) float32 {
		return p1 + 0.5*t*(p2-p0+t*(2*p0-5*p1+4*p2-p3+t*(3*(p1-p2)+p3-p0)))
	}
	return cr(h[0][0], h[1][0], h[2][0], h[3][0]), cr(h[0][1], h[1][1], h[2][1], h[3][1])
}

// mix fills out, interleaved stereo, with the voices playing, and reports
// whether any is.
func mix(out []float32) bool {
	clear(out)
	mu.Lock()
	defer mu.Unlock()
	n := 0
	for _, v := range voices {
		if v.closed || !v.mix(out, volume) {
			v.active = false
			if v.orphan {
				v.close()
			}
			continue
		}
		voices[n] = v
		n++
	}
	clear(voices[n:])
	voices = voices[:n]
	for i, s := range out {
		out[i] = min(max(s, -1), 1)
	}
	return n > 0
}
