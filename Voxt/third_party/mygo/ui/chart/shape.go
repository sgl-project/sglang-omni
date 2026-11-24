package chart

import (
	"math"

	"github.com/egoist/mygo/ui"
)

// X returns the x of f, a fraction along a plot's width, in r, the plot's
// box a Draw function gets.
func X(r ui.Rect, f float32) float32 { return r.X + f*r.W }

// Y returns the y of f, a fraction up a plot's height, in r: 0 is its
// bottom and 1 its top.
func Y(r ui.Rect, f float32) float32 { return r.Y + (1-f)*r.H }

// Curve is how a line goes from a point to the next, as Swift Charts'
// interpolation methods and d3's curves.
type Curve int

const (
	// CurveLinear joins the points with straight lines.
	CurveLinear Curve = iota
	// CurveMonotone joins them with a smooth curve that keeps between the
	// values of each two points, never going past them, as d3's
	// curveMonotoneX (Steffen's method).
	CurveMonotone
	// CurveStepAfter keeps a point's value until the next point, as a value
	// that holds until it changes.
	CurveStepAfter
	// CurveStepBefore takes on a point's value from the point before it.
	CurveStepBefore
	// CurveStep changes the value halfway between points.
	CurveStep
)

// segment is a piece of a line, from where the one before ended to to:
// straight unless cubic, with the control points c1 and c2.
type segment struct {
	cubic      bool
	c1, c2, to [2]float32
}

// curve returns the segments through pts, after the first point, as the
// curve goes.
func curve(pts [][2]float32, k Curve) []segment {
	n := len(pts)
	segs := make([]segment, 0, n)
	switch k {
	case CurveMonotone:
		if n < 3 {
			break
		}
		// Each point's slope, as d3's curveMonotoneX: Steffen's at the
		// points inside, and at the ends one the last segment's slope and
		// the neighbor's give.
		m := make([]float32, n)
		for i := 1; i < n-1; i++ {
			m[i] = monotoneSlope(pts[i-1], pts[i], pts[i+1])
		}
		m[0] = endSlope(pts[0], pts[1], m[1])
		m[n-1] = endSlope(pts[n-2], pts[n-1], m[n-2])
		for i := 1; i < n; i++ {
			p0, p1 := pts[i-1], pts[i]
			dx := (p1[0] - p0[0]) / 3
			segs = append(segs, segment{cubic: true,
				c1: [2]float32{p0[0] + dx, p0[1] + dx*m[i-1]},
				c2: [2]float32{p1[0] - dx, p1[1] - dx*m[i]},
				to: p1})
		}
		return segs
	case CurveStepAfter, CurveStepBefore, CurveStep:
		for i := 1; i < n; i++ {
			p0, p1 := pts[i-1], pts[i]
			switch k {
			case CurveStepAfter:
				segs = append(segs, segment{to: [2]float32{p1[0], p0[1]}})
			case CurveStepBefore:
				segs = append(segs, segment{to: [2]float32{p0[0], p1[1]}})
			default:
				mid := (p0[0] + p1[0]) / 2
				segs = append(segs, segment{to: [2]float32{mid, p0[1]}}, segment{to: [2]float32{mid, p1[1]}})
			}
			segs = append(segs, segment{to: p1})
		}
		return segs
	}
	for _, p := range pts[1:] {
		segs = append(segs, segment{to: p})
	}
	return segs
}

// monotoneSlope returns the slope at b between a and c, as d3's slope3.
func monotoneSlope(a, b, c [2]float32) float32 {
	h0, h1 := b[0]-a[0], c[0]-b[0]
	if h0 == 0 || h1 == 0 {
		return 0
	}
	s0, s1 := (b[1]-a[1])/h0, (c[1]-b[1])/h1
	p := (s0*h1 + s1*h0) / (h0 + h1)
	if math.Signbit(float64(s0)) != math.Signbit(float64(s1)) || s0 == 0 || s1 == 0 {
		return 0
	}
	m := min(abs(s0), abs(s1), 0.5*abs(p))
	if s0 < 0 {
		return -2 * m
	}
	return 2 * m
}

// endSlope returns the slope at an end of a line, from the slope t at the
// point next to it, as d3's slope2.
func endSlope(a, b [2]float32, t float32) float32 {
	h := b[0] - a[0]
	if h == 0 {
		return t
	}
	return (3*(b[1]-a[1])/h - t) / 2
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// defined reports whether every value is a number, not NaN: a missing
// value breaks a line, as d3's line.defined.
func defined(vs ...float32) bool {
	for _, v := range vs {
		if v != v {
			return false
		}
	}
	return true
}

// runs calls fn with each run of points whose values are all defined.
func runs(n int, ok func(i int) bool, fn func(from, to int)) {
	from := -1
	for i := 0; i <= n; i++ {
		if i < n && ok(i) {
			if from < 0 {
				from = i
			}
			continue
		}
		if from >= 0 {
			fn(from, i)
			from = -1
		}
	}
}

// Line returns the path of a line through the points at xs and ys,
// fractions along and up the plot whose box is r, joined as curve says.
// A NaN value breaks the line there. Stroke it in a Draw function of the
// plot:
//
//	area.Draw(func(p *ui.Painter, r ui.Rect) {
//		p.StrokePath(chart.Line(r, xs, ys, chart.CurveMonotone), 2, t.Accent)
//	})
func Line(r ui.Rect, xs, ys []float32, k Curve) *ui.Path {
	return LineInto(&ui.Path{}, r, xs, ys, k)
}

// LineInto replaces path with Line's geometry, reusing its storage. It also
// clears any Trim. Keep one path per series when drawing live waveforms.
func LineInto(path *ui.Path, r ui.Rect, xs, ys []float32, k Curve) *ui.Path {
	path.Reset()
	n := min(len(xs), len(ys))
	runs(n, func(i int) bool { return defined(xs[i], ys[i]) }, func(from, to int) {
		if k == CurveLinear {
			path.MoveTo(X(r, xs[from]), Y(r, ys[from]))
			for i := from + 1; i < to; i++ {
				path.LineTo(X(r, xs[i]), Y(r, ys[i]))
			}
			return
		}
		pts := make([][2]float32, 0, to-from)
		for i := from; i < to; i++ {
			pts = append(pts, [2]float32{X(r, xs[i]), Y(r, ys[i])})
		}
		path.MoveTo(pts[0][0], pts[0][1])
		addSegments(path, curve(pts, k))
	})
	return path
}

// Area returns the path of the area between two lines through the points
// at xs, lo below and hi above, joined as curve says, for areas and bands
// and the layers of a stack. Nil lo puts the bottom line at the bottom of
// the plot. A NaN value breaks the area there.
//
//	area.Draw(func(p *ui.Painter, r ui.Rect) {
//		p.FillPath(chart.Area(r, xs, nil, ys, chart.CurveMonotone), fill)
//	})
func Area(r ui.Rect, xs, lo, hi []float32, k Curve) *ui.Path {
	path := &ui.Path{}
	n := min(len(xs), len(hi))
	if lo != nil {
		n = min(n, len(lo))
	}
	low := func(i int) float32 {
		if lo == nil {
			return 0
		}
		return lo[i]
	}
	runs(n, func(i int) bool { return defined(xs[i], hi[i], low(i)) }, func(from, to int) {
		top := make([][2]float32, 0, to-from)
		bottom := make([][2]float32, 0, to-from)
		for i := from; i < to; i++ {
			top = append(top, [2]float32{X(r, xs[i]), Y(r, hi[i])})
			bottom = append(bottom, [2]float32{X(r, xs[i]), Y(r, low(i))})
		}
		path.MoveTo(top[0][0], top[0][1])
		addSegments(path, curve(top, k))
		// Back along the bottom line, the same curve the other way.
		segs := curve(bottom, k)
		last := bottom[len(bottom)-1]
		path.LineTo(last[0], last[1])
		for i := len(segs) - 1; i >= 0; i-- {
			from := bottom[0]
			if i > 0 {
				from = segs[i-1].to
			}
			if s := segs[i]; s.cubic {
				path.CubeTo(s.c2[0], s.c2[1], s.c1[0], s.c1[1], from[0], from[1])
			} else {
				path.LineTo(from[0], from[1])
			}
		}
		path.Close()
	})
	return path
}

func addSegments(path *ui.Path, segs []segment) {
	for _, s := range segs {
		if s.cubic {
			path.CubeTo(s.c1[0], s.c1[1], s.c2[0], s.c2[1], s.to[0], s.to[1])
		} else {
			path.LineTo(s.to[0], s.to[1])
		}
	}
}

// Nearest returns the index of the value of xs, in ascending order, that
// is nearest f, or -1 for none: the point of a line under the pointer.
func Nearest(xs []float32, f float32) int {
	if len(xs) == 0 {
		return -1
	}
	lo, hi := 0, len(xs)
	for lo < hi { // the first at or after f
		m := (lo + hi) / 2
		if xs[m] < f {
			lo = m + 1
		} else {
			hi = m
		}
	}
	switch {
	case lo == len(xs):
		return lo - 1
	case lo > 0 && f-xs[lo-1] <= xs[lo]-f:
		return lo - 1
	}
	return lo
}

// Slice is a value's part of a pie, from the angle Start to End, in
// radians clockwise from twelve o'clock, as d3's pie and Swift Charts'
// sector marks. Pad is the gap Sector leaves between it and the next.
type Slice struct {
	Start, End, Pad float64
}

// Pie divides a circle among values, in their order from twelve o'clock,
// each slice as large as its share of their sum. Pad is the angle between
// slices, in radians; values below 0, and NaN, take no room.
//
//	slices := chart.Pie(shares, 0.02)
func Pie(values []float64, pad float64) []Slice {
	sum := 0.0
	for _, v := range values {
		if v > 0 {
			sum += v
		}
	}
	n := float64(len(values))
	pad = max(0, min(pad, 2*math.Pi/max(n, 1)))
	k := 0.0
	if sum > 0 {
		k = (2*math.Pi - n*pad) / sum
	}
	out := make([]Slice, len(values))
	a := 0.0
	for i, v := range values {
		end := a + pad
		if v > 0 {
			end += v * k
		}
		out[i] = Slice{Start: a, End: end, Pad: pad}
		a = end
	}
	return out
}

// Mid returns the angle halfway through the slice, where its label goes.
func (s Slice) Mid() float64 { return (s.Start + s.End) / 2 }

// Polar returns the point at the angle a, r from the center (cx, cy), as
// where a slice's label goes.
func Polar(cx, cy, r float32, a float64) (x, y float32) {
	return cx + r*float32(math.Sin(a)), cy - r*float32(math.Cos(a))
}

// Sector returns the path of a slice between the radii inner and outer
// around (cx, cy): a wedge of a pie with inner 0, of a donut otherwise.
// The slice's pad keeps a gap of the same width along it, as d3's arcs do.
//
//	ring.Draw(func(p *ui.Painter, r ui.Rect) {
//		cx, cy := r.X+r.W/2, r.Y+r.H/2
//		for i, s := range slices {
//			p.FillPath(chart.Sector(cx, cy, 40, 80, s), colors[i])
//		}
//	})
func Sector(cx, cy, inner, outer float32, s Slice) *ui.Path {
	path := &ui.Path{}
	if outer <= 0 || s.End <= s.Start {
		return path
	}
	inner = max(0, min(inner, outer))
	// The gap's width, at the radius between the two, as d3's padRadius.
	gap := s.Pad * math.Sqrt(float64(inner*inner+outer*outer))
	angles := func(r float32) (a0, a1 float64) {
		a0, a1 = s.Start, s.End
		if gap > 0 && r > 0 {
			d := math.Asin(min(1, gap/2/float64(r)))
			if a1-a0 > 2*d {
				return a0 + d, a1 - d
			}
			m := (a0 + a1) / 2
			return m, m
		}
		return a0, a1
	}
	a0, a1 := angles(outer)
	x, y := Polar(cx, cy, outer, a0)
	path.MoveTo(x, y)
	arc(path, cx, cy, outer, a0, a1)
	if inner > 0 {
		b0, b1 := angles(inner)
		x, y = Polar(cx, cy, inner, b1)
		path.LineTo(x, y)
		arc(path, cx, cy, inner, b1, b0)
	} else {
		path.LineTo(cx, cy)
	}
	path.Close()
	return path
}

// arc adds an arc of radius r around (cx, cy) from the angle a0 to a1 to
// path, in cubic curves of up to a quarter turn.
func arc(path *ui.Path, cx, cy, r float32, a0, a1 float64) {
	n := int(math.Ceil(math.Abs(a1-a0) / (math.Pi / 2)))
	if n == 0 {
		return
	}
	step := (a1 - a0) / float64(n)
	k := float32(4.0 / 3 * math.Tan(step/4))
	for i := range n {
		b0, b1 := a0+float64(i)*step, a0+float64(i+1)*step
		x0, y0 := Polar(cx, cy, r, b0)
		x1, y1 := Polar(cx, cy, r, b1)
		// The tangent at angle a, going clockwise, is (cos a, sin a).
		t0x, t0y := float32(math.Cos(b0)), float32(math.Sin(b0))
		t1x, t1y := float32(math.Cos(b1)), float32(math.Sin(b1))
		path.CubeTo(x0+k*r*t0x, y0+k*r*t0y, x1-k*r*t1x, y1-k*r*t1y, x1, y1)
	}
}

// SliceAt returns the slice at (x, y), relative to the center of a pie
// whose slices go from inner to outer, and whether there is one there:
// the slice under the pointer.
func SliceAt(slices []Slice, x, y, inner, outer float32) (int, bool) {
	d := float32(math.Hypot(float64(x), float64(y)))
	if d < inner || d > outer {
		return 0, false
	}
	a := math.Atan2(float64(x), -float64(y))
	if a < 0 {
		a += 2 * math.Pi
	}
	for i, s := range slices {
		if a >= s.Start && a < s.End {
			return i, true
		}
	}
	return 0, false
}
