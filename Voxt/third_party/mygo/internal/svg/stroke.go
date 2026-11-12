package svg

import "math"

// Line caps and joins.
const (
	capButt uint8 = iota
	capRound
	capSquare
)

const (
	joinMiter uint8 = iota
	joinRound
	joinBevel
)

// stroker outlines polylines as polygons that all turn the same way, so
// that the non-zero rule fills their union: a quadrilateral per segment,
// and the shapes of the joins and caps.
type stroker struct {
	hw        float64 // half the width
	join, cap uint8
	miter     float64 // the miter limit
	tol       float64 // how far arcs may stray from circles
	out       *polys
	pts       []point
}

func (s *stroker) stroke(in *polys) {
	for i := range in.len() {
		closed := in.closed[i]
		pts := s.pts[:0]
		for _, p := range in.poly(i) {
			if len(pts) == 0 || p != pts[len(pts)-1] {
				pts = append(pts, p)
			}
		}
		if closed && len(pts) > 1 && pts[0] == pts[len(pts)-1] {
			pts = pts[:len(pts)-1]
		}
		s.pts = pts
		n := len(pts)
		if n == 1 {
			s.dot(pts[0])
			continue
		}
		segs := n - 1
		if closed {
			segs = n
		}
		for k := range segs {
			s.segment(pts[k], pts[(k+1)%n])
		}
		if closed {
			for k := range n {
				s.joinAt(pts[(k+n-1)%n], pts[k], pts[(k+1)%n])
			}
			continue
		}
		for k := 1; k < n-1; k++ {
			s.joinAt(pts[k-1], pts[k], pts[k+1])
		}
		s.capAt(pts[0], pts[1])
		s.capAt(pts[n-1], pts[n-2])
	}
}

// full reports whether the outline has all the points it may have.
func (s *stroker) full() bool { return len(s.out.pts) >= 2*maxPoints }

// polygon adds a polygon to out, turned the way they all turn.
func (s *stroker) polygon(pts ...point) {
	if s.full() {
		return
	}
	o := s.out
	o.start(pts[0])
	for _, p := range pts[1:] {
		o.add(p)
	}
	s.finish()
}

// finish ends the polygon being built, reversing it if it turns the other
// way.
func (s *stroker) finish() {
	o := s.out
	begin := o.begin
	o.end(true)
	if len(o.starts) == 0 || o.starts[len(o.starts)-1] != begin {
		return
	}
	pts := o.pts[begin:]
	var area float64
	a := pts[len(pts)-1]
	for _, b := range pts {
		area += a.x*b.y - b.x*a.y
		a = b
	}
	if area < 0 {
		for i, j := 0, len(pts)-1; i < j; i, j = i+1, j-1 {
			pts[i], pts[j] = pts[j], pts[i]
		}
	}
}

func unit(a, b point) point {
	dx, dy := b.x-a.x, b.y-a.y
	l := math.Sqrt(dx*dx + dy*dy)
	return point{dx / l, dy / l}
}

func (s *stroker) segment(a, b point) {
	d := unit(a, b)
	n := point{-d.y * s.hw, d.x * s.hw}
	s.polygon(point{a.x + n.x, a.y + n.y}, point{b.x + n.x, b.y + n.y}, point{b.x - n.x, b.y - n.y}, point{a.x - n.x, a.y - n.y})
}

// joinAt joins the segments a→p and p→b on the outer side of the turn.
func (s *stroker) joinAt(a, p, b point) {
	d0, d1 := unit(a, p), unit(p, b)
	cross := d0.x*d1.y - d0.y*d1.x
	dot := d0.x*d1.x + d0.y*d1.y
	if math.Abs(cross) < 1e-12 && dot > 0 {
		return // straight on
	}
	side := s.hw
	if cross > 0 {
		side = -s.hw
	}
	n0 := point{-d0.y * side, d0.x * side}
	n1 := point{-d1.y * side, d1.x * side}
	o0 := point{p.x + n0.x, p.y + n0.y}
	o1 := point{p.x + n1.x, p.y + n1.y}
	switch s.join {
	case joinRound:
		turn := math.Atan2(math.Abs(cross), dot)
		if s.hw*(1-math.Cos(turn/2)) < s.tol {
			s.polygon(p, o0, o1)
			return
		}
		s.wedge(p, n0, math.Atan2(n0.x*n1.y-n0.y*n1.x, n0.x*n1.x+n0.y*n1.y))
	case joinBevel:
		s.polygon(p, o0, o1)
	default:
		// The miter's length over the stroke's width is 1/cos(turn/2).
		cosHalf := math.Sqrt(math.Max((1+dot)/2, 0))
		if cosHalf*s.miter < 1 {
			s.polygon(p, o0, o1)
			return
		}
		u := unit(point{}, point{n0.x + n1.x, n0.y + n1.y})
		l := s.hw / cosHalf
		s.polygon(p, o0, point{p.x + u.x*l, p.y + u.y*l}, o1)
	}
}

// capAt caps the end p of a polyline whose last segment comes from prev.
func (s *stroker) capAt(p, prev point) {
	d := unit(prev, p)
	n := point{d.y * s.hw, -d.x * s.hw}
	switch s.cap {
	case capRound:
		// Half a turn from n, through the direction d.
		s.wedge(p, n, math.Pi)
	case capSquare:
		e := point{d.x * s.hw, d.y * s.hw}
		s.polygon(point{p.x + n.x, p.y + n.y}, point{p.x + n.x + e.x, p.y + n.y + e.y},
			point{p.x - n.x + e.x, p.y - n.y + e.y}, point{p.x - n.x, p.y - n.y})
	}
}

// dot draws a subpath of no length: a disc with round caps, a square with
// square caps.
func (s *stroker) dot(p point) {
	switch s.cap {
	case capRound:
		s.wedge(p, point{s.hw, 0}, 2*math.Pi)
	case capSquare:
		h := s.hw
		s.polygon(point{p.x - h, p.y - h}, point{p.x + h, p.y - h}, point{p.x + h, p.y + h}, point{p.x - h, p.y + h})
	}
}

// wedge adds the slice of the disc around c of radius hw from the vector
// from, sweeping sweep radians.
func (s *stroker) wedge(c, from point, sweep float64) {
	if s.full() {
		return
	}
	step := 2 * math.Acos(math.Max(1-s.tol/s.hw, -1))
	n := 1
	if step > 0 {
		n = max(1, min(int(math.Ceil(math.Abs(sweep)/step)), 256))
	}
	a0 := math.Atan2(from.y, from.x)
	o := s.out
	full := math.Abs(sweep) >= 2*math.Pi
	if full {
		n = max(n, 8)
	}
	// Points between the ends lie a little outside the circle, so that
	// the polygon covers as much as the slice of the disc does.
	a := math.Abs(sweep) / float64(n)
	r := s.hw * math.Sqrt(a/math.Sin(a))
	if full {
		o.start(point{c.x + from.x*r/s.hw, c.y + from.y*r/s.hw})
	} else {
		o.start(c)
		o.add(point{c.x + from.x, c.y + from.y})
	}
	for k := 1; k < n; k++ {
		sin, cos := math.Sincos(a0 + sweep*float64(k)/float64(n))
		o.add(point{c.x + cos*r, c.y + sin*r})
	}
	if !full {
		sin, cos := math.Sincos(a0 + sweep)
		o.add(point{c.x + cos*s.hw, c.y + sin*s.hw})
	}
	s.finish()
}

// dash cuts the polylines of in into the dashes of a dash array (of even
// length), starting offset into it, as open polylines in out. limit
// bounds the number of dashes.
func dash(in *polys, dashes []float64, offset float64, out *polys, limit int) {
	var total float64
	for _, d := range dashes {
		total += d
	}
	offset = math.Mod(offset, total)
	if offset < 0 {
		offset += total
	}
	for i := range in.len() {
		pts := in.poly(i)
		// Every subpath starts the pattern anew, offset into it: in dash
		// or gap k, with left of it to go.
		k, o := 0, offset
		for o > dashes[k] {
			o -= dashes[k]
			k = (k + 1) % len(dashes)
		}
		left := dashes[k] - o
		on := k%2 == 0
		if on {
			out.start(pts[0])
		}
		n := len(pts)
		segs := n - 1
		if in.closed[i] {
			segs = n
		}
		for sg := range segs {
			a, b := pts[sg], pts[(sg+1)%n]
			l := math.Hypot(b.x-a.x, b.y-a.y)
			pos := 0.0
			for l-pos > left {
				pos += left
				t := pos / l
				p := point{a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t}
				if on {
					out.add(p)
					out.end(false)
					limit--
					if limit <= 0 {
						return
					}
				} else {
					out.start(p)
				}
				on = !on
				k = (k + 1) % len(dashes)
				left = dashes[k]
			}
			left -= l - pos
			if on {
				out.add(b)
			}
		}
		out.end(false)
	}
}
