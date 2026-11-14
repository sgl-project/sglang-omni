package svg

import (
	"math"
	"strconv"
)

type point struct{ x, y float64 }

// box is a rectangle in user units.
type box struct{ x, y, w, h float64 }

func (b box) union(o box) box {
	x0, y0 := math.Min(b.x, o.x), math.Min(b.y, o.y)
	x1, y1 := math.Max(b.x+b.w, o.x+o.w), math.Max(b.y+b.h, o.y+o.h)
	return box{x0, y0, x1 - x0, y1 - y0}
}

// transform returns the box around b transformed by m.
func (b box) transform(m matrix) box {
	out := box{}
	for i, p := range [4]point{{b.x, b.y}, {b.x + b.w, b.y}, {b.x, b.y + b.h}, {b.x + b.w, b.y + b.h}} {
		q := m.apply(p)
		if i == 0 {
			out = box{q.x, q.y, 0, 0}
		} else {
			out = out.union(box{q.x, q.y, 0, 0})
		}
	}
	return out
}

// matrix is an affine transform: it maps (x, y) to (m[0]x + m[2]y + m[4],
// m[1]x + m[3]y + m[5]), as SVG's matrix(a, b, c, d, e, f) does.
type matrix [6]float64

var identity = matrix{1, 0, 0, 1, 0, 0}

// mul returns the transform that applies n, then m.
func (m matrix) mul(n matrix) matrix {
	return matrix{
		m[0]*n[0] + m[2]*n[1],
		m[1]*n[0] + m[3]*n[1],
		m[0]*n[2] + m[2]*n[3],
		m[1]*n[2] + m[3]*n[3],
		m[0]*n[4] + m[2]*n[5] + m[4],
		m[1]*n[4] + m[3]*n[5] + m[5],
	}
}

func (m matrix) apply(p point) point {
	return point{m[0]*p.x + m[2]*p.y + m[4], m[1]*p.x + m[3]*p.y + m[5]}
}

// invert returns the inverse of m, or false when m flattens the plane.
func (m matrix) invert() (matrix, bool) {
	d := m[0]*m[3] - m[1]*m[2]
	if d == 0 || math.IsNaN(d) || math.IsInf(d, 0) {
		return identity, false
	}
	return matrix{
		m[3] / d, -m[1] / d, -m[2] / d, m[0] / d,
		(m[2]*m[5] - m[3]*m[4]) / d,
		(m[1]*m[4] - m[0]*m[5]) / d,
	}, true
}

// stretch returns the most m lengthens a line: its largest singular value.
func (m matrix) stretch() float64 {
	a, b, c, d := m[0], m[1], m[2], m[3]
	e := a*a + b*b - c*c - d*d
	f := a*c + b*d
	return math.Sqrt((a*a + b*b + c*c + d*d + math.Sqrt(e*e+4*f*f)) / 2)
}

func translate(x, y float64) matrix { return matrix{1, 0, 0, 1, x, y} }

func scaling(x, y float64) matrix { return matrix{x, 0, 0, y, 0, 0} }

func rotation(deg float64) matrix {
	s, c := math.Sincos(deg * math.Pi / 180)
	return matrix{c, s, -s, c, 0, 0}
}

// boxMatrix maps the unit square to b, for objectBoundingBox units.
func boxMatrix(b box) matrix { return matrix{b.w, 0, 0, b.h, b.x, b.y} }

// parseTransform parses a transform list, such as "translate(10 20)
// rotate(45)". A list in error transforms nothing, as in browsers.
func parseTransform(s string) matrix {
	m := identity
	p := scanner{s: s}
	for {
		p.skipSep()
		if p.done() {
			return m
		}
		name := p.ident()
		p.skipSpace()
		if !p.eat('(') {
			return identity
		}
		var a [6]float64
		n := 0
		for {
			p.skipSep()
			if p.eat(')') {
				break
			}
			v, ok := p.number()
			if !ok || n == len(a) {
				return identity
			}
			a[n] = v
			n++
		}
		var t matrix
		switch {
		case name == "matrix" && n == 6:
			t = matrix(a)
		case name == "translate" && (n == 1 || n == 2):
			t = translate(a[0], a[1])
		case name == "scale" && n == 1:
			t = scaling(a[0], a[0])
		case name == "scale" && n == 2:
			t = scaling(a[0], a[1])
		case name == "rotate" && n == 1:
			t = rotation(a[0])
		case name == "rotate" && n == 3:
			t = translate(a[1], a[2]).mul(rotation(a[0])).mul(translate(-a[1], -a[2]))
		case name == "skewX" && n == 1:
			t = matrix{1, 0, math.Tan(a[0] * math.Pi / 180), 1, 0, 0}
		case name == "skewY" && n == 1:
			t = matrix{1, math.Tan(a[0] * math.Pi / 180), 0, 1, 0, 0}
		default:
			return identity
		}
		m = m.mul(t)
	}
}

// scanner reads the numbers, names and separators of attribute values.
type scanner struct {
	s string
	i int
}

func (p *scanner) done() bool { return p.i >= len(p.s) }

func (p *scanner) peek() byte {
	if p.i < len(p.s) {
		return p.s[p.i]
	}
	return 0
}

func (p *scanner) eat(c byte) bool {
	if p.i < len(p.s) && p.s[p.i] == c {
		p.i++
		return true
	}
	return false
}

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' }

func (p *scanner) skipSpace() {
	for p.i < len(p.s) && isSpace(p.s[p.i]) {
		p.i++
	}
}

// skipSep skips white space with at most one comma in it.
func (p *scanner) skipSep() {
	p.skipSpace()
	if p.eat(',') {
		p.skipSpace()
	}
}

// ident reads a name of letters, digits, hyphens and underscores.
func (p *scanner) ident() string {
	start := p.i
	for p.i < len(p.s) {
		c := p.s[p.i]
		if c == '-' || c == '_' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= 0x80 {
			p.i++
			continue
		}
		break
	}
	return p.s[start:p.i]
}

// number reads a finite number: an optional sign, digits with an optional
// fraction, and an optional exponent.
func (p *scanner) number() (float64, bool) {
	s, start := p.s, p.i
	i := start
	if i < len(s) && (s[i] == '+' || s[i] == '-') {
		i++
	}
	digits := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
		digits++
	}
	if i < len(s) && s[i] == '.' {
		i++
		for i < len(s) && s[i] >= '0' && s[i] <= '9' {
			i++
			digits++
		}
	}
	if digits == 0 {
		return 0, false
	}
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		// An exponent, unless the e starts a unit such as em.
		j := i + 1
		if j < len(s) && (s[j] == '+' || s[j] == '-') {
			j++
		}
		if j < len(s) && s[j] >= '0' && s[j] <= '9' {
			for j < len(s) && s[j] >= '0' && s[j] <= '9' {
				j++
			}
			i = j
		}
	}
	v, err := strconv.ParseFloat(s[start:i], 64)
	if err != nil || math.IsInf(v, 0) {
		return 0, false
	}
	p.i = i
	return v, true
}

// numbers parses a list of numbers separated by white space or commas,
// up to the first that is not one.
func numbers(s string) []float64 {
	var out []float64
	p := scanner{s: s}
	for {
		p.skipSep()
		v, ok := p.number()
		if !ok {
			return out
		}
		out = append(out, v)
	}
}

// path is a path in user units. Its curves are cubic Bézier curves:
// quadratic curves and arcs are converted.
type path struct {
	ops []pathOp
	pts []point
	// start is the first point of the subpath, cur the pen; open is false
	// before the first move and after a close.
	start, cur point
	open       bool
}

type pathOp uint8

const (
	opMove  pathOp = iota // a point
	opLine                // a point
	opCubic               // two control points and the end
	opClose               // no point
)

func (pa *path) moveTo(p point) {
	pa.ops = append(pa.ops, opMove)
	pa.pts = append(pa.pts, p)
	pa.start, pa.cur, pa.open = p, p, true
}

// begin starts a subpath where the pen is, when drawing follows a close:
// the subpath after a close starts where the closed one did.
func (pa *path) begin() {
	if !pa.open {
		pa.moveTo(pa.cur)
	}
}

func (pa *path) lineTo(p point) {
	pa.begin()
	pa.ops = append(pa.ops, opLine)
	pa.pts = append(pa.pts, p)
	pa.cur = p
}

func (pa *path) cubicTo(c1, c2, p point) {
	pa.begin()
	pa.ops = append(pa.ops, opCubic)
	pa.pts = append(pa.pts, c1, c2, p)
	pa.cur = p
}

// quadTo draws a quadratic curve, as the cubic curve it is.
func (pa *path) quadTo(c, p point) {
	a := pa.cur
	pa.cubicTo(point{a.x + 2.0/3*(c.x-a.x), a.y + 2.0/3*(c.y-a.y)}, point{p.x + 2.0/3*(c.x-p.x), p.y + 2.0/3*(c.y-p.y)}, p)
}

func (pa *path) close() {
	if pa.open {
		pa.ops = append(pa.ops, opClose)
		pa.cur, pa.open = pa.start, false
	}
}

// arcTo draws an elliptical arc to p as cubic curves, converting SVG's
// endpoint parameters as appendix F.6 of SVG 1.1 says.
func (pa *path) arcTo(rx, ry, rotation float64, large, sweep bool, p point) {
	a := pa.cur
	if a == p {
		return
	}
	rx, ry = math.Abs(rx), math.Abs(ry)
	if rx == 0 || ry == 0 {
		pa.lineTo(p)
		return
	}
	sin, cos := math.Sincos(rotation * math.Pi / 180)
	// Half the chord, in the ellipse's axes.
	hx, hy := (a.x-p.x)/2, (a.y-p.y)/2
	x1 := cos*hx + sin*hy
	y1 := -sin*hx + cos*hy
	// Radii too small to reach p grow.
	if l := x1*x1/(rx*rx) + y1*y1/(ry*ry); l > 1 {
		l = math.Sqrt(l)
		rx, ry = rx*l, ry*l
	}
	num := rx*rx*ry*ry - rx*rx*y1*y1 - ry*ry*x1*x1
	den := rx*rx*y1*y1 + ry*ry*x1*x1
	k := 0.0
	if den > 0 {
		k = math.Sqrt(math.Max(num/den, 0))
	}
	if large == sweep {
		k = -k
	}
	cx1, cy1 := k*rx*y1/ry, -k*ry*x1/rx
	cx := cos*cx1 - sin*cy1 + (a.x+p.x)/2
	cy := sin*cx1 + cos*cy1 + (a.y+p.y)/2
	ux, uy := (x1-cx1)/rx, (y1-cy1)/ry
	vx, vy := (-x1-cx1)/rx, (-y1-cy1)/ry
	theta := math.Atan2(uy, ux)
	delta := math.Atan2(ux*vy-uy*vx, ux*vx+uy*vy)
	if !sweep && delta > 0 {
		delta -= 2 * math.Pi
	} else if sweep && delta < 0 {
		delta += 2 * math.Pi
	}
	// Quarter turns at most per curve.
	n := max(int(math.Ceil(math.Abs(delta)/(math.Pi/2)-1e-7)), 1)
	step := delta / float64(n)
	t := 4.0 / 3 * math.Tan(step/4)
	on := func(ex, ey float64) point {
		x, y := ex*rx, ey*ry
		return point{cos*x - sin*y + cx, sin*x + cos*y + cy}
	}
	for i := range n {
		s1, c1 := math.Sincos(theta)
		s2, c2 := math.Sincos(theta + step)
		end := on(c2, s2)
		if i == n-1 {
			end = p
		}
		pa.cubicTo(on(c1-t*s1, s1+t*c1), on(c2+t*s2, s2-t*c2), end)
		theta += step
	}
}

// kappa places the control points of a quarter circle's cubic curve.
const kappa = 0.5522847498307936

func (pa *path) ellipse(cx, cy, rx, ry float64) {
	kx, ky := rx*kappa, ry*kappa
	pa.moveTo(point{cx + rx, cy})
	pa.cubicTo(point{cx + rx, cy + ky}, point{cx + kx, cy + ry}, point{cx, cy + ry})
	pa.cubicTo(point{cx - kx, cy + ry}, point{cx - rx, cy + ky}, point{cx - rx, cy})
	pa.cubicTo(point{cx - rx, cy - ky}, point{cx - kx, cy - ry}, point{cx, cy - ry})
	pa.cubicTo(point{cx + kx, cy - ry}, point{cx + rx, cy - ky}, point{cx + rx, cy})
	pa.close()
}

// rect adds a rectangle with corners rounded rx by ry.
func (pa *path) rect(x, y, w, h, rx, ry float64) {
	if rx <= 0 || ry <= 0 {
		pa.moveTo(point{x, y})
		pa.lineTo(point{x + w, y})
		pa.lineTo(point{x + w, y + h})
		pa.lineTo(point{x, y + h})
		pa.close()
		return
	}
	kx, ky := rx*kappa, ry*kappa
	r, b := x+w, y+h
	pa.moveTo(point{x + rx, y})
	pa.lineTo(point{r - rx, y})
	pa.cubicTo(point{r - rx + kx, y}, point{r, y + ry - ky}, point{r, y + ry})
	pa.lineTo(point{r, b - ry})
	pa.cubicTo(point{r, b - ry + ky}, point{r - rx + kx, b}, point{r - rx, b})
	pa.lineTo(point{x + rx, b})
	pa.cubicTo(point{x + rx - kx, b}, point{x, b - ry + ky}, point{x, b - ry})
	pa.lineTo(point{x, y + ry})
	pa.cubicTo(point{x, y + ry - ky}, point{x + rx - kx, y}, point{x + rx, y})
	pa.close()
}

func isCommand(c byte) bool {
	switch c {
	case 'M', 'm', 'L', 'l', 'H', 'h', 'V', 'v', 'C', 'c', 'S', 's', 'Q', 'q', 'T', 't', 'A', 'a', 'Z', 'z':
		return true
	}
	return false
}

// parsePath parses path data up to its first error, as SVG asks.
func parsePath(d string) path {
	var pa path
	p := scanner{s: d}
	var cmd, last byte // the command, and the last one in upper case
	var ctrl point     // the last control point, for S and T
	coord := func() (float64, bool) { p.skipSep(); return p.number() }
	pt := func(base point) (point, bool) {
		x, ok := coord()
		if !ok {
			return point{}, false
		}
		y, ok := coord()
		return point{base.x + x, base.y + y}, ok
	}
	flag := func() (bool, bool) {
		p.skipSep()
		switch p.peek() {
		case '0':
			p.i++
			return false, true
		case '1':
			p.i++
			return true, true
		}
		return false, false
	}
	for {
		p.skipSpace()
		if p.done() {
			break
		}
		if c := p.peek(); isCommand(c) {
			cmd = c
			p.i++
		} else if cmd == 0 || cmd == 'Z' || cmd == 'z' {
			break // numbers without a command
		}
		if len(pa.ops) == 0 && cmd != 'M' && cmd != 'm' {
			break // a path starts with a move
		}
		var base point
		if cmd >= 'a' {
			base = pa.cur
		}
		up := cmd &^ 0x20
		ok := true
		switch up {
		case 'M':
			var a point
			if a, ok = pt(base); ok {
				pa.moveTo(a)
				// More coordinates are lines.
				cmd = 'L' | cmd&0x20
			}
		case 'L':
			var a point
			if a, ok = pt(base); ok {
				pa.lineTo(a)
			}
		case 'H':
			var x float64
			if x, ok = coord(); ok {
				pa.lineTo(point{base.x + x, pa.cur.y})
			}
		case 'V':
			var y float64
			if y, ok = coord(); ok {
				pa.lineTo(point{pa.cur.x, base.y + y})
			}
		case 'C', 'S':
			c1 := pa.cur
			if up == 'C' {
				c1, ok = pt(base)
			} else if last == 'C' || last == 'S' {
				c1 = point{2*pa.cur.x - ctrl.x, 2*pa.cur.y - ctrl.y}
			}
			var c2, e point
			if ok {
				c2, ok = pt(base)
			}
			if ok {
				e, ok = pt(base)
			}
			if ok {
				pa.cubicTo(c1, c2, e)
				ctrl = c2
			}
		case 'Q', 'T':
			c := pa.cur
			if up == 'Q' {
				c, ok = pt(base)
			} else if last == 'Q' || last == 'T' {
				c = point{2*pa.cur.x - ctrl.x, 2*pa.cur.y - ctrl.y}
			}
			var e point
			if ok {
				e, ok = pt(base)
			}
			if ok {
				pa.quadTo(c, e)
				ctrl = c
			}
		case 'A':
			var rx, ry, rot float64
			var large, sweep bool
			var e point
			rx, ok = coord()
			if ok {
				ry, ok = coord()
			}
			if ok {
				rot, ok = coord()
			}
			if ok {
				large, ok = flag()
			}
			if ok {
				sweep, ok = flag()
			}
			if ok {
				e, ok = pt(base)
			}
			if ok {
				pa.arcTo(rx, ry, rot, large, sweep, e)
			}
		case 'Z':
			pa.close()
		}
		if !ok {
			break
		}
		last = up
	}
	return pa
}

// bounds returns the box around the path's geometry, in user units.
func (pa *path) bounds() (box, bool) {
	var b box
	found := false
	add := func(p point) {
		if !found {
			b, found = box{p.x, p.y, 0, 0}, true
			return
		}
		b = b.union(box{p.x, p.y, 0, 0})
	}
	var cur point
	i := 0
	for _, op := range pa.ops {
		switch op {
		case opMove:
			cur = pa.pts[i]
			i++
		case opLine:
			add(cur)
			cur = pa.pts[i]
			add(cur)
			i++
		case opCubic:
			c1, c2, e := pa.pts[i], pa.pts[i+1], pa.pts[i+2]
			add(cur)
			add(e)
			for _, t := range extrema(cur.x, c1.x, c2.x, e.x) {
				add(bezier(cur, c1, c2, e, t))
			}
			for _, t := range extrema(cur.y, c1.y, c2.y, e.y) {
				add(bezier(cur, c1, c2, e, t))
			}
			cur = e
			i += 3
		}
	}
	return b, found
}

// extrema returns where in (0, 1) a cubic curve's coordinate turns.
func extrema(a, b, c, d float64) []float64 {
	qa := -a + 3*b - 3*c + d
	qb := 2 * (a - 2*b + c)
	qc := b - a
	var roots [2]float64
	n := 0
	if math.Abs(qa) < 1e-12 {
		if qb != 0 {
			roots[0] = -qc / qb
			n = 1
		}
	} else if disc := qb*qb - 4*qa*qc; disc >= 0 {
		s := math.Sqrt(disc)
		roots[0], roots[1] = (-qb+s)/(2*qa), (-qb-s)/(2*qa)
		n = 2
	}
	out := roots[:0]
	for _, t := range roots[:n] {
		if t > 0 && t < 1 {
			out = append(out, t)
		}
	}
	return out
}

func bezier(a, b, c, d point, t float64) point {
	u := 1 - t
	k0, k1, k2, k3 := u*u*u, 3*u*u*t, 3*u*t*t, t*t*t
	return point{k0*a.x + k1*b.x + k2*c.x + k3*d.x, k0*a.y + k1*b.y + k2*c.y + k3*d.y}
}

// steps returns how many lines follow a cubic curve, in pixels, to a
// tenth of a pixel: the error of n lines is at most 3/4 of the curve's
// largest second difference over n².
func steps(a, b, c, d point) int {
	ddx := math.Max(math.Abs(a.x-2*b.x+c.x), math.Abs(b.x-2*c.x+d.x))
	ddy := math.Max(math.Abs(a.y-2*b.y+c.y), math.Abs(b.y-2*c.y+d.y))
	dd := math.Hypot(ddx, ddy)
	if !(dd >= 0) {
		return 1
	}
	return max(1, min(int(math.Ceil(math.Sqrt(7.5*dd))), 1000))
}

// polys holds polylines: polyline i is pts[starts[i]:starts[i+1]], the
// last up to the end of pts.
type polys struct {
	pts    []point
	starts []int
	closed []bool
	// begin is where the polyline being built begins, if building.
	begin    int
	building bool
}

func (ps *polys) reset() {
	ps.pts, ps.starts, ps.closed, ps.building = ps.pts[:0], ps.starts[:0], ps.closed[:0], false
}

func (ps *polys) len() int { return len(ps.starts) }

func (ps *polys) poly(i int) []point {
	end := len(ps.pts)
	if i+1 < len(ps.starts) {
		end = ps.starts[i+1]
	}
	return ps.pts[ps.starts[i]:end]
}

// start begins a polyline at p, ending the one being built.
func (ps *polys) start(p point) {
	ps.end(false)
	ps.begin, ps.building = len(ps.pts), true
	ps.pts = append(ps.pts, p)
}

func (ps *polys) add(p point) { ps.pts = append(ps.pts, p) }

// end ends the polyline being built. One of a single point, a move with
// nothing drawn, is dropped.
func (ps *polys) end(closed bool) {
	if !ps.building {
		return
	}
	ps.building = false
	if len(ps.pts)-ps.begin < 2 {
		ps.pts = ps.pts[:ps.begin]
		return
	}
	ps.starts = append(ps.starts, ps.begin)
	ps.closed = append(ps.closed, closed)
}

func (ps *polys) transform(m matrix) {
	for i, p := range ps.pts {
		ps.pts[i] = m.apply(p)
	}
}

// maxPoints bounds the points of a flattened path: beyond it, curves are
// drawn as lines, and strokes are left out.
const maxPoints = 1 << 20

// flatten adds the path's subpaths to out as polylines, with points
// enough that they stray less than a tenth of a pixel from its curves
// once transformed by m, the transform to pixels: in pixels with device,
// in user units otherwise.
func (pa *path) flatten(m matrix, device bool, out *polys) {
	at := func(p point) point {
		if device {
			return m.apply(p)
		}
		return p
	}
	var cur point
	i := 0
	for _, op := range pa.ops {
		switch op {
		case opMove:
			cur = pa.pts[i]
			i++
			out.start(at(cur))
		case opLine:
			cur = pa.pts[i]
			i++
			out.add(at(cur))
		case opCubic:
			c1, c2, e := pa.pts[i], pa.pts[i+1], pa.pts[i+2]
			i += 3
			n := 1
			if len(out.pts) < maxPoints {
				n = steps(m.apply(cur), m.apply(c1), m.apply(c2), m.apply(e))
			}
			for k := 1; k < n; k++ {
				out.add(at(bezier(cur, c1, c2, e, float64(k)/float64(n))))
			}
			out.add(at(e))
			cur = e
		case opClose:
			if out.building && len(out.pts)-out.begin == 1 {
				// A subpath of no length, which round and square caps
				// stroke.
				out.add(out.pts[out.begin])
			}
			out.end(true)
		}
	}
	out.end(false)
}
