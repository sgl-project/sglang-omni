package scene

import "math"

// ClipShape is a clip in the coordinate system in which it was pushed.
type ClipShape struct {
	Rect       Rect
	Radii      [4]float32
	Transform  Affine
	Continuous bool
}

// Affine maps drawing coordinates to frame coordinates: an affine map,
// or, when G or H is set, a projective one, which shows a plane turned in
// 3D in perspective, x and y going to
//
//	((A*x + C*y + X) / w, (B*x + D*y + Y) / w), with w = G*x + H*y + 1.
//
// The zero value is identity; Set distinguishes a singular matrix
// (including scale zero).
type Affine struct {
	A, B, C, D, X, Y float32
	G, H             float32
	Set              bool
}

func Matrix(a, b, c, d, x, y float32) Affine { return Projective(a, b, c, d, x, y, 0, 0) }

// Projective returns the projective map of the matrix
//
//	| a c x |
//	| b d y |
//	| g h 1 |
func Projective(a, b, c, d, x, y, g, h float32) Affine {
	if a == 1 && b == 0 && c == 0 && d == 1 && x == 0 && y == 0 && g == 0 && h == 0 {
		return Affine{}
	}
	return Affine{a, b, c, d, x, y, g, h, true}
}

func Translation(x, y float32) Affine { return Matrix(1, 0, 0, 1, x, y) }
func Scaling(x, y float32) Affine     { return Matrix(x, 0, 0, y, 0, 0) }
func Rotation(degrees float32) Affine {
	angle := math.Remainder(float64(degrees), 360)
	switch angle {
	case 0:
		return Affine{}
	case 90:
		return Matrix(0, 1, -1, 0, 0, 0)
	case -90:
		return Matrix(0, -1, 1, 0, 0, 0)
	case 180, -180:
		return Matrix(-1, 0, 0, -1, 0, 0)
	}
	s, c := math.Sincos(angle * math.Pi / 180)
	return Matrix(float32(c), float32(s), -float32(s), float32(c), 0, 0)
}

// Perspective reports whether m is projective, not affine.
func (m Affine) Perspective() bool { return m.Set && (m.G != 0 || m.H != 0) }

// matrix returns m's matrix, by rows.
func (m Affine) matrix() [3][3]float64 {
	if !m.Set {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	return [3][3]float64{
		{float64(m.A), float64(m.C), float64(m.X)},
		{float64(m.B), float64(m.D), float64(m.Y)},
		{float64(m.G), float64(m.H), 1},
	}
}

// fromMatrix returns the map of matrix p, scaled so that its last entry
// is 1, and whether it has one.
func fromMatrix(p [3][3]float64) (Affine, bool) {
	k := p[2][2]
	if math.Abs(k) < 1e-12 || math.IsNaN(k) || math.IsInf(k, 0) {
		return Affine{}, false
	}
	v := [8]float64{p[0][0] / k, p[1][0] / k, p[0][1] / k, p[1][1] / k, p[0][2] / k, p[1][2] / k, p[2][0] / k, p[2][1] / k}
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) > math.MaxFloat32 {
			return Affine{}, false
		}
	}
	return Projective(float32(v[0]), float32(v[1]), float32(v[2]), float32(v[3]), float32(v[4]), float32(v[5]), float32(v[6]), float32(v[7])), true
}

// singular is a map without an inverse, which shows nothing.
var singular = Affine{Set: true}

// Mul returns m*n: n acts first, then m.
func (m Affine) Mul(n Affine) Affine {
	if !m.Set {
		return n
	}
	if !n.Set {
		return m
	}
	if !m.Perspective() && !n.Perspective() {
		return Matrix(m.A*n.A+m.C*n.B, m.B*n.A+m.D*n.B,
			m.A*n.C+m.C*n.D, m.B*n.C+m.D*n.D,
			m.A*n.X+m.C*n.Y+m.X, m.B*n.X+m.D*n.Y+m.Y)
	}
	a, b := m.matrix(), n.matrix()
	var p [3][3]float64
	for i := range 3 {
		for j := range 3 {
			p[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	r, ok := fromMatrix(p)
	if !ok {
		return singular
	}
	return r
}

// W returns the projective divisor of (x, y): positive in front of the
// viewer, as all of an affine map's plane is.
func (m Affine) W(x, y float32) float32 {
	if !m.Set {
		return 1
	}
	return m.G*x + m.H*y + 1
}

// Point returns where m maps (x, y). A point behind the viewer of a
// projective map (W not positive) goes far away.
func (m Affine) Point(x, y float32) (float32, float32) {
	if !m.Set {
		return x, y
	}
	px, py := m.A*x+m.C*y+m.X, m.B*x+m.D*y+m.Y
	if m.G == 0 && m.H == 0 {
		return px, py
	}
	w := max(m.W(x, y), 1e-6)
	return px / w, py / w
}

// VectorAt returns how far m moves the point (x, y) moved by (dx, dy):
// the same for every point of an affine map.
func (m Affine) VectorAt(x, y, dx, dy float32) (float32, float32) {
	if !m.Set {
		return dx, dy
	}
	if m.G == 0 && m.H == 0 {
		return m.A*dx + m.C*dy, m.B*dx + m.D*dy
	}
	x0, y0 := m.Point(x, y)
	x1, y1 := m.Point(x+dx, y+dy)
	return x1 - x0, y1 - y0
}

func (m Affine) Inverse() (Affine, bool) {
	if !m.Set {
		return m, true
	}
	if m.G == 0 && m.H == 0 {
		det := float64(m.A)*float64(m.D) - float64(m.B)*float64(m.C)
		if det == 0 || math.IsNaN(det) || math.IsInf(det, 0) {
			return Affine{}, false
		}
		a, b, c, d := float32(float64(m.D)/det), float32(-float64(m.B)/det), float32(-float64(m.C)/det), float32(float64(m.A)/det)
		n := Matrix(a, b, c, d, -a*m.X-c*m.Y, -b*m.X-d*m.Y)
		for _, v := range [6]float32{n.A, n.B, n.C, n.D, n.X, n.Y} {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return Affine{}, false
			}
		}
		return n, true
	}
	q := m.matrix()
	cof := func(r0, r1, c0, c1 int) float64 { return q[r0][c0]*q[r1][c1] - q[r0][c1]*q[r1][c0] }
	adj := [3][3]float64{
		{cof(1, 2, 1, 2), -cof(0, 2, 1, 2), cof(0, 1, 1, 2)},
		{-cof(1, 2, 0, 2), cof(0, 2, 0, 2), -cof(0, 1, 0, 2)},
		{cof(1, 2, 0, 1), -cof(0, 2, 0, 1), cof(0, 1, 0, 1)},
	}
	det := q[0][0]*adj[0][0] + q[0][1]*adj[1][0] + q[0][2]*adj[2][0]
	if math.Abs(det) < 1e-12 || math.IsNaN(det) || math.IsInf(det, 0) {
		return Affine{}, false
	}
	return fromMatrix(adj) // the inverse up to a factor, which fromMatrix drops
}

func (m Affine) Bounds(r Rect) Rect {
	if !m.Set {
		return r
	}
	corners := [4][2]float32{{r.X, r.Y}, {r.X + r.W, r.Y}, {r.X, r.Y + r.H}, {r.X + r.W, r.Y + r.H}}
	if m.Perspective() {
		for _, p := range corners {
			if m.W(p[0], p[1]) <= 1e-6 {
				// Part of it is behind the viewer: it may show anywhere.
				return Rect{-1e6, -1e6, 2e6, 2e6}
			}
		}
	}
	x0, y0 := m.Point(r.X, r.Y)
	x1, y1 := x0, y0
	for _, p := range corners[1:] {
		x, y := m.Point(p[0], p[1])
		x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
	}
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

// Pixels expresses a DIP transform in device pixels.
func (m Affine) Pixels(scale float32) Affine {
	if m.Set {
		m.X *= scale
		m.Y *= scale
		m.G /= scale
		m.H /= scale
	}
	return m
}
