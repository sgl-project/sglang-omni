// Package vec rasterizes vector paths (lines, quadratic and cubic Bézier
// curves) into anti-aliased coverage masks, with the signed area
// accumulation of font-rs: every edge adds the area it covers to the cells
// it crosses, and a running sum along each row gives the coverage.
package vec

import "math"

// Rasterizer accumulates the area of a path's edges. Reuse it with Reset.
type Rasterizer struct {
	w, h           int
	acc            []float32
	penX, penY     float32
	firstX, firstY float32
	open           bool
}

// Reset prepares the rasterizer for a w×h mask.
func (z *Rasterizer) Reset(w, h int) {
	z.w, z.h = w, h
	if cap(z.acc) < w*h {
		z.acc = make([]float32, w*h)
	} else {
		z.acc = z.acc[:w*h]
		clear(z.acc)
	}
	z.open = false
}

// MoveTo starts a contour, closing the previous one.
func (z *Rasterizer) MoveTo(x, y float32) {
	z.ClosePath()
	z.firstX, z.firstY, z.penX, z.penY = x, y, x, y
	z.open = true
}

// ClosePath draws a line back to the start of the contour.
func (z *Rasterizer) ClosePath() {
	if z.open {
		z.LineTo(z.firstX, z.firstY)
		z.open = false
	}
}

// QuadTo draws a quadratic Bézier curve with control point (bx, by).
func (z *Rasterizer) QuadTo(bx, by, cx, cy float32) {
	ax, ay := z.penX, z.penY
	dx, dy := ax-2*bx+cx, ay-2*by+cy
	n := int(math.Ceil(math.Sqrt(math.Sqrt(float64(dx*dx+dy*dy)) * 2)))
	n = max(1, min(n, 100))
	for i := 1; i <= n; i++ {
		t := float32(i) / float32(n)
		u := 1 - t
		z.LineTo(u*u*ax+2*u*t*bx+t*t*cx, u*u*ay+2*u*t*by+t*t*cy)
	}
}

// CubeTo draws a cubic Bézier curve with control points (bx, by) and (cx,
// cy).
func (z *Rasterizer) CubeTo(bx, by, cx, cy, dx, dy float32) {
	ax, ay := z.penX, z.penY
	ex, ey := ax-2*bx+cx, ay-2*by+cy
	fx, fy := bx-2*cx+dx, by-2*cy+dy
	dd := max(ex*ex+ey*ey, fx*fx+fy*fy)
	n := int(math.Ceil(math.Sqrt(math.Sqrt(float64(dd)) * 3)))
	n = max(1, min(n, 100))
	for i := 1; i <= n; i++ {
		t := float32(i) / float32(n)
		u := 1 - t
		a, b, c, d := u*u*u, 3*u*u*t, 3*u*t*t, t*t*t
		z.LineTo(a*ax+b*bx+c*cx+d*dx, a*ay+b*by+c*cy+d*dy)
	}
}

// LineTo draws a line from the pen.
func (z *Rasterizer) LineTo(bx, by float32) {
	ax, ay := z.penX, z.penY
	z.penX, z.penY = bx, by
	dir := float32(1)
	if ay > by {
		dir = -1
		ax, ay, bx, by = bx, by, ax, ay
	}
	if by-ay <= 0 || by <= 0 || ay >= float32(z.h) {
		return
	}
	dxdy := (bx - ax) / (by - ay)
	x := ax
	y0 := int(math.Floor(float64(ay)))
	y1 := int(math.Ceil(float64(by)))
	if y0 < 0 {
		// Skip the rows above the mask.
		x += dxdy * (0 - ay)
		ay = 0
		y0 = 0
	}
	y1 = min(y1, z.h)
	w := z.w
	for y := y0; y < y1; y++ {
		fy := float32(y)
		dy := min(fy+1, by) - max(fy, ay)
		xNext := x + dy*dxdy
		d := dy * dir
		row := z.acc[y*w : (y+1)*w]
		x0, x1 := x, xNext
		if x0 > x1 {
			x0, x1 = x1, x0
		}
		x0i := int(math.Floor(float64(x0)))
		x0f := float32(x0i)
		x1i := int(math.Ceil(float64(x1)))
		x1c := float32(x1i)
		if x1i <= x0i+1 {
			xmf := 0.5*(x+xNext) - x0f
			add(row, x0i, d-d*xmf)
			add(row, x0i+1, d*xmf)
		} else {
			s := 1 / (x1 - x0)
			x0fr := x0 - x0f
			omx := 1 - x0fr
			a0 := 0.5 * s * omx * omx
			x1fr := x1 - x1c + 1
			am := 0.5 * s * x1fr * x1fr
			add(row, x0i, d*a0)
			if x1i == x0i+2 {
				add(row, x0i+1, d*(1-a0-am))
			} else {
				a1 := s * (1.5 - x0fr)
				add(row, x0i+1, d*(a1-a0))
				for xi := x0i + 2; xi < x1i-1; xi++ {
					add(row, xi, d*s)
				}
				a2 := a1 + s*float32(x1i-x0i-3)
				add(row, x1i-1, d*(1-a2-am))
			}
			add(row, x1i, d*am)
		}
		x = xNext
	}
}

// add adds v at x; area left of the mask counts at its first pixel and
// area right of it is dropped, since rows are summed from the left.
func add(row []float32, x int, v float32) {
	if x < 0 {
		x = 0
	}
	if x < len(row) {
		row[x] += v
	}
}

// Mask writes the coverage of the accumulated path (non-zero winding) as
// one byte per pixel into dst, rows of stride bytes.
func (z *Rasterizer) Mask(dst []byte, stride int) {
	z.ClosePath()
	for y := 0; y < z.h; y++ {
		var acc float32
		row := z.acc[y*z.w : (y+1)*z.w]
		out := dst[y*stride:]
		for x, v := range row {
			acc += v
			a := acc
			if a < 0 {
				a = -a
			}
			if a > 1 {
				a = 1
			}
			out[x] = uint8(a*255 + 0.5)
		}
	}
}

// MaskEvenOdd is Mask with the even-odd rule: what the path's contours
// cover an odd number of times is inside it.
func (z *Rasterizer) MaskEvenOdd(dst []byte, stride int) {
	z.ClosePath()
	for y := 0; y < z.h; y++ {
		var acc float32
		row := z.acc[y*z.w : (y+1)*z.w]
		out := dst[y*stride:]
		for x, v := range row {
			acc += v
			a := acc
			if a < 0 {
				a = -a
			}
			// Fold the winding number: 0, 2, 4… are outside, 1, 3, 5…
			// inside, and coverage in between ramps.
			a -= 2 * float32(int(a/2))
			if a > 1 {
				a = 2 - a
			}
			out[x] = uint8(a*255 + 0.5)
		}
	}
}
