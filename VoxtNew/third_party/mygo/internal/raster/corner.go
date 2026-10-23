package raster

import "math"

// Continuous corners are Core Animation's, the corners macOS shows for
// CALayer's continuous corner curve and SwiftUI's rounded rectangles,
// computed as its shaders compute them (QuartzCore's supercircle_sdf),
// which a capture of the screen matches within a few 255ths a pixel.
//
// In a corner's own terms, u and v are how far a point is inside its
// vertical and horizontal edges, r its radius and e = contExtent·r
// how far from the corner the curve leaves the edges. With a = max(0,
// 1 − (u, v)/e) the point's place in the box of the curve, and ρ the ratio
// of a's smaller coordinate to its larger, the curve is where
//
//	|a| + 1 − 1/(1 − ρ²·min(|a|, 1)·P(ρ)) = 1,
//
// P a quartic: a quarter circle of radius r around the diagonal, which
// bends less and less toward the edges. Where a side is too short for both
// of its corners' curves, the curves blend along it toward the quarter
// circles by the side's clamp factor: 0 while the curves fit, 1 once the
// side is no longer than two of their mean radius, as a pill's ends; a
// circle's corners, clamped both ways, are quarter circles. The shape is
// the intersection of its corners' (the largest of their values), so a
// curve may reach past the middle of a side whose other corner is smaller,
// and edges are antialiased by the value over the sum of its derivatives
// (fwidth). The shaders compute the same.
const contExtent = 1.528665

// contCorner is a continuous corner of radius r, whose curve leaves the
// edges e from the corner, blended toward a quarter circle by cx along
// its horizontal edge and cy along its vertical one; eff is where the
// curve's box ends as clamped.
type contCorner struct{ r, e, eff, cx, cy float32 }

// newContCorner returns the corner of radius r of a rectangle w×h whose
// horizontal edge it shares with a corner of radius rh, and its vertical
// one with one of radius rv.
func newContCorner(r, rh, rv, w, h float32) contCorner {
	clamp := func(side, ra, rb float32) float32 {
		return clamp01((contExtent - side/(ra+rb)) / (contExtent - 1))
	}
	c := contCorner{r: r, e: contExtent * r, cx: clamp(w, r, rh), cy: clamp(h, r, rv)}
	c.eff = c.e + (r-c.e)*max(c.cx, c.cy)
	return c
}

// dist returns the corner's value at the point u inside its vertical edge
// and v inside its horizontal one: about the signed distance to its edge
// near it, positive outside.
func (c *contCorner) dist(u, v float32) float32 {
	ax, ay := max(0, 1-u/c.e), max(0, 1-v/c.e)
	l := float32(math.Sqrt(float64(ax*ax + ay*ay)))
	hi, lo := max(ax, ay), min(ax, ay)
	var rho float32
	if hi > 0 {
		rho = min(lo/hi, 1)
	}
	p := (((-0.926054*rho+3.15601)*rho-3.64122)*rho+1.26803)*rho + 0.268531
	f1 := l + 1 - 1/(1-rho*rho*min(l, 1)*p)
	qx := max(0, ax*contExtent-(contExtent-1))
	qy := max(0, ay*contExtent-(contExtent-1))
	f2 := float32(math.Sqrt(float64(qx*qx+qy*qy)))*0.654166 + 0.345834
	// The clamp along the edge the point is nearer to, blended across the
	// diagonal.
	s := float32(-1)
	if ay > ax {
		s = 1
	}
	w := clamp01(0.5 - s + s*rho)
	fx, fy := f1+(f2-f1)*c.cx, f1+(f2-f1)*c.cy
	f := fx + (fy-fx)*w
	return min(max(c.eff-u, c.eff-v), 0) + c.e*(f-1)
}

// inset returns how far inside its vertical edge the corner's edge is v
// inside its horizontal one, and a bound at least as far: where dist is 0,
// which lies between the quarter circles of radius r and e, found by
// regula falsi (Illinois) within a hundredth of a pixel at radii of a few
// hundred.
func (c *contCorner) inset(v float32) (at, bound float32) {
	if v >= c.e || c.r <= 0 {
		return 0, 0
	}
	circle := func(r float32) float32 {
		if v >= r {
			return 0
		}
		return r - float32(math.Sqrt(float64(v*(2*r-v))))
	}
	lo, hi := circle(c.r), circle(c.e)
	dlo, dhi := c.dist(lo, v), c.dist(hi, v)
	if dlo <= 0 {
		return lo, lo
	}
	if dhi >= 0 {
		return hi, hi
	}
	side := 0
	for range 5 {
		m := (lo*dhi - hi*dlo) / (dhi - dlo)
		dm := c.dist(m, v)
		if dm > 0 {
			lo, dlo = m, dm
			if side == 1 {
				dhi /= 2
			}
			side = 1
		} else {
			hi, dhi = m, dm
			if side == -1 {
				dlo /= 2
			}
			side = -1
		}
	}
	return (lo*dhi - hi*dlo) / (dhi - dlo), hi
}
