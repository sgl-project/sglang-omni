package ui

import (
	"encoding/binary"
	"hash/maphash"
	"math"

	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/vec"
)

// Path is a vector shape, in DIPs, for Painter.FillPath and StrokePath:
// icons, check marks, charts.
type Path struct {
	cmds []pathCmd
	// trim is the part of the path's length it keeps (Trim), when trimmed.
	trim    [2]float32
	trimmed bool
}

type pathCmd struct {
	op  uint8 // 0 move, 1 line, 2 quad, 3 cubic, 4 close
	pts [3][2]float32
}

// Reset empties the path and removes Trim, retaining its command storage.
func (p *Path) Reset() *Path {
	p.cmds = p.cmds[:0]
	p.trim, p.trimmed = [2]float32{}, false
	return p
}

// MoveTo starts a new subpath at (x, y).
func (p *Path) MoveTo(x, y float32) *Path {
	p.cmds = append(p.cmds, pathCmd{op: 0, pts: [3][2]float32{{x, y}}})
	return p
}

// LineTo draws a line to (x, y).
func (p *Path) LineTo(x, y float32) *Path {
	p.cmds = append(p.cmds, pathCmd{op: 1, pts: [3][2]float32{{x, y}}})
	return p
}

// QuadTo draws a quadratic Bézier curve to (x, y) with control point
// (cx, cy).
func (p *Path) QuadTo(cx, cy, x, y float32) *Path {
	p.cmds = append(p.cmds, pathCmd{op: 2, pts: [3][2]float32{{cx, cy}, {x, y}}})
	return p
}

// CubeTo draws a cubic Bézier curve to (x, y) with control points (c1x,
// c1y) and (c2x, c2y).
func (p *Path) CubeTo(c1x, c1y, c2x, c2y, x, y float32) *Path {
	p.cmds = append(p.cmds, pathCmd{op: 3, pts: [3][2]float32{{c1x, c1y}, {c2x, c2y}, {x, y}}})
	return p
}

// Close closes the subpath.
func (p *Path) Close() *Path {
	p.cmds = append(p.cmds, pathCmd{op: 4})
	return p
}

// Trim keeps of the path, as a Painter fills or strokes it, the part of
// its length from from to to, as fractions of it, its subpaths one after
// another, and returns the path: 0 to 0.5 draws its first half, and a check
// mark draws itself as to animates from 0 to 1, as SwiftUI's trim(from:to:)
// and Motion's pathLength. Trimmed, closed subpaths draw open, and a
// stroke trimmed shorter than its width fades with its length, rather than
// its round caps showing a dot that comes or goes at once.
func (p *Path) Trim(from, to float32) *Path {
	finiteMotion("trim", float64(from), float64(to))
	p.trim, p.trimmed = [2]float32{max(0, min(from, 1)), max(0, min(to, 1))}, true
	return p
}

// Bounds returns the smallest box holding the path's points and the
// control points of its curves, in DIPs: the box of a circle (Circle), of
// lines, and of curves whose control points lie on it; others may fit in
// less. Empty paths have an empty box.
func (p *Path) Bounds() Rect {
	first := true
	var x0, y0, x1, y1 float32
	for _, c := range p.cmds {
		n := [5]int{1, 1, 2, 3, 0}[c.op]
		for _, pt := range c.pts[:n] {
			if first {
				x0, y0, x1, y1, first = pt[0], pt[1], pt[0], pt[1], false
				continue
			}
			x0, y0, x1, y1 = min(x0, pt[0]), min(y0, pt[1]), max(x1, pt[0]), max(y1, pt[1])
		}
	}
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

// Circle adds a circle as a subpath.
func (p *Path) Circle(cx, cy, r float32) *Path {
	const k = 0.5522847498
	p.MoveTo(cx+r, cy)
	p.CubeTo(cx+r, cy+r*k, cx+r*k, cy+r, cx, cy+r)
	p.CubeTo(cx-r*k, cy+r, cx-r, cy+r*k, cx-r, cy)
	p.CubeTo(cx-r, cy-r*k, cx-r*k, cy-r, cx, cy-r)
	p.CubeTo(cx+r*k, cy-r, cx+r, cy-r*k, cx+r, cy)
	return p.Close()
}

// flatPath is a path flattened to polylines in device pixels: polyline i
// holds pts[starts[i]:starts[i+1]], the last up to the end of pts. The
// engine keeps one for the paths it paints, reusing its memory.
type flatPath struct {
	pts    [][2]float32
	starts []int
	closed []bool
	// trimmed tells that Trim kept kept pixels of the path's length.
	trimmed bool
	kept    float32
	// The polylines that trimming keeps, swapped with the others.
	keptPts    [][2]float32
	keptStarts []int
	keptClosed []bool
}

func (f *flatPath) polys() int { return len(f.starts) }

func (f *flatPath) poly(i int) [][2]float32 {
	end := len(f.pts)
	if i+1 < len(f.starts) {
		end = f.starts[i+1]
	}
	return f.pts[f.starts[i]:end]
}

// flatten sets f to the path's subpaths as polylines in device pixels.
func (p *Path) flatten(f *flatPath, scale float32) {
	f.pts, f.starts, f.closed = f.pts[:0], f.starts[:0], f.closed[:0]
	f.trimmed, f.kept = false, 0
	begin, isClosed := 0, false
	var origin, pen [2]float32
	flush := func() {
		if len(f.pts)-begin > 1 {
			f.starts = append(f.starts, begin)
			f.closed = append(f.closed, isClosed)
		} else {
			f.pts = f.pts[:begin]
		}
		begin, isClosed = len(f.pts), false
	}
	for _, c := range p.cmds {
		pts := c.pts
		for i := range pts {
			pts[i][0] *= scale
			pts[i][1] *= scale
		}
		switch c.op {
		case 0:
			flush()
			origin, pen = pts[0], pts[0]
			f.pts = append(f.pts, pen)
		case 1:
			pen = pts[0]
			f.pts = append(f.pts, pen)
		case 2:
			n := curveSteps(pen, pts[0], pts[1], pts[1])
			for i := 1; i <= n; i++ {
				t := float32(i) / float32(n)
				u := 1 - t
				f.pts = append(f.pts, [2]float32{u*u*pen[0] + 2*u*t*pts[0][0] + t*t*pts[1][0], u*u*pen[1] + 2*u*t*pts[0][1] + t*t*pts[1][1]})
			}
			pen = pts[1]
		case 3:
			n := curveSteps(pen, pts[0], pts[1], pts[2])
			for i := 1; i <= n; i++ {
				t := float32(i) / float32(n)
				u := 1 - t
				a, b, cc, d := u*u*u, 3*u*u*t, 3*u*t*t, t*t*t
				f.pts = append(f.pts, [2]float32{a*pen[0] + b*pts[0][0] + cc*pts[1][0] + d*pts[2][0], a*pen[1] + b*pts[0][1] + cc*pts[1][1] + d*pts[2][1]})
			}
			pen = pts[2]
		case 4:
			if len(f.pts) > begin {
				isClosed = true
				flush()
				pen = origin
			}
		}
	}
	flush()
	if p.trimmed {
		f.trim(p.trim[0], p.trim[1])
	}
}

// trim keeps the part of f's length from from to to, as fractions of it,
// as open polylines.
func (f *flatPath) trim(from, to float32) {
	// segments returns how many segments polyline i has, a closed one's
	// back to its start included.
	segments := func(i int) int {
		n := len(f.poly(i)) - 1
		if f.closed[i] {
			n++
		}
		return n
	}
	var total float32
	for i := range f.polys() {
		poly := f.poly(i)
		for j := range segments(i) {
			total += dist(poly[j], poly[(j+1)%len(poly)])
		}
	}
	lo, hi := from*total, to*total
	f.trimmed, f.kept = true, max(hi-lo, 0)
	pts, starts, closed := f.keptPts[:0], f.keptStarts[:0], f.keptClosed[:0]
	var at float32
	for i := range f.polys() {
		poly := f.poly(i)
		open := false // a kept polyline goes on to the next segment
		for j := range segments(i) {
			a, b := poly[j], poly[(j+1)%len(poly)]
			l := dist(a, b)
			s, e := max(lo, at), min(hi, at+l)
			at += l
			if l == 0 || s >= e {
				open = open && l == 0
				continue
			}
			k0, k1 := (s-(at-l))/l, (e-(at-l))/l
			if !open {
				starts, closed = append(starts, len(pts)), append(closed, false)
				pts = append(pts, [2]float32{a[0] + (b[0]-a[0])*k0, a[1] + (b[1]-a[1])*k0})
			}
			pts = append(pts, [2]float32{a[0] + (b[0]-a[0])*k1, a[1] + (b[1]-a[1])*k1})
			open = k1 >= 1
		}
	}
	f.keptPts, f.pts = f.pts, pts
	f.keptStarts, f.starts = f.starts, starts
	f.keptClosed, f.closed = f.closed, closed
}

func curveSteps(a, b, c, d [2]float32) int {
	l := dist(a, b) + dist(b, c) + dist(c, d)
	return max(2, min(int(math.Sqrt(float64(l))*2), 64))
}

func dist(a, b [2]float32) float32 {
	return float32(math.Hypot(float64(a[0]-b[0]), float64(a[1]-b[1])))
}

var pathSeed = maphash.MakeSeed()

// paths holds what the engine reuses to paint paths, frame after frame:
// the flattened path, the job of drawing its mask, and the rasterizer
// and pixels that draw it.
type paths struct {
	flat   flatPath
	job    pathJob
	raster vec.Rasterizer
	pix    []byte
	loop   [][2]float32 // a polyline there and back, as strokeInto outlines it
}

// pathJob is the mask drawPath asks the text system for, which draws it
// with the engine's rasterizer and buffer when it is not cached.
type pathJob struct {
	f         *flatPath
	w, h      int
	hw        float32 // half the stroke's width; 0 fills
	x0, y0    float32
	rasterize func() (w, h int, pix []byte)
}

// drawPath paints a flattened path with c, from a mask cached by the shape:
// filled (non-zero winding), or with width > 0 stroked that many pixels
// wide with round joins and caps.
func (p *Painter) drawPath(f *flatPath, width float32, c Color, g *LinearGradient) {
	p.drawPathPaint(f, width, c, g, nil)
}

// drawPathPaint is drawPath in a linear gradient (g) or a conic one (cg).
func (p *Painter) drawPathPaint(f *flatPath, width float32, c Color, g *LinearGradient, cg *ConicGradient) {
	if f.polys() == 0 {
		return
	}
	hw := max(width, 0) / 2
	minX, minY := float32(math.MaxFloat32), float32(math.MaxFloat32)
	maxX, maxY := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
	for _, pt := range f.pts {
		minX, maxX = min(minX, pt[0]), max(maxX, pt[0])
		minY, maxY = min(minY, pt[1]), max(maxY, pt[1])
	}
	x0, y0 := float32(math.Floor(float64(minX-hw))), float32(math.Floor(float64(minY-hw)))
	w, h := int(math.Ceil(float64(maxX+hw-x0)))+1, int(math.Ceil(float64(maxY+hw-y0)))+1
	if w <= 0 || h <= 0 || w > 4096 || h > 4096 {
		return
	}
	// The key covers the shape relative to the pixel grid, in quarters of
	// a pixel; a stroke's is its center line's, and its width. The points
	// move to those quarters, so that the mask of a key is the same
	// whichever path drew it first: a path looks the same wherever it is
	// on the grid, whatever drew before it.
	for i, pt := range f.pts {
		f.pts[i] = [2]float32{x0 + round((pt[0]-x0)*4)/4, y0 + round((pt[1]-y0)*4)/4}
	}
	key := uint64(1)
	if hw > 0 {
		key = 2 + uint64(math.Float32bits(hw))<<8
	}
	var buf [8]byte
	var hs maphash.Hash
	hs.SetSeed(pathSeed)
	binary.LittleEndian.PutUint64(buf[:], key)
	hs.Write(buf[:])
	for i := range f.polys() {
		for _, pt := range f.poly(i) {
			binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(round((pt[0]-x0)*4)))
			binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(round((pt[1]-y0)*4)))
			hs.Write(buf[:])
		}
		end := byte(0xff)
		if f.closed[i] {
			end = 0xfe
		}
		hs.WriteByte(end)
	}
	j := &p.rt.paths.job
	if j.rasterize == nil {
		j.rasterize = p.rt.rasterizePath
	}
	j.f, j.w, j.h, j.hw, j.x0, j.y0 = f, w, h, hw, x0, y0
	gi := p.rt.text.Mask(hs.Sum64(), j.rasterize)
	j.f = nil
	if !gi.OK {
		return
	}
	start := int32(len(p.s.Glyphs))
	c = c.Alpha(p.opacity)
	p.s.Glyphs = append(p.s.Glyphs, scene.Glyph{X: x0, Y: y0, W: float32(gi.W), H: float32(gi.H), U: gi.X, V: gi.Y, UW: gi.W, VH: gi.H, Color: c.scene(), Wide: p.glyphWide(c)})
	op := scene.Op{Kind: scene.OpGlyphs, Start: start, End: start + 1}
	if g != nil || cg != nil {
		// The gradient spans the path's bounds.
		op.Rect = scene.Rect{X: minX - hw, Y: minY - hw, W: maxX - minX + 2*hw, H: maxY - minY + 2*hw}
		op.Opacity = p.opacity
		if g != nil {
			p.gradient(&op, *g)
			op.Wide = p.wide(g.From, g.To, Color{})
		} else {
			p.conic(&op, *cg)
		}
	}
	p.add(op)
}

// rasterizePath draws the mask of the path job, into memory the engine
// reuses: the text system copies it into its atlas.
func (rt *engine) rasterizePath() (int, int, []byte) {
	j := &rt.paths.job
	z := &rt.paths.raster
	n := j.w * j.h
	if cap(rt.paths.pix) < n {
		rt.paths.pix = make([]byte, n)
	}
	pix := rt.paths.pix[:n]
	z.Reset(j.w, j.h)
	if j.hw > 0 {
		strokeInto(z, &rt.paths.loop, j.f, j.hw, j.x0, j.y0)
	} else {
		fillInto(z, j.f, j.x0, j.y0)
	}
	z.Mask(pix, j.w)
	return j.w, j.h, pix
}

// FillPath fills a path (non-zero winding).
func (p *Painter) FillPath(path *Path, c Color) {
	f := &p.rt.paths.flat
	path.flatten(f, p.scale)
	p.drawPath(f, 0, c, nil)
}

// FillPathGradient fills a path with a gradient across its bounds, as the
// area under a chart's line.
func (p *Painter) FillPathGradient(path *Path, g LinearGradient) {
	f := &p.rt.paths.flat
	path.flatten(f, p.scale)
	p.drawPath(f, 0, g.From, &g)
}

// StrokePath draws the outline of a path, width DIPs wide, with round
// joins and caps.
func (p *Painter) StrokePath(path *Path, width float32, c Color) {
	f := &p.rt.paths.flat
	path.flatten(f, p.scale)
	p.drawPath(f, width*p.scale, c.Alpha(f.capFade(width*p.scale)), nil)
}

// capFade returns how opaque a stroke width pixels wide of f shows:
// trimmed shorter than its width, which its round caps would show as a
// dot appearing or going at once, it fades with its length.
func (f *flatPath) capFade(width float32) float32 {
	if !f.trimmed || width <= 0 || f.kept >= width {
		return 1
	}
	return f.kept / width
}

// FillPathConicGradient fills a path with a conic gradient around its
// bounds.
func (p *Painter) FillPathConicGradient(path *Path, g ConicGradient) {
	f := &p.rt.paths.flat
	path.flatten(f, p.scale)
	p.drawPathPaint(f, 0, Color{R: 255, G: 255, B: 255, A: 255}, nil, &g)
}

// StrokePathConicGradient draws the outline of a path as StrokePath does,
// in a conic gradient around its bounds.
func (p *Painter) StrokePathConicGradient(path *Path, width float32, g ConicGradient) {
	f := &p.rt.paths.flat
	path.flatten(f, p.scale)
	// Trimmed short, it fades as its caps would (capFade).
	saved := p.opacity
	p.opacity *= f.capFade(width * p.scale)
	p.drawPathPaint(f, width*p.scale, Color{R: 255, G: 255, B: 255, A: 255}, nil, &g)
	p.opacity = saved
}

// StrokePathGradient draws the outline of a path as StrokePath does, in a
// gradient across its bounds.
func (p *Painter) StrokePathGradient(path *Path, width float32, g LinearGradient) {
	f := &p.rt.paths.flat
	path.flatten(f, p.scale)
	k := f.capFade(width * p.scale)
	g.From, g.To = g.From.Alpha(k), g.To.Alpha(k)
	p.drawPath(f, width*p.scale, g.From, &g)
}

func fillInto(z *vec.Rasterizer, f *flatPath, x0, y0 float32) {
	for i := range f.polys() {
		poly := f.poly(i)
		z.MoveTo(poly[0][0]-x0, poly[0][1]-y0)
		for _, pt := range poly[1:] {
			z.LineTo(pt[0]-x0, pt[1]-y0)
		}
		z.ClosePath()
	}
}

// strokeInto draws a stroke hw pixels to each side of a flattened path,
// with round joins and caps, as an outline that the rasterizer fills: the
// side of a polyline, an arc around each outer corner and a turn through
// the corner on the inside, and back along the other side. Each edge of
// the stroke is drawn once, so that its antialiasing is exact, as
// rasterizing a piece per segment and per corner and adding them up is
// not: where their edges meet, as along the outside of curves, their
// coverage would add up and thicken the stroke.
func strokeInto(z *vec.Rasterizer, loop *[][2]float32, f *flatPath, hw, x0, y0 float32) {
	for i := range f.polys() {
		// The polyline without repeated points, moved to the mask.
		pts := (*loop)[:0]
		for _, pt := range f.poly(i) {
			pt = [2]float32{pt[0] - x0, pt[1] - y0}
			if n := len(pts); n == 0 || pts[n-1] != pt {
				pts = append(pts, pt)
			}
		}
		if f.closed[i] && len(pts) > 2 && pts[0] == pts[len(pts)-1] {
			pts = pts[:len(pts)-1]
		}
		switch {
		case len(pts) == 1:
			disc(z, pts[0][0], pts[0][1], hw)
		case f.closed[i] && len(pts) > 2:
			// Each side of a closed polyline is a loop of its own, the
			// other side going the other way, so that the inside is not
			// filled.
			n := len(pts)
			offsetLoop(z, pts, hw)
			for k := range n / 2 {
				pts[k], pts[n-1-k] = pts[n-1-k], pts[k]
			}
			offsetLoop(z, pts, hw)
		default:
			// An open polyline is outlined as the loop that goes there and
			// back, whose turns at the ends are its caps.
			n := len(pts)
			for k := n - 2; k > 0; k-- {
				pts = append(pts, pts[k])
			}
			offsetLoop(z, pts, hw)
		}
		*loop = pts[:0]
	}
}

// offsetLoop draws the side of a closed polyline hw pixels to its left
// (a segment's normal), turning around the outside of its corners in arcs
// and through the corners on the inside.
func offsetLoop(z *vec.Rasterizer, pts [][2]float32, hw float32) {
	n := len(pts)
	normal := func(k int) (float32, float32, float32, float32) {
		a, b := pts[k], pts[(k+1)%n]
		dx, dy := b[0]-a[0], b[1]-a[1]
		l := float32(math.Hypot(float64(dx), float64(dy)))
		dx, dy = dx/l, dy/l
		return dx, dy, -dy * hw, dx * hw
	}
	dx, dy, nx, ny := normal(0)
	z.MoveTo(pts[0][0]+nx, pts[0][1]+ny)
	for k := range n {
		b := pts[(k+1)%n]
		z.LineTo(b[0]+nx, b[1]+ny)
		ex, ey, mx, my := normal((k + 1) % n)
		// Rounded apart, as fused multiply-adds would leave a polyline
		// turning straight back, a cap's turn, a hair off zero.
		cross, dot := float32(dx*ey)-float32(dy*ex), float32(dx*ex)+float32(dy*ey)
		switch {
		case cross > 0:
			// The inside of the corner: through it, where the stroke
			// covers itself.
			z.LineTo(b[0], b[1])
			z.LineTo(b[0]+mx, b[1]+my)
		case cross < 0 || dot < 0:
			// The outside, or the polyline turning back: an arc around
			// the corner, the shorter way, or the way the segment went.
			from := math.Atan2(float64(ny), float64(nx))
			turn := math.Atan2(float64(cross), float64(dot))
			if cross == 0 {
				turn = -math.Pi
			}
			steps := int(math.Ceil(math.Abs(turn) / (2 * math.Acos(max(1-0.02/float64(hw), -1)))))
			for s := 1; s <= steps; s++ {
				a := from + turn*float64(s)/float64(steps)
				sin, cos := math.Sincos(a)
				z.LineTo(b[0]+hw*float32(cos), b[1]+hw*float32(sin))
			}
		}
		dx, dy, nx, ny = ex, ey, mx, my
	}
	z.ClosePath()
}

// disc draws a circle of radius r, in four cubic curves.
func disc(z *vec.Rasterizer, cx, cy, r float32) {
	const k = 0.5522847498
	z.MoveTo(cx+r, cy)
	z.CubeTo(cx+r, cy+r*k, cx+r*k, cy+r, cx, cy+r)
	z.CubeTo(cx-r*k, cy+r, cx-r, cy+r*k, cx-r, cy)
	z.CubeTo(cx-r, cy-r*k, cx-r*k, cy-r, cx, cy-r)
	z.CubeTo(cx+r*k, cy-r, cx+r, cy-r*k, cx+r, cy)
	z.ClosePath()
}
