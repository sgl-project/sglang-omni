package terminal

import (
	"math"

	"github.com/egoist/mygo/ui"
)

// Characters of box drawing, blocks and Powerline's separators are drawn
// rather than taken from the font, as Ghostty draws them: fonts leave gaps
// between the lines of cells taller than their glyphs, and many lack the
// separators.

// Weights of the arms of box drawing characters.
const (
	armNone = iota
	armLight
	armHeavy
	armDouble
)

// boxArms holds the arms of U+2500 to U+257F, from their Unicode names:
// the weights of up, right, down and left, a digit each; "" for those
// drawn otherwise (arcs, diagonals) or by the font (dashes).
var boxArms = [128]string{
	"0101", "0202", "1010", "2020", "", "", "", "", "", "", "", "", "0110", "0210", "0120", "0220",
	"0011", "0012", "0021", "0022", "1100", "1200", "2100", "2200", "1001", "1002", "2001", "2002", "1110", "1210", "2110", "1120",
	"2120", "2210", "1220", "2220", "1011", "1012", "2011", "1021", "2021", "2012", "1022", "2022", "0111", "0112", "0211", "0212",
	"0121", "0122", "0221", "0222", "1101", "1102", "1201", "1202", "2101", "2102", "2201", "2202", "1111", "1112", "1211", "1212",
	"2111", "1121", "2121", "2112", "2211", "1122", "1221", "2212", "1222", "2122", "2221", "2222", "", "", "", "",
	"0303", "3030", "0310", "0130", "0330", "0013", "0031", "0033", "1300", "3100", "3300", "1003", "3001", "3003", "1310", "3130",
	"3330", "1013", "3031", "3033", "0313", "0131", "0333", "1303", "3101", "3303", "1313", "3131", "3333", "", "", "",
	"", "", "", "", "0001", "1000", "0100", "0010", "0002", "2000", "0200", "0020", "0201", "1020", "0102", "2010",
}

// drawn reports whether the terminal draws r itself.
func drawn(r rune) bool {
	switch {
	case 0x2500 <= r && r <= 0x257f:
		return boxArms[r-0x2500] != "" || 0x256d <= r && r <= 0x2573
	case 0x2580 <= r && r <= 0x259f:
		return true
	case 0xe0b0 <= r && r <= 0xe0b3:
		return true
	}
	return false
}

// drawBox draws r in cell c.
func drawBox(p *ui.Painter, r rune, c ui.Rect, col ui.Color) {
	s := p.Scale()
	// The cell in device pixels.
	x0, y0 := float32(math.Round(float64(c.X*s))), float32(math.Round(float64(c.Y*s)))
	x1, y1 := float32(math.Round(float64((c.X+c.W)*s))), float32(math.Round(float64((c.Y+c.H)*s)))
	fill := func(ax, ay, bx, by float32) {
		if bx > ax && by > ay {
			p.Fill(ui.Rect{X: ax / s, Y: ay / s, W: (bx - ax) / s, H: (by - ay) / s}, col, 0)
		}
	}
	w, h := x1-x0, y1-y0
	switch {
	case 0x2580 <= r && r <= 0x259f:
		drawBlock(r, x0, y0, w, h, fill, p, col, s)
		return
	case 0xe0b0 <= r && r <= 0xe0b3:
		var path ui.Path
		if r == 0xe0b0 || r == 0xe0b1 {
			path.MoveTo(x0/s, y0/s).LineTo(x1/s, (y0+h/2)/s).LineTo(x0/s, y1/s)
		} else {
			path.MoveTo(x1/s, y0/s).LineTo(x0/s, (y0+h/2)/s).LineTo(x1/s, y1/s)
		}
		if r == 0xe0b0 || r == 0xe0b2 {
			path.Close()
			p.FillPath(&path, col)
		} else {
			p.StrokePath(&path, max(1, float32(math.Round(float64(s))))/s, col)
		}
		return
	}
	light := max(1, float32(math.Round(float64(s))))
	cx := x0 + float32(math.Floor(float64(w-light)/2))
	cy := y0 + float32(math.Floor(float64(h-light)/2))
	if 0x256d <= r && r <= 0x2573 {
		var path ui.Path
		mx, my := (cx+light/2)/s, (cy+light/2)/s
		rad := min(w, h) / 2 / s
		switch r {
		case 0x256d: // ╭
			path.MoveTo(mx, y1/s).LineTo(mx, my+rad).QuadTo(mx, my, mx+rad, my).LineTo(x1/s, my)
		case 0x256e: // ╮
			path.MoveTo(mx, y1/s).LineTo(mx, my+rad).QuadTo(mx, my, mx-rad, my).LineTo(x0/s, my)
		case 0x256f: // ╯
			path.MoveTo(mx, y0/s).LineTo(mx, my-rad).QuadTo(mx, my, mx-rad, my).LineTo(x0/s, my)
		case 0x2570: // ╰
			path.MoveTo(mx, y0/s).LineTo(mx, my-rad).QuadTo(mx, my, mx+rad, my).LineTo(x1/s, my)
		case 0x2571: // ╱
			path.MoveTo(x1/s, y0/s).LineTo(x0/s, y1/s)
		case 0x2572: // ╲
			path.MoveTo(x0/s, y0/s).LineTo(x1/s, y1/s)
		case 0x2573: // ╳
			path.MoveTo(x1/s, y0/s).LineTo(x0/s, y1/s)
			p.StrokePath(&path, light/s, col)
			path = ui.Path{}
			path.MoveTo(x0/s, y0/s).LineTo(x1/s, y1/s)
		}
		p.StrokePath(&path, light/s, col)
		return
	}
	arms := boxArms[r-0x2500]
	if arms == "" {
		return
	}
	drawArms(fill, [4]int{int(arms[0] - '0'), int(arms[1] - '0'), int(arms[2] - '0'), int(arms[3] - '0')}, x0, y0, x1, y1, cx+light/2, cy+light/2, light)
}

// drawArms draws the arms of a box drawing character (up, right, down,
// left) in the cell from x0, y0 to x1, y1, whose lines cross at mx, my,
// light pixels wide. Light and heavy arms are lines; a double arm is two
// light lines, a light line apart, which turn into the arm across them as
// a corner does, or go through.
func drawArms(fill func(ax, ay, bx, by float32), a [4]int, x0, y0, x1, y1, mx, my, light float32) {
	up, right, down, left := a[0], a[1], a[2], a[3]
	off := light // from the middle to each line of a double arm
	vDouble := up == armDouble || down == armDouble
	hDouble := left == armDouble || right == armDouble
	thick := func(w int) float32 {
		switch w {
		case armHeavy:
			return 2 * light
		case armDouble:
			return 2*off + light
		}
		return light
	}
	// The joint is as wide as the widest arm across.
	hw := max(thick(left), thick(right))
	vw := max(thick(up), thick(down))
	single := func(w int) bool { return w == armLight || w == armHeavy }
	if single(up) {
		t, end := thick(up), my+hw/2
		if hDouble && down == armNone {
			end = my - off + light/2 // up to the upper line
		}
		fill(mx-t/2, y0, mx+t/2, end)
	}
	if single(down) {
		t, start := thick(down), my-hw/2
		if hDouble && up == armNone {
			start = my + off - light/2
		}
		fill(mx-t/2, start, mx+t/2, y1)
	}
	if single(left) {
		t, end := thick(left), mx+vw/2
		if vDouble && right == armNone {
			end = mx - off + light/2
		}
		fill(x0, my-t/2, end, my+t/2)
	}
	if single(right) {
		t, start := thick(right), mx-vw/2
		if vDouble && left == armNone {
			start = mx + off - light/2
		}
		fill(start, my-t/2, x1, my+t/2)
	}
	// The lines of double arms. The one on a side with a double arm across
	// is inside the corner it turns, and the other outside.
	for _, side := range []float32{-1, 1} {
		if vDouble {
			lx := mx + side*off - light/2
			across := left
			if side > 0 {
				across = right
			}
			if up == armDouble && down == armDouble && across != armDouble {
				fill(lx, y0, lx+light, y1)
			} else {
				// Where the line meets the arms across: inside or outside
				// their double lines, or at their single line.
				in, out := my-off+light/2, my+off+light/2
				if across == armDouble {
					out = in
				} else if !hDouble {
					out = my + light/2
				}
				if up == armDouble {
					fill(lx, y0, lx+light, out)
				}
				in, out = my+off-light/2, my-off-light/2
				if across == armDouble {
					out = in
				} else if !hDouble {
					out = my - light/2
				}
				if down == armDouble {
					fill(lx, out, lx+light, y1)
				}
			}
		}
		if hDouble {
			ly := my + side*off - light/2
			across := up
			if side > 0 {
				across = down
			}
			if left == armDouble && right == armDouble && across != armDouble {
				fill(x0, ly, x1, ly+light)
			} else {
				in, out := mx-off+light/2, mx+off+light/2
				if across == armDouble {
					out = in
				} else if !vDouble {
					out = mx + light/2
				}
				if left == armDouble {
					fill(x0, ly, out, ly+light)
				}
				in, out = mx+off-light/2, mx-off-light/2
				if across == armDouble {
					out = in
				} else if !vDouble {
					out = mx - light/2
				}
				if right == armDouble {
					fill(out, ly, x1, ly+light)
				}
			}
		}
	}
}

// drawBlock draws a block element (U+2580 to U+259F).
func drawBlock(r rune, x0, y0, w, h float32, fill func(ax, ay, bx, by float32), p *ui.Painter, col ui.Color, s float32) {
	rx := func(f float32) float32 { return x0 + float32(math.Round(float64(w*f))) }
	ry := func(f float32) float32 { return y0 + float32(math.Round(float64(h*f))) }
	x1, y1 := x0+w, y0+h
	switch {
	case r == 0x2580:
		fill(x0, y0, x1, ry(0.5))
	case 0x2581 <= r && r <= 0x2588:
		fill(x0, ry(1-float32(r-0x2580)/8), x1, y1)
	case 0x2589 <= r && r <= 0x258f:
		fill(x0, y0, rx(float32(0x2590-r)/8), y1)
	case r == 0x2590:
		fill(rx(0.5), y0, x1, y1)
	case 0x2591 <= r && r <= 0x2593:
		a := float32(r-0x2590) / 4
		p.Fill(ui.Rect{X: x0 / s, Y: y0 / s, W: w / s, H: h / s}, col.Alpha(a), 0)
	case r == 0x2594:
		fill(x0, y0, x1, ry(1.0/8))
	case r == 0x2595:
		fill(rx(7.0/8), y0, x1, y1)
	default:
		// Quadrants: upper left, upper right, lower left, lower right.
		q := map[rune]string{0x2596: "0010", 0x2597: "0001", 0x2598: "1000", 0x2599: "1011", 0x259a: "1001",
			0x259b: "1110", 0x259c: "1101", 0x259d: "0100", 0x259e: "0110", 0x259f: "0111"}[r]
		mx, my := rx(0.5), ry(0.5)
		if len(q) == 4 {
			if q[0] == '1' {
				fill(x0, y0, mx, my)
			}
			if q[1] == '1' {
				fill(mx, y0, x1, my)
			}
			if q[2] == '1' {
				fill(x0, my, mx, y1)
			}
			if q[3] == '1' {
				fill(mx, my, x1, y1)
			}
		}
	}
}
