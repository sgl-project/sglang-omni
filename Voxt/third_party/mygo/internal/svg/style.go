package svg

import (
	"math"
	"strings"
)

// rgba is a straight color, its components from 0 to 1.
type rgba struct{ r, g, b, a float64 }

var black = rgba{0, 0, 0, 1}

// Kinds of paint specifications.
const (
	specNone uint8 = iota
	specColor
	specCurrent
	specURL
)

// paintSpec is a fill or stroke as written: none, a color, currentColor,
// or a reference to a gradient with what to paint when it is not one.
type paintSpec struct {
	kind     uint8
	color    rgba
	id       string
	fallback uint8 // specNone, specColor or specCurrent, with color
}

// style is the computed style of an element.
type style struct {
	// Inherited.
	fill, stroke  paintSpec
	fillOpacity   float64
	strokeOpacity float64
	fillEvenOdd   bool
	clipEvenOdd   bool
	strokeWidth   float64
	miterLimit    float64
	cap, join     uint8
	dashes        []float64
	dashOffset    float64
	// color is the color property, which currentColor stands for; until a
	// document sets it, currentColor is the color it is drawn with.
	color       rgba
	colorSet    bool
	visible     bool
	strokeFirst bool

	// Not inherited.
	opacity     float64
	display     bool
	clipPath    string
	mask        string
	maskAlpha   bool
	stopColor   paintSpec
	stopOpacity float64
}

func initialStyle() style {
	return style{
		fill:        paintSpec{kind: specColor, color: black},
		fillOpacity: 1, strokeOpacity: 1, strokeWidth: 1, miterLimit: 4,
		visible: true, opacity: 1, display: true,
		stopColor: paintSpec{kind: specColor, color: black}, stopOpacity: 1,
	}
}

// properties are the properties that attributes, style sheets and style
// attributes set.
var properties = map[string]bool{
	"fill": true, "fill-opacity": true, "fill-rule": true,
	"stroke": true, "stroke-width": true, "stroke-opacity": true, "stroke-linecap": true,
	"stroke-linejoin": true, "stroke-miterlimit": true, "stroke-dasharray": true, "stroke-dashoffset": true,
	"opacity": true, "color": true, "display": true, "visibility": true, "paint-order": true,
	"clip-path": true, "clip-rule": true, "mask": true, "mask-type": true,
	"stop-color": true, "stop-opacity": true,
}

// computeStyle computes the style of e, a child of an element of style
// parent, with lengths in percent of the viewport vw×vh.
func computeStyle(e *elem, parent *style, vw, vh float64) style {
	s := *parent
	s.opacity, s.display, s.clipPath, s.mask, s.maskAlpha = 1, true, "", "", false
	s.stopColor, s.stopOpacity = paintSpec{kind: specColor, color: black}, 1
	diag := math.Sqrt((vw*vw + vh*vh) / 2)
	for _, d := range e.props {
		v := d.value
		if v == "inherit" {
			s.inherit(d.name, parent)
			continue
		}
		switch d.name {
		case "fill":
			if p, ok := parsePaint(v); ok {
				s.fill = p
			}
		case "stroke":
			if p, ok := parsePaint(v); ok {
				s.stroke = p
			}
		case "fill-opacity":
			s.fillOpacity = parseOpacity(v, s.fillOpacity)
		case "stroke-opacity":
			s.strokeOpacity = parseOpacity(v, s.strokeOpacity)
		case "opacity":
			s.opacity = parseOpacity(v, s.opacity)
		case "stop-opacity":
			s.stopOpacity = parseOpacity(v, s.stopOpacity)
		case "fill-rule":
			if v == "evenodd" || v == "nonzero" {
				s.fillEvenOdd = v == "evenodd"
			}
		case "clip-rule":
			if v == "evenodd" || v == "nonzero" {
				s.clipEvenOdd = v == "evenodd"
			}
		case "stroke-width":
			if w, ok := parseLength(v, diag); ok && w >= 0 {
				s.strokeWidth = w
			}
		case "stroke-miterlimit":
			if l, ok := parseNumber(v); ok && l >= 1 {
				s.miterLimit = l
			}
		case "stroke-linecap":
			switch v {
			case "butt":
				s.cap = capButt
			case "round":
				s.cap = capRound
			case "square":
				s.cap = capSquare
			}
		case "stroke-linejoin":
			switch v {
			case "miter", "miter-clip", "arcs":
				s.join = joinMiter
			case "round":
				s.join = joinRound
			case "bevel":
				s.join = joinBevel
			}
		case "stroke-dasharray":
			s.dashes = parseDashes(v, diag)
		case "stroke-dashoffset":
			if o, ok := parseLength(v, diag); ok {
				s.dashOffset = o
			}
		case "color":
			if strings.EqualFold(v, "currentColor") {
				break // the color it inherits
			}
			if c, ok := parseColor(v); ok {
				s.color, s.colorSet = c, true
			}
		case "display":
			s.display = v != "none"
		case "visibility":
			if v == "visible" || v == "hidden" || v == "collapse" {
				s.visible = v == "visible"
			}
		case "paint-order":
			s.strokeFirst = strokeFirst(v)
		case "clip-path":
			s.clipPath = urlID(v)
		case "mask":
			s.mask = urlID(v)
		case "mask-type":
			s.maskAlpha = v == "alpha"
		case "stop-color":
			if strings.EqualFold(v, "currentColor") {
				s.stopColor = paintSpec{kind: specCurrent}
			} else if c, ok := parseColor(v); ok {
				s.stopColor = paintSpec{kind: specColor, color: c}
			}
		}
	}
	return s
}

// inherit takes a property from the parent's style, undoing what an
// earlier declaration set.
func (s *style) inherit(name string, p *style) {
	switch name {
	case "fill":
		s.fill = p.fill
	case "stroke":
		s.stroke = p.stroke
	case "fill-opacity":
		s.fillOpacity = p.fillOpacity
	case "stroke-opacity":
		s.strokeOpacity = p.strokeOpacity
	case "fill-rule":
		s.fillEvenOdd = p.fillEvenOdd
	case "clip-rule":
		s.clipEvenOdd = p.clipEvenOdd
	case "stroke-width":
		s.strokeWidth = p.strokeWidth
	case "stroke-miterlimit":
		s.miterLimit = p.miterLimit
	case "stroke-linecap":
		s.cap = p.cap
	case "stroke-linejoin":
		s.join = p.join
	case "stroke-dasharray":
		s.dashes = p.dashes
	case "stroke-dashoffset":
		s.dashOffset = p.dashOffset
	case "color":
		s.color, s.colorSet = p.color, p.colorSet
	case "visibility":
		s.visible = p.visible
	case "paint-order":
		s.strokeFirst = p.strokeFirst
	case "opacity":
		s.opacity = p.opacity
	case "display":
		s.display = p.display
	case "clip-path":
		s.clipPath = p.clipPath
	case "mask":
		s.mask = p.mask
	case "mask-type":
		s.maskAlpha = p.maskAlpha
	case "stop-color":
		s.stopColor = p.stopColor
	case "stop-opacity":
		s.stopOpacity = p.stopOpacity
	}
}

// strokeFirst reports whether a paint-order paints the stroke before the
// fill.
func strokeFirst(v string) bool {
	for _, f := range strings.Fields(v) {
		switch f {
		case "stroke":
			return true
		case "fill":
			return false
		}
	}
	return false
}

// urlID returns the id url(#id) refers to.
func urlID(v string) string {
	v = strings.TrimSpace(v)
	if !strings.HasPrefix(v, "url(") {
		return ""
	}
	end := strings.IndexByte(v, ')')
	if end < 0 {
		return ""
	}
	ref := strings.Trim(strings.TrimSpace(v[4:end]), `"'`)
	if !strings.HasPrefix(ref, "#") {
		return ""
	}
	return ref[1:]
}

func parsePaint(v string) (paintSpec, bool) {
	switch {
	case v == "none":
		return paintSpec{kind: specNone}, true
	case strings.EqualFold(v, "currentColor"):
		return paintSpec{kind: specCurrent}, true
	case strings.HasPrefix(v, "url("):
		id := urlID(v)
		if id == "" {
			return paintSpec{}, false
		}
		p := paintSpec{kind: specURL, id: id}
		if rest := strings.TrimSpace(v[strings.IndexByte(v, ')')+1:]); rest != "" {
			fb, ok := parsePaint(rest)
			if !ok || fb.kind == specURL {
				return paintSpec{}, false
			}
			p.fallback, p.color = fb.kind, fb.color
		}
		return p, true
	}
	c, ok := parseColor(v)
	return paintSpec{kind: specColor, color: c}, ok
}

// parseColor parses a CSS color: a name, #rgb, #rgba, #rrggbb, #rrggbbaa,
// rgb(), rgba(), hsl(), hsla() or transparent.
func parseColor(v string) (rgba, bool) {
	v = strings.TrimSpace(v)
	// An ICC color follows the sRGB one, which is all that is drawn.
	if i := strings.Index(v, "icc-color("); i > 0 {
		v = strings.TrimSpace(v[:i])
	}
	if v == "" {
		return rgba{}, false
	}
	if v[0] == '#' {
		return parseHex(v[1:])
	}
	lower := strings.ToLower(v)
	if open := strings.IndexByte(lower, '('); open > 0 && strings.HasSuffix(lower, ")") {
		name, args := lower[:open], lower[open+1:len(lower)-1]
		switch name {
		case "rgb", "rgba":
			return parseRGB(args)
		case "hsl", "hsla":
			return parseHSL(args)
		}
		return rgba{}, false
	}
	if lower == "transparent" {
		return rgba{}, true
	}
	if c, ok := namedColors[lower]; ok {
		return rgba{float64(c>>16) / 255, float64(c>>8&0xff) / 255, float64(c&0xff) / 255, 1}, true
	}
	return rgba{}, false
}

func parseHex(h string) (rgba, bool) {
	var d [8]float64
	for i := range len(h) {
		c := h[i]
		switch {
		case i >= len(d):
			return rgba{}, false
		case c >= '0' && c <= '9':
			d[i] = float64(c - '0')
		case c >= 'a' && c <= 'f':
			d[i] = float64(c - 'a' + 10)
		case c >= 'A' && c <= 'F':
			d[i] = float64(c - 'A' + 10)
		default:
			return rgba{}, false
		}
	}
	switch len(h) {
	case 3, 4:
		c := rgba{d[0] * 17 / 255, d[1] * 17 / 255, d[2] * 17 / 255, 1}
		if len(h) == 4 {
			c.a = d[3] * 17 / 255
		}
		return c, true
	case 6, 8:
		c := rgba{(d[0]*16 + d[1]) / 255, (d[2]*16 + d[3]) / 255, (d[4]*16 + d[5]) / 255, 1}
		if len(h) == 8 {
			c.a = (d[6]*16 + d[7]) / 255
		}
		return c, true
	}
	return rgba{}, false
}

// colorArgs splits the arguments of rgb() or hsl(), separated by commas or
// spaces, with the alpha after a slash in the latter syntax.
func colorArgs(s string) []string {
	s = strings.ReplaceAll(s, "/", " / ")
	f := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' || r == '\n' || r == '\r' })
	if len(f) == 5 && f[3] == "/" {
		return []string{f[0], f[1], f[2], f[4]}
	}
	for _, a := range f {
		if a == "/" {
			return nil
		}
	}
	return f
}

func parseRGB(s string) (rgba, bool) {
	args := colorArgs(s)
	if len(args) != 3 && len(args) != 4 {
		return rgba{}, false
	}
	var c [4]float64
	c[3] = 1
	for i, a := range args {
		pct := strings.HasSuffix(a, "%")
		n, ok := parseNumber(strings.TrimSuffix(a, "%"))
		if !ok {
			return rgba{}, false
		}
		switch {
		case i == 3 && pct:
			c[i] = n / 100
		case i == 3:
			c[i] = n
		case pct:
			c[i] = n / 100
		default:
			c[i] = n / 255
		}
		c[i] = math.Max(0, math.Min(1, c[i]))
	}
	return rgba{c[0], c[1], c[2], c[3]}, true
}

func parseHSL(s string) (rgba, bool) {
	args := colorArgs(s)
	if len(args) != 3 && len(args) != 4 {
		return rgba{}, false
	}
	h := args[0]
	scale := 1.0
	switch {
	case strings.HasSuffix(h, "deg"):
		h = h[:len(h)-3]
	case strings.HasSuffix(h, "grad"):
		h, scale = h[:len(h)-4], 0.9
	case strings.HasSuffix(h, "rad"):
		h, scale = h[:len(h)-3], 180/math.Pi
	case strings.HasSuffix(h, "turn"):
		h, scale = h[:len(h)-4], 360
	}
	hue, ok := parseNumber(h)
	if !ok {
		return rgba{}, false
	}
	var v [3]float64
	v[2] = 1
	for i, a := range args[1:] {
		pct := strings.HasSuffix(a, "%")
		n, ok := parseNumber(strings.TrimSuffix(a, "%"))
		if !ok || i < 2 && !pct {
			return rgba{}, false
		}
		if pct {
			n /= 100
		}
		v[i] = math.Max(0, math.Min(1, n))
	}
	hue = math.Mod(hue*scale, 360)
	if hue < 0 {
		hue += 360
	}
	sat, light := v[0], v[1]
	f := func(n float64) float64 {
		k := math.Mod(n+hue/30, 12)
		a := sat * math.Min(light, 1-light)
		return light - a*math.Max(-1, math.Min(math.Min(k-3, 9-k), 1))
	}
	return rgba{f(0), f(8), f(4), v[2]}, true
}

func parseNumber(v string) (float64, bool) {
	p := scanner{s: strings.TrimSpace(v)}
	n, ok := p.number()
	return n, ok && p.done()
}

// parseOpacity parses a number or percentage, clamped to [0, 1], or
// returns old.
func parseOpacity(v string, old float64) float64 {
	pct := strings.HasSuffix(v, "%")
	n, ok := parseNumber(strings.TrimSuffix(v, "%"))
	if !ok {
		return old
	}
	if pct {
		n /= 100
	}
	return math.Max(0, math.Min(1, n))
}

// parseLength parses a length: a number with an optional unit, in pixels,
// or a percentage of base.
func parseLength(v string, base float64) (float64, bool) {
	p := scanner{s: strings.TrimSpace(v)}
	n, ok := p.number()
	if !ok {
		return 0, false
	}
	switch strings.ToLower(p.s[p.i:]) {
	case "", "px":
		return n, true
	case "%":
		return n * base / 100, true
	case "pt":
		return n * 4 / 3, true
	case "pc":
		return n * 16, true
	case "in":
		return n * 96, true
	case "cm":
		return n * 96 / 2.54, true
	case "mm":
		return n * 96 / 25.4, true
	case "q":
		return n * 96 / 101.6, true
	case "em", "rem":
		return n * 16, true
	case "ex":
		return n * 8, true
	}
	return 0, false
}

// parseDashes parses a dash array: nil for a solid line, as none or an
// array in error asks, and of even length otherwise.
func parseDashes(v string, diag float64) []float64 {
	if v == "none" {
		return nil
	}
	var out []float64
	var total float64
	for _, f := range strings.FieldsFunc(v, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' || r == '\n' || r == '\r' }) {
		l, ok := parseLength(f, diag)
		if !ok || l < 0 {
			return nil
		}
		out = append(out, l)
		total += l
	}
	if total <= 0 {
		return nil
	}
	if len(out)%2 == 1 {
		out = append(out, out...)
	}
	return out
}

// aspect is a preserveAspectRatio: how a viewBox fits its viewport.
type aspect struct {
	none   bool
	ax, ay float64 // where it aligns: 0 at the start, 0.5 in the middle, 1 at the end
	slice  bool
}

func parseAspect(v string) aspect {
	a := aspect{ax: 0.5, ay: 0.5}
	f := strings.Fields(v)
	if len(f) > 0 && f[0] == "defer" {
		f = f[1:]
	}
	if len(f) == 0 {
		return a
	}
	if f[0] == "none" {
		a.none = true
		return a
	}
	pos := map[string]float64{"Min": 0, "Mid": 0.5, "Max": 1}
	al := f[0]
	if len(al) != 8 || al[0] != 'x' || al[4] != 'Y' {
		return a
	}
	x, okX := pos[al[1:4]]
	y, okY := pos[al[5:8]]
	if !okX || !okY {
		return a
	}
	a.ax, a.ay = x, y
	a.slice = len(f) > 1 && f[1] == "slice"
	return a
}

// viewBoxTransform maps a viewBox to a viewport w×h as a says.
func viewBoxTransform(vb box, a aspect, w, h float64) matrix {
	if vb.w <= 0 || vb.h <= 0 {
		return identity
	}
	sx, sy := w/vb.w, h/vb.h
	if a.none {
		return matrix{sx, 0, 0, sy, -vb.x * sx, -vb.y * sy}
	}
	s := math.Min(sx, sy)
	if a.slice {
		s = math.Max(sx, sy)
	}
	return matrix{s, 0, 0, s, -vb.x*s + (w-vb.w*s)*a.ax, -vb.y*s + (h-vb.h*s)*a.ay}
}

func parseViewBox(v string) (box, bool) {
	n := numbers(v)
	if len(n) != 4 || n[2] <= 0 || n[3] <= 0 {
		return box{}, false
	}
	return box{n[0], n[1], n[2], n[3]}, true
}
