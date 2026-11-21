package svg

import (
	"math"
	"slices"
	"strings"
)

// node is a drawing compiled from an element: a shape, or a group of
// nodes. m maps its user space to its parent's.
type node struct {
	m       matrix
	opacity float64
	clip    *clipDef
	mask    *maskDef
	shape   *shape
	kids    []*node
	// bbox is the box around its geometry in its user space, for what
	// is in objectBoundingBox units.
	bbox   box
	hasBox bool
	// single is true when the node paints one thing at most, so that its
	// opacity can apply to it instead of to a layer; flat is true when
	// its parent's opacity can apply to it instead.
	single, flat bool
}

// Kinds of paint.
const (
	paintNone uint8 = iota
	paintColor
	paintCurrent // the color the document is drawn with
	paintGradient
)

type paint struct {
	kind  uint8
	color rgba
	grad  *gradient
}

type shape struct {
	path          path
	fill, stroke  paint
	fillOpacity   float64
	strokeOpacity float64
	evenOdd       bool
	width, miter  float64
	cap, join     uint8
	dashes        []float64
	dashOffset    float64
	strokeFirst   bool
}

// Spread methods.
const (
	spreadPad uint8 = iota
	spreadReflect
	spreadRepeat
)

type gradient struct {
	radial bool
	bbox   bool   // in objectBoundingBox units
	m      matrix // gradientTransform
	spread uint8
	// The line of a linear gradient, the circles of a radial one.
	x1, y1, x2, y2        float64
	cx, cy, r, fx, fy, fr float64
	stops                 []stop
}

type stop struct {
	offset  float64
	color   rgba
	current bool // the color the document is drawn with
	opacity float64
}

// clipDef is a clipPath: the union of its shapes clips.
type clipDef struct {
	bbox bool   // clipPathUnits is objectBoundingBox
	m    matrix // its transform
	kids []*node
	clip *clipDef // its own clip-path
}

// maskDef is a mask: the luminance, or alpha, of its content masks, within
// its region.
type maskDef struct {
	units, content bool // maskUnits and maskContentUnits are objectBoundingBox
	region         box
	alpha          bool
	kids           []*node
}

// compiler compiles a document's elements into nodes.
type compiler struct {
	root   *elem
	ids    map[string]*elem
	styles map[*elem]*style
	grads  map[*elem]*gradient
	clips  map[*elem]*clipDef
	masks  map[*elem]*maskDef
	// using holds the elements being drawn by use elements, which may not
	// use themselves.
	using []*elem
	// budget is how many more nodes the document may compile to.
	budget int
	// vw and vh are the size of the viewport, for percentages.
	vw, vh float64
	base   style
	// current is set once a paint is currentColor.
	current bool
}

func newCompiler(root *elem) *compiler {
	c := &compiler{
		root: root, ids: map[string]*elem{}, styles: map[*elem]*style{},
		grads: map[*elem]*gradient{}, clips: map[*elem]*clipDef{}, masks: map[*elem]*maskDef{},
		budget: 1 << 17, base: initialStyle(),
	}
	walk(root, func(e *elem) {
		if id := e.attrs["id"]; id != "" {
			if _, ok := c.ids[id]; !ok {
				c.ids[id] = e
			}
		}
	})
	return c
}

func (c *compiler) diag() float64 { return math.Sqrt((c.vw*c.vw + c.vh*c.vh) / 2) }

// length parses the attribute name of e as a length, with def when it is
// missing or in error, percentages of base.
func (c *compiler) length(e *elem, name string, base, def float64) float64 {
	if v, ok := e.attrs[name]; ok {
		if l, ok := parseLength(v, base); ok {
			return l
		}
	}
	return def
}

func (c *compiler) style(e *elem, parent *style) style { return computeStyle(e, parent, c.vw, c.vh) }

// domStyle returns the style of e where the document has it: for the
// content of gradients, clip paths and masks, which inherit from where
// they are, not from what they paint.
func (c *compiler) domStyle(e *elem) *style {
	if s, ok := c.styles[e]; ok {
		return s
	}
	parent := &c.base
	if e.parent != nil {
		parent = c.domStyle(e.parent)
	}
	s := c.style(e, parent)
	c.styles[e] = &s
	return &s
}

// doc compiles the document.
func (c *compiler) doc() *Doc {
	root := c.root
	vb, hasVB := parseViewBox(root.attrs["viewBox"])
	w, wok := absLength(root.attrs["width"])
	h, hok := absLength(root.attrs["height"])
	switch {
	case wok && hok:
	case hasVB && wok:
		h = w * vb.h / vb.w
	case hasVB && hok:
		w = h * vb.w / vb.h
	case hasVB:
		w, h = vb.w, vb.h
	default:
		if !wok {
			w = 300
		}
		if !hok {
			h = 150
		}
	}
	if !hasVB {
		vb = box{0, 0, w, h}
	}
	c.vw, c.vh = vb.w, vb.h
	d := &Doc{view: vb, par: parseAspect(root.attrs["preserveAspectRatio"]), w: math.Max(w, 0), h: math.Max(h, 0)}
	st := c.style(root, &c.base)
	if !st.display {
		return d
	}
	n := &node{m: identity}
	c.group(n, root, &st)
	c.finish(n, &st)
	d.root, d.current = n, c.current
	return d
}

// absLength parses an absolute length, ignoring percentages.
func absLength(v string) (float64, bool) {
	if strings.HasSuffix(strings.TrimSpace(v), "%") {
		return 0, false
	}
	return parseLength(v, 0)
}

// group compiles the children of e into n.
func (c *compiler) group(n *node, e *elem, st *style) {
	for _, k := range e.kids {
		if kn := c.elem(k, st); kn != nil {
			n.kids = append(n.kids, kn)
		}
	}
}

// finish gives a node its opacity, clipping and masking, and works out its
// box and whether it is single and flat.
func (c *compiler) finish(n *node, st *style) {
	n.opacity = st.opacity
	n.clip = c.clipRef(st.clipPath)
	n.mask = c.maskRef(st.mask)
	if s := n.shape; s != nil {
		n.bbox, n.hasBox = s.path.bounds()
		n.single = s.fill.kind == paintNone || s.stroke.kind == paintNone
	} else {
		for _, k := range n.kids {
			if !k.hasBox {
				continue
			}
			b := k.bbox.transform(k.m)
			if n.hasBox {
				b = n.bbox.union(b)
			}
			n.bbox, n.hasBox = b, true
		}
		n.single = len(n.kids) == 0 || len(n.kids) == 1 && n.kids[0].flat
	}
	n.flat = n.clip != nil || n.mask != nil || n.single
}

// elem compiles a graphics element, or returns nil for others.
func (c *compiler) elem(e *elem, parent *style) *node {
	switch e.name {
	case "g", "a", "switch", "svg", "use", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon":
	default:
		return nil
	}
	if c.budget <= 0 {
		return nil
	}
	c.budget--
	st := c.style(e, parent)
	if !st.display {
		return nil
	}
	n := &node{m: identity}
	switch e.name {
	case "g", "a":
		c.group(n, e, &st)
	case "switch":
		// The first child whose conditions hold, of which only
		// requiredExtensions fails: MyGo supports no extension.
		for _, k := range e.kids {
			if k.attrs["requiredExtensions"] != "" || k.name == "title" || k.name == "desc" || k.name == "metadata" {
				continue
			}
			if kn := c.elem(k, &st); kn != nil {
				n.kids = append(n.kids, kn)
			}
			break
		}
	case "svg":
		x, y := c.length(e, "x", c.vw, 0), c.length(e, "y", c.vh, 0)
		w, h := c.length(e, "width", c.vw, c.vw), c.length(e, "height", c.vh, c.vh)
		if w <= 0 || h <= 0 {
			return nil
		}
		n.kids = c.viewport(e, &st, x, y, w, h)
	case "use":
		if !c.use(n, e, &st) {
			return nil
		}
	default:
		if !st.visible {
			return nil
		}
		pa := c.shapePath(e)
		if len(pa.ops) == 0 {
			return nil
		}
		n.shape = &shape{
			path: pa, fill: c.paint(st.fill, &st), stroke: c.paint(st.stroke, &st),
			fillOpacity: st.fillOpacity, strokeOpacity: st.strokeOpacity, evenOdd: st.fillEvenOdd,
			width: st.strokeWidth, miter: st.miterLimit, cap: st.cap, join: st.join,
			dashes: st.dashes, dashOffset: st.dashOffset, strokeFirst: st.strokeFirst,
		}
	}
	if n.shape == nil && len(n.kids) == 0 {
		return nil
	}
	if t, ok := e.attrs["transform"]; ok {
		n.m = parseTransform(t).mul(n.m)
	}
	c.finish(n, &st)
	return n
}

// viewport compiles the children of an svg or symbol element e drawn in a
// viewport w×h at (x, y), which clips them.
func (c *compiler) viewport(e *elem, st *style, x, y, w, h float64) []*node {
	vw, vh := c.vw, c.vh
	defer func() { c.vw, c.vh = vw, vh }()
	inner := &node{m: identity}
	if vb, ok := parseViewBox(e.attrs["viewBox"]); ok {
		inner.m = viewBoxTransform(vb, parseAspect(e.attrs["preserveAspectRatio"]), w, h)
		c.vw, c.vh = vb.w, vb.h
	} else {
		c.vw, c.vh = w, h
	}
	c.group(inner, e, st)
	if len(inner.kids) == 0 {
		return nil
	}
	c.finish(inner, &c.base)
	var r path
	r.rect(0, 0, w, h, 0, 0)
	clipped := &node{m: translate(x, y), kids: []*node{inner}}
	cs := c.base
	c.finish(clipped, &cs)
	clipped.clip = &clipDef{m: identity, kids: []*node{{m: identity, shape: &shape{path: r}}}}
	clipped.flat = true
	return []*node{clipped}
}

// use compiles the element a use element e draws into n.
func (c *compiler) use(n *node, e *elem, st *style) bool {
	ref := c.ids[hrefID(e)]
	if ref == nil || slices.Contains(c.using, ref) {
		return false
	}
	for p := e; p != nil; p = p.parent {
		if p == ref {
			return false // it would draw itself
		}
	}
	c.using = append(c.using, ref)
	defer func() { c.using = c.using[:len(c.using)-1] }()
	x, y := c.length(e, "x", c.vw, 0), c.length(e, "y", c.vh, 0)
	if ref.name == "symbol" {
		sst := c.style(ref, st)
		if !sst.display {
			return false
		}
		w, h := c.length(e, "width", c.vw, c.vw), c.length(e, "height", c.vh, c.vh)
		if w <= 0 || h <= 0 {
			return false
		}
		n.kids = c.viewport(ref, &sst, x, y, w, h)
		return len(n.kids) > 0
	}
	kid := c.elem(ref, st)
	if kid == nil {
		return false
	}
	n.m = translate(x, y)
	n.kids = []*node{kid}
	return true
}

func hrefID(e *elem) string {
	v := strings.TrimSpace(e.attrs["href"])
	if !strings.HasPrefix(v, "#") {
		return ""
	}
	return v[1:]
}

func isShape(name string) bool {
	switch name {
	case "path", "rect", "circle", "ellipse", "line", "polyline", "polygon":
		return true
	}
	return false
}

// shapePath returns the path of a shape element.
func (c *compiler) shapePath(e *elem) path {
	var pa path
	vw, vh, diag := c.vw, c.vh, c.diag()
	switch e.name {
	case "path":
		pa = parsePath(e.attrs["d"])
	case "rect":
		x, y := c.length(e, "x", vw, 0), c.length(e, "y", vh, 0)
		w, h := c.length(e, "width", vw, 0), c.length(e, "height", vh, 0)
		if w <= 0 || h <= 0 {
			break
		}
		rx, ry := c.radii(e, vw, vh)
		pa.rect(x, y, w, h, math.Min(rx, w/2), math.Min(ry, h/2))
	case "circle":
		if r := c.length(e, "r", diag, 0); r > 0 {
			pa.ellipse(c.length(e, "cx", vw, 0), c.length(e, "cy", vh, 0), r, r)
		}
	case "ellipse":
		if rx, ry := c.radii(e, vw, vh); rx > 0 && ry > 0 {
			pa.ellipse(c.length(e, "cx", vw, 0), c.length(e, "cy", vh, 0), rx, ry)
		}
	case "line":
		pa.moveTo(point{c.length(e, "x1", vw, 0), c.length(e, "y1", vh, 0)})
		pa.lineTo(point{c.length(e, "x2", vw, 0), c.length(e, "y2", vh, 0)})
	case "polyline", "polygon":
		n := numbers(e.attrs["points"])
		for i := 0; i+1 < len(n); i += 2 {
			if i == 0 {
				pa.moveTo(point{n[i], n[i+1]})
			} else {
				pa.lineTo(point{n[i], n[i+1]})
			}
		}
		if e.name == "polygon" {
			pa.close()
		}
	}
	return pa
}

// radii returns the rx and ry of a rect or ellipse: one that is missing,
// or in error, is the other.
func (c *compiler) radii(e *elem, vw, vh float64) (float64, float64) {
	rx, okX := parseLength(e.attrs["rx"], vw)
	ry, okY := parseLength(e.attrs["ry"], vh)
	okX, okY = okX && rx >= 0, okY && ry >= 0
	switch {
	case okX && !okY:
		ry = rx
	case okY && !okX:
		rx = ry
	case !okX && !okY:
		rx, ry = 0, 0
	}
	return rx, ry
}

// paint resolves a fill or stroke of an element of style st.
func (c *compiler) paint(ps paintSpec, st *style) paint {
	switch ps.kind {
	case specColor:
		return paint{kind: paintColor, color: ps.color}
	case specCurrent:
		if st.colorSet {
			return paint{kind: paintColor, color: st.color}
		}
		c.current = true
		return paint{kind: paintCurrent}
	case specURL:
		if el := c.ids[ps.id]; el != nil && (el.name == "linearGradient" || el.name == "radialGradient") {
			g := c.gradient(el)
			if g == nil || len(g.stops) == 0 {
				return paint{}
			}
			return paint{kind: paintGradient, grad: g}
		}
		// What the reference falls back to, or nothing.
		return c.paint(paintSpec{kind: ps.fallback, color: ps.color}, st)
	}
	return paint{}
}

// gradient compiles a gradient element, with what it inherits from the
// gradients it refers to.
func (c *compiler) gradient(e *elem) *gradient {
	if g, ok := c.grads[e]; ok {
		return g
	}
	c.grads[e] = nil
	chain := []*elem{e}
	for cur := e; len(chain) < 32; {
		ref := c.ids[hrefID(cur)]
		if ref == nil || ref.name != "linearGradient" && ref.name != "radialGradient" || slices.Contains(chain, ref) {
			break
		}
		chain = append(chain, ref)
		cur = ref
	}
	g := &gradient{radial: e.name == "radialGradient", bbox: true, m: identity}
	// attr finds an attribute along the chain; the coordinates come from
	// gradients of the same kind only.
	attr := func(name string, own bool) (string, bool) {
		for _, el := range chain {
			if own && el.name != e.name {
				continue
			}
			if v, ok := el.attrs[name]; ok {
				return v, true
			}
		}
		return "", false
	}
	if v, ok := attr("gradientUnits", false); ok {
		g.bbox = strings.TrimSpace(v) != "userSpaceOnUse"
	}
	if v, ok := attr("gradientTransform", false); ok {
		g.m = parseTransform(v)
	}
	if v, ok := attr("spreadMethod", false); ok {
		switch strings.TrimSpace(v) {
		case "reflect":
			g.spread = spreadReflect
		case "repeat":
			g.spread = spreadRepeat
		}
	}
	coord := func(name, def string, base float64) float64 {
		v, ok := attr(name, true)
		if !ok {
			v = def
		}
		if g.bbox {
			base = 1
		}
		l, ok := parseLength(v, base)
		if !ok {
			l, _ = parseLength(def, base)
		}
		return l
	}
	diag := c.diag()
	if g.radial {
		g.cx, g.cy, g.r = coord("cx", "50%", c.vw), coord("cy", "50%", c.vh), coord("r", "50%", diag)
		g.fx, g.fy = g.cx, g.cy
		if _, ok := attr("fx", true); ok {
			g.fx = coord("fx", "50%", c.vw)
		}
		if _, ok := attr("fy", true); ok {
			g.fy = coord("fy", "50%", c.vh)
		}
		g.fr = coord("fr", "0%", diag)
		// SVG 1.1 moves a focal point outside the circle onto it.
		if d := math.Hypot(g.fx-g.cx, g.fy-g.cy); d > g.r*0.999 && d > 0 {
			k := g.r * 0.999 / d
			g.fx, g.fy = g.cx+(g.fx-g.cx)*k, g.cy+(g.fy-g.cy)*k
		}
	} else {
		g.x1, g.y1 = coord("x1", "0%", c.vw), coord("y1", "0%", c.vh)
		g.x2, g.y2 = coord("x2", "100%", c.vw), coord("y2", "0%", c.vh)
	}
	// The stops of the first gradient along the chain that has some.
	for _, el := range chain {
		if g.stops = c.stops(el); len(g.stops) > 0 {
			break
		}
	}
	c.grads[e] = g
	return g
}

func (c *compiler) stops(e *elem) []stop {
	var out []stop
	last := 0.0
	for _, k := range e.kids {
		if k.name != "stop" {
			continue
		}
		st := c.domStyle(k)
		off := 0.0
		if v, ok := k.attrs["offset"]; ok {
			off = parseOpacity(strings.TrimSpace(v), 0)
		}
		off = math.Max(off, last)
		last = off
		s := stop{offset: off, color: st.stopColor.color, opacity: st.stopOpacity}
		if st.stopColor.kind == specCurrent {
			if st.colorSet {
				s.color = st.color
			} else {
				s.current, c.current = true, true
			}
		}
		out = append(out, s)
	}
	return out
}

// clipRef compiles the clipPath an element refers to, if it is one.
func (c *compiler) clipRef(id string) *clipDef {
	el := c.ids[id]
	if id == "" || el == nil || el.name != "clipPath" {
		return nil
	}
	if cd, ok := c.clips[el]; ok {
		return cd // nil while it is being compiled: it clips itself
	}
	c.clips[el] = nil
	cd := &clipDef{m: parseTransform(el.attrs["transform"]), bbox: strings.TrimSpace(el.attrs["clipPathUnits"]) == "objectBoundingBox"}
	st := c.domStyle(el)
	for _, k := range el.kids {
		if n := c.clipKid(k, st, 0); n != nil {
			cd.kids = append(cd.kids, n)
		}
	}
	cd.clip = c.clipRef(st.clipPath)
	c.clips[el] = cd
	return cd
}

// clipKid compiles a child of a clipPath: a shape, or a use of one. Only
// their geometry clips, filled with the clip-rule.
func (c *compiler) clipKid(k *elem, parent *style, depth int) *node {
	if c.budget <= 0 || depth > 1 {
		return nil
	}
	c.budget--
	st := c.style(k, parent)
	if !st.display || !st.visible {
		return nil
	}
	n := &node{m: parseTransform(k.attrs["transform"])}
	switch {
	case k.name == "use":
		ref := c.ids[hrefID(k)]
		if ref == nil || !isShape(ref.name) {
			return nil
		}
		kid := c.clipKid(ref, &st, depth+1)
		if kid == nil {
			return nil
		}
		n.m = n.m.mul(translate(c.length(k, "x", c.vw, 0), c.length(k, "y", c.vh, 0)))
		n.kids = []*node{kid}
	case isShape(k.name):
		pa := c.shapePath(k)
		if len(pa.ops) == 0 {
			return nil
		}
		n.shape = &shape{path: pa, evenOdd: st.clipEvenOdd}
	default:
		return nil
	}
	return n
}

// maskRef compiles the mask an element refers to, if it is one.
func (c *compiler) maskRef(id string) *maskDef {
	el := c.ids[id]
	if id == "" || el == nil || el.name != "mask" {
		return nil
	}
	if md, ok := c.masks[el]; ok {
		return md
	}
	c.masks[el] = nil
	md := &maskDef{
		units:   strings.TrimSpace(el.attrs["maskUnits"]) != "userSpaceOnUse",
		content: strings.TrimSpace(el.attrs["maskContentUnits"]) == "objectBoundingBox",
	}
	bw, bh := c.vw, c.vh
	if md.units {
		bw, bh = 1, 1
	}
	md.region = box{
		c.length(el, "x", bw, -0.1*bw), c.length(el, "y", bh, -0.1*bh),
		c.length(el, "width", bw, 1.2*bw), c.length(el, "height", bh, 1.2*bh),
	}
	st := c.domStyle(el)
	md.alpha = st.maskAlpha
	for _, k := range el.kids {
		if n := c.elem(k, st); n != nil {
			md.kids = append(md.kids, n)
		}
	}
	c.masks[el] = md
	return md
}
