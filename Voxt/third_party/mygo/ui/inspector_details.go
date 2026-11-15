//go:build !mygo_noinspector

package ui

import (
	"fmt"
	"hash/maphash"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// inspDetails describes the element chosen in the inspector, for the
// sidebar of its Elements tab: its styles as CSS rules, its box model and
// computed values, and its properties.
type inspDetails struct {
	name   string
	id     uint64
	source string
	// rules are the element's own styles, then those it inherits, by the
	// element setting them.
	rules []inspRule
	// margin, border and padding are the sides of the box model (top,
	// right, bottom, left), w and h the content's size; margin's sides may
	// be Auto.
	margin, border, padding [4]float32
	w, h                    float32
	computed                []inspDecl
	props                   []inspSection
}

// inspRule is a rule of the Styles tab: the element's own, or those of an
// element it inherits from.
type inspRule struct {
	selector  string
	inherited bool
	decls     []inspDecl
}

// inspDecl is a declaration: a property and its value, with a color
// swatch before the value when swatch is set.
type inspDecl struct {
	name, value string
	color       Color
	swatch      bool
}

// inspSection is a section of the Properties tab.
type inspSection struct {
	title string
	rows  []inspDecl
}

func (d *inspDetails) hash() uint64 {
	if d.name == "" {
		return 0
	}
	var b strings.Builder
	b.WriteString(d.name)
	b.WriteString(d.source)
	for _, r := range d.rules {
		b.WriteString(r.selector)
		for _, dc := range r.decls {
			b.WriteString(dc.name)
			b.WriteString(dc.value)
		}
	}
	for _, dc := range d.computed {
		b.WriteString(dc.value)
	}
	for _, s := range d.props {
		for _, dc := range s.rows {
			b.WriteString(dc.value)
		}
	}
	return maphash.String(keySeed, b.String())
}

// describe returns the details of e.
func describe(rt *engine, e *node) inspDetails {
	d := inspDetails{name: elementName(e), id: e.id, source: rt.insp.source}
	sel := d.name
	if k, ok := rt.insp.keys[e.id]; ok {
		sel += "#" + fmt.Sprint(k)
	}
	d.rules = append(d.rules, inspRule{selector: sel, decls: ownStyles(rt, e)})
	// What it inherits: the text styles that the elements around it set
	// and nothing nearer overrides.
	taken := e.ts.set
	for p := e.parent; p != nil && taken != setAll; p = p.parent {
		set := p.ts.set &^ taken
		if set == 0 {
			continue
		}
		taken |= set
		d.rules = append(d.rules, inspRule{selector: elementName(p), inherited: true, decls: textStyles(&p.ts, set)})
	}
	d.margin, d.border, d.padding = e.margin, e.border, e.pad
	d.w, d.h = max(e.w-e.padX(), 0), max(e.h-e.padY(), 0)
	d.computed = computed(e)
	d.props = properties(rt, e)
	return d
}

// ownStyles returns the styles e sets, as CSS declarations.
func ownStyles(rt *engine, e *node) []inspDecl {
	var ds []inspDecl
	add := func(name, value string) { ds = append(ds, inspDecl{name: name, value: value}) }
	addColor := func(name string, c Color) {
		ds = append(ds, inspDecl{name: name, value: colorText(c), color: c, swatch: true})
	}
	if e.kind == kindBox {
		switch {
		case e.grid:
			add("display", "grid")
			if len(e.cols) > 0 {
				add("grid-template-columns", fmt.Sprintf("%d tracks", len(e.cols)))
			}
			if len(e.rows) > 0 {
				add("grid-template-rows", fmt.Sprintf("%d tracks", len(e.rows)))
			}
		case e.row:
			if e.reverse {
				add("flex-direction", "row-reverse")
			} else {
				add("flex-direction", "row")
			}
		case e.reverse:
			add("flex-direction", "column-reverse")
		}
		switch {
		case e.wrapReverse:
			add("flex-wrap", "wrap-reverse")
		case e.wrap:
			add("flex-wrap", "wrap")
		}
		switch {
		case e.gapX == e.gapY && e.gapX != 0:
			add("gap", pxText(e.gapX))
		case e.gapX != e.gapY:
			add("row-gap", pxText(e.gapY))
			add("column-gap", pxText(e.gapX))
		}
		if e.justify != alignAuto {
			add("justify-content", cssAlign(e.justify))
		}
		if e.align != alignAuto {
			add("align-items", cssAlign(e.align))
		}
		if e.alignContent != alignAuto {
			add("align-content", cssAlign(e.alignContent))
		}
	}
	if e.self != alignAuto {
		add("align-self", cssAlign(e.self))
	}
	if e.dir&dirSet != 0 {
		add("direction", Direction(e.dir&dirRTL).String())
	}
	switch {
	case e.attach != 0:
		at, self := e.attach.anchors()
		add("position", "absolute")
		add("attach", anchorText(at)+" / "+anchorText(self))
	case e.flags&flagAbsolute != 0:
		add("position", "absolute")
	case e.inset != [4]length{}:
		add("position", "relative")
	}
	for i, side := range [4]string{"top", "right", "bottom", "left"} {
		if e.inset[i].u != unitAuto {
			add(side, cssLength(e.inset[i]))
		}
	}
	for _, l := range []struct {
		name string
		v    length
	}{{"width", e.width}, {"height", e.height}, {"min-width", e.minW}, {"min-height", e.minH}, {"max-width", e.maxW}, {"max-height", e.maxH}} {
		if l.v.u != unitAuto {
			add(l.name, cssLength(l.v))
		}
	}
	if e.aspect > 0 {
		add("aspect-ratio", num(e.aspect))
	}
	if e.grow != 0 {
		add("flex-grow", num(e.grow))
	}
	if e.shrink != 1 {
		add("flex-shrink", num(e.shrink))
	}
	if e.basis.u != unitAuto {
		add("flex-basis", cssLength(e.basis))
	}
	if v, ok := edgesCSS(e.pad); ok {
		add("padding", v)
	}
	if v, ok := edgesCSS(e.margin); ok {
		add("margin", v)
	}
	if v, ok := edgesCSS(e.barInset); ok {
		add("scrollbar-insets", v)
	}
	if v, ok := edgesCSS(e.border); ok {
		style := "solid"
		if e.borderStyle == BorderDashed {
			style = "dashed"
		}
		if e.grads != nil && e.grads.border != borderColor {
			add("border", v+" "+style)
			add("border-image", "gradient")
		} else {
			ds = append(ds, inspDecl{name: "border", value: v + " " + style + " " + colorText(e.borderC), color: e.borderC, swatch: true})
		}
	}
	if e.blur > 0 {
		add("filter", "blur("+num(e.blur)+"px)")
	}
	if v, ok := edgesCSS(e.radius); ok {
		add("border-radius", v)
	}
	switch {
	case e.flags&(flagScrollX|flagScrollY) == flagScrollX|flagScrollY:
		add("overflow", "scroll")
	case e.flags&flagScrollX != 0:
		add("overflow-x", "scroll")
	case e.flags&flagScrollY != 0:
		add("overflow-y", "scroll")
	case e.flags&flagClip == flagClip:
		add("overflow", "clip")
	case e.flags&flagClipX != 0:
		add("overflow-x", "clip")
	case e.flags&flagClipY != 0:
		add("overflow-y", "clip")
	}
	if e.kind == kindBox || e.kind == kindText && e.bg.A > 0 {
		switch {
		case e.fill == fillGradient && len(e.grad.Stops) > 0:
			add("background", fmt.Sprintf("linear-gradient(%sdeg, %d stops)", num(e.grad.Angle), len(e.grad.Stops)))
		case e.fill == fillGradient:
			g := e.grad
			add("background", fmt.Sprintf("linear-gradient(%sdeg, %s, %s)", num(g.Angle), colorText(g.From), colorText(g.To)))
		case e.fill == fillConic:
			g := e.grads.conic
			add("background", fmt.Sprintf("conic-gradient(from %sdeg, %d stops)", num(g.Angle), len(g.Stops)))
		case e.fill == fillStripes:
			add("background", "stripes "+colorText(e.stripes.c))
		case e.fill == fillMaterial:
			// A material describes itself as a fmt.Stringer.
			v := "material"
			if m, ok := e.material.(fmt.Stringer); ok {
				v = m.String()
			}
			add("background", v)
		case e.bg.A > 0:
			addColor("background-color", e.bg)
		}
	}
	for _, sh := range e.shadows {
		inset := ""
		if sh.inset {
			inset = "inset "
		}
		add("box-shadow", fmt.Sprintf("%s%s %s %s %s %s", inset, pxText(sh.x), pxText(sh.y), pxText(sh.blur), pxText(sh.spread), colorText(sh.color)))
	}
	if e.opacitySet && e.opacity < 1 {
		add("opacity", num(e.opacity))
	}
	if e.flags&flagInvisible != 0 {
		add("visibility", "hidden")
	}
	if e.flags&flagPassThrough != 0 {
		add("pointer-events", "none")
	}
	if e.cursor != 0 {
		add("cursor", cursorText(e.cursor-1))
	}
	if e.flags&flagDividers != 0 {
		for _, d := range e.c.dividers {
			if d.e == e {
				add("dividers", pxText(d.width)+" "+colorText(d.color))
			}
		}
	}
	if r := e.st.trec; r != nil && r.built == rt.frame && r.elem == e {
		var what []string
		p := r.properties()
		if p&propPosition != 0 {
			what = append(what, "position")
		}
		if p&propSize != 0 {
			what = append(what, "size")
		}
		if p&propColors != 0 {
			what = append(what, "colors")
		}
		add("transition", strings.Join(what, " ")+" "+strconv.Itoa(int(r.duration().Milliseconds()))+"ms")
	}
	ds = append(ds, textStyles(&e.ts, e.ts.set)...)
	if e.kind == kindText {
		if e.maxLines > 0 {
			add("line-clamp", strconv.Itoa(e.maxLines))
		}
		if e.noWrap || e.single {
			add("white-space", "nowrap")
		}
		if e.ellipsis != "" {
			add("text-overflow", strconv.Quote(e.ellipsis))
		}
	}
	return ds
}

// textStyles returns the text styles of ts in set, as CSS declarations.
func textStyles(ts *textStyle, set uint32) []inspDecl {
	var ds []inspDecl
	add := func(name, value string) { ds = append(ds, inspDecl{name: name, value: value}) }
	if set&setFamily != 0 {
		family := ts.family
		if family == "" {
			family = "system-ui"
		}
		add("font-family", family)
	}
	if set&setSize != 0 {
		add("font-size", pxText(ts.size))
	}
	if set&setWeight != 0 {
		add("font-weight", strconv.Itoa(ts.weight))
	}
	if set&setItalic != 0 && ts.italic {
		add("font-style", "italic")
	}
	if set&setColor != 0 {
		ds = append(ds, inspDecl{name: "color", value: colorText(ts.color), color: ts.color, swatch: true})
	}
	if set&setLineHeight != 0 {
		if ts.fixedLine {
			add("line-height", pxText(ts.lineHeight))
		} else {
			add("line-height", num(ts.lineHeight))
		}
	}
	if set&setAlign != 0 {
		add("text-align", map[Align]string{Start: "start", Center: "center", End: "end"}[ts.align])
	}
	if set&setSpacing != 0 {
		add("letter-spacing", pxText(ts.spacing))
	}
	if set&setFeatures != 0 {
		add("font-feature-settings", ts.features)
	}
	if set&setLang != 0 {
		add("lang", ts.lang)
	}
	var deco []string
	if set&setUnderline != 0 && ts.underline {
		if ts.wavy {
			deco = append(deco, "underline wavy")
		} else {
			deco = append(deco, "underline")
		}
	}
	if set&setStrike != 0 && ts.strike {
		deco = append(deco, "line-through")
	}
	if len(deco) > 0 {
		add("text-decoration", strings.Join(deco, " "))
	}
	if set&setBackground != 0 {
		ds = append(ds, inspDecl{name: "background-color", value: colorText(ts.background), color: ts.background, swatch: true})
	}
	if set&setSelection != 0 {
		ds = append(ds, inspDecl{name: "selection-color", value: colorText(ts.selection), color: ts.selection, swatch: true})
	}
	return ds
}

// computed returns the values e has as laid out, by name.
func computed(e *node) []inspDecl {
	var ds []inspDecl
	add := func(name, value string) { ds = append(ds, inspDecl{name: name, value: value}) }
	display := "flex"
	switch {
	case e.grid:
		display = "grid"
	case e.kind == kindText:
		display = "text"
	case e.kind == kindImage, e.kind == kindIcon:
		display = "image"
	}
	add("display", display)
	add("direction", Direction(e.dir&dirRTL).String())
	if e.kind == kindBox && !e.grid {
		dir := "column"
		if e.row {
			dir = "row"
		}
		add("flex-direction", dir)
	}
	add("width", pxText(e.w))
	add("height", pxText(e.h))
	add("left", pxText(e.x))
	add("top", pxText(e.y))
	ts := e.resolvedText()
	family := ts.family
	if family == "" {
		family = "system-ui"
	}
	add("font-family", family)
	add("font-size", pxText(ts.size))
	weight := ts.weight
	if weight == 0 {
		weight = 400
	}
	add("font-weight", strconv.Itoa(weight))
	ds = append(ds, inspDecl{name: "color", value: colorText(ts.color), color: ts.color, swatch: true})
	o := float32(1)
	if e.opacitySet {
		o = e.opacity
	}
	add("opacity", num(o))
	if e.scrolls() {
		add("scroll-left", pxText(float32(e.st.scrollX)))
		add("scroll-top", pxText(float32(e.st.scrollY)))
		add("scroll-width", pxText(float32(e.contentW)))
		add("scroll-height", pxText(float32(e.contentH)))
	}
	slices.SortStableFunc(ds, func(a, b inspDecl) int { return strings.Compare(a.name, b.name) })
	return ds
}

// properties returns what e is, its state, and what assistive technology
// sees of it.
func properties(rt *engine, e *node) []inspSection {
	yes := func(b bool) string {
		if b {
			return "true"
		}
		return "false"
	}
	el := inspSection{title: "Element"}
	el.rows = append(el.rows, inspDecl{name: "type", value: elementName(e)}, inspDecl{name: "id", value: fmt.Sprintf("%016x", e.id)})
	if k, ok := rt.insp.keys[e.id]; ok {
		el.rows = append(el.rows, inspDecl{name: "key", value: fmt.Sprintf("%#v", k)})
	}
	if s := rt.insp.source; s != "" {
		el.rows = append(el.rows, inspDecl{name: "built at", value: s})
	}
	if e.kind == kindText && e.text != "" {
		el.rows = append(el.rows, inspDecl{name: "text", value: strconv.Quote(clip(e.text, 200))})
	}
	hovered := false
	for _, id := range rt.hover {
		hovered = hovered || id == e.id
	}
	moving := false
	if r := e.st.trec; r != nil && r.built == rt.frame && r.elem == e {
		moving = r.moving != 0 || r.colorMoving
	}
	state := inspSection{title: "State", rows: []inspDecl{
		{name: "focused", value: yes(rt.focused == e.id)},
		{name: "hovered", value: yes(hovered)},
		{name: "pressed", value: yes(e.st.pressed)},
		{name: "disabled", value: yes(e.IsDisabled())},
		{name: "moving", value: yes(moving)},
	}}
	if e.leaving != 0 {
		state.rows = append(state.rows, inspDecl{name: "leaving", value: "true"})
	}
	role, seen := accessibleRole(e)
	name := e.label
	if name == "" && e.kind == kindText {
		name = clip(e.text, 80)
	}
	acc := inspSection{title: "Accessibility", rows: []inspDecl{
		{name: "role", value: role},
		{name: "name", value: strconv.Quote(name)},
		{name: "keyboard-focusable", value: yes(e.flags&flagFocusable != 0 && !e.IsDisabled())},
		{name: "ignored", value: yes(!seen || e.flags&(flagInvisible|flagInert) != 0)},
	}}
	if e.description != "" {
		acc.rows = append(acc.rows, inspDecl{name: "description", value: strconv.Quote(e.description)})
	}
	return []inspSection{el, state, acc}
}

func cssAlign(a Align) string {
	switch a {
	case Start:
		return "flex-start"
	case Center:
		return "center"
	case End:
		return "flex-end"
	case Stretch:
		return "stretch"
	case SpaceBetween:
		return "space-between"
	case SpaceAround:
		return "space-around"
	case SpaceEvenly:
		return "space-evenly"
	}
	return "auto"
}

func cssLength(l length) string {
	switch l.u {
	case unitPx:
		return pxText(l.v)
	case unitPercent:
		return num(l.v) + "%"
	}
	return "auto"
}

// edgesCSS returns the four sides top, right, bottom and left as CSS
// writes them, and whether one is not zero.
func edgesCSS(v [4]float32) (string, bool) {
	if v == [4]float32{} {
		return "", false
	}
	f := func(x float32) string {
		if isAuto(x) {
			return "auto"
		}
		return pxText(x)
	}
	switch {
	case v[0] == v[1] && v[1] == v[2] && v[2] == v[3]:
		return f(v[0]), true
	case v[0] == v[2] && v[1] == v[3]:
		return f(v[0]) + " " + f(v[1]), true
	}
	return f(v[0]) + " " + f(v[1]) + " " + f(v[2]) + " " + f(v[3]), true
}

func anchorText(a Anchor) string {
	return [...]string{"top left", "top", "top right", "left", "center", "right", "bottom left", "bottom", "bottom right",
		"top start", "start", "bottom start", "top end", "end", "bottom end"}[min(int(a), 14)]
}

func cursorText(c Cursor) string {
	switch c {
	case CursorPointer:
		return "pointer"
	case CursorText:
		return "text"
	case CursorMove:
		return "move"
	case CursorResizeEW:
		return "ew-resize"
	case CursorResizeNS:
		return "ns-resize"
	case CursorNotAllowed:
		return "not-allowed"
	case CursorCrosshair:
		return "crosshair"
	case CursorGrab:
		return "grab"
	case CursorGrabbing:
		return "grabbing"
	case CursorNone:
		return "none"
	}
	return "default"
}

// shortSource returns the file name and line of a source location.
func shortSource(s string) string {
	if s == "" {
		return ""
	}
	return filepath.Base(s)
}
