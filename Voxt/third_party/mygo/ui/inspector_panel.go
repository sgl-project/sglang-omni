//go:build !mygo_noinspector

package ui

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// inspPalette is the inspector's colors, Chrome's developer tools'.
type inspPalette struct {
	bg, toolbar, border, text, muted, accent Color
	// selected is the row chosen while the tree has the focus, unfocused
	// while it does not, hover the row under the pointer.
	selected, unfocused, hover Color
	// tag, attr and value color markup, prop the properties of styles,
	// str and lit the strings and other values of properties.
	tag, attr, value, prop, str, lit Color
	warn                             Color
}

func inspPaletteFor(dark bool) inspPalette {
	if dark {
		return inspPalette{
			bg: Hex("#202124"), toolbar: Hex("#292a2d"), border: Hex("#494c50"), text: Hex("#e8eaed"), muted: Hex("#9aa0a6"),
			accent: Hex("#8ab4f8"), selected: Hex("#0f3a66"), unfocused: Hex("#3c4043"), hover: Hex("#2f3237"),
			tag: Hex("#5db0d7"), attr: Hex("#9bbbdc"), value: Hex("#f29766"), prop: Hex("#5cd5fb"),
			str: Hex("#f28b54"), lit: Hex("#9980ff"), warn: Hex("#fdd663"),
		}
	}
	return inspPalette{
		bg: Hex("#ffffff"), toolbar: Hex("#f1f3f4"), border: Hex("#cbcdd1"), text: Hex("#202124"), muted: Hex("#5f6368"),
		accent: Hex("#1a73e8"), selected: Hex("#cfe8fc"), unfocused: Hex("#e3e3e3"), hover: Hex("#ebf1fb"),
		tag: Hex("#881280"), attr: Hex("#994500"), value: Hex("#1a1aa6"), prop: Hex("#c80000"),
		str: Hex("#c41a16"), lit: Hex("#1a1aa6"), warn: Hex("#e37400"),
	}
}

// inspIcons are the inspector's icons, parsed once a window opens it.
var inspIcons = sync.OnceValue(func() (icons struct{ pick, close, warn, arrow *SVG }) {
	stroked := func(shapes string) *SVG {
		return MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` + shapes + `</svg>`))
	}
	icons.pick = stroked(`<path d="M21 11V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h6"/><path d="m12 12 4.2 10 1.6-4.4L22 16z"/>`)
	icons.close = stroked(`<path d="M18 6 6 18M6 6l12 12"/>`)
	icons.warn = stroked(`<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3z"/><path d="M12 9v4M12 17h.01"/>`)
	icons.arrow = MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M3 1.5 7.5 5 3 8.5z" fill="currentColor"/></svg>`))
	return icons
})

// buildInspector builds the panel, right of the content appW wide, in a
// window w×h.
func (rt *engine) buildInspector(c *context, appW, w, h float32) {
	in := &rt.insp
	in.shown, in.changed = in.sum, false
	if in.opened == nil {
		in.opened = map[uint64]bool{}
	}
	// The display's refresh rate, which the window may move to another.
	if in.hz = rt.host.refreshRate(); in.hz <= 0 {
		in.hz = 60
	}
	// The panel looks as Chrome's tools do, whatever theme the view set.
	in.pal = inspPaletteFor(rt.dark)
	pal := &in.pal
	in.theme = rt.theme
	t := &in.theme
	t.Background, t.Surface, t.SurfaceHover, t.Border = pal.bg, pal.toolbar, pal.hover, pal.border
	t.Text, t.TextMuted, t.FontSize = pal.text, pal.muted, 12
	// Rows chosen span the tree, square, as Chrome's.
	t.Radius = 0
	// The tree's chosen row, in its colors, whether the tree has the focus.
	t.Accent, t.AccentText = pal.unfocused, pal.text
	if rt.focused == in.listID && in.listID != 0 {
		t.Accent = pal.selected
	}
	saved, savedParent := c.theme, c.parent
	c.theme, c.parent = t, c.root
	defer func() { c.theme, c.parent = saved, savedParent }()

	// The panel reads left to right whatever the window, as browsers'
	// developer tools do.
	panel := coreColumn(c).Key(inspectorKey).Direction(LeftToRight)
	in.panel = panel.id
	panel.Absolute().Left(appW).Top(0).Width(w-appW).Height(h).Background(pal.bg).
		BorderWidth(0, 0, 0, 1).BorderColor(pal.border).FontSize(12).TextColor(pal.text).Label("Developer tools")
	panel.Children(func() {
		in.resizer(c)
		in.toolbar(c, rt)
		switch in.tab {
		case inspPerformance:
			in.performance(c)
		case inspIssues:
			in.issues(c, rt)
		default:
			if in.split == 0 {
				in.split = max(h*0.55, 120)
			}
			coreSplitVertical(c, &in.split, func() { in.treePane(c, rt) }, func() { in.sidebar(c) }).Grow(1).MinHeight(0)
		}
	})
	if panel.Shortcut(Cmd, KeyF) {
		in.finding, in.tab = true, inspElements
	}
}

// resizer is the panel's left edge, which the pointer drags to resize it.
func (in *inspector) resizer(c *context) {
	edge := coreBox(c).Absolute().Left(-3).Top(0).Width(6).FillHeight().Cursor(CursorResizeEW)
	if dx, _, ok := edge.Dragged(); ok && dx != 0 {
		in.width = max(in.width-dx, 260)
	}
}

// toolbar builds the bar at the top of the panel: the picker, the tabs
// and the close button.
func (in *inspector) toolbar(c *context, rt *engine) {
	pal := &in.pal
	icons := inspIcons()
	coreRow(c).Height(30).Shrink(0).PaddingX(4).AlignItems(Center).Background(pal.toolbar).
		BorderWidth(0, 0, 1, 0).BorderColor(pal.border).Children(func() {
		pick := in.iconButton(c, icons.pick, in.picking, "Select an element in the window to inspect it")
		if pick.Clicked() {
			in.picking, in.hovered, in.tab = !in.picking, 0, inspElements
		}
		coreBox(c).Width(1).Height(16).MarginX(4).Background(pal.border)
		issues := "Issues"
		if n := len(rt.warnings); n > 0 {
			issues = fmt.Sprintf("Issues %d", n)
		}
		for i, label := range []string{"Elements", "Performance", issues} {
			if in.tab0(c, label, in.tab == i, 30).Clicked() {
				in.tab = i
			}
		}
		coreSpacer(c)
		// Lays the window out right to left, or back, to try it without
		// translations, as GTK's inspector can.
		rtl := rt.direction()
		dir := in.tab0(c, "RTL", rtl, 30).Role(RoleToggleButton).Tooltip("Lay the window out right to left")
		if dir.Clicked() {
			rt.dirOverride = 2
			if rtl {
				rt.dirOverride = 1
			}
			rt.requestFrame()
		}
		if n := len(rt.warnings); n > 0 {
			badge := coreRow(c).Gap(3).PaddingX(6).AlignItems(Center).Cursor(CursorPointer).Label(fmt.Sprintf("%d issues", n))
			badge.Children(func() {
				coreIcon(c, icons.warn).FontSize(13).TextColor(pal.warn)
				coreText(c, fmt.Sprint(n)).FontSize(11).TextColor(pal.muted)
			})
			if badge.Clicked() {
				in.tab = inspIssues
			}
		}
		if in.iconButton(c, icons.close, false, "Close").Clicked() {
			rt.toggleInspector()
		}
	})
}

// iconButton builds a button of the toolbar showing icon, in the accent
// color while on. It takes no focus, so as not to take it from the
// content.
func (in *inspector) iconButton(c *context, icon *SVG, on bool, label string) *node {
	pal := &in.pal
	b := coreBox(c).Size(26, 24).Radius(4).Center().Label(label).Role(RoleButton).Tooltip(label)
	color := pal.muted
	switch {
	case on:
		color = pal.accent
	case b.Hovered():
		color = pal.text
		b.Background(pal.hover)
	}
	b.Children(func() { coreIcon(c, icon).FontSize(16).TextColor(color) })
	return b
}

// tab0 builds a tab of a bar height high, underlined in the accent color
// while selected.
func (in *inspector) tab0(c *context, label string, selected bool, height float32) *node {
	pal := &in.pal
	t := coreBox(c).Height(height).PaddingX(10).Center().Label(label).Role(RoleTab)
	t.checked = 1
	color := pal.muted
	switch {
	case selected:
		t.checked = 2
		color = pal.text
		t.BorderWidth(0, 0, 2, 0).BorderColor(pal.accent)
	case t.Hovered():
		color = pal.text
		t.Background(pal.hover)
	}
	t.Children(func() { coreText(c, label).TextColor(color).SingleLine() })
	return t
}

// treePane builds the tree of elements, with its find bar and the
// breadcrumbs of the element chosen.
func (in *inspector) treePane(c *context, rt *engine) {
	coreColumn(c).Fill().Children(func() {
		in.find(c)
		in.tree(c, rt)
		in.breadcrumbs(c)
	})
}

// treeRows lists the rows the tree shows.
func (in *inspector) treeRows() {
	in.rows = in.rows[:0]
	var open []int32 // the nodes open around the node, innermost last
	hide := int32(-1)
	for i := range in.nodes {
		n := &in.nodes[i]
		for len(open) > 0 && in.nodes[open[len(open)-1]].depth >= n.depth {
			in.rows = append(in.rows, inspRow{node: open[len(open)-1], close: true})
			open = open[:len(open)-1]
		}
		if hide >= 0 && n.depth > hide {
			continue
		}
		hide = -1
		in.rows = append(in.rows, inspRow{node: int32(i)})
		if n.kids {
			if in.isOpen(n) {
				open = append(open, int32(i))
			} else {
				hide = n.depth
			}
		}
	}
	for i := len(open) - 1; i >= 0; i-- {
		in.rows = append(in.rows, inspRow{node: open[i], close: true})
	}
}

// tree builds the tree of elements as markup, Chrome's Elements panel.
func (in *inspector) tree(c *context, rt *engine) {
	in.treeRows()
	if in.reveal && in.selected != 0 {
		in.reveal = false
		if in.expandTo(in.selected) {
			in.treeRows()
		}
		if r := in.rowOf(in.selected); r >= 0 {
			in.list.ScrollIntoView(r)
		}
	}
	in.row = in.rowOf(in.selected)
	in.list.Selected = &in.row
	in.list.Key = func(i int) any {
		r := in.rows[i]
		if r.close {
			return ^in.nodes[r.node].key
		}
		return in.nodes[r.node].key
	}
	in.list.Label = func(i int) string {
		n := &in.nodes[in.rows[i].node]
		return strings.TrimSpace(n.name + " " + clip(n.text, 40))
	}
	hovered := uint64(0)
	list := coreList(c, &in.list, len(in.rows), func(i int) {
		r := in.rows[i]
		n := &in.nodes[r.node]
		row := coreRow(c).Height(20).Padding(0, 6, 0, float32(n.depth)*14+4).AlignItems(Center)
		if row.Hovered() {
			hovered = n.id
		}
		if n.leaving {
			row.Opacity(0.5)
		}
		row.Children(func() { in.treeRow(c, r, n) })
	}).Grow(1).MinHeight(0).PaddingY(2).FocusRing(false).Label("Elements")
	in.listID = list.id
	list.afterInput(func() {
		if list.Changed() && in.row >= 0 && in.row < len(in.rows) {
			in.choose(in.nodes[in.rows[in.row].node].id)
		}
	})
	// Left closes the node chosen, else goes to the node around it; Right
	// opens it, else goes into it.
	left, right := list.Shortcut(0, KeyLeft), list.Shortcut(0, KeyRight)
	if i := in.nodeOf(in.selected); i >= 0 && (left || right) {
		n := &in.nodes[i]
		switch {
		case left && n.kids && in.isOpen(n):
			in.setOpen(i, false, false)
		case left && n.parent >= 0:
			in.choose(in.nodes[n.parent].id)
			in.reveal = true
		case right && n.kids && !in.isOpen(n):
			in.setOpen(i, true, false)
		case right && n.kids:
			in.choose(in.nodes[i+1].id)
			in.reveal = true
		}
	}
	if !in.picking {
		in.hovered = hovered
	}
}

// rowOf returns the row of the opening tag of the node id, -1 for none.
func (in *inspector) rowOf(id uint64) int {
	for i, r := range in.rows {
		if !r.close && in.nodes[r.node].id == id {
			return i
		}
	}
	return -1
}

// treeRow builds the markup of row r, of node n.
func (in *inspector) treeRow(c *context, r inspRow, n *inspNode) {
	pal := &in.pal
	if !r.close && n.kids {
		open := in.isOpen(n)
		arrow := coreIcon(c, inspIcons().arrow).FontSize(10).TextColor(pal.muted).Margin(0, 2, 0, 0).Cursor(CursorDefault)
		if open {
			arrow.Rotate(90)
		}
		if arrow.Clicked() {
			// Alt opens or closes everything inside, as in Chrome.
			in.setOpen(r.node, !open, arrow.ClickModifiers()&Alt != 0)
		}
	} else {
		coreBox(c).Width(12).Shrink(0)
	}
	tag := func(s string) Span { return Span{Text: s, Color: pal.tag} }
	var spans []Span
	switch {
	case r.close:
		spans = append(spans, tag("</"+n.name+">"))
	default:
		spans = append(spans, tag("<"+n.name))
		attr := func(name, value string) {
			spans = append(spans, Span{Text: " " + name, Color: pal.attr}, tag("="), Span{Text: `"` + clip(value, 40) + `"`, Color: pal.value})
		}
		if n.keyText != "" {
			attr("key", n.keyText)
		}
		if n.label != "" && n.label != n.text {
			attr("label", n.label)
		}
		switch {
		case n.kids && in.isOpen(n):
			spans = append(spans, tag(">"))
		case n.kids:
			spans = append(spans, tag(">"), Span{Text: "…", Color: pal.muted}, tag("</"+n.name+">"))
		case n.text != "":
			spans = append(spans, tag(">"), Span{Text: clip(n.text, 80), Color: pal.text}, tag("</"+n.name+">"))
		default:
			spans = append(spans, tag("></"+n.name+">"))
		}
	}
	coreRichText(c, spans...).Font("monospace").FontSize(11.5).SingleLine().Ellipsis("…").Shrink(1).MinWidth(0)
	if !r.close && n.hasTrans {
		coreText(c, "transition").Font("monospace").FontSize(10).TextColor(pal.muted).Padding(0, 4).Margin(0, 0, 0, 6).
			Radius(3).Border(1, pal.border).Shrink(0)
	}
}

// find builds the find bar of the tree (Cmd+F), which finds elements by
// their tag, text, label or key.
func (in *inspector) find(c *context) {
	in.findMatches()
	before := in.query
	bar := coreFindBar(c, &in.finding, &in.query, len(in.matches), &in.match)
	bar.afterInput(func() {
		if in.query != before {
			// Typed: the first match of the new query, which the bar counts in
			// the next pass.
			in.findMatches()
			in.match = 0
			c.rt.consumed = true
		}
		if (bar.Changed() || in.query != before) && len(in.matches) > 0 {
			in.match = max(0, min(in.match, len(in.matches)-1))
			in.choose(in.nodes[in.matches[in.match]].id)
			in.reveal = true
		}
	})
}

// findMatches lists the nodes whose tag, text, label or key holds the
// query, whatever their case.
func (in *inspector) findMatches() {
	in.matches = in.matches[:0]
	if !in.finding || in.query == "" {
		return
	}
	q := strings.ToLower(in.query)
	for i := range in.nodes {
		n := &in.nodes[i]
		if strings.Contains(strings.ToLower(n.name), q) || strings.Contains(strings.ToLower(n.text), q) ||
			strings.Contains(strings.ToLower(n.label), q) || strings.Contains(strings.ToLower(n.keyText), q) {
			in.matches = append(in.matches, int32(i))
		}
	}
}

// breadcrumbs builds the path of the element chosen, from the root, whose
// elements a click chooses.
func (in *inspector) breadcrumbs(c *context) {
	pal := &in.pal
	var path []int32
	for i := in.nodeOf(in.selected); i >= 0; i = in.nodes[i].parent {
		path = append(path, i)
	}
	if len(path) == 0 {
		return
	}
	coreRow(c).Height(24).Shrink(0).Background(pal.toolbar).BorderWidth(1, 0, 0, 0).BorderColor(pal.border).Children(func() {
		coreScrollHorizontal(c).TrackScroll(&in.crumbs).Grow(1).MinWidth(0).Children(func() {
			coreRow(c).Gap(1).PaddingX(4).FillHeight().AlignItems(Center).Children(func() {
				for k := len(path) - 1; k >= 0; k-- {
					n := &in.nodes[path[k]]
					crumb := coreBox(c).Key(n.key).PaddingX(6).PaddingY(2).Radius(3).Label(n.name)
					switch {
					case k == 0:
						crumb.Background(pal.selected)
					case crumb.Hovered():
						crumb.Background(pal.hover)
						in.hovered = n.id
					}
					crumb.Children(func() { coreText(c, n.name).Font("monospace").FontSize(11).TextColor(pal.tag).SingleLine() })
					if crumb.Clicked() {
						in.choose(n.id)
						in.reveal = true
					}
				}
			})
		})
	})
}

// sidebar builds the tabs below the tree: Styles, Computed and
// Properties of the element chosen.
func (in *inspector) sidebar(c *context) {
	pal := &in.pal
	coreColumn(c).Fill().Children(func() {
		coreRow(c).Height(26).Shrink(0).PaddingX(4).Background(pal.toolbar).BorderWidth(0, 0, 1, 0).BorderColor(pal.border).Children(func() {
			for i, label := range []string{"Styles", "Computed", "Properties"} {
				if in.tab0(c, label, in.side == i, 26).Clicked() {
					in.side = i
				}
			}
		})
		coreScroll(c).Grow(1).MinHeight(0).TrackScroll(&in.sideScroll).Children(func() {
			if in.sel.name == "" {
				coreText(c, "Choose an element in the tree, or pick one in the window.").TextColor(pal.muted).Padding(12)
				return
			}
			switch in.side {
			case inspComputed:
				in.computedPane(c)
			case inspProperties:
				in.propertiesPane(c)
			default:
				in.stylesPane(c)
			}
		})
	})
}

// mono creates a line of monospaced spans, as the panes show declarations.
func mono(c *context, spans ...Span) *node {
	return coreRichText(c, spans...).Font("monospace").FontSize(11.5)
}

// decl builds declaration d: its property in color name, a colon, and its
// value, after a swatch of its color.
func (in *inspector) decl(c *context, d inspDecl, name Color) {
	pal := &in.pal
	if !d.swatch {
		// One text, so that the space after the colon stays.
		mono(c, Span{Text: d.name, Color: name}, Span{Text: ": " + d.value + ";", Color: pal.text}).Selectable()
		return
	}
	coreRow(c).AlignItems(Center).Children(func() {
		mono(c, Span{Text: d.name, Color: name}, Span{Text: ":", Color: pal.text}).Shrink(0)
		coreBox(c).Size(10, 10).Margin(0, 4, 0, 6).Border(1, pal.border).Background(d.color).Shrink(0)
		mono(c, Span{Text: d.value + ";", Color: pal.text}).Shrink(1).MinWidth(0).Selectable()
	})
}

// stylesPane lists the element's styles as CSS rules: its own, then
// those it inherits, as Chrome's Styles tab.
func (in *inspector) stylesPane(c *context) {
	pal := &in.pal
	for i, r := range in.sel.rules {
		if r.inherited {
			coreRow(c).Padding(4, 10).Background(pal.toolbar).Children(func() {
				coreRichText(c, Span{Text: "Inherited from ", Color: pal.muted}, Span{Text: r.selector, Color: pal.tag, Font: "monospace"}).FontSize(11)
			})
		}
		coreColumn(c).Padding(6, 10, 8).Gap(1).BorderWidth(0, 0, 1, 0).BorderColor(pal.border).Children(func() {
			coreRow(c).AlignItems(Center).Children(func() {
				mono(c, Span{Text: r.selector, Color: pal.text}, Span{Text: " {", Color: pal.text})
				coreSpacer(c)
				if i == 0 && in.sel.source != "" {
					coreText(c, shortSource(in.sel.source)).FontSize(11).TextColor(pal.muted).Underline().
						SingleLine().Ellipsis("…").Shrink(1).MinWidth(0).Tooltip(in.sel.source)
				}
			})
			coreColumn(c).Padding(0, 0, 0, 16).Gap(1).Children(func() {
				for _, d := range r.decls {
					in.decl(c, d, pal.prop)
				}
			})
			mono(c, Span{Text: "}", Color: pal.text})
		})
	}
}

// computedPane shows the box model of the element and its computed values,
// as Chrome's Computed tab.
func (in *inspector) computedPane(c *context) {
	pal := &in.pal
	d := &in.sel
	side := func(v float32) string {
		switch {
		case isAuto(v):
			return "auto"
		case v == 0:
			return "-"
		}
		return num(v)
	}
	ink := Hex("#202124")
	var layer func(name string, bg, line Color, dashed bool, sides [4]float32, inner func())
	layer = func(name string, bg, line Color, dashed bool, sides [4]float32, inner func()) {
		b := coreColumn(c).Background(bg).Border(1, line).AlignItems(Stretch).FontSize(11).TextColor(ink)
		if dashed {
			b.BorderStyle(BorderDashed)
		}
		b.Children(func() {
			coreText(c, name).FontSize(10).TextColor(ink.Alpha(0.7)).Attach(AnchorTopLeft, AnchorTopLeft).Left(4).Top(2)
			coreText(c, side(sides[0])).TextAlign(Center).MarginY(3)
			coreRow(c).AlignItems(Center).Children(func() {
				coreText(c, side(sides[3])).Width(30).TextAlign(Center).Shrink(0)
				coreColumn(c).Grow(1).Children(inner)
				coreText(c, side(sides[1])).Width(30).TextAlign(Center).Shrink(0)
			})
			coreText(c, side(sides[2])).TextAlign(Center).MarginY(3)
		})
	}
	coreColumn(c).Padding(12).Children(func() {
		layer("margin", Hex("#f9cc9d"), Hex("#333333"), true, d.margin, func() {
			layer("border", Hex("#fddd9b"), Hex("#000000"), false, d.border, func() {
				layer("padding", Hex("#c3d08b"), Hex("#808080"), true, d.padding, func() {
					coreColumn(c).Background(Hex("#8cb6c0")).Border(1, Hex("#808080")).Padding(6, 4).Center().Children(func() {
						coreText(c, num(d.w)+" × "+num(d.h)).FontSize(11).TextColor(ink).SingleLine()
					})
				})
			})
		})
	})
	coreColumn(c).Padding(0, 10, 10).Gap(1).Children(func() {
		for _, dc := range d.computed {
			in.decl(c, dc, pal.prop)
		}
	})
}

// propertiesPane lists what the element is, its state, and what
// assistive technology sees of it.
func (in *inspector) propertiesPane(c *context) {
	pal := &in.pal
	for _, s := range in.sel.props {
		coreColumn(c).Padding(6, 10, 8).Gap(2).BorderWidth(0, 0, 1, 0).BorderColor(pal.border).Children(func() {
			coreText(c, s.title).Bold().FontSize(11).Margin(0, 0, 2, 0)
			for _, d := range s.rows {
				color := pal.text
				switch {
				case strings.HasPrefix(d.value, `"`):
					color = pal.str
				case d.value == "true", d.value == "false":
					color = pal.lit
				}
				coreRow(c).Padding(0, 0, 0, 8).Children(func() {
					mono(c, Span{Text: d.name, Color: pal.tag}, Span{Text: ": ", Color: pal.text}, Span{Text: d.value, Color: color}).
						Shrink(1).MinWidth(0).Selectable()
				})
			}
		})
	}
}

// The colors of the Performance tab: building as Chrome's scripting,
// laying out as rendering, painting as painting.
var (
	inspBuild  = Hex("#f2b630")
	inspLayout = Hex("#9a7fd1")
	inspPaint  = Hex("#71b362")
)

// performance builds the Performance tab: how long the last frames took to
// build, lay out and paint, against the time between two refreshes of the
// display.
func (in *inspector) performance(c *context) {
	pal := &in.pal
	hz := in.hz
	budget := time.Duration(float64(time.Second) / float64(hz))
	n := min(in.frames, inspHistory)
	var total, slowest time.Duration
	for i := 0; i < n; i++ {
		f := in.history[i]
		d := f[0] + f[1] + f[2]
		total += d
		slowest = max(slowest, d)
	}
	coreScroll(c).Grow(1).MinHeight(0).Padding(12).Gap(12).Children(func() {
		coreRow(c).Gap(16).Wrap().Children(func() {
			stat := func(label, value string) {
				coreColumn(c).Gap(2).Children(func() {
					coreText(c, label).FontSize(11).TextColor(pal.muted)
					coreText(c, value).FontSize(18).FontWeight(600).FontFeatures("tnum")
				})
			}
			last := in.times[0] + in.times[1] + in.times[2]
			stat("Last frame", ms(last)+" ms")
			avg := time.Duration(0)
			if n > 0 {
				avg = total / time.Duration(n)
			}
			stat("Average", ms(avg)+" ms")
			stat("Slowest", ms(slowest)+" ms")
			stat("Display", num(hz)+" Hz")
			stat("Elements", fmt.Sprint(in.elements))
		})
		coreBox(c).Height(150).Shrink(0).Radius(4).Border(1, pal.border).Draw(func(p *Painter, r Rect) { in.paintFrames(p, r) })
		coreRow(c).Gap(14).Wrap().Children(func() {
			for _, l := range []struct {
				name  string
				color Color
				d     time.Duration
			}{{"Build", inspBuild, in.times[0]}, {"Layout", inspLayout, in.times[1]}, {"Paint", inspPaint, in.times[2]}} {
				coreRow(c).Gap(6).AlignItems(Center).Children(func() {
					coreBox(c).Size(10, 10).Radius(2).Background(l.color)
					coreText(c, l.name+" "+ms(l.d)+" ms").FontSize(11).FontFeatures("tnum")
				})
			}
			coreText(c, fmt.Sprintf("— %s ms, a frame at %s Hz", ms(budget), num(hz))).FontSize(11).TextColor(pal.muted)
		})
		coreText(c, "Frames happen only when something changes: input, an update, or motion. "+
			"Set MYGO_FRAME_STATS to log slow frames.").FontSize(11).TextColor(pal.muted)
	})
}

// paintFrames draws the last frames as bars in r, the latest at the right,
// stacked by what they spent their time on.
func (in *inspector) paintFrames(p *Painter, r Rect) {
	pal := &in.pal
	n := min(in.frames, inspHistory)
	budget := float64(time.Second) / float64(in.hz)
	scale := 2 * budget
	for i := 0; i < n; i++ {
		f := in.history[i]
		scale = max(scale, float64(f[0]+f[1]+f[2])*1.1)
	}
	r = Rect{r.X + 4, r.Y + 6, r.W - 8, r.H - 10}
	y := r.Y + r.H - float32(budget/scale)*r.H
	p.Line(r.X, y, r.X+r.W, y, 1, pal.muted.Alpha(0.6))
	bw := r.W / inspHistory
	for k := 0; k < n; k++ {
		f := in.history[(in.frames-n+k)%inspHistory]
		x := r.X + r.W - float32(n-k)*bw
		bottom := r.Y + r.H
		for j, color := range [3]Color{inspBuild, inspLayout, inspPaint} {
			hgt := float32(float64(f[j])/scale) * r.H
			if hgt <= 0 {
				continue
			}
			p.Fill(Rect{x + bw*0.15, bottom - hgt, bw * 0.7, hgt}, color, 0)
			bottom -= hgt
		}
	}
}

// issues builds the Issues tab: the warnings, latest first.
func (in *inspector) issues(c *context, rt *engine) {
	pal := &in.pal
	coreScroll(c).Grow(1).MinHeight(0).TrackScroll(&in.issuesScroll).Children(func() {
		if len(rt.warnings) == 0 {
			coreText(c, "No issues so far.").TextColor(pal.muted).Padding(12)
			return
		}
		for i := len(rt.warnings) - 1; i >= 0; i-- {
			coreRow(c).Padding(8, 10).Gap(8).BorderWidth(0, 0, 1, 0).BorderColor(pal.border).Children(func() {
				coreIcon(c, inspIcons().warn).FontSize(14).TextColor(pal.warn).Shrink(0)
				coreText(c, rt.warnings[i]).FontSize(11.5).Grow(1).MinWidth(0).Selectable()
			})
		}
	})
}
