package ui

import "slices"

// treeBuild is a tree being built: the depth of the items, the item each
// level is in, and the items built so far, in order.
type treeBuild struct {
	depth   int
	parents []uint64
	items   []uint64
	// last is the order of the items in the last frame, for the arrows.
	last []uint64
}

// Tree creates a tree, whose items TreeItem builds in fn. While an item
// has the keyboard focus, Up and Down move it to the items around, Right
// opens the item or moves to its first child, and Left closes it or moves
// to its parent.
//
//	ui.Tree(c, func() {
//		ui.TreeItem(c, "src", &app.srcOpen, func() {
//			ui.TreeItem(c, "main.go", nil, nil)
//		})
//		ui.TreeItem(c, "go.mod", nil, nil)
//	})
func coreTree(c *context, fn func()) *node {
	tree := coreColumn(c).Role(RoleTree)
	tree.widget = "Tree"
	last := coreLocal(tree, "items", func() []uint64 { return nil })
	saved := c.tree
	tb := &treeBuild{last: *last}
	c.tree = tb
	tree.Children(fn)
	c.tree = saved
	*last = tb.items
	return tree
}

// TreeItem creates an item of a Tree showing label, inside the item it is
// built in. children builds the items inside it, which show while *open
// is true; nil children, or a nil open, makes a leaf. A click on its arrow
// opens or closes it; Clicked reports a click elsewhere on it, or Enter,
// for choosing it, which Selected shows.
func coreTreeItem(c *context, label string, open *bool, children func()) *node {
	t := c.theme
	tb := c.tree
	if tb == nil {
		tb = &treeBuild{}
	}
	parent := uint64(0)
	if n := len(tb.parents); n > 0 {
		parent = tb.parents[n-1]
	}
	branch := children != nil && open != nil
	item := coreRow(c).Height(t.Space(7)).Gap(t.Space(0.5)).AlignItems(Center).Padding(0, t.Space(2)).PaddingStart(t.Space(1 + 4*float32(tb.depth))).Focusable().Shrink(0).Role(RoleTreeItem)
	item.widget = "TreeItem"
	item.flags |= flagClickable | flagHover | flagOwnRing
	tb.items = append(tb.items, item.id)
	rt := c.rt
	focus := func(id uint64) {
		if id != 0 && rt.states[id] != nil {
			rt.focused, rt.focusVisible = id, true
			c.rt.consumed = true
		}
	}
	at := slices.Index(tb.last, item.id)
	// Right opens, Left closes, the other way right to left.
	back, forth := item.arrows()
	switch {
	case item.Shortcut(0, KeyDown):
		if at >= 0 && at+1 < len(tb.last) {
			focus(tb.last[at+1])
		}
	case item.Shortcut(0, KeyUp):
		if at > 0 {
			focus(tb.last[at-1])
		}
	case item.Shortcut(0, forth):
		if branch && !*open {
			*open = true
			c.rt.consumed = true
		} else if branch && at >= 0 && at+1 < len(tb.last) {
			focus(tb.last[at+1])
		}
	case item.Shortcut(0, back):
		if branch && *open {
			*open = false
			c.rt.consumed = true
		} else {
			focus(parent)
		}
	}
	if s := item.st; branch && s.expand != 0 {
		// Assistive technology opening or closing it.
		*open = s.expand > 0
		s.expand = 0
		c.rt.consumed = true
	}
	expanded := branch && *open
	item.expanded, item.expandable, item.level = expanded, branch, tb.depth+1
	item.styleFn = func(item *node) {
		if item.checked != 2 && item.Hovered() {
			item.bg = t.SurfaceHover
		}
	}
	item.DrawOver(func(p *Painter, r Rect) {
		if item.FocusVisible() {
			p.FocusRing(r, [4]float32{t.Radius, t.Radius, t.Radius, t.Radius})
		}
	})
	item.Radius(t.Radius)
	item.Children(func() {
		arrow := coreBox(c).Size(t.Space(4), t.Space(4)).Shrink(0).Role(RoleNone)
		if branch {
			arrow.flags |= flagClickable
			if arrow.Clicked() {
				*open = !*open
				c.rt.consumed = true
			}
			rtl := arrow.rtl()
			arrow.Draw(func(p *Painter, r Rect) {
				var path Path
				switch {
				case expanded:
					path.MoveTo(r.X+r.W/4, r.Y+r.H*3/8).LineTo(r.X+r.W/2, r.Y+r.H*5/8).LineTo(r.X+r.W*3/4, r.Y+r.H*3/8)
				case rtl:
					path.MoveTo(r.X+r.W*5/8, r.Y+r.H/4).LineTo(r.X+r.W*3/8, r.Y+r.H/2).LineTo(r.X+r.W*5/8, r.Y+r.H*3/4)
				default:
					path.MoveTo(r.X+r.W*3/8, r.Y+r.H/4).LineTo(r.X+r.W*5/8, r.Y+r.H/2).LineTo(r.X+r.W*3/8, r.Y+r.H*3/4)
				}
				p.StrokePath(&path, 1.5, arrowColor(t, item))
			})
		}
		coreText(c, label).SingleLine()
	})
	if expanded {
		tb.parents = append(tb.parents, item.id)
		tb.depth++
		children()
		tb.depth--
		tb.parents = tb.parents[:len(tb.parents)-1]
	}
	return item
}
