package ui

// Orientation says which arrows move the focus in a FocusGroup: Left and
// Right (Horizontal), Up and Down (Vertical), or all four
// (Horizontal|Vertical).
type Orientation uint8

const (
	Horizontal Orientation = 1 << iota
	Vertical
)

// FocusGroup makes the elements inside the element that take the focus one
// stop of Tab, as the controls of a toolbar, a radio group or a tab list
// are: Tab moves the focus into the group, to the element of it that had
// it last (else the radio button or tab chosen, else the first), and on
// out of it; the
// arrows that o names move it among them, round from the last to the
// first, and Home and End to the first and the last. An element inside
// that takes those keys itself, as a slider or a text input does, keeps
// them. Groups inside a group are part of it.
func (e *node) FocusGroup(o Orientation) *node {
	e.focusGroup = o
	return e
}

// groupInfo is what the frame committed of a focus group.
type groupInfo struct {
	orient Orientation
	// rtl tells that the group is laid out right to left, where Left
	// moves to the next element.
	rtl bool
	// selects makes moving the focus with the arrows click the element
	// it moves to, as in a radio group.
	selects bool
	// checked is the first element of the group that is checked, 0 for
	// none.
	checked uint64
}

// groupOf returns the focus group of an element of the focus order, 0
// for none.
func (rt *engine) groupOf(id uint64) uint64 { return rt.memberOf[id] }

// groupEntry returns the element Tab moves the focus to in group g: the
// one that had it last, the one checked, or the first.
func (rt *engine) groupEntry(g uint64) uint64 {
	if last := rt.groupLast[g]; last != 0 && rt.memberOf[last] == g {
		return last
	}
	if c := rt.groups[g].checked; c != 0 {
		return c
	}
	for _, id := range rt.focusOrder {
		if rt.memberOf[id] == g {
			return id
		}
	}
	return 0
}

// tabStop reports whether Tab stops at id: an element in no group, or the
// entry of its group.
func (rt *engine) tabStop(id uint64) bool {
	if s := rt.states[id]; s != nil && s.flags&flagFocusTarget != 0 {
		return false
	}
	g := rt.memberOf[id]
	return g == 0 || rt.groupEntry(g) == id
}

// noteGroups records the focus groups of the committed frame, and which
// of their elements had the focus.
func (rt *engine) noteGroups() {
	clear(rt.memberOf)
	if rt.memberOf == nil {
		rt.memberOf = map[uint64]uint64{}
	}
	for i, sc := range rt.focusScopes {
		if sc.group != 0 {
			rt.memberOf[rt.focusOrder[i]] = sc.group
		}
	}
	for g := range rt.groupLast {
		if _, ok := rt.groups[g]; !ok {
			delete(rt.groupLast, g)
		}
	}
	if g := rt.memberOf[rt.focused]; g != 0 {
		if rt.groupLast == nil {
			rt.groupLast = map[uint64]uint64{}
		}
		rt.groupLast[g] = rt.focused
	}
}

// groupKey moves the focus among the elements of the focused element's
// group with an arrow, Home or End, and reports whether it did.
func (rt *engine) groupKey(mods Modifiers, key Key) bool {
	g := rt.memberOf[rt.focused]
	if g == 0 || mods != 0 {
		return false
	}
	info := rt.groups[g]
	var members []uint64
	at := -1
	for _, id := range rt.focusOrder {
		if rt.memberOf[id] == g {
			if id == rt.focused {
				at = len(members)
			}
			members = append(members, id)
		}
	}
	n := len(members)
	if n == 0 || at < 0 {
		return false
	}
	to := -1
	back, forth := KeyLeft, KeyRight
	if info.rtl {
		back, forth = forth, back
	}
	switch {
	case key == forth && info.orient&Horizontal != 0, key == KeyDown && info.orient&Vertical != 0:
		to = (at + 1) % n
	case key == back && info.orient&Horizontal != 0, key == KeyUp && info.orient&Vertical != 0:
		to = (at - 1 + n) % n
	case key == KeyHome:
		to = 0
	case key == KeyEnd:
		to = n - 1
	default:
		return false
	}
	id := members[to]
	rt.focused, rt.focusVisible, rt.blinkStart = id, true, rt.now()
	if rt.groupLast == nil {
		rt.groupLast = map[uint64]uint64{}
	}
	rt.groupLast[g] = id
	if s := rt.states[id]; s != nil && info.selects && s.flags&flagClickable != 0 {
		s.clicks++
		s.clickMods = 0
	}
	rt.reveal(id)
	rt.requestFrame()
	return true
}
