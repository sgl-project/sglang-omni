package ui

import (
	"runtime"
	"slices"
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// event handles a surface event on the main thread. It reports whether
// an element takes files dragged over or dropped at the event's position.
func (rt *engine) event(ev platform.SurfaceEvent) (taken bool) {
	if ev.Kind != platform.SurfaceFrame {
		// Whatever the event changes, the next frame builds anew.
		rt.redraw = false
	}
	x, y := float32(ev.X), float32(ev.Y)
	if ev.Kind == platform.PointerDown || ev.Kind == platform.WebViewPress {
		// A press is of no element until countPress finds it one.
		rt.lastPress.prev, rt.lastPress.id = rt.lastPress.id, 0
	}
	if rt.insp.pointer(rt, ev, x, y) {
		return true
	}
	switch ev.Kind {
	case platform.PointerMove, platform.PointerDown, platform.PointerUp, platform.PointerScroll:
		rt.mods = Modifiers(ev.Mods)
	}
	switch ev.Kind {
	case platform.SurfaceFrame:
		rt.surfaceFrame()
	case platform.SurfaceResize:
		rt.requestFrame()
	case platform.SurfaceShown:
		if rt.held {
			// What moves goes on from where the time puts it.
			rt.held = false
			rt.requestFrame()
		}
	case platform.PointerMove:
		rt.pointerMove(x, y)
	case platform.PointerDown:
		rt.pointerMove(x, y)
		rt.pointerDown(x, y, ev.Button, Modifiers(ev.Mods), ev.Clicks)
	case platform.PointerUp:
		rt.pointerMove(x, y)
		rt.pointerUp(ev.Button, ev.Clicks)
	case platform.WebViewPress:
		rt.webViewPress(x, y, ev.Button)
	case platform.PointerLeave:
		rt.pointerIn = false
		if rt.pressed == nil {
			rt.setHover(nil)
		} else {
			// The elements around the one pressed hover no more.
			rt.requestFrame()
		}
	case platform.PointerScroll:
		rt.pointerMove(x, y)
		rt.scroll(float32(ev.DX), float32(ev.DY), Modifiers(ev.Mods), ev.Precise)
	case platform.KeyPressed:
		rt.modsChanged(Modifiers(ev.Mods))
		taken = rt.keyDownPhysical(Modifiers(ev.Mods), Key(ev.Key), Key(ev.PhysicalKey), ev.Repeat)
	case platform.ModifiersChanged:
		rt.modsChanged(Modifiers(ev.Mods))
	case platform.KeyReleased:
		rt.modsChanged(Modifiers(ev.Mods))
		key, physical := Key(ev.Key), Key(ev.PhysicalKey)
		if key == KeyUnknown && rt.byPosition[physical] {
			key = physical
		}
		delete(rt.byPosition, physical)
		if h := rt.focusHandler(); h != nil {
			rt.deliver(h, InputEvent{Kind: InputKeyUp, Key: key, PhysicalKey: physical, Mods: Modifiers(ev.Mods)})
		}
	case platform.TextInput:
		rt.editEvent(rt.replaced(editEvent{kind: editInsert, text: ev.Text}, ev))
	case platform.TextComposition:
		rt.editEvent(rt.replaced(editEvent{kind: editCompose, text: ev.Text, caret: ev.Caret}, ev))
	case platform.SurfaceCommand:
		rt.editEvent(editEvent{kind: editCommand, text: ev.Text})
	case platform.SurfaceFocus:
		rt.windowFocused = true
		rt.blinkStart = rt.now()
		rt.focusChanged()
	case platform.SurfaceBlur:
		rt.windowFocused = false
		rt.selection.dragging = false
		// Keys let go of elsewhere never come back up here.
		rt.modsChanged(0)
		clear(rt.byPosition)
		if p := rt.pressed; p != nil {
			// The release will not come: an element taking its input
			// gets one now.
			if p.input != nil {
				rt.deliver(p, InputEvent{Kind: InputPointerUp, Button: rt.pressButton, Mods: rt.mods, Clicks: rt.pressClicks})
			}
			p.pressed = false
			rt.pressed = nil
			rt.requestFrame()
		}
		rt.focusChanged()
	case platform.FileDragOver:
		taken = rt.fileDrag(x, y)
	case platform.FileDragLeave:
		rt.fileDrag(-1, -1)
	case platform.FileDrop:
		taken = rt.fileDrop(x, y, ev.Files)
	case platform.DataDragOver:
		taken = rt.dataDragOver(x, y, ev.Drag)
	case platform.DataDragLeave:
		rt.clearDataOver()
	case platform.DataDrop:
		taken = rt.dataDrop(x, y, ev.Drag)
	case platform.AccessibilityOn:
		rt.accessibilityOn()
	case platform.AccessAction:
		rt.accessAction(ev)
	}
	if ev.Kind != platform.SurfaceFrame {
		// A press or Tab may have moved the focus: tell the host now, not
		// after the next frame, or the text of keys typed before it would
		// never reach the input method.
		rt.updateTextInput()
	}
	return taken
}

// hitChain returns the ids of the topmost element at (x, y) and of its
// ancestors, innermost first.
func (rt *engine) hitChain(x, y float32) []uint64 { return rt.appendHitChain(nil, x, y) }

// appendHitChain appends hitChain(x, y) to chain.
func (rt *engine) appendHitChain(chain []uint64, x, y float32) []uint64 {
	for i := len(rt.hits) - 1; i >= 0; i-- {
		h := &rt.hits[i]
		if !rt.contains(h.st, h.r, x, y) {
			continue
		}
		for s := h.st; s != nil; s = rt.states[s.parent] {
			chain = append(chain, s.id)
			if s.parent == 0 {
				break
			}
		}
		return chain
	}
	return chain
}

// setHover makes chain the elements under the pointer, and reports
// whether it took chain, which then must not change.
func (rt *engine) setHover(chain []uint64) bool {
	changed := len(chain) != len(rt.hover)
	if !changed {
		for i := range chain {
			if chain[i] != rt.hover[i] {
				changed = true
				break
			}
		}
	}
	if !changed {
		return false
	}
	// Only elements that look at their hover need a frame.
	need := false
	for _, list := range [][]uint64{chain, rt.hover} {
		for _, id := range list {
			if s := rt.states[id]; s != nil && s.flags&(flagHover|flagTrackPointer) != 0 {
				need = true
			}
		}
	}
	rt.hover = chain
	if need {
		rt.requestFrame()
	}
	rt.updateCursor()
	return true
}

func (rt *engine) pointerMove(x, y float32) {
	if d := &rt.scrollDrag; d.st != nil {
		s := d.st
		g := d.bars(rt.c.theme.scrollbarWidth())
		lx, ly := s.local(x, y)
		lx += s.x
		ly += s.y
		if d.horizontal {
			if travel := g.hTrack.W - 4 - g.h.W; travel > 0 {
				moved := lx - d.start
				if s.rtl {
					// The offset counts from the right.
					moved = -moved
				}
				s.scrollTo(dragTo(d.from, moved, travel, d.contentW-float64(s.w), s.contentW-float64(s.w)), s.scrollY)
			}
		} else if travel := g.vTrack.H - 4 - g.v.H; travel > 0 {
			s.scrollTo(s.scrollX, dragTo(d.from, ly-d.start, travel, d.contentH-float64(s.h), s.contentH-float64(s.h)))
		}
		rt.pointerX, rt.pointerY = x, y
		rt.requestFrame()
		return
	}
	if rt.pressed != nil && rt.pointerIn {
		dx, dy := rt.pressed.vector(rt.pointerX, rt.pointerY, x-rt.pointerX, y-rt.pointerY)
		rt.pressed.dragX += dx
		rt.pressed.dragY += dy
		rt.pressed.drag.move(rt.now(), dx, dy, x-rt.pointerX, y-rt.pointerY)
		if rt.pressed.flags&(flagTrackPointer|flagDraggable|flagEditable|flagSelectable) != 0 {
			rt.requestFrame()
		} else {
			rt.pressMove(rt.pointerX, rt.pointerY, x, y)
		}
		rt.dragMove(x, y)
	}
	moved := x != rt.pointerX || y != rt.pointerY
	rt.pointerX, rt.pointerY, rt.pointerIn = x, y, true
	if rt.selection.dragging && rt.pressed != nil {
		rt.moveTextSelection(x, y)
	}
	if rt.pressed == nil {
		// The chain goes in a buffer, which the hover's last chain
		// becomes when the hover takes it.
		old, chain := rt.hover, rt.appendHitChain(rt.chain[:0], x, y)
		if rt.setHover(chain) {
			rt.chain = old[:0]
		} else {
			rt.chain = chain[:0]
		}
	}
	if moved {
		// The element pressed takes the moves, else the one under the
		// pointer.
		if p := rt.pressed; p != nil && p.input != nil {
			rt.deliver(p, InputEvent{Kind: InputPointerMove, Button: rt.pressButton, Mods: rt.mods})
		} else if p == nil {
			if h := rt.handler(rt.hover); h != nil {
				rt.deliver(h, InputEvent{Kind: InputPointerMove, Button: -1, Mods: rt.mods})
			}
		}
	}
	if moved {
		for _, id := range rt.hover {
			if s := rt.states[id]; s != nil && s.flags&flagTrackPointer != 0 {
				rt.requestFrame()
				break
			}
		}
	}
}

const interactive = flagClickable | flagFocusable | flagEditable | flagSelectable | flagDragWindow | flagDraggable | flagTrackPointer

func (rt *engine) pointerDown(x, y float32, button int, mods Modifiers, clicks int) {
	chain := rt.hitChain(x, y)
	rt.setHover(chain)
	rt.pressDragger = 0
	if len(chain) > 0 {
		rt.downs = append(rt.downs, chain[0])
	} else {
		rt.downs = append(rt.downs, 0)
	}
	if button == 0 && rt.scrollbarPress(chain, x, y) {
		return
	}
	var target, focus *state
	for _, id := range chain {
		s := rt.states[id]
		if s == nil {
			continue
		}
		if target == nil && s.flags&interactive != 0 {
			target = s
		}
		if focus == nil && s.flags&(flagFocusable|flagEditable|flagSelectable) != 0 && s.flags&flagDisabled == 0 {
			focus = s
		}
	}
	keep := slices.ContainsFunc(chain, func(id uint64) bool {
		s := rt.states[id]
		return s != nil && s.flags&flagKeepFocus != 0
	})
	if button == 0 && !keep {
		newFocus := uint64(0)
		if focus != nil {
			newFocus = focus.id
		}
		if newFocus != rt.focused {
			rt.focused = newFocus
			rt.focusVisible = false
			rt.blinkStart = rt.now()
		}
	}
	h := rt.handler(chain)
	clicks = rt.countPress(chain, target, h, max(clicks, 1))
	if h != nil && rt.deliver(h, InputEvent{Kind: InputPointerDown, Button: button, Mods: mods, Clicks: clicks}) {
		// It takes the moves and the release.
		rt.pressed, rt.pressButton, rt.pressClicks = h, button, clicks
		h.pressed = true
		h.pressPending = true
		h.pressX, h.pressY = h.local(x, y)
		h.drag.reset(rt.now(), x, y)
		return
	}
	if button == 1 && rt.menuPress(chain, x, y) {
		return
	}
	rt.requestFrame()
	if target == nil {
		return
	}
	if button == 0 && target.flags&flagMenuButton != 0 && target.flags&flagDisabled == 0 {
		// A menu button's menu opens as the button goes down, and takes
		// the release.
		rt.openMenuButton(target)
		return
	}
	if button == 0 && target.flags&flagDragWindow != 0 && target.flags&(interactive&^flagDragWindow) == 0 {
		if clicks == 2 {
			rt.host.titleBarDoubleClicked()
		} else {
			rt.host.startDrag()
		}
		return
	}
	rt.pressed, rt.pressButton, rt.pressClicks = target, button, clicks
	rt.pressDragger = pressDragger(rt, chain, target)
	target.pressed, target.pressMods = true, mods
	target.pressPending = true
	target.pressX, target.pressY = target.local(x, y)
	target.drag.reset(rt.now(), x, y)
	if button == 0 && rt.pressTextSelection(target, x, y, mods, clicks) {
		return
	}
	if target.editor != nil {
		target.editor.pressMods = mods
		target.editor.press(target.pressX, target.pressY, clicks, button)
	}
	if clicks == 2 && button == 0 {
		target.doubleClicks++
	}
}

// countPress counts the backend's n-th quick press from the first on the
// element pressed: the innermost of the target and the handler, so that
// two handlers in one focusable element, or two buttons in one handler,
// count apart.
func (rt *engine) countPress(chain []uint64, target, h *state, n int) int {
	id := uint64(0)
	for _, c := range chain {
		if target != nil && c == target.id || h != nil && c == h.id {
			id = c
			break
		}
	}
	lp := &rt.lastPress
	if id != lp.prev || n <= lp.base {
		lp.base = n - 1
	}
	lp.id = id
	return n - lp.base
}

func (rt *engine) pointerUp(button, clicks int) {
	if button == rt.pressButton {
		clicks = rt.pressClicks // the engine's, which may have started over
	}
	if rt.scrollDrag.st != nil {
		rt.scrollDrag.st = nil
		rt.requestFrame()
		return
	}
	if s := rt.pressed; s != nil && button == rt.pressButton && rt.dragEnd() {
		// A drag ended, dropping its value: no click.
		rt.pressed = nil
		s.pressed = false
		rt.setHover(rt.hitChain(rt.pointerX, rt.pointerY))
		return
	}
	if p := rt.pressed; p != nil && p.input != nil {
		rt.deliver(p, InputEvent{Kind: InputPointerUp, Button: button, Mods: rt.mods, Clicks: max(clicks, 1)})
	}
	if button == 1 && rt.menuRelease() {
		return
	}
	s := rt.pressed
	if s == nil || button != rt.pressButton {
		return
	}
	if rt.selection.dragging {
		rt.moveTextSelection(rt.pointerX, rt.pointerY)
		rt.selection.dragging = false
	}
	rt.pressed = nil
	s.pressed = false
	// A press that moved ends a drag; a click does not.
	s.released = s.drag.n > 1
	v := s.drag.velocities(rt.now())
	s.releaseVX, s.releaseVY, s.releaseWX, s.releaseWY = v[0], v[1], v[2], v[3]
	if s.editor != nil {
		s.editor.release()
	}
	inside := rt.contains(s, Rect{s.x, s.y, s.w, s.h}, rt.pointerX, rt.pointerY)
	if inside && s.flags&flagDisabled == 0 {
		switch button {
		case 0:
			if s.dragRec.dragged(rt, s) {
				break // it moved the element (Draggable): no click
			}
			s.clicks++
			s.clickMods = s.pressMods
		case 1:
			s.rightClicks++
		}
	}
	rt.setHover(rt.hitChain(rt.pointerX, rt.pointerY))
	rt.requestFrame()
}

func (rt *engine) scroll(dx, dy float32, mods Modifiers, precise bool) {
	chain := rt.appendHitChain(rt.chain[:0], rt.pointerX, rt.pointerY)
	rt.chain = chain[:0]
	if h := rt.handler(chain); h != nil && rt.deliver(h, InputEvent{Kind: InputScroll, DX: dx, DY: dy, Mods: mods, Precise: precise}) {
		return
	}
	// Shift turns the wheel sideways: down is toward the end, which is
	// to the left right to left.
	sideways := mods&Shift != 0 && dx == 0
	if sideways {
		dx, dy = dy, 0
	}
	for _, id := range chain {
		s := rt.states[id]
		if s == nil {
			continue
		}
		sx, sy := s.vector(rt.pointerX, rt.pointerY, dx, dy)
		if sideways && s.rtl {
			sx = -sx
		}
		if scrollBy(s, sx, sy) {
			rt.requestFrame()
			return
		}
	}
}

// dragTo returns the offset of content whose thumb moved by moved DIPs
// along a track with travel DIPs of room since the offset was from, the
// content scrolling as far as reach when the drag started and as far as
// now: the thumb goes as far through the content as along the track, as
// rows a List measures meanwhile change the size of its content, and the
// end of the track shows the end.
func dragTo(from float64, moved, travel float32, reach, now float64) float64 {
	if reach <= 0 || now <= 0 {
		return 0
	}
	along := from/reach + float64(moved)/float64(travel)
	return max(0, min(along, 1)) * now
}

// bars returns the scroll bars of the container being dragged, with the
// content's size as the drag started.
func (d *scrollDrag) bars(width float32) scrollGeometry {
	s := d.st
	return scrollBars(Rect{s.x, s.y, s.w, s.h}, s.barInset, float32(d.contentW), float32(d.contentH), float32(s.scrollX), float32(s.scrollY), s.flags, width, s.rtl)
}

// scrollBy scrolls a container by dx, dy within its content, and reports
// whether it moved. dx moves the content's view to the right, which,
// right to left, is back toward its start.
func scrollBy(s *state, dx, dy float32) bool {
	if s.rtl {
		dx = -dx
	}
	x, y := s.scrollX, s.scrollY
	if dy != 0 && s.flags&flagScrollY != 0 {
		y = max(0, min(y+float64(dy), s.contentH-float64(s.h)))
	}
	if dx != 0 && s.flags&flagScrollX != 0 {
		x = max(0, min(x+float64(dx), s.contentW-float64(s.w)))
	}
	if x == s.scrollX && y == s.scrollY {
		return false
	}
	if s.track != nil {
		s.track.glide = nil // the user takes over from ScrollTo
	}
	s.scrollTo(x, y)
	return true
}

// scrollKey scrolls with the keys that scroll pages in browsers the
// innermost container around the focus that can go that way, or under the
// pointer without a focus, and reports whether one moved.
func (rt *engine) scrollKey(mods Modifiers, key Key) bool {
	const line, far = 40, 1e9
	var dx, dy, page float32
	switch {
	case mods == 0 && key == KeyDown:
		dy = line
	case mods == 0 && key == KeyUp:
		dy = -line
	case mods == 0 && key == KeyRight:
		dx = line
	case mods == 0 && key == KeyLeft:
		dx = -line
	case mods == 0 && (key == KeyPageDown || key == KeySpace):
		page = 1
	case mods == 0 && key == KeyPageUp, mods == Shift && key == KeySpace:
		page = -1
	case (mods == 0 || mods == Cmd) && key == KeyHome, mods == Cmd && key == KeyUp && runtime.GOOS == "darwin":
		dy = -far
	case (mods == 0 || mods == Cmd) && key == KeyEnd, mods == Cmd && key == KeyDown && runtime.GOOS == "darwin":
		dy = far
	default:
		return false
	}
	scroller := func(id uint64) bool {
		s := rt.states[id]
		return s != nil && s.flags&(flagScrollX|flagScrollY) != 0
	}
	chain := rt.focusChain()
	if !slices.ContainsFunc(chain, scroller) && rt.pointerIn {
		chain = rt.hitChain(rt.pointerX, rt.pointerY)
	}
	for _, id := range chain {
		if !scroller(id) {
			continue
		}
		s := rt.states[id]
		if page != 0 {
			// A page keeps a line of the last one in view.
			dy = page * max(s.h-line, s.h/2)
		}
		if scrollBy(s, dx, dy) {
			rt.requestFrame()
			return true
		}
	}
	return false
}

// focusChain returns the focused element and its ancestors, innermost
// first.
func (rt *engine) focusChain() []uint64 {
	var chain []uint64
	for s := rt.states[rt.focused]; s != nil; s = rt.states[s.parent] {
		chain = append(chain, s.id)
		if s.parent == 0 {
			break
		}
	}
	return chain
}

// claimed reports whether an element around the focus, or the window,
// handles the key as a shortcut.
func (rt *engine) claimed(k keyEvent) bool { return rt.claimedBy(k, true) }

// claimedBy reports whether an element around the focus handles the key as
// a shortcut, or, when window is set, the window.
func (rt *engine) claimedBy(k keyEvent, window bool) bool {
	var chain []uint64
	for _, r := range rt.regs {
		if r.mods != k.mods || r.key != k.key || r.overlay {
			continue // an overlay takes what the focus leaves
		}
		if r.id == 0 {
			if window {
				return true
			}
			continue
		}
		if chain == nil {
			chain = rt.focusChain()
		}
		if slices.Contains(chain, r.id) {
			return true
		}
	}
	return false
}

// keyDown handles a key pressed, and reports whether an element took it as
// it came (HandleInput).
func (rt *engine) keyDown(mods Modifiers, key Key, repeat bool) bool {
	return rt.keyDownPhysical(mods, key, KeyUnknown, repeat)
}

// keyDownPhysical is keyDown with the key by position (InputEvent.PhysicalKey).
// A chord of a key the layout does not type as a US key, as S on a Cyrillic
// layout, matches shortcuts by that position; plain typing is untouched.
func (rt *engine) keyDownPhysical(mods Modifiers, key, physical Key, repeat bool) bool {
	if platform.KeyByPosition(platform.Key(key), platform.Key(physical), platform.Modifiers(mods)) {
		if rt.byPosition == nil {
			rt.byPosition = map[Key]bool{}
		}
		rt.byPosition[physical] = true
		key = physical
	} else if !repeat {
		delete(rt.byPosition, physical)
	}
	if rt.inspectKey(mods, key) {
		return true
	}
	k := keyEvent{mods, key}
	if key == KeyEscape && mods == 0 && rt.dragCancel() {
		return true
	}
	if h := rt.focusHandler(); h != nil && !rt.claimed(k) && rt.deliver(h, InputEvent{Kind: InputKeyDown, Key: key, PhysicalKey: physical, Mods: mods, Repeat: repeat}) {
		rt.blinkStart = rt.now()
		return true
	}
	if s := rt.states[rt.focused]; s != nil && s.editor != nil && s.editor.onTextKey != nil && s.flags&flagEditable != 0 &&
		!typesText(k) && (s.editor.wants(k) || !rt.claimed(k)) && s.editor.textKey(k) {
		// OnTextKey took it as it came, as HandleInput does, but after the
		// keys and text before it, which it reads.
		rt.blinkStart = rt.now()
		rt.requestFrame()
		return true
	}
	if (key == KeyContextMenu && mods == 0 || key == KeyF10 && mods == Shift) && !rt.claimed(k) && rt.menuKey() {
		return false
	}
	if s := rt.states[rt.focused]; s != nil && s.editor != nil && s.flags&(flagEditable|flagSelectable) != 0 && s.editor.wants(k) {
		if rt.textSelectionKey(s, k) {
			rt.requestFrame()
			return false
		}
		s.editor.queue = append(s.editor.queue, editEvent{kind: editKey, mods: mods, key: key})
		rt.blinkStart = rt.now()
		rt.requestFrame()
		return false
	}
	if s := rt.states[rt.focused]; s != nil && s.flags&flagTypeSelect != 0 && !rt.claimed(k) {
		// The first letters of a row of a list: they add up while typed
		// quickly enough, with spaces once the first came.
		now := rt.now()
		if now.Sub(s.typedAt) >= typePause {
			s.typed = ""
		}
		if r, ok := typedRune(mods, key); ok && (r != ' ' || s.typed != "") {
			s.typed += string(r)
			s.typedAt = now
			s.markTyping()
			rt.requestFrame()
			return false
		}
	}
	if !rt.claimedBy(k, false) && rt.groupKey(mods, key) {
		return false
	}
	if key == KeyTab && (mods == 0 || mods == Shift) && !rt.claimed(k) {
		rt.moveFocus(mods == Shift)
		rt.requestFrame()
		return false
	}
	if s := rt.states[rt.focused]; s != nil && s.flags&flagMenuButton != 0 && s.flags&flagDisabled == 0 && !rt.claimedBy(k, false) &&
		(mods == 0 && (key == KeyEnter || key == KeySpace || key == KeyDown) || mods == Alt && key == KeyDown) {
		rt.focusVisible = true
		rt.openMenuButton(s)
		return false
	}
	if (key == KeyEnter || key == KeySpace) && mods == 0 {
		// A focused button or link takes them before the window's
		// shortcuts, such as a dialog's default button; a toggle takes
		// Space before them, and Enter after.
		s := rt.states[rt.focused]
		window := s != nil && key == KeyEnter && s.flags&flagToggle != 0
		if s != nil && !rt.claimedBy(k, window) && s.flags&flagClickable != 0 && s.flags&flagDisabled == 0 {
			s.clicks++
			s.clickMods = 0
			rt.focusVisible = true
			rt.requestFrame()
			return false
		}
	}
	if !rt.claimed(k) && rt.scrollKey(mods, key) {
		return false
	}
	rt.keys = append(rt.keys, k)
	rt.requestFrame()
	return false
}

// typePause is how long after a letter typed to choose a row the next
// one starts anew.
const typePause = time.Second

// typedRune returns the letter, digit or space a key types without
// modifiers but Shift, in lowercase, for choosing a row by its text.
func typedRune(mods Modifiers, key Key) (rune, bool) {
	switch {
	case mods&^Shift != 0:
		return 0, false
	case key >= KeyA && key <= KeyZ:
		return 'a' + rune(key-KeyA), true
	case key >= Key0 && key <= Key9:
		return '0' + rune(key-Key0), true
	case key == KeySpace:
		return ' ', true
	}
	return 0, false
}

// moveFocus focuses the next (or previous) element that takes the focus.
func (rt *engine) moveFocus(back bool) {
	// The dialog on top keeps the focus among its elements.
	order := rt.focusOrder
	if m := rt.modal; m != 0 {
		order = nil
		for i, sc := range rt.focusScopes {
			if sc.modal == m {
				order = append(order, rt.focusOrder[i])
			}
		}
	}
	n := len(order)
	if n == 0 {
		return
	}
	i := -1
	for j, id := range order {
		if id == rt.focused {
			i = j
			break
		}
	}
	// Tab stops once in a focus group, at its entry, and leaves it.
	g := rt.groupOf(rt.focused)
	d := 1
	if back {
		d = -1
	}
	next := -1
	for k := 1; k <= n; k++ {
		j := (i + d*k + 2*n) % n
		if i < 0 {
			j = (k - 1) % n
			if back {
				j = n - k
			}
		}
		if id := order[j]; rt.tabStop(id) && (g == 0 || rt.groupOf(id) != g) {
			next = j
			break
		}
	}
	if next < 0 {
		return // the group is all there is
	}
	rt.focused = order[next]
	rt.focusVisible = true
	rt.blinkStart = rt.now()
	s := rt.states[rt.focused]
	if s != nil && s.editor != nil {
		s.editor.selectAll()
	}
	rt.reveal(rt.focused)
	if s != nil && s.webView != nil {
		// Its page takes the keyboard, and Tab leaving it moves on from
		// here (webViewTabOut).
		rt.host.tabIntoWebView(s.webView, back)
	}
}

// routeKeys delivers the keys pressed since the last frame to the
// innermost element around the focus that handles them, else to the
// overlay on top that does, else to the window's shortcuts, as the last
// frame registered them. Those it finds no shortcut for wait for what
// this frame builds (settleKeys).
func (rt *engine) routeKeys() {
	rt.unrouted = rt.unrouted[:0]
	if len(rt.keys) == 0 && len(rt.retried) == 0 {
		return
	}
	chain := rt.focusChain()
	route := func(k keyEvent, retried bool) {
		r := shortcutReg{mods: k.mods, key: k.key, retried: retried}
		if target, found := routeKey(rt.regs, chain, k); found {
			r.id = target
			rt.delivered = append(rt.delivered, r)
		} else {
			rt.unrouted = append(rt.unrouted, r)
		}
	}
	// Keys retried came before those pressed since.
	for _, k := range rt.retried {
		route(k, true)
	}
	for _, k := range rt.keys {
		route(k, false)
	}
	rt.retried, rt.keys = rt.retried[:0], rt.keys[:0]
}

// routeKey returns the element that takes key k by shortcuts regs, the
// focus in the elements of chain: the innermost of them, else the overlay
// on top, else the window (0). found is false when none does.
func routeKey(regs []shortcutReg, chain []uint64, k keyEvent) (target uint64, found bool) {
	for _, id := range chain {
		for _, r := range regs {
			if r.id == id && !r.overlay && r.mods == k.mods && r.key == k.key {
				return id, true
			}
		}
	}
	// The overlay on top: the one made last, as it paints last.
	var top int32 = -1
	for _, r := range regs {
		if r.overlay && r.mods == k.mods && r.key == k.key && r.serial >= top {
			target, found, top = r.id, true, r.serial
		}
	}
	if found {
		return target, true
	}
	for _, r := range regs {
		if r.id == 0 && r.mods == k.mods && r.key == k.key {
			return 0, true
		}
	}
	return 0, false
}

// settleKeys decides, once the frame is built, about the keys no shortcut
// took in it: those the frame found no shortcut for, or whose shortcut
// the view did not ask for again. The shortcuts the frame registered take
// one in the next frame, as the view had changed by the time it came: a
// key pressed as the view changes, before a frame shows the change (an
// Escape typed ahead into a flyout whose first frame builds a shortcut for
// it), goes where the view shows it goes. Else an Escape is the window's
// (a flyout closes on it), and other keys go nowhere.
func (rt *engine) settleKeys() {
	if len(rt.unrouted) == 0 {
		return
	}
	chain := rt.focusChain()
	for _, r := range rt.unrouted {
		k := keyEvent{r.mods, r.key}
		if _, found := routeKey(rt.nextRegs, chain, k); found && !r.retried {
			rt.retried = append(rt.retried, k) // runFrame asks for the frame
		} else if !found && k == (keyEvent{0, KeyEscape}) {
			rt.host.escape()
		}
	}
	rt.unrouted = rt.unrouted[:0]
}

// shortcut registers that element id (0 for the window) handles mods+key
// and reports whether such a key was delivered to it.
func (rt *engine) shortcut(id uint64, mods Modifiers, key Key) bool {
	rt.nextRegs = append(rt.nextRegs, shortcutReg{id: id, mods: mods, key: key})
	for i, d := range rt.delivered {
		if d.id == id && d.mods == mods && d.key == key {
			rt.delivered = append(rt.delivered[:i], rt.delivered[i+1:]...)
			rt.consumed = true
			return true
		}
	}
	return false
}

func (rt *engine) editEvent(ev editEvent) {
	if s := rt.states[rt.focused]; s != nil && s.textClient != nil && rt.windowFocus() && s.flags&flagDisabled == 0 && (s.editor == nil || !s.editor.readOnly) {
		if ev.kind == editInsert || ev.kind == editCompose {
			var r *TextInputRange
			if ev.replace {
				v := TextInputRange{Start: ev.from, End: ev.to}
				r = &v
			}
			if ev.kind == editInsert {
				s.textAdapter.ReplaceText(r, ev.text)
			} else {
				caret := platform.UTF16Len(string([]rune(ev.text)[:max(0, min(ev.caret, len([]rune(ev.text))))]))
				s.textAdapter.SetMarkedText(r, ev.text, TextInputRange{Start: caret, End: caret})
			}
			return
		}
	}
	if h := rt.focusHandler(); h != nil && h.editor == nil {
		kind := map[editKind]InputKind{editInsert: InputText, editCompose: InputCompose, editCommand: InputCommand}[ev.kind]
		handled := kind != 0 && rt.deliver(h, InputEvent{Kind: kind, Text: ev.text, Caret: ev.caret})
		if !handled && kind == InputCommand && ev.text == "pasteAndMatchStyle" {
			// A handler of text without styles pastes it as Paste does.
			handled = rt.deliver(h, InputEvent{Kind: kind, Text: "paste"})
		}
		if handled {
			rt.blinkStart = rt.now()
		}
		return
	}
	s := rt.states[rt.focused]
	if s == nil || s.editor == nil {
		return
	}
	// Selectable text takes the menus' commands, as Copy, not text.
	if s.flags&flagEditable == 0 && (s.flags&flagSelectable == 0 || ev.kind != editCommand) {
		return
	}
	if ev.kind == editCommand && rt.textSelectionCommand(s, ev.text) {
		rt.requestFrame()
		return
	}
	s.editor.queue = append(s.editor.queue, ev)
	rt.blinkStart = rt.now()
	rt.requestFrame()
}

// imeContext is how many runes around the selection input methods see.
const imeContext = 512

// updateTextInput tells the host where text input goes, and the text
// around the caret.
func (rt *engine) updateTextInput() {
	var t platform.TextInputState
	base := 0
	if s := rt.states[rt.focused]; s != nil && s.textClient != nil && rt.windowFocus() && s.flags&flagDisabled == 0 && (s.editor == nil || !s.editor.readOnly) {
		t.Active, t.Client, t.NoInputMethod = true, s.textAdapter, s.noIME
		sel := s.textAdapter.Selection()
		caret := sel.Caret()
		if bounds, _, ok := s.textAdapter.BoundsForRange(TextInputRange{Start: caret, End: caret}); ok {
			t.Caret = bounds
		} else {
			r := transformRect(s.world, Rect{X: s.x, Y: s.y, W: 1, H: s.h})
			t.Caret = platform.RectF{X: float64(r.X), Y: float64(r.Y), W: float64(r.W), H: float64(r.H)}
		}
	} else if s != nil && s.editor == nil && s.input != nil && s.takesText && rt.windowFocus() {
		// An element taking text itself: no text around the caret.
		t.Active, t.NoInputMethod = true, s.noIME
		r := transformRect(s.world, Rect{s.x + s.caret.X, s.y + s.caret.Y, s.caret.W, s.caret.H})
		t.Caret = platform.RectF{X: float64(r.X), Y: float64(r.Y), W: float64(r.W), H: float64(r.H)}
	}
	if t != rt.ime.state {
		if t.Client != rt.ime.state.Client {
			if old, ok := rt.ime.state.Client.(*textInputAdapter); ok {
				old.release()
			}
		}
		rt.ime.state, rt.ime.base = t, base
		rt.host.setTextInput(t)
	}
}

// replaced makes an edit replace the runes an input method named, from
// the text it was last given, rather than the selection.
func (rt *engine) replaced(ev editEvent, sev platform.SurfaceEvent) editEvent {
	if rt.ime.state.Client != nil {
		if sev.Replace {
			ev.replace, ev.from, ev.to = true, sev.From, sev.To
		}
		return ev
	}
	if sev.Replace && rt.ime.state.Active {
		n := len([]rune(rt.ime.state.Text))
		from, to := max(0, min(sev.From, n)), max(0, min(sev.To, n))
		ev.replace, ev.from, ev.to = true, rt.ime.base+min(from, to), rt.ime.base+max(from, to)
	}
	return ev
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// Clicked reports whether the element was clicked, with the primary
// button or by Enter or Space while focused, since the last frame.
func (e *node) Clicked() bool {
	if !e.hasState() {
		return false
	}
	e.flags |= flagClickable
	if e.disabled() || e.st.clicks == 0 {
		return false
	}
	e.c.rt.consumed = true
	return true
}

// Clicks returns how many times the element was clicked since the last
// frame.
func (e *node) Clicks() int {
	if !e.hasState() {
		return 0
	}
	e.flags |= flagClickable
	if e.disabled() {
		return 0
	}
	if e.st.clicks > 0 {
		e.c.rt.consumed = true
	}
	return e.st.clicks
}

// ClickModifiers returns the modifier keys held as the element was last
// clicked, none for a click by the keyboard: with Clicked, a click with
// Shift or Cmd does something else, as extending a choice.
func (e *node) ClickModifiers() Modifiers {
	if !e.hasState() {
		return 0
	}
	return e.st.clickMods
}

// DoubleClicked reports a double click on the element.
func (e *node) DoubleClicked() bool {
	if !e.hasState() {
		return false
	}
	e.flags |= flagClickable
	if e.disabled() || e.st.doubleClicks == 0 {
		return false
	}
	e.c.rt.consumed = true
	return true
}

// RightClicked reports a click with the secondary button, as for a
// context menu.
func (e *node) RightClicked() bool {
	if !e.hasState() {
		return false
	}
	e.flags |= flagClickable
	if e.disabled() || e.st.rightClicks == 0 {
		return false
	}
	e.c.rt.consumed = true
	return true
}

// Hovered reports whether the pointer is over the element. While the
// pointer presses an element, the elements it was over as the press began,
// as those around the element pressed, stay hovered as long as it is over
// them, as in CSS, and the others hover no more: a button that shows over
// a row stays as it is pressed, and dragging over other elements does not
// light them up.
func (e *node) Hovered() bool {
	if !e.hasState() {
		return false
	}
	e.flags |= flagHover
	if e.IsDisabled() {
		return false
	}
	rt := e.c.rt
	if !slices.Contains(rt.hover, e.id) {
		return false
	}
	if p := rt.pressed; p != nil && p.id != e.id {
		// The hover holds the elements under the pointer as the press
		// began.
		s := e.st
		return rt.pointerIn && rt.contains(s, Rect{s.x, s.y, s.w, s.h}, rt.pointerX, rt.pointerY)
	}
	return true
}

// pressMove asks for a frame when the pointer, pressing an element, moves
// in or out of an element around it that looks at its hover (Hovered).
func (rt *engine) pressMove(x0, y0, x1, y1 float32) {
	for _, id := range rt.hover {
		s := rt.states[id]
		if s == nil || s == rt.pressed || s.flags&flagHover == 0 {
			continue
		}
		r := Rect{s.x, s.y, s.w, s.h}
		if rt.contains(s, r, x0, y0) != rt.contains(s, r, x1, y1) {
			rt.requestFrame()
			return
		}
	}
}

// Pressed reports whether the element is being pressed with the pointer,
// unless it is disabled.
func (e *node) Pressed() bool {
	if !e.hasState() {
		return false
	}
	e.flags |= flagClickable | flagHover
	s := e.st
	rt := e.c.rt
	if !s.pressed || e.disabled() || !rt.contains(s, Rect{s.x, s.y, s.w, s.h}, rt.pointerX, rt.pointerY) {
		return false
	}
	if s.pressPending {
		// A press may choose a row after the view built its selection and
		// focus colors. Show that choice in this frame, as for a click,
		// without rebuilding again while the pointer remains held.
		rt.consumed = true
	}
	return true
}

// Focused reports whether the element has the keyboard focus.
func (e *node) Focused() bool {
	return e.hasState() && e.c.rt.focused == e.id && e.c.rt.windowFocus()
}

// FocusVisible reports whether the element has the keyboard focus and
// should show it, because it came from the keyboard.
func (e *node) FocusVisible() bool { return e.Focused() && e.c.rt.focusVisible }

// FocusWithin reports whether the element or one of its descendants has
// the keyboard focus.
func (e *node) FocusWithin() bool {
	if !e.hasState() {
		return false
	}
	rt := e.c.rt
	for s := rt.states[rt.focused]; s != nil; s = rt.states[s.parent] {
		if s.id == e.id {
			return true
		}
		if s.parent == 0 {
			break
		}
	}
	return false
}

// Focus gives the element the keyboard focus. Called in every frame, it
// keeps it there; AutoFocus gives it once.
func (e *node) Focus() *node {
	if !e.hasState() {
		return e
	}
	e.flags |= flagFocusable
	rt := e.c.rt
	if rt.focused != e.id {
		rt.focused = e.id
		rt.blinkStart = rt.now()
	}
	return e
}

// AutoFocus gives the element the keyboard focus in the frame it appears,
// as the first field of a dialog.
func (e *node) AutoFocus() *node {
	if e.hasState() && e.st.born == e.c.rt.frame {
		e.Focus()
	}
	return e
}

// Shortcut reports whether the key with exactly the modifiers mods was
// pressed while the element or one of its descendants had the focus. The
// innermost element handling a key gets it; a disabled one handles none.
func (e *node) Shortcut(mods Modifiers, key Key) bool {
	if e.disabled() {
		return false
	}
	return e.c.rt.shortcut(e.id, mods, key)
}

// PointerPosition returns the pointer's position relative to the
// element's box and whether it is over the element. Elements asking for it
// get a frame whenever the pointer moves over them.
func (e *node) PointerPosition() (x, y float32, over bool) {
	if !e.hasState() {
		return 0, 0, false
	}
	e.flags |= flagTrackPointer
	rt := e.c.rt
	s := e.st
	x, y = s.local(rt.pointerX, rt.pointerY)
	return x, y, rt.pointerIn && rt.contains(s, Rect{s.x, s.y, s.w, s.h}, rt.pointerX, rt.pointerY)
}

// Dragged reports how far the pointer moved in the element's local axes
// since the last frame while pressing it.
func (e *node) Dragged() (dx, dy float32, ok bool) {
	if !e.hasState() {
		return 0, 0, false
	}
	e.flags |= flagDraggable
	s := e.st
	if !s.pressed {
		return 0, 0, false
	}
	if s.dragX != 0 || s.dragY != 0 {
		e.c.rt.consumed = true
	}
	return s.dragX, s.dragY, true
}

// DragVelocity returns how fast, in DIPs per second along the element's
// local axes, the pointer pressing the element moves now: zero once it
// stops for a moment, or while nothing presses the element.
func (e *node) DragVelocity() (vx, vy float32) {
	if !e.hasState() {
		return 0, 0
	}
	e.flags |= flagDraggable
	s := e.st
	if !s.pressed {
		return 0, 0
	}
	return s.drag.velocity(e.c.rt.now())
}

// DragEnded reports whether the pointer pressing the element, and moving,
// let go since the last frame, with how fast it moved then, as
// DragVelocity: a spring taking over (SpringOptions.Velocity) keeps the
// fling going. A click, which does not move, ends no drag.
func (e *node) DragEnded() (vx, vy float32, ok bool) {
	if !e.hasState() {
		return 0, 0, false
	}
	e.flags |= flagDraggable
	s := e.st
	if !s.released {
		return 0, 0, false
	}
	e.c.rt.consumed = true
	return s.releaseVX, s.releaseVY, true
}

// dragTrack keeps the last moves of a press, to tell how fast it goes: in
// the element's axes (x, y), and in the window's (wx, wy, from where the
// press began, ox and oy). far tells that the press went dragStart DIPs
// from there.
type dragTrack struct {
	x, y    float32
	ox, oy  float32
	wx, wy  float32
	far     bool
	samples [8]dragSample
	n, last int
}

type dragSample struct {
	at           time.Time
	x, y, wx, wy float32
}

// reset starts tracking a press at the window's point (x, y).
func (d *dragTrack) reset(now time.Time, x, y float32) {
	*d = dragTrack{ox: x, oy: y}
	d.add(now)
}

func (d *dragTrack) add(now time.Time) {
	d.last = (d.last + 1) % len(d.samples)
	d.samples[d.last] = dragSample{now, d.x, d.y, d.wx, d.wy}
	d.n = min(d.n+1, len(d.samples))
}

// move adds a move by dx, dy in the element's axes, wdx, wdy in the
// window's, at now.
func (d *dragTrack) move(now time.Time, dx, dy, wdx, wdy float32) {
	if wdx == 0 && wdy == 0 {
		return // as with the release's position
	}
	d.x += dx
	d.y += dy
	d.wx += wdx
	d.wy += wdy
	d.far = d.far || d.wx*d.wx+d.wy*d.wy >= dragStart*dragStart
	d.add(now)
}

// velocity returns how fast the press went over its last 100 ms in the
// element's axes, zero once it has not moved for 50 ms.
func (d *dragTrack) velocity(now time.Time) (float32, float32) {
	v := d.velocities(now)
	return v[0], v[1]
}

// windowVelocity returns how fast the press went, as velocity does, in
// the window's axes.
func (d *dragTrack) windowVelocity(now time.Time) (float32, float32) {
	v := d.velocities(now)
	return v[2], v[3]
}

// velocities returns how fast the press went in the element's axes and
// in the window's.
func (d *dragTrack) velocities(now time.Time) [4]float32 {
	if d.n < 2 {
		return [4]float32{}
	}
	newest := d.samples[d.last]
	if now.Sub(newest.at) > 50*time.Millisecond {
		return [4]float32{}
	}
	oldest := newest
	for i := 1; i < d.n; i++ {
		s := d.samples[(d.last-i+len(d.samples))%len(d.samples)]
		if newest.at.Sub(s.at) > 100*time.Millisecond {
			break
		}
		oldest = s
	}
	dt := float32(newest.at.Sub(oldest.at).Seconds())
	if dt <= 0 {
		return [4]float32{}
	}
	return [4]float32{(newest.x - oldest.x) / dt, (newest.y - oldest.y) / dt, (newest.wx - oldest.wx) / dt, (newest.wy - oldest.wy) / dt}
}

// Changed reports whether this widget's value changed in this build pass,
// by the input it took as its constructor returned.
func (e *node) Changed() bool {
	if e.hasState() {
		e.c.rt.applyInputs()
	}
	if e.hasState() && e.st.changed {
		e.c.rt.consumed = true
		return true
	}
	return false
}

// Submitted reports whether Enter was pressed in a single-line text input
// in this build pass.
func (e *node) Submitted() bool {
	if !e.hasState() {
		return false
	}
	e.c.rt.applyInputs()
	return e.st.submitted
}

// Dismissed reports whether pressing Escape or clicking outside closed the
// dialog or popover (Modal, DialogBase, Popover, PopoverBase) returning the
// element, in the frame it did, setting its *open to false: one that must
// stay open sets it back to true.
//
//	if ui.Modal(c, &app.editing, edit).Dismissed() && app.unsaved {
//		app.editing = true
//	}
func (e *node) Dismissed() bool { return e.dismissed }

// scrollbarPress starts dragging the thumb of the scroll container under
// the pointer when the press is on its scroll bar, or pages toward the
// press on the bar's track.
func (rt *engine) scrollbarPress(chain []uint64, x, y float32) bool {
	for _, id := range chain {
		s := rt.states[id]
		if s == nil || s.flags&(flagScrollX|flagScrollY) == 0 {
			continue
		}
		lx, ly := s.local(x, y)
		lx += s.x
		ly += s.y
		g := scrollBars(Rect{s.x, s.y, s.w, s.h}, s.barInset, float32(s.contentW), float32(s.contentH), float32(s.scrollX), float32(s.scrollY), s.flags, rt.c.theme.scrollbarWidth(), s.rtl)
		d := &rt.scrollDrag
		w, h := float64(s.w), float64(s.h)
		switch {
		case g.vertical && g.vTrack.Contains(lx, ly):
			switch {
			case ly >= g.v.Y && ly < g.v.Y+g.v.H:
				d.st, d.start, d.from, d.horizontal = s, ly, s.scrollY, false
				if s.track != nil {
					s.track.glide = nil
				}
				d.contentW, d.contentH = s.contentW, s.contentH
			case ly < g.v.Y:
				s.scrollTo(s.scrollX, max(0, s.scrollY-h*0.9))
			default:
				s.scrollTo(s.scrollX, min(s.contentH-h, s.scrollY+h*0.9))
			}
		case g.horizontal && g.hTrack.Contains(lx, ly):
			switch {
			case lx >= g.h.X && lx < g.h.X+g.h.W:
				d.st, d.start, d.from, d.horizontal = s, lx, s.scrollX, true
				if s.track != nil {
					s.track.glide = nil
				}
				d.contentW, d.contentH = s.contentW, s.contentH
			case (lx < g.h.X) != s.rtl:
				// Toward the start.
				s.scrollTo(max(0, s.scrollX-w*0.9), s.scrollY)
			default:
				s.scrollTo(min(s.contentW-w, s.scrollX+w*0.9), s.scrollY)
			}
		default:
			continue
		}
		rt.requestFrame()
		return true
	}
	return false
}

// windowFocus reports whether the window has the keyboard focus, noting
// that the frame depends on it.
func (rt *engine) windowFocus() bool {
	rt.focusRead = true
	return rt.windowFocused
}

// focusChanged draws a frame after the window gained or lost the keyboard
// focus, when the last frame read it (a focus ring, a caret, a selection):
// another frame would draw the same, as a window hides or shows again, and
// cost the GPU's driver tens of MB for a second or two.
func (rt *engine) focusChanged() {
	if rt.focusRead {
		rt.requestFrame()
	}
}

// modsChanged takes the modifier keys held now, drawing a frame when they
// changed, for views that show what a held key would do.
func (rt *engine) modsChanged(mods Modifiers) {
	if rt.mods != mods {
		rt.mods = mods
		rt.requestFrame()
	}
}

// Modifiers returns the modifier keys held now, as the last key, pointer
// or modifier event said: a view showing each row's shortcut while Cmd is
// held reads it, and draws again as it changes.
func (c *context) Modifiers() Modifiers { return c.rt.mods }
