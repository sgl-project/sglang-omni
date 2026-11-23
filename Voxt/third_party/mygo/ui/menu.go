package ui

import (
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/internal/platform"
)

// ContextMenu gives the element a context menu, which build fills with
// items. The system shows it where the element is clicked with the
// secondary button, or Control-clicked on macOS, and, while the element or
// one inside it has the focus, below it when the menu key or Shift+F10 is
// pressed. build runs when the menu opens, and again in the frame after an
// item was chosen, where the item's Chosen reports it:
//
//	row.ContextMenu(func(m *ui.Menu) {
//		if m.Item("Rename").Chosen() {
//			app.renaming = i
//		}
//		m.Separator()
//		if m.Item("Delete").Disabled(app.locked).Chosen() {
//			app.delete(i)
//		}
//	})
//
// The innermost element with a context menu gets the click. Text inputs
// and selectable text have one with their editing commands, which
// ContextMenu replaces: on a text input, on a selectable text, or on the
// Selectable container whose text was clicked. EditItems puts those
// commands in a menu of one's own:
//
//	message.Selectable().ContextMenu(func(m *ui.Menu) {
//		if m.Item("Reply").Chosen() {
//			app.reply(msg)
//		}
//		m.Separator()
//		m.EditItems()
//	})
func (e *node) ContextMenu(build func(m *Menu)) *node {
	e.flags |= flagContextMenu
	e.buildMenu(false, build)
	return e
}

// Menu makes the element a menu button: it opens a menu below it, which
// build fills with items, as the pointer goes down on it, as a pop-up
// button does, and for Enter, Space or Down while it has the focus. It
// takes the focus. build runs when the menu opens, and again in the frame
// after an item was chosen, where the item's Chosen reports it, as for
// ContextMenu. MenuButton is a button with a menu:
//
//	more := ui.ButtonBase(c).Label("More").Padding(4).Radius(6)
//	more.Children(func() { ui.Icon(c, moreIcon).Size(16, 16) })
//	more.Menu(func(m *ui.Menu) {
//		if m.Item("Duplicate").Chosen() {
//			app.duplicate()
//		}
//	})
func (e *node) Menu(build func(m *Menu)) *node {
	e.flags |= flagMenuButton | flagClickable | flagFocusable
	if e.role == RoleAuto || e.role == RoleButton {
		e.role = RoleMenuButton
	}
	e.buildMenu(true, build)
	return e
}

// buildMenu builds the element's menu, its menu button's or its context
// menu, as it opens, and again for the item chosen from it.
func (e *node) buildMenu(button bool, build func(m *Menu)) {
	rt := e.c.rt
	mr := &rt.menu
	switch {
	case mr.asked == e.id && mr.button == button:
		mr.asked = 0
		if e.disabled() {
			break
		}
		run := &menuRun{collect: true}
		pm := &platform.Menu{}
		build(&Menu{run: run, menu: pm, rt: rt, text: rt.states[mr.text]})
		var commands []string
		if slices.ContainsFunc(run.commands, func(c string) bool { return c != "" }) {
			commands = run.commands
		}
		rt.openMenu(e.id, button, pm, run.labels, commands)
	case mr.chosen == e.id && mr.chosenButton == button:
		mr.chosen = 0
		run := &menuRun{choice: mr.choice, label: mr.label}
		build(&Menu{run: run, rt: rt, text: rt.states[mr.chosenText]})
		if run.chosen {
			// The choice may change what was built before: build again.
			rt.consumed = true
		}
	}
}

// Menu is a menu being built, by the function given to ContextMenu or
// Element.Menu. Its items show in the order they are added.
type Menu struct {
	run  *menuRun
	menu *platform.Menu // what the menu shows, when it opens
	rt   *engine
	// text is the text input or selectable text the menu opened on, whose
	// editing items EditItems adds.
	text *state
}

// menuRun is a run of a context menu's function: the one collecting what
// the menu shows, or the one after an item was chosen, which finds it by
// its place among the items and its label.
type menuRun struct {
	collect bool
	seq     int
	labels  []string // the items' labels by place, when collecting
	// commands are the editing commands of EditItems' items by place, ""
	// for the others, when collecting.
	commands []string
	choice   int
	label    string
	chosen   bool // an item reported the choice
}

// MenuItem is an item of a Menu. Its methods set it up and return it, for
// chaining, and Chosen reports the choice.
type MenuItem struct {
	run      *menuRun
	seq      int
	label    string
	disabled bool
	sub      bool
	p        *platform.MenuItem // when the menu opens
}

func (m *Menu) add(label string, typ platform.MenuItemType) *MenuItem {
	r := m.run
	r.seq++
	it := &MenuItem{run: r, seq: r.seq, label: label}
	if r.collect {
		it.p = &platform.MenuItem{ID: r.seq, Label: label, Type: typ, Enabled: true, Visible: true}
		m.menu.Items = append(m.menu.Items, it.p)
		r.labels = append(r.labels, label)
		r.commands = append(r.commands, "")
	}
	return it
}

// EditItems adds the editing items of the text the menu opened on, as its
// menu has them without ContextMenu: Copy and Select All for selectable
// text, the platform's text field items for a text input. Choosing one does
// its command on the text, without reaching the menu's function. It adds
// nothing to a menu that opened on no text.
func (m *Menu) EditItems() {
	if m.text == nil || m.text.editor == nil {
		return
	}
	for _, c := range m.rt.editCommands(m.text) {
		if c.name == "" {
			m.Separator()
			continue
		}
		it := m.Item(c.label).Disabled(!c.on)
		if m.run.collect {
			m.run.commands[it.seq-1] = c.name
		}
	}
}

// Item adds an item showing label.
func (m *Menu) Item(label string) *MenuItem { return m.add(label, platform.MenuItemNormal) }

// Separator adds a line between groups of items.
func (m *Menu) Separator() { m.add("", platform.MenuItemSeparator) }

// Submenu adds an item showing label that opens a submenu, which build
// fills as ContextMenu's function fills the menu.
func (m *Menu) Submenu(label string, build func(m *Menu)) *MenuItem {
	it := m.add(label, platform.MenuItemSubmenu)
	it.sub = true
	sub := &Menu{run: m.run}
	if it.p != nil {
		it.p.Submenu = &platform.Menu{}
		sub.menu = it.p.Submenu
	}
	build(sub)
	return it
}

// Disabled shows the item grayed out when d is true: it cannot be chosen.
func (it *MenuItem) Disabled(d bool) *MenuItem {
	it.disabled = d
	if it.p != nil {
		it.p.Enabled = !d
	}
	return it
}

// Checked shows a check mark by the item when on is true, as for a setting
// that choosing the item toggles.
func (it *MenuItem) Checked(on bool) *MenuItem {
	if it.p != nil && !it.sub {
		it.p.Type, it.p.Checked = platform.MenuItemCheckbox, on
	}
	return it
}

// Shortcut shows a key with modifiers by the item, as the shortcut that
// does the same; the view handles the key itself, as with
// Context.Shortcut.
func (it *MenuItem) Shortcut(mods Modifiers, key Key) *MenuItem {
	if it.p != nil {
		it.p.Accelerator = accelerator(mods, key)
	}
	return it
}

// Chosen reports whether the item was chosen, in the frame after the user
// chose it.
func (it *MenuItem) Chosen() bool {
	r := it.run
	if r.collect || it.disabled || it.sub || r.choice != it.seq || r.label != it.label {
		return false
	}
	r.chosen = true
	return true
}

// menuState is what the engine knows of context menus: the one asked for,
// which the next frame builds, the one built, which shows after the frame,
// and the item chosen from the one shown.
type menuState struct {
	asked uint64 // the element whose menu opens, at (x, y)
	x, y  float32
	// text is the text input or selectable text the menu opens on, which
	// may be inside the element with the menu, and chosenText the one the
	// item chosen was from.
	text, chosenText uint64
	// button is set when the menu asked for is a menu button's, and
	// chosenButton when the item chosen is from one.
	button, chosenButton bool
	// release is the element whose menu opens when the secondary button
	// goes up, as on Windows, on the text releaseText.
	release, releaseText uint64
	pending              *shownMenu
	chosen               uint64
	choice               int
	label                string
}

// shownMenu is a context menu built for the element id. commands are the
// editing commands of a text input's items, nil for a view's menu.
type shownMenu struct {
	id       uint64
	button   bool
	menu     *platform.Menu
	labels   []string
	commands []string
	x, y     float32
	// text is the text the menu opened on, which takes its commands.
	text uint64
}

// textMenu returns the element whose context menu opens on the text input
// or selectable text s: s, or the Selectable container whose text it is
// when the container has a ContextMenu and s has none.
func (rt *engine) textMenu(s *state) *state {
	if s.flags&flagContextMenu == 0 && s.textScope != 0 {
		if c := rt.states[s.textScope]; c != nil && c.flags&flagContextMenu != 0 && c.flags&flagDisabled == 0 {
			return c
		}
	}
	return s
}

// menuTarget returns the innermost element of chain with a context menu, a
// text input's or the view's, or nil.
func (rt *engine) menuTarget(chain []uint64) *state {
	for _, id := range chain {
		s := rt.states[id]
		if s == nil || s.flags&flagDisabled != 0 {
			continue
		}
		if s.flags&(flagContextMenu|flagEditable|flagSelectable) != 0 {
			return s
		}
		if s.flags&flagUnselectable != 0 {
			return nil
		}
	}
	return nil
}

// askMenu has the next frame build the context menu of s, or with button
// its menu button's, to show at (x, y): where it was clicked, or below the
// focus.
func (rt *engine) askMenu(s *state, x, y float32, button bool) {
	mr := &rt.menu
	mr.asked, mr.x, mr.y, mr.button, mr.text = s.id, x, y, button, 0
	// The menu takes the press: the release goes to it.
	if rt.pressed != nil {
		rt.pressed.pressed = false
		rt.pressed = nil
	}
	rt.requestFrame()
}

// menuPress handles a press of the secondary button over chain: when an
// element there has a context menu, the menu opens, as the button goes
// down on macOS and Linux and up on Windows, and menuPress reports true.
// A text input takes the focus, and the caret unless the press is in the
// selection.
func (rt *engine) menuPress(chain []uint64, x, y float32) bool {
	s := rt.menuTarget(chain)
	if s == nil {
		return false
	}
	if s.flags&flagSelectable != 0 && s.editor == nil {
		if p := rt.textSelectionAt(s.id, x, y); p.id != 0 {
			s = rt.states[p.id]
		}
	}
	// A right-click in any selected paragraph keeps the whole selection.
	if !rt.textSelectionMenu(s, x, y) && s.flags&(flagEditable|flagSelectable) != 0 && s.editor != nil {
		if rt.focused != s.id {
			rt.focused = s.id
			rt.focusVisible = false
		}
		rt.blinkStart = rt.now()
		if ed := s.editor; ed.laidOut() {
			lx, ly := s.local(x, y)
			i := ed.hit(lx, ly)
			if a, b := ed.selection(); a == b || i < a || i > b {
				ed.commitCompose()
				ed.move(i, false)
			}
		}
	}
	text := uint64(0)
	if s.editor != nil {
		text = s.id
		s = rt.textMenu(s)
	}
	if runtime.GOOS == "windows" {
		rt.menu.release, rt.menu.releaseText = s.id, text
		rt.requestFrame()
		return true
	}
	rt.askMenu(s, x, y, false)
	rt.menu.text = text
	return true
}

// menuRelease opens the menu menuPress left for the release of the
// button, where it is released, and reports whether there was one.
func (rt *engine) menuRelease() bool {
	id := rt.menu.release
	if id == 0 {
		return false
	}
	text := rt.menu.releaseText
	rt.menu.release, rt.menu.releaseText = 0, 0
	if s := rt.states[id]; s != nil {
		rt.askMenu(s, rt.pointerX, rt.pointerY, false)
		rt.menu.text = text
	}
	return true
}

// menuKey opens the context menu of the focused element, or of the
// innermost around it with one, for the menu key or Shift+F10: below a
// text input's caret, or the element's top. It reports whether there was
// one.
func (rt *engine) menuKey() bool {
	s := rt.menuTarget(rt.focusChain())
	if s == nil {
		return false
	}
	x, y := s.vx, s.vy+min(s.vh, 32)
	if s.rtl {
		x += s.vw
	}
	text := uint64(0)
	if s.flags&(flagEditable|flagSelectable) != 0 && s.editor != nil {
		r := transformRect(s.world, s.editor.caretRect(s))
		x, y = r.X, r.Y+r.H
		text = s.id
		s = rt.textMenu(s)
	}
	rt.askMenu(s, x, y, false)
	rt.menu.text = text
	return true
}

// openMenuButton has the next frame build the menu of menu button s, to
// show below it.
func (rt *engine) openMenuButton(s *state) {
	x, y := s.menuCorner()
	rt.askMenu(s, x, y, true)
}

// menuCorner returns where the menu of a button goes: at its bottom left
// corner, or its bottom right one right to left, where the menu opens to
// the left (platform.Menu.RightToLeft).
func (s *state) menuCorner() (x, y float32) {
	if s.rtl {
		return s.vx + s.vw, s.vy + s.vh
	}
	return s.vx, s.vy + s.vh
}

// openMenu shows the context menu built for the element id once the frame
// is done.
func (rt *engine) openMenu(id uint64, button bool, pm *platform.Menu, labels, commands []string) {
	if len(pm.Items) == 0 {
		return
	}
	mr := &rt.menu
	if s := rt.states[id]; s != nil && s.rtl {
		pm.RightToLeft = true
	}
	mr.pending = &shownMenu{id: id, button: button, menu: pm, labels: labels, commands: commands, x: mr.x, y: mr.y, text: mr.text}
}

// resolveMenu opens the menu asked for that no ContextMenu built, after a
// pass of the view: a text input's, with the editing commands.
func (rt *engine) resolveMenu() {
	mr := &rt.menu
	id := mr.asked
	if id == 0 {
		return
	}
	mr.asked = 0
	if s := rt.states[id]; s != nil && s.seen == rt.frame && s.editor != nil && !mr.button {
		rt.editMenu(s)
	}
}

// showMenu hands the menu built in the frame to the host, which shows it
// after the event being handled.
func (rt *engine) showMenu() {
	p := rt.menu.pending
	if p == nil {
		return
	}
	rt.menu.pending = nil
	rt.host.popupMenu(p.menu, p.x, p.y, func(id int) { rt.menuChosen(p, id) })
}

// menuChosen takes the item id chosen from the menu p: the next frame
// reports it to ContextMenu's function, or the text input does the
// command.
func (rt *engine) menuChosen(p *shownMenu, id int) {
	if id < 1 || id > len(p.labels) {
		return
	}
	if p.commands != nil && p.commands[id-1] != "" {
		// An editing command, of the text's own menu or of EditItems.
		target := p.text
		if target == 0 {
			target = p.id
		}
		if s := rt.states[target]; s != nil && s.editor != nil {
			if !rt.textSelectionCommand(s, p.commands[id-1]) {
				s.editor.queue = append(s.editor.queue, editEvent{kind: editCommand, text: p.commands[id-1]})
			}
			rt.blinkStart = rt.now()
		}
	} else {
		mr := &rt.menu
		mr.chosen, mr.chosenButton, mr.choice, mr.label, mr.chosenText = p.id, p.button, id, p.labels[id-1], p.text
	}
	rt.requestFrame()
}

// editItem is an item of a text's editing menu: a separator without a
// name.
type editItem struct {
	label, name string
	on          bool
}

// editMenu opens the context menu of a text input or selectable text, with
// the editing commands its platform's text fields have in theirs.
func (rt *engine) editMenu(s *state) {
	commands := rt.editCommands(s)
	pm := &platform.Menu{}
	labels := make([]string, len(commands))
	names := make([]string, len(commands))
	for i, c := range commands {
		it := &platform.MenuItem{ID: i + 1, Label: c.label, Enabled: c.on, Visible: true}
		if c.name == "" {
			it.Type = platform.MenuItemSeparator
		}
		pm.Items = append(pm.Items, it)
		labels[i], names[i] = c.label, c.name
	}
	rt.openMenu(s.id, false, pm, labels, names)
}

// editCommands are the items of the editing menu of s: Copy and Select All
// for selectable text, the platform's text field items for a text input.
func (rt *engine) editCommands(s *state) []editItem {
	ed := s.editor
	a, b := ed.selection()
	selected := a != b
	if s.textScope != 0 && rt.selection.scope == s.textScope {
		_, _, selected = rt.textSelectionBounds()
	}
	type command = editItem
	texts := l10n.In(rt.lang)
	var (
		undo      = command{texts.Undo(), "undo", len(ed.undo) > 0}
		cut       = command{texts.Cut(), "cut", selected && !ed.password}
		copyText  = command{texts.Copy(), "copy", selected && !ed.password}
		paste     = command{texts.Paste(), "paste", true}
		del       = command{texts.Delete(), "delete", selected}
		selectAll = command{texts.SelectAll(), "selectAll", ed.buf.n > 0}
		separator = command{}
		commands  []command
	)
	switch {
	case ed.readOnly:
		commands = []command{copyText, separator, selectAll}
	case runtime.GOOS == "darwin":
		commands = []command{cut, copyText, paste, separator, selectAll}
	case runtime.GOOS == "windows":
		commands = []command{undo, separator, cut, copyText, paste, del, separator, selectAll}
	default:
		commands = []command{cut, copyText, paste, del, separator, selectAll}
	}
	return commands
}

// accelerator writes a key with modifiers as menus take it, as in
// "Super+Shift+Z".
func accelerator(mods Modifiers, key Key) string {
	name := keyName(key)
	if name == "" {
		return ""
	}
	var b strings.Builder
	for _, m := range []struct {
		mod  Modifiers
		name string
	}{{Super, "Super"}, {Ctrl, "Ctrl"}, {Alt, "Alt"}, {Shift, "Shift"}} {
		if mods&m.mod != 0 {
			b.WriteString(m.name)
			b.WriteByte('+')
		}
	}
	b.WriteString(name)
	return b.String()
}

func keyName(k Key) string {
	switch {
	case k >= KeyA && k <= KeyZ:
		return string(rune('a' + k - KeyA))
	case k >= Key0 && k <= Key9:
		return string(rune('0' + k - Key0))
	case k >= KeyF1 && k <= KeyF12:
		return "F" + strconv.Itoa(int(k-KeyF1)+1)
	}
	switch k {
	case KeyEnter:
		return "Enter"
	case KeyEscape:
		return "Escape"
	case KeyBackspace:
		return "Backspace"
	case KeyTab:
		return "Tab"
	case KeySpace:
		return "Space"
	case KeyDelete:
		return "Delete"
	case KeyInsert:
		return "Insert"
	case KeyHome:
		return "Home"
	case KeyEnd:
		return "End"
	case KeyPageUp:
		return "PageUp"
	case KeyPageDown:
		return "PageDown"
	case KeyLeft:
		return "Left"
	case KeyRight:
		return "Right"
	case KeyUp:
		return "Up"
	case KeyDown:
		return "Down"
	case KeyMinus:
		return "-"
	case KeyEqual:
		return "="
	case KeyComma:
		return ","
	case KeyPeriod:
		return "."
	case KeySlash:
		return "/"
	case KeySemicolon:
		return ";"
	case KeyQuote:
		return "'"
	case KeyBracketLeft:
		return "["
	case KeyBracketRight:
		return "]"
	case KeyBackslash:
		return "\\"
	case KeyBackquote:
		return "`"
	}
	return ""
}
