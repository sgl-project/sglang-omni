//go:build js

package web

import (
	"strings"
	"syscall/js"

	"github.com/egoist/mygo/internal/platform"
)

// menuCSS styles context menus, which the page draws with HTML: browsers
// show no menus of a page's own.
const menuCSS = `
.mygo-menu{position:fixed;z-index:2147483647;min-width:180px;padding:5px;box-sizing:border-box;border-radius:8px;
font:13px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1d1d1f;background:rgba(246,246,246,.96);
-webkit-backdrop-filter:blur(20px);backdrop-filter:blur(20px);box-shadow:0 0 0 .5px rgba(0,0,0,.2),0 8px 24px rgba(0,0,0,.18);
user-select:none;-webkit-user-select:none;cursor:default;outline:none}
.mygo-menu-item{display:flex;align-items:center;gap:16px;height:22px;padding:0 10px 0 22px;border-radius:4px;position:relative;white-space:nowrap}
.mygo-menu-item[data-checked]::before{content:"✓";position:absolute;left:7px;font-size:11px}
.mygo-menu-item>.label{flex:1}
.mygo-menu-item>.accel{opacity:.5}
.mygo-menu-item.on{background:#0a64d6;color:#fff}
.mygo-menu-item.on>.accel{opacity:.8}
.mygo-menu-item[aria-disabled=true]{opacity:.35}
.mygo-menu-separator{height:1px;margin:5px 10px;background:rgba(0,0,0,.12)}
@media (prefers-color-scheme:dark){.mygo-menu{color:#f5f5f7;background:rgba(40,40,42,.96);box-shadow:0 0 0 .5px rgba(255,255,255,.18),0 8px 24px rgba(0,0,0,.4)}
.mygo-menu-separator{background:rgba(255,255,255,.14)}}
`

var menuStyled bool

// popup is a context menu shown on the page.
type popup struct {
	b     *Backend
	w     *window
	menus []js.Value // the menu and its submenus open
	funcs []js.Func
	// removers remove the listeners of the page's events.
	removers []func()
	done     bool
	// rtl lays the menus out right to left (Menu.RightToLeft).
	rtl bool
}

// PopupMenu shows m at pos in w's content, or where the pointer last
// pressed, and waits until the user chooses an item or dismisses it,
// handling events meanwhile.
func (b *Backend) PopupMenu(m *platform.Menu, pw platform.Window, pos *platform.Point) {
	w, _ := pw.(*window)
	if w == nil {
		b.mu.Lock()
		if n := len(b.windows); n > 0 {
			w = b.windows[n-1]
		}
		b.mu.Unlock()
	}
	if w == nil || w.closed || m == nil || len(m.Items) == 0 {
		return
	}
	doc := js.Global().Get("document")
	if !menuStyled {
		menuStyled = true
		style := doc.Call("createElement", "style")
		style.Set("textContent", menuCSS)
		doc.Get("head").Call("appendChild", style)
	}
	x, y := w.s.downX, w.s.downY
	if pos != nil {
		x, y = float64(pos.X), float64(pos.Y)
	}
	r := w.s.canvas.Call("getBoundingClientRect")
	x += r.Get("left").Float()
	y += r.Get("top").Float()

	p := &popup{b: b, w: w, rtl: m.RightToLeft}
	chosen := 0
	p.open(m, x, y, 0, func(id int) { chosen = id })
	// Dismissed by a press elsewhere, Escape, or the page losing focus.
	p.listen(doc, "pointerdown", func(e js.Value) {
		if p.inside(e.Get("target")) {
			// The window keeps the keyboard.
			e.Call("preventDefault")
			return
		}
		p.done = true
	})
	p.listen(doc, "keydown", func(e js.Value) {
		if e.Get("key").String() == "Escape" {
			e.Call("preventDefault")
			e.Call("stopPropagation")
			p.done = true
		}
	})
	p.listen(js.Global(), "blur", func(e js.Value) {
		// The page's, not an element's, which capturing sees too.
		if e.Get("target").Equal(js.Global()) {
			p.done = true
		}
	})
	for !p.done && !w.closed {
		b.Step()
	}
	for _, el := range p.menus {
		el.Call("remove")
	}
	for _, remove := range p.removers {
		remove()
	}
	for _, l := range p.funcs {
		l.Release()
	}
	w.s.focus()
	if chosen != 0 {
		w.h.MenuItemClicked(chosen)
	}
}

// listen adds a listener for the life of the menu, capturing, so that the
// window's own listeners do not see what the menu takes.
func (p *popup) listen(target js.Value, event string, fn func(e js.Value)) {
	f := js.FuncOf(func(_ js.Value, args []js.Value) any {
		p.b.sync(func() { fn(args[0]) })
		return nil
	})
	p.funcs = append(p.funcs, f)
	target.Call("addEventListener", event, f, true)
	p.removers = append(p.removers, func() { target.Call("removeEventListener", event, f, true) })
}

func (p *popup) inside(el js.Value) bool {
	for _, m := range p.menus {
		if m.Call("contains", el).Bool() {
			return true
		}
	}
	return false
}

// open shows menu m with its top left at (x, y), page pixels, as the menu
// of depth level, closing deeper ones.
func (p *popup) open(m *platform.Menu, x, y float64, level int, choose func(int)) {
	for len(p.menus) > level {
		p.menus[len(p.menus)-1].Call("remove")
		p.menus = p.menus[:len(p.menus)-1]
	}
	doc := js.Global().Get("document")
	el := doc.Call("createElement", "div")
	el.Set("className", "mygo-menu")
	el.Call("setAttribute", "role", "menu")
	if p.rtl {
		el.Call("setAttribute", "dir", "rtl")
	}
	for _, it := range m.Items {
		if !it.Visible {
			continue
		}
		if it.Type == platform.MenuItemSeparator {
			sep := doc.Call("createElement", "div")
			sep.Set("className", "mygo-menu-separator")
			sep.Call("setAttribute", "role", "separator")
			el.Call("appendChild", sep)
			continue
		}
		row := doc.Call("createElement", "div")
		row.Set("className", "mygo-menu-item")
		row.Call("setAttribute", "role", "menuitem")
		row.Call("setAttribute", "aria-disabled", !it.Enabled)
		if it.Checked && (it.Type == platform.MenuItemCheckbox || it.Type == platform.MenuItemRadio) {
			row.Call("setAttribute", "data-checked", "")
		}
		label := doc.Call("createElement", "span")
		label.Set("className", "label")
		label.Set("textContent", strings.ReplaceAll(it.Label, "&", ""))
		row.Call("appendChild", label)
		accel := doc.Call("createElement", "span")
		accel.Set("className", "accel")
		if it.Type == platform.MenuItemSubmenu {
			if p.rtl {
				accel.Set("textContent", "‹")
			} else {
				accel.Set("textContent", "›")
			}
		} else {
			accel.Set("textContent", accelText(it.Accelerator, p.b.mac))
		}
		row.Call("appendChild", accel)
		el.Call("appendChild", row)
		p.on(row, "pointerenter", func(js.Value) {
			for k := range el.Get("children").Length() {
				el.Get("children").Index(k).Get("classList").Call("remove", "on")
			}
			if !it.Enabled {
				return
			}
			row.Get("classList").Call("add", "on")
			if it.Type == platform.MenuItemSubmenu && it.Submenu != nil {
				r := row.Call("getBoundingClientRect")
				x := r.Get("right").Float() + 2
				if p.rtl {
					x = r.Get("left").Float() - 2
				}
				p.open(it.Submenu, x, r.Get("top").Float()-5, level+1, choose)
			} else {
				p.open(&platform.Menu{}, 0, 0, level+1, choose)
			}
		})
		p.on(row, "pointerup", func(js.Value) {
			if it.Enabled && it.Type != platform.MenuItemSubmenu {
				choose(it.ID)
				p.done = true
			}
		})
	}
	if el.Get("children").Length() == 0 {
		return
	}
	doc.Get("body").Call("appendChild", el)
	// Kept on the page.
	vw, vh := js.Global().Get("innerWidth").Float(), js.Global().Get("innerHeight").Float()
	r := el.Call("getBoundingClientRect")
	if p.rtl {
		// Its top right corner at x.
		x = max(x-r.Get("width").Float(), 4)
	}
	if x+r.Get("width").Float() > vw {
		x = max(vw-r.Get("width").Float()-4, 0)
	}
	if y+r.Get("height").Float() > vh {
		y = max(vh-r.Get("height").Float()-4, 0)
	}
	st := el.Get("style")
	st.Set("left", ftoa(x)+"px")
	st.Set("top", ftoa(y)+"px")
	p.menus = append(p.menus, el)
}

// on adds a listener to an element of the menu.
func (p *popup) on(el js.Value, event string, fn func(e js.Value)) {
	f := js.FuncOf(func(_ js.Value, args []js.Value) any {
		p.b.sync(func() {
			if !p.done {
				fn(args[0])
			}
		})
		return nil
	})
	p.funcs = append(p.funcs, f)
	el.Call("addEventListener", event, f)
}

// accelText writes an accelerator as menus show it: with the symbols of
// macOS on a Mac.
func accelText(a string, mac bool) string {
	if a == "" {
		return ""
	}
	parts := strings.Split(a, "+")
	var b strings.Builder
	for i, part := range parts {
		last := i == len(parts)-1
		switch strings.ToLower(part) {
		case "cmdorctrl", "commandorcontrol":
			if mac {
				b.WriteString("⌘")
			} else {
				b.WriteString("Ctrl+")
			}
		case "cmd", "command", "super", "meta":
			if mac {
				b.WriteString("⌘")
			} else {
				b.WriteString("Win+")
			}
		case "ctrl", "control":
			if mac {
				b.WriteString("⌃")
			} else {
				b.WriteString("Ctrl+")
			}
		case "alt", "option":
			if mac {
				b.WriteString("⌥")
			} else {
				b.WriteString("Alt+")
			}
		case "shift":
			if mac {
				b.WriteString("⇧")
			} else {
				b.WriteString("Shift+")
			}
		default:
			if last {
				b.WriteString(strings.ToUpper(part[:1]) + part[1:])
			}
		}
	}
	return b.String()
}
