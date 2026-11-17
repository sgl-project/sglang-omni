//go:build darwin

package darwin

import "github.com/egoist/mygo/internal/platform"

// The Dock tile shows progress (Window.SetProgressBar) under the app icon.
// NSProgressIndicator does not draw in a Dock tile, which renders its
// content view offscreen, so the bar is made of two custom NSBoxes, which
// draw themselves.
type dockProgress struct {
	view, icon, track, fill id
}

func (b *Backend) setDockProgress(state string, value float64) {
	withPool(func() { b.updateDockProgress(state, value) })
}

func (b *Backend) updateDockProgress(state string, value float64) {
	tile := send(b.app, "dockTile")
	p := &b.dockProgress
	if state == "" {
		if p.view != 0 {
			send(tile, "setContentView:", 0)
			for _, v := range []id{p.view, p.icon, p.track, p.fill} {
				release(v)
			}
			*p = dockProgress{}
			send(tile, "display")
		}
		return
	}
	size := msgSize(tile, sel("size"))
	const height, inset = 14.0, 6.0
	if p.view == 0 {
		frame := NSRect{Size: size}
		p.view = alloc("NSView")
		msgSetRect(p.view, sel("setFrame:"), frame)
		p.icon = alloc("NSImageView")
		msgSetRect(p.icon, sel("setFrame:"), frame)
		send(p.view, "addSubview:", uintptr(p.icon))
		box := func(fill, border platform.Color, width float64) id {
			v := alloc("NSBox")
			send(v, "setBoxType:", 4)       // NSBoxCustom
			send(v, "setTitlePosition:", 0) // NSNoTitle
			send(v, "setFillColor:", uintptr(nsColor(fill)))
			send(v, "setBorderColor:", uintptr(nsColor(border)))
			msgSetFloat(v, sel("setBorderWidth:"), width)
			msgSetFloat(v, sel("setCornerRadius:"), height/2)
			send(p.view, "addSubview:", uintptr(v))
			return v
		}
		p.track = box(platform.Color{R: 40, G: 40, B: 40, A: 200}, platform.Color{R: 160, G: 160, B: 160, A: 255}, 1.5)
		p.fill = box(platform.Color{R: 51, G: 153, B: 255, A: 255}, platform.Color{R: 51, G: 153, B: 255, A: 255}, 0)
		send(tile, "setContentView:", uintptr(p.view))
	}
	send(p.icon, "setImage:", uintptr(send(b.app, "applicationIconImage")))
	track := NSRect{Origin: NSPoint{inset, inset}, Size: NSSize{size.Width - 2*inset, height}}
	msgSetRect(p.track, sel("setFrame:"), track)
	if state == "indeterminate" {
		value = 1
	}
	fill := track
	fill.Size.Width = max(height, track.Size.Width*value)
	send(p.fill, "setHidden:", boolArg(value <= 0))
	msgSetRect(p.fill, sel("setFrame:"), fill)
	send(tile, "display")
}
