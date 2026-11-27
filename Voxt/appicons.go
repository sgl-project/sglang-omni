package main

// App-icon rendering. The controller owns icon requests and bitmap decoding;
// this view receives only the in-memory bitmap source and display reference.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type appIconBitmap = ui.Bitmap

var decodeAppIconBitmap = ui.DecodeBitmap

type appEnhancementIconSource interface {
	appEnhancementIcon(ref appEnhancementIconRef, want bool) (*appIconBitmap, bool)
}

// appIcon shows an app's icon, its generic fallback, or blank space while it loads.
func appIcon(c *ui.Context, source appEnhancementIconSource, ref appEnhancementIconRef, size float32, want bool) ui.Element {
	theme := c.Theme()
	bitmap, generic := source.appEnhancementIcon(ref, want)
	if bitmap != nil {
		return ui.Image(c, bitmap).Size(px(theme, size), px(theme, size)).Fit(ui.Contain).Shrink(0)
	}
	box := ui.Box(c).Size(px(theme, size), px(theme, size)).Center().Shrink(0)
	if generic {
		box.Children(func() {
			ui.Icon(c, icons.Widget2.Linear).Size(px(theme, size*0.8), px(theme, size*0.8)).TextColor(tok(theme).Text2)
		})
	}
	return box
}
