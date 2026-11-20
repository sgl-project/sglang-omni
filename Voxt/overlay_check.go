package main

import (
	"strconv"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
)

// ovCheckBackdropContent is the native check's backdrop content.
func ovCheckBackdropContent(dark bool) mygo.Content {
	return ui.View(func(c *ui.Context) { ovCheckBackdrop(c, dark) })
}

// ovCheckBackdrop is the backdrop: a page of text lines, light or dark.
func ovCheckBackdrop(c *ui.Context, dark bool) {
	desk, line := ui.RGB(246, 246, 248), ui.RGB(150, 156, 166)
	if dark {
		desk, line = ui.RGB(30, 31, 36), ui.RGB(120, 128, 140)
	}
	ui.Column(c).Fill().Background(desk).Padding(12, 16).Gap(9).Children(func() {
		for j := range 14 {
			ui.Box(c.Key(strconv.Itoa(j))).Height(6).WidthPercent(45 + 40*float32((j*37)%10)/10).Radius(3).Background(line)
		}
	})
}
