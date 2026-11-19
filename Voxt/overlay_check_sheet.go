package main

// The overlay check's contact sheet puts the captured screens side by side,
// with each step's name and the first pill's anchor over them.

import (
	"fmt"
	"image"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// ovCheckPicture is a decoded capture and the geometry its sheet cell needs.
type ovCheckPicture struct {
	name   string
	image  image.Image
	region mygo.Rectangle
	anchor mygo.Rectangle
}

// ovCheckSheetImage renders the captured screens side by side, a column a
// backdrop, each with its step's name and the anchor over it: the line
// of the first pill's face's bottom edge and its centre.
func ovCheckSheetImage(columns [][]ovCheckPicture) (image.Image, error) {
	type pic struct {
		shot ovCheckPicture
		bmp  *ui.Bitmap
	}
	var cols [][]pic
	cellW, rows := 0, 0
	for _, col := range columns {
		var ps []pic
		for _, s := range col {
			ps = append(ps, pic{s, ui.NewBitmap(s.image)})
			cellW = max(cellW, s.region.Width)
		}
		rows = max(rows, len(ps))
		cols = append(cols, ps)
	}
	if cellW == 0 {
		return nil, fmt.Errorf("nothing captured")
	}
	cellH := columns[0][0].region.Height
	const labelH, gap = 22, 14
	w := len(cols)*cellW + (len(cols)+1)*gap
	h := rows*(cellH+labelH+gap) + gap
	guide := ui.RGBA(255, 59, 48, 0.85)
	tt := ui.NewTester(func(c *ui.Context) {
		ui.Row(c).Fill().Background(ui.RGB(255, 255, 255)).Padding(gap).Gap(gap).AlignItems(ui.Start).Children(func() {
			for i, col := range cols {
				ui.Column(c.Key(i)).Width(float32(cellW)).Gap(gap).Children(func() {
					for j, p := range col {
						r, an := p.shot.region, p.shot.anchor
						ui.Column(c.Key(j)).Children(func() {
							ui.Text(c, p.shot.name).FontSize(12).Height(labelH).TextColor(ui.RGB(60, 60, 67))
							ui.Box(c).Size(float32(r.Width), float32(r.Height)).Children(func() {
								ui.Image(c, p.bmp).Absolute().Fill().Fit(ui.FillBox)
								bottom := float32(an.Y + an.Height - layout.Pad - r.Y)
								centre := float32(an.X + an.Width/2 - r.X)
								ui.Box(c).Absolute().Left(0).Right(0).Top(bottom).Height(1).Background(guide)
								ui.Box(c).Absolute().Top(0).Bottom(0).Left(centre).Width(1).Background(guide)
							})
						})
					}
				})
			}
		})
	}, w, h)
	tt.SetScale(2)
	for range 3 {
		tt.Frame()
	}
	return tt.Image(), nil
}
