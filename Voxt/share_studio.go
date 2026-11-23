package main

// The hat studio, over the share card, as its mockup draws it: on the
// left a white board the shape of the hat's box, Voxt's head at its foot
// and "My Voxt!" on top, where the pen draws; its corner button shows a
// small Voxt wearing the drawing on the card's background instead, and
// back. On the right the kit, centred: a palette strip whose colors run
// into each other, the brushes in the chosen color, the eraser, undo,
// clear and the wardrobe, then × and the hat that puts it on.

import (
	"image"
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// shareStudioPalette is the palette strip, three rows of ten: pastel,
// vivid, deep; the last column white, grey and ink.
var shareStudioPalette = [3][10]string{
	{"#FFB3B8", "#FFC9A3", "#FFE6A3", "#D9F2A6", "#A8E6CF", "#A9D8F5", "#C3B5F5", "#F5B5DC", "#F0D9C2", "#FFFFFF"},
	{"#E0303F", "#FF7A2F", "#FFC21A", "#8ACB3A", "#1FAE8C", "#2E8BEA", "#7A4FD6", "#E0458B", "#B97A4A", "#B8B5C4"},
	{"#8E1426", "#B8471C", "#B88112", "#4F7F1E", "#0F6E5E", "#1D4E9E", "#46288C", "#8E1F57", "#6B4226", "#1E1B2E"},
}

const (
	studioPad    = 18
	studioCellW  = 16 // a palette cell
	studioCellH  = 15
	studioTileW  = 26 // a wardrobe tile
	studioTileH  = 22
	studioKitPad = 14
)

var (
	shareHeadGuide = ui.Hex("#ECEAF2")
	shareHeadDash  = ui.Hex("#B9B5C8")
	shareTileHead  = ui.RGBA(140, 136, 156, .22)
)

// studioKitColors are the kit's ink, its frosted panel, its solid one
// (the wardrobe's), a hover, a chosen face and its hairlines.
type studioKitColors struct {
	ink, panel, ring, solid, hover, on, line ui.Color
}

func studioColors(dark bool) studioKitColors {
	if dark {
		return studioKitColors{ui.Hex("#F4F2FA"), shareWhite.Alpha(.08), shareWhite.Alpha(.10), ui.Hex("#2B2935"),
			shareWhite.Alpha(.10), shareWhite.Alpha(.18), shareWhite.Alpha(.16)}
	}
	return studioKitColors{ui.Hex("#1E1B2E"), shareWhite.Alpha(.9), ui.RGBA(30, 27, 46, .06), ui.Hex("#FBFAFC"),
		ui.RGBA(30, 27, 46, .06), ui.RGBA(30, 27, 46, .11), ui.RGBA(30, 27, 46, .12)}
}

func shareStudio(c *ui.Context, state *shareCardViewState, actions shareActions) {
	studio := state.Studio
	ui.Box(c.Key("studio")).Absolute().Left(0).Top(0).Size(shareCardW, shareCardH).Children(func() {
		shareBackdrop(c, state.Background, shareCardW, shareCardH)
		ui.Box(c).Absolute().Left(0).Top(0).Size(shareCardW, shareCardH).Background(ui.RGBA(20, 18, 30, .12))
		ui.Row(c.Key("studio-row")).Absolute().Left(0).Top(0).Size(shareCardW, shareCardH).Padding(studioPad).Gap(16).AlignItems(ui.Stretch).Children(func() {
			h := float32(shareCardH - 2*studioPad)
			w := h * shareHatBox[2] / shareHatBox[3]
			studioBoard(c, state, studio, actions, w, h)
			// The kit's inside: the row's width less the board, the gap,
			// its padding and its hairline.
			studioKit(c, state, studio, actions, shareCardW-2*studioPad-16-w-2*studioKitPad-2)
		})
	})
}

// studioBoard is the drawing board w×h DIPs, or the preview in its place.
func studioBoard(c *ui.Context, state *shareCardViewState, studio *shareStudioViewState, actions shareActions, w, h float32) {
	scale := w / shareHatBox[2] // DIPs a unit of the mascot
	board := ui.Box(c.Key("studio-board")).Size(w, h).Radius(14).Clip().Shrink(0)
	pen := ui.Local(board, "pen", func() bool { return false })
	if !studio.Peek {
		board.Background(shareWhite)
		x, y, _ := board.PointerPosition()
		switch {
		case board.Pressed():
			ux, uy := shareBoardPoint(x, y, w, h)
			actions.studioPoint(ux, uy, !*pen)
			*pen = true
		case *pen:
			*pen = false
			actions.studioEnd()
		}
	}
	// "My Voxt!" sits where the mockup's board puts it: 15 units of 270
	// high, its baseline 25 units from the top.
	hintSize := 15 * h / shareHatBox[3]
	hintTop := 25*h/shareHatBox[3] - hintSize*0.85
	board.Children(func() {
		hint := ui.Hex("#A7A3B6")
		if studio.Peek {
			shareBackdrop(c, state.Background, w, h)
			hint = shareWhite.Alpha(.85)
			studioPreview(c, studio, actions, w, h)
		} else {
			studioDrawing(c, studio, actions, w, h, scale)
		}
		ui.Text(c, i18n.T("My Voxt!")).FontSize(hintSize).TextColor(hint).Absolute().Top(hintTop).Left(0).Width(w).TextAlign(ui.Center).SingleLine()
		studioPeekButton(c, studio, actions, w, h)
	})
}

// studioHeadPath is the top of Voxt's head, in the mascot's units mapped
// by at: its sides from y 60 down to the shoulders' start and the arch
// over the top.
func studioHeadPath(path *ui.Path, at func(x, y float32) (float32, float32), closed bool) {
	move := func(x, y float32) { px, py := at(x, y); path.MoveTo(px, py) }
	line := func(x, y float32) { px, py := at(x, y); path.LineTo(px, py) }
	cube := func(x1, y1, x2, y2, x, y float32) {
		a, b := at(x1, y1)
		c2, d := at(x2, y2)
		e, f := at(x, y)
		path.CubeTo(a, b, c2, d, e, f)
	}
	move(75, 60)
	line(75, 128.5)
	cube(75, 57.53, 132.53, 0, 203.5, 0)
	cube(274.47, 0, 332, 57.53, 332, 128.5)
	if closed {
		line(332, 60)
		path.Close()
	}
}

// studioDrawing is the board: the head's guide (filled, its arch dashed),
// what the drawing starts from, and the strokes.
func studioDrawing(c *ui.Context, studio *shareStudioViewState, actions shareActions, w, h, scale float32) {
	at := func(x, y float32) (float32, float32) {
		return (x - shareHatBox[0]) * scale, (y - shareHatBox[1]) * scale
	}
	ui.Box(c.Key("studio-guide")).Absolute().Left(0).Top(0).Size(w, h).Draw(func(p *ui.Painter, r ui.Rect) {
		in := func(x, y float32) (float32, float32) { px, py := at(x, y); return r.X + px, r.Y + py }
		var head, arch ui.Path
		studioHeadPath(&head, in, true)
		p.FillPath(&head, shareHeadGuide)
		studioHeadPath(&arch, in, false)
		// Dashes of 10 units, gaps of 8, along the arch.
		length := studioPathLength(in)
		for from := float32(0); from < length; from += 18 * scale {
			dash := arch
			p.StrokePath(dash.Trim(from/length, min(from+10*scale, length)/length), 3*scale, shareHeadDash)
		}
	})
	studioBase(c, studio.Base, actions, at, scale)
	studioStrokes(c, "studio-ink", studio.Strokes, w, h, at, scale)
}

// studioPathLength is the arch's length, mapped by at, measured along a
// polyline of it.
func studioPathLength(at func(x, y float32) (float32, float32)) float32 {
	var pts [][2]float32
	add := func(x, y float32) { px, py := at(x, y); pts = append(pts, [2]float32{px, py}) }
	bez := func(x0, y0, x1, y1, x2, y2, x3, y3 float32) {
		for i := 1; i <= 24; i++ {
			t := float32(i) / 24
			u := 1 - t
			add(u*u*u*x0+3*u*u*t*x1+3*u*t*t*x2+t*t*t*x3, u*u*u*y0+3*u*u*t*y1+3*u*t*t*y2+t*t*t*y3)
		}
	}
	add(75, 60)
	add(75, 128.5)
	bez(75, 128.5, 75, 57.53, 132.53, 0, 203.5, 0)
	bez(203.5, 0, 274.47, 0, 332, 57.53, 332, 128.5)
	var length float32
	for i := 1; i < len(pts); i++ {
		dx, dy := pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]
		length += float32(math.Hypot(float64(dx), float64(dy)))
	}
	return length
}

// studioBase draws what the drawing starts from: a built-in hat or a
// drawn one.
func studioBase(c *ui.Context, base string, actions shareActions, at func(x, y float32) (float32, float32), scale float32) {
	switch {
	case len(base) > len("outfit:") && base[:len("outfit:")] == "outfit:":
		for _, piece := range mascotPieces(base[len("outfit:"):]) {
			x, y := at(piece.Box[0], piece.Box[1])
			ui.Image(c, piece.SVG).Absolute().Left(x).Top(y).Size(piece.Box[2]*scale, piece.Box[3]*scale)
		}
	case base != "":
		if bm := actions.shareBitmap(base); bm != nil {
			x, y := at(shareHatBox[0], shareHatBox[1])
			ui.Image(c, bm).Absolute().Left(x).Top(y).Size(shareHatBox[2]*scale, shareHatBox[3]*scale)
		}
	}
}

func studioStrokes(c *ui.Context, key string, strokes []shareStroke, w, h float32, at func(x, y float32) (float32, float32), scale float32) {
	ui.Box(c.Key(key)).Absolute().Left(0).Top(0).Size(w, h).Draw(func(p *ui.Painter, r ui.Rect) {
		for _, s := range strokes {
			points := make([][2]float32, len(s.Points))
			for i, pt := range s.Points {
				x, y := at(pt[0], pt[1])
				points[i] = [2]float32{r.X + x, r.Y + y}
			}
			p.StrokePolyline(points, s.Width*scale, ui.Hex(s.Color))
		}
	})
}

// studioPreview is a small Voxt wearing the drawing on the card's own
// background, placed as the mockup places it: the board shows the
// mascot's units -180..590 × -250..380, Voxt at 0.62 of that around its
// microphone's axis.
func studioPreview(c *ui.Context, studio *shareStudioViewState, actions shareActions, w, h float32) {
	k := w / 770
	u := 0.62 * k
	left := (203.5 + (-100-203.5)*0.62 + 180) * k
	top := (-16*0.62 + 250) * k
	ui.Box(c.Key("studio-preview")).Absolute().Left(left).Top(top).Size(600*u, 600*u).Children(func() {
		shareMicrophone(c, false, u)
		at := func(x, y float32) (float32, float32) { return (x + 100) * u, (y + 16) * u }
		studioBase(c, studio.Base, actions, at, u)
		studioStrokes(c, "preview-ink", studio.Strokes, 600*u, 600*u, at, u)
	})
}

// studioPeekButton is the board's corner: the mascot's grey mark to see
// it on Voxt, a white hat to go back to drawing.
func studioPeekButton(c *ui.Context, studio *shareStudioViewState, actions shareActions, w, h float32) {
	label, mark, color, face := i18n.T("See it on Voxt"), icons.MascotSmall, ui.Hex("#8E8B9C"), ui.RGBA(30, 27, 46, .06)
	if studio.Peek {
		label, mark, color, face = i18n.T("Back to drawing"), icons.ShareHat, shareWhite, shareWhite.Alpha(.22)
	}
	b := ui.ButtonBase(c.Key("studio-peek")).Absolute().Left(w-44).Top(h-44).Size(34, 34).Radius(17).
		AlignItems(ui.Center).Justify(ui.Center).Background(face).Label(label).Tooltip(label)
	if b.Children(func() { ui.Icon(c, mark.Linear).Size(21, 21).TextColor(color) }).Clicked() {
		actions.studioPeek()
	}
}

func studioKit(c *ui.Context, state *shareCardViewState, studio *shareStudioViewState, actions shareActions, inner float32) {
	k := studioColors(state.Dark)
	// Three rows: the palette as wide as the kit, the brushes, the tools.
	ui.Column(c.Key("studio-kit")).Grow(1).MinWidth(0).Radius(14).Padding(studioKitPad).Gap(10).AlignItems(ui.Center).
		Background(k.panel).Border(1, k.ring).Children(func() {
		studioPalette(c, studio, actions, inner)
		ui.Row(c.Key("studio-brushes")).Gap(6).AlignItems(ui.Center).Children(func() {
			for i, dot := range []float32{5, 9, 13} {
				label := []string{i18n.T("Thin"), i18n.T("Medium"), i18n.T("Thick")}[i]
				b := studioTool(c.Key(i), k, studio.Brush == i && !studio.Erasing, label)
				if b.Children(func() {
					d := ui.Box(c).Size(dot, dot).Radius(dot / 2).Background(ui.Hex(studio.Color))
					if shareLight(studio.Color) {
						d.Border(1, ui.RGBA(30, 27, 46, .28))
					}
				}).Clicked() {
					actions.studioBrush(i)
				}
			}
		})
		ui.Row(c.Key("studio-tools")).Gap(2).AlignItems(ui.Center).Children(func() {
			tools := []struct {
				key   string
				ic    icons.Icon
				label string
				on    bool
				do    func()
			}{
				{"studio-eraser", icons.ShareEraser, i18n.T("Eraser"), studio.Erasing, actions.studioEraser},
				{"studio-undo", icons.ShareUndo, i18n.T("Undo"), false, actions.studioUndo},
				{"studio-clear", icons.ShareClear, i18n.T("Clear"), false, actions.studioClear},
				{"studio-closet", icons.ShareWardrobe, i18n.T("Wardrobe"), false, func() { actions.studioCloset(true) }},
			}
			for _, tool := range tools {
				if studioTool(c.Key(tool.key), k, tool.on, tool.label).Children(func() {
					ui.Icon(c, tool.ic.Linear).Size(17, 17).TextColor(k.ink)
				}).Clicked() {
					tool.do()
				}
			}
		})
		ui.Spacer(c)
		ui.Row(c.Key("studio-actions")).Gap(12).Children(func() {
			cancel := ui.ButtonBase(c.Key("studio-cancel")).Size(36, 36).Radius(18).Background(k.hover).
				AlignItems(ui.Center).Justify(ui.Center).Label(i18n.T("Cancel")).Tooltip(i18n.T("Cancel"))
			if cancel.Children(func() { ui.Icon(c, icons.ShareClose.Linear).Size(16, 16).TextColor(k.ink) }).Clicked() {
				actions.studioCancel()
			}
			wear := ui.ButtonBase(c.Key("studio-wear")).Size(36, 36).Radius(18).Background(ui.Hex(studio.Color)).
				AlignItems(ui.Center).Justify(ui.Center).Label(i18n.T("Put on")).Tooltip(i18n.T("Put on"))
			mark := shareWhite
			if shareLight(studio.Color) {
				mark = ui.Hex("#1E1B2E")
				wear.Border(1, ui.RGBA(30, 27, 46, .18))
			}
			if wear.Children(func() { ui.Icon(c, icons.ShareHatBold.Linear).Size(20, 20).TextColor(mark) }).Clicked() {
				actions.studioWear()
			}
		})
		if studio.Closet {
			studioCloset(c, studio, actions, k)
		}
	})
}

// shareLight reports whether a "#RRGGBB" color is light enough to want
// dark marks on it.
func shareLight(hex string) bool {
	col := ui.Hex(hex)
	return 0.2126*float32(col.R)+0.7152*float32(col.G)+0.0722*float32(col.B) > 0.72*255
}

// studioPalette is the palette strip (share_paint.go), w DIPs wide, with
// the chosen cell ringed: white inside, ink outside.
func studioPalette(c *ui.Context, studio *shareStudioViewState, actions shareActions, w float32) {
	h := w * (3 * studioCellH) / (10 * studioCellW)
	cw, ch := w/10, h/3
	strip := ui.ButtonBase(c.Key("studio-palette")).Size(w, h).Radius(7).Clip().Label(i18n.T("Colors"))
	x, y, _ := strip.PointerPosition()
	strip.Children(func() {
		ui.Image(c, sharePaletteBitmap()).Absolute().Left(0).Top(0).Size(w, h)
		for row, colors := range shareStudioPalette {
			for i, hex := range colors {
				if hex == studio.Color {
					ui.Box(c.Key("pick")).Absolute().Left(float32(i)*cw).Top(float32(row)*ch).
						Size(cw, ch).Radius(4).Border(1.5, ui.RGBA(30, 27, 46, .85)).
						InsetShadow(0, 0, 0, 2, shareWhite)
				}
			}
		}
	})
	if strip.Clicked() {
		col, row := int(x/cw), int(y/ch)
		if row >= 0 && row < 3 && col >= 0 && col < 10 {
			actions.studioColor(shareStudioPalette[row][col])
		}
	}
}

// studioTool is a kit button 28 DIPs square, its face chosen or hovered.
func studioTool(c *ui.Context, k studioKitColors, chosen bool, label string) ui.Element {
	b := ui.ButtonBase(c).Size(28, 28).Radius(8).AlignItems(ui.Center).Justify(ui.Center).Label(label).Tooltip(label)
	switch {
	case chosen:
		b.Background(k.on)
	case b.Hovered():
		b.Background(k.hover)
	}
	return b
}

// studioCloset is the wardrobe over the kit: the built-in hats, then the
// drawn ones, four tiny tiles to a row; past twenty it scrolls down. A
// click puts one on the board.
func studioCloset(c *ui.Context, studio *shareStudioViewState, actions shareActions, k studioKitColors) {
	ui.Column(c.Key("closet")).Absolute().Left(0).Top(0).Right(0).Bottom(0).Radius(14).Padding(12, 14).Gap(10).
		Background(k.solid).Children(func() {
		ui.Row(c.Key("closet-top")).AlignSelf(ui.Stretch).Children(func() {
			if studioTool(c.Key("closet-back"), k, false, i18n.T("Back")).Children(func() {
				ui.Icon(c, icons.ShareBack.Linear).Size(18, 18).TextColor(k.ink)
			}).Clicked() {
				actions.studioCloset(false)
			}
		})
		ui.Scroll(c.Key("closet-scroll")).Grow(1).MinHeight(0).AlignSelf(ui.Stretch).Children(func() {
			ui.Row(c.Key("closet-grid")).Wrap().Gap(5).Width(4*studioTileW + 3*5).AlignSelf(ui.Center).Children(func() {
				scale := float32(studioTileW-4) / shareHatBox[2]
				at := func(x, y float32) (float32, float32) {
					return 2 + (x-shareHatBox[0])*scale, 2 + (y-shareHatBox[1])*scale
				}
				for _, item := range studio.Wardrobe {
					tile := ui.ButtonBase(c.Key(item)).Size(studioTileW, studioTileH).Radius(5).Background(k.hover).
						Label(i18n.T("Wear this hat")).Clip()
					if tile.Children(func() {
						ui.Box(c).Absolute().Left(0).Top(0).Size(studioTileW, studioTileH).Draw(func(p *ui.Painter, r ui.Rect) {
							var head ui.Path
							studioHeadPath(&head, func(x, y float32) (float32, float32) { px, py := at(x, y); return r.X + px, r.Y + py }, true)
							p.FillPath(&head, shareTileHead)
						})
						studioBase(c, item, actions, at, scale)
					}).Clicked() {
						actions.studioLoad(item)
					}
				}
			})
		})
	})
}

// renderHatPicture is the board's hat at 2×: what a drawn hat is saved as.
func renderHatPicture(base string, strokes []shareStroke, actions shareActions) image.Image {
	return renderTransparent(func(c *ui.Context) { shareHatPicture(c, base, strokes, actions) }, int(shareHatBox[2]), int(shareHatBox[3]), 2)
}

// shareHatPicture is the board's hat alone, at one DIP a unit of the
// mascot: what a drawn hat is saved as.
func shareHatPicture(c *ui.Context, base string, strokes []shareStroke, actions shareActions) {
	w, h := shareHatBox[2], shareHatBox[3]
	at := func(x, y float32) (float32, float32) { return x - shareHatBox[0], y - shareHatBox[1] }
	ui.Box(c.Key("hat-picture")).Size(w, h).Children(func() {
		studioBase(c, base, actions, at, 1)
		studioStrokes(c, "hat-ink", strokes, w, h, at, 1)
	})
}

// shareBoardPoint maps a point on the board, (bx, by) of a board w×h
// DIPs, to the mascot's units.
func shareBoardPoint(bx, by, w, h float32) (float32, float32) {
	return shareHatBox[0] + bx/w*shareHatBox[2], shareHatBox[1] + by/h*shareHatBox[3]
}
