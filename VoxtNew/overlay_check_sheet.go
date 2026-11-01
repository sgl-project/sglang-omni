package main

// The overlay check's pictures (overlay_check.go): the ink each step
// should have, a step's window as an image over its backdrop, and the
// contact sheet of all of them.

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/ink"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// overlayInkWant is what the face shows now, for the ink check: the
// waveform's row even on both sides; the text under it, scrolled when it
// is wider than its line. A notice's icon is coloured, not ink: only its
// size is checked.
func (a *app) overlayInkWant() ink.Want {
	w := ink.Want{W: a.ov.w, H: a.ov.h}
	if a.ov.mode == ovAnswer {
		return cardInkWant(w, a.cardNow())
	}
	if a.ov.mode != ovRecording && a.ov.mode != ovWorking {
		return w
	}
	// The loader's dots spend most of their time dim (Voxt's: 15 % for
	// 40 % of a turn): caught there, the row is the icon alone. On the Mac
	// (it moves) a working pill's size is checked; TestOverlayInk checks
	// its row still.
	w.Row = a.ov.mode == ovRecording
	w.Even = a.ov.mode == ovRecording && !a.prefs.Overlay.ShowTime && uitext.SceneShort(a.sess) == ""
	if text := liveLineText(a.sess); text != "" {
		tw, _ := a.ovTextWidth(liveTextKey(text))
		w.Text, w.Scrolled = true, tw > layout.LiveTextW
	}
	return w
}

// cardInkWant is what the answer card v should show: its ink from its
// left padding down to its text's last line (a text's line, 14 at most
// 22 high, and the small line under it when there is one, whose ink may
// be too dim to count), or to its bottom padding when it scrolls.
func cardInkWant(w ink.Want, v cardView) ink.Want {
	w.Card, w.Gap = true, 8
	if !v.conversation {
		w.Gap = 22
	} else {
		// A bubble's fill is not ink: its padding, and the room under its
		// last line's glyphs (more with Core Text's lines: 35.5 on the
		// Mac, 10-15, for Latin text without descenders).
		w.Gap = layout.BubblePadY + 12
	}
	if v.msg != "" {
		w.Gap += layout.CardMsgGap + layout.CardMsgH
	}
	return w
}

// ovCheckImage decodes a capture.
func ovCheckImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}

// ovCheckBackdrop is the backdrop: a page of text lines, light or dark.
func ovCheckBackdrop(c *ui.Context, dark bool) {
	desk, line := ui.RGB(246, 246, 248), ui.RGB(150, 156, 166)
	if dark {
		desk, line = ui.RGB(30, 31, 36), ui.RGB(120, 128, 140)
	}
	ui.Column(c).Fill().Background(desk).Padding(12, 16).Gap(9).Children(func() {
		for j := range 14 {
			ui.Box(c.Key(itoa(j))).Height(6).WidthPercent(45 + 40*float32((j*37)%10)/10).Radius(3).Background(line)
		}
	})
}

// ovCheckSheet renders the captured screens side by side, a column a
// backdrop, each with its step's name and the anchor over it: the line
// of the first pill's face's bottom edge and its centre.
func ovCheckSheet(path string, columns [][]ovCheckShot) error {
	type pic struct {
		shot ovCheckShot
		bmp  *ui.Bitmap
	}
	var cols [][]pic
	cellW, rows := 0, 0
	for _, col := range columns {
		var ps []pic
		for _, s := range col {
			data, err := os.ReadFile(s.file)
			if err != nil {
				return err
			}
			b, err := ui.DecodeBitmap(data)
			if err != nil {
				return err
			}
			ps = append(ps, pic{s, b})
			cellW = max(cellW, s.region.Width)
		}
		rows = max(rows, len(ps))
		cols = append(cols, ps)
	}
	if cellW == 0 {
		return fmt.Errorf("nothing captured")
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
	var b bytes.Buffer
	if err := png.Encode(&b, tt.Image()); err != nil {
		return err
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}
