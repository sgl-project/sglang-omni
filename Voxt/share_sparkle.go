package main

// Share, at the top of Home and under Looking back: under the pointer it
// grows and the logo's two stars come out at its corner; a click throws a
// handful of stars out of it, and what it opens comes up once they have
// flown (shareSparkleOpenAt), so they are seen before the window covers
// them.

import (
	"math"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	// shareSparkleLength is how long a click's stars fly; what Share opens
	// comes up at shareSparkleOpenAt of it.
	shareSparkleLength = 560 * time.Millisecond
	shareSparkleOpenAt = .4
)

// shareSparkleStars are a click's stars: the direction each flies in
// (degrees, 0 to the right), how far (DIPs) and how large (DIPs), in the
// logo's orange and peach by turns.
var shareSparkleStars = [...][3]float32{
	{-95, 34, 12}, {-45, 46, 8}, {-5, 52, 11}, {35, 40, 7},
	{80, 30, 9}, {140, 42, 8}, {185, 50, 11}, {-140, 40, 9},
}

type shareSparkle struct {
	// clicks counts the clicks, waiting whether one is still to open
	// what Share opens.
	clicks  uint64
	waiting bool
}

// shareSparkleButton is Share with label for assistive technology and
// text on it, at the size of a page header's button (header) or of a
// section title's. It reports true once, as a click's stars have flown
// far enough for what it opens to come up.
func shareSparkleButton(c *ui.Context, label, text string, header bool) bool {
	t := c.Theme()
	k := tok(t)
	icon, size, padY, padX := px(t, 12), px(t, fsCaption), px(t, 4), px(t, 10)
	if header {
		icon, size, padY, padX = labelIcon(t), t.FontSize, t.Space(1.5), t.Space(2.5)
	}
	b := ui.ButtonBase(c).Radius(px(t, radiusControl)).Padding(padY, padX).Gap(t.Space(1.5)).AlignItems(ui.Center).
		Label(label).Shrink(0)
	local := ui.Local(b, "sparkle", func() shareSparkle { return shareSparkle{} })
	if b.Clicked() {
		local.clicks++
		local.waiting = true
	}
	target, ink, mark := float32(0), k.Text2, k.Text2
	if !header {
		ink, mark = k.Text2, k.Text3
	}
	switch {
	case b.Pressed():
		target, ink, mark = 1, k.Text, k.Text
		b.Background(k.Pressed)
	case b.Hovered() || b.FocusVisible():
		target, ink, mark = 1, k.Text, k.Text
		b.Background(k.Hover)
	}
	hot := b.AnimateTo("hot", target, ui.Bouncy)
	shown := min(max(hot, 0), 1)
	fly := float32(1)
	if local.clicks > 0 {
		tl := b.Timeline("burst", ui.TimelineOptions{Run: local.clicks})
		fly = tl.Tween(ui.Tween{From: 0, To: 1, Duration: shareSparkleLength, Ease: ui.Linear})
	}
	open := local.waiting && fly >= shareSparkleOpenAt
	if open {
		local.waiting = false
	}
	b.Scale(1+.12*hot, 1+.12*hot).Children(func() {
		ui.Icon(c, icons.Share.Linear).Size(icon, icon).TextColor(mark)
		ui.Text(c, text).FontSize(size).TextColor(ink).SingleLine()
		// The two stars at the corner while the pointer is on it.
		ui.Icon(c, icons.ShareStar).Absolute().Right(px(t, -3)).Top(px(t, -5)).Size(px(t, 10), px(t, 10)).TextColor(shareOrange).
			Opacity(shown).Scale(hot, hot)
		ui.Icon(c, icons.ShareStar).Absolute().Right(px(t, -9)).Top(px(t, 5)).Size(px(t, 6), px(t, 6)).TextColor(sharePeach).
			Opacity(shown).Scale(hot, hot)
		if fly < 1 {
			shareSparkleBurst(c, t, fly)
		}
	})
	return open
}

// shareSparkleBurst is a click's stars at fly (0 to 1) of their flight:
// out from the button's middle, growing, then shrinking away.
func shareSparkleBurst(c *ui.Context, t *ui.Theme, fly float32) {
	out := 1 - float32(math.Pow(float64(1-fly), 3))
	grow := float32(math.Sin(math.Pi * float64(min(fly*1.15, 1))))
	ui.Box(c.Key("burst")).Absolute().LeftPercent(50).TopPercent(50).Size(0, 0).Children(func() {
		for i, star := range shareSparkleStars {
			a := float64(star[0]) * math.Pi / 180
			d, s := px(t, star[1])*out, px(t, star[2])
			color := shareOrange
			if i%2 == 1 {
				color = sharePeach
			}
			// Wider than tall: Home's Share sits near the window's top.
			ui.Icon(c.Key(i), icons.ShareStar).Absolute().Left(d*float32(math.Cos(a))-s/2).Top(.6*d*float32(math.Sin(a))-s/2).
				Size(s, s).TextColor(color).Scale(grow, grow).Opacity(min(1, 2*(1-fly)))
		}
	})
}
