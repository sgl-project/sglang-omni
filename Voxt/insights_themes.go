package main

// The insights cards' looks (insights_cards.go): twelve themes, each a
// painted background or a sheet of colored paper, the colors of what
// sits on it, and a mark large and faint behind each card
// (insights_paint.go). Every background is deep enough for the white
// Voxt and its lockup.

import (
	"math"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// insightsTheme is a background (its tiles are glass when it is Dark),
// the title's color on it, the color of the calendar's squares, and the
// three colors the marks on its tiles run through (the week's ticks, the
// bars, the ring, the dial): deep enough for a white tick on a light
// theme, light enough for a dark one on a dark theme.
type insightsTheme struct {
	shareBackground
	Title, Accent string
	Marks         [3]string
	// Paper makes the background a sheet of paper in the theme's first
	// color (insights_paint.go), its mark layers cut from the same paper.
	Paper bool
}

// insightsThemes are the looks, the first six the cards' own, in their
// order: a painted background and a paper one in turn.
var insightsThemes = [...]insightsTheme{
	{shareBackground{false, "#4F6FE8", "#9A7CF2", "#FFAE86"}, "#FFFFFF", "#6C63E8", [3]string{"#4F6FE8", "#9A7CF2", "#EE8468"}, false}, // iris: Voxt's own
	{shareBackground{false, "#2347C5", "", ""}, "#EEF3FF", "#2F5BE0", [3]string{"#2347C5", "#3F7BEA", "#35B4E4"}, true},                // blue paper
	{shareBackground{false, "#FF9A3D", "#F4623A", "#D63C5E"}, "#FFF6E3", "#F0562F", [3]string{"#FA8A34", "#F4623A", "#D63C5E"}, false}, // ember
	{shareBackground{false, "#355037", "", ""}, "#F6F2DF", "#3E7A46", [3]string{"#2F6B3F", "#4E8F4A", "#9CB84A"}, true},                // green paper
	{shareBackground{true, "#15131D", "#1D1A2B", "#2C2247"}, "#FFFFFF", "#6FE3C4", [3]string{"#6FE3C4", "#6FBDF2", "#B3A4FA"}, false},  // night
	{shareBackground{false, "#3A1A8C", "", ""}, "#F3EEFF", "#6A3FD0", [3]string{"#4A25B0", "#7A4FE0", "#D060C8"}, true},                // violet paper
	{shareBackground{false, "#0B7F78", "#14A08C", "#63CFA4"}, "#EFFFF8", "#0F9C8A", [3]string{"#0B7F78", "#14A08C", "#3DBE8A"}, false}, // lagoon
	{shareBackground{false, "#04646F", "", ""}, "#ECFBFA", "#0B8A94", [3]string{"#04646F", "#0E93A0", "#3DBE8A"}, true},                // teal paper
	{shareBackground{false, "#2456D6", "#3F8AEA", "#7ED3F2"}, "#FFFFFF", "#2F6BE8", [3]string{"#2456D6", "#3F8AEA", "#38B6E6"}, false}, // sky
	{shareBackground{false, "#6E1F2B", "", ""}, "#FFF1E6", "#C2452F", [3]string{"#A8323A", "#D8603A", "#EFA63A"}, true},                // burgundy paper
	{shareBackground{false, "#C93A72", "#E8648C", "#FFB088"}, "#FFF1F4", "#D8477A", [3]string{"#C93A72", "#E8648C", "#F58A62"}, false}, // rose
	{shareBackground{true, "#0F4A40", "#14121F", "#4A2A1C"}, "#FFFFFF", "#F2B56B", [3]string{"#F2B56B", "#F29A6B", "#F28AA0"}, false},  // forest
}

// insightsInks are a theme's colors: the title and the quieter text on
// the background, the text and the muted text on a tile, the calendar's
// color, the marks' three and the color of what is drawn on a mark.
type insightsInks struct {
	Title, Soft, Ink, Muted, Accent, Track, On ui.Color
	Marks                                      [3]ui.Color
	Dark                                       bool
}

func insightsThemeInks(theme int) insightsInks {
	th := insightsThemes[theme%len(insightsThemes)]
	ink, _, muted := shareInk(th.Dark)
	inks := insightsInks{Title: ui.Hex(th.Title), Soft: shareWhite.Alpha(.78), Ink: ink, Muted: muted, Accent: ui.Hex(th.Accent), Dark: th.Dark}
	inks.Track, inks.On = ui.RGBA(30, 27, 46, .09), shareWhite
	if th.Dark {
		inks.Track, inks.On = shareWhite.Alpha(.12), ui.Hex("#1E1B2E")
	}
	for i, mark := range th.Marks {
		inks.Marks[i] = ui.Hex(mark)
	}
	return inks
}

// at is the marks' color t of the way along their gradient (0 to 1).
func (inks insightsInks) at(t float32) ui.Color {
	t = min(max(t, 0), 1)
	if t < .5 {
		return inks.Marks[0].Mix(inks.Marks[1], t*2)
	}
	return inks.Marks[1].Mix(inks.Marks[2], t*2-1)
}

// gradient is the marks' gradient at angle degrees (90: left to right).
func (inks insightsInks) gradient(angle float32) ui.LinearGradient {
	return shareGradient(inks.Marks[0], inks.Marks[1], inks.Marks[2], angle, .5)
}

// insightsTile is a panel on the card: white, or glass on a dark theme.
func insightsTile(c *ui.Context, inks insightsInks) ui.Element {
	e := ui.Column(c).Radius(18)
	if inks.Dark {
		return e.Background(shareWhite.Alpha(.08)).Border(1, shareWhite.Alpha(.12))
	}
	return e.Background(shareWhite.Alpha(.94))
}

// insightsBackdrop is the card's background (its theme's colors, or its
// paper) and, over it, its mark.
func insightsBackdrop(c *ui.Context, face *insightsFaceViewState) {
	th := insightsThemes[face.Theme%len(insightsThemes)]
	var background *ui.Bitmap
	if th.Paper {
		background = insightsPaperBitmap(th.A)
	} else {
		background = sharePaintedBackground(th.shareBackground, insightsCardW, insightsCardH)
	}
	ui.Image(c.Key("insights-bg"), background).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH)
	strength := float32(1)
	if th.Dark {
		strength = .5
	}
	ui.Image(c.Key("insights-mark"), insightsMarkBitmap(face.Kind, th.Paper)).Absolute().Left(0).Top(0).Size(insightsCardW, insightsCardH).Opacity(strength)
}

// insightsProgress is how far an entrance that starts at start and takes
// length has come by elapsed, easing out: 0 before, 1 after.
func insightsProgress(elapsed, start, length time.Duration) float32 {
	progress := min(max(float64(elapsed-start)/float64(length), 0), 1)
	return float32(1 - math.Pow(1-progress, 3))
}

// insightsPop is insightsProgress with a little overshoot at the end, for
// what pops in.
func insightsPop(elapsed, start, length time.Duration) float32 {
	progress := min(max(float64(elapsed-start)/float64(length), 0), 1)
	const back = 1.70158
	shifted := progress - 1
	return float32(1 + (back+1)*shifted*shifted*shifted + back*shifted*shifted)
}

// insightsCount is n as the count toward it shows at progress, with
// thousands separators.
func insightsCount(n int, progress float32) string {
	shown := int(float64(n)*float64(progress) + .5)
	if progress >= 1 {
		shown = n
	}
	digits := []byte{}
	for i := 0; shown > 0 || i == 0; i++ {
		if i > 0 && i%3 == 0 {
			digits = append(digits, ',')
		}
		digits = append(digits, byte('0'+shown%10))
		shown /= 10
	}
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}

// insightsLockup is Voxt's lockup on one line, h DIPs tall: the mascot,
// the wordmark, "by SGL-omni". The mascot makes move as the card comes
// up (elapsed), as Home's does when clicked.
func insightsLockup(c *ui.Context, h float32, move int, elapsed time.Duration) ui.Element {
	return ui.Row(c.Key("insights-lockup")).Height(h).Gap(h * .27).AlignItems(ui.Center).Shrink(0).Children(func() {
		pose := mascotPose{orange: 1, peach: 1}
		if p := float32(elapsed-insightsMascotAt) / float32(mascotLength); p > 0 && p < 1 {
			pose = mascotPoseAt(move, p)
		}
		insightsMascotMark(c, h, pose)
		ui.Image(c, shareWordmark()).Size(h*.8*443/173.0, h*.8)
		ui.Image(c, shareAttribution()).Size(h*.5*777/118.0, h*.5).Margin(0, 0, 0, h*.07)
	})
}

// insightsMascotAt is when, after its card came up, the mascot moves.
const insightsMascotAt = 420 * time.Millisecond

// insightsMascotMark is the mascot as a mark h DIPs tall, in pose: the
// white microphone with its dark eyes and the logo's two stars, without
// a hat.
func insightsMascotMark(c *ui.Context, h float32, pose mascotPose) {
	u := h / 600
	ui.Box(c.Key("insights-mascot")).Size(650*u, h).Shrink(0).Children(func() {
		ui.Box(c.Key("mic")).Absolute().Left(0).Top(0).Size(600*u, 600*u).TransformOrigin(.5, .97).Rotate(pose.tilt).Children(func() {
			shareMicrophone(c, pose.blink, u)
		})
		ui.Icon(c, icons.ShareStar).Absolute().Left(480*u).Top(6*u).Size(130*u, 130*u).TextColor(shareOrange).Scale(pose.orange, pose.orange)
		ui.Icon(c, icons.ShareStar).Absolute().Left(570*u).Top(136*u).Size(80*u, 80*u).TextColor(sharePeach).Scale(pose.peach, pose.peach)
	})
}
