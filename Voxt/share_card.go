package main

// The share card: a picture to send, 488×244 DIPs, the size and type of
// its mockup (deliverables/voxt-share-card) as the user approved it. On the left, Voxt in a hat over its
// wordmark and "by SGL-omni"; on the right, "Voxt and me:" and a sentence
// the user says to it. Its corner offers another background and hat, the
// hat studio and Copy; those buttons, the record controls and the studio
// stay out of the copied picture. It floats over the window, scaled down
// when the window is smaller; a click outside it closes it.

import (
	_ "embed"
	"image"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	shareCardW, shareCardH = 488, 244
	shareCardRadius        = 22
	shareLeftW             = shareCardW * 0.42
	// shareFoot is how far the logo's foot sits from the card's bottom:
	// level with the panel's footer.
	shareFoot = 46
	// shareLockupH is the wordmark (34), the gap (10) and the logo (20).
	shareLockupH = 34 + 10 + 20
)

// shareFigureBox is the mascot's box in its units (its microphone's
// viewBox is -100 -16 600 600): centred on the microphone, wide enough for
// the stars, tall enough for a hat.
var shareFigureBox = [4]float32{-156.5, -250, 720, 850}

var (
	//go:embed assets/share/attribution.png
	shareAttributionPNG []byte
	//go:embed assets/share/voxt-wordmark.png
	shareWordmarkPNG []byte

	shareAttribution = sync.OnceValue(func() *ui.Bitmap { b, _ := ui.DecodeBitmap(shareAttributionPNG); return b })
	shareWordmark    = sync.OnceValue(func() *ui.Bitmap { b, _ := ui.DecodeBitmap(shareWordmarkPNG); return b })
)

// sharePicture is a drawn hat, read by loadHat (share_controller.go)
// with decodeSharePicture.
type sharePicture = ui.Bitmap

var decodeSharePicture = ui.DecodeBitmap

// renderShareCard is the card without its buttons, at 3× (1464×732
// pixels): what Copy copies. Its corners are square: chat apps lay a
// picture's transparent corners on white, and round pictures themselves.
func renderShareCard(state *shareCardViewState, actions shareActions) image.Image {
	return ui.Render(func(c *ui.Context) { shareCardFace(c, state, actions, false) }, shareCardW, shareCardH, 3)
}

var (
	shareWhite  = ui.RGB(255, 255, 255)
	shareEye    = ui.Hex("#211D2E")
	shareOrange = ui.Hex("#FF8A3D")
	sharePeach  = ui.Hex("#FFC9A8")
)

// shareColors are a background's three colors; lifted halfway to white
// on a dark card, for the controls drawn in them (waveform, buttons).
func shareColors(state *shareCardViewState, lifted bool) (a, b, c ui.Color) {
	bg := shareBackgrounds[state.Background]
	a, b, c = ui.Hex(bg.A), ui.Hex(bg.B), ui.Hex(bg.C)
	if lifted && bg.Dark {
		a, b, c = a.Mix(shareWhite, .5), b.Mix(shareWhite, .5), c.Mix(shareWhite, .5)
	}
	return a, b, c
}

func shareGradient(a, b, c ui.Color, angle, mid float32) ui.LinearGradient {
	return ui.LinearGradient{Angle: angle, Stops: []ui.GradientStop{{Color: a, At: 0}, {Color: b, At: mid}, {Color: c, At: 1}}}
}

// shareBackdrop fills a w×h box with background index, as the card
// paints it (share_paint.go).
func shareBackdrop(c *ui.Context, index int, w, h float32) {
	ui.Image(c, shareBackgroundBitmap(index, int(w), int(h))).Absolute().Left(0).Top(0).Size(w, h)
}

// shareFit is how much a w×h card shrinks to fit the window, with a
// margin around it.
func shareFit(c *ui.Context, w, h float32) float32 {
	ww, wh := c.Size()
	if ww <= 0 || wh <= 0 {
		return 1
	}
	return max(.4, min(1, (ww-48)/w, (wh-48)/h))
}

// shareCardDialog is the card floating over the main window.
func shareCardDialog(c *ui.Context, state *shareCardViewState, actions shareActions) {
	open := true
	s := shareFit(c, shareCardW, shareCardH)
	// The card is the dialog: no panel of the theme's around it.
	ui.Modal(c.Key("share-dialog"), &open, func() {
		ui.Box(c.Key("share-fit")).Size(shareCardW*s, shareCardH*s).Children(func() {
			ui.Box(c.Key("share-scaled")).Absolute().Left(0).Top(0).Size(shareCardW, shareCardH).TransformOrigin(0, 0).Scale(s, s).Children(func() {
				shareCardFace(c, state, actions, true)
			})
		})
	}).Padding(0).Background(ui.RGBA(0, 0, 0, 0)).Radius(shareCardRadius * s)
	if !open {
		actions.closeShare()
	}
}

// shareCardFace draws the card; chrome false leaves out its buttons, the
// record controls and the studio (the copied picture).
func shareCardFace(c *ui.Context, state *shareCardViewState, actions shareActions, chrome bool) {
	radius := float32(shareCardRadius)
	if !chrome {
		radius = 0 // the copied picture (renderShareCard)
	}
	ui.Box(c.Key("share-card")).Size(shareCardW, shareCardH).Radius(radius).Clip().Children(func() {
		shareBackdrop(c, state.Background, shareCardW, shareCardH)
		ui.Row(c.Key("share-body")).Absolute().Left(0).Top(0).Size(shareCardW, shareCardH).AlignItems(ui.Stretch).Children(func() {
			shareCardLeft(c, state, actions, chrome)
			shareCardPanel(c, state, actions, chrome)
		})
		if !chrome {
			return
		}
		shareCorner(c, state, actions)
		if state.Studio != nil {
			shareStudio(c, state, actions)
		}
	})
}

// shareCardLeft is Voxt over its lockup, all on one centre line: the
// mascot fills the height the lockup leaves, at most 70% of the column
// wide.
func shareCardLeft(c *ui.Context, state *shareCardViewState, actions shareActions, chrome bool) {
	figureH := float32(shareCardH - 26 - shareFoot - shareLockupH)
	u := min(figureH/shareFigureBox[3], shareLeftW*0.7/shareFigureBox[2])
	ui.Column(c.Key("share-left")).Width(shareLeftW).Shrink(0).AlignItems(ui.Center).Padding(26, 0, shareFoot, 0).Children(func() {
		ui.Column(c.Key("share-figure")).Height(figureH).Justify(ui.Center).AlignItems(ui.Center).Children(func() {
			shareMascot(c, state, actions, u, chrome)
		})
		ui.Image(c, shareWordmark()).Size(34*443/173.0, 34)
		ui.Image(c, shareAttribution()).Size(20*777/118.0, 20).Margin(10, 0, 0, 0)
	})
}

// shareMascot is Voxt, white with dark eyes as in the opening, in the
// card's hat, at u DIPs a unit of it. When a hat goes on (state.Dressed)
// it sways or blinks twice, at random.
func shareMascot(c *ui.Context, state *shareCardViewState, actions shareActions, u float32, animate bool) {
	at := func(x, y float32) (float32, float32) { return (x - shareFigureBox[0]) * u, (y - shareFigureBox[1]) * u }
	box := ui.Box(c.Key("share-mascot")).Size(shareFigureBox[2]*u, shareFigureBox[3]*u).Shrink(0)
	type look struct {
		dressed, runs uint64
		move          int
	}
	local := ui.Local(box, "dress", func() look { return look{dressed: state.Dressed} })
	if animate && local.dressed != state.Dressed {
		local.dressed = state.Dressed
		local.runs++
		local.move = mascotDressMoves[rand.IntN(len(mascotDressMoves))]
	}
	pose := mascotPose{orange: 1, peach: 1}
	if animate && local.runs > 0 {
		tl := box.Timeline("dress", ui.TimelineOptions{Run: local.runs, Delay: 80 * time.Millisecond})
		if p := tl.Tween(ui.Tween{From: 0, To: 1, Duration: mascotLength}); p < 1 {
			pose = mascotPoseAt(local.move, p)
		}
	}
	box.Children(func() {
		mx, my := at(-100, -16)
		ui.Box(c.Key("share-mic")).Absolute().Left(mx).Top(my).Size(600*u, 600*u).TransformOrigin(.5, .97).Rotate(pose.tilt).Children(func() {
			shareMicrophone(c, pose.blink, u)
			shareHat(c, state.Outfit, state.Hat, actions, func(x, y float32) (float32, float32) { return (x + 100) * u, (y + 16) * u }, u)
		})
		x, y := at(380, -10)
		ui.Icon(c, icons.ShareStar).Absolute().Left(x).Top(y).Size(130*u, 130*u).TextColor(shareOrange).Scale(pose.orange, pose.orange)
		x, y = at(470, 120)
		ui.Icon(c, icons.ShareStar).Absolute().Left(x).Top(y).Size(80*u, 80*u).TextColor(sharePeach).Scale(pose.peach, pose.peach)
	})
}

// shareMicrophone is the white microphone over its dark eyes, its box
// 600 units square at u DIPs a unit.
func shareMicrophone(c *ui.Context, blink bool, u float32) {
	for _, x := range []float32{114, 212} { // the eyes, under the microphone's holes
		ui.Box(c).Absolute().Left((x+100)*u).Top((99+16)*u).Size(82*u, 110*u).Radius(41 * u).Background(shareEye)
	}
	microphone := icons.MascotMicrophone
	if blink {
		microphone = icons.MascotMicrophoneBlink
	}
	ui.Icon(c, microphone).Absolute().Left(0).Top(0).Size(600*u, 600*u).TextColor(shareWhite)
}

// shareHat draws a built-in outfit's pieces or a drawn hat's picture,
// placed by at (the mascot's units to the box's DIPs).
func shareHat(c *ui.Context, outfit, hat string, actions shareActions, at func(x, y float32) (float32, float32), u float32) {
	for _, piece := range mascotPieces(outfit) {
		x, y := at(piece.Box[0], piece.Box[1])
		ui.Image(c, piece.SVG).Absolute().Left(x).Top(y).Size(piece.Box[2]*u, piece.Box[3]*u)
	}
	if hat == "" || actions == nil {
		return
	}
	if bm := actions.shareBitmap("hat:" + hat); bm != nil {
		x, y := at(shareHatBox[0], shareHatBox[1])
		ui.Image(c, bm).Absolute().Left(x).Top(y).Size(shareHatBox[2]*u, shareHatBox[3]*u)
	}
}

// shareInk is the panel's text colors: the sentence, the label, and the
// muted footer.
func shareInk(dark bool) (ink, label, muted ui.Color) {
	if dark {
		return ui.Hex("#F4F2FA"), ui.Hex("#C9C3E8"), ui.Hex("#A9A5BC")
	}
	return ui.Hex("#1E1B2E"), ui.Hex("#4A4560"), ui.Hex("#8A879A")
}

// sharePanelStyle is the frosted panel the cards' text sits on.
func sharePanelStyle(e ui.Element, dark bool) ui.Element {
	if dark {
		return e.Background(shareWhite.Alpha(.07)).Border(1, shareWhite.Alpha(.10))
	}
	return e.Background(shareWhite.Alpha(.88)).Border(1, ui.RGBA(30, 27, 46, .06))
}

// shareCardPanel is the right half: "Voxt and me:" and the sentence in
// its phase.
func shareCardPanel(c *ui.Context, state *shareCardViewState, actions shareActions, chrome bool) {
	_, label, muted := shareInk(state.Dark)
	panel := ui.Column(c.Key("share-panel")).Grow(1).MinWidth(0).Margin(22, 22, 22, 0).Radius(16).Padding(24, 26)
	sharePanelStyle(panel, state.Dark).Children(func() {
		ui.Text(c, i18n.T("Voxt and me:")).FontSize(15).FontWeight(600).TextColor(label).SingleLine().Margin(0, 0, 10, 0)
		phase := state.Phase
		if !chrome && phase != sharePhaseSaid {
			phase = sharePhaseEmpty // the picture shows the sentence or the prompt
		}
		switch phase {
		case sharePhaseListening:
			shareListening(c, state, actions)
		case sharePhaseWorking:
			ui.Text(c, "…").FontSize(17).FontWeight(600).TextColor(muted)
		case sharePhaseSaid:
			shareSaid(c, state, actions, chrome)
		default:
			shareEmpty(c, state, actions, chrome)
		}
	})
}

// shareEmpty is the prompt where the sentence will be; a click on it
// records.
func shareEmpty(c *ui.Context, state *shareCardViewState, actions shareActions, chrome bool) {
	hint := ui.RGBA(30, 27, 46, .32)
	hover, hoverInk := ui.RGBA(30, 27, 46, .04), ui.RGBA(30, 27, 46, .6)
	if state.Dark {
		hint = ui.Hex("#F4F2FA").Alpha(.45)
		hover, hoverInk = shareWhite.Alpha(.06), ui.Hex("#F4F2FA").Alpha(.75)
	}
	text := i18n.T("What would you like to say?")
	if !chrome {
		ui.Text(c, text).FontSize(17).FontWeight(600).LineHeight(1.45).TextColor(hint)
		return
	}
	b := ui.ButtonBase(c.Key("share-prompt")).AlignSelf(ui.Start).Radius(8).Padding(4, 8).Margin(-4, -8, 0, -8).
		Label(text).Tooltip(i18n.T("Click, or press a feature's shortcut, to start"))
	color := hint
	if b.Hovered() || b.FocusVisible() {
		b.Background(hover)
		color = hoverInk
	}
	if b.Children(func() { ui.Text(c, text).FontSize(17).FontWeight(600).LineHeight(1.45).TextColor(color) }).Clicked() {
		actions.startShareTry()
	}
	if state.Notice != "" {
		ui.Text(c, state.Notice).FontSize(12).TextColor(ui.Hex("#E0503A")).Margin(8, 0, 0, 0)
	}
}

// shareListening is the waveform in the card's colors over a pulsing dot,
// "Listening", the time and a stop button.
func shareListening(c *ui.Context, state *shareCardViewState, actions shareActions) {
	a, b, cc := shareColors(state, true)
	ink, _, _ := shareInk(state.Dark)
	ui.Column(c.Key("share-listen")).Grow(1).Justify(ui.Center).Gap(18).Children(func() {
		ui.Row(c.Key("share-wave")).Height(46).Gap(4).AlignItems(ui.Center).Children(func() {
			const bars = 22
			levels := state.Levels
			for i := range bars {
				level := 0.0
				if j := len(levels) - bars + i; j >= 0 && j < len(levels) {
					level = levels[j]
				}
				t := float32(i) / (bars - 1)
				color := a.Mix(b, min(t/.55, 1))
				if t > .55 {
					color = b.Mix(cc, (t-.55)/.45)
				}
				ui.Box(c.Key(i)).Size(5, 6+40*float32(min(max(level, 0), 1))).Radius(3).Background(color)
			}
		})
		ui.Row(c.Key("share-listen-row")).Gap(8).AlignItems(ui.Center).Children(func() {
			dot := ui.Box(c.Key("share-rec")).Size(8, 8).Radius(4).Background(ui.Hex("#F0503A"))
			tl := dot.Timeline("pulse", ui.TimelineOptions{Repeat: ui.RepeatForever, Alternate: true})
			dot.Opacity(tl.Tween(ui.Tween{From: 1, To: .3, Duration: 500 * time.Millisecond, Ease: ui.EaseInOut}))
			ui.Text(c, i18n.T("Listening")).FontSize(14).TextColor(ink.Alpha(.6)).SingleLine()
			ui.Text(c, state.Elapsed).FontSize(14).FontWeight(600).TextColor(ink).FontFeatures("tnum").SingleLine()
			ui.Spacer(c)
			if shareOrb(c.Key("share-stop"), state, i18n.T("Done"), func() {
				ui.Box(c).Size(10, 10).Radius(2.5).Background(shareWhite)
			}).Clicked() {
				actions.stopShareTry()
			}
		})
	})
}

// shareSaid is the sentence, three lines at most, over its footer: the
// mascot's mark, its length and its day, and a mic to say it again.
func shareSaid(c *ui.Context, state *shareCardViewState, actions shareActions, chrome bool) {
	ink, _, muted := shareInk(state.Dark)
	ellipsis := "…"
	if i18n.Locale() == "zh-Hans" {
		ellipsis = "……"
	}
	ui.Text(c, state.Text).FontSize(17).FontWeight(600).LineHeight(1.5).TextColor(ink).MaxLines(3).Ellipsis(ellipsis)
	ui.Spacer(c)
	ui.Row(c.Key("share-foot")).Gap(6).AlignItems(ui.Center).Height(30).Children(func() {
		ui.Icon(c, icons.MascotSmall.Linear).Size(15, 15).TextColor(muted.Alpha(.9))
		ui.Text(c, state.Meta).FontSize(12.5).LetterSpacing(.25).TextColor(muted).FontFeatures("tnum").SingleLine().Shrink(0)
		ui.Spacer(c)
		if chrome && shareOrb(c.Key("share-resay"), state, i18n.T("Say it again"), func() {
			ui.Icon(c, icons.ShareMic.Linear).Size(16, 16).TextColor(shareWhite)
		}).Clicked() {
			actions.startShareTry()
		}
	})
}

// shareOrb is a round button filled with the card's own gradient.
func shareOrb(c *ui.Context, state *shareCardViewState, label string, inner func()) ui.Element {
	a, b, cc := shareColors(state, true)
	return ui.ButtonBase(c).Size(30, 30).Radius(15).Shrink(0).Label(label).Tooltip(label).
		LinearGradient(shareGradient(a, b, cc, 135, .6)).AlignItems(ui.Center).Justify(ui.Center).Children(inner)
}

// shareCorner is the card's buttons, top left: another background and
// hat, the hat studio, Copy.
func shareCorner(c *ui.Context, state *shareCardViewState, actions shareActions) {
	ui.Column(c.Key("share-corner")).Absolute().Left(14).Top(14).Gap(8).Children(func() {
		if shareCornerButton(c.Key("share-shuffle"), icons.ShareShuffle, i18n.T("Another background and hat")).Clicked() {
			actions.shuffleShare()
		}
		if shareCornerButton(c.Key("share-studio"), icons.ShareHat, i18n.T("Draw a hat")).Clicked() {
			actions.openStudio()
		}
		if shareCopyButton(c.Key("share-copy"), state.Copied).Clicked() {
			actions.copyShare()
		}
	})
}

// shareCopyButton is Copy, a tick for a moment after it copied.
func shareCopyButton(c *ui.Context, copied bool) ui.Element {
	if copied {
		c.After(400 * time.Millisecond) // the tick goes back to Copy
		return shareCornerButton(c, icons.ShareCopied, i18n.T("Copied"))
	}
	return shareCornerButton(c, icons.ShareCopy, i18n.T("Copy picture"))
}

func shareCornerButton(c *ui.Context, ic icons.Icon, label string) ui.Element {
	b := ui.ButtonBase(c).Size(32, 32).Radius(10).AlignItems(ui.Center).Justify(ui.Center).Label(label).Tooltip(label)
	if b.Hovered() {
		b.Background(shareWhite.Alpha(.28))
	} else {
		b.Background(shareWhite.Alpha(.16))
	}
	return b.Children(func() { ui.Icon(c, ic.Linear).Size(17, 17).TextColor(shareWhite) })
}
