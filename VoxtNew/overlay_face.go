package main

// The overlay's face (overlay.go): its view, the pill's material and row,
// the live line, the group's name, the working and notice texts and the
// features' icons.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// overlayView is the overlay window's content.
func (a *app) overlayView(c *ui.Context) {
	c.SetTheme(overlayTheme())
	// The root draws the theme's background unless told not to: in this
	// transparent window only the face may show. Last, once the view is
	// built: the kit's ink buttons swap the theme while they build
	// (AudioSmith's pill).
	defer c.Root().Background(ui.Transparent)
	a.overlayMeasure(c)
	if a.ov.reveal {
		// This frame draws the new face: opaque from the next one.
		a.ov.reveal = false
		a.post(a.revealOverlay)
	}
	motion := a.overlayMotion()
	if !motion {
		c.DisableAnimations()
	}
	switch {
	case motion && (a.ov.mode == ovRecording || a.ov.mode == ovWorking || a.ov.mode == ovAnswer && (a.cardRecording() || a.cardNow().spinning())):
		c.After(layout.Tick)
	case a.ov.mode == ovRecording && a.prefs.Overlay.ShowTime:
		c.After(250 * time.Millisecond) // the time
	}
	look := pillLook{time: a.prefs.Overlay.ShowTime, motion: motion, phase: a.overlayPhase(), w: a.ov.w}
	ui.Column(c).Fill().Padding(layout.Pad).Children(func() {
		o := a.prefs.Overlay
		switch a.ov.mode {
		case ovAnswer:
			// Voxt's card does not move (isMovableByWindowBackground off):
			// its text is selectable.
			faceStyle(ui.Column(c.Key("ov-face")), o, ovAnswer, a.ov.h).Children(func() { a.overlayCard(c) })
		case ovNotice:
			a.overlayNotice(c)
		default:
			// The pointer goes through it (passThrough): nothing to drag.
			faceStyle(ui.Column(c.Key("ov-face")), o, a.ov.mode, a.ov.h).Children(func() {
				overlayPill(c, a.pillSession(), a.ov.mode, look) // mute_link.go: the output's note
			})
		}
	})
}

// Colours of the face; the live text's is Voxt's: white, 85 %.
var (
	ovText      = ui.RGB(245, 245, 247)
	ovText2     = ui.RGBA(235, 235, 245, 0.62)
	ovIcon      = ui.RGBA(255, 255, 255, 0.92)
	ovLiveColor = ui.RGBA(255, 255, 255, ovLiveAlpha)
)

const ovLiveAlpha = 0.85

// faceRadius is the face's corner radius in mode, h DIPs high: the
// setting's, at most a capsule for the pill.
func faceRadius(o settings.Overlay, mode overlayMode, h int) float32 {
	radius := float32(o.Radius)
	if mode != ovAnswer {
		radius = min(radius, float32(h)/2)
	}
	return radius
}

// faceStyle gives e, the face, its look in mode, h DIPs high: the radius,
// the translucent fill, the hairline and the shadow. Give it children.
func faceStyle(e ui.Element, o settings.Overlay, mode overlayMode, h int) ui.Element {
	return e.Fill().Radius(faceRadius(o, mode, h)).Background(ui.RGBA(22, 22, 26, layout.FaceAlpha(o))).
		Border(1, ui.RGBA(255, 255, 255, 0.12)).Shadow(0, layout.ShadowY, layout.ShadowBlur, 0, ui.RGBA(0, 0, 0, layout.ShadowA))
}

// pillLook is how the pill draws: with the time, moving, at phase
// seconds, w DIPs wide.
type pillLook struct {
	time   bool
	motion bool
	phase  float64
	w      int
}

// overlayPill is the recording and working pill of s in mode, look.w
// wide: the feature's icon, the waveform or the loader, the time and the
// matched group when they show, in a row centred in the pill (Voxt's
// compact card); with live text the row and the text's line under it,
// both centred, in the wider padding of Voxt's card with text.
func overlayPill(c *ui.Context, s session.Snapshot, mode overlayMode, look pillLook) {
	text, hint := pillLine(s)
	// The face's border takes layout.Border of its padding (MyGo's border box).
	col := ui.Column(c).Fill().AlignItems(ui.Center).Justify(ui.Center).Padding(layout.PillPadY-layout.Border, layout.PillPadX-layout.Border)
	if text != "" {
		col.Padding(layout.LivePadY-layout.Border, layout.LivePadX-layout.Border).Gap(layout.LiveGap)
	}
	col.Children(func() {
		ui.Row(c).Height(layout.RowH).Shrink(0).Gap(layout.Gap).AlignItems(ui.Center).Children(func() {
			if mode == ovRecording && s.Waiting != session.WaitNone {
				startingIcon(c, s, look)
			} else {
				ui.Box(c).Size(layout.IconSlot, layout.RowH).Shrink(0).Center().Children(func() {
					ui.Icon(c, overlayKindIcon(s.Kind).Bold).Size(layout.IconSize, layout.IconSize).TextColor(ovIcon).Shrink(0).Label(uitext.KindTitle(s.Kind))
				})
			}
			if mode == ovRecording {
				waveform(c, s.Levels, !s.HeardAt.IsZero(), look)
				if look.time {
					ui.Box(c).Size(layout.TimeW, layout.RowH).Shrink(0).Center().Children(func() {
						ui.Text(c, clock(s.Elapsed(time.Now()))).FontSize(12).Font("monospace").TextColor(ovText2).SingleLine().Shrink(0)
					})
				}
			} else {
				overlayLoader(c, uitext.WorkingText(s), look)
			}
			if name := uitext.SceneShort(s); name != "" {
				ui.Text(c, name).FontSize(layout.SceneSize).TextColor(ovText2).SingleLine().Ellipsis("…").MaxWidth(layout.SceneMaxW).MinWidth(0).Shrink(1).
					Label(i18n.T("App enhancement: %s", uitext.SceneLabel(s)))
			}
		})
		switch {
		case hint:
			// What it waits for, centred and dim, until the live text comes.
			ui.Row(c.Key("ov-wait")).Width(float32(max(look.w, layout.PillW) - 2*layout.LivePadX)).Height(layout.LineH).Shrink(0).
				AlignItems(ui.Center).Justify(ui.Center).Clip().Children(func() {
				ui.Text(c, text).FontSize(layout.LiveSize).FontWeight(layout.LiveWeight).TextColor(ovText2).SingleLine().Shrink(0)
			})
		case text != "":
			liveLine(c.Key("ov-live-"+s.ID), text, float32(max(look.w, layout.PillW)-2*layout.LivePadX), look.motion)
		}
	})
}

// pillLine is the line under the pill's row: the live text, else (hint)
// "Starting the microphone…" while the microphone gives no audio (what
// is said then is lost); "" for none. A speech model that is still
// loading shows only the starting icon (the user, 10-21): the recording
// works meanwhile, so no line is needed.
func pillLine(s session.Snapshot) (text string, hint bool) {
	if text := liveLineText(s); text != "" {
		return text, false
	}
	if s.Waiting == session.WaitMic {
		if w := uitext.WaitText(s); w != "" {
			return w, true
		}
	}
	return "", false
}

// startingIcon is the feature's icon while the microphone or the speech
// model starts (Voxt's initializing icon): it breathes, and a ring pings
// out of it (layout.MicRing, layout.MicPulse); still without motion.
func startingIcon(c *ui.Context, s session.Snapshot, look pillLook) {
	d, ringA := layout.MicRing(look.phase, look.motion)
	iconA, scale := layout.MicPulse(look.phase, look.motion)
	ui.Box(c.Key("ov-starting")).Size(layout.IconSlot, layout.RowH).Shrink(0).Center().Label(uitext.WaitText(s)).Children(func() {
		ring := float32(d)
		ui.Box(c).Absolute().Left((layout.IconSlot-ring)/2).Top((layout.RowH-ring)/2).Size(ring, ring).Radius(ring/2).
			Border(1.2, ui.RGBA(255, 255, 255, float32(ringA)))
		ui.Icon(c, overlayKindIcon(s.Kind).Bold).Size(layout.IconSize, layout.IconSize).TextColor(ui.RGBA(255, 255, 255, float32(iconA))).
			Scale(float32(scale), float32(scale)).Shrink(0)
	})
}

// liveLineText is the live text as its line shows it: its spaces and
// line breaks one space each.
func liveLineText(s session.Snapshot) string {
	return strings.Join(strings.Fields(liveText(s)), " ")
}

// liveLine is the live text on one line, width DIPs wide: centred while
// it fits, else scrolled so that its newest words show at the right
// edge, sliding left as more come (180 ms, Voxt's 0.2 s ease-out), the
// letters at the left edge fading out to say that more came before. The
// key is the session's: a new one starts where its text is, not sliding.
func liveLine(c *ui.Context, text string, width float32, motion bool) {
	measure := func(t string) float32 { w, _ := c.MeasureText(0, liveTextKey(t).span()); return w }
	box := ui.Row(c).Width(width).Height(layout.LineH).Shrink(0).AlignItems(ui.Center).Clip().Label(text)
	// Centred under the row while it fits; else its end at the right edge.
	target := width - measure(text)
	if target > 0 {
		target /= 2
	}
	x := target
	if motion {
		x = box.AnimateTo("x", target, ui.Animation{Duration: 180 * time.Millisecond, Ease: ui.EaseOut})
	}
	shown, at, spans := text, x, []ui.Span{{Text: text, Size: layout.LiveSize, Weight: layout.LiveWeight, Color: ovLiveColor}}
	if x < 0 {
		shown, at, spans = fadeSpans(text, -x, measure)
		at = x + at
	}
	box.Children(func() {
		ui.RichText(c, spans...).NoWrap().Shrink(0).FixedLineHeight(layout.LineH).Translate(at, 0).Label(shown)
	})
}

// fadeSpans are the part of text that shows when it is scrolled off DIPs
// to the left, measure telling how wide a part of it is: the letters past
// the edge, where the shown part starts (from the text's start), and its
// spans: the letters in the layout.FadeW after the edge fading in from it, the
// rest as ovText.
func fadeSpans(text string, off float32, measure func(string) float32) (string, float32, []ui.Span) {
	r := []rune(text)
	// The first letter that ends past the edge.
	lo, hi := 0, len(r)
	for lo < hi {
		mid := (lo + hi) / 2
		if measure(string(r[:mid+1])) <= off {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	from := measure(string(r[:lo]))
	var spans []ui.Span
	i, start := lo, from-off
	for ; i < len(r) && start < layout.FadeW; i++ {
		end := measure(string(r[:i+1])) - off
		mid := max(0, (start+end)/2)
		a := min(mid/layout.FadeW, 1)
		spans = append(spans, ui.Span{Text: string(r[i]), Size: layout.LiveSize, Weight: layout.LiveWeight, Color: ui.RGBA(255, 255, 255, ovLiveAlpha*(0.1+0.9*a))})
		start = end
	}
	if i < len(r) {
		spans = append(spans, ui.Span{Text: string(r[i:]), Size: layout.LiveSize, Weight: layout.LiveWeight, Color: ovLiveColor})
	}
	return string(r[lo:]), from, spans
}

// liveText is the text the pill shows under it, the same in every flow
// (10-15, third round): the answer as the model writes it; until then
// what it works on: what the speech model heard so far (a file
// service's whole text, while it is cleaned up), else the selection it
// translates.
func liveText(s session.Snapshot) string {
	working := s.State == session.Composing || s.State == session.Delivering
	if d := strings.TrimSpace(s.Draft); d != "" && working {
		return d
	}
	if p := strings.TrimSpace(s.Partial); p != "" || !working {
		return p
	}
	return strings.TrimSpace(s.Selection)
}

// overlayKindIcon is a feature's icon in the overlay: Voxt's mode icons
// (10-15: the overlay's icons are Voxt's, the app's pages keep Solar's).
func overlayKindIcon(k store.Kind) icons.Icon {
	switch k {
	case store.KindTranslate:
		return icons.VoxtTranslate
	case store.KindRewrite:
		return icons.VoxtRewrite
	}
	return icons.VoxtTranscribe
}

// kindIcon is a feature's icon; uitext.KindTitle its name.
func kindIcon(k store.Kind) icons.Icon {
	switch k {
	case store.KindTranslate:
		return translateIcon
	case store.KindRewrite:
		return rewriteIcon
	}
	return transcribeIcon
}

// overlayNotice is a notice: the pill with an icon and one line in place
// of the waveform; with an action, a click on it runs the action.
func (a *app) overlayNotice(c *ui.Context) {
	t := c.Theme()
	n := a.notice
	if n == nil {
		return
	}
	label := noticeActionLabel(n.action)
	content := func() {
		// Centred, as the pill's row: a notice narrower than layout.NoticeW
		// keeps its sides even. The border takes layout.Border of the padding.
		ui.Row(c).Fill().Padding(0, layout.PillPadX-layout.Border).Gap(layout.Gap).AlignItems(ui.Center).Justify(ui.Center).Children(func() {
			ic, col := icons.DangerCircle, tok(t).Danger
			if n.warn {
				ic, col = icons.DangerTriangle, tok(t).Warning
			}
			ui.Box(c).Size(layout.IconSlot, layout.RowH).Shrink(0).Center().Children(func() {
				ui.Icon(c, ic.Bold).Size(layout.IconSize, layout.IconSize).TextColor(col).Shrink(0)
			})
			ui.Text(c, n.text).FontSize(layout.NoticeSize).TextColor(ovText).SingleLine().Ellipsis("…").Shrink(1).MinWidth(0)
			if label != "" {
				ui.Icon(c, icons.AltArrowRight.Linear).Size(14, 14).TextColor(ovText2).Shrink(0).Margin(0, 0, 0, -2)
			}
		})
	}
	o := a.prefs.Overlay
	if label == "" {
		faceStyle(ui.Column(c.Key("ov-face")), o, ovNotice, a.ov.h).DragWindow().Label(n.text).Children(content)
		return
	}
	b := faceStyle(ui.ButtonBase(c.Key("ov-notice")), o, ovNotice, a.ov.h).Label(n.text + " · " + label)
	if b.Hovered() {
		b.Background(ui.RGBA(34, 34, 40, min(layout.FaceAlpha(o)+0.06, 1)))
	}
	if b.Children(content).Clicked() {
		a.noticeAction(n.action)
	}
}
