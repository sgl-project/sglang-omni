package main

// The overlay's face (overlay.go): the pill's material and row, the live
// line, the group's name, the working and notice texts and the features'
// icons.

import (
	"math"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// Colours of the face; the live text's is Voxt's: white, 85 %.
var (
	ovText      = ui.RGB(245, 245, 247)
	ovText2     = ui.RGBA(235, 235, 245, 0.62)
	ovIcon      = ui.RGBA(255, 255, 255, 0.92)
	ovLiveColor = ui.RGBA(255, 255, 255, ovLiveAlpha)
)

const ovLiveAlpha = 0.85

// faceStyle gives e, the face, its radius and translucent fill, hairline and
// shadow. Give it children.
func faceStyle(e ui.Element, radius, alpha float32) ui.Element {
	return e.Fill().Radius(radius).Background(ui.RGBA(22, 22, 26, alpha)).
		Border(1, ui.RGBA(255, 255, 255, 0.12)).Shadow(0, layout.ShadowY, layout.ShadowBlur, 0, ui.RGBA(0, 0, 0, layout.ShadowA))
}

// pillLook is how the pill draws: with the time, its live text in lines
// (Overlay.LiveLines) and their scroll (nil: a preview's, following the
// end), moving, at phase seconds, w DIPs wide.
type pillLook struct {
	time   bool
	lines  bool
	scroll *linesScroll
	motion bool
	phase  float64
	w      int
}

// overlayPill draws the recording or working pill from state: the feature's
// icon, waveform or loader, optional time and scene, live line, and any skip
// or thinking capsule. Its controls go through actions.
func overlayPill(c *ui.Context, state sessionOverlayViewState, look pillLook, actions sessionOverlayActions) {
	pill := state.Pill
	text, hint := pill.LiveText, pill.LiveHint
	var skip, thinkClicked bool
	// The face's border takes layout.Border of its padding (MyGo's border box).
	col := ui.Column(c).Fill().AlignItems(ui.Center).Justify(ui.Center).Padding(layout.PillPadY-layout.Border, layout.PillPadX-layout.Border)
	if text != "" || look.lines {
		col.Padding(layout.LivePadY-layout.Border, layout.LivePadX-layout.Border).Gap(layout.LiveGap)
	}
	if look.lines {
		col.Justify(ui.Start) // the row stays put as the lines under it grow
	}
	col.Children(func() {
		ui.Row(c).Height(layout.RowH).Shrink(0).Gap(layout.Gap).AlignItems(ui.Center).Children(func() {
			if state.Mode == ovRecording && pill.Starting {
				startingIcon(c, pill, look)
			} else {
				ui.Box(c).Size(layout.IconSlot, layout.RowH).Shrink(0).Center().Children(func() {
					ui.Icon(c, overlayKindIcon(pill.Kind).Bold).Size(layout.IconSize, layout.IconSize).TextColor(ovIcon).Shrink(0).Label(pill.Title)
				})
			}
			if state.Mode == ovRecording && !pill.Starting {
				ui.Text(c, pill.Title).FontSize(12).FontWeight(500).TextColor(ovText).SingleLine().Shrink(0).
					Label(i18n.T("Recording mode: %s", pill.Title))
			}
			if state.Mode == ovRecording {
				waveform(c, pill.Levels[:], pill.Heard, look)
				if look.time {
					ui.Box(c).Size(layout.TimeW, layout.RowH).Shrink(0).Center().Children(func() {
						ui.Text(c, pill.ElapsedText).FontSize(12).Font("monospace").TextColor(ovText2).SingleLine().Shrink(0)
					})
				}
			} else {
				overlayLoader(c, pill.WorkingLabel, look, layout.WaveSlotW)
			}
			if pill.SceneText != "" {
				ui.Text(c, pill.SceneText).FontSize(layout.SceneSize).TextColor(ovText2).SingleLine().Ellipsis("…").MaxWidth(layout.SceneMaxW).MinWidth(0).Shrink(1).
					Label(i18n.T("App enhancement: %s", pill.SceneLabel))
			}
			if pill.CanSkip {
				skip = cardCapsule(c.Key("ov-skip"), pill.SkipLabel, pill.SkipTooltip)
			}
			if pill.CanThink && pill.ThinkingLabel != "" {
				thinkClicked = cardCapsule(c.Key("ov-think"), pill.ThinkingLabel, "")
			}
		})
		switch {
		case hint:
			// What it waits for, centred and dim, until the live text comes.
			ui.Row(c.Key("ov-wait")).Width(float32(max(look.w, layout.PillW) - 2*layout.LivePadX)).Height(layout.LineH).Shrink(0).
				AlignItems(ui.Center).Justify(ui.Center).Clip().Children(func() {
				ui.Text(c, text).FontSize(layout.LiveSize).FontWeight(layout.LiveWeight).TextColor(ovText2).SingleLine().Shrink(0)
			})
		case text != "" && look.lines:
			liveLines(c.Key("ov-lines"), look.scroll, state.SessionID, text)
		case text != "":
			liveLine(c.Key("ov-live-"+state.SessionID), text, float32(max(look.w, layout.PillW)-2*layout.LivePadX), look.motion)
		case look.lines:
			ui.Box(c).Height(layout.LineH).Shrink(0) // the first line's room, before any text
		}
	})
	if skip {
		actions.skipLLM()
	}
	if thinkClicked {
		actions.togglePillThinkMenu()
	}
}

// startingIcon is the feature's icon while the microphone or the speech
// model starts (Voxt's initializing icon): it breathes, and a ring pings
// out of it (layout.MicRing, layout.MicPulse); still without motion.
func startingIcon(c *ui.Context, pill sessionPillViewState, look pillLook) {
	d, ringA := layout.MicRing(look.phase, look.motion)
	iconA, scale := layout.MicPulse(look.phase, look.motion)
	ui.Box(c.Key("ov-starting")).Size(layout.IconSlot, layout.RowH).Shrink(0).Center().Label(pill.WaitingLabel).Children(func() {
		ring := float32(d)
		ui.Box(c).Absolute().Left((layout.IconSlot-ring)/2).Top((layout.RowH-ring)/2).Size(ring, ring).Radius(ring/2).
			Border(1.2, ui.RGBA(255, 255, 255, float32(ringA)))
		ui.Icon(c, overlayKindIcon(pill.Kind).Bold).Size(layout.IconSize, layout.IconSize).TextColor(ui.RGBA(255, 255, 255, float32(iconA))).
			Scale(float32(scale), float32(scale)).Shrink(0)
	})
}

// linesScroll is the scroll of session id's live text in lines: following
// its newest words unless the wheel took it up.
type linesScroll struct {
	id     string
	st     ui.ScrollState
	follow bool
}

// liveLines is session id's live text wrapped at layout.LinesTextW, as a
// text box types it (Overlay.LiveLines): as high as its lines up to
// layout.LinesMax of them, then it scrolls (l, nil for none kept),
// following the newest words unless the wheel took it up to read what
// came before, until it is back at the end (#3644: the lines before the
// last ten were out of reach).
func liveLines(c *ui.Context, l *linesScroll, id, text string) {
	if l == nil {
		l = &linesScroll{}
	}
	if l.id != id {
		l.id, l.st, l.follow = id, ui.ScrollState{}, true
	}
	l.follow = l.st.Y >= l.st.MaxY-1 // as the last frame laid it out: the wheel moved Y since
	if l.follow {
		l.st.Y = math.MaxFloat32 // the end, wherever the new words put it
	}
	ui.Scroll(c).Width(layout.LinesTextW).MaxHeight(layout.LinesMax * layout.LineH).Shrink(0).TrackScroll(&l.st).Label(text).Children(func() {
		ui.Text(c, text).FontSize(layout.LiveSize).FontWeight(layout.LiveWeight).TextColor(ovText).Shrink(0)
	})
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

// overlayKindIcon is a feature's icon in the overlay: Voxt's mode icons
// (10-15: the overlay's icons are Voxt's, the app's pages keep Solar's).
func overlayKindIcon(kind string) icons.Icon {
	switch kind {
	case "translate":
		return icons.VoxtTranslate
	case "rewrite":
		return icons.VoxtRewrite
	}
	return icons.VoxtTranscribe
}

// overlayNotice is a notice: the pill with an icon and one line in place
// of the waveform; with an action, a click on it runs the action.
func overlayNotice(c *ui.Context, notice *sessionNoticeViewState, radius, alpha float32, actions sessionOverlayActions) {
	if notice == nil {
		return
	}
	t := c.Theme()
	content := func() {
		// Centred, as the pill's row: a notice narrower than layout.NoticeW
		// keeps its sides even. The border takes layout.Border of the padding.
		ui.Row(c).Fill().Padding(0, layout.PillPadX-layout.Border).Gap(layout.Gap).AlignItems(ui.Center).Justify(ui.Center).Children(func() {
			icon, color := icons.DangerCircle, tok(t).Danger
			switch notice.Tone {
			case sessionNoticeSuccess:
				icon, color = icons.CheckCircle, tok(t).Success
			case sessionNoticeWarning:
				icon, color = icons.DangerTriangle, tok(t).Warning
			}
			ui.Box(c).Size(layout.IconSlot, layout.RowH).Shrink(0).Center().Children(func() {
				ui.Icon(c, icon.Bold).Size(layout.IconSize, layout.IconSize).TextColor(color).Shrink(0)
			})
			ui.Text(c, notice.Text).FontSize(layout.NoticeSize).TextColor(ovText).SingleLine().Ellipsis("…").Shrink(1).MinWidth(0)
			if notice.ActionLabel != "" {
				ui.Icon(c, icons.AltArrowRight.Linear).Size(14, 14).TextColor(ovText2).Shrink(0).Margin(0, 0, 0, -2)
			}
		})
	}
	if notice.ActionLabel == "" {
		faceStyle(ui.Column(c.Key("ov-face")), radius, alpha).DragWindow().Label(notice.Text).Children(content)
		return
	}
	button := faceStyle(ui.ButtonBase(c.Key("ov-notice")), radius, alpha).Label(notice.Text + " · " + notice.ActionLabel)
	if button.Hovered() {
		button.Background(ui.RGBA(34, 34, 40, min(alpha+0.06, 1)))
	}
	if button.Children(content).Clicked() {
		actions.noticeAction(notice.Action)
	}
}
