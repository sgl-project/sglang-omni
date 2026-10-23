package main

// The answer card's view (overlay_card.go): its header with the actions,
// the language capsule, the waveform of a follow-up, the text or a
// conversation's bubbles, and the line under them.

import (
	"math"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// overlayCard is the answer card's content, in the face.
func (a *app) overlayCard(c *ui.Context) {
	v := a.cardNow()
	msgH := 0
	if v.msg != "" {
		msgH = layout.CardMsgGap + layout.CardMsgH
	}
	bodyH := float32(max(a.ov.h-layout.CardChrome-msgH, 0))
	// The border takes layout.Border of the padding (MyGo's border box).
	ui.Column(c).Fill().Padding(layout.CardPadTop-layout.Border, layout.CardPadX-layout.Border, layout.CardPadY-layout.Border).Children(func() {
		ui.Row(c).Height(layout.CardHeadH).Shrink(0).Gap(layout.CardHeadGap).AlignItems(ui.Center).Children(func() {
			a.cardHeader(c, v)
		})
		ui.Box(c).Height(1).Shrink(0).Margin(layout.CardRuleTop, 0, 0, 0).Background(ovCardRule)
		a.cardBody(c, v, bodyH)
		if v.msg != "" {
			ui.Row(c).Height(layout.CardMsgH).Shrink(0).Margin(layout.CardMsgGap, 0, 0, 0).Gap(6).AlignItems(ui.Center).Label(v.msg).Children(func() {
				ic, col := icons.InfoCircle, ovText2
				if v.failed != nil {
					ic, col = icons.DangerCircle, tok(c.Theme()).Danger
				}
				ui.Icon(c, ic.Linear).Size(12, 12).TextColor(col).Shrink(0)
				ui.Text(c, v.msg).FontSize(12).TextColor(ovText2).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1).Tooltip(v.msg)
			})
		}
	})
}

// cardHeader is the card's header: what it is, then what it offers.
func (a *app) cardHeader(c *ui.Context, v cardView) {
	// Voxt's: one answer icon for every single result, the microphone over
	// a conversation (its waveform beside it).
	ic := icons.VoxtAnswer
	if v.conversation {
		ic = icons.VoxtTranscribe
	}
	ui.Icon(c, ic.Bold).Size(layout.CardIcon, layout.CardIcon).TextColor(ui.RGB(255, 255, 255)).Opacity(ovCardIconA(v)).Shrink(0).Label(uitext.KindTitle(v.kind))
	if v.conversation {
		// Voxt's: a small waveform in the title's place, moving while a
		// follow-up records.
		cardWave(c, v, a.overlayMotion(), a.overlayPhase())
		ui.Spacer(c)
	} else {
		ui.Text(c, v.title).FontSize(layout.CardTitleSize).FontWeight(600).TextColor(ui.RGB(255, 255, 255)).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
		if v.busy {
			// Voxt's spinner: an arc turning once in 0.9 s, 88 %.
			ui.Icon(c, icons.VoxtSpinner.Linear).Size(12, 12).TextColor(ui.RGBA(255, 255, 255, 0.88)).Shrink(0).
				Rotate(float32(math.Mod(a.overlayPhase()*400, 360))).Label(i18n.T("Translating…"))
		}
		ui.Box(c).Grow(1).MinWidth(0) // Voxt's Spacer(minLength: 12): the gap
	}
	if v.kind == store.KindTranslate && v.failed == nil {
		a.cardLanguage(c, v)
	}
	switch {
	case v.canStop:
		if cardCapsule(c.Key("ov-card-stop"), i18n.T("Done"), i18n.T("Stop recording the follow-up")) {
			a.closeLangMenu()
			a.stopSession()
		}
	case v.canFollowUp:
		tip := i18n.T("Follow up by voice")
		if v.follow != "" {
			tip = i18n.T("Follow up by voice (%s)", v.follow)
		}
		if cardCapsule(c.Key("ov-card-continue"), i18n.T("Continue"), tip) {
			a.closeLangMenu()
			a.followUpAnswer("")
		}
	case v.canRetry:
		if cardCapsule(c.Key("ov-card-retry"), i18n.T("Try again"), "") {
			a.closeLangMenu()
			a.retryAnswer("")
		}
	}
	if v.failed == nil {
		tip := i18n.T("Insert into %s", cmpOr(v.target, i18n.T("the app")))
		if v.why != "" {
			tip = v.why
		}
		if cardIconButton(c.Key("ov-card-insert"), i18n.T("Insert"), tip, icons.Pen, v.canInsert) {
			a.closeLangMenu()
			a.insertAnswer()
		}
		copyIcon, copyLabel := icons.Copy, i18n.T("Copy")
		if v.copied {
			copyIcon, copyLabel = icons.CopyCheck, i18n.T("Copied")
		}
		if cardIconButton(c.Key("ov-card-copy"), copyLabel, copyLabel, copyIcon, v.canCopy) {
			a.closeLangMenu()
			a.copyAnswer(c.Services().WriteClipboard)
		}
	}
	if cardIconButton(c.Key("ov-card-close"), i18n.T("Close"), "", icons.Close, true) {
		a.closeAnswer()
	}
}

// ovCardIconA is how opaque the card's icon is: Voxt's microphone over a
// conversation at 92 %, its answer icon whole (its bubble is dim itself).
func ovCardIconA(v cardView) float32 {
	if v.conversation {
		return 0.92
	}
	return 1
}

// cardSpinning says whether the card shows a spinner (it turns).
func (v cardView) spinning() bool { return !v.conversation && v.busy }

// cardIconButton is one of the header's round buttons (Voxt's
// AnswerHeaderActionButton): 24, white 8 % (16 % and a hairline under
// the pointer); dim when it cannot. It reports a click.
func cardIconButton(c *ui.Context, label, tip string, icon icons.Icon, enabled bool) bool {
	b := ui.ButtonBase(c).Size(layout.CardBtn, layout.CardBtn).Radius(layout.CardBtn / 2).Center().Shrink(0).Label(label).Disabled(!enabled)
	if tip != "" {
		b.Tooltip(tip)
	}
	hover := enabled && b.Hovered()
	bg, line := ui.RGBA(255, 255, 255, 0.08), ui.RGBA(255, 255, 255, 0)
	switch {
	case !enabled:
		bg = ui.RGBA(255, 255, 255, 0.04)
	case hover:
		bg, line = ui.RGBA(255, 255, 255, 0.16), ui.RGBA(255, 255, 255, 0.18)
	}
	b.Background(bg).Border(1, line)
	alpha := float32(0.92)
	if !enabled {
		alpha = 0.42 * 0.92
	}
	return b.Children(func() {
		ui.Icon(c, icon.Linear).Size(layout.CardBtnIcon, layout.CardBtnIcon).TextColor(ui.RGBA(255, 255, 255, alpha))
	}).Clicked()
}

// cardCapsule is the header's text button (Voxt's AnswerContinueButton):
// 24 high, 11 semibold. It reports a click.
func cardCapsule(c *ui.Context, label, tip string) bool {
	b := ui.ButtonBase(c).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Center().Shrink(0).Label(label)
	if tip != "" {
		b.Tooltip(tip)
	}
	bg, line := ui.RGBA(255, 255, 255, 0.08), ui.RGBA(255, 255, 255, 0.1)
	if b.Hovered() {
		bg, line = ui.RGBA(255, 255, 255, 0.16), ui.RGBA(255, 255, 255, 0.18)
	}
	b.Background(bg).Border(1, line)
	return b.Children(func() {
		ui.Text(c, label).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.94)).SingleLine()
	}).Clicked()
}

// cardLanguage is Translate's target in the card (Voxt's capsule): its
// name and a chevron; a click opens our own menu of the languages
// (overlay_lang.go), where another translates the card's text again,
// into the card.
func (a *app) cardLanguage(c *ui.Context, v cardView) {
	open := a.answer.langOpen && v.canLang
	b := ui.ButtonBase(c.Key("ov-card-lang")).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Shrink(0).
		Label(i18n.T("Translate into")).Disabled(!v.canLang).Checked(open)
	line, bg := ui.RGBA(255, 255, 255, 0.12), ui.RGBA(255, 255, 255, 0.08)
	switch {
	case open:
		line, bg = tok(c.Theme()).Accent.Alpha(0.5), ui.RGBA(255, 255, 255, 0.14)
	case v.canLang && b.Hovered():
		line = tok(c.Theme()).Accent.Alpha(0.28)
	}
	b.Background(bg).Border(1, line)
	alpha := float32(1)
	if !v.canLang && !v.busy {
		alpha = 0.6
	}
	chevron := icons.AltArrowDown
	if open && a.prefs.Overlay.Position != "top" {
		chevron = icons.AltArrowUp // it opens upward
	}
	defer func() {
		if b.Clicked() {
			a.toggleLangMenu()
		}
	}()
	b.Children(func() {
		ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Text(c, uitext.TargetLanguageLabel(v.lang)).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.92*alpha)).
				SingleLine().Ellipsis("…").MaxWidth(layout.CardLangMaxW).MinWidth(0)
			ui.Icon(c, chevron.Bold).Size(10, 10).TextColor(ui.RGBA(255, 255, 255, 0.72*alpha)).Shrink(0)
		})
	})
}

// cardWave is a conversation's waveform (Voxt's AnswerConversationWaveView:
// its 16 bars at 0.84 × 0.82), 96 wide: the levels while a follow-up
// records, a quiet pattern otherwise.
func cardWave(c *ui.Context, v cardView, motion bool, phase float64) {
	const sx, sy = 0.84, 0.82
	bars := layout.WaveBars(v.levels)
	quiet := [...]float32{3.2, 3.9, 4.6, 5.1, 4.2, 3.5, 4.4, 4.9}
	label := i18n.T("Input level")
	ui.Row(c.Key("ov-card-wave")).Width(layout.CardWaveW).Height(layout.CardIcon).Shrink(0).Gap(layout.WaveBarGap * sx).AlignItems(ui.Center).Label(label).Children(func() {
		for i, lv := range bars {
			h := quiet[i%len(quiet)] * sy
			if v.recording && v.heard {
				h = (layout.WaveBarMinH + (layout.WaveBarMaxH-layout.WaveBarMinH)*float32(lv)) * sy
				if motion {
					h += float32(layout.WaveIdle(i, lv, phase)) * sy
				}
			}
			ui.Box(c).Width(layout.WaveBarW*sx).Height(h).Radius(layout.WaveBarR).Gradient(ui.RGBA(255, 255, 255, 0.98), ui.RGBA(255, 255, 255, 0.8), 180).Shrink(0)
		}
	})
}

// cardBody is the card's text, bodyH high: it scrolls past that. A
// conversation follows its newest turn while it shows the end.
func (a *app) cardBody(c *ui.Context, v cardView, bodyH float32) {
	st := &a.ov.cardScroll
	key := a.answer.id + "/" + a.sess.ID
	if v.conversation {
		key = a.answer.id + "/conversation"
	}
	if key != a.ov.cardScrollKey {
		a.ov.cardScrollKey = key
		st.X, st.Y = 0, 0
		if v.conversation {
			st.Y = math.MaxFloat32
		}
	} else if v.conversation && st.Y >= st.MaxY-2 {
		st.Y = math.MaxFloat32
	}
	ui.Scroll(c.Key("ov-card-body")).Height(bodyH).Shrink(0).Margin(layout.CardRuleGap, 0, 0, 0).TrackScroll(st).Children(func() {
		ui.Column(c).Padding(0, layout.CardInset, 0, 0).Children(func() {
			for i, b := range cardBlocks(v) {
				a.cardBlockView(c.Key(i), b)
			}
		})
	})
}

// cardBlockView draws one piece of the card's text.
func (a *app) cardBlockView(c *ui.Context, b cardBlock) {
	switch {
	case b.bubble:
		a.cardBubbleView(c, b)
	case b.problem:
		ui.Row(c).Gap(layout.ProblemGap).AlignItems(ui.Start).Margin(b.gap, 0, 0, 0).Children(func() {
			ui.Icon(c, icons.DangerCircle.Bold).Size(layout.ProblemIcon, layout.ProblemIcon).TextColor(tok(c.Theme()).Danger).Shrink(0)
			ui.Text(c, b.text.text).FontSize(b.text.size).FontWeight(b.text.weight).TextColor(b.color).Grow(1).MinWidth(0)
		})
	default:
		t := ui.Text(c, b.text.text).FontSize(b.text.size).TextColor(b.color).Margin(b.gap, 0, 0, 0)
		if b.text.weight != 0 {
			t.FontWeight(b.text.weight)
		}
		if b.selectable {
			t.Selectable()
		}
	}
}

// cardBubbleView is a conversation's turn (Voxt's
// RewriteConversationBubble): 340 wide, the user's on the right; an
// answer shows Copy under the pointer.
func (a *app) cardBubbleView(c *ui.Context, b cardBlock) {
	justify := ui.Start
	bg, line := ui.RGBA(255, 255, 255, 0.06), ui.RGBA(255, 255, 255, 0.08)
	titleA, textA := float32(0.64), float32(0.92)
	if b.user {
		justify, bg, line, titleA = ui.End, ui.RGBA(255, 255, 255, 0.1), ui.RGBA(255, 255, 255, 0.12), 0.7
	}
	if b.live {
		textA = 0.84
		if !b.user {
			titleA = 0.54
		}
	}
	ui.Row(c).Justify(justify).Margin(b.gap, 0, 0, 0).Children(func() {
		col := ui.Column(c.Key("bubble")).Width(layout.BubbleW).Shrink(0).Padding(layout.BubblePadY-layout.Border, layout.BubblePadX-layout.Border).Gap(layout.BubbleTitleGap).
			Radius(layout.BubbleRadius).Background(bg).Border(1, line)
		if b.user {
			col.AlignItems(ui.End)
		}
		hover := !b.user && !b.live && col.Hovered()
		col.Children(func() {
			if b.title.text != "" {
				ui.Text(c, b.title.text).FontSize(layout.BubbleTitleSize).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, titleA)).SingleLine().Ellipsis("…").MaxWidth(layout.BubbleW - 2*layout.BubblePadX)
			}
			t := ui.Text(c, b.text.text).FontSize(layout.BubbleTextSize).FontWeight(500).TextColor(ui.RGBA(255, 255, 255, textA))
			if b.user {
				t.TextAlign(ui.End)
			}
			if !b.live {
				t.Selectable()
			}
			if hover {
				copied := a.ov.bubbleCopied == b.text.text
				label, ic := i18n.T("Copy"), icons.Copy
				if copied {
					label, ic = i18n.T("Copied"), icons.CopyCheck
				}
				cb := ui.ButtonBase(c.Key("copy")).Absolute().Top(8).Right(8).Height(22).Padding(0, 8-layout.Border).Radius(11).
					Background(ui.RGBA(0, 0, 0, 0.74)).Border(1, ui.RGBA(255, 255, 255, 0.12)).Label(label)
				if cb.Children(func() {
					ui.Row(c).Gap(4).AlignItems(ui.Center).Children(func() {
						ui.Icon(c, ic.Linear).Size(12, 12).TextColor(ui.RGBA(255, 255, 255, 0.94))
						ui.Text(c, label).FontSize(10).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.94)).SingleLine()
					})
				}).Clicked() {
					a.copyBubble(c.Services().WriteClipboard, b)
				}
			}
		})
	})
}

// bubbleMark is how long a bubble's Copy shows its mark (Voxt's).
const bubbleMark = 1200 * time.Millisecond

// copyBubble copies an answer's bubble: its title, a blank line, its
// text (Voxt's).
func (a *app) copyBubble(copyText func(string), b cardBlock) {
	text := b.text.text
	if b.title.text != "" {
		text = b.title.text + "\n\n" + text
	}
	copyText(text)
	a.ov.bubbleCopied = b.text.text
	a.ov.bubbleSeq++
	seq := a.ov.bubbleSeq
	a.later(bubbleMark, func() {
		if a.ov.bubbleSeq == seq {
			a.ov.bubbleCopied = ""
			if a.ov.win != nil {
				a.ov.win.Invalidate()
			}
		}
	})
}
