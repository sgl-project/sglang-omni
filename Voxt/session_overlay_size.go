package main

import (
	"math"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (a *app) overlaySize(mode overlayMode) (width, height int, exact bool) {
	if mode != ovAnswer {
		a.refreshSessionOverlay(time.Now())
	}
	switch mode {
	case ovAnswer:
		return a.cardSize()
	case ovNotice:
		notice := a.ov.projection.view.Notice
		if notice == nil {
			return layout.NoticeW, layout.PillH, true
		}
		textWidth, exact := a.ovTextWidth(noticeText(notice.Text))
		return layout.NoticeWidth(textWidth, notice.Action != actionNone), layout.PillH, exact
	case ovRecording, ovWorking:
		pill := a.ov.projection.view.Pill
		scene, exact := float32(0), true
		if pill.SceneText != "" {
			scene, exact = a.ovTextWidth(sceneText(pill.SceneText))
		}
		row, rowExact := recordingPillRow(pill, mode == ovRecording, mode == ovRecording && a.prefs.Overlay.ShowTime, scene, a.ovTextWidth)
		exact = exact && rowExact
		if pill.CanSkip {
			width, ok := a.ovTextWidth(capsuleText(pill.SkipLabel))
			row += layout.Gap + 2*10 + int(math.Ceil(float64(width)))
			exact = exact && ok
		}
		if pill.CanThink {
			width, ok := a.ovTextWidth(capsuleText(pill.ThinkingLabel))
			row += layout.Gap + 2*10 + int(math.Ceil(float64(width)))
			exact = exact && ok
		}
		text, hint := pill.LiveText, pill.LiveHint
		if a.prefs.Overlay.LiveLines {
			if text == "" || hint {
				return layout.LinesW, layout.LinesHeight(0), true
			}
			textHeight, known := a.ovTextHeight(liveLinesKey(text))
			return layout.LinesW, layout.LinesHeight(textHeight), known
		}
		if text == "" {
			return row + 2*layout.PillPadX, layout.PillH, exact
		}
		textWidth, known := a.ovTextWidth(liveTextKey(text))
		width := layout.LiveWidth(row, textWidth)
		if a.ov.live.id == a.sess.ID {
			width = max(width, a.ov.live.w)
		}
		return width, layout.LiveH, exact && known
	}
	return layout.PillW, layout.PillH, true
}

func recordingPillFaceW(snapshot session.Snapshot, showTime bool) int {
	pill := sessionPillViewState{Title: uitext.KindTitle(snapshot.Kind), Starting: snapshot.Waiting != session.WaitNone}
	scene := float32(0)
	if name := uitext.SceneShort(snapshot); name != "" {
		scene = float32(textWidth(name, float64(layout.SceneSize)))
	}
	row, _ := recordingPillRow(pill, true, showTime, scene, func(key ovSized) (float32, bool) {
		return float32(textWidth(key.text, float64(key.size))), true
	})
	return row + 2*layout.PillPadX
}

func faceRadius(overlay settings.Overlay, mode overlayMode, height int) float32 {
	radius := float32(overlay.Radius)
	if mode != ovAnswer {
		radius = min(radius, float32(height)/2)
	}
	return radius
}

func (a *app) ovTextWidth(key ovSized) (float32, bool) {
	measured, ok := a.ovText(key)
	return measured.w, ok
}

func (a *app) ovTextHeight(key ovSized) (float32, bool) {
	measured, ok := a.ovText(key)
	return measured.h, ok
}

func (a *app) ovText(key ovSized) (ovMeasured, bool) {
	if a.measureHook != nil {
		width, height := a.measureHook(key)
		return ovMeasured{width, height}, true
	}
	if measured, ok := a.ov.textW[key]; ok {
		return measured, true
	}
	width, height := guessText(key)
	return ovMeasured{width, height}, false
}

func (a *app) postSessionOverlayMeasurements(measurements []sessionOverlayMeasurement) {
	if len(measurements) == 0 {
		return
	}
	a.post(func() {
		if a.ov.textW == nil || len(a.ov.textW) > 256 {
			a.ov.textW = map[ovSized]ovMeasured{}
		}
		for _, measurement := range measurements {
			a.ov.textW[measurement.Key] = ovMeasured{measurement.Width, measurement.Height}
		}
		a.overlayRestyle()
	})
}

var ovTestArea = mygo.Rectangle{X: 0, Y: 25, Width: 1440, Height: 875}

func (a *app) overlayArea() mygo.Rectangle {
	if !a.overlayNative() {
		return ovTestArea
	}
	return mygo.Screen.DisplayNearestPoint(mygo.Screen.CursorScreenPoint()).WorkArea
}

const streamGrowStep = 4 * 19

func (a *app) blockHeight(card *sessionCardViewState, block sessionCardBlock) (float32, bool) {
	text, title, quote := sessionCardBlockText(card, block)
	height, exact := a.ovTextHeight(text)
	if block.Kind == sessionCardProblem {
		height = max(height, layout.ProblemIcon)
	}
	if block.Kind == sessionCardBubble {
		height += 2 * layout.BubblePadY
		if title.text != "" {
			titleHeight, ok := a.ovTextHeight(title)
			height, exact = height+titleHeight+layout.BubbleTitleGap, exact && ok
		}
		if quote.text != "" {
			quoteHeight, ok := a.ovTextHeight(quote)
			height, exact = height+quoteGap+quoteHeight, exact && ok
			if block.Cut {
				height += quoteGap + quoteToggleH
			}
		}
	}
	return height, exact
}

func (a *app) cardBodyHeight(card *sessionCardViewState) (int, bool) {
	bodyMax := a.cardBodyMax(card)
	var height float32
	exact := true
	for index := 0; index < sessionCardBlockCount(card); index++ {
		block := sessionCardBlockAt(card, index)
		blockHeight, ok := a.blockHeight(card, block)
		height += sessionCardBlockGap(block, index) + blockHeight
		exact = exact && ok
	}
	if card.Writing || card.Conversation && card.Busy && !card.Recording {
		height = float32(math.Ceil(float64(height/streamGrowStep))) * streamGrowStep
		if reveal := &a.ov.stream; exact {
			reveal.h = height
		} else if reveal.h > 0 {
			height = reveal.h
		}
	}
	return min(int(math.Ceil(float64(height))), bodyMax), exact
}

func (a *app) cardBodyMax(card *sessionCardViewState) int {
	room := a.ov.area.Height - int(math.Round(a.prefs.Overlay.Margin)) - layout.Pad - layout.CardChrome
	if card != nil && card.Message != "" {
		room -= layout.CardMsgGap + layout.CardMsgH
	}
	return max(room, layout.CardBodyMax)
}

func (a *app) cardSize() (width, height int, exact bool) {
	a.refreshSessionOverlay(time.Now())
	card := a.ov.projection.view.Card
	if card == nil {
		return layout.CardSizes[0].W, layout.CardChrome, true
	}
	body, exact := a.cardBodyHeight(card)
	height = layout.CardChrome + body
	if card.Message != "" {
		height += layout.CardMsgGap + layout.CardMsgH
	}
	return layout.CardSizes[card.SizeStep].W, height, exact
}
