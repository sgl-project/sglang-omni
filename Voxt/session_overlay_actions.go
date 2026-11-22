package main

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) copyOverlayText(text string) {
	if a.copyText != nil {
		_ = a.copyText(text)
		return
	}
	_ = mygo.Clipboard.Write(transfer.TextData(text))
}

func (a *app) copyAnswer() {
	a.closeLangMenu()
	if !a.answer.open || a.answer.text == "" {
		return
	}
	a.copyOverlayText(a.answer.text)
	a.setDelivery(a.answer.id, "copy", proto.App{})
	a.answer.copySeq++
	sequence, id := a.answer.copySeq, a.answer.id
	a.answer.copied = true
	a.later(copyMark, func() {
		if a.answer.open && a.answer.copySeq == sequence && a.answer.id == id {
			a.answer.copied = false
			if a.ov.win != nil {
				a.ov.win.Invalidate()
			}
		}
	})
}

func (a *app) cycleOverlayCardSize() {
	card := a.ov.projection.view.Card
	if card == nil {
		return
	}
	a.closeLangMenu()
	next := a.prefs.Features.CardSize(card.Kind) + 1
	if next == settings.CardSizeSteps {
		next = 0
	}
	a.setCardSize(card.Kind, next)
}

func (a *app) toggleQuote(turn int) {
	if a.answer.ui.quotesOpen == nil {
		a.answer.ui.quotesOpen = map[int]bool{}
	}
	a.answer.ui.quotesOpen[turn] = !a.answer.ui.quotesOpen[turn]
	a.ov.projection.cardReady = false
	a.overlaySync()
}

const bubbleMark = 1200 * time.Millisecond

// copyBubble copies the displayed block bytes, including their trimming and
// empty-text ellipsis, rather than reconstructing a raw conversation turn.
func (a *app) copyBubble(block sessionCardBlock) {
	text := block.Text
	if block.Title != "" {
		text = block.Title + "\n\n" + text
	}
	a.copyOverlayText(text)
	a.ov.ui.bubbleCopied = block.Text
	a.ov.ui.bubbleSeq++
	sequence := a.ov.ui.bubbleSeq
	a.later(bubbleMark, func() {
		if a.ov.ui.bubbleSeq == sequence {
			a.ov.ui.bubbleCopied = ""
			if a.ov.win != nil {
				a.ov.win.Invalidate()
			}
		}
	})
}
