package main

// Answer-card text measurement keys and fixed block spacing. Session logic
// builds the displayed blocks; this file only maps them to their draw metrics.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

var (
	ovCardText = ui.RGBA(255, 255, 255, 0.9)
	ovCardRule = ui.RGBA(255, 255, 255, 0.08)
)

// answerCardUI is the card's view-local menu, quote and thumbnail state.
type answerCardUI struct {
	langOpen, thinkMenu, pillMenu bool
	quotesOpen                    map[int]bool
	thumbs                        []*ui.Bitmap
}

const quoteGap, quoteToggleH, quoteSize = layout.BubbleTitleGap, 16, 12

type sessionOverlayPreviewRequest struct {
	key             string
	text            ovSized
	textColor, link uiColor
	background      uiColor
	blockIndex      int
	streaming       bool
	bodyHeight      float32
}

func sessionCardPreviewKey(sessionID string, card *sessionCardViewState, blockIndex int) string {
	key := card.ID
	if key == "" {
		key = sessionID
	}
	if blockIndex >= 0 {
		key += "/" + strconv.Itoa(blockIndex)
	}
	return key
}

func newSessionOverlayPreviewRequest(context *ui.Context, sessionID string, card *sessionCardViewState, block sessionCardBlock, blockIndex int, textColor uiColor, bodyHeight float32) sessionOverlayPreviewRequest {
	text, _, _ := sessionCardBlockText(card, block)
	background := ui.RGB(36, 36, 40)
	if blockIndex < 0 {
		background = ui.RGB(22, 22, 26)
	}
	return sessionOverlayPreviewRequest{
		key: sessionCardPreviewKey(sessionID, card, blockIndex), text: text,
		textColor: textColor, link: context.Theme().Accent, background: background,
		blockIndex: blockIndex, streaming: block.Live, bodyHeight: bodyHeight,
	}
}

func sessionCardBlockCount(card *sessionCardViewState) int {
	if card == nil {
		return 0
	}
	return len(card.Blocks) + card.LiveCount
}

func sessionCardBlockAt(card *sessionCardViewState, index int) sessionCardBlock {
	if index < len(card.Blocks) {
		return card.Blocks[index]
	}
	return card.Live[index-len(card.Blocks)]
}

func sessionCardBlockText(card *sessionCardViewState, block sessionCardBlock) (text, title, quote ovSized) {
	bodyWidth := float32(layout.CardSizes[card.SizeStep].BodyW())
	switch block.Kind {
	case sessionCardProblem:
		return ovSized{block.Text, layout.CardTextSize, layout.CardTextWeight, bodyWidth - layout.ProblemIcon - layout.ProblemGap}, ovSized{}, ovSized{}
	case sessionCardProblemMore:
		return ovSized{block.Text, 12, 0, bodyWidth}, ovSized{}, ovSized{}
	case sessionCardSource:
		return ovSized{block.Text, 13, 0, bodyWidth}, ovSized{}, ovSized{}
	case sessionCardBubble:
		inner := float32(layout.CardSizes[card.SizeStep].BubbleW() - 2*layout.BubblePadX)
		text = ovSized{block.Text, layout.BubbleTextSize, 500, inner}
		if block.Title != "" {
			title = ovSized{block.Title, layout.BubbleTitleSize, 600, 0}
		}
		if block.Quote != "" {
			quote = ovSized{block.Quote, quoteSize, 0, inner}
		}
		return text, title, quote
	default:
		return ovSized{block.Text, layout.CardTextSize, layout.CardTextWeight, bodyWidth}, ovSized{}, ovSized{}
	}
}

func sessionCardBlockGap(block sessionCardBlock, index int) float32 {
	switch block.Kind {
	case sessionCardBubble:
		if index > 0 {
			return layout.BubbleGap
		}
	case sessionCardProblemMore:
		return 4
	case sessionCardSource:
		return 10
	}
	return 0
}
