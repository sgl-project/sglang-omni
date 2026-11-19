package main

// The overlay's transcript previews. The view asks through
// sessionOverlayActions; this controller owns the WebViews and their heights.

import (
	"math"

	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func (a *app) overlayTranscriptResult(context *uiContext, request sessionOverlayPreviewRequest) bool {
	if a.ov.win == nil || request.text.text == "" {
		return false
	}
	input := transcriptPreviewInput{
		key: request.key, source: request.text.text, size: request.text.size,
		text: request.textColor, link: request.link, background: request.background,
		dark: true, streaming: request.streaming,
	}
	if request.blockIndex < 0 {
		preview := a.overlayTranscript.place(context, a.ov.win, input, request.bodyHeight, request.bodyHeight)
		if !preview.Valid() {
			return false
		}
		// The page fills the card's body, not only its last measured text:
		// the card grows a line ahead of a streaming answer
		// (streamHeadroom), and a page as high as its text had no room for
		// the next line, so it scrolled to its end and flashed its scroll
		// bar until the card caught up (#3836 feedback).
		preview.Height(request.bodyHeight).Margin(layout.CardRuleGap, layout.CardInset, 0, 0)
		return true
	}
	for len(a.bubbleTranscripts) <= request.blockIndex {
		a.bubbleTranscripts = append(a.bubbleTranscripts, nil)
	}
	preview := a.bubbleTranscripts[request.blockIndex]
	if preview == nil {
		preview = &transcriptPreview{}
		a.bubbleTranscripts[request.blockIndex] = preview
	}
	measured, _ := a.ovText(request.text)
	return preview.place(context, a.ov.win, input, measured.h, math.MaxFloat32).Valid()
}

// Conversation previews keep their identity when a live answer becomes a
// completed turn. Drop pages the current card no longer draws.
func (a *app) trimBubblePreviews(count int) {
	if count >= len(a.bubbleTranscripts) {
		return
	}
	for _, preview := range a.bubbleTranscripts[count:] {
		if preview == nil {
			continue
		}
		preview.closed = true
		if a.ov.win != nil {
			a.ov.win.Update(func() {
				if preview.view != nil && !preview.view.IsDestroyed() {
					preview.view.Destroy()
				}
			})
		}
	}
	a.bubbleTranscripts = a.bubbleTranscripts[:count]
}

func (a *app) sessionOverlayPreviewHeights(state *sessionOverlayViewState) map[string]int {
	card := state.Card
	if card == nil {
		return nil
	}
	var heights map[string]int
	for index := 0; index < sessionCardBlockCount(card); index++ {
		if !card.Conversation && (card.Failed || index > 0) {
			continue
		}
		block := sessionCardBlockAt(card, index)
		if card.Conversation && (block.Kind != sessionCardBubble || block.User) {
			continue
		}
		blockIndex, preview := index, &a.overlayTranscript
		if card.Conversation {
			if index >= len(a.bubbleTranscripts) {
				continue
			}
			preview = a.bubbleTranscripts[index]
		} else {
			blockIndex = -1
		}
		key := sessionCardPreviewKey(state.SessionID, card, blockIndex)
		height, ok := transcriptPreviewHeight(preview, key, block.Text, block.Live)
		if !ok {
			continue
		}
		if heights == nil {
			heights = map[string]int{}
		}
		heights[key] = height
	}
	return heights
}

func transcriptPreviewHeight(preview *transcriptPreview, key, source string, streaming bool) (int, bool) {
	if preview == nil || !preview.ready || preview.input.key != key ||
		preview.input.source != source && !streaming || preview.height < 1 {
		return 0, false
	}
	return preview.height, true
}
