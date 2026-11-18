package main

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"
)

func (a *app) cardPreviewKey() string { return cmpOr(a.answer.id, a.sess.ID) }

// Conversation previews keep their identity when a live answer becomes a
// completed turn. Drop unused pages when the conversation shrinks or another answer replaces it.
func (a *app) trimBubblePreviews(count int) {
	if count >= len(a.bubbleTranscripts) {
		return
	}
	for _, p := range a.bubbleTranscripts[count:] {
		if p == nil {
			continue
		}
		p.closed = true
		if a.ov.win != nil {
			a.ov.win.Update(func() {
				if p.view != nil && !p.view.IsDestroyed() {
					p.view.Destroy()
				}
			})
		}
	}
	a.bubbleTranscripts = a.bubbleTranscripts[:count]
}

func (a *app) bubblePreview(c *ui.Context, b cardBlock, color ui.Color) bool {
	if b.user || a.ov.win == nil {
		return false
	}
	for len(a.bubbleTranscripts) <= b.index {
		a.bubbleTranscripts = append(a.bubbleTranscripts, nil)
	}
	p := a.bubbleTranscripts[b.index]
	if p == nil {
		p = &transcriptPreview{}
		a.bubbleTranscripts[b.index] = p
	}
	in := transcriptPreviewInput{key: fmt.Sprintf("%s/%d", a.cardPreviewKey(), b.index), source: b.text.text,
		size: b.text.size, text: color, link: c.Theme().Accent, background: ui.RGB(36, 36, 40), dark: true, streaming: b.live}
	fallback, _ := a.ovTextHeight(b.text)
	return p.place(c.Key("preview"), a.ov.win, in, fallback, math.MaxFloat32).Valid()
}

func (a *app) bubblePreviewHeight(b cardBlock) (float32, bool) {
	if b.user || b.index >= len(a.bubbleTranscripts) {
		return 0, false
	}
	p := a.bubbleTranscripts[b.index]
	key := fmt.Sprintf("%s/%d", a.cardPreviewKey(), b.index)
	if p == nil || !p.ready || p.input.key != key || p.input.source != b.text.text && !b.live || p.height < 1 {
		return 0, false
	}
	return float32(p.height), true
}
