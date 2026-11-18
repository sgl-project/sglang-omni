package main

import (
	"context"
	"fmt"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/transcriptpreview"
)

type transcriptPreviewInput struct {
	key, source            string
	size, rightPadding     float32
	text, link, background uiColor
	dark                   bool
	streaming              bool
	bubble                 bool
	index                  int
	measurement            ovSized
}

type transcriptRenderFunc func(*uiContext, transcriptPreviewInput, float32, float32) uiElement

// transcriptPreview owns one native page on its host window's main thread.
type transcriptPreview struct {
	input                          transcriptPreviewInput
	view                           *mygo.WebView
	version                        uint64
	height                         int
	ready, loaded, running, closed bool
}

var transcriptUpdate = mygo.NewEvent[transcriptpreview.Update]("transcript-preview")

func (preview *transcriptPreview) place(context *uiContext, window *mygo.Window, input transcriptPreviewInput, fallback, limit float32) uiElement {
	if window == nil || input.source == "" {
		return uiElement{}
	}
	if preview.input != input || preview.view != nil && preview.view.IsDestroyed() {
		old := preview.input
		preview.input = input
		preview.version++
		if old.key != input.key || old.size != input.size || old.rightPadding != input.rightPadding || old.text != input.text || old.link != input.link || old.dark != input.dark || old.background != input.background || preview.view != nil && preview.view.IsDestroyed() {
			preview.height, preview.ready, preview.loaded = 0, false, false
		}
		window.Update(func() { preview.update(window) })
	}
	if !preview.ready || preview.view == nil || preview.view.IsDestroyed() {
		return uiElement{}
	}
	height := fallback
	if preview.height > 0 {
		height = float32(preview.height)
	}
	return transcriptWebView(context, preview.view).Height(min(height, limit)).MinWidth(0).Shrink(0)
}

// Only one parse runs per preview. Updates arriving meanwhile replace the
// pending input, so a fast stream cannot queue an unbounded number of jobs.
func (preview *transcriptPreview) update(window *mygo.Window) {
	if preview.running || preview.closed || window.IsDestroyed() {
		return
	}
	input, version := preview.input, preview.version
	preview.running = true
	go func() {
		markup := transcriptMarkup(input.source, input.streaming)
		window.Update(func() {
			preview.running = false
			if preview.closed || window.IsDestroyed() {
				return
			}
			if preview.version != version {
				preview.update(window)
				return
			}
			if markup == "" {
				preview.ready = false
				return
			}
			if preview.view == nil || preview.view.IsDestroyed() {
				view, err := window.NewWebView(mygo.WebViewOptions{BackgroundColor: cssColor(input.background)})
				if err != nil {
					return
				}
				preview.view, preview.loaded = view, false
			}
			update := transcriptpreview.Update{HTML: markup, Version: version, Streaming: input.streaming}
			if !preview.loaded {
				preview.view.Page().LoadHTML(transcriptpreview.Page(update, input.size, input.rightPadding, cssColor(input.text), cssColor(input.link), cssColor(input.background), input.dark), "")
				preview.loaded = true
			} else {
				_ = transcriptUpdate.EmitPage(preview.view.Page(), update)
			}
			preview.ready = true
		})
	}()
}

func cssColor(color uiColor) string {
	return fmt.Sprintf("#%02x%02x%02x%02x", color.R, color.G, color.B, color.A)
}

func (a *app) mainTranscriptRenderer() transcriptRenderFunc {
	return func(context *uiContext, input transcriptPreviewInput, fallback, limit float32) uiElement {
		return a.mainTranscript.place(context, a.win, input, fallback, limit)
	}
}

func (a *app) overlayTranscriptRenderer() transcriptRenderFunc {
	return func(context *uiContext, input transcriptPreviewInput, fallback, limit float32) uiElement {
		if !input.bubble {
			return a.overlayTranscript.place(context, a.ov.win, input, fallback, limit)
		}
		for len(a.bubbleTranscripts) <= input.index {
			a.bubbleTranscripts = append(a.bubbleTranscripts, nil)
		}
		preview := a.bubbleTranscripts[input.index]
		if preview == nil {
			preview = &transcriptPreview{}
			a.bubbleTranscripts[input.index] = preview
		}
		return preview.place(context, a.ov.win, input, fallback, limit)
	}
}

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

func (a *app) closeTranscriptPreviews() {
	previews := append([]*transcriptPreview{&a.mainTranscript, &a.overlayTranscript}, a.bubbleTranscripts...)
	for _, preview := range previews {
		if preview != nil {
			preview.closed = true
		}
	}
}

type transcriptPreviewBridge struct{ app *app }

func (bridge transcriptPreviewBridge) Resize(ctx context.Context, version uint64, height int) {
	page := mygo.CallerPage(ctx)
	if page == nil || page.WebView() == nil || height < 1 {
		return
	}
	host := page.Window()
	host.Update(func() {
		a := bridge.app
		previews := append([]*transcriptPreview{&a.mainTranscript, &a.overlayTranscript}, a.bubbleTranscripts...)
		for _, preview := range previews {
			if preview == nil || !preview.ready || preview.view != page.WebView() || preview.version != version || preview.height == height {
				continue
			}
			preview.height = height
			if preview != &a.mainTranscript && preview.input.measurement.text != "" {
				a.postSessionOverlayMeasurements([]sessionOverlayMeasurement{{
					Key: preview.input.measurement, Width: preview.input.measurement.wrap, Height: float32(height),
				}})
			}
			break
		}
	})
}
