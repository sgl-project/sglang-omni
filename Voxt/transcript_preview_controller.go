package main

// The transcript preview (#4105): a result with Markdown, code or math
// shows in a WebView with KaTeX; plain dictation keeps the native text
// (transcript_preview.go draws it). The views reach it through their
// actions' transcriptResult.

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
}

// Preview state belongs to its host window's main thread.
type transcriptPreview struct {
	input                          transcriptPreviewInput
	view                           *mygo.WebView
	version                        uint64
	height                         int
	ready, loaded, running, closed bool
	// restyle relayouts an overlay card when a preview it measured goes
	// away; the main window's preview leaves it nil.
	restyle func()
	// queue replaces Window.Update in tests, so a test observes the parse
	// and its completion without a running event loop.
	queue func(*mygo.Window, func())
}

var transcriptUpdate = mygo.NewEvent[transcriptpreview.Update]("transcript-preview")

// schedule runs fn on the window's queue.
func (p *transcriptPreview) schedule(win *mygo.Window, fn func()) {
	if p.queue != nil {
		p.queue(win, fn)
		return
	}
	win.Update(fn)
}

// request takes the input a frame shows and renders it when it changed.
func (p *transcriptPreview) request(win *mygo.Window, in transcriptPreviewInput) {
	if !p.adopt(in) {
		return
	}
	p.schedule(win, func() { p.update(win) })
}

// adopt records the input a frame shows and reports whether the markup
// must be parsed anew. A view that died with its input unchanged (the
// overlay hid) renders again; a frame revives a closed preview, even one
// closed before its first parse finished. A stream that ends drops the
// height its partial markup measured: the final render measures it anew
// (Resize).
func (p *transcriptPreview) adopt(in transcriptPreviewInput) bool {
	if !p.closed && p.input == in && (p.view == nil || !p.view.IsDestroyed()) {
		return false
	}
	old := p.input
	p.input = in
	p.version++
	p.closed = false
	if old.key != in.key || old.size != in.size || old.rightPadding != in.rightPadding || old.text != in.text || old.link != in.link || old.dark != in.dark || old.background != in.background || p.view != nil && p.view.IsDestroyed() {
		p.height, p.ready, p.loaded = 0, false, false
	}
	if old.streaming && !in.streaming {
		p.height = 0
	}
	return true
}

// Only one parse runs per preview. Updates arriving meanwhile replace the
// pending input, so a fast stream cannot queue an unbounded number of jobs.
func (p *transcriptPreview) update(win *mygo.Window) {
	if p.running || p.closed || win.IsDestroyed() {
		return
	}
	in, version := p.input, p.version
	p.running = true
	go func() {
		markup := transcriptMarkup(in.source, in.streaming)
		p.schedule(win, func() {
			p.running = false
			if p.closed || win.IsDestroyed() {
				return
			}
			if p.version != version {
				p.update(win)
				return
			}
			p.parsed(win, in, markup)
		})
	}()
}

// parsed applies a finished parse: plain text takes the shown page down
// (dropPage); markup loads into the page, creating it first when needed.
func (p *transcriptPreview) parsed(win *mygo.Window, in transcriptPreviewInput, markup string) {
	if markup == "" {
		p.dropPage()
		return
	}
	if p.view == nil || p.view.IsDestroyed() {
		view, err := win.NewWebView(mygo.WebViewOptions{BackgroundColor: cssColor(in.background)})
		if err != nil {
			return
		}
		p.view, p.loaded = view, false
	}
	data := transcriptpreview.Update{HTML: markup, Version: p.version, Streaming: in.streaming}
	if !p.loaded {
		p.view.Page().LoadHTML(transcriptpreview.Page(data, in.size, in.rightPadding, cssColor(in.text), cssColor(in.link), cssColor(in.background), in.dark), "")
		p.loaded = true
	} else {
		_ = transcriptUpdate.EmitPage(p.view.Page(), data)
	}
	p.ready = true
}

// dropPage takes the shown page down when its text turned plain: the
// native text view takes over, and a card that showed the page relayouts
// without it. A nil view keeps adopt from parsing the same input again.
func (p *transcriptPreview) dropPage() {
	replaced := p.ready || p.height > 0 || p.view != nil && !p.view.IsDestroyed()
	p.height, p.ready, p.loaded = 0, false, false
	if p.view != nil {
		if !p.view.IsDestroyed() {
			p.view.Destroy()
		}
		p.view = nil
	}
	if replaced && p.restyle != nil {
		p.restyle()
	}
}

func cssColor(c uiColor) string { return fmt.Sprintf("#%02x%02x%02x%02x", c.R, c.G, c.B, c.A) }

// transcriptResult shows a result (history, Try) in the main window's
// preview and says whether it did; the view draws the plain text if not.
// Part of historyActions and tryCardActions.
// A transcription's preview sits on its card's transcript face.
func (a *app) transcriptResult(c *uiContext, key, source string, transcript, streaming bool) bool {
	t := c.Theme()
	in := transcriptPreviewInput{key: key, source: source, size: t.FontSize, rightPadding: px(t, sp3-2), text: t.Text, link: t.Accent,
		background: sessionTextBackground(t, transcript), dark: t.Dark, streaming: streaming}
	view := a.mainTranscript.place(c, a.win, in, px(t, 110), px(t, 510))
	if view.Valid() {
		view.Margin(0, -in.rightPadding, 0, 0)
	}
	return view.Valid()
}

type transcriptPreviewBridge struct{ app *app }

func (b transcriptPreviewBridge) Resize(ctx context.Context, version uint64, height int) {
	page := mygo.CallerPage(ctx)
	if page == nil || page.WebView() == nil || page.WebView().IsDestroyed() || height < 1 {
		return
	}
	page.Window().Update(func() {
		a := b.app
		previews := append([]*transcriptPreview{&a.mainTranscript, &a.overlayTranscript}, a.bubbleTranscripts...)
		for _, preview := range previews {
			if preview == nil || !preview.ready || preview.view != page.WebView() || preview.view.IsDestroyed() || preview.version != version || preview.height == height {
				continue
			}
			preview.height = height
			if preview != &a.mainTranscript {
				a.overlayRestyle()
			}
			break
		}
	})
}
