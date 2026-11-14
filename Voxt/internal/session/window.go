package session

import (
	"context"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// windowTimeout bounds taking the picture: the language model never
// waits longer for it.
const windowTimeout = 3 * time.Second

// shootWindow starts taking a picture of the target app's window, when
// the settings send one to the language model (Privacy.SendWindow): at
// the start, while the window the user speaks into is still in front.
// Not for a try, nor in Voxt itself, nor for Transcribe without its
// clean-up (no language model).
func (r *run) shootWindow() {
	cfg := r.m.cfg
	t := r.req.Target
	noLLM := r.req.Kind == store.KindTranscribe && !r.settings.Features.Transcribe.Enhance
	if !r.settings.Privacy.SendWindow || noLLM || cfg.Window == nil || r.req.Try || t.PID == 0 || t.PID == cfg.Self {
		return
	}
	done := make(chan []byte, 1)
	go func() {
		ctx, cancel := context.WithTimeout(r.ctx, windowTimeout)
		defer cancel()
		jpeg, err := cfg.Window(ctx, t.PID)
		if err != nil {
			r.logf("warning", "no picture of the window: %v", err)
		}
		done <- jpeg
	}()
	r.shot = sync.OnceValue(func() []byte { return <-done })
}

// readClipboardImage reads the picture on the clipboard for Rewrite's
// language model, when the settings send it
// (Privacy.SendClipboardImage).
func (r *run) readClipboardImage() {
	cfg := r.m.cfg
	if !r.settings.Privacy.SendClipboardImage || cfg.ClipboardImage == nil {
		return
	}
	ctx, cancel := context.WithTimeout(r.ctx, windowTimeout)
	defer cancel()
	jpeg, err := cfg.ClipboardImage(ctx)
	if err != nil {
		r.logf("warning", "no picture from the clipboard: %v", err)
		return
	}
	if jpeg != nil {
		r.logf("info", "the clipboard's picture goes to the language model (%d bytes)", len(jpeg))
	}
	r.clip = jpeg
}
