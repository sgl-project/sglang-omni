package session

import (
	"context"
	"sync"
	"time"
)

// windowTimeout bounds taking the picture: the language model never
// waits longer for it.
const windowTimeout = 3 * time.Second

// shootWindow starts taking a picture of the target app's window, when
// the settings send one to the language model (Privacy.SendWindow): at
// the start, while the window the user speaks into is still in front.
// Not for a try, nor in Voxt itself.
func (r *run) shootWindow() {
	cfg := r.m.cfg
	t := r.req.Target
	if !r.settings.Privacy.SendWindow || cfg.Window == nil || r.req.Try || t.PID == 0 || t.PID == cfg.Self {
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
