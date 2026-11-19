package main

import (
	"context"
	"slices"
	"sync"
)

// boxShare is the box's content as the sending session reads it
// (Config.BoxInput, on the session's goroutine): stored on every box
// change and every frame the card shows (a keystroke is one). Its
// conversion count covers automatic recording stops as well as Enter.
type boxShare struct {
	mu                       sync.Mutex
	text                     string
	images                   [][]byte
	selectionSuppressed      bool
	automaticImageSuppressed bool
	pending                  int
	err                      error
	changed                  chan struct{}
}

func (b *boxShare) store(text string, images [][]byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.text, b.images = text, slices.Clone(images)
}

func (b *boxShare) start() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pending++
	if b.changed == nil {
		b.changed = make(chan struct{})
	}
}

func (b *boxShare) done(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pending--
	if err != nil {
		b.err = err
	}
	if b.pending == 0 || err != nil {
		b.signalLocked()
	}
}

func (b *boxShare) signalLocked() {
	if b.changed != nil {
		close(b.changed)
	}
	b.changed = make(chan struct{})
}

func (b *boxShare) clearError() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.err = nil
}

func (b *boxShare) reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.text, b.images, b.err = "", nil, nil
	b.signalLocked()
}

func (b *boxShare) load(ctx context.Context) (string, [][]byte, error) {
	for {
		b.mu.Lock()
		if b.pending == 0 || b.err != nil {
			text, images, err := b.text, b.images, b.err
			b.mu.Unlock()
			return text, images, err
		}
		changed := b.changed
		b.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
	}
}

func (b *boxShare) suppressAutomatic(selection, image bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.selectionSuppressed, b.automaticImageSuppressed = selection, image
}

func (b *boxShare) automaticSuppressed() (bool, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.selectionSuppressed, b.automaticImageSuppressed
}
