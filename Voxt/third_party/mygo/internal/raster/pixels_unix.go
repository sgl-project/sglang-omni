//go:build darwin || linux

package raster

import (
	"runtime"
	"syscall"
)

// allocPixels returns n bytes for the image of a Renderer, mapped outside
// the Go heap: the GC would keep about as much again beside a frame it
// holds, and only slowly give back one it freed.
func allocPixels(n int) *pixels {
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return &pixels{b: make([]byte, n)}
	}
	p := &pixels{b: b, mapped: true}
	p.cleanup = runtime.AddCleanup(p, func(b []byte) { syscall.Munmap(b) }, b)
	return p
}

// free gives the memory back at once.
func (p *pixels) free() {
	if p == nil {
		return
	}
	if p.mapped {
		p.cleanup.Stop()
		syscall.Munmap(p.b)
	}
	p.b = nil
}
