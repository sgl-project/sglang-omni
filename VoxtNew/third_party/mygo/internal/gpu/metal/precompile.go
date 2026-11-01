//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// PrecompileShader starts making what a shader of package ui/shader draws
// with, whose library's source is src (ShaderSource), on a goroutine of
// its own, with the system's default GPU: its library, which Metal
// compiles from the source in about 0.1 s, and its pipeline for windows
// of 8-bit color, about as long again. Renderers on that GPU then draw
// with them, waiting for what is left, rather than make them on the
// shader's first draw. Metal keeps both on disk for the next launches.
// Package ui/shader calls it as it compiles a shader.
func PrecompileShader(src string) {
	shaders.once.Do(func() {
		shaders.byKey = map[string]*shaderEntry{}
		shaders.sem = make(chan struct{}, max(1, runtime.NumCPU()-1))
		if load() == nil {
			shaders.device = createDevice() // owned, for the program's life
		}
	})
	if shaders.device == 0 {
		return
	}
	shaders.mu.Lock()
	if shaders.byKey[src] != nil {
		shaders.mu.Unlock()
		return
	}
	e := &shaderEntry{done: make(chan struct{})}
	shaders.byKey[src] = e
	shaders.mu.Unlock()
	go func() {
		shaders.sem <- struct{}{}
		defer func() { <-shaders.sem }()
		defer close(e.done)
		pool(func() { e.make(src) })
	}()
}

// shaders holds what PrecompileShader made, by source.
var shaders struct {
	once   sync.Once
	mu     sync.Mutex
	byKey  map[string]*shaderEntry
	sem    chan struct{} // the goroutines making them at once
	device id
}

// shaderEntry is what PrecompileShader made of a source: its library and
// pipeline (owned), or why not, once done is closed.
type shaderEntry struct {
	done      chan struct{}
	lib, pipe id
	err       error
}

// precompiledFormat is the format of the pipeline PrecompileShader makes:
// that of windows that draw no wide gamut.
const precompiledFormat = pixelFormatBGRA8Unorm

func (e *shaderEntry) make(src string) {
	var errObj id
	e.lib = send(shaders.device, "newLibraryWithSource:options:error:", nsString(src), 0, uintptr(unsafe.Pointer(&errObj)))
	if e.lib == 0 {
		e.err = fmt.Errorf("metal: cannot compile the shader: %s", describe(errObj))
		return
	}
	r := &Renderer{device: shaders.device}
	e.pipe, _ = r.newPipeline(e.lib, "shader_ps", precompiledFormat, blendOver)
}

// sharedShader returns what PrecompileShader made of src for a renderer
// on device drawing format, waiting for it, retained: its library, and its
// pipeline when made for format (0 otherwise). ok is false when it made
// nothing for src on device.
func sharedShader(device id, src string, format uint) (lib, pipe id, ok bool) {
	if device == 0 || device != shaders.device {
		return 0, 0, false
	}
	shaders.mu.Lock()
	e := shaders.byKey[src]
	shaders.mu.Unlock()
	if e == nil {
		return 0, 0, false
	}
	<-e.done
	if e.lib == 0 {
		return 0, 0, false
	}
	lib = send(e.lib, "retain")
	if format == precompiledFormat && e.pipe != 0 {
		pipe = send(e.pipe, "retain")
	}
	return lib, pipe, true
}
