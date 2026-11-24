//go:build js

package ui

import (
	"github.com/egoist/mygo/internal/gpu/gl"
	"github.com/egoist/mygo/internal/platform"
)

// newGPURenderer returns a WebGL 2 renderer, package gl's, for a canvas
// whose context is current while it draws a frame. Canvases draw in
// memory where the browser has no WebGL 2, which the backend tells.
func newGPURenderer(n platform.SurfaceNative) (gpuRenderer, error) {
	if n.Canvas == 0 {
		return nil, nil
	}
	r, err := gl.New()
	if err != nil {
		return nil, err
	}
	return r, nil
}
