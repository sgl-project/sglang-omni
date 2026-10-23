//go:build linux && (amd64 || arm64)

package ui

import (
	"github.com/egoist/mygo/internal/gpu/gl"
	"github.com/egoist/mygo/internal/platform"
)

// newGPURenderer returns an OpenGL renderer for a surface drawing in a
// GtkGLArea's render signal. Surfaces draw in memory where OpenGL does
// not run on a GPU, which the backend tells.
func newGPURenderer(n platform.SurfaceNative) (gpuRenderer, error) {
	if n.GLArea == 0 {
		return nil, nil
	}
	r, err := gl.New()
	if err != nil {
		return nil, err
	}
	return r, nil
}
