//go:build darwin

package ui

import (
	"errors"

	"github.com/egoist/mygo/internal/gpu/metal"
	"github.com/egoist/mygo/internal/platform"
)

func newGPURenderer(n platform.SurfaceNative) (gpuRenderer, error) {
	if n.Layer == 0 {
		return nil, errors.New("the surface has no layer")
	}
	return metal.New(n.Layer)
}
