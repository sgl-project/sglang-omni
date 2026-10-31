package ui

import (
	"errors"

	"github.com/egoist/mygo/internal/gpu/d3d11"
	"github.com/egoist/mygo/internal/platform"
)

func newGPURenderer(n platform.SurfaceNative) (gpuRenderer, error) {
	if n.HWND == 0 {
		return nil, errors.New("the surface has no window")
	}
	if n.Composed {
		// The window shows its material where the frames are transparent.
		return d3d11.NewComposed(n.HWND)
	}
	return d3d11.New(n.HWND)
}
