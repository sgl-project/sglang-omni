//go:build darwin

package metal

import "github.com/egoist/mygo/internal/scene"

// RenderOffscreen draws s with a renderer of its own into a texture,
// without a layer, and returns its premultiplied BGRA rows: for tests, as
// packages drawing effects have.
func RenderOffscreen(s *scene.Scene) ([]byte, error) {
	r, err := newRenderer()
	if err != nil {
		return nil, err
	}
	defer r.Release()
	return r.renderOffscreen(s)
}
