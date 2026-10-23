//go:build !darwin && !windows

package shader

import (
	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/shaderir"
	"github.com/egoist/mygo/internal/shaderir/glsl"
)

// code gives fx p's code for the OpenGL and WebGL 2 renderer, as an
// effect's, with the uniforms in the std140 block ShaderUniforms and the
// images in the samplers T0 to T3, and returns where the uniforms' lanes
// lie in the block and its size. The renderer draws with GL's y going up,
// which negates dFdy, and keeps canvases' rows going up, which it tells
// the shader in uShaderFlipRows.
func code(p *shaderir.Program, fx *scene.Effect) ([]int, int) {
	fx.GLSL = glsl.Fragment(p, "shaderFragment", glsl.Options{FlipY: "true", Block: "ShaderUniforms", FlipRows: "uShaderFlipRows"}) + `
vec4 effect(vec2 p, Effect e) {
	return shaderFragment(vec4(p, 0.0, 1.0), (p - (e.p4.z > 0.5 ? e.p4.xy : e.rect.xy)) / U2);
}
`
	return glsl.UniformLanes(p)
}
