package shader

import (
	"github.com/egoist/mygo/internal/gpu/d3d11"
	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/shaderir"
	"github.com/egoist/mygo/internal/shaderir/hlsl"
)

// code gives fx p's code for the Direct3D renderer, as an effect's, with
// the uniforms in the cbuffer of register b2 and the images in registers
// t6 to t9, after the renderer's own, and returns where the uniforms'
// lanes lie in the cbuffer and its size. It starts compiling the code,
// which takes Direct3D's compiler from a tenth of a second to more than
// one, on a goroutine of its own, which the renderers of all windows then
// share, with a cache on disk for the next launches.
func code(p *shaderir.Program, fx *scene.Effect) ([]int, int) {
	fx.HLSL.Source = hlsl.Fragment(p, "shaderFragment", 2, 6) + `
float4 effect(float2 p, Effect e) {
	return shaderFragment(float4(p, 0, 1), (p - (e.p4.z > 0.5 ? e.p4.xy : e.rect.xy)) / U2);
}
`
	d3d11.PrecompileEffect(fx.HLSL.Source)
	return hlsl.UniformLanes(p)
}
