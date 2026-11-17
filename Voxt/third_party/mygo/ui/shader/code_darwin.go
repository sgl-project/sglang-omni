package shader

import (
	"github.com/egoist/mygo/internal/gpu/metal"
	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/shaderir"
	"github.com/egoist/mygo/internal/shaderir/msl"
)

// code gives fx p's code for the Metal renderer, and returns where the
// uniforms' lanes lie in its buffer and the buffer's size: the whole
// library, whose tail (ShaderSource in package metal) calls shaderMain.
// It starts making the library and its pipeline, which take Metal 0.1 s
// each the first time, on a goroutine of its own, which the renderers
// then share.
func code(p *shaderir.Program, fx *scene.Effect) ([]int, int) {
	fx.Metal.Source = metal.ShaderSource(msl.Fragment(p, "shaderFragment") + `
float4 shaderMain(constant Uniforms& uniforms, texture2d<float> T0, texture2d<float> T1, texture2d<float> T2, texture2d<float> T3, float2 p, float4 rect) {
	return shaderFragment(uniforms, T0, T1, T2, T3, float4(p, 0.0f, 1.0f), (p - rect.xy) / uniforms.U2);
}
`)
	metal.PrecompileShader(fx.Metal.Source)
	return msl.UniformLanes(p)
}
