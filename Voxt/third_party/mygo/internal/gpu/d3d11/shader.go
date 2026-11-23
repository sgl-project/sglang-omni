//go:build windows

package d3d11

import (
	_ "embed"
	"fmt"
	"strings"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/gpu"
	"github.com/egoist/mygo/internal/gpu/d3d11/device"
)

//go:embed shader.hlsl
var shaderSource string

// effectSource is the head and the tail of an effect's pixel shader,
// around the line "// effect" (see EffectSource).
//
//go:embed effect.hlsl
var effectSource string

// EffectSource returns the source of an effect's pixel shader: shader.hlsl
// with EFFECT defined, the head of effect.hlsl, the effect's shader (see
// scene.Effect), and the tail, which draws its instances with it
// (effectps). Effects compile it ahead of time with CompileEffect.
func EffectSource(src string) string {
	head, tail, _ := strings.Cut(effectSource, "\n// effect\n")
	return "#define EFFECT\n" + shaderSource + "\n" + head + "\n" + src + "\n" + tail
}

// CompileEffect compiles the pixel shader of an effect whose shader is
// src, with the compiler Windows has, as go generate does for code
// compiled ahead of time.
func CompileEffect(src string) ([]byte, error) {
	return compile(EffectSource(src), "effectps", "ps_4_0")
}

// compileShaders makes shaderCode compile shader.hlsl even when the
// bytecode of shaders.go comes from it, for tests.
var compileShaders bool

// shaders is the bytecode of the shader's functions: the instances' vertex
// and pixel shaders, and the passes computing backdrops (see package gpu).
type shaders struct {
	vs, ps, passVS, down, blur []byte
}

// shaderCode returns the bytecode of the shaders: that compiled ahead of
// time into shaders.go, or, when shader.hlsl changed since (go generate
// ./internal/gpu/d3d11 was not run), that compiled now from shader.hlsl,
// with the compiler Windows has.
func shaderCode() (code shaders, err error) {
	if gpu.SourceSum(shaderSource) == shaderSum && !compileShaders {
		return shaders{vertexShader, pixelShader, passVertexShader, downShader, blurShader}, nil
	}
	for _, f := range []struct {
		code          *[]byte
		entry, target string
	}{{&code.vs, "vs", "vs_4_0"}, {&code.ps, "ps", "ps_4_0"}, {&code.passVS, "passvs", "vs_4_0"}, {&code.down, "downps", "ps_4_0"}, {&code.blur, "blurps", "ps_4_0"}} {
		if *f.code, err = compileShader(f.entry, f.target); err != nil {
			return shaders{}, err
		}
	}
	return code, nil
}

// procD3DCompile is the shader compiler of Windows 10 and later, which gen.go
// compiles shaders.go with too.
var procD3DCompile = syscall.NewLazyDLL(device.SystemDir() + `\d3dcompiler_47.dll`).NewProc("D3DCompile")

// compileShader compiles the function entry of shader.hlsl for target, as
// gen.go does.
func compileShader(entry, target string) ([]byte, error) {
	return compile(shaderSource, entry, target)
}

// compile compiles the function entry of source for target.
func compile(source, entry, target string) ([]byte, error) {
	if err := procD3DCompile.Find(); err != nil {
		return nil, fmt.Errorf("d3d11: no shader compiler: %w", err)
	}
	src := []byte(source)
	name, _ := syscall.BytePtrFromString("shader.hlsl")
	e, _ := syscall.BytePtrFromString(entry)
	t, _ := syscall.BytePtrFromString(target)
	// IEEE strictness avoids the native optimizer's stack overflow on
	// transformed coverage sampling while retaining level 3 optimization.
	const optimize3IEEE = 1<<15 | 1<<13
	var blob, errs uintptr
	hr, _, _ := procD3DCompile.Call(uintptr(unsafe.Pointer(&src[0])), uintptr(len(src)), uintptr(unsafe.Pointer(name)),
		0, 0, uintptr(unsafe.Pointer(e)), uintptr(unsafe.Pointer(t)), optimize3IEEE, 0,
		uintptr(unsafe.Pointer(&blob)), uintptr(unsafe.Pointer(&errs)))
	if errs != 0 {
		defer free(&errs)
	}
	if failed(hr) || blob == 0 {
		msg := "unknown error"
		if errs != 0 {
			msg = string(blobBytes(errs))
		}
		return nil, fmt.Errorf("d3d11: cannot compile the %s shader: %s", entry, msg)
	}
	defer free(&blob)
	return append([]byte(nil), blobBytes(blob)...), nil
}

// blobBytes returns the contents of an ID3DBlob, until it is released.
func blobBytes(blob uintptr) []byte {
	const blobGetBufferPointer, blobGetBufferSize = 3, 4
	p, n := call(blob, blobGetBufferPointer), call(blob, blobGetBufferSize)
	return unsafe.Slice((*byte)(ptr(p)), n)
}
