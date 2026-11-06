package main

import (
	"os"

	"github.com/egoist/mygo/internal/gpu/d3d11"
)

// compile compiles the shader file within the Direct3D renderer's head
// and tail.
func compile(file string) (compiled string, code []byte, err error) {
	src, err := os.ReadFile(file)
	if err != nil {
		return "", nil, err
	}
	code, err = d3d11.CompileEffect(string(src))
	return d3d11.EffectSource(string(src)), code, err
}
