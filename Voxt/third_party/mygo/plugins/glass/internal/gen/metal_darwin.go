package main

import (
	"os"

	"github.com/egoist/mygo/internal/gpu/metal"
)

// compile compiles the shader file within the Metal renderer's head and
// tail.
func compile(file string) (compiled string, code []byte, err error) {
	src, err := os.ReadFile(file)
	if err != nil {
		return "", nil, err
	}
	compiled = metal.EffectSource(string(src))
	code, err = metal.CompileLibrary(compiled)
	return compiled, code, err
}
