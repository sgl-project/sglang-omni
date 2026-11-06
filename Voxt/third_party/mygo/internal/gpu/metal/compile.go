//go:build darwin

package metal

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// CompileLibrary compiles Metal Shading Language into a Metal library with
// Xcode's metal tools, for code compiled ahead of time: go generate runs
// it, for the renderer's shader (shaderlib.go) and effects' (EffectSource).
func CompileLibrary(src string) ([]byte, error) {
	dir, err := os.MkdirTemp("", "metal")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	in, air, lib := filepath.Join(dir, "shader.metal"), filepath.Join(dir, "shader.air"), filepath.Join(dir, "shader.metallib")
	if err := os.WriteFile(in, []byte(src), 0o644); err != nil {
		return nil, err
	}
	// macOS 12 is the oldest MyGo runs on.
	for _, args := range [][]string{
		{"-sdk", "macosx", "metal", "-c", in, "-o", air, "-mmacosx-version-min=12.0"},
		{"-sdk", "macosx", "metallib", air, "-o", lib},
	} {
		if out, err := exec.Command("xcrun", args...).CombinedOutput(); err != nil {
			return nil, fmt.Errorf("metal: %s: %v\n%s", args[2], err, strings.TrimSpace(string(out)))
		}
	}
	return os.ReadFile(lib)
}
