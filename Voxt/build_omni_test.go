package main

// scripts/build-omni.sh stages silero_vad_server, a copy of
// qwen3_asr_server, so that Activity Monitor tells Silero VAD from the
// speech model (internal/omni Spec.Binary). Runs on Linux too: the
// runtime's build, uname, uv and codesign are fakes.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestBuildOmniStagesSileroCopy(t *testing.T) {
	root := t.TempDir()
	script, err := os.ReadFile("scripts/build-omni.sh")
	if err != nil {
		t.Fatal(err)
	}
	write := func(path, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(root, "scripts/build-omni.sh"), string(script))
	// The native build makes the server, which refuses an unknown kind as
	// the real one does, and MLX beside it.
	native := filepath.Join(root, "native")
	write(filepath.Join(native, "scripts/build_runtime.sh"), `#!/bin/bash
mkdir -p "$1/bin"
printf '#!/bin/sh\necho "--model-kind must be one of qwen3_asr, silero_vad" >&2\nexit 2\n' > "$1/bin/qwen3_asr_server"
chmod +x "$1/bin/qwen3_asr_server"
echo lib > "$1/bin/libmlx.dylib"
echo metal > "$1/bin/mlx.metallib"
`)
	fakeBin := filepath.Join(root, "fakebin")
	write(filepath.Join(fakeBin, "uname"), "#!/bin/sh\ncase \"$1\" in -m) echo arm64 ;; *) echo Darwin ;; esac\n")
	write(filepath.Join(fakeBin, "uv"), "#!/bin/sh\nexit 0\n")
	write(filepath.Join(fakeBin, "codesign"), "#!/bin/sh\nexit 0\n")
	out := filepath.Join(root, "staged")
	cmd := exec.Command("bash", filepath.Join(root, "scripts/build-omni.sh"), "--out", out)
	cmd.Env = append(os.Environ(), "PATH="+fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"), "OMNI_NATIVE="+native)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build-omni.sh failed: %v\n%s", err, output)
	}
	server, err := os.ReadFile(filepath.Join(out, "bin/qwen3_asr_server"))
	if err != nil {
		t.Fatal(err)
	}
	silero := filepath.Join(out, "bin/silero_vad_server")
	copied, err := os.ReadFile(silero)
	if err != nil || !bytes.Equal(copied, server) {
		t.Fatalf("silero_vad_server is not a copy of qwen3_asr_server: %v", err)
	}
	if fi, err := os.Stat(silero); err != nil || fi.Mode()&0o111 == 0 {
		t.Fatalf("silero_vad_server is not executable: %v", fi.Mode())
	}
}
