package main

// scripts/build-omni.sh stages silero_vad_server, a copy of
// qwen3_asr_server, so that Activity Monitor tells Silero VAD from the
// speech model, and the server of every other speech model (internal/omni
// Spec.Binary). Runs on Linux too: the runtime's build, uname, uv and
// codesign are fakes.

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
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
	// The native build makes the servers, of which qwen3_asr_server
	// refuses an unknown kind as the real one does, and MLX beside them.
	var others []string
	for _, s := range omni.Models() {
		if s.Binary != "" && s.Binary != "silero_vad_server" {
			others = append(others, s.Binary)
		}
	}
	native := filepath.Join(root, "native")
	write(filepath.Join(native, "scripts/build_runtime.sh"), `#!/bin/bash
mkdir -p "$1/bin"
printf '#!/bin/sh\necho "--model-kind must be one of qwen3_asr, silero_vad" >&2\nexit 2\n' > "$1/bin/qwen3_asr_server"
chmod +x "$1/bin/qwen3_asr_server"
for s in `+strings.Join(others, " ")+`; do echo "$s" > "$1/bin/$s"; done
echo lib > "$1/bin/libmlx.dylib"
echo metal > "$1/bin/mlx.metallib"
`)
	fakeBin := filepath.Join(root, "fakebin")
	write(filepath.Join(fakeBin, "uname"), "#!/bin/sh\ncase \"$1\" in -m) echo arm64 ;; *) echo Darwin ;; esac\n")
	write(filepath.Join(fakeBin, "uv"), "#!/bin/sh\nexit 0\n")
	write(filepath.Join(fakeBin, "codesign"), "#!/bin/sh\nexit 0\n")
	out := filepath.Join(root, "staged")
	cmd := shellWithFakeBin("bash", fakeBin, filepath.Join(root, "scripts/build-omni.sh"), "--out", out)
	cmd.Env = append(cmd.Env, "OMNI_NATIVE="+native)
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
	if runtime.GOOS == "windows" {
		t.Log("skipping executable mode-bit assertions: Windows does not expose Unix permission bits")
	} else {
		if fi, err := os.Stat(silero); err != nil || fi.Mode()&0o111 == 0 {
			t.Fatalf("silero_vad_server is not executable: %v", fi.Mode())
		}
		for _, s := range others {
			if fi, err := os.Stat(filepath.Join(out, "bin", s)); err != nil || fi.Mode()&0o111 == 0 {
				t.Errorf("%s is not staged: %v", s, err)
			}
		}
	}
	if len(others) < 6 {
		t.Fatalf("the specs name %v", others)
	}
}
