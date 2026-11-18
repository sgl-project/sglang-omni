package omni

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

// TestVADServer: Silero runs as silero_vad_server, so that Activity Monitor
// tells the voice detector from the speech model's qwen3_asr_server (the
// team took the two for two copies of the speech model); a runtime without
// that copy runs it on qwen3_asr_server.
func TestVADServer(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	withCopy := func(c *Config) {
		bin := t.TempDir()
		for _, name := range []string{"qwen3_asr_server", "silero_vad_server"} {
			if err := os.Symlink(exe, filepath.Join(bin, name)); err != nil {
				t.Fatal(err)
			}
		}
		c.Binary = filepath.Join(bin, "qwen3_asr_server")
	}
	for _, c := range []struct {
		tune func(*Config)
		want string
	}{{withCopy, "silero_vad_server"}, {nil, filepath.Base(exe)}} {
		f := newFake(t, omnitest.Opts{}, c.tune)
		v, err := NewVAD(f.rt, f.model(VADSilero))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := v.Speech(context.Background(), tone(time.Second, 0.3), 16000); err != nil {
			t.Fatal(err)
		}
		if l := f.records("launch"); len(l) != 1 || l[0].Binary != c.want || !strings.Contains(strings.Join(l[0].Args, " "), "--model-kind silero_vad") {
			t.Errorf("launched %+v, want %s", l, c.want)
		}
	}
}
