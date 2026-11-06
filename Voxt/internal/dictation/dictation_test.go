package dictation

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

// The test binary is also the fake helper (helpertest).
func TestMain(m *testing.M) {
	if helpertest.Enabled() {
		helpertest.Main()
	}
	os.Exit(m.Run())
}

// startHelper is a supervisor over the fake helper with opts, ready.
func startHelper(t *testing.T, opts string) helper.Supervisor {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	sup := helper.New(helper.Config{Bin: self, Env: helpertest.Env(opts), Logf: t.Logf})
	t.Cleanup(sup.Stop)
	sup.Start(context.Background())
	for deadline := time.Now().Add(5 * time.Second); sup.Status().State != helper.Ready; time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("helper %+v", sup.Status())
		}
	}
	return sup
}

// TestRecognize: System dictation streams what the helper hears, partial
// by partial, and says why it cannot: Speech Recognition not allowed, or
// a language macOS has no on-device model for.
func TestRecognize(t *testing.T) {
	pcm := make([]byte, 5*asr.SampleRate/10*2) // five 100 ms frames
	for _, c := range []struct {
		name, opts, lang, want, partials string
		err                              error
	}{
		{"heard", "speech=你好 Voxt", "zh", "你好 Voxt", "[你好 你好 Voxt]", nil},
		{"no on-device model", "speech", "fr", "", "[]", ErrNoLanguage},
		{"not allowed", "speech,speech-perm=denied", "", "", "[]", ErrNotAllowed},
	} {
		t.Run(c.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			var partials []string
			res, err := asr.StreamAll(ctx, Provider{Helper: startHelper(t, c.opts)}, asr.StreamRequest{SampleRate: asr.SampleRate, Lang: c.lang}, pcm,
				asr.Feed{OnPartial: func(text string) { partials = append(partials, text) }})
			if !errors.Is(err, c.err) || res.Text != c.want || fmt.Sprint(partials) != c.partials {
				t.Fatalf("got %q %v, partials %q; want %q %v, %s", res.Text, err, partials, c.want, c.err, c.partials)
			}
			if c.err != nil && asr.KindOf(err) != asr.KindConfig {
				t.Errorf("kind %s", asr.KindOf(err))
			}
		})
	}
	// History's Retry hands it a whole recording.
	var ft asr.FileTranscriber = Provider{Helper: startHelper(t, "speech=hello again")}
	res, err := ft.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: asr.SampleRate, Lang: "en"})
	if err != nil || res.Text != "hello again" {
		t.Errorf("Transcribe %q %v", res.Text, err)
	}
}

// oneASR is a local model library with one speech model installed.
type oneASR struct{ local.NotInstalled }

var qwen = local.Model{Kind: local.ASR, ID: "asr.accurate", Name: "Qwen3-ASR 1.7B (8bit)"}

func (oneASR) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{qwen}
	}
	return nil
}

// TestServices: System dictation is listed after the installed speech
// models (Automatic keeps a downloaded one), only while a helper that
// recognises runs.
func TestServices(t *testing.T) {
	s := NewServices(oneASR{}, func(id string) string { return "name of " + id })
	ids := func() []string {
		var out []string
		for _, m := range s.Installed(local.ASR) {
			out = append(out, m.ID)
		}
		return out
	}
	if got := ids(); !slices.Equal(got, []string{qwen.ID}) {
		t.Fatalf("without a helper %v", got)
	}
	s.SetHelper(startHelper(t, "speech"))
	if got := ids(); !slices.Equal(got, []string{qwen.ID, ModelID}) || len(s.Installed(local.LLM)) != 0 {
		t.Fatalf("with a helper %v", got)
	}
	p, m, err := s.ASR(context.Background(), ModelID)
	if _, ok := p.(Provider); !ok || err != nil || m.Name != "name of "+ModelID {
		t.Fatalf("ASR %T %+v %v", p, m, err)
	}
	s.SetHelper(nil)
	if p, _, _ := s.ASR(context.Background(), ModelID); p != nil {
		t.Fatalf("ASR without a helper %T", p)
	}
}
