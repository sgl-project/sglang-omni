package omni

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

// serverLinks makes the runtime's bin/ a folder of links to the test
// binary under every server's name, so that each model's server is the
// fake playing that server.
func serverLinks(t *testing.T) func(*Config) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	bin := t.TempDir()
	for _, name := range []string{"qwen3_asr_server", "silero_vad_server", "whisper_server", "moss_transcribe_diarize_server", "parakeet_server", "sensevoice_server"} {
		if err := os.Symlink(exe, filepath.Join(bin, name)); err != nil {
			t.Fatal(err)
		}
	}
	return func(c *Config) { c.Binary = filepath.Join(bin, "qwen3_asr_server") }
}

// mossRaw is MOSS's output with its timestamps, speaker labels and an
// acoustic event.
const mossRaw = "[0.00][S01]你好。[sniff] [1.20]\n[1.30][S02]Hello there.[2.50]"

// TestKinds: each speech model runs on its own server, launched with only
// the flags that server knows, and its Final carries Voxt's fields for
// it; Whisper's server finds the tokenizer beside the weights; MOSS's text
// loses its tags; only Qwen3-ASR and MOSS stream.
func TestKinds(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir()) // the joined folder of Whisper's (joinDirs)
	f := newFake(t, omnitest.Opts{Text: mossRaw}, serverLinks(t))
	tokenizer := t.TempDir()
	for _, p := range []string{filepath.Join(tokenizer, "vocab.json"), filepath.Join(f.model(ASRWhisperTurbo), "weights.safetensors")} {
		if err := os.WriteFile(p, []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cases := []struct {
		id, binary string
		aux        []string
		live       bool
		fields     map[string]string // "": not sent
		text       string
	}{
		{ASRAccurate, "qwen3_asr_server", nil, true, map[string]string{"language": "zh", "prompt": "Voxt\nMLX", "audio_layout": "voxt_swift", "chunk_duration": ""}, mossRaw},
		{ASRWhisperTurbo, "whisper_server", []string{tokenizer}, false, map[string]string{"language": "zh", "max_new_tokens": "256", "prompt": "", "audio_layout": ""}, mossRaw},
		{ASRMoss, "moss_transcribe_diarize_server", nil, true, map[string]string{"language": "", "prompt": mossPlainPrompt + " Hotwords: Voxt, MLX", "max_new_tokens": "256"}, "你好。Hello there."},
		{ASRParakeet, "parakeet_server", nil, false, map[string]string{"language": "", "prompt": "", "chunk_duration": "1200"}, mossRaw},
		{ASRSenseVoice, "sensevoice_server", nil, false, map[string]string{"language": "zh", "prompt": "", "use_itn": "false", "vad_model_directory": ""}, mossRaw},
	}
	for i, c := range cases {
		a, err := NewASR(f.rt, c.id, f.model(c.id), c.aux...)
		if err != nil {
			t.Fatal(err)
		}
		if a.Caps().Stream != c.live {
			t.Errorf("%s: streams %v", c.id, a.Caps().Stream)
		}
		if !c.live {
			if _, err := a.Open(context.Background(), asr.StreamRequest{}); asr.KindOf(err) != asr.KindConfig {
				t.Errorf("%s: Open without a live preview: %v", c.id, err)
			}
		}
		res, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", Lang: "zh", Hints: []string{"Voxt", "MLX"}})
		if err != nil {
			t.Fatalf("%s: %v", c.id, err)
		}
		if res.Text != c.text {
			t.Errorf("%s: text %q, want %q", c.id, res.Text, c.text)
		}
		launch := f.records("launch")
		if got := launch[len(launch)-1]; got.Binary != c.binary || slices.Contains(got.Args, "--first-decode-ms") != (c.binary == "qwen3_asr_server") {
			t.Errorf("%s: launched %s %v", c.id, got.Binary, got.Args)
		} else if c.aux != nil && !(slices.Contains(got.Files, "vocab.json") && slices.Contains(got.Files, "weights.safetensors")) {
			t.Errorf("%s: the server's folder has %v", c.id, got.Files)
		}
		finals := f.records("final")
		if len(finals) != i+1 {
			t.Fatalf("%s: %d Finals", c.id, len(finals))
		}
		for k, v := range c.fields {
			if got := finals[i].Fields[k]; got != v {
				t.Errorf("%s: %s %q, want %q", c.id, k, got, v)
			}
		}
	}
}

// TestVADServer: Silero runs as silero_vad_server, so that Activity Monitor
// tells it from the speech model's qwen3_asr_server; a runtime without that
// copy runs it on qwen3_asr_server.
func TestVADServer(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		tune func(*Config)
		want string
	}{{serverLinks(t), "silero_vad_server"}, {nil, filepath.Base(exe)}} {
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

// TestSenseVoiceLong: a SenseVoice recording over 30 s is cut at speech by
// the server, with the speech model's Silero VAD and Voxt's settings.
func TestSenseVoiceLong(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, serverLinks(t))
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	a := f.asr(t, ASRSenseVoice).WithVAD(v)
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(31*time.Second, 0.3), Format: "pcm"}); err != nil {
		t.Fatal(err)
	}
	fields := f.records("final")[0].Fields
	if fields["vad_model_directory"] != f.model(VADSilero) || fields["vad_max_chunk_seconds"] != "24" || fields["vad_chunk_overlap_seconds"] != "0.35" {
		t.Errorf("fields %v", fields)
	}
}

// TestMossLive: MOSS's live session gets its instruction and hot words,
// and its partial text comes without the tags.
func TestMossLive(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: mossRaw, Partial: "[0.00][S01]"}, serverLinks(t))
	st, err := f.asr(t, ASRMoss).Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh", Hints: []string{"Voxt"}})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, tone(1500*time.Millisecond, 0.3))
	var first asr.Event
	select {
	case first = <-st.Events():
	case <-time.After(10 * time.Second):
		t.Fatal("no partial before Finish")
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := append([]asr.Event{first}, drain(t, st)...)
	if first.Kind != asr.EventPartial || !strings.HasPrefix(first.Text, "partial ") || evs[len(evs)-1].Result.Text != "你好。Hello there." {
		t.Errorf("events %+v", evs)
	}
	eventually(t, "the live record", func() bool { return len(f.records("live")) == 1 })
	if live := f.records("live")[0].Fields; live["prompt"] != mossPlainPrompt+" Hotwords: Voxt" || live["language"] != "" {
		t.Errorf("session %v", live)
	}
}
