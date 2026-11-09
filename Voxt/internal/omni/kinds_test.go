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

// TestKinds: Voxt 1.x's other speech models run on servers of their own
// with only the common launch flags (their servers refuse Qwen3-ASR's),
// take no context, and spell the language hint their server's way.
func TestKinds(t *testing.T) {
	for _, s := range ASRModels() {
		qwen := s.Kind == kindQwen3ASR
		if qwen != (s.Binary == "") || qwen != (s.ASR.ContextTerms > 0) || qwen != (s.ASR.TokensPerSecond > 0) {
			t.Errorf("%s: kind %s, binary %q, context %d, %v tokens a second", s.ID, s.Kind, s.Binary, s.ASR.ContextTerms, s.ASR.TokensPerSecond)
		}
		if args := strings.Join(s.launchArgs(true), " "); !qwen && args != "--model-kind "+s.Kind {
			t.Errorf("%s: launch args %q", s.ID, args)
		}
		if realtime := s.Kind == kindQwen3ASR || s.Kind == kindNemotron || s.Kind == kindMoss; realtime != s.ASR.Realtime {
			t.Errorf("%s: realtime %v", s.ID, s.ASR.Realtime)
		}
	}
	for _, c := range []struct{ id, code, want string }{
		{ASRNemotron, "zh-Hans", "zh-CN"}, {ASRNemotron, "en-US", "en"}, {ASRNemotron, "ja", "ja-JP"},
		{ASRWhisperTurbo, "zh-Hans", "zh"}, {ASRSenseVoice, "yue", "yue"}, {ASRSenseVoice, "fr", ""},
		{ASRParakeet, "de-DE", "de"}, {ASRParakeet, "zh", ""}, {ASRCohere, "ko", "ko"},
	} {
		if got := again(c.id).languageHint(c.code); got != c.want {
			t.Errorf("%s: hint for %q = %q, want %q", c.id, c.code, got, c.want)
		}
	}
	if b := again(ASRWhisperTurbo).ASR.contextBias([]string{"Voxt"}, "names"); b.context != "" || len(b.terms) != 0 {
		t.Errorf("whisper bias %+v", b)
	}
	moss := again(ASRMoss)
	if b := moss.ASR.contextBias([]string{"Voxt"}, "names"); b.context != moss.ASR.Instruction || len(b.terms) != 0 || !strings.Contains(b.context, "plain text") {
		t.Errorf("moss bias %+v", b)
	}
	if w := again(ASRWhisperTurbo).ASR; w.tokenBudget(time.Hour) != 1024 || w.tokenBudget(time.Second) != 1024 {
		t.Error("whisper's budget is 1024 a request")
	}
	if p := again(ASRParakeet).ASR; p.tokenBudget(time.Minute) != 0 {
		t.Error("parakeet sends no budget")
	}
}

// withServers puts the runtime's binary and the named servers, all the
// test binary (the fake), in a folder of their own.
func withServers(t *testing.T, names ...string) func(*Config) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return func(c *Config) {
		bin := t.TempDir()
		for _, name := range append([]string{"qwen3_asr_server"}, names...) {
			if err := os.Symlink(exe, filepath.Join(bin, name)); err != nil {
				t.Fatal(err)
			}
		}
		c.Binary = filepath.Join(bin, "qwen3_asr_server")
	}
}

// TestBatchPreview: a model without a realtime API (Whisper) shows the
// text of Final requests over the audio while it records, never opens a
// socket, and its Final carries the per-request budget and no context.
func TestBatchPreview(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "text {n}"}, withServers(t, "whisper_server"))
	a := f.asr(t, ASRWhisperTurbo)
	if c := a.Caps(); c.Hints || c.Prompt || !c.Stream {
		t.Errorf("caps %+v", c)
	}
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "zh-Hans", Hints: []string{"Voxt"}})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, tone(2500*time.Millisecond, 0.3))
	var first asr.Event
	select {
	case first = <-st.Events():
	case <-time.After(10 * time.Second):
		t.Fatal("no preview before Finish")
	}
	if first.Kind != asr.EventPartial || !strings.HasPrefix(first.Text, "text ") {
		t.Errorf("preview %+v", first)
	}
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if len(evs) == 0 || evs[len(evs)-1].Kind != asr.EventFinal || !strings.HasPrefix(evs[len(evs)-1].Text, "text ") {
		t.Fatalf("events %+v", evs)
	}
	if l := f.records("launch"); len(l) != 1 || l[0].Binary != "whisper_server" || strings.Contains(strings.Join(l[0].Args, " "), "--decode-interval-ms") {
		t.Errorf("launched %+v", l)
	}
	finals := f.records("final")
	if len(finals) < 2 {
		t.Fatalf("finals %+v: a preview and the Final", finals)
	}
	for _, r := range finals {
		if r.Fields["prompt"] != "" || r.Fields["max_new_tokens"] != "1024" || r.Fields["language"] != "zh" {
			t.Errorf("request %+v", r.Fields)
		}
	}
	if !slices.ContainsFunc(finals, func(r omnitest.Record) bool { return r.Samples == 40000 }) {
		t.Errorf("no request had the whole recording: %+v", finals)
	}
	f.rt.Close()
	if l := f.records("live"); len(l) != 0 {
		t.Errorf("a socket was opened: %+v", l)
	}
}

// TestMossLive: MOSS's live session and Final carry its plain-text
// instruction, not the dictionary.
func TestMossLive(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "en", Hints: []string{"Voxt"}})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, tone(1500*time.Millisecond, 0.3))
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	if evs := drain(t, st); len(evs) == 0 || evs[len(evs)-1].Kind != asr.EventFinal {
		t.Fatalf("events %+v", evs)
	}
	want := again(ASRMoss).ASR.Instruction
	if finals := f.records("final"); len(finals) != 1 || finals[0].Fields["prompt"] != want {
		t.Errorf("finals %+v", finals)
	}
	f.rt.Close()
	if l := f.records("live"); len(l) != 1 || l[0].Fields["prompt"] != want || l[0].Fields["language"] != "en" {
		t.Errorf("live %+v", l)
	}
}

// TestMissingServer: a runtime built without a model's server says so
// instead of running the model on qwen3_asr_server.
func TestMissingServer(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, withServers(t))
	a := f.asr(t, ASRParakeet)
	_, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", SampleRate: 16000})
	if err == nil || !strings.Contains(err.Error(), "parakeet_server") {
		t.Errorf("err %v", err)
	}
	if l := f.records("launch"); len(l) != 0 {
		t.Errorf("launched %+v", l)
	}
}
