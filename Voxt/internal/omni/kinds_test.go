package omni

import (
	"bytes"
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

func TestWhisperFinalBatchesQuietCuts(t *testing.T) {
	pcm := append(tone(32*time.Second, 0.3), silence(time.Second)...)
	pcm = append(pcm, tone(32*time.Second, 0.3)...)

	for _, tc := range []struct {
		name        string
		windowTexts []string
		want        string
		requests    int
	}{
		{"batch", []string{"第一段", "末尾", "第二段", ""}, "第一段 末尾第二段", 1},
		{"older runtime", nil, "片段片段", 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, omnitest.Opts{Text: "片段", WindowTexts: tc.windowTexts, RecordAudio: true}, withServers(t, "whisper_server"))
			a := f.asr(t, ASRWhisperTurbo)
			chunks := a.spec.ASR.splitChunks(pcm)
			if len(chunks) != 2 || len(chunks[0].pcm) <= whisperWindowBytes || len(chunks[1].pcm) <= whisperWindowBytes {
				t.Fatal("the fixture should cut into two chunks of more than 30 seconds")
			}
			res, err := a.Transcribe(t.Context(), asr.FileRequest{Audio: pcm, Format: "pcm", Lang: "zh-Hans"})
			if err != nil {
				t.Fatal(err)
			}
			finals := f.records("final")
			if res.Text != tc.want || len(finals) != tc.requests {
				t.Fatalf("%q in %d requests, want %q in %d requests", res.Text, len(finals), tc.want, tc.requests)
			}
			audio, err := base64.StdEncoding.DecodeString(finals[0].Audio)
			if err != nil {
				t.Fatal(err)
			}
			if len(audio) != 4*whisperWindowBytes {
				t.Fatalf("sent %d bytes, want four 30-second windows", len(audio))
			}
			for i, chunk := range chunks {
				padded := audio[i*2*whisperWindowBytes : (i+1)*2*whisperWindowBytes]
				if !bytes.Equal(padded[:len(chunk.pcm)], chunk.pcm) ||
					len(bytes.Trim(padded[len(chunk.pcm):], "\x00")) != 0 {
					t.Errorf("chunk %d audio or padding changed", i)
				}
			}
		})
	}
}

func TestWhisperFinalSkipsDigitalSilenceTail(t *testing.T) {
	for _, tc := range []struct {
		name         string
		speechTime   time.Duration
		chunkSeconds float64
		cutSearch    time.Duration
		lastSample   byte
		wantSamples  int
		wantText     string
	}{
		{"digital silence", 2 * time.Second, 2, 0, 0, 2 * sampleRate, "speech"},
		{"quiet audio", 2 * time.Second, 2, 0, 1, 60 * sampleRate, "speech tail"},
		{"zero window within a chunk", 30 * time.Second, 30, 5 * time.Second, 0, 30 * sampleRate, "speech"},
		{"nonzero window within a chunk", 30 * time.Second, 30, 5 * time.Second, 1, 31 * sampleRate, "speech"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, omnitest.Opts{Text: "speech", WindowTexts: []string{"speech", "tail"}}, withServers(t, "whisper_server"))
			a := f.asr(t, ASRWhisperTurbo)
			a.spec.ASR.ChunkSeconds, a.spec.ASR.CutSearch = tc.chunkSeconds, tc.cutSearch
			pcm := append(tone(tc.speechTime, 0.3), silence(time.Second)...)
			pcm[len(pcm)-2] = tc.lastSample
			result, err := a.Transcribe(t.Context(), asr.FileRequest{Audio: pcm, Format: "pcm"})
			if err != nil {
				t.Fatal(err)
			}
			finals := f.records("final")
			if result.Text != tc.wantText || len(finals) != 1 || finals[0].Samples != tc.wantSamples {
				t.Fatalf("got %q and %+v, want %q from one request of %d samples", result.Text, finals, tc.wantText, tc.wantSamples)
			}
		})
	}
}

// TestKinds: Voxt 1.x's other speech models run on servers of their own
// with only the common launch flags (their servers refuse Qwen3-ASR's),
// MOSS takes dictionary hotwords, and language hints use each server's spelling.
func TestKinds(t *testing.T) {
	for _, s := range ASRModels() {
		qwen := s.Kind == kindQwen3ASR
		takesContext := qwen || s.Kind == kindMoss
		if qwen != (s.Binary == "") || takesContext != (s.ASR.ContextTerms > 0) || qwen != (s.ASR.TokensPerSecond > 0) {
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

func TestWhisperFinishCancelsInFlightPreview(t *testing.T) {
	f := newFake(t, omnitest.Opts{Delay: time.Second, Text: "text {n}"}, withServers(t, "whisper_server"))
	a := f.asr(t, ASRWhisperTurbo)
	st, err := a.Open(t.Context(), asr.StreamRequest{SampleRate: sampleRate, Lang: "en"})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	sendAll(t, st, tone(time.Second, 0.3))
	eventually(t, "the preview request reached the server", func() bool {
		return len(f.records("final")) == 1
	})
	sendAll(t, st, tone(time.Second/2, 0.3))
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the server saw the preview disconnect", func() bool {
		return len(f.records("final-cancelled")) == 1
	})
	events := drain(t, st)
	if len(events) != 1 || events[0].Kind != asr.EventFinal || events[0].Text != "text 2" {
		t.Fatalf("events %+v, want just the second request's final", events)
	}
	requests := f.records("final")
	if len(requests) != 2 || requests[1].Samples != 3*sampleRate/2 {
		t.Fatalf("requests %+v, want the final to include all samples", requests)
	}
	if requests[0].Fields["preview"] != "true" || requests[1].Fields["preview"] != "" {
		t.Fatalf("requests %+v, want only the preview to request fast disconnect detection", requests)
	}
}

// TestMossLive: MOSS's live preview carries only its plain-text instruction;
// its post-stop Final also carries the dictionary as hotwords.
func TestMossLive(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: "en", Hints: []string{"Zephyria"}})
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
	instruction := again(ASRMoss).ASR.Instruction
	if finals := f.records("final"); len(finals) != 1 || finals[0].Fields["prompt"] != instruction+" Hotwords: Zephyria" {
		t.Errorf("finals %+v", finals)
	}
	f.rt.Close()
	if l := f.records("live"); len(l) != 1 || l[0].Fields["prompt"] != instruction || l[0].Fields["language"] != "en" {
		t.Errorf("live %+v", l)
	}
}

// TestMossTranscribeHotwords: a direct Final gets the dictionary in MOSS's
// hotword format; an empty dictionary leaves its instruction untouched.
func TestMossTranscribeHotwords(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	instruction := again(ASRMoss).ASR.Instruction
	for _, tc := range []struct {
		name  string
		hints []string
		want  string
	}{
		{"one rare term", []string{"Zephyria"}, instruction + " Hotwords: Zephyria"},
		{"empty dictionary", nil, instruction},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", SampleRate: 16000, Lang: "en", Hints: tc.hints}); err != nil {
				t.Fatal(err)
			}
		})
	}
	if _, err := a.WithInstruction("").Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", SampleRate: 16000, Lang: "en", Hints: []string{"Zephyria"}}); err != nil {
		t.Fatal(err)
	}
	finals := f.records("final")
	if len(finals) != 3 || finals[0].Fields["prompt"] != instruction+" Hotwords: Zephyria" || finals[1].Fields["prompt"] != instruction {
		t.Errorf("finals %+v", finals)
	}
	if got := finals[2].Fields["prompt"]; !strings.HasPrefix(got, "Transcribe the audio into text. Start each segment") || !strings.HasSuffix(got, " Hotwords: Zephyria") {
		t.Errorf("default diarization prompt = %q", got)
	}
}

// A post-meeting MOSS request keeps the whole Other track in one request,
// sends the request's prompt (the meeting's carry-over instruction), and
// returns timed speakers.
func TestMossDiarize(t *testing.T) {
	f := newFake(t, omnitest.Opts{Segments: true}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	res, err := a.Diarize(context.Background(), asr.FileRequest{Audio: tone(65*time.Second, 0.3), Format: "pcm", SampleRate: 16000, Lang: "en", Prompt: "the instruction", Hints: []string{"Zephyria"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Segments) != 2 || res.Segments[0].Start != 500*time.Millisecond || res.Segments[0].End != 1500*time.Millisecond || res.Segments[0].Speaker != "S01" || res.Segments[0].Text != "first" || res.Segments[1].Speaker != "S02" {
		t.Fatalf("segments %+v", res.Segments)
	}
	finals := f.records("final")
	if len(finals) != 1 || finals[0].Samples != 65*16000 || finals[0].Fields["prompt"] != "the instruction Hotwords: Zephyria" || finals[0].Fields["language"] != "en" {
		t.Errorf("finals %+v", finals)
	}
}

func TestMossDiarizeTokenLimit(t *testing.T) {
	f := newFake(t, omnitest.Opts{Segments: true, TokenLimit: true}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss)
	if _, err := a.Diarize(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm"}); asr.KindOf(err) != asr.KindServer {
		t.Fatalf("token-limit error %v", err)
	}
}

// TestWithInstruction: a copy with another instruction sends it as the
// prompt; the model's own Instruction is unchanged.
func TestWithInstruction(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, withServers(t, "moss_transcribe_diarize_server"))
	a := f.asr(t, ASRMoss).WithInstruction("another task")
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm", SampleRate: 16000}); err != nil {
		t.Fatal(err)
	}
	if finals := f.records("final"); len(finals) != 1 || finals[0].Fields["prompt"] != "another task" {
		t.Errorf("finals %+v", finals)
	}
	if got := again(ASRMoss).ASR.Instruction; !strings.Contains(got, "plain text") {
		t.Errorf("the model's Instruction changed: %q", got)
	}
	f.rt.Close()

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
