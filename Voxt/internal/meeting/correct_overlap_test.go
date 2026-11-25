package meeting

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// wholeSpeech is a vad.Model whose Speech covers the whole recording,
// so the final pass chunks it by the fake ASR's duration cap alone.
type wholeSpeech struct{}

func (wholeSpeech) Speech(_ context.Context, pcm []byte, _ int) ([]vad.Segment, error) {
	return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
}
func (wholeSpeech) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

// overlapFileASR answers one text per Transcribe call, in order.
type overlapFileASR struct {
	mu     sync.Mutex
	texts  []string
	n      int
	onCall func(n int) // called with the call's index before answering
}

func (f *overlapFileASR) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *overlapFileASR) Transcribe(_ context.Context, _ asr.FileRequest) (asr.Result, error) {
	f.mu.Lock()
	n := f.n
	f.n++
	f.mu.Unlock()
	if f.onCall != nil {
		f.onCall(n)
	}
	if n < len(f.texts) {
		return asr.Result{Text: f.texts[n]}, nil
	}
	return asr.Result{Text: fmt.Sprintf("words %d", n)}, nil
}

// recordingChat corrects "Box" to "Voxt" like fixingChat, records every
// request's line texts, and tracks the most requests in flight at once.
type recordingChat struct {
	mu       sync.Mutex
	batches  [][]string
	cur, max int
	first    chan struct{}
	once     sync.Once
}

func (c *recordingChat) Chat(_ context.Context, req llm.Request) (llm.Response, error) {
	c.mu.Lock()
	c.cur++
	if c.cur > c.max {
		c.max = c.cur
	}
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		c.cur--
		c.mu.Unlock()
	}()
	if c.first != nil {
		c.once.Do(func() { close(c.first) })
	}
	last := req.Messages[len(req.Messages)-1].Content
	var lines []correctLine
	if err := json.Unmarshal([]byte(last), &lines); err != nil {
		return llm.Response{}, fmt.Errorf("not a correction request: %w", err)
	}
	texts := make([]string, 0, len(lines))
	var out []correctLine
	for _, l := range lines {
		texts = append(texts, l.Text)
		out = append(out, correctLine{I: l.I, Text: strings.ReplaceAll(l.Text, "Box", "Voxt")})
	}
	c.mu.Lock()
	c.batches = append(c.batches, texts)
	c.mu.Unlock()
	b, _ := json.Marshal(out)
	return llm.Response{Text: "```json\n" + string(b) + "\n```"}, nil
}

// requested reports every line text ever sent, in order.
func (c *recordingChat) requested() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, b := range c.batches {
		out = append(out, b...)
	}
	return out
}

// overlapFixture builds a two-track meeting with micSeconds/systemSeconds
// of tone audio each; the fake ASR caps a chunk at two seconds (a one-second
// hard overlap merges the tail), so a track becomes seconds-1 segments:
// pass one more second than the number of texts.
func overlapFixture(t *testing.T, micSeconds, systemSeconds int, micTexts, sysTexts []string) (context.Context, Config, store.Meetings, store.Meeting, *recordingChat) {
	t.Helper()
	ctx := context.Background()
	root := t.TempDir()
	db, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	meetDir := filepath.Join(root, "meetings", "m1")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	tone := func(seconds int) []byte {
		out := make([]byte, seconds*asr.SampleRate*2)
		for i := 0; i+1 < len(out); i += 2 {
			out[i], out[i+1] = 0x10, 0x7f
		}
		return out
	}
	for name, seconds := range map[string]int{"mic.wav": micSeconds, "system.wav": systemSeconds} {
		pcm := tone(seconds)
		if err := os.WriteFile(filepath.Join(meetDir, name), asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mt := store.Meeting{
		ID: "m1", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Audio: "m1",
		DurationMS: int64(micSeconds+systemSeconds) * 1000,
	}
	created, err := db.Meetings().CreateMeeting(ctx, mt)
	if err != nil {
		t.Fatal(err)
	}
	micASR := &overlapFileASR{texts: micTexts}
	sysASR := &overlapFileASR{texts: sysTexts}
	chat := &recordingChat{}
	cfg := Config{
		Store:    db.Meetings(),
		Dir:      filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(_ context.Context, src Source) (TrackASR, error) {
			ta := TrackASR{Caps: asr.Caps{File: true, MaxDuration: 2 * time.Second}}
			if src == SourceMic {
				ta.File = micASR
			} else {
				ta.File = sysASR
			}
			return ta, nil
		},
		VAD: wholeSpeech{},
		LLM: func(context.Context, string) (SummaryChat, error) { return chat, nil },
		Context: func(context.Context) TranscriptContext {
			return TranscriptContext{Terms: []compose.Term{{Term: "Voxt", Aliases: []string{"Box"}}}}
		},
	}
	return ctx, cfg, db.Meetings(), created, chat
}

// The mic track's correction runs while the system track's final pass is
// still transcribing: the system track's first chunk does not finish
// until a correction request has arrived.
func TestMeetingCorrectionOverlapStartsDuringSystemTrack(t *testing.T) {
	micTexts := []string{"mic Box one", "mic Box two"}
	sysTexts := []string{"system Box three", "system Box four"}
	ctx, cfg, repo, mt, chat := overlapFixture(t, 3, 3, micTexts, sysTexts)
	cfg.correctOverlap = &correctionOverlap{meeting: mt.ID, fixed: map[string]string{}}
	chat.first = make(chan struct{})
	sysASR := &overlapFileASR{texts: sysTexts}
	cfg.ASR = func(_ context.Context, src Source) (TrackASR, error) {
		ta := TrackASR{Caps: asr.Caps{File: true, MaxDuration: 2 * time.Second}}
		if src == SourceMic {
			ta.File = &overlapFileASR{texts: micTexts}
		} else {
			sysASR.onCall = func(n int) {
				if n != 0 {
					return
				}
				select {
				case <-chat.first:
				case <-time.After(5 * time.Second):
					t.Error("no correction request overlapped the system track's transcription")
				}
			}
			ta.File = sysASR
		}
		return ta, nil
	}
	if err := FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	m := NewManager(cfg)
	if err := m.CorrectTranscript(ctx, mt.ID); err != nil {
		t.Fatal(err)
	}
	_, segs, err := repo.Meeting(ctx, mt.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range segs {
		if strings.Contains(s.Text, "Box") {
			t.Fatalf("line %d not corrected: %q", s.Idx, s.Text)
		}
	}
}

// Both tracks are corrected in full (the r298c class of bug: a track
// silently never sent), the batches the overlap finished are not asked
// again, one request is in flight at a time, and at least 95% of the
// recognised characters went through a correction request.
func TestMeetingCorrectionOverlapCoversBothTracks(t *testing.T) {
	// The line counts below are sized for 30-line, 3000-rune batches: 31
	// mic lines make two batches, like a long meeting's mic track (#4539
	// raised the real limits).
	lines, runes := correctBatchLines, correctBatchRunes
	correctBatchLines, correctBatchRunes = 30, 3000
	t.Cleanup(func() { correctBatchLines, correctBatchRunes = lines, runes })
	var micTexts, sysTexts []string
	for i := 0; i < 31; i++ {
		micTexts = append(micTexts, fmt.Sprintf("mic Box line %02d", i))
	}
	// 80 different characters: a repeated one is a recognition loop the
	// final pass collapses, and the stored line would no longer match.
	var filler []rune
	for r := '一'; len(filler) < 80; r++ {
		filler = append(filler, r)
	}
	for i := 0; i < 30; i++ {
		// The words must not form a shared three-word run with the mic
		// lines, or the final pass drops them as echoes (#3982).
		sysTexts = append(sysTexts, fmt.Sprintf("system Box utterance %02d %s", i, string(filler)))
	}
	ctx, cfg, repo, mt, chat := overlapFixture(t, 32, 31, micTexts, sysTexts)
	cfg.correctOverlap = &correctionOverlap{meeting: mt.ID, fixed: map[string]string{}}
	if err := FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	m := NewManager(cfg)
	if err := m.CorrectTranscript(ctx, mt.ID); err != nil {
		t.Fatal(err)
	}
	_, segs, err := repo.Meeting(ctx, mt.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 61 {
		t.Fatalf("%d lines, want 61", len(segs))
	}
	for _, s := range segs {
		if strings.Contains(s.Text, "Box") {
			t.Fatalf("line %d not corrected: %q", s.Idx, s.Text)
		}
	}
	// Two speculative mic batches hid under the system track; the one
	// batch made only of finished lines was not asked again: 2 + 2, not
	// the sequential 3 plus 2.
	if got := len(chat.batches); got != 4 {
		t.Fatalf("%d requests, want 4 (2 overlapped mic + 2 after)", got)
	}
	asked := map[string]int{}
	for _, text := range chat.requested() {
		asked[text]++
	}
	covered, total := 0, 0
	for _, s := range segs {
		n := len([]rune(s.Text))
		total += n
		// The stored text is corrected; its recognised form is the same
		// line with Box for Voxt.
		recognised := strings.ReplaceAll(s.Text, "Voxt", "Box")
		if asked[recognised] > 0 {
			covered += n
		}
	}
	if total == 0 || covered*100 < total*95 {
		t.Fatalf("correction coverage %d/%d runes, under 95%%", covered, total)
	}
	for _, text := range sysTexts {
		if asked[text] == 0 {
			t.Fatalf("system line never corrected: %q", text[:20])
		}
	}
	if chat.max != 1 {
		t.Fatalf("%d correction requests in flight at once, want 1", chat.max)
	}
}

// Echo dropping only removes whole mic lines: a surviving line's
// speculative correction lands, a dropped line's one is discarded with
// it, and the system track is corrected too.
func TestMeetingCorrectionOverlapEchoDroppedLine(t *testing.T) {
	echo := "the quick brown fox jumps over"
	micTexts := []string{echo, "mic Box own words", echo}
	sysTexts := []string{echo}
	ctx, cfg, repo, mt, chat := overlapFixture(t, 4, 2, micTexts, sysTexts)
	cfg.correctOverlap = &correctionOverlap{meeting: mt.ID, fixed: map[string]string{}}
	if err := FinalizePass(ctx, cfg, mt, nil); err != nil {
		t.Fatal(err)
	}
	m := NewManager(cfg)
	if err := m.CorrectTranscript(ctx, mt.ID); err != nil {
		t.Fatal(err)
	}
	_, segs, err := repo.Meeting(ctx, mt.ID)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, s := range segs {
		got = append(got, string(s.Source)+":"+s.Text)
	}
	want := []string{"mic:mic Voxt own words", "system:" + echo}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("lines %q, want %q", got, want)
	}
	for _, text := range micTexts[:2] {
		if asked := chat.requested(); !contains(asked, text) {
			t.Fatalf("mic line %q never corrected: %q", text, asked)
		}
	}
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

// The overlap is armed only with a remote language model; a local one
// would share the GPU and memory with recognition (5.31b).
func TestStartCorrectOverlapGates(t *testing.T) {
	m, _, _ := correctFixture(t, store.MeetingSegment{Idx: -1, Source: store.SegmentMic, Text: "Box"})
	if o := m.startCorrectOverlap("m"); o == nil {
		t.Fatal("a remote language model arms the overlap")
	}
	m.cfg.LLMIsLocal = func(string) bool { return true }
	if o := m.startCorrectOverlap("m"); o != nil {
		t.Fatal("a local language model must not overlap recognition")
	}
	m.cfg.LLMIsLocal = nil
	m.cfg.LLM = nil
	if o := m.startCorrectOverlap("m"); o != nil {
		t.Fatal("no language model, no overlap")
	}
}
