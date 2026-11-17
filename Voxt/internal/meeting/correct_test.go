package meeting

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// fixingChat corrects "Box" to "Voxt" in each line it is asked about,
// leaving out the line "skip me", and answers a summary request with a
// summary of what it was given.
type fixingChat struct {
	mu   sync.Mutex
	reqs []llm.Request
}

func (f *fixingChat) Chat(_ context.Context, req llm.Request) (llm.Response, error) {
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	last := req.Messages[len(req.Messages)-1].Content
	if req.Messages[0].Role != llm.System {
		b, _ := json.Marshal(map[string]any{"transcript_summary": map[string]string{"title": "T", "content": last}, "todo_list": []string{}})
		return llm.Response{Text: string(b)}, nil
	}
	var lines []correctLine
	_ = json.Unmarshal([]byte(last), &lines)
	var out []correctLine
	for _, l := range lines {
		if l.Text != "skip me" {
			out = append(out, correctLine{I: l.I, Text: strings.ReplaceAll(l.Text, "Box", "Voxt")})
		}
	}
	b, _ := json.Marshal(out)
	return llm.Response{Text: "```json\n" + string(b) + "\n```"}, nil
}

func correctFixture(t *testing.T, segs ...store.MeetingSegment) (*Manager, store.Meetings, *fixingChat) {
	t.Helper()
	ctx := context.Background()
	s, err := sqlitestore.Open(ctx, filepath.Join(t.TempDir(), "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	created, err := s.Meetings().CreateMeeting(ctx, store.Meeting{ID: "m", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Meetings().AppendSegments(ctx, created.Seq, segs); err != nil {
		t.Fatal(err)
	}
	chat := &fixingChat{}
	m := NewManager(Config{
		Store:    s.Meetings(),
		Settings: func() settings.Meeting { return settings.Meeting{} },
		LLM:      func(context.Context, string) (SummaryChat, error) { return chat, nil },
		Context: func(context.Context) TranscriptContext {
			return TranscriptContext{Terms: []compose.Term{{Term: "Voxt", Aliases: []string{"Box"}}}, Notes: "We say Voxt.", Speech: "A Voxt product meeting"}
		},
	})
	return m, s.Meetings(), chat
}

// #4018: the lines are corrected in place with the dictionary and speech
// context; what the answer leaves out stays as recognised.
func TestCorrectTranscript(t *testing.T) {
	m, repo, chat := correctFixture(t,
		store.MeetingSegment{Idx: -1, Source: store.SegmentMic, Speaker: "me", StartMS: 0, EndMS: 900, Text: "Box is free"},
		store.MeetingSegment{Idx: -1, Source: store.SegmentSystem, StartMS: 1000, EndMS: 1900, Text: "skip me"},
		store.MeetingSegment{Idx: -1, Source: store.SegmentMic, Speaker: "me", StartMS: 2000, EndMS: 2900, Text: "try Box Up", Translation: "kept"},
	)
	ctx := context.Background()
	if err := m.CorrectTranscript(ctx, "m"); err != nil {
		t.Fatal(err)
	}
	_, segs, _ := repo.Meeting(ctx, "m")
	got := []string{}
	for _, s := range segs {
		got = append(got, s.Text)
	}
	if strings.Join(got, "|") != "Voxt is free|skip me|try Voxt Up" {
		t.Fatalf("texts %q", got)
	}
	if s := segs[2]; s.StartMS != 2000 || s.Speaker != "me" || s.Translation != "kept" || s.Source != store.SegmentMic {
		t.Fatalf("line changed beyond its text: %+v", s)
	}
	sys := chat.reqs[0].Messages[0].Content
	for _, want := range []string{"- Voxt (heard as: Box)", "We say Voxt.", "A Voxt product meeting"} {
		if !strings.Contains(sys, want) {
			t.Fatalf("system prompt lacks %q:\n%s", want, sys)
		}
	}
}

func TestCorrectSystemBlankStyleUsesDefault(t *testing.T) {
	for _, style := range []string{"", " \n\t"} {
		if got := correctSystem(TranscriptContext{}, "Box", style); got != correctPrompt {
			t.Fatalf("blank style %q changed the prompt:\n%s", style, got)
		}
	}
}

// #4134: every correction batch uses the chosen style, while the summary
// keeps its own prompt and the JSON correction contract stays intact.
func TestCorrectTranscriptCustomStyle(t *testing.T) {
	var segs []store.MeetingSegment
	for i := range 35 {
		segs = append(segs, store.MeetingSegment{Idx: i, Source: store.SegmentMic, Text: "Box"})
	}
	m, _, chat := correctFixture(t, segs...)
	style := strings.TrimSpace(strings.Repeat("保持原语气，句子简洁。Keep English product names. ", 20))
	m.cfg.Settings = func() settings.Meeting {
		return settings.Meeting{CorrectionStyle: style, Feature: settings.Feature{Prompt: "SUMMARY STYLE ONLY"}}
	}
	m.correctAndSummarize(context.Background(), "m")
	if len(chat.reqs) != 3 {
		t.Fatalf("%d requests, want two correction batches and a summary", len(chat.reqs))
	}
	for i, req := range chat.reqs[:2] {
		sys := req.Messages[0].Content
		for _, want := range []string{style, "You proofread a meeting transcript", "Answer with only a JSON array", "We say Voxt."} {
			if !strings.Contains(sys, want) {
				t.Fatalf("batch %d system prompt lacks %q:\n%s", i, want, sys)
			}
		}
		for _, unwanted := range []string{"Drop stutters", "SUMMARY STYLE ONLY"} {
			if strings.Contains(sys, unwanted) {
				t.Fatalf("batch %d system prompt contains %q:\n%s", i, unwanted, sys)
			}
		}
	}
	var summary string
	for _, msg := range chat.reqs[2].Messages {
		summary += msg.Content
	}
	if !strings.Contains(summary, "SUMMARY STYLE ONLY") || strings.Contains(summary, style) {
		t.Fatalf("summary mixed up the prompts:\n%s", summary)
	}
}

// After a meeting the summary reads the corrected transcript.
func TestSummaryReadsCorrectedTranscript(t *testing.T) {
	m, repo, chat := correctFixture(t, store.MeetingSegment{Idx: -1, Source: store.SegmentMic, Speaker: "me", Text: "Box ships today"})
	ctx := context.Background()
	updated := ""
	m.cfg.Updated = func(id string) { updated = id } // the app shows the result (11-15)
	m.correctAndSummarize(ctx, "m")
	if updated != "m" {
		t.Fatalf("Updated(%q), want the meeting's id", updated)
	}
	mt, _, _ := repo.Meeting(ctx, "m")
	sum, _ := ParseStoredSummary(mt.Summary)
	if len(chat.reqs) != 2 || !strings.Contains(sum.Body, "Voxt ships today") {
		t.Fatalf("%d requests, summary %q", len(chat.reqs), sum.Body)
	}
}

// Long transcripts go in several requests, each within its limits.
func TestCorrectionBatches(t *testing.T) {
	var segs []store.MeetingSegment
	for i := range 70 {
		segs = append(segs, store.MeetingSegment{Idx: i, Text: strings.Repeat("字", 100)})
	}
	segs = append(segs, store.MeetingSegment{Idx: 70, Text: "  "})
	b := correctionBatches(segs)
	n := 0
	for _, batch := range b {
		runes := 0
		for _, s := range batch {
			runes += len([]rune(s.Text))
		}
		if len(batch) > correctBatchLines || runes > correctBatchRunes {
			t.Fatalf("batch of %d lines, %d runes", len(batch), runes)
		}
		n += len(batch)
	}
	if n != 70 || len(b) != 3 {
		t.Fatalf("%d lines in %d batches", n, len(b))
	}
}

func TestParseCorrectionIgnoresOtherLines(t *testing.T) {
	batch := []store.MeetingSegment{{Idx: 4, Text: "a"}}
	if got := parseCorrection(`[{"i":4,"text":" b "},{"i":9,"text":"x"}]`, batch); len(got) != 1 || got[4] != "b" {
		t.Fatalf("%v", got)
	}
	if got := parseCorrection("sorry, I cannot", batch); len(got) != 0 {
		t.Fatalf("%v", got)
	}
}

// failingChat fails every request.
type failingChat struct{}

func (failingChat) Chat(context.Context, llm.Request) (llm.Response, error) {
	return llm.Response{}, errors.New("rate limited")
}

// A correction or summary that fails after the meeting is in the log, not
// dropped (11-15).
func TestCorrectAndSummarizeWarns(t *testing.T) {
	m, _, _ := correctFixture(t, store.MeetingSegment{Idx: -1, Source: store.SegmentMic, Speaker: "me", Text: "Box ships today"})
	var got []string
	m.cfg.LLM = func(context.Context, string) (SummaryChat, error) { return failingChat{}, nil }
	m.cfg.Warn = func(msg string) { got = append(got, msg) }
	m.correctAndSummarize(context.Background(), "m")
	if len(got) != 2 || !strings.HasPrefix(got[0], "correct: ") || !strings.Contains(got[0], "rate limited") || !strings.HasPrefix(got[1], "summary: ") {
		t.Fatalf("warnings %q", got)
	}
}

// A line's live translation goes under that line (11-15: the snapshot's
// lines all had store index 0, so the first line's translation went under
// every line and the others' under none).
func TestLiveTranslationFindsItsLine(t *testing.T) {
	m, _, _ := correctFixture(t)
	m.commitSegment(Segment{Source: SourceSystem, Text: "one"})
	m.commitSegment(Segment{Source: SourceSystem, Start: time.Second, Text: "two"})
	m.segWrites.Wait()
	m.applyTranslation(1, "二")
	segs := m.Current().Segments
	if len(segs) != 2 || segs[0].Translation != "" || segs[1].Translation != "二" {
		t.Fatalf("%+v", segs)
	}
}
