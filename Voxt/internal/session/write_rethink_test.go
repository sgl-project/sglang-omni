package session

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type rethinkDraftLLM struct {
	kind          store.Kind
	firstStarted  chan struct{}
	secondStarted chan struct{}
	releaseSecond chan struct{}
	requests      int
}

func (m *rethinkDraftLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	m.requests++
	switch m.requests {
	case 1:
		partial := "Old partial translation"
		if m.kind == store.KindRewrite {
			partial = `{"title":"Old title","content":"Old partial answer`
		}
		req.OnDelta(partial)
		close(m.firstStarted)
		<-ctx.Done()
		return llm.Response{}, ctx.Err()
	case 2:
		close(m.secondStarted)
		select {
		case <-m.releaseSecond:
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
		if m.kind == store.KindRewrite {
			answer := `{"title":"New title","content":"New answer"}`
			req.OnDelta(answer)
			return llm.Response{Text: answer}, nil
		}
		req.OnDelta("新翻译")
		return llm.Response{Text: "新翻译"}, nil
	default:
		return llm.Response{}, context.Canceled
	}
}

func TestRethinkClearsRewriteAndTranslateDrafts(t *testing.T) {
	for _, tc := range []struct {
		kind      store.Kind
		wantDraft string
		wantTitle string
	}{
		{kind: store.KindRewrite, wantDraft: "New answer", wantTitle: "New title"},
		{kind: store.KindTranslate, wantDraft: "新翻译"},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			model := &rethinkDraftLLM{
				kind: tc.kind, firstStarted: make(chan struct{}),
				secondStarted: make(chan struct{}), releaseSecond: make(chan struct{}),
			}
			h := newHarness(t, &fakeFile{text: "hello"}, model, func(c *Config, _ *settings.Settings) {
				c.Select = &fakeSelector{}
			})
			id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			select {
			case <-model.firstStarted:
			case <-time.After(5 * time.Second):
				t.Fatal("first answer did not start")
			}
			h.rec.waitFor(t, "the first partial answer", func(s Snapshot) bool {
				return s.ID == id && s.State == Composing && s.Draft != "" && (tc.kind != store.KindRewrite || s.Title != "")
			})

			h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOff})
			select {
			case <-model.secondStarted:
			case <-time.After(5 * time.Second):
				t.Fatal("second answer did not start")
			}
			if snapshot := h.m.Current(); snapshot.Draft != "" || snapshot.Title != "" {
				t.Fatalf("obsolete answer reappeared before the new answer's first token: draft=%q title=%q", snapshot.Draft, snapshot.Title)
			}

			close(model.releaseSecond)
			end := h.rec.wait(t)
			if end.State != Done || end.Draft != tc.wantDraft || end.Title != tc.wantTitle {
				t.Fatalf("new answer did not publish normally: %+v", end)
			}
		})
	}
}
