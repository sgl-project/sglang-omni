package session

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestLLMDeadlines: the default deadlines grow with the input; the first
// token has none of its own.
func TestLLMDeadlines(t *testing.T) {
	cfg := New(Config{}).cfg
	s := time.Second
	for _, c := range []struct {
		chars        int
		floor        time.Duration
		stall, total time.Duration
	}{
		{0, cfg.LLMTimeout, 5 * s, 20 * s},
		{40, cfg.LLMTimeout, 5 * s, 20 * s},
		{300, cfg.LLMTimeout, 5 * s, 46500 * time.Millisecond},
		{366, cfg.LLMTimeout, 5 * s, 53430 * time.Millisecond},
		{100, cfg.WriteTimeout, 5 * s, 60 * s},
		{3000, cfg.WriteTimeout, 5 * s, 330 * s},
	} {
		req := cfg.llmDeadlines(llm.Request{}, c.chars, c.floor)
		if req.FirstTokenTimeout != 0 || req.StallTimeout != c.stall || req.Timeout != c.total {
			t.Errorf("%d characters: first token %v, stall %v, total %v; want none, %v, %v", c.chars,
				req.FirstTokenTimeout, req.StallTimeout, req.Timeout, c.stall, c.total)
		}
	}
}

// script is one answer of pacedLLM: its first delta after first (-1:
// never), then the text in deltas of two runes every pause, waiting
// stall (forever when -1) before delta stallAt (0: none).
type script struct {
	text         string
	first, pause time.Duration
	stallAt      int
	stall        time.Duration
}

// pacedLLM is a backend that answers each request with the next script
// (the last repeats), honouring its context.
type pacedLLM struct {
	scripts []script
	mu      sync.Mutex
	reqs    []llm.Request
}

func (p *pacedLLM) Info() llm.BackendInfo { return llm.BackendInfo{ID: "paced"} }

func (p *pacedLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	p.mu.Lock()
	p.reqs = append(p.reqs, req)
	s := p.scripts[min(len(p.reqs), len(p.scripts))-1]
	p.mu.Unlock()
	wait := func(d time.Duration) error {
		if d < 0 {
			<-ctx.Done()
			return ctx.Err()
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(d):
			return nil
		}
	}
	if err := wait(s.first); err != nil {
		return llm.Response{}, err
	}
	rs := []rune(s.text)
	for i, n := 0, 0; i < len(rs); i, n = i+2, n+1 {
		pause := s.pause
		if n == 0 {
			pause = 0
		} else if s.stallAt > 0 && n == s.stallAt {
			pause = s.stall
		}
		if err := wait(pause); err != nil {
			return llm.Response{}, err
		}
		if req.OnDelta != nil {
			req.OnDelta(string(rs[i:min(i+2, len(rs))]))
		}
	}
	return llm.Response{Text: s.text, FinishReason: llm.FinishStop}, nil
}

// TestCleanupDeadlines: Transcribe's clean-up over llm.Client with the
// progress deadlines. A piece that never answers, or that stalls, keeps
// its text as recognised (a cut-off answer would lose words), and the
// next piece runs at once and is cleaned up; a slow but steady answer
// passes, however much longer than the stall deadline it takes; a stream
// past the request's cap ends.
func TestCleanupDeadlines(t *testing.T) {
	ms := time.Millisecond
	pieces := compose.Pieces(longZH)
	if len(pieces) != 2 {
		t.Fatalf("%d pieces", len(pieces))
	}
	second := script{text: "讨论预算。", first: 10 * ms, pause: 10 * ms}
	steady := strings.Repeat("我们下周三开会。", 4) // 16 deltas
	for _, c := range []struct {
		name    string
		text    string // recognised
		llm     *pacedLLM
		cfg     func(*Config)
		result  string
		problem string // the logged warning ("" for none)
		most    time.Duration
	}{
		{"no answer", longZH, &pacedLLM{scripts: []script{{text: "x", first: -1}, second}}, nil,
			pieces[0].Text + "讨论预算。", "piece 1 of 2: no answer within", 4 * time.Second},
		{"a piece stalls", longZH, &pacedLLM{scripts: []script{{text: steady, pause: 10 * ms, stallAt: 3, stall: -1}, second}}, nil,
			pieces[0].Text + "讨论预算。", "piece 1 of 2: no new token for 500ms after 3 pieces", 3 * time.Second},
		{"slow but steady", longZH, &pacedLLM{scripts: []script{{text: steady, first: 800 * ms, pause: 150 * ms}, second}},
			func(c *Config) { c.LLMPerChar = 20 * ms },
			steady + "讨论预算。", "", 6 * time.Second},
		{"past the cap", "hello aura link", &pacedLLM{scripts: []script{{text: steady, pause: 100 * ms}}},
			func(c *Config) { c.LLMBase, c.LLMTimeout = 200*ms, 600*ms }, "hello VoxtNew", "no answer within 600ms", 3 * time.Second},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: c.text}, &llm.Client{Backend: c.llm}, func(cfg *Config, _ *settings.Settings) {
				if c.cfg != nil {
					c.cfg(cfg)
				}
			})
			h.speak(h.start(), 5)
			end := h.rec.wait(t)
			if end.Result != c.result {
				t.Errorf("result %q, want %q", end.Result, c.result)
			}
			if c.problem == "" {
				if end.Problem != nil {
					t.Errorf("problem %+v", end.Problem)
				}
			} else if end.Problem == nil || end.Problem.Code != ProblemLLM || !end.Problem.Degraded || !h.logged(c.problem) {
				t.Errorf("problem %+v, logs %q", end.Problem, h.logs)
			}
			if end.Timings.LLM > c.most {
				t.Errorf("the clean-up took %v", end.Timings.LLM)
			}
			c.llm.mu.Lock()
			reqs := c.llm.reqs
			c.llm.mu.Unlock()
			for _, req := range reqs {
				if req.FirstTokenTimeout != 0 || req.StallTimeout != 500*ms || req.Timeout < h.m.cfg.LLMTimeout {
					t.Errorf("deadlines %v, %v, %v", req.FirstTokenTimeout, req.StallTimeout, req.Timeout)
				}
			}
		})
	}
}

// TestWriteDeadlines: Translate (and Rewrite, the same call) has the same
// progress deadlines: a slow but steady answer passes, and so does one
// whose model thinks past LLMBase before its first token (a remote
// reasoning model; 15 s failed such translations).
func TestWriteDeadlines(t *testing.T) {
	ms := time.Millisecond
	steady := strings.Repeat("我们下周三开会。", 4)
	for _, c := range []struct {
		name string
		llm  *pacedLLM
		want string // the result, or the error's start
	}{
		{"slow but steady", &pacedLLM{scripts: []script{{text: steady, first: 300 * ms, pause: 120 * ms}}}, steady},
		{"a long wait for the first token", &pacedLLM{scripts: []script{{text: steady, first: 1500 * ms, pause: 20 * ms}}}, steady},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "x"}, &llm.Client{Backend: c.llm}, func(cfg *Config, _ *settings.Settings) {
				cfg.Select = &fakeSelector{text: "Let's meet on Wednesday."}
				cfg.LLMStall, cfg.WriteTimeout = 200*ms, 5*time.Second
			})
			h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
			end := h.rec.wait(t)
			got := end.Result
			if end.Problem != nil {
				got = end.Problem.Err.Error()
			}
			if !strings.HasPrefix(got, c.want) {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// TestFollowUpDeadlines: a follow-up's request resends the conversation's
// earlier selections and answers, which the model reads before it
// answers, so its deadline counts them too.
func TestFollowUpDeadlines(t *testing.T) {
	long := strings.Repeat("a long selection ", 1250)
	model := &pacedLLM{scripts: []script{{text: "Shorter."}}}
	h := newHarness(t, &fakeFile{text: "make it shorter"}, &llm.Client{Backend: model}, nil)
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack,
		Previous: []compose.Turn{{Instruction: "summarise this", Selection: long, Answer: compose.Answer{Content: "A summary."}}}})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 5)
	if end := h.rec.wait(t); end.State != Done {
		t.Fatalf("end %s %+v", end.State, end.Problem)
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	want := h.m.cfg.LLMBase + time.Duration(len(long))*llmPrefillPerChar
	if len(model.reqs) != 1 || model.reqs[0].Timeout < want {
		t.Fatalf("deadline %v, want at least %v", model.reqs[0].Timeout, want)
	}
}
