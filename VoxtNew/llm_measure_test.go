package main

// TestLLMMeasure runs VoxtNew's own prompts (compose) through the real
// voxtnew-llm on a Mac, each twice, and logs what each chat took and
// answered: the first token, the rate, the memory, the text. It is the
// measurement behind docs/llm-models.md (VOXTNEW_LLM_REAL=1 and
// VOXTNEW_DEV_LLM as llm_e2e_test.go's).

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
)

// measureCase is one request in VoxtNew's shapes.
type measureCase struct {
	name  string
	skill llm.Skill
	in    compose.Input
	json  bool
}

var measureCases = []measureCase{
	{"transcribe zh-en", llm.SkillTranscribeEnhance, compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: "zh",
		Glossary: []compose.Term{{Term: "VoxtNew"}, {Term: "MLX"}},
		Text:     "嗯那个我们下周二呃下周三下午三点开个会讨论一下 voxt new 的发布计划然后 m l x 那边的模型也要一起看一下"}, false},
	{"transcribe en", llm.SkillTranscribeEnhance, compose.Input{Task: compose.Transcribe, Cleanup: true, Structure: true, MainLanguage: "en",
		Text: "so um the first thing is we need to update the docs second we should uh fix the login bug and third let's ship it on friday no wait thursday"}, false},
	{"translate en→zh", llm.SkillTranslateText, compose.Input{Task: compose.Translate, Selected: true, MainLanguage: "zh", TargetLanguage: "zh-Hans",
		Text: "Good morning, team. The release moves to Thursday; please review the checklist in docs/release.md before then."}, false},
	{"translate zh→en (speech)", llm.SkillTranslateText, compose.Input{Task: compose.Translate, Cleanup: true, MainLanguage: "zh", TargetLanguage: "en",
		Text: "呃今天天气真好我们下午三点在会议室见面吧记得带上电脑"}, false},
	{"rewrite selection", llm.SkillRewrite, compose.Input{Task: compose.Rewrite, Cleanup: true, MainLanguage: "en",
		Instruction: "make it more polite and a bit shorter", Text: "send me the report by friday, i told you twice already and it's still not here"}, false},
	{"rewrite answer (JSON)", llm.SkillRewrite, compose.Input{Task: compose.Rewrite, Cleanup: true, MainLanguage: "zh", Answer: true,
		Instruction: "帮我写一条消息告诉团队周五的会议改到下周一上午十点"}, true},
}

func TestLLMMeasure(t *testing.T) {
	if os.Getenv("VOXTNEW_LLM_REAL") != "1" {
		t.Skip("VOXTNEW_LLM_REAL=1 on a Mac, with VOXTNEW_DEV_LLM")
	}
	ms, err := llmrt.ParseDev(os.Getenv(llmrt.DevEnv))
	if err != nil || len(ms) == 0 {
		t.Fatalf("%s: %v", llmrt.DevEnv, err)
	}
	for _, m := range ms {
		t.Run(m.ID, func(t *testing.T) {
			_, l := llmServices(t, &fakeTranscriptions{}, "", m.ID)
			p, _ := llmrt.Lookup(m.ID)
			b, err := llmrt.NewModel(l.rt, m.ID, m.Dir)
			if err != nil {
				t.Fatal(err)
			}
			t0 := time.Now()
			if err := b.Warm(context.Background()); err != nil {
				t.Fatal(err)
			}
			t.Logf("start and load %v; %s", time.Since(t0).Round(time.Millisecond), strings.Join(l.lines("loaded"), " "))
			client := &llm.Client{Backend: b}
			for _, c := range measureCases {
				if p.TranslateOnly() && c.skill != llm.SkillTranslateText {
					continue
				}
				out, err := compose.Build(c.in)
				if err != nil {
					t.Fatal(err)
				}
				req := llm.Request{Skill: c.skill, Timeout: 60 * time.Second}
				for _, msg := range out.Messages() {
					req.Messages = append(req.Messages, llm.Message{Role: llm.Role(msg.Role), Content: msg.Content})
				}
				if c.in.Task == compose.Translate {
					req.Translate = &llm.TranslateHint{Text: c.in.Text, Target: c.in.TargetLanguage}
				}
				for run := 1; run <= 2; run++ {
					before := len(l.lines(m.ID + " "))
					var resp llm.Response
					if c.json {
						resp, err = client.JSON(context.Background(), req, nil, 0)
					} else {
						resp, err = client.Chat(context.Background(), req)
					}
					line := ""
					if ls := l.lines(m.ID + " "); len(ls) > before {
						line = ls[len(ls)-1]
					}
					text := resp.Text
					if c.json && err == nil {
						text = string(resp.JSON)
					}
					if err != nil {
						t.Errorf("%s #%d: %v", c.name, run, err)
						continue
					}
					t.Logf("%s #%d: %s\n    → %s", c.name, run, line, oneLine(compose.Clean(text, c.in.Text)))
				}
			}
			if res, err := l.rt.Memory(context.Background()); err == nil {
				t.Logf("memory: footprint %d MB, peak %d MB, MLX active %d MB, peak %d MB", res.Memory.FootprintBytes>>20,
					res.Memory.PeakFootprintBytes>>20, res.Memory.MLXActiveBytes>>20, res.Memory.MLXPeakBytes>>20)
			}
		})
	}
}

func oneLine(s string) string { return fmt.Sprintf("%q", s) }

// TestLLMRealCancel: a long answer stops within a moment of its cancel,
// and the process takes the next chat (VOXTNEW_LLM_REAL=1).
func TestLLMRealCancel(t *testing.T) {
	if os.Getenv("VOXTNEW_LLM_REAL") != "1" {
		t.Skip("VOXTNEW_LLM_REAL=1 on a Mac, with VOXTNEW_DEV_LLM")
	}
	ms, _ := llmrt.ParseDev(os.Getenv(llmrt.DevEnv))
	for _, m := range ms {
		t.Run(m.ID, func(t *testing.T) {
			_, l := llmServices(t, &fakeTranscriptions{}, "", m.ID)
			b, err := llmrt.NewModel(l.rt, m.ID, m.Dir)
			if err != nil {
				t.Fatal(err)
			}
			req := llm.Request{Skill: llm.SkillRewrite, Messages: []llm.Message{{Role: llm.User, Content: "Write a long story about a lighthouse keeper, at least 800 words."}}}
			if p, _ := llmrt.Lookup(m.ID); p.TranslateOnly() {
				req = llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: strings.Repeat("今天天气很好，我们去公园散步，然后去图书馆看书。", 30), Target: "en"}}
			}
			ctx, cancel := context.WithCancel(context.Background())
			var cancelledAt time.Time
			deltas := 0
			req.OnDelta = func(string) {
				deltas++
				if deltas == 20 {
					cancelledAt = time.Now()
					cancel()
				}
			}
			_, err = b.Chat(ctx, req)
			stopped := time.Since(cancelledAt)
			if !strings.Contains(fmt.Sprint(err), "context canceled") || cancelledAt.IsZero() || stopped > 2*time.Second {
				t.Fatalf("err %v after %d deltas, %v after the cancel", err, deltas, stopped)
			}
			t.Logf("cancelled after %d deltas; the chat ended %v later", deltas, stopped.Round(time.Millisecond))
			req.OnDelta, req.MaxTokens = nil, 16
			if _, err := b.Chat(context.Background(), req); err != nil {
				t.Fatalf("the next chat: %v", err)
			}
			if st := l.rt.Status(); st.Starts != 1 {
				t.Errorf("the process was restarted: %+v", st)
			}
		})
	}
}

// realSetup is a runtime over the real voxtnew-llm with env added to its
// environment, logging into an llmSetup.
func realSetup(t *testing.T, env ...string) *llmSetup {
	l := &llmSetup{real: true}
	bin := os.Getenv("VOXTNEW_LLM")
	if bin == "" {
		bin, _ = filepath.Abs(filepath.Join("resources", "darwin-arm64", "llm", llmrt.BinName))
	}
	l.rt = llmrt.New(llmrt.Config{Bin: bin, AppVersion: "test", LogDir: t.TempDir(), IdleTimeout: -1, Env: env, Log: func(e core.LogEntry) {
		l.mu.Lock()
		l.logs = append(l.logs, e.Message)
		l.mu.Unlock()
	}})
	t.Cleanup(l.rt.Stop)
	return l
}

// cacheTexts are a second input per measure case: the same system
// prompt with another user message, the prompt cache's usual hit.
var cacheTexts = map[string]string{
	"transcribe zh-en":         "呃那个明天上午十点我们先过一下 m l x 的测试结果然后再定 voxt new 的发布时间",
	"transcribe en":            "okay so uh first we book the room second we send the invites and um third we order lunch",
	"translate en→zh":          "Thanks for the quick fix. Please tag the build tonight so QA can start in the morning.",
	"translate zh→en (speech)": "嗯明天早上九点出发别忘了带护照和充电器",
	"rewrite selection":        "can u check the numbers again, they look off and the client is asking",
}

// TestLLMRealPromptCache: with the prompt cache, greedy answers are the
// ones without it (VOXTNEW_LLM_PROMPT_CACHE=0: nothing kept, the prompt
// split at the same place), on a miss, on a hit with another input and on
// a hit with the same input; and a prefill (Go's
// warm-up) makes the next chat a hit. It logs the first tokens and the
// cache's memory (VOXTNEW_LLM_REAL=1, VOXTNEW_DEV_LLM with MLX models).
func TestLLMRealPromptCache(t *testing.T) {
	if os.Getenv("VOXTNEW_LLM_REAL") != "1" {
		t.Skip("VOXTNEW_LLM_REAL=1 on a Mac, with VOXTNEW_DEV_LLM")
	}
	ms, _ := llmrt.ParseDev(os.Getenv(llmrt.DevEnv))
	for _, m := range ms {
		if p, _ := llmrt.Lookup(m.ID); p.TranslateOnly() {
			continue
		}
		t.Run(m.ID, func(t *testing.T) {
			off, on := realSetup(t, "VOXTNEW_LLM_PROMPT_CACHE=0"), realSetup(t)
			offModel, _ := llmrt.NewModel(off.rt, m.ID, m.Dir)
			onModel, _ := llmrt.NewModel(on.rt, m.ID, m.Dir)
			for _, b := range []*llmrt.Backend{offModel, onModel} {
				if err := b.Warm(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			chat := func(l *llmSetup, b *llmrt.Backend, c measureCase, text string) (string, string) {
				in := c.in
				in.Text = text
				out, err := compose.Build(in)
				if err != nil {
					t.Fatal(err)
				}
				seed := int64(7) // Rewrite samples: the same draws with the cache and without
				req := llm.Request{Skill: c.skill, Timeout: 60 * time.Second, Seed: &seed}
				for _, msg := range out.Messages() {
					req.Messages = append(req.Messages, llm.Message{Role: llm.Role(msg.Role), Content: msg.Content})
				}
				before := len(l.lines(m.ID + " "))
				resp, err := b.Chat(context.Background(), req)
				if err != nil {
					t.Fatalf("%s: %v", c.name, err)
				}
				ls := l.lines(m.ID + " ")
				if len(ls) <= before {
					t.Fatalf("%s: no log line", c.name)
				}
				return resp.Text, ls[len(ls)-1]
			}
			// A small dictionary's glossary is in the dictionary's order,
			// whatever the text mentions (compose.Glossary): the system
			// prompt stays the same from one input to the next.
			glossary := measureCase{"transcribe en, glossary", llm.SkillTranscribeEnhance, compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: "en",
				Glossary: []compose.Term{{Term: "VoxtNew"}, {Term: "MLX"}, {Term: "Qwen"}, {Term: "SGLang"}},
				Text:     "um so the qwen model runs on m l x now"}, false}
			cases := append([]measureCase{glossary}, measureCases...)
			texts := map[string]string{glossary.name: "uh we should ask the s g lang team about it tomorrow"}
			maps.Copy(texts, cacheTexts)
			for _, c := range cases {
				second, ok := texts[c.name]
				if !ok {
					continue
				}
				// The first chat may start from another kind's branch point;
				// another input with the same system prompt is a hit, and so
				// is the first input again; with a glossary too.
				list := []string{c.in.Text, second, c.in.Text}
				if c.name == glossary.name {
					list = append(list, "and voxt new should load it at start up")
				}
				for i, text := range list {
					want, offLine := chat(off, offModel, c, text)
					got, onLine := chat(on, onModel, c, text)
					hit := strings.Contains(onLine, "prompt cache hit")
					if i > 0 && !hit || got != want {
						t.Errorf("%s #%d: hit %v\n    without: %q\n    with:    %q", c.name, i+1, hit, want, got)
					}
					t.Logf("%s #%d:\n    off: %s\n    on:  %s", c.name, i+1, offLine, onLine)
				}
			}
			for _, l := range []*llmSetup{off, on} {
				if res, err := l.rt.Memory(context.Background()); err == nil {
					t.Logf("memory: footprint %d MB, peak %d MB, MLX active %d MB, peak %d MB", res.Memory.FootprintBytes>>20,
						res.Memory.PeakFootprintBytes>>20, res.Memory.MLXActiveBytes>>20, res.Memory.MLXPeakBytes>>20)
				}
			}
			t.Logf("%s", strings.Join(on.lines("prompt cache: kept"), "\n"))

			// Go's warm-up: the system prompt alone, then the chat.
			warm := realSetup(t)
			b, _ := llmrt.NewModel(warm.rt, m.ID, m.Dir)
			if err := b.Warm(context.Background()); err != nil {
				t.Fatal(err)
			}
			c := measureCases[0]
			out, _ := compose.Build(c.in)
			req := llm.Request{Skill: c.skill}
			for _, msg := range out.Messages() {
				req.Messages = append(req.Messages, llm.Message{Role: llm.Role(msg.Role), Content: msg.Content})
			}
			t0 := time.Now()
			if err := b.Prefill(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			took := time.Since(t0)
			_, line := chat(warm, b, c, cacheTexts[c.name])
			_, again := chat(warm, b, c, c.in.Text)
			if !strings.Contains(line, "prompt cache hit") || !strings.Contains(again, "prompt cache hit") {
				t.Errorf("after a prefill: %s\n%s", line, again)
			}
			t.Logf("prefill %v (%s)\n    then: %s\n    and:  %s", took.Round(time.Millisecond), strings.Join(warm.lines("prefill:"), ""), line, again)
		})
	}
}
