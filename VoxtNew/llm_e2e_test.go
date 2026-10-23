package main

// Translate and Rewrite end to end over the language models' runtime
// (internal/llmrt): the fake helper process, a fake speech model, and
// voxtnew-llm behind llmrt.DevServices (the VOXTNEW_DEV_LLM hook's
// Services). On Linux the process is the fake (llmrttest); with
// VOXTNEW_LLM_REAL=1 on a Mac it is the real one ($VOXTNEW_LLM, else the
// staged resources/darwin-arm64/llm) with the models of $VOXTNEW_DEV_LLM,
// and the tests log what the chats took:
//
//	VOXTNEW_LLM_REAL=1 VOXTNEW_DEV_LLM=llm.balanced=<snapshot>,llm.translate=<dir> \
//	  go test -run TestLLM -v .

import (
	"context"
	fammodels "github.com/hehehai/actnowkit/models"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// llmSetup is the runtime and its log lines.
type llmSetup struct {
	rt   *llmrt.Runtime
	real bool
	mu   sync.Mutex
	logs []string
}

func (l *llmSetup) lines(sub string) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []string
	for _, s := range l.logs {
		if strings.Contains(s, sub) {
			out = append(out, s)
		}
	}
	return out
}

// llmServices is speech model tr and the language models over voxtnew-llm:
// fake ones answering reply on Linux (fake lists the ids), or the real
// process and $VOXTNEW_DEV_LLM's models.
func llmServices(t *testing.T, tr *fakeTranscriptions, reply string, fake ...string) (local.Services, *llmSetup) {
	t.Helper()
	l := &llmSetup{real: os.Getenv("VOXTNEW_LLM_REAL") == "1"}
	cfg := llmrt.Config{AppVersion: "test", LogDir: t.TempDir(), IdleTimeout: -1, Log: func(e core.LogEntry) {
		l.mu.Lock()
		l.logs = append(l.logs, e.Message)
		l.mu.Unlock()
	}}
	var models []llmrt.DevModel
	if l.real {
		cfg.Bin = os.Getenv("VOXTNEW_LLM")
		if cfg.Bin == "" {
			cfg.Bin, _ = filepath.Abs(filepath.Join("resources", "darwin-arm64", "llm", llmrt.BinName))
		}
		var err error
		if models, err = llmrt.ParseDev(os.Getenv(llmrt.DevEnv)); err != nil || len(models) == 0 {
			t.Fatalf("VOXTNEW_LLM_REAL needs %s: %v", llmrt.DevEnv, err)
		}
		var keep []llmrt.DevModel
		for _, m := range models {
			for _, id := range fake {
				if m.ID == id {
					keep = append(keep, m)
				}
			}
		}
		if len(keep) == 0 {
			t.Skipf("%s names none of %v", llmrt.DevEnv, fake)
		}
		models = keep
		t.Setenv(llmrt.DevEnv, "") // the app's own hook (withDevLLM) stays out
	} else {
		cfg.Bin, _ = os.Executable()
		cfg.Env = llmrttest.Env(llmrttest.Opt("reply", reply))
		for _, id := range fake {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0o644)
			os.WriteFile(filepath.Join(dir, "Hy-MT2-1.8B-Q4_K_M.gguf"), []byte("GGUF"), 0o644)
			models = append(models, llmrt.DevModel{ID: id, Dir: dir})
		}
	}
	l.rt = llmrt.New(cfg)
	t.Cleanup(l.rt.Stop)
	return llmrt.DevServices(&fakeLocal{ASRP: tr}, l.rt, models), l
}

// pumpLong is pump with a longer deadline (a real model loads first).
func (s *sessionApp) pumpLong(what string, d time.Duration, cond func() bool) {
	s.t.Helper()
	deadline := time.After(d)
	for !cond() {
		select {
		case fn := <-s.queue:
			fn()
		case <-deadline:
			s.t.Fatalf("timed out waiting for %s (session %s %+v)", what, s.sess.State, s.sess.Problem)
		}
	}
}

func hasHan(s string) bool {
	for _, r := range s {
		if unicode.Is(unicode.Han, r) {
			return true
		}
	}
	return false
}

// TestLLMTranslateSelection: the selection is translated by the default
// model, streamed, pasted over it and kept.
func TestLLMTranslateSelection(t *testing.T) {
	svc, l := llmServices(t, &fakeTranscriptions{text: "unused"}, "早上好，团队。今天下午三点开会。", llmrt.Balanced)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	src := "Good morning, team. The meeting moves to three this afternoon; please bring the release checklist."
	a := newSessionApp(t, helpertest.Opt("selection", src)+","+ev, func(s *settings.Settings) {
		s.Features.Translate.Target = "zh-Hans"
	}, svc)
	drafts := 0
	a.pumpLong("the end", 90*time.Second, func() bool {
		if a.sess.Draft != "" {
			drafts++
		}
		return a.sess.State.Ended()
	})
	end := a.sess
	if end.State != session.Done || !hasHan(end.Result) || !end.Inserted || end.LLM.ID != llmrt.Balanced {
		t.Fatalf("end %s %q %+v (llm %+v)", end.State, end.Result, end.Problem, end.LLM)
	}
	if p := a.pasted(); len(p) != 1 || p[0].Text != end.Result {
		t.Errorf("pasted %+v", p)
	}
	if es := a.history(); len(es) != 1 || es[0].Kind != store.KindTranslate || es[0].LLMProvider != llmrt.Balanced {
		t.Errorf("history %+v", es)
	}
	if drafts == 0 {
		t.Error("nothing streamed")
	}
	chats := l.lines("translate_text")
	if len(chats) == 0 {
		t.Errorf("no chat line: %q", l.logs)
	}
	t.Logf("%q → %q (llm %v)\n%s", src, end.Result, end.Timings.LLM.Round(time.Millisecond), strings.Join(chats, "\n"))
}

// TestLLMRewriteSelection: a spoken instruction rewrites the selection;
// the answer waits in the card, Insert pastes it over the selection.
func TestLLMRewriteSelection(t *testing.T) {
	tr := &fakeTranscriptions{text: "make this more polite and formal"}
	svc, l := llmServices(t, tr, "Could you please send me the report by Friday? Thank you.", llmrt.Balanced)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "send me the report by friday")+","+ev, func(*settings.Settings) {}, svc)
	a.pumpLong("the end", 90*time.Second, func() bool { return a.sess.State.Ended() })
	end := a.sess
	if end.State != session.Done || end.Result == "" || !a.answer.open || strings.Contains(end.Result, "<think>") {
		t.Fatalf("end %s %q %+v card %+v", end.State, end.Result, end.Problem, a.answer)
	}
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].Text != end.Result {
		t.Errorf("pasted %+v", p)
	}
	chats := l.lines("rewrite")
	if len(chats) == 0 {
		t.Errorf("no chat line: %q", l.logs)
	}
	t.Logf("→ %q (llm %v)\n%s", end.Result, end.Timings.LLM.Round(time.Millisecond), strings.Join(chats, "\n"))
}

// TestLLMTranslateModel: Translate on the translation model (Hy-MT2): its
// own prompt, the answer pasted.
func TestLLMTranslateModel(t *testing.T) {
	svc, l := llmServices(t, &fakeTranscriptions{text: "unused"}, "The weather is really nice today; let's meet at three.", llmrt.Translate)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "今天天气真好，我们三点见面吧。")+","+ev, func(s *settings.Settings) {
		s.Features.Translate.Target = "en"
		s.Features.Translate.LLM = llmrt.Translate
	}, svc)
	a.pumpLong("the end", 90*time.Second, func() bool { return a.sess.State.Ended() })
	end := a.sess
	if end.State != session.Done || end.Result == "" || hasHan(end.Result) || end.LLM.ID != llmrt.Translate {
		t.Fatalf("end %s %q %+v", end.State, end.Result, end.Problem)
	}
	chats := l.lines("llm.translate translate_text")
	if len(chats) == 0 {
		t.Errorf("no chat line: %q", l.logs)
	}
	t.Logf("→ %q (llm %v)\n%s", end.Result, end.Timings.LLM.Round(time.Millisecond), strings.Join(chats, "\n"))
}

// TestLLMOpenerThroughLibrary: the catalogue's installed models open
// through local.Openers' LLM (models_runtimes.go), and deleting the loaded
// one stops the process first (Openers.Unload).
func TestLLMOpenerThroughLibrary(t *testing.T) {
	// The fake process and seeded models, also beside VOXTNEW_LLM_REAL's
	// variables (scripts/mac-regress.sh runs TestLLM* with them).
	t.Setenv(llmrt.DevEnv, "")
	t.Setenv("VOXTNEW_LLM", "")
	a, _, _ := newTestApp(t)
	self, _ := os.Executable()
	rt := llmrt.New(llmrt.Config{Bin: self, Env: llmrttest.Env(llmrttest.Opt("reply", "Done.")), IdleTimeout: -1})
	a.llmrt = rt
	t.Cleanup(rt.Stop)
	seedModels(t, a, models.LLMBalanced, models.LLMTranslate)
	chat, m, err := a.local.LLM(context.Background(), "")
	if err != nil || m.ID != models.LLMBalanced {
		t.Fatalf("LLM: %+v %v", m, err)
	}
	resp, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, Skill: llm.SkillRewrite})
	if err != nil || resp.Text != "Done." {
		t.Fatalf("%+v %v", resp, err)
	}
	if st := rt.Status(); st.Model == nil || st.Model.ID != models.LLMBalanced || st.Model.Path != a.models.Dir(models.LLMBalanced) {
		t.Errorf("status %+v", st)
	}
	// The translation model's file is the catalogue's.
	tr, _, err := a.local.LLM(context.Background(), models.LLMTranslate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Chat(context.Background(), llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: "你好", Target: "en"}}); err != nil {
		t.Fatal(err)
	}
	cm, _ := models.Lookup(models.LLMTranslate)
	if st := rt.Status(); st.Model == nil || st.Model.Path != filepath.Join(a.models.Dir(models.LLMTranslate), cm.Files[0].Path) {
		t.Errorf("status %+v", st)
	}
	// Deleting another model leaves the process; deleting the loaded one stops it.
	lib := a.local.(*local.Library)
	if _, err := lib.Delete(context.Background(), models.LLMBalanced, fammodels.ThisApp); err != nil || rt.Status().State != llmrt.Ready {
		t.Errorf("delete balanced: %v %+v", err, rt.Status())
	}
	if _, err := lib.Delete(context.Background(), models.LLMTranslate, fammodels.ThisApp); err != nil || rt.Status().State != llmrt.Stopped {
		t.Errorf("delete translate: %v %+v", err, rt.Status())
	}
}
