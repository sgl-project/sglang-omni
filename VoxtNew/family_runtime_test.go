package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"testing"
	"time"

	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// runtimeFile is the app's runtime file (zero when there is none).
func (a *app) runtimeFile(t *testing.T) (famruntime.File, bool) {
	t.Helper()
	if _, err := os.Stat(a.fam.Dir().RuntimeFile(a.fam.BundleID())); os.IsNotExist(err) {
		return famruntime.File{}, false
	}
	f, err := famruntime.Read(a.fam.Dir(), a.fam.BundleID())
	if err != nil {
		t.Fatal(err)
	}
	return f, true
}

// loadedModel is model id's entry in the runtime file.
func (a *app) loadedModel(t *testing.T, id string) (famruntime.Loaded, bool) {
	t.Helper()
	m, _ := models.Lookup(id)
	f, _ := a.runtimeFile(t)
	for _, l := range f.Loaded {
		if l.Repo == m.Repo {
			return l, true
		}
	}
	return famruntime.Loaded{}, false
}

// TestFamilyRuntimeOmni: across a Transcribe session on the fake Omni
// server the runtime file says the speech model and Silero loaded (their
// repos, commits, sizes, worker "omni"), the speech model busy while it
// records, both idle after; the paste waits for another app's; an
// unload for another app takes the model out; quitting removes the file.
// The fake servers cannot be measured (procMemory: on a Mac the test's
// own process would be, a few MB), so their sizes are written.
func TestFamilyRuntimeOmni(t *testing.T) {
	fakeOmni(t, omnitest.Opts{Text: "hello from omni"})
	(&fakeProcMemory{mem: map[int]int64{}}).install(t)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(2500, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.General.MainLanguage = "en"
	}, realLocal{installed: []string{models.ASRFast, models.VADSilero}})
	if _, ok := a.runtimeFile(t); ok {
		t.Error("a runtime file before any model loaded")
	}
	// Another voice-input app pastes, and never says it is done.
	if err := famruntime.Write(a.fam.Dir(), famruntime.File{BundleID: "dev.hehehai.auralink", PID: os.Getpid(), Pasting: true}); err != nil {
		t.Fatal(err)
	}
	busy := false
	a.pump("the end", func() bool {
		if a.sess.State == session.Recording && !busy {
			l, ok := a.loadedModel(t, models.ASRFast)
			busy = ok && l.IdleSince == nil
		}
		return a.sess.State.Ended()
	})
	end := a.sess
	if end.State != session.Done || end.Raw != "hello from omni" {
		t.Fatalf("end %s %+v %q", end.State, end.Problem, end.Raw)
	}
	if !busy {
		t.Error("the speech model was never busy while recording")
	}
	if pasted, _ := os.ReadFile(a.pasteLog); bytes.Count(pasted, []byte("\n")) != 1 || end.Timings.Deliver < famruntime.PasteWait-50*time.Millisecond {
		t.Errorf("the paste took %v (%q)", end.Timings.Deliver, pasted)
	}
	for _, id := range []string{models.ASRFast, models.VADSilero} {
		m, _ := models.Lookup(id)
		eventually(t, id+" idle", func() bool { l, ok := a.loadedModel(t, id); return ok && l.IdleSince != nil })
		l, _ := a.loadedModel(t, id)
		if rev := a.models.Status(id).Revision; l.Revision != rev || rev == "" || l.Worker != workerOmni || l.Bytes != m.Size() {
			t.Errorf("%s: %+v (revision %q, size %d)", id, l, rev, m.Size())
		}
	}
	if f, _ := a.runtimeFile(t); f.Pasting || f.BudgetBytes <= 0 {
		t.Errorf("file %+v", f)
	}

	a.unloadIdleModel(models.ASRFast)
	eventually(t, "the unload", func() bool { _, ok := a.loadedModel(t, models.ASRFast); return !ok })
	if _, ok := a.loadedModel(t, models.VADSilero); !ok {
		t.Error("Silero went too")
	}
	a.quit()
	if _, ok := a.runtimeFile(t); ok {
		t.Error("the runtime file is left after quitting")
	}
}

// TestFamilyRuntimeLLM: the language model in the runtime file: loaded
// with the memory the process reported (worker "llm"; the fake process
// cannot be measured, as in TestFamilyRuntimeOmni), idle after its chat,
// and gone when another app's request unloads it.
func TestFamilyRuntimeLLM(t *testing.T) {
	t.Setenv(llmrt.DevEnv, "")
	(&fakeProcMemory{mem: map[int]int64{}}).install(t)
	a, _, _ := newTestApp(t)
	seedModels(t, a, models.LLMFast)
	self, _ := os.Executable()
	env := append(llmrttest.Env(""), omnitest.EnvFake+"=") // this one is no Omni server
	a.llmrt = llmrt.New(llmrt.Config{Bin: self, Env: env, IdleTimeout: -1,
		OnModel: a.modelEvents(workerLLM), BeforeLoad: a.beforeLoad()})
	chat, _, err := a.local.LLM(context.Background(), models.LLMFast)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, Skill: llm.SkillRewrite}); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the model idle", func() bool { l, ok := a.loadedModel(t, models.LLMFast); return ok && l.IdleSince != nil })
	l, _ := a.loadedModel(t, models.LLMFast)
	if rev := a.models.Status(models.LLMFast).Revision; l.Revision != rev || rev == "" || l.Worker != workerLLM || l.Bytes != 2<<30 {
		t.Errorf("loaded %+v (revision %q)", l, rev)
	}
	a.unloadIdleModel(models.LLMFast)
	eventually(t, "the unload", func() bool { f, _ := a.runtimeFile(t); return len(f.Loaded) == 0 })
	if st := a.llmrt.Status(); st.State != llmrt.Stopped {
		t.Errorf("the process %+v", st)
	}
}

// TestModelBytes: a loaded model's memory in the runtime file is its
// process's measure, else what the runtime reported, else its files'
// size.
func TestModelBytes(t *testing.T) {
	pm := (&fakeProcMemory{mem: map[int]int64{}}).install(t)
	m, _ := models.Lookup(models.LLMFast)
	loaded := func(pid int, reported int64) int64 {
		return modelBytes(resident.Event{Kind: resident.Loaded, ID: m.ID, PID: pid, Bytes: reported}, m)
	}
	pm.set(41, 3<<30)
	pm.set(42, 0)
	for _, c := range []struct {
		name     string
		pid      int
		reported int64
		want     int64
	}{
		{"measured", 41, 2 << 30, 3 << 30},
		{"measured zero: the report", 42, 2 << 30, 2 << 30},
		{"not measured: the report", 43, 2 << 30, 2 << 30},
		{"no process: the report", 0, 2 << 30, 2 << 30},
		{"nothing: the size", 43, 0, m.Size()},
	} {
		if got := loaded(c.pid, c.reported); got != c.want {
			t.Errorf("%s: %d, want %d", c.name, got, c.want)
		}
	}
	if m.Size() <= 0 {
		t.Errorf("%s has no size", m.ID)
	}
	if slices.Contains(pm.asked, 0) {
		t.Errorf("pid 0 measured: %v", pm.asked)
	}
}

// TestFamilyPaste: the answer card's Insert and paste-last say "pasting"
// in the runtime file while the helper pastes, and not after.
func TestFamilyPaste(t *testing.T) {
	a, fs, _ := newTestApp(t)
	fs.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
		return proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}, nil
	}
	var during []bool
	fs.handlers[proto.MethodTextPaste] = func(json.RawMessage) (any, error) {
		f, _ := a.runtimeFile(t)
		during = append(during, f.Pasting)
		return proto.PasteResult{Posted: true}, nil
	}
	snap(a, session.Snapshot{ID: "c1", State: session.Recording, Target: proto.App{PID: 812}})
	snap(a, session.Snapshot{ID: "c1", State: session.Done, Result: "Answer", Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 812}})
	a.insertAnswer()
	a.mgr.SetLast("Last result")
	a.pasteLast(proto.App{PID: 812})
	if len(during) != 2 || !during[0] || !during[1] {
		t.Errorf("pasting during the pastes: %v", during)
	}
	if f, ok := a.runtimeFile(t); !ok || f.Pasting {
		t.Errorf("after the pastes: %+v", f)
	}
}

// TestFamilyPasteAfterMute: the output's mute and the family's paste
// rule stay apart. Muting writes nothing to the runtime file; the
// recording's mute is given back as it stops, before the text is pasted,
// and the paste still says "pasting" around the helper's paste only.
func TestFamilyPasteAfterMute(t *testing.T) {
	r := newMuteRig(t)
	r.fs.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
		return proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}, nil
	}
	type at struct{ pasting, muted bool }
	var during []at
	r.fs.handlers[proto.MethodTextPaste] = func(json.RawMessage) (any, error) {
		f, _ := r.runtimeFile(t)
		during = append(during, at{f.Pasting, r.fs.out.Sys.Muted(testSpeakers)})
		return proto.PasteResult{Posted: true}, nil
	}
	before, _ := r.runtimeFile(t)
	r.feed("c1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted while recording")
	}
	if f, _ := r.runtimeFile(t); f.Pasting || !reflect.DeepEqual(f.Loaded, before.Loaded) {
		t.Errorf("the mute changed the runtime file: %+v", f)
	}
	r.sessionChanged(session.Snapshot{Rev: r.sess.Rev + 1, ID: "c1", State: session.Done, Result: "Answer", Card: true,
		Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 812}})
	r.settle()
	r.insertAnswer()
	r.settle()
	if len(during) != 1 || !during[0].pasting || during[0].muted {
		t.Errorf("during the paste (pasting, muted): %+v", during)
	}
	if f, _ := r.runtimeFile(t); f.Pasting || r.muted() {
		t.Errorf("after the paste: %+v, muted %v", f, r.muted())
	}
}
