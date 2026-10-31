package main

// Whole sessions on Linux: the app over a real supervisor and the fake
// helper process (helpertest: a tone as the microphone, scripted
// shortcuts, text I/O that logs what it pastes), and in-process local
// models (local_fakes_test.go): a speech model that transcribes the
// recording or streams, and a language model. A shortcut starts the
// session, another stops it (or Escape cancels it), and the text lands in
// the paste log and the history.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// sessionApp is the app over the fake helper process, its post queue
// pumped by the test.
type sessionApp struct {
	*app
	t        *testing.T
	queue    chan func()
	pasteLog string
}

// newSessionApp starts the app with helper options opts (helpertest),
// the settings set changes and the local models svc (nil: none).
func newSessionApp(t *testing.T, opts string, set func(*settings.Settings), svc local.Services) *sessionApp {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	lib, err := openLibrary(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.close)
	if err := lib.settings.Update(context.Background(), set); err != nil {
		t.Fatal(err)
	}
	if svc == nil {
		svc = local.NotInstalled{}
	}
	if real, ok := svc.(realLocal); ok {
		// The app's own local models (the Library and its runtimes),
		// these installed: seeded in its library, or a real library's.
		if real.hub != "" {
			testHubs.Store(lib.store.Dir(), real.hub)
			t.Cleanup(func() { testHubs.Delete(lib.store.Dir()) })
		} else if err := modelstest.Seed(filepath.Join(lib.store.Dir(), "hub"), real.installed...); err != nil {
			t.Fatal(err)
		}
	} else {
		old := newLocalServices
		newLocalServices = func(*app) local.Services { return svc }
		t.Cleanup(func() { newLocalServices = old })
	}
	pasteLog := filepath.Join(t.TempDir(), "paste.jsonl")
	a := newApp(lib, &helperHost{bin: self, env: helpertest.Env("frame-ms=20," + helpertest.Opt("paste-log", pasteLog) + "," + opts)})
	a.measureHook = guessText
	s := &sessionApp{app: a, t: t, queue: make(chan func(), 4096), pasteLog: pasteLog}
	a.postQueue = func(fn func()) { s.queue <- fn }
	a.muteJournal = outputJournal(t.TempDir(), "") // as main.go: the helper's claims survive its crash
	a.connectHelper()
	a.startHelper()
	t.Cleanup(func() {
		a.quit()
		// Let the session's goroutines finish posting into the queue.
		for {
			select {
			case fn := <-s.queue:
				fn()
			case <-time.After(50 * time.Millisecond):
				return
			}
		}
	})
	s.pump("the helper", func() bool { return a.helperStatus().State == helper.Ready })
	return s
}

// realLocal makes newSessionApp keep the app's own local models, with
// the models installed (modelstest.Seed), or a real model library's.
type realLocal struct {
	local.NotInstalled
	installed []string
	hub       string // the library of real models (realHub, eval_e2e_test.go): read, never written
}

// pump runs posted functions until cond holds.
func (s *sessionApp) pump(what string, cond func() bool) {
	s.t.Helper()
	deadline := time.After(10 * time.Second)
	for !cond() {
		select {
		case fn := <-s.queue:
			fn()
		case <-deadline:
			s.t.Fatalf("timed out waiting for %s (session %s %+v)", what, s.sess.State, s.sess.Problem)
		}
	}
}

// pasted is what the fake helper pasted.
func (s *sessionApp) pasted() []proto.PasteParams {
	data, _ := os.ReadFile(s.pasteLog)
	var out []proto.PasteParams
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var p proto.PasteParams
		json.Unmarshal([]byte(line), &p)
		out = append(out, p)
	}
	return out
}

func (s *sessionApp) history() []store.Entry {
	es, err := s.lib.store.History().List(context.Background(), store.HistoryQuery{})
	if err != nil {
		s.t.Fatal(err)
	}
	return es
}

// script writes an events= script: hotkey events after delays.
func script(t *testing.T, lines ...string) string {
	p := filepath.Join(t.TempDir(), "events.jsonl")
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return helpertest.Opt("events", p)
}

func hotkeyLine(afterMS int, binding, phase string) string {
	front := helpertest.Apps["com.tinyspeck.slackmacgap"]
	b, _ := json.Marshal(helpertest.Script{AfterMS: afterMS, Method: proto.MethodHotkeyEvent,
		Params: mustJSON(proto.HotkeyEventParams{Binding: binding, Phase: phase, Behavior: proto.BehaviorTap, T: 1, Front: &front})})
	return string(b)
}

func mustJSON(v any) json.RawMessage { b, _ := json.Marshal(v); return b }

// fileSetup is a speech model that transcribes recordings (asrText)
// and chat as the language model; the settings need nothing (set is a
// no-op, for the tests that change their own).
func fileSetup(t *testing.T, asrText string, chat *fakeChat) (func(*settings.Settings), *fakeLocal, *fakeTranscriptions) {
	tr := &fakeTranscriptions{text: asrText}
	return func(*settings.Settings) {}, withChat(&fakeLocal{ASRP: tr}, chat), tr
}

// TestSessionEndToEnd: fn, speak, fn: the cleaned-up text is pasted into
// the app that was in front, kept in the history, and ⌃⌘V pastes it again.
func TestSessionEndToEnd(t *testing.T) {
	chat := &fakeChat{text: "Hello, VoxtNew."}
	set, keys, tr := fileSetup(t, "hello aura link", chat)
	ev := script(t,
		hotkeyLine(0, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(400, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(1500, "paste-last", proto.PhaseTrigger),
	)
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Transcribe.Prompt = "Keep it short."
	}, keys)
	var modes []overlayMode
	seen := func() {
		if (len(modes) == 0 && a.ov.mode != ovHidden) || len(modes) > 0 && modes[len(modes)-1] != a.ov.mode {
			modes = append(modes, a.ov.mode)
		}
	}
	a.pump("the recording", func() bool { seen(); return a.sess.State == session.Recording })
	if !strings.Contains(a.hotkeysSent, "Active:true") {
		t.Errorf("not active while recording: %s", a.hotkeysSent)
	}
	a.pump("the end", func() bool { seen(); return a.sess.State.Ended() })
	end := a.sess
	if end.State != session.Done || end.Result != "Hello, VoxtNew." || end.Raw != "hello aura link" || !end.Inserted || end.Target.Name != "Slack" {
		t.Fatalf("end %+v", end)
	}
	if got := fmt.Sprint(modes); got != "[recording working hidden]" { // nothing after the insert
		t.Errorf("overlay modes %s", got)
	}
	a.pump("the second paste", func() bool { return len(a.pasted()) == 2 })
	p := a.pasted()
	if p[0].Text != "Hello, VoxtNew." || p[0].TargetPID != 812 || p[1].Text != "Hello, VoxtNew." {
		t.Errorf("pasted %+v", p)
	}
	if strings.Contains(a.hotkeysSent, "Active:true") {
		t.Errorf("still active: %s", a.hotkeysSent)
	}
	es := a.history()
	if len(es) != 1 || es[0].Result != "Hello, VoxtNew." || es[0].RawText != "hello aura link" || es[0].ASRProvider != fakeASRModel.ID ||
		es[0].LLMProvider != fakeLLMModel.ID || es[0].LLMModel != fakeLLMModel.ID || es[0].Delivery != "insert" || es[0].AppName != "Slack" || es[0].AudioMS < 300 {
		t.Errorf("history %+v", es)
	}
	if calls, size := tr.uploads(); calls != 1 || size < 9000 {
		t.Errorf("uploads %d, %d bytes", calls, size)
	}
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	msgs, _ := reqs[0]["messages"].([]any)
	sys, _ := msgs[0].(map[string]any)["content"].(string)
	user, _ := msgs[len(msgs)-1].(map[string]any)["content"].(string) // after the clean-up's example turn
	if len(msgs) != 4 || !strings.Contains(sys, "Keep it short.") || strings.Contains(sys, "hello aura link") || user != "<input>\nhello aura link\n</input>" ||
		reqs[0]["temperature"] != 0.1 {
		t.Errorf("chat %v", reqs[0])
	}
}

// TestSessionStreamingCard: a streaming model shows its partial while
// recording; the app in front takes no text, so the result waits in the
// answer card, whose Insert pastes it anyway.
func TestSessionStreamingCard(t *testing.T) {
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(600, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, "readonly,"+ev, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
	}, &fakeLocal{ASRP: fakeStreamer{final: "你好，VoxtNew"}})
	partial := false
	a.pump("the end", func() bool {
		partial = partial || a.sess.State == session.Recording && a.sess.Partial == "hello" && a.ov.mode == ovRecording
		return a.sess.State.Ended()
	})
	if !partial {
		t.Error("no partial while recording")
	}
	end := a.sess
	if end.State != session.Done || end.Result != "你好，VoxtNew" || end.Inserted || !end.Card || a.ov.mode != ovAnswer || !a.answer.open {
		t.Fatalf("end %+v, overlay %s", end, a.ov.mode)
	}
	if len(a.pasted()) != 0 {
		t.Fatalf("pasted %+v", a.pasted())
	}
	if !strings.Contains(a.hotkeysSent, "Active:true") {
		t.Error("Escape should close the card")
	}
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "你好，VoxtNew" {
		t.Errorf("pasted %+v", p)
	}
	a.pump("the history", func() bool {
		es := a.history()
		return len(es) == 1 && es[0].Delivery == "insert"
	})
	if e := a.history()[0]; e.AppName != "Slack" || e.InsertedBundle != "" || e.InsertedName != "" {
		t.Errorf("inserted into the start app: %+v", e)
	}
	if a.ov.mode != ovHidden {
		t.Errorf("overlay %s", a.ov.mode)
	}
}

// TestSessionEscapeAndFailures: Escape cancels (nothing pasted or kept);
// a failing language model inserts the text as recognised with a warning.
func TestSessionEscapeAndFailures(t *testing.T) {
	t.Run("escape", func(t *testing.T) {
		chat := &fakeChat{text: "x"}
		set, keys, tr := fileSetup(t, "never", chat)
		ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(300, proto.HotkeyEscape, proto.PhaseTrigger))
		a := newSessionApp(t, ev, set, keys)
		a.pump("the end", func() bool { return a.sess.State.Ended() })
		if calls, _ := tr.uploads(); a.sess.State != session.Cancelled || a.ov.mode != ovHidden || len(a.pasted()) != 0 || len(a.history()) != 0 || calls != 0 {
			t.Errorf("end %+v", a.sess)
		}
	})
	t.Run("llm fails", func(t *testing.T) {
		chat := &fakeChat{status: 500}
		set, keys, _ := fileSetup(t, "hello there", chat)
		ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
		a := newSessionApp(t, ev, set, keys)
		a.pump("the end", func() bool { return a.sess.State.Ended() })
		if a.sess.State != session.Done || a.sess.Result != "hello there" || a.sess.Problem == nil || a.sess.Problem.Code != session.ProblemLLM {
			t.Fatalf("end %+v", a.sess)
		}
		if p := a.pasted(); len(p) != 1 || p[0].Text != "hello there" {
			t.Errorf("pasted %+v", p)
		}
		if a.notice == nil || !a.notice.warn || a.ov.mode != ovNotice || a.notice.text != "Inserted as recognised: the language model failed." {
			t.Errorf("notice %+v, overlay %s", a.notice, a.ov.mode)
		}
		if es := a.history(); len(es) != 1 || es[0].Error == "" || es[0].Status != store.StatusDone {
			t.Errorf("history %+v", es)
		}
	})
	t.Run("no speech model", func(t *testing.T) {
		ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger))
		a := newSessionApp(t, ev, func(*settings.Settings) {}, nil)
		a.pump("the notice", func() bool { return a.notice != nil })
		if a.sess.State != session.Idle || a.notice.action != actionModels || a.notice.text != "No speech model is installed." || a.ov.mode != ovNotice {
			t.Errorf("session %s, notice %+v", a.sess.State, a.notice)
		}
		a.noticeAction(a.notice.action)
		if a.router.Path() != modelsPath || a.ov.mode != ovHidden {
			t.Errorf("went to %s, overlay %s", a.router.Path(), a.ov.mode)
		}
	})
	t.Run("microphone denied", func(t *testing.T) {
		chat := &fakeChat{text: "x"}
		set, keys, _ := fileSetup(t, "x", chat)
		ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger))
		a := newSessionApp(t, "deny-mic,"+ev, set, keys)
		a.pump("the end", func() bool { return a.sess.State.Ended() })
		if a.sess.State != session.Failed || a.sess.Problem.Code != session.ProblemMicrophone || a.notice == nil || a.notice.action != actionPermissions {
			t.Errorf("end %+v notice %+v", a.sess, a.notice)
		}
	})
}

// TestSessionLongPress: a long-press binding records while held.
func TestSessionLongPress(t *testing.T) {
	chat := &fakeChat{text: "held"}
	set, keys, _ := fileSetup(t, "held", chat)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseDown), hotkeyLine(500, "transcribe/0", proto.PhaseUp))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Transcribe.Hotkeys = []settings.Hotkey{{Keys: "RightCommand", Trigger: settings.TriggerLong}}
	}, keys)
	a.pump("the recording", func() bool { return a.sess.State == session.Recording })
	if a.sess.HoldBinding != "transcribe/0" {
		t.Errorf("hold %q", a.sess.HoldBinding)
	}
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Done || a.sess.Result != "held" || a.sess.Timings.Audio < 300*time.Millisecond {
		t.Errorf("end %+v", a.sess)
	}
}

// chatMessages are a chat request's messages: role and content.
func chatMessages(req map[string]any) [][2]string {
	var out [][2]string
	msgs, _ := req["messages"].([]any)
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		role, _ := mm["role"].(string)
		content, _ := mm["content"].(string)
		out = append(out, [2]string{role, content})
	}
	return out
}
