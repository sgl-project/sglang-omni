package main

// Whole sessions replaying the two failures of the 2026-11-17 demo
// (#4152, #4153; the fakes are in session_e2e_test.go).

import (
	"strings"
	"testing"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestDemoRewriteSelectedText: the demo's first failure (#4152). Text is
// selected, the user says what to do with it, and the language model got
// a picture without the text. The selection reaches the model as text
// however the helper read it: by Accessibility (select all in a text
// box) or by ⌘C (a block selection in a Lark doc, which Accessibility
// reads as a blank). The request has a picture only when the user turned
// one on, and the text still goes with it.
func TestDemoRewriteSelectedText(t *testing.T) {
	const selection = "Voxt 2.1 改了三件事。\n\nMeetings split speakers. Translate goes both ways."
	const instruction = "Summarize the selected text in one sentence."
	const answer = "Voxt 2.1 changes three things."
	for _, tc := range []struct {
		name, read, via string
		window          bool
	}{
		{"read by Accessibility", "selection", proto.ViaAX, false},
		{"read by ⌘C", "copy-selection", proto.ViaCopy, false},
		{"read by ⌘C, the window's picture on", "copy-selection", proto.ViaCopy, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fakeChat{text: answer}
			set, keys, _ := fileSetup(t, instruction, chat)
			ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
			a := newSessionApp(t, helpertest.Opt(tc.read, selection)+","+ev, func(s *settings.Settings) {
				set(s)
				s.Privacy.SendWindow = tc.window
			}, keys)
			if tc.window { // Rewrite shows the window's picture for confirmation before the request
				a.pump("the window review", func() bool { return a.sess.State == session.Recognizing && len(a.sess.ImagePreview) > 0 })
				a.postQueue(a.boxSend)
			}
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.State != session.Done || a.sess.Result != answer || a.sess.Selection != selection || a.sess.SelectionVia != tc.via {
				t.Fatalf("end %+v\n%s", a.sess, a.logText())
			}
			reqs := chat.requests()
			if len(reqs) != 1 {
				t.Fatalf("%d chats", len(reqs))
			}
			msgs := chatMessages(reqs[0])
			want := [2]string{"user", "<selected_text>\n" + selection + "\n</selected_text>\n\n<instruction>\n" + instruction + "\n</instruction>"}
			if len(msgs) == 0 || msgs[len(msgs)-1] != want {
				t.Errorf("the model did not get the selected text with the instruction: %q", msgs)
			}
			if img := reqs[0]["image"]; !tc.window && img != nil {
				t.Errorf("a picture went with the selected text: %q", img)
			}
			if !a.answer.open || a.answer.text != answer {
				t.Errorf("the answer card %+v", a.answer)
			}
		})
	}
}

// TestDemoTranslateChineseSpeech: the demo's second failure (#4153).
// English is the main language and the target, the user speaks Chinese,
// and the Chinese came back as the translation. The model is asked to
// translate into English (its main-language note does not forbid the
// target, its example translates Chinese into English); when it answers
// with the Chinese all the same, it is asked again, strictly, and only
// the English is pasted and kept in the history.
func TestDemoTranslateChineseSpeech(t *testing.T) {
	const speech, english = "嗯那个周报帮我发给小李", "Send the weekly report to Xiao Li."
	chat := &fakeChat{texts: []string{speech, english}}
	set, keys, _ := fileSetup(t, speech, chat)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger), hotkeyLine(500, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.General.MainLanguage = "en"
		s.Features.Translate.Target = "en"
		s.Features.Translate.Cleanup = true
	}, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Done || a.sess.Result != english || !a.sess.Retried || !a.sess.Inserted {
		t.Fatalf("end %+v\n%s", a.sess, a.logText())
	}
	reqs := chat.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d chats, want the first and the strict one", len(reqs))
	}
	first := chatMessages(reqs[0])
	if len(first) != 4 || first[3][1] != "<input>\n"+speech+"\n</input>" {
		t.Fatalf("first request %q", first)
	}
	if sys := first[0][1]; !strings.Contains(sys, "into English") || strings.Contains(sys, "never as a language to translate into") {
		t.Errorf("the system prompt does not ask for English, or forbids it:\n%s", sys)
	}
	if !hasHan(first[1][1]) || hasHan(first[2][1]) {
		t.Errorf("the example does not translate Chinese into English: %q -> %q", first[1][1], first[2][1])
	}
	if sys := chatMessages(reqs[1])[0][1]; !strings.Contains(sys, "Strict:") {
		t.Errorf("the second request is not strict:\n%s", sys)
	}
	if p := a.pasted(); len(p) != 1 || p[0].Text != english {
		t.Errorf("pasted %+v, want only the English", p)
	}
	if es := a.history(); len(es) != 1 || es[0].Kind != store.KindTranslate || es[0].TargetLang != "en" || es[0].Result != english {
		t.Errorf("history %+v", es)
	}
}

func hasHan(s string) bool {
	return strings.ContainsFunc(s, func(r rune) bool { return unicode.Is(unicode.Han, r) })
}
