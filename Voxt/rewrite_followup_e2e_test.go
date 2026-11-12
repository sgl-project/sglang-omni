package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// Continue takes the text selected now (#3816): a follow-up reads the
// selection again; text the conversation does not hold yet goes with the
// new instruction, its own <selected_text> block, and shows in that
// turn's bubble. Nothing new selected: the follow-up is as before.

// selectNow makes path the fake helper's selection: what the user selects
// in the app ("" for nothing).
func selectNow(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

// rewriteSelecting is a Rewrite of first, the card's, with the selection
// read from a file the test changes; its path.
func rewriteSelecting(t *testing.T, chat *fakeChat, first, said string) (*sessionApp, *fakeTranscriptions, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "selection")
	selectNow(t, path, first)
	a, tr := rewriteOnce(t, helpertest.Opt("selection-file", path)+",", chat, said)
	if !a.answer.open || a.answer.input.Selection != first {
		t.Fatalf("end %+v card %+v", a.sess, a.answer)
	}
	return a, tr, path
}

// lastUser is the last user message of the chat's request i.
func lastUser(chat *fakeChat, i int) string {
	msgs := chatMessages(chat.requests()[i])
	return msgs[len(msgs)-1][1]
}

const (
	chatA = "Anna: the docs are ready.\nBen: tests still running."
	chatB = "Anna: tests passed.\nBen: ship it on Friday."
)

// TestRewriteFollowUpNewSelection: the user's case. A chat summarized,
// more messages selected, Continue: the request holds both selections, in
// order, each with its instruction; the card shows the new one under the
// follow-up; Insert pastes the last answer.
func TestRewriteFollowUpNewSelection(t *testing.T) {
	chat := &fakeChat{texts: []string{"Docs ready, tests running.", "Docs ready, tests passed, ship Friday."}}
	a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
	selectNow(t, path, chatB)
	a.recordFollowUp(t, tr, "add the new messages")
	msgs := chatMessages(chat.requests()[1])
	if len(msgs) != 4 || msgs[1][1] != "<instruction>\nsummarize\n</instruction>\n\n<selected_text>\n"+chatA+"\n</selected_text>" ||
		msgs[3][1] != "<instruction>\nadd the new messages\n</instruction>\n\n<selected_text>\n"+chatB+"\n</selected_text>" {
		t.Errorf("follow-up messages %q", msgs)
	}
	if !strings.Contains(msgs[0][1], "the user selected it since") {
		t.Errorf("the system prompt does not say what the new selection is:\n%s", msgs[0][1])
	}
	v := a.cardNow()
	if len(v.bubbles) != 4 || v.bubbles[0].quote != "" || v.bubbles[2].text != "add the new messages" || v.bubbles[2].quote != chatB || v.bubbles[2].cut {
		t.Errorf("bubbles %+v", v.bubbles)
	}
	if len(a.answer.turns) != 2 || a.answer.turns[1].Selection != chatB {
		t.Errorf("turns %+v", a.answer.turns)
	}
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "Docs ready, tests passed, ship Friday." {
		t.Errorf("pasted %+v", p)
	}
}

// TestRewriteFollowUpSelectionKinds: the same text still selected goes
// once; a selection that holds the first and more goes as new text;
// nothing selected is the follow-up as before.
func TestRewriteFollowUpSelectionKinds(t *testing.T) {
	for _, tc := range []struct {
		name, now, want string
	}{
		{"the same selection", chatA, ""},
		{"the same, other spaces", "\n" + chatA + "  ", ""},
		{"one that extends the first", chatA + "\n" + chatB, chatA + "\n" + chatB},
		{"nothing selected", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fakeChat{texts: []string{"one", "two"}}
			a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
			selectNow(t, path, tc.now)
			a.recordFollowUp(t, tr, "shorter")
			want := "<instruction>\nshorter\n</instruction>"
			if tc.want != "" {
				want += "\n\n<selected_text>\n" + tc.want + "\n</selected_text>"
			}
			if got := lastUser(chat, 1); got != want {
				t.Errorf("follow-up %q, want %q", got, want)
			}
			quote, _ := quoteText(tc.want, false)
			if v := a.cardNow(); len(v.bubbles) != 4 || v.bubbles[2].quote != quote {
				t.Errorf("bubbles %+v", v.bubbles)
			}
		})
	}
}

// TestRewriteThreeFollowUpsSelecting: three follow-ups, each with text
// selected since: every turn keeps its own, and the third request holds
// all four, in order.
func TestRewriteThreeFollowUpsSelecting(t *testing.T) {
	chat := &fakeChat{texts: []string{"r1", "r2", "r3", "r4"}}
	a, tr, path := rewriteSelecting(t, chat, "s0", "summarize")
	for i, sel := range []string{"s1", "s2", "s3"} {
		selectNow(t, path, sel)
		a.recordFollowUp(t, tr, "more "+sel)
		if got := lastUser(chat, i+1); got != "<instruction>\nmore "+sel+"\n</instruction>\n\n<selected_text>\n"+sel+"\n</selected_text>" {
			t.Errorf("follow-up %d: %q", i+1, got)
		}
	}
	msgs := chatMessages(chat.requests()[3])
	var sels []string
	for _, m := range msgs {
		if _, rest, ok := strings.Cut(m[1], "<selected_text>\n"); ok {
			sels = append(sels, strings.TrimSuffix(rest, "\n</selected_text>"))
		}
	}
	if strings.Join(sels, " ") != "s0 s1 s2 s3" {
		t.Errorf("selections %q in %q", sels, msgs)
	}
	v := a.cardNow()
	if len(v.bubbles) != 8 || v.bubbles[2].quote != "s1" || v.bubbles[4].quote != "s2" || v.bubbles[6].quote != "s3" {
		t.Errorf("bubbles %+v", v.bubbles)
	}
}

// TestRewriteFollowUpSelectionEscape: a follow-up with text selected,
// cancelled while it records: the earlier turns stay, nothing was asked;
// the next Continue takes the selection then.
func TestRewriteFollowUpSelectionEscape(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
	selectNow(t, path, chatB)
	before := a.sess.ID
	if !a.followUpAnswer("") {
		t.Fatal("Continue did nothing")
	}
	a.pump("the follow-up's selection", func() bool {
		return a.sess.ID != before && a.sess.State == session.Recording && a.sess.Selection == chatB
	})
	if v := a.cardNow(); len(v.bubbles) != 3 || !v.bubbles[2].live || v.bubbles[2].quote != chatB {
		t.Errorf("while it records %+v", v.bubbles)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the cancel", func() bool { return a.sess.State == session.Cancelled })
	if !a.answer.open || len(a.answer.turns) != 1 || a.answer.text != "one" || len(chat.requests()) != 1 {
		t.Errorf("after Escape %+v", a.answer)
	}
	// The cancelled recording lets the microphone go a moment later; the
	// user takes longer than that to press Continue again.
	time.Sleep(500 * time.Millisecond)
	selectNow(t, path, "")
	a.recordFollowUp(t, tr, "shorter")
	if got := lastUser(chat, 1); got != "<instruction>\nshorter\n</instruction>" {
		t.Errorf("the next follow-up %q", got)
	}
}

// TestRewriteShortcutFollowUpSelection: the Rewrite shortcut, instead of
// Continue, takes the text selected now too.
func TestRewriteShortcutFollowUpSelection(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
	selectNow(t, path, chatB)
	tr.setText("add these")
	front := helpertest.Apps["com.tinyspeck.slackmacgap"]
	first := a.sess.ID
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
	a.pump("the follow-up's recording", func() bool {
		return a.sess.ID != first && a.sess.State == session.Recording && a.sess.HeardAt.After(a.sess.StartedAt)
	})
	a.pump("the follow-up's audio", func() bool { return len(a.sess.Levels) >= 5 })
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
	a.pump("the follow-up's end", func() bool { return a.sess.State.Ended() && a.answer.text == "two" })
	if got := lastUser(chat, 1); got != "<instruction>\nadd these\n</instruction>\n\n<selected_text>\n"+chatB+"\n</selected_text>" {
		t.Errorf("follow-up %q", got)
	}
}

// TestRewriteFollowUpSelectionRetry: the model fails on a follow-up with
// text selected: Try again asks it again in the conversation, with that
// text, and the card keeps the earlier turns.
func TestRewriteFollowUpSelectionRetry(t *testing.T) {
	chat := &fakeChat{text: "one"}
	a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
	selectNow(t, path, chatB)
	chat.set("", 503)
	a.recordFollowUp(t, tr, "add the new messages")
	if a.answer.problem == nil || len(a.answer.turns) != 1 || a.answer.input.Selection != chatB {
		t.Fatalf("after the failure %+v", a.answer)
	}
	chat.set("two", 0)
	failed := a.sess.ID
	a.retryAnswer("")
	a.pump("the retry", func() bool { return a.sess.ID != failed && a.sess.State.Ended() })
	msgs := chatMessages(chat.requests()[2])
	if len(msgs) != 4 || !strings.Contains(msgs[1][1], chatA) || msgs[3][1] != "<instruction>\nadd the new messages\n</instruction>\n\n<selected_text>\n"+chatB+"\n</selected_text>" {
		t.Errorf("the retry's messages %q", msgs)
	}
	if a.answer.text != "two" || len(a.answer.turns) != 2 || a.answer.turns[1].Selection != chatB || !a.answer.conversation {
		t.Errorf("card %+v", a.answer)
	}
}
