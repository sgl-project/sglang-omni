package session

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
)

// note (luojiaxuan): liveSentences are n English sentences of 57 characters:
// five of them are 289, a block; a few words more are over
// compose.PieceChars.
func liveSentences(n int) []string {
	var out []string
	for i := range n {
		out = append(out, strings.Repeat("word ", 9)+"item "+string(rune('a'+i))+" ends.")
	}
	return out
}

// note (luojiaxuan): speakLive hears the sentences one by one, each cleaned
// before the next comes: with five to a block, the sixth commits the first
// five.
func speakLive(t *testing.T, r *run, sentences []string) {
	t.Helper()
	for i := range sentences {
		start := i / 5 * 5
		r.hearLive(strings.Join(sentences[:i+1], " "))
		waitLive(t, r, strings.Join(sentences[start:i+1], " "))
	}
}

// note (luojiaxuan): TestLivePolishRedoesWhatWasSaidBefore: when the text
// left at Stop is several sentences, the cleaned sentence before them goes
// back to the model with them all the same: a correction in the first of
// them reaches it.
func TestLivePolishRedoesWhatWasSaidBefore(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Meet Tuesday. No,")
	waitLive(t, r, "Meet Tuesday.")
	got, err := liveCleanUp(r, "Meet Tuesday. No, Thursday. Send the invite.")
	if err != nil {
		t.Fatal(err)
	}
	asked := "[Meet Tuesday.] No, Thursday. Send the invite."
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.UserMessage(asked) || got != "["+asked+"]" {
		t.Errorf("after Stop %q gave %q, want the cleaned sentence asked again with the rest", user, got)
	}
}

// note (luojiaxuan): TestLivePolishRedoesAcrossABlock: right after a block
// was committed, the cleaned sentences asked again with the text left at
// Stop are the block's last two, the rest of the block the context.
func TestLivePolishRedoesAcrossABlock(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	s := liveSentences(5)
	speakLive(t, r, s)
	first := strings.Join(s, " ")
	r.hearLive(first + " Thursday, no")
	waitLiveFor(t, r, "a committed block", func(lp *livePolish) bool { return len(lp.blocks) == 1 })
	got, err := liveCleanUp(r, first+" Thursday, no, Friday.")
	if err != nil {
		t.Fatal(err)
	}
	three := "[" + strings.Join(s[:3], " ")
	asked := s[3] + " " + s[4] + "] Thursday, no, Friday."
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.BeforeMessage(three, asked) {
		t.Errorf("the request after Stop is %q, want the block's last two sentences with the rest", user)
	}
	if want := three + " [" + asked + "]"; got != want {
		t.Errorf("result %q, want %q", got, want)
	}
}

// note (luojiaxuan): TestLivePolishBoundsTheContext: the third block is asked
// with the end of the cleaned text before it, the second block's answer
// (compose.PieceChars at most), not the whole dictation.
func TestLivePolishBoundsTheContext(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	s := liveSentences(15)
	speakLive(t, r, s)
	want := compose.BeforeMessage("["+strings.Join(s[5:10], " ")+"]", s[10])
	for _, req := range model.requests() {
		if inputOf(req) == s[10] {
			if user := req.Messages[len(req.Messages)-1].Content; user != want {
				t.Errorf("the third block was asked with %q, want %q", user, want)
			}
			return
		}
	}
	t.Fatal("the third block was not asked alone")
}
