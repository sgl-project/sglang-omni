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

// note (luojiaxuan): TestLivePolishBacksUpOverWhatWasSaidAfter: when the
// text left at Stop is several sentences, the sentence before them is
// cleaned again with them all the same: a correction in the first of them
// reaches it.
func TestLivePolishBacksUpOverWhatWasSaidAfter(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Meet Tuesday. No,")
	waitLive(t, r, "Meet Tuesday.")
	full := "Meet Tuesday. No, Thursday. Send the invite."
	got, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.UserMessage(full) || got != "["+full+"]" {
		t.Errorf("after Stop %q gave %q, want the whole text cleaned again", user, got)
	}
}

// note (luojiaxuan): TestLivePolishBacksUpIntoABlock: right after a block was
// committed, the sentence cleaned again with the text left at Stop is the
// block's last: Stop goes back to the block's clean-up of its first four
// sentences.
func TestLivePolishBacksUpIntoABlock(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	s := liveSentences(5)
	speakLive(t, r, s)
	first := strings.Join(s, " ")
	r.hearLive(first + " Thursday, no")
	waitLiveFor(t, r, "a committed block", func(lp *livePolish) bool { return len(lp.blocks) == 1 })
	full := first + " Thursday, no, Friday."
	got, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	four := strings.Join(s[:4], " ")
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.BeforeMessage("["+four+"]", s[4]+" Thursday, no, Friday.") {
		t.Errorf("the request after Stop is %q, want the block's last sentence with the rest", user)
	}
	if want := "[" + four + "] [" + s[4] + " Thursday, no, Friday.]"; got != want {
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
