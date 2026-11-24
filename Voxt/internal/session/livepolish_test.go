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
)

// liveLLM answers a clean-up with its input in brackets. A request whose
// input has a gate waits for the gate to close; answered gets each answered
// input.
type liveLLM struct {
	mu       sync.Mutex
	reqs     []llm.Request
	gates    map[string]chan struct{}
	answered chan string
}

func newLiveLLM() *liveLLM {
	return &liveLLM{gates: map[string]chan struct{}{}, answered: make(chan string, 64)}
}

// inputOf is the text between a request's <input> tags.
func inputOf(req llm.Request) string {
	msg := req.Messages[len(req.Messages)-1].Content
	_, after, _ := strings.Cut(msg, "<input>\n")
	in, _, _ := strings.Cut(after, "\n</input>")
	return in
}

func (m *liveLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	in := inputOf(req)
	m.mu.Lock()
	m.reqs = append(m.reqs, req)
	gate := m.gates[in]
	m.mu.Unlock()
	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
	}
	m.answered <- in
	return llm.Response{Text: "[" + in + "]"}, nil
}

func (m *liveLLM) gate(input string) chan struct{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	g := make(chan struct{})
	m.gates[input] = g
	return g
}

func (m *liveLLM) requests() []llm.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.Request(nil), m.reqs...)
}

// waitAnswered waits until the model answered input.
func (m *liveLLM) waitAnswered(t *testing.T, input string) {
	t.Helper()
	timeout := time.After(5 * time.Second)
	for {
		select {
		case in := <-m.answered:
			if in == input {
				return
			}
		case <-timeout:
			t.Fatalf("%q was not cleaned", input)
		}
	}
}

// waitLive waits until the live clean-up of r has a clean-up of src (open
// or committed).
func waitLive(t *testing.T, r *run, src string) {
	t.Helper()
	lp := r.live
	got := make(chan struct{})
	go func() {
		lp.mu.Lock()
		defer lp.mu.Unlock()
		for !lp.closed {
			for _, res := range append(append([]liveResult(nil), lp.blocks...), lp.done...) {
				if res.src == src {
					close(got)
					return
				}
			}
			lp.cond.Wait()
		}
	}()
	select {
	case <-got:
	case <-time.After(5 * time.Second):
		t.Fatalf("no clean-up of %q while recording", src)
	}
}

// TestLivePolishLeavesNothingAfterStop: the whole sentences of the live
// text are cleaned while the user speaks; when the final text is the text
// cleaned last, Stop asks the language model nothing and the answer goes
// in (reverted: one request after Stop).
func TestLivePolishLeavesNothingAfterStop(t *testing.T) {
	full := "Hello there. How are you?"
	model := newLiveLLM()
	h := newHarness(t, &fakeStreamer{final: full, partials: []string{"Hello there.", "Hello there. How are", full}}, model, nil)
	id := h.start()
	model.waitAnswered(t, full)
	before := len(model.requests())
	h.m.Stop(id)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "["+full+"]" || end.Problem != nil {
		t.Fatalf("end %s %q %+v, want the live answer", end.State, end.Result, end.Problem)
	}
	if n := len(model.requests()); n != before {
		t.Errorf("%d requests after Stop, want none", n-before)
	}
}

// TestLivePolishCleansOnlyTheTail: what the live text finished is kept;
// after Stop only the rest is asked, with the cleaned text before it, and
// the answers join (no sentence cleaned again: RecleanSentences 0).
func TestLivePolishCleansOnlyTheTail(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Features.Transcribe.RecleanSentences = 0
	r.hearLive("Hello there.")
	r.hearLive("Hello there. How are")
	waitLive(t, r, "Hello there.")
	got, _, err := r.cleanUpFirst("Hello there. How are you?")
	if err != nil {
		t.Fatal(err)
	}
	if want := "[Hello there.] [How are you?]"; got != want {
		t.Errorf("result %q, want %q", got, want)
	}
	reqs := model.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests, want the live one and the tail", len(reqs))
	}
	tail := reqs[1].Messages
	if user := tail[len(tail)-1].Content; user != compose.BeforeMessage("[Hello there.]", "How are you?") {
		t.Errorf("the tail's user message is %q", user)
	}
	if !strings.Contains(tail[0].Content, "<before>") {
		t.Errorf("the tail's system prompt does not say what <before> is")
	}
	if strings.Contains(reqs[0].Messages[0].Content, "<before>") || reqs[0].Messages[1].Content != compose.UserMessage("Hello there.") {
		t.Errorf("the live request is not a whole dictation's: %q", reqs[0].Messages[1].Content)
	}
}

// TestLivePolishCleansTheLastSentenceAgain: by default the sentence before
// the text left at Stop is cleaned again with it, after the clean-up of the
// text before that sentence, so that a correction said last reaches it.
func TestLivePolishCleansTheLastSentenceAgain(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Book the room. Let's")
	waitLive(t, r, "Book the room.")
	r.hearLive("Book the room. Let's meet Tuesday.")
	waitLive(t, r, "Book the room. Let's meet Tuesday.")
	r.hearLive("Book the room. Let's meet Tuesday. No, make")
	got, _, err := r.cleanUpFirst("Book the room. Let's meet Tuesday. No, make it Thursday.")
	if err != nil {
		t.Fatal(err)
	}
	if want := "[Book the room.] [Let's meet Tuesday. No, make it Thursday.]"; got != want {
		t.Errorf("result %q, want %q", got, want)
	}
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.BeforeMessage("[Book the room.]", "Let's meet Tuesday. No, make it Thursday.") {
		t.Errorf("the request after Stop is %q, want the last two sentences after the first", user)
	}
}

// TestLivePolishAsksAgainWhenRevised: a sentence the speech model changes
// after it was sent is cleaned again; the answer of the old one never goes
// in.
func TestLivePolishAsksAgainWhenRevised(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello their. How")
	waitLive(t, r, "Hello their.")
	r.hearLive("Hello there. How are you?")
	waitLive(t, r, "Hello there. How are you?")
	got, _, err := r.cleanUpFirst("Hello there. How are you?")
	if err != nil {
		t.Fatal(err)
	}
	if want := "[Hello there. How are you?]"; got != want {
		t.Errorf("result %q, want %q", got, want)
	}
	if n := len(model.requests()); n != 2 {
		t.Errorf("%d requests, want the first sentence and the revised text, none after Stop", n)
	}
}

// TestLivePolishFinalDiffers: when the final text changed the first
// sentence, nothing cleaned while recording applies: the whole text is
// cleaned after Stop, as without the live clean-up.
func TestLivePolishFinalDiffers(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello their. How are you?")
	waitLive(t, r, "Hello their. How are you?")
	full := "Hello there. How are you?"
	got, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	if got != "["+full+"]" {
		t.Errorf("result %q", got)
	}
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if len(reqs) != 2 || last[len(last)-1].Content != compose.UserMessage(full) {
		t.Errorf("%d requests, the last %q: want the whole text after Stop", len(reqs), last[len(last)-1].Content)
	}
}

// TestLivePolishWaitsForTheRequestUnderWay: at Stop, a request already
// under way for exactly the final text is awaited, not asked again.
func TestLivePolishWaitsForTheRequestUnderWay(t *testing.T) {
	full := "Hello there. How are you?"
	model := newLiveLLM()
	release := model.gate(full)
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello there.")
	waitLive(t, r, "Hello there.")
	r.hearLive(full)
	// Stop comes once the request for the whole text is under way.
	waitStarted := time.After(5 * time.Second)
	for len(model.requests()) < 2 {
		select {
		case <-waitStarted:
			t.Fatal("the request for the whole text did not start")
		case <-time.After(time.Millisecond):
		}
	}
	result := make(chan string, 1)
	go func() {
		got, _, _ := r.cleanUpFirst(full)
		result <- got
	}()
	// The answer comes once Stop has taken the final text (finish closed
	// the live clean-up and waits for the request).
	for closed := false; !closed; {
		select {
		case <-waitStarted:
			t.Fatal("the final text was not taken")
		case <-time.After(time.Millisecond):
		}
		r.live.mu.Lock()
		closed = r.live.closed
		r.live.mu.Unlock()
	}
	close(release)
	select {
	case got := <-result:
		if got != "["+full+"]" {
			t.Errorf("result %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the clean-up did not end")
	}
	if n := len(model.requests()); n != 2 {
		t.Errorf("%d requests, want 2: the request under way is awaited", n)
	}
}

// TestLivePolishBlocks: past compose.PieceChars characters the text so far
// is kept as a block, and the next block is asked with the cleaned text
// before it.
func TestLivePolishBlocks(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	var sentences []string
	for i := range 7 {
		sentences = append(sentences, strings.Repeat("word ", 9)+"number "+string(rune('a'+i))+".")
	}
	first := strings.Join(sentences[:5], " ") // 290 characters: one block
	second := strings.Join(sentences[5:], " ")
	full := first + " " + second
	r.hearLive(first)
	waitLive(t, r, first)
	r.hearLive(first + " " + sentences[5])
	r.hearLive(full)
	waitLive(t, r, second)
	got, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	if want := "[" + first + "] [" + second + "]"; got != want {
		t.Errorf("result %q, want %q", got, want)
	}
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.BeforeMessage("["+first+"]", second) {
		t.Errorf("the second block's request is %q", user)
	}
}

// TestLivePolishOff: with Clean up while you speak off, nothing goes to the
// language model before Stop: neither the live text nor the speech model's
// own stretches.
func TestLivePolishOff(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Features.Transcribe.CleanAfterStop = true
	r.hearLive("Hello there.")
	r.hearPiece("Hello there.")
	if r.live != nil || r.early != nil || len(model.requests()) != 0 {
		t.Fatalf("the live clean-up ran with the switch off")
	}
	if _, _, err := r.cleanUpFirst("Hello there."); err != nil {
		t.Fatal(err)
	}
	if n := len(model.requests()); n != 1 {
		t.Errorf("%d requests, want 1 after Stop", n)
	}
}

// TestLivePolishThinking: a clean-up asked with thinking off (the text then
// shorter than ThinkingOffUnder) does not count for a final text long
// enough for thinking.
func TestLivePolishThinking(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Providers.LLM = []settings.Provider{{ID: "deepseek", Generation: settings.Generation{ThinkingOffUnder: 20}}}
	r.hearLive("Short one. And")
	waitLive(t, r, "Short one.")
	full := "Short one. And a longer second sentence."
	got, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	if got != "["+full+"]" {
		t.Errorf("result %q, want the whole text cleaned after Stop", got)
	}
}

// TestLivePolishEndsWithPieces: once the speech model reports stretches of
// its own, the live clean-up stops and they are cleaned instead.
func TestLivePolishEndsWithPieces(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello there.")
	waitLive(t, r, "Hello there.")
	r.hearPiece("Hello there.")
	r.hearLive("Hello there. How are you?")
	model.waitAnswered(t, "Hello there.")
	if got, _, err := r.cleanUpFirst("Hello there. How are you?"); err != nil || got != "[Hello there.] [How are you?]" {
		t.Errorf("result %q (%v), want the stretch's answer and the tail's", got, err)
	}
}
