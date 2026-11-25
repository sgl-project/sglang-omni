package session

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// note (luojiaxuan): liveLLM answers a clean-up with its input in brackets. A request whose
// input has a gate waits for the gate to close; answered gets each answered
// input.
type liveLLM struct {
	mu       sync.Mutex
	reqs     []llm.Request
	gates    map[string]chan struct{}
	stuck    map[string]chan struct{} // note (luojiaxuan): inputs whose request ignores its cancellation until the channel closes, then answers
	answered chan string
}

func newLiveLLM() *liveLLM {
	return &liveLLM{gates: map[string]chan struct{}{}, stuck: map[string]chan struct{}{}, answered: make(chan string, 64)}
}

// note (luojiaxuan): liveCleanUp is r's first clean-up of the final text:
// its text and error.
func liveCleanUp(r *run, text string) (string, error) {
	r.m.mu.Lock()
	if r.m.cur == nil {
		r.m.cur = r
		r.m.rev++
		r.m.snap = Snapshot{Rev: r.m.rev, ID: r.id, Kind: store.KindTranscribe, State: Composing}
	}
	r.m.mu.Unlock()
	out := r.cleanUpFirst(text)
	return out.text, out.err
}

// note (luojiaxuan): inputOf is the text between a request's <input> tags.
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
	gate, stuck := m.gates[in], m.stuck[in]
	m.mu.Unlock()
	if stuck != nil {
		<-stuck
	} else if gate != nil {
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

// note (luojiaxuan): waitAnswered waits until the model answered input.
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

// note (luojiaxuan): waitLive waits until the live clean-up of r has a
// clean-up of src (open or committed).
func waitLive(t *testing.T, r *run, src string) {
	t.Helper()
	waitLiveFor(t, r, "clean-up of "+src, func(lp *livePolish) bool {
		return slices.ContainsFunc(append(slices.Clone(lp.blocks), lp.done...), func(res liveResult) bool { return res.src == src })
	})
}

// note (luojiaxuan): waitLiveFor waits until ok holds of the live clean-up
// of r (under its lock, at each change), failing when it ends first.
func waitLiveFor(t *testing.T, r *run, what string, ok func(*livePolish) bool) {
	t.Helper()
	lp := r.live
	got := make(chan bool, 1)
	go func() {
		lp.mu.Lock()
		defer lp.mu.Unlock()
		for !ok(lp) {
			if lp.closed {
				got <- false
				return
			}
			lp.cond.Wait()
		}
		got <- true
	}()
	select {
	case found := <-got:
		if !found {
			t.Fatalf("the live clean-up ended without %s", what)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("no %s while recording", what)
	}
}

// note (luojiaxuan): TestLivePolishLeavesNothingAfterStop: the whole sentences of the live
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

// note (luojiaxuan): TestLivePolishCleansOnlyTheTail: what the live text finished is kept;
// after Stop only the rest is asked, with the cleaned text before it, and
// the answers join (no sentence cleaned again: RecleanSentences 0).
func TestLivePolishCleansOnlyTheTail(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Features.Transcribe.RecleanSentences = 0
	r.hearLive("Hello there.")
	r.hearLive("Hello there. How are")
	waitLive(t, r, "Hello there.")
	got, err := liveCleanUp(r, "Hello there. How are you?")
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

// note (luojiaxuan): TestLivePolishCleansTheLastSentencesAgain: by default
// the last two sentences of the cleaned text go back to the model with the
// text left at Stop, so that a correction said last reaches them; the
// cleaned text before them is the context.
func TestLivePolishCleansTheLastSentencesAgain(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	said := "Book the room. Bring the slides. Let's meet Tuesday."
	r.hearLive("Book the room. Bring the slides. Let's")
	waitLive(t, r, "Book the room. Bring the slides.")
	r.hearLive(said)
	waitLive(t, r, said)
	r.hearLive(said + " No, make")
	got, err := liveCleanUp(r, said+" No, make it Thursday.")
	if err != nil {
		t.Fatal(err)
	}
	asked := "Bring the slides. Let's meet Tuesday.] No, make it Thursday."
	if want := "[Book the room. [" + asked + "]"; got != want {
		t.Errorf("result %q, want %q", got, want)
	}
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.BeforeMessage("[Book the room.", asked) {
		t.Errorf("the request after Stop is %q, want the last two cleaned sentences with the rest", user)
	}
}

// note (luojiaxuan): TestLivePolishAsksAgainWhenRevised: a sentence the speech model changes
// after it was sent is cleaned again; the answer of the old one never goes
// in.
func TestLivePolishAsksAgainWhenRevised(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello their. How")
	waitLive(t, r, "Hello their.")
	r.hearLive("Hello there. How are you?")
	waitLive(t, r, "Hello there. How are you?")
	got, err := liveCleanUp(r, "Hello there. How are you?")
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

// note (luojiaxuan): TestLivePolishFinalDiffers: when the final text changed the first
// sentence, nothing cleaned while recording applies: the whole text is
// cleaned after Stop, as without the live clean-up.
func TestLivePolishFinalDiffers(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello their. How are you?")
	waitLive(t, r, "Hello their. How are you?")
	full := "Hello there. How are you?"
	got, err := liveCleanUp(r, full)
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

// note (luojiaxuan): TestLivePolishWaitsForTheRequestUnderWay: at Stop, a request already
// under way for exactly the final text is awaited, not asked again.
func TestLivePolishWaitsForTheRequestUnderWay(t *testing.T) {
	full := "Hello there. How are you?"
	model := newLiveLLM()
	release := model.gate(full)
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello there.")
	waitLive(t, r, "Hello there.")
	r.hearLive(full)
	// note (luojiaxuan): Stop comes once the request for the whole text is under way.
	waitLiveFor(t, r, "the request for the whole text", func(lp *livePolish) bool { return lp.flight != nil && lp.flight.src == full })
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, full)
		result <- got
	}()
	// note (luojiaxuan): The answer comes once Stop has taken the final text (finish closed
	// the live clean-up and waits for the request).
	waitLiveFor(t, r, "the final text taken", func(lp *livePolish) bool { return lp.closed })
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

// note (luojiaxuan): TestLivePolishBlocks: past compose.PieceChars characters the text so far
// is kept as a block, and the next block is asked with the cleaned text
// before it.
func TestLivePolishBlocks(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	var sentences []string
	for i := range 7 {
		sentences = append(sentences, strings.Repeat("word ", 9)+"number "+string(rune('a'+i))+".")
	}
	first := strings.Join(sentences[:5], " ") // note (luojiaxuan): 274 characters: one block
	second := strings.Join(sentences[5:], " ")
	full := first + " " + second
	r.hearLive(first)
	waitLive(t, r, first)
	r.hearLive(first + " " + sentences[5])
	r.hearLive(full)
	waitLive(t, r, second)
	got, err := liveCleanUp(r, full)
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

// note (luojiaxuan): TestLivePolishOff: nothing goes to the language model
// before Stop with Clean up while you speak off (neither the live text nor
// the speech model's own stretches), when a picture goes with the clean-up,
// or when the language model runs on this Mac.
func TestLivePolishOff(t *testing.T) {
	for _, c := range []struct {
		name string
		set  func(r *run)
	}{
		{"switch off", func(r *run) { r.settings.Features.Transcribe.CleanAfterStop = true }},
		{"window picture", func(r *run) { r.shot = func() []byte { return []byte("jpeg") } }},
		{"model on this Mac", func(r *run) { r.svc.LLMInfo.ID = "ollama" }},
	} {
		t.Run(c.name, func(t *testing.T) {
			model := newLiveLLM()
			r := newEarlyUnitRun(t, model, nil)
			c.set(r)
			r.hearLive("Hello there.")
			if c.name == "switch off" {
				r.hearPiece("Hello there.")
			}
			if r.live != nil || r.early != nil || len(model.requests()) != 0 {
				t.Fatalf("the clean-up ran while recording")
			}
			if _, err := liveCleanUp(r, "Hello there."); err != nil {
				t.Fatal(err)
			}
			if n := len(model.requests()); n != 1 {
				t.Errorf("%d requests, want 1 after Stop", n)
			}
		})
	}
}

// note (luojiaxuan): TestLivePolishThinking: a clean-up asked with thinking off (the text then
// shorter than ThinkingOffUnder) does not count for a final text long
// enough for thinking.
func TestLivePolishThinking(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Providers.LLM = []settings.Provider{{ID: "deepseek", Generation: settings.Generation{ThinkingOffUnder: 20}}}
	r.hearLive("Short one. And")
	waitLive(t, r, "Short one.")
	full := "Short one. And a longer second sentence."
	got, err := liveCleanUp(r, full)
	if err != nil {
		t.Fatal(err)
	}
	if got != "["+full+"]" {
		t.Errorf("result %q, want the whole text cleaned after Stop", got)
	}
}

// note (luojiaxuan): TestLivePolishEndsWithPieces: once the speech model reports stretches of
// its own, the live clean-up stops and they are cleaned instead.
func TestLivePolishEndsWithPieces(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello there.")
	waitLive(t, r, "Hello there.")
	r.hearPiece("Hello there.")
	r.hearLive("Hello there. How are you?")
	model.waitAnswered(t, "Hello there.")
	if got, err := liveCleanUp(r, "Hello there. How are you?"); err != nil || got != "[Hello there.] [How are you?]" {
		t.Errorf("result %q (%v), want the stretch's answer and the tail's", got, err)
	}
}

// note (luojiaxuan): TestLivePolishStopsOnFailure: a request the service
// refuses ends the live clean-up after that one request (reverted, the
// worker asked it again at once, as a cancellation), and Stop cleans the
// whole text: what was cleaned before the failure is dropped too.
func TestLivePolishStopsOnFailure(t *testing.T) {
	model := &fakeLLM{text: "clean", err: errors.New("401 unauthorized"), failOn: 2, failDelta: "private failed text"}
	h, r := newEarlyUnitHarnessRun(t, model, nil)
	r.settings.Features.Transcribe.RecleanSentences = 0
	r.hearLive("Hello there. How")
	waitLive(t, r, "Hello there.")
	r.hearLive("Hello there. How are you?")
	waitLiveFor(t, r, "the end of the live clean-up", func(lp *livePolish) bool { return lp.closed })
	if n := len(model.requests()); n != 2 {
		t.Fatalf("%d requests while recording, want 2: the refused one is not asked again", n)
	}
	if draft := h.m.Current().Draft; draft != "" {
		t.Fatalf("failed private text was published as Draft %q", draft)
	}
	full := "Hello there. How are you? Fine."
	result, err := liveCleanUp(r, full)
	if err != nil {
		t.Fatal(err)
	}
	if result != "clean" {
		t.Fatalf("full-text cleanup %q, want clean", result)
	}
	reqs := model.requests()
	last := reqs[len(reqs)-1].Messages
	if user := last[len(last)-1].Content; user != compose.UserMessage(full) {
		t.Errorf("the request after Stop is %q, want the whole text", user)
	}
	for _, snapshot := range h.rec.all() {
		if strings.Contains(snapshot.Draft, "private failed text") {
			t.Fatalf("failed private text reached a session snapshot: %+v", snapshot)
		}
	}
}

// note (luojiaxuan): TestLivePolishLeavesATriggerAlone: a dictation that is a
// snippet's trigger alone never goes to the language model, while it is
// spoken either: the snippet is typed as saved.
func TestLivePolishLeavesATriggerAlone(t *testing.T) {
	model := newLiveLLM()
	h := newHarness(t, &fakeStreamer{final: "My email address.", partials: []string{"My email", "My email address."}}, model, withSnippets)
	id := h.start()
	h.speak(id, 50)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "ruiling@example.com" {
		t.Fatalf("%s %q, want the snippet", end.State, end.Result)
	}
	if n := len(model.requests()); n != 0 {
		t.Errorf("%d requests, want none for a trigger alone", n)
	}
}

// note (luojiaxuan): TestLivePolishCountsEachRequest: a block in two
// languages is asked in two parts (compose.LanguagePieces), and the Stop
// summary counts both requests.
func TestLivePolishCountsEachRequest(t *testing.T) {
	model := newLiveLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello there. 你好。")
	waitLive(t, r, "Hello there. 你好。")
	r.live.mu.Lock()
	asked := r.live.asked
	r.live.mu.Unlock()
	if n := len(model.requests()); n != 2 || asked != n {
		t.Errorf("%d requests, %d counted; want 2 and 2", n, asked)
	}
}

// note (luojiaxuan): TestLivePolishDoesNotWaitForAStuckRequest: a request
// under way for text the final text does not have is cancelled at Stop and
// not waited for, even when it ignores the cancellation: Stop cleans the
// rest at once (reverted, it waited for the request to return).
func TestLivePolishDoesNotWaitForAStuckRequest(t *testing.T) {
	model := newLiveLLM()
	stuck := make(chan struct{})
	t.Cleanup(func() { close(stuck) })
	model.mu.Lock()
	model.stuck["Hello there. How are you?"] = stuck
	model.mu.Unlock()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Features.Transcribe.RecleanSentences = 0
	r.hearLive("Hello there. How")
	waitLive(t, r, "Hello there.")
	r.hearLive("Hello there. How are you?")
	waitLiveFor(t, r, "the request for the whole text", func(lp *livePolish) bool {
		return lp.flight != nil && lp.flight.src == "Hello there. How are you?"
	})
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, "Hello there. How are you doing?")
		result <- got
	}()
	select {
	case got := <-result:
		if want := "[Hello there.] [How are you doing?]"; got != want {
			t.Errorf("result %q, want %q", got, want)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Stop waited for the stuck request")
	}
}

// note (luojiaxuan): TestLivePolishStopsBetweenParts: a clean-up in two parts
// (two languages) that Stop cancels during its first part never asks the
// second, even when the first answers after the cancellation.
func TestLivePolishStopsBetweenParts(t *testing.T) {
	model := newLiveLLM()
	stuck := make(chan struct{})
	model.mu.Lock()
	model.stuck["Hi there."] = stuck
	model.mu.Unlock()
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Features.Transcribe.RecleanSentences = 0
	live := "Hi there. 你好。"
	r.hearLive(live)
	var flight *liveFlight
	waitLiveFor(t, r, "the first part's request", func(lp *livePolish) bool { flight = lp.flight; return flight != nil })
	if _, err := liveCleanUp(r, "Hello there, friend. 你好。"); err != nil {
		t.Fatal(err)
	}
	close(stuck)
	select {
	case <-flight.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the cancelled clean-up did not end")
	}
	chars := utf8.RuneCountInString(live)
	for _, req := range model.requests() {
		if req.InputChars == chars && inputOf(req) != "Hi there." {
			t.Errorf("after Stop the live clean-up asked its next part %q", inputOf(req))
		}
	}
}
