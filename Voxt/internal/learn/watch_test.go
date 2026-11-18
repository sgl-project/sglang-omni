package learn

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf16"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// fieldRead is what the field gives on one textio.read.
type fieldRead struct {
	value   string
	element int
	err     string // proto.Read*: nothing read
	failed  error  // the call itself fails
}

func saw(value string) fieldRead { return fieldRead{value: value, element: 1} }

var (
	fieldNotFront = fieldRead{err: proto.ReadNotFront}
	fieldNoFocus  = fieldRead{err: proto.ReadNoFocus}
	fieldSecure   = fieldRead{err: proto.ReadSecure}
)

// fakeField answers Reader with the script; the last entry repeats. It
// clamps ranges as the helper does.
type fakeField struct {
	mu     sync.Mutex
	script []fieldRead
	calls  []proto.ReadParams
}

func (f *fakeField) Read(ctx context.Context, p proto.ReadParams) (proto.ReadResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return proto.ReadResult{}, err
	}
	r := f.script[min(len(f.calls), len(f.script)-1)]
	f.calls = append(f.calls, p)
	if r.failed != nil {
		return proto.ReadResult{}, r.failed
	}
	if r.err != "" {
		return proto.ReadResult{Error: r.err}, nil
	}
	units := utf16.Encode([]rune(r.value))
	start := min(max(0, p.Range.Location), len(units))
	end := min(max(start, p.Range.Location+p.Range.Length), len(units), start+proto.MaxRead)
	return proto.ReadResult{Text: string(utf16.Decode(units[start:end])), Range: proto.TextRange{Location: start, Length: end - start},
		Length: len(units), Element: r.element}, nil
}

func (f *fakeField) reads() []proto.ReadParams {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]proto.ReadParams(nil), f.calls...)
}

// fakeClock is Learner.Sleep: time passes when asked, and actions fire
// when it reaches them.
type fakeClock struct {
	now time.Duration
	at  map[time.Duration]func()
}

func (c *fakeClock) sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.now += d
	for when, fn := range c.at {
		if c.now >= when {
			delete(c.at, when)
			fn()
		}
	}
	return ctx.Err()
}

// learnTest is a Learner over a fake field and clock and a real Book.
type learnTest struct {
	*Learner
	field *fakeField
	clock *fakeClock
	book  *Book
	st    interface{ Dictionary() store.Dictionary }
	logs  *[]string
}

func newLearnTest(t *testing.T, script ...fieldRead) *learnTest {
	t.Helper()
	b, st, _ := newBook(t)
	f := &fakeField{script: script}
	c := &fakeClock{at: map[time.Duration]func(){}}
	logs := &[]string{}
	var logged sync.Mutex
	l := &Learner{Read: f, Book: b, Log: func(level, msg string) {
		logged.Lock()
		defer logged.Unlock()
		*logs = append(*logs, level+" "+msg)
	}, Sleep: c.sleep}
	return &learnTest{Learner: l, field: f, clock: c, book: b, st: st, logs: logs}
}

const insertedText = "open Go Host now"

// inField is a value of the field with the dictated text at position 10.
func inField(text string) string { return "0123456789" + text + " tail" }

func insertion() Insertion {
	return Insertion{Text: insertedText, Landing: deliver.Landing{App: proto.App{PID: 42}, At: 10, Role: "AXTextArea"}}
}

// settle is the time of the baseline read (Learner's first wait).
const settleAt = 900 * time.Millisecond

// afterPoll is the clock's reading poll number n (1 s apart after the baseline).
func afterPoll(n int) time.Duration { return settleAt + time.Duration(n)*time.Second }

func TestWatch(t *testing.T) {
	base, edit := saw(inField(insertedText)), saw(inField("open Ghostty now"))
	deleted := saw(inField("open  now"))
	unchanged := []fieldRead{base, base}
	for _, tc := range []struct {
		name   string
		script []fieldRead
		reads  int
		term   string // "" : nothing proposed
	}{
		{name: "testStableFocusedEditDoesNotFinalizeObservationImmediately: edit at second 3, read to second 30",
			script: []fieldRead{base, base, base, edit}, reads: 31, term: "Ghostty"},
		{name: "left the app after the edit, idle 4 s: judged",
			script: []fieldRead{base, edit, edit, edit, edit, fieldNotFront, fieldNotFront, fieldNotFront}, reads: 8, term: "Ghostty"},
		{name: "left the app twice and came back: the watch goes on",
			script: []fieldRead{base, edit, fieldNotFront, fieldNotFront, edit}, reads: 31, term: "Ghostty"},
		{name: "unreadable three times before any edit: nothing",
			script: []fieldRead{base, fieldNotFront, fieldNoFocus, {failed: errors.New("helper gone")}}, reads: 4},
		{name: "one good read clears the count",
			script: []fieldRead{base, fieldNotFront, fieldNotFront, base, fieldNotFront, fieldNotFront, base, fieldNotFront, fieldNotFront, edit}, reads: 31, term: "Ghostty"},
		{name: "only deleted: keeps waiting and judges nothing",
			script: []fieldRead{base, deleted, deleted, deleted, deleted, fieldNotFront, fieldNotFront, fieldNotFront, deleted}, reads: 31},
		{name: "deleted, then the replacement written",
			script: []fieldRead{base, deleted, deleted, edit}, reads: 31, term: "Ghostty"},
		{name: "secure ends the watch and drops the edit",
			script: []fieldRead{base, edit, fieldSecure}, reads: 3},
		{name: "secure on the first read", script: []fieldRead{fieldSecure}, reads: 1},
		{name: "another element counts as unreadable",
			script: []fieldRead{base, {value: inField("other"), element: 2}, {value: inField("other"), element: 2}, {value: inField("other"), element: 2}}, reads: 4},
		{name: "baseline retried three times, then given up",
			script: []fieldRead{fieldNotFront, fieldNoFocus, {err: proto.ReadUnreadable}}, reads: 3},
		{name: "baseline on the third try still works",
			script: []fieldRead{fieldNotFront, fieldNotFront, base, edit}, reads: 33, term: "Ghostty"},
		{name: "the pasted text is not in the field: given up after one read",
			script: []fieldRead{saw(inField("the app rewrote it")), edit}, reads: 1},
		{name: "no edit in 30 s: nothing", script: unchanged, reads: 31},
		{name: "edit outside the inserted text: nothing",
			script: []fieldRead{base, saw("0123456789" + insertedText + " tailed")}, reads: 31},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lt := newLearnTest(t, tc.script...)
			n := lt.Learn(context.Background(), insertion())
			if got := len(lt.field.reads()); got != tc.reads {
				t.Errorf("%d reads, want %d", got, tc.reads)
			}
			got := pendingTerms(t, lt.book)
			if tc.term == "" {
				if n != 0 || len(got) != 0 {
					t.Fatalf("Learn %d, proposed %v", n, got)
				}
				return
			}
			if n != 1 || len(got) != 1 || got[0] != tc.term {
				t.Fatalf("Learn %d, proposed %v, want %s", n, got, tc.term)
			}
		})
	}
}

func TestLearnNeverReads(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*Insertion)
	}{
		{"testAutoKeyPressSkipsObservationScheduling", func(i *Insertion) { i.Pressed = true }},
		{"the position is unknown", func(i *Insertion) { i.At = -1 }},
		{"nothing but spaces was dictated", func(i *Insertion) { i.Text = "   " }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lt := newLearnTest(t, saw("anything"))
			ins := insertion()
			tc.edit(&ins)
			if lt.Learn(context.Background(), ins) != 0 || len(lt.field.reads()) != 0 || lt.clock.now != 0 {
				t.Fatalf("reads %v, clock %v", lt.field.reads(), lt.clock.now)
			}
		})
	}
	// testTranscriptionWithoutAutoKeyPressSchedulesObservation: no skip reads
	// at Settle, the first read.
	lt := newLearnTest(t, saw(inField(insertedText)))
	lt.Learn(context.Background(), insertion())
	if got := lt.field.reads(); len(got) == 0 || lt.clock.now == 0 {
		t.Fatalf("Learn read nothing: %v, clock %v", got, lt.clock.now)
	}
}

func TestLearnFinishEndsNowAndJudges(t *testing.T) {
	lt := newLearnTest(t, saw(inField(insertedText)), saw(inField("open Ghostty now")))
	lt.clock.at[afterPoll(3)] = lt.Finish
	if n := lt.Learn(context.Background(), insertion()); n != 1 {
		t.Fatalf("Learn %d", n)
	}
	if got := len(lt.field.reads()); got != 3 {
		t.Errorf("%d reads, want 3: none after Finish", got)
	}
	if got := pendingTerms(t, lt.book); len(got) != 1 || got[0] != "Ghostty" {
		t.Errorf("proposed %v", got)
	}
}

func TestLearnStopAndCancelDrop(t *testing.T) {
	for _, tc := range []struct {
		name string
		end  func(*learnTest, context.CancelFunc)
	}{
		{"Stop", func(lt *learnTest, _ context.CancelFunc) { lt.Stop() }},
		{"ctx", func(_ *learnTest, cancel context.CancelFunc) { cancel() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lt := newLearnTest(t, saw(inField(insertedText)), saw(inField("open Ghostty now")))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			lt.clock.at[afterPoll(2)] = func() { tc.end(lt, cancel) }
			if n := lt.Learn(ctx, insertion()); n != 0 {
				t.Fatalf("Learn %d", n)
			}
			if got := len(lt.field.reads()); got != 2 {
				t.Errorf("%d reads, want 2: none after the end", got)
			}
			if got := pendingTerms(t, lt.book); len(got) != 0 {
				t.Errorf("proposed %v", got)
			}
		})
	}
}

func TestLearnChecksTheDictionary(t *testing.T) {
	put := func(t *testing.T, lt *learnTest, term string, aliases ...string) {
		t.Helper()
		if err := lt.st.Dictionary().Put(context.Background(), store.Term{ID: term, Term: term, Aliases: aliases, Enabled: true}); err != nil {
			t.Fatal(err)
		}
	}
	base, edit := saw(inField(insertedText)), saw(inField("open Ghostty now"))
	for _, tc := range []struct {
		name string
		seed func(*testing.T, *learnTest)
	}{
		{"the correction is already a term", func(t *testing.T, lt *learnTest) { put(t, lt, "Ghostty") }},
		{"the correction is an alias", func(t *testing.T, lt *learnTest) { put(t, lt, "Terminal", "Ghostty") }},
		{"what the user changed is a dictionary term", func(t *testing.T, lt *learnTest) { put(t, lt, "Go Host") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lt := newLearnTest(t, base, edit)
			tc.seed(t, lt)
			if n := lt.Learn(context.Background(), insertion()); n != 0 {
				t.Fatalf("Learn %d, proposed %v", n, pendingTerms(t, lt.book))
			}
		})
	}
}

func TestReadRange(t *testing.T) {
	const long = "0123456789" // 10 units before the text
	base := saw(long + insertedText + " end")
	grown := saw(long + insertedText + " end and then many more typed words")
	lt := newLearnTest(t, base, grown, grown, grown)
	lt.Learn(context.Background(), insertion())
	reads := lt.field.reads()
	if len(reads) < 3 {
		t.Fatalf("reads %v", reads)
	}
	if reads[0].PID != 42 || reads[0].Range.Location != 0 {
		t.Errorf("baseline read %+v: at most Pad before", reads[0])
	}
	want := utf16Len(long+insertedText) + watchPad
	if reads[0].Range.Length != want {
		t.Errorf("baseline length %d, want %d", reads[0].Range.Length, want)
	}
	base16 := utf16Len(base.value) // the baseline read's clamped range ended here
	if got := reads[1].Range; got.Location != 0 || got.Length != base16 {
		t.Errorf("the second read %+v follows the field's length as it was, %d", got, base16)
	}
	grew := utf16Len(grown.value) - base16
	if got := reads[2].Range; got.Location != 0 || got.Length != base16+grew {
		t.Errorf("the third read %+v: start kept, length grows by %d from %d", got, grew, base16)
	}
}

func TestLogNeverHasTheText(t *testing.T) {
	const correction = "Ghostty"
	lt := newLearnTest(t, saw(inField(insertedText)), saw(inField("open Ghostty now")))
	lt.Learn(context.Background(), insertion())
	if len(*lt.logs) == 0 {
		t.Fatal("no log lines")
	}
	for _, line := range *lt.logs {
		for _, text := range []string{insertedText, "Go Host", correction, "open", "tail", "0123456789"} {
			if strings.Contains(line, text) {
				t.Errorf("log line %q has %q", line, text)
			}
		}
	}
}

// TestLearnLaterWatchEndsThePreviousOne: a new insertion while another is
// watched judges that one; the later watch stays the current one, so
// Stop still drops it. Two goroutines, in a fixed order by channels.
func TestLearnLaterWatchEndsThePreviousOne(t *testing.T) {
	first, firstEdit := saw(inField(insertedText)), saw(inField("open Ghostty now"))
	secondInsertion := Insertion{Text: "use Wechart here", Landing: deliver.Landing{App: proto.App{PID: 43}, At: 0}}
	lt := newLearnTest(t, first, firstEdit)
	second := &fakeField{script: []fieldRead{saw("use Wechart here"), saw("use WeChat here")}}
	lt.Read = byPID{42: lt.field, 43: second}
	firstHolding := make(chan struct{})
	secondHolding := make(chan struct{})
	var once sync.Once
	var sleeps sync.Map
	lt.Sleep = func(ctx context.Context, d time.Duration) error {
		pid := ctx.Value(pidKey{})
		n, _ := sleeps.LoadOrStore(pid, new(int))
		count := n.(*int)
		*count++
		if *count == 3 { // settle, one poll, then wait here until the watch is ended
			switch pid {
			case 42:
				once.Do(func() { close(firstHolding) })
			case 43:
				close(secondHolding)
			}
			<-ctx.Done()
			return ctx.Err()
		}
		return ctx.Err()
	}
	firstDone := make(chan int)
	go func() { firstDone <- lt.Learn(context.WithValue(context.Background(), pidKey{}, 42), insertion()) }()
	<-firstHolding
	secondDone := make(chan int)
	go func() {
		secondDone <- lt.Learn(context.WithValue(context.Background(), pidKey{}, 43), secondInsertion)
	}()
	if n := <-firstDone; n != 1 {
		t.Fatalf("the first watch proposed %d", n)
	}
	<-secondHolding
	lt.Stop()
	if n := <-secondDone; n != 0 {
		t.Fatalf("Stop kept the second watch: %d", n)
	}
	if got := pendingTerms(t, lt.book); len(got) != 1 || got[0] != "Ghostty" {
		t.Fatalf("proposed %v", got)
	}
}

// byPID routes reads to the field of the app.
type byPID map[int]*fakeField

func (b byPID) Read(ctx context.Context, p proto.ReadParams) (proto.ReadResult, error) {
	return b[p.PID].Read(ctx, p)
}

type pidKey struct{}

// TestWatchStateMissing is 1.x's observeMissingSnapshot table.
func TestWatchStateMissing(t *testing.T) {
	const idle = watchIdle + 100*time.Millisecond
	const before = watchIdle - 100*time.Millisecond
	baseline := "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
	edited := "我们配合 Ghostty 来实现 Terminal CLI 的输入。"
	incomplete := "特别是 Claude Code 和我们的 。我们为了识别 Terminal CLI，做了一些优化。"
	incompleteBaseline := "特别是 Cloud Code 和我们的 Go Host。我们为了识别 Terminal CLI，做了一些优化。"
	for _, tc := range []struct {
		name  string
		state watchState
		idle  time.Duration
		last  watchDecision // the decision of the third miss
	}{
		{"testObservationStopsAfterConsecutiveMissingSnapshotsBeforeAnyEdit", watchState{baseline: "baseline", latest: "baseline"}, 0, watchGiveUp},
		{"testObservationSettlesAfterConsecutiveMissingSnapshotsOnceEditIsIdle", watchState{baseline: baseline, latest: edited, changed: true}, idle, watchSettle},
		{"testObservationDoesNotSettleAfterMissingSnapshotsBeforeIdleThreshold", watchState{baseline: baseline, latest: edited, changed: true}, before, watchGoOn},
		{"testObservationDoesNotSettleAfterMissingSnapshotsWhenReplacementStillIncomplete", watchState{baseline: incompleteBaseline, latest: incomplete, changed: true}, idle, watchGoOn},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := tc.state
			for i, want := range []watchDecision{watchGoOn, watchGoOn, tc.last} {
				if got := state.missed(tc.idle); got != want {
					t.Fatalf("miss %d: got %v, want %v", i+1, got, want)
				}
			}
		})
	}
}

// TestWatchStateFocused is 1.x's observeSnapshot rows: a focused field
// never ends the watch; whether the edit is judged shows in its end.
func TestWatchStateFocused(t *testing.T) {
	baseline := "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
	replaced := "我们配合 Ghostty 来实现 Terminal CLI 的输入。"
	deleted := "我们配合  来实现 Terminal CLI 的输入。"
	groupBaseline := "你好，你帮我看一下我们的微信中有没有新的消息，特别是 Cloud Code 和我们的 Go Host。我们为了识别 Terminal CLI，做了一些优化。"
	groupLatest := "你好，你帮我看一下我们的微信中有没有新的消息，特别是 Claude Code 和我们的 。我们为了识别 Terminal CLI，做了一些优化。"
	for _, tc := range []struct {
		name     string
		baseline string
		latest   string
		changed  bool
		see      []string
		end      string // the text the end of the window judges; "": nothing
	}{
		{name: "testObservationContinuesWhenIdleSnapshotStillLooksLikeDeletionOnly", baseline: baseline, latest: deleted, changed: true, see: []string{deleted}},
		{name: "testObservationContinuesWhenIdleSnapshotStillContainsUnfinishedDeletionGroup", baseline: groupBaseline, latest: groupLatest, changed: true, see: []string{groupLatest}},
		{name: "testObservationSettlesWhenIdleSnapshotContainsCompletedReplacement", baseline: baseline, latest: replaced, changed: true, see: []string{replaced}, end: replaced},
		{name: "testObservationResetsMissingCounterWhenFocusedSnapshotReturns", baseline: "baseline", latest: "baseline", see: []string{"baseline"}},
		{name: "testObservationSettlesAfterDeletionIntermediateWhenReplacementIsCompleted", baseline: baseline, latest: deleted, changed: true, see: []string{replaced, replaced}, end: replaced},
		{name: "testObservationRegistersNewInputChangeAndResetsMissingCounter", baseline: "baseline", latest: "baseline", see: []string{"baseline updated"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := watchState{baseline: tc.baseline, latest: tc.latest, changed: tc.changed, missing: 2}
			var moved bool
			for _, s := range tc.see {
				moved = state.saw(s) || moved
			}
			if state.missing != 0 {
				t.Errorf("missing %d after a read", state.missing)
			}
			if tc.name == "testObservationRegistersNewInputChangeAndResetsMissingCounter" {
				if !moved || !state.changed || state.latest != "baseline updated" {
					t.Errorf("state %+v, moved %v", state, moved)
				}
				return
			}
			end, ok := state.conclude()
			if (tc.end == "") == ok || (ok && end != tc.end) {
				t.Fatalf("conclude: %q %v, want %q", end, ok, tc.end)
			}
		})
	}
}

func TestScopedCurrent(t *testing.T) {
	const inserted = "看一下我们投坑中有没有新的词源了。这个新的词源是也需要接飞的。"
	baseline, ok := scopedBaseline(inserted, "~/x/doit/voxt-service  main !20 ?9\n> "+inserted, -1)
	if !ok || baseline != inserted {
		t.Fatalf("baseline %q %v", baseline, ok)
	}
	// testObservationScopedTextPrefersEchoedCommandLineAfterPromptClears
	got := scopedCurrent(inserted, baseline, "> 看一下我们 token 中有没有新的词元了。这个新的词元也需要计费。\nzsh: command not found: 看一下我们\n\n~/x/doit/voxt-service  main !20 ?9\n>")
	if want := "看一下我们 token 中有没有新的词元了。这个新的词元也需要计费。"; got != want {
		t.Fatalf("got %q", got)
	}
}

func TestScopedBaselineExpectedPosition(t *testing.T) {
	text := "hello ok\nother ok\nlast"
	// The same word on two lines: the position tells which.
	if got, ok := scopedBaseline("ok", text, 15); !ok || got != "other ok" {
		t.Fatalf("got %q %v", got, ok)
	}
	if got, ok := scopedBaseline("ok", text, -1); !ok || got != "hello ok" {
		t.Fatalf("the search finds the first: %q %v", got, ok)
	}
	if got, ok := scopedBaseline("OK", "say Ok", 4); !ok || got != "say Ok" {
		t.Fatalf("ignoring case: %q %v", got, ok)
	}
	if got, ok := scopedBaseline("two  words", "one two\n words end", -1); !ok || got != "one two\n words end" {
		t.Fatalf("whitespace relaxed: %q %v", got, ok)
	}
	if _, ok := scopedBaseline("missing", text, -1); ok {
		t.Fatal("found text that is not there")
	}
}

func TestUTF16Offsets(t *testing.T) {
	if n := utf16Len("a😀b中"); n != 5 {
		t.Fatalf("utf16Len %d", n)
	}
	for _, tc := range []struct {
		unit, rune int
	}{{0, 0}, {1, 1}, {3, 2}, {4, 3}, {5, 4}, {-1, -1}, {6, -1}, {2, -1}} {
		if got := runeAtUTF16("a😀b中", tc.unit); got != tc.rune {
			t.Errorf("unit %d: rune %d, want %d", tc.unit, got, tc.rune)
		}
	}
}
