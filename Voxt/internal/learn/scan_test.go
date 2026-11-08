package learn

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

const scanModelID = "scan-model"

// scanChat is the language model: answer gives the text for each request
// (1-based), and it keeps the requests.
type scanChat struct {
	answer   func(request int, records []string) (string, error)
	requests []llm.Request
}

func (f *scanChat) Chat(_ context.Context, req llm.Request) (llm.Response, error) {
	f.requests = append(f.requests, req)
	text, err := f.answer(len(f.requests), recordsOf(req))
	return llm.Response{Text: text}, err
}

// recordsOf is the records of a request's user message.
func recordsOf(req llm.Request) []string {
	var out []string
	for _, part := range strings.Split(req.Messages[len(req.Messages)-1].Content, "<record>\n")[1:] {
		out = append(out, strings.TrimSuffix(strings.TrimSpace(part), "\n</record>"))
	}
	return out
}

func (f *scanChat) open(context.Context) (local.Chat, local.Model, error) {
	return f, local.Model{ID: scanModelID, Name: "the scan model"}, nil
}

// answerFirst proposes the first word of each batch's first record.
func answerFirst(_ int, records []string) (string, error) {
	return fmt.Sprintf(`{"terms": [%q]}`, strings.Fields(records[0])[0]), nil
}

func transcription(minute int, text string) store.Entry {
	return store.Entry{ID: fmt.Sprintf("01J%04d", minute), Kind: store.KindTranscribe, Status: store.StatusDone,
		CreatedAt: bookStart.Add(time.Duration(minute) * time.Minute), Result: text}
}

// numbered is count transcriptions of distinct first words ("Term03 said so").
func numbered(count int) []store.Entry {
	var out []store.Entry
	for i := range count {
		out = append(out, transcription(i, fmt.Sprintf("Term%02d said so", i)))
	}
	return out
}

func newScanner(t *testing.T, entries ...store.Entry) (*Scanner, *Book, *sqlitestore.Store) {
	t.Helper()
	b, st, _ := newBook(t)
	s := &Scanner{History: st.History(), Book: b}
	addEntries(t, st, entries...)
	return s, b, st
}

func addEntries(t *testing.T, st *sqlitestore.Store, entries ...store.Entry) {
	t.Helper()
	for _, e := range entries {
		if err := st.History().Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
}

func scanWith(s *Scanner, chat *scanChat) (int, error) {
	return s.Scan(context.Background(), scanModelID, chat.open, Languages{Main: "en"}, func(int, int, int) {})
}

func pendingCount(t *testing.T, s *Scanner) int {
	t.Helper()
	n, err := s.Pending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func TestScanBatchesOldestFirstAndMovesTheCheckpoint(t *testing.T) {
	s, b, _ := newScanner(t, numbered(30)...)
	if n := pendingCount(t, s); n != 30 {
		t.Fatalf("pending %d", n)
	}
	chat := &scanChat{answer: answerFirst}
	type step struct{ done, total, found, left int }
	var steps []step
	found, err := s.Scan(context.Background(), scanModelID, chat.open, Languages{Main: "en"}, func(done, total, found int) {
		steps = append(steps, step{done, total, found, pendingCount(t, s)})
	})
	if err != nil || found != 3 {
		t.Fatalf("found %d, %v", found, err)
	}
	want := []step{{12, 30, 1, 18}, {24, 30, 2, 6}, {30, 30, 3, 0}}
	if !slices.Equal(steps, want) {
		t.Errorf("steps %+v, want %+v", steps, want)
	}
	if len(chat.requests) != 3 {
		t.Fatalf("%d requests", len(chat.requests))
	}
	for i, size := range []int{12, 12, 6} {
		records := recordsOf(chat.requests[i])
		if len(records) != size || !strings.HasPrefix(records[0], fmt.Sprintf("Term%02d", i*12)) {
			t.Errorf("request %d: %q", i, records)
		}
	}
	if got := pendingTerms(t, b); !slices.Equal(got, []string{"Term24", "Term12", "Term00"}) {
		t.Errorf("suggestions %q", got)
	}
	for _, sug := range pending(t, b) {
		if sug.Source != FromHistory || len(sug.Heard) != 0 {
			t.Errorf("suggestion %+v", sug)
		}
	}
	chat.requests = nil
	if found, err := scanWith(s, chat); found != 0 || err != nil || len(chat.requests) != 0 {
		t.Errorf("a second scan: %d, %v, %d requests", found, err, len(chat.requests))
	}
}

func TestScanStopsAtAFailedBatchAndResumesThere(t *testing.T) {
	s, b, _ := newScanner(t, numbered(30)...)
	failing := errors.New("the service is down")
	chat := &scanChat{answer: func(request int, records []string) (string, error) {
		if request == 2 {
			return "", failing
		}
		return answerFirst(request, records)
	}}
	found, err := scanWith(s, chat)
	if !errors.Is(err, failing) || found != 1 || len(chat.requests) != 2 {
		t.Fatalf("found %d, %v, %d requests", found, err, len(chat.requests))
	}
	if n := pendingCount(t, s); n != 18 {
		t.Errorf("pending %d: the checkpoint is not at the end of batch 1", n)
	}
	if got := pendingTerms(t, b); !slices.Equal(got, []string{"Term00"}) {
		t.Errorf("suggestions %q", got)
	}
	again := &scanChat{answer: answerFirst}
	if _, err := scanWith(s, again); err != nil || len(again.requests) != 2 {
		t.Fatalf("%v, %d requests", err, len(again.requests))
	}
	if records := recordsOf(again.requests[0]); !strings.HasPrefix(records[0], "Term12") {
		t.Errorf("the second scan starts at %q", records[0])
	}
}

func TestScanCancelledKeepsWhatItFound(t *testing.T) {
	s, b, _ := newScanner(t, numbered(30)...)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	chat := &scanChat{answer: answerFirst}
	found, err := s.Scan(ctx, scanModelID, chat.open, Languages{Main: "en"}, func(done, _, _ int) {
		if done == 12 {
			cancel() // between the first request and the second
		}
	})
	if !errors.Is(err, context.Canceled) || found != 1 || len(chat.requests) != 1 {
		t.Fatalf("found %d, %v, %d requests", found, err, len(chat.requests))
	}
	if got := pendingTerms(t, b); !slices.Equal(got, []string{"Term00"}) || pendingCount(t, s) != 18 {
		t.Errorf("suggestions %q, pending %d", got, pendingCount(t, s))
	}
}

func TestScanLegacyEntriesOnceAndNewOnesAfter(t *testing.T) {
	legacy := transcription(0, "Legacyname was here")
	legacy.ID = "legacy-1234"
	recent := transcription(5, "Recentname was here")
	s, _, st := newScanner(t, recent, legacy)
	chat := &scanChat{answer: answerFirst}
	if _, err := scanWith(s, chat); err != nil {
		t.Fatal(err)
	}
	if records := recordsOf(chat.requests[0]); len(records) != 2 || !strings.HasPrefix(records[0], "Legacyname") {
		t.Fatalf("records %q", records)
	}
	chat.requests = nil
	if found, err := scanWith(s, chat); found != 0 || err != nil || len(chat.requests) != 0 {
		t.Fatalf("legacy read again: %d, %v, %d requests", found, err, len(chat.requests))
	}
	addEntries(t, st, transcription(10, "Newname was here"))
	if _, err := scanWith(s, chat); err != nil {
		t.Fatal(err)
	}
	if len(chat.requests) != 1 {
		t.Fatalf("%d requests after a new entry", len(chat.requests))
	}
	if records := recordsOf(chat.requests[0]); len(records) != 1 || !strings.HasPrefix(records[0], "Newname") {
		t.Errorf("records %q", records)
	}
	older := transcription(1, "Importedlater was here")
	older.ID = "legacy-5678"
	addEntries(t, st, older)
	if n := pendingCount(t, s); n != 0 {
		t.Errorf("an entry older than the checkpoint, imported later: pending %d", n)
	}
}

func TestScanSameInstantEntriesEachReadOnce(t *testing.T) {
	var same []store.Entry
	for i := range 13 {
		e := transcription(0, fmt.Sprintf("Word%02d said so", i))
		e.ID = fmt.Sprintf("01J%04d", i)
		same = append(same, e)
	}
	s, _, _ := newScanner(t, same...)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := &scanChat{answer: answerFirst}
	if _, err := s.Scan(ctx, scanModelID, first.open, Languages{Main: "en"}, func(int, int, int) { cancel() }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if n := pendingCount(t, s); n != 1 {
		t.Fatalf("pending %d after the first batch", n)
	}
	second := &scanChat{answer: answerFirst}
	if _, err := scanWith(s, second); err != nil || len(second.requests) != 1 {
		t.Fatalf("%v, %d requests", err, len(second.requests))
	}
	if records := recordsOf(second.requests[0]); len(records) != 1 || !strings.HasPrefix(records[0], "Word12") {
		t.Errorf("the next scan: %q", records)
	}
}

func TestScanReadsDoneTranscriptionsOnly(t *testing.T) {
	entryOf := func(minute int, edit func(*store.Entry)) store.Entry {
		e := transcription(minute, fmt.Sprintf("Skipped%d said so", minute))
		edit(&e)
		return e
	}
	s, _, _ := newScanner(t,
		transcription(0, "Kept said so"),
		entryOf(1, func(e *store.Entry) { e.Kind = store.KindTranslate }),
		entryOf(2, func(e *store.Entry) { e.Kind = store.KindRewrite }),
		entryOf(3, func(e *store.Entry) { e.Status = store.StatusFailed }),
		entryOf(4, func(e *store.Entry) { e.Status = store.StatusCancelled }),
		entryOf(5, func(e *store.Entry) { e.Result, e.RawText = "", "" }),
		entryOf(6, func(e *store.Entry) { e.Result, e.RawText = "", "Rawonly said so" }),
		entryOf(7, func(e *store.Entry) { e.Result, e.RawText = "Result said so", "Rawignored said so" }),
	)
	if n := pendingCount(t, s); n != 3 {
		t.Errorf("pending %d", n)
	}
	chat := &scanChat{answer: answerFirst}
	if _, err := scanWith(s, chat); err != nil {
		t.Fatal(err)
	}
	if got := recordsOf(chat.requests[0]); !slices.Equal(got, []string{"Kept said so", "Rawonly said so", "Result said so"}) {
		t.Errorf("records %q", got)
	}
}

func TestScanCutsARecordAt320Characters(t *testing.T) {
	long := strings.Repeat("词", 400)
	s, _, _ := newScanner(t, transcription(0, long))
	chat := &scanChat{answer: func(int, []string) (string, error) { return `{"terms":[]}`, nil }}
	if _, err := scanWith(s, chat); err != nil {
		t.Fatal(err)
	}
	if got := recordsOf(chat.requests[0]); len(got) != 1 || []rune(got[0])[319] != '词' || len([]rune(got[0])) != 320 {
		t.Errorf("record of %d characters", len([]rune(got[0])))
	}
}

func TestScanFiltersTheAnswer(t *testing.T) {
	s, b, st := newScanner(t, transcription(0, "Ghostty and OpenAI and email, also Knownterm in Kubernetes"))
	putTerm(t, st, store.Term{ID: "known", Term: "Knownterm", Enabled: true})
	chat := &scanChat{answer: func(int, []string) (string, error) {
		return "```json\n" + `{"terms":["Ghostty","GHOSTTY","Invented","email","Knownterm","OpenAI"]}` + "\n```", nil
	}}
	found, err := scanWith(s, chat)
	if err != nil || found != 2 {
		t.Fatalf("found %d, %v", found, err)
	}
	if got := pendingTerms(t, b); !slices.Equal(got, []string{"OpenAI", "Ghostty"}) {
		t.Errorf("suggestions %q", got)
	}
}

func TestScanTakesAtMostTwelveTermsARequest(t *testing.T) {
	var terms []string
	for i := range 16 {
		terms = append(terms, fmt.Sprintf(`"Name%02d"`, i))
	}
	s, b, _ := newScanner(t, transcription(0, strings.ReplaceAll(strings.Join(terms, " "), `"`, "")))
	chat := &scanChat{answer: func(int, []string) (string, error) {
		return `{"terms":[` + strings.Join(terms, ",") + `]}`, nil
	}}
	if found, err := scanWith(s, chat); err != nil || found != 12 {
		t.Fatalf("found %d, %v", found, err)
	}
	if got := pendingTerms(t, b); len(got) != 12 || got[0] != "Name11" {
		t.Errorf("suggestions %q", got)
	}
}

func TestScanStopsOnAnAnswerThatIsNotATermList(t *testing.T) {
	s, _, _ := newScanner(t, numbered(1)...)
	chat := &scanChat{answer: func(int, []string) (string, error) { return "I found Ghostty.", nil }}
	var codes *llm.Error
	if _, err := scanWith(s, chat); !errors.As(err, &codes) || codes.Code != llm.CodeInvalidJSON {
		t.Fatalf("err %v", err)
	}
	if pendingCount(t, s) != 1 {
		t.Errorf("the checkpoint moved")
	}
}

func TestScanSendsRecordsNotTheDictionaryInTheUsersLanguage(t *testing.T) {
	s, _, st := newScanner(t, transcription(0, "Ghostty said so"))
	putTerm(t, st, store.Term{ID: "x", Term: "SecretDictionaryWord", Aliases: []string{"SecretAlias"}, Note: "SecretNote", Enabled: true})
	chat := &scanChat{answer: answerFirst}
	if _, err := s.Scan(context.Background(), scanModelID, chat.open, Languages{Main: "zh", Others: []string{"en"}}, func(int, int, int) {}); err != nil {
		t.Fatal(err)
	}
	req := chat.requests[0]
	if req.Format != llm.FormatJSON || req.Schema == nil {
		t.Errorf("not asked for JSON: %+v", req)
	}
	var whole strings.Builder
	for _, m := range req.Messages {
		whole.WriteString(m.Content)
	}
	for _, secret := range []string{"SecretDictionaryWord", "SecretAlias", "SecretNote"} {
		if strings.Contains(whole.String(), secret) {
			t.Errorf("the request has %q: %s", secret, whole.String())
		}
	}
	system := req.Messages[0].Content
	if !strings.Contains(system, "用户主要口语语言：中文") || !strings.Contains(system, "其他常用语言：英语") {
		t.Errorf("not in the user's languages: %s", system)
	}
}

func TestScanWithAnotherModelSendsNothing(t *testing.T) {
	s, b, _ := newScanner(t, numbered(3)...)
	chat := &scanChat{answer: answerFirst}
	open := func(context.Context) (local.Chat, local.Model, error) {
		return chat, local.Model{ID: "the first model"}, nil
	}
	found, err := s.Scan(context.Background(), scanModelID, open, Languages{Main: "en"}, func(int, int, int) {})
	if err == nil || found != 0 {
		t.Fatalf("found %d, %v", found, err)
	}
	if len(chat.requests) != 0 || len(pendingTerms(t, b)) != 0 || pendingCount(t, s) != 3 {
		t.Errorf("%d requests", len(chat.requests))
	}
}

func TestScanCheckpointSurvivesReopeningTheBook(t *testing.T) {
	s, _, st := newScanner(t, numbered(3)...)
	if _, err := scanWith(s, &scanChat{answer: answerFirst}); err != nil {
		t.Fatal(err)
	}
	reopened := &Scanner{History: st.History(), Book: NewBook(st.Documents(), st.Dictionary())}
	if n := pendingCount(t, reopened); n != 0 {
		t.Errorf("pending %d", n)
	}
}
