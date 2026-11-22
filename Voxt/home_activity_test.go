package main

import (
	"context"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// untilActivity runs frames until ok, or fails after a few seconds (as
// waitFor does).
func untilActivity(t *testing.T, tt *ui.Tester, what string, ok func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		frames(tt)
		if ok() {
			return
		}
	}
	t.Fatalf("no %s", what)
}

// TestActivityCountsHistoryThenSessions: the first read counts the
// history into the log and stores it; a later result adds to today; a
// try, a failure and an empty result do not; the history off counts
// nothing.
func TestActivityCountsHistoryThenSessions(t *testing.T) {
	a, tt := homeTester(t)
	ctx := context.Background()
	now := time.Now()
	h := a.lib.store.History()
	for i, e := range []store.Entry{
		{Kind: store.KindTranscribe, CreatedAt: now.AddDate(0, 0, -2), Status: store.StatusDone, Result: "two days ago"},
		{Kind: store.KindTranslate, CreatedAt: now.AddDate(0, 0, -1), Status: store.StatusDone, Result: "yesterday"},
		{Kind: store.KindRewrite, CreatedAt: now.AddDate(0, 0, -1), Status: store.StatusFailed},
	} {
		e.ID = "act-" + string(rune('a'+i))
		if err := h.Put(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	// Home's first visit stored a log of the empty history: forget it, as
	// a store from before the calendar has none.
	docs := a.lib.store.Documents()
	untilActivity(t, tt, "the first log stored", func() bool { _, ok, _ := docs.Get(ctx, activity.Key); return ok })
	if err := docs.Delete(ctx, activity.Key); err != nil {
		t.Fatal(err)
	}
	a.act = homeActivity{}
	a.loadActivity()
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	if got := activeDays(a.act.log); got != 2 {
		t.Fatalf("from the history: %d active days", got)
	}

	rev := func() uint64 { return a.sess.Rev + 1 }
	end := func(id string, s session.Snapshot) {
		a.sessionChanged(session.Snapshot{Rev: rev(), ID: id, Kind: store.KindTranscribe, State: session.Recording, Try: s.Try})
		s.Rev, s.ID, s.Kind = rev(), id, store.KindTranscribe
		a.sessionChanged(s)
	}
	end("done", session.Snapshot{State: session.Done, Result: "Hello."})
	end("try", session.Snapshot{State: session.Done, Result: "Hello.", Try: true})
	end("failed", session.Snapshot{State: session.Failed})
	end("blank", session.Snapshot{State: session.Done, Result: "  "})
	if got := a.act.log.Get(now); got.Count != 1 {
		t.Fatalf("today: %+v", got)
	}
	untilActivity(t, tt, "the log stored with today", func() bool {
		data, ok, err := a.lib.store.Documents().Get(ctx, activity.Key)
		if err != nil || !ok {
			return false
		}
		log, err := activity.Unmarshal(data)
		return err == nil && log.Get(now).Count == 1
	})

	a.prefs.Privacy.KeepHistory = false
	end("off", session.Snapshot{State: session.Done, Result: "Not kept."})
	if got := a.act.log.Get(now).Count; got != 1 {
		t.Fatalf("the history off counted a result: %d", got)
	}
}

// TestActivityPendingBeforeRead: a result that ends before the stored
// log is read is added to it once it lands.
func TestActivityPendingBeforeRead(t *testing.T) {
	a, tt := homeTester(t)
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	var stored activity.Log
	stored.Add(time.Now().AddDate(0, 0, -1), activity.Result{})
	data, _ := stored.Marshal()
	if err := a.lib.store.Documents().Put(context.Background(), activity.Key, data); err != nil {
		t.Fatal(err)
	}
	a.act = homeActivity{}
	rev := func() uint64 { return a.sess.Rev + 1 }
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "p", Kind: store.KindTranscribe, State: session.Recording})
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "p", Kind: store.KindTranscribe, State: session.Done, Result: "Hi."})
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	if got := activeDays(a.act.log); got != 2 {
		t.Fatalf("stored day and the pending result: %d active days", got)
	}
}

func TestHomeUsageNumbersAndHeatmap(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.Local)
	var log activity.Log
	log.Add(now, activity.Result{Chars: 120, Spoken: 120, Audio: 30 * time.Second})
	log.Add(now.AddDate(0, 0, -10), activity.Result{Chars: 30})
	log.Add(now.AddDate(0, 0, -30), activity.Result{Chars: 999})
	state := buildHomeViewState(i18n.NewTranslator("en"), homeViewStateInput{
		Activity: log, ActivityRead: true, Now: now,
	})
	want := []homeActivityStatViewState{
		{Value: "120", Label: "Today"},
		{Value: "150", Label: "30 days"},
		{Value: "240", Label: "Per minute"},
		{Value: "—", Label: "Latency"},
	}
	if state.Activity == nil || len(state.Activity.Stats) != len(want) {
		t.Fatalf("usage stats %+v, want %+v", state.Activity, want)
	}
	for index := range want {
		if state.Activity.Stats[index] != want[index] {
			t.Errorf("stat %d = %+v, want %+v", index, state.Activity.Stats[index], want[index])
		}
	}
	if len(state.Activity.Weeks) != homeActivityWeeks {
		t.Fatalf("the existing heatmap changed: %d weeks", len(state.Activity.Weeks))
	}
}

func TestHomeActivityViewState(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.Local) // a Friday
	var log activity.Log
	for _, d := range []int{-1, 0} {
		log.Add(now.AddDate(0, 0, d), activity.Result{})
	}
	for range 12 {
		log.Add(now.AddDate(0, 0, -3), activity.Result{})
	}
	log.Add(now.AddDate(0, -9, 0), activity.Result{}) // before this page
	tr := i18n.NewTranslator("en")
	state := buildHomeActivityViewState(tr, log, 0, now, homeUsage{})
	if len(state.Weeks) != homeActivityWeeks || !state.CanBack || state.CanForward || state.Levels != activity.Levels {
		t.Fatalf("%d weeks, back %v forward %v", len(state.Weeks), state.CanBack, state.CanForward)
	}
	last := state.Weeks[len(state.Weeks)-1]
	if friday := last[5]; friday.Level != 1 || friday.Future || friday.Date != "Nov 20, 2026" || friday.Count != "1 result" {
		t.Fatalf("today %+v", friday)
	}
	if tuesday := last[2]; tuesday.Level != 4 {
		t.Fatalf("12 results: %+v", tuesday)
	}
	if saturday := last[6]; !saturday.Future {
		t.Fatalf("tomorrow is not future: %+v", saturday)
	}
	if monday := last[1]; monday.Date != "Nov 16, 2026" || monday.Count != "No activity" {
		t.Fatalf("an empty day: %q %q", monday.Date, monday.Count)
	}
	if n := len(state.Months); n < 5 || state.Months[n-1].Label != "Nov" {
		t.Fatalf("months %+v", state.Months)
	}
	zh := buildHomeActivityViewState(i18n.NewTranslator("zh-Hans"), log, 0, now, homeUsage{})
	if day := zh.Weeks[len(zh.Weeks)-1][5]; day.Date != "2026年11月20日" || day.Count != "1 次" || zh.Weekdays[1] != "一" {
		tip := day.Date + " " + day.Count
		t.Fatalf("zh-Hans: %q %q", tip, zh.Weekdays)
	}
	older := buildHomeActivityViewState(tr, log, 1, now, homeUsage{})
	if !older.CanForward || older.Weeks[len(older.Weeks)-1][5].Date != "Jun 19, 2026" {
		t.Fatalf("a page back: forward %v, %q", older.CanForward, older.Weeks[len(older.Weeks)-1][5].Date)
	}
}

// TestHomeActivityShows: Home shows the numbers, a day's square names it
// and its results, and the arrows page.
func TestHomeActivityShows(t *testing.T) {
	a, tt := homeTester(t)
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	a.act.log.Add(time.Now(), activity.Result{})
	a.act.gen++
	frames(tt)
	for _, s := range []string{"Activity", "Less often", "More often"} {
		if !hasExact(tt, s) {
			t.Errorf("no %q: %q", s, firstTexts(tt))
		}
	}
	tr := i18n.CurrentTranslator()
	today := homeActivityDate(tr, time.Now()) + " " + homeActivityCount(tr, 1)
	if _, ok := tt.Find(today); !ok {
		t.Fatalf("no square for today (%q)", today)
	}
	a.act.log.Add(time.Now().AddDate(-1, 0, 0), activity.Result{})
	a.act.gen++
	click(t, tt, "Earlier weeks")
	if a.act.page != 1 {
		t.Fatalf("page %d", a.act.page)
	}
	click(t, tt, "Later weeks")
	if a.act.page != 0 {
		t.Fatalf("back to today: page %d", a.act.page)
	}
}

// TestActivityPendingSeenByTheCount: a result that ended while the first
// count read the history is added once: not again when the count saw its
// entry, and when the count read that day before the entry was written.
func TestActivityPendingSeenByTheCount(t *testing.T) {
	a, tt := homeTester(t)
	ctx := context.Background()
	docs := a.lib.store.Documents()
	untilActivity(t, tt, "the first log stored", func() bool { _, ok, _ := docs.Get(ctx, activity.Key); return ok })
	if err := docs.Delete(ctx, activity.Key); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	if err := a.lib.store.History().Put(ctx, store.Entry{ID: "counted", Kind: store.KindTranscribe, CreatedAt: now, Status: store.StatusDone, Result: "In the history."}); err != nil {
		t.Fatal(err)
	}
	a.act = homeActivity{loading: true} // the count is on its way
	a.recordActivity(session.Snapshot{ID: "counted", State: session.Done, Result: "In the history.", StartedAt: now})
	a.recordActivity(session.Snapshot{ID: "missed", State: session.Done, Result: "Written after the count.", StartedAt: now})
	a.act.loading = false
	a.loadActivity()
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	if got := a.act.log.Get(now).Count; got != 2 {
		t.Fatalf("today counted %d, want the history's entry once and the missed result", got)
	}
}

// activeDays is how many days of the log had a result.
func activeDays(log activity.Log) int {
	n := 0
	for _, day := range log.Days {
		if day.Count > 0 {
			n++
		}
	}
	return n
}
