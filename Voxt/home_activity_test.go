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
	if got := a.act.log.Stats(now); got != (activity.Stats{Active: 2, Current: 2, Longest: 2}) {
		t.Fatalf("from the history: %+v", got)
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
	if got := a.act.log.Get(now); got.Count != 1 || got.Chars != countChars("Hello.") {
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
	stored.Add(time.Now().AddDate(0, 0, -1), 5)
	data, _ := stored.Marshal()
	if err := a.lib.store.Documents().Put(context.Background(), activity.Key, data); err != nil {
		t.Fatal(err)
	}
	a.act = homeActivity{}
	rev := func() uint64 { return a.sess.Rev + 1 }
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "p", Kind: store.KindTranscribe, State: session.Recording})
	a.sessionChanged(session.Snapshot{Rev: rev(), ID: "p", Kind: store.KindTranscribe, State: session.Done, Result: "Hi."})
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	if got := a.act.log.Stats(time.Now()); got.Active != 2 || got.Current != 2 {
		t.Fatalf("stored day and the pending result: %+v", got)
	}
}

func TestHomeActivityViewState(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.Local) // a Friday
	var log activity.Log
	for _, d := range []int{-1, 0} {
		log.Add(now.AddDate(0, 0, d), 3)
	}
	for range 12 {
		log.Add(now.AddDate(0, 0, -3), 1)
	}
	log.Add(now.AddDate(0, -9, 0), 1) // before this page
	tr := i18n.NewTranslator("en")
	state := buildHomeActivityViewState(tr, log, 0, now)
	if got := []string{state.Stats[0].Value, state.Stats[1].Value, state.Stats[2].Value}; got[0] != "4" || got[1] != "2" || got[2] != "2" {
		t.Fatalf("stats %q", got)
	}
	if state.Stats[0].Unit != "days" || state.Stats[1].Label != "Current streak" {
		t.Fatalf("stat words %+v", state.Stats)
	}
	if len(state.Weeks) != homeActivityWeeks || !state.CanBack || state.CanForward || state.Levels != activity.Levels {
		t.Fatalf("%d weeks, back %v forward %v", len(state.Weeks), state.CanBack, state.CanForward)
	}
	last := state.Weeks[len(state.Weeks)-1]
	if friday := last[5]; friday.Level != 1 || friday.Future || friday.Tip != "Nov 20, 2026: 1 result" {
		t.Fatalf("today %+v", friday)
	}
	if tuesday := last[2]; tuesday.Level != 4 {
		t.Fatalf("12 results: %+v", tuesday)
	}
	if saturday := last[6]; !saturday.Future {
		t.Fatalf("tomorrow is not future: %+v", saturday)
	}
	if monday := last[1]; monday.Tip != "Nov 16, 2026: no results" {
		t.Fatalf("an empty day: %q", monday.Tip)
	}
	if n := len(state.Months); n < 5 || state.Months[n-1].Label != "Nov" {
		t.Fatalf("months %+v", state.Months)
	}
	zh := buildHomeActivityViewState(i18n.NewTranslator("zh-Hans"), log, 0, now)
	if tip := zh.Weeks[len(zh.Weeks)-1][5].Tip; tip != "2026年11月20日：1 次" || zh.Stats[0].Unit != "天" || zh.Weekdays[1] != "一" {
		t.Fatalf("zh-Hans: %q %q %q", tip, zh.Stats[0].Unit, zh.Weekdays)
	}
	older := buildHomeActivityViewState(tr, log, 1, now)
	if !older.CanForward || older.Weeks[len(older.Weeks)-1][5].Tip != "May 22, 2026: no results" {
		t.Fatalf("a page back: forward %v, %q", older.CanForward, older.Weeks[len(older.Weeks)-1][5].Tip)
	}
}

// TestHomeActivityShows: Home shows the numbers, a day's square names it
// and its results, and the arrows page.
func TestHomeActivityShows(t *testing.T) {
	a, tt := homeTester(t)
	untilActivity(t, tt, "log read", func() bool { return a.act.loaded })
	a.act.log.Add(time.Now(), 4)
	a.act.gen++
	frames(tt)
	for _, s := range []string{"Activity", "Active days", "Current streak", "Longest streak", "Less often", "More often"} {
		if !hasExact(tt, s) {
			t.Errorf("no %q: %q", s, firstTexts(tt))
		}
	}
	today := homeActivityDayTip(i18n.CurrentTranslator(), time.Now(), 1)
	if _, ok := tt.Find(today); !ok {
		t.Fatalf("no square for today (%q)", today)
	}
	a.act.log.Add(time.Now().AddDate(-1, 0, 0), 1)
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
