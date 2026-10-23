package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// Sample history entries (seedHistory), newest first.
const (
	histInserted = "01HIST0000000000000000INS1" // done, inserted, with LLM and a recording
	histCard     = "01HIST0000000000000000CRD1" // done, left in the answer card (Chinese)
	histFailed   = "01HIST0000000000000000FAI1" // failed: the speech model did not answer
	histCopied   = "01HIST0000000000000000CPY1" // done, copied from the card
)

// seedHistory writes sample entries to the app's history: what the
// History page's tests and snapshots show.
func seedHistory(t *testing.T, a *app) {
	t.Helper()
	now := time.Now()
	for _, e := range []store.Entry{
		{
			ID: histInserted, Kind: store.KindTranscribe, CreatedAt: now.Add(-5 * time.Minute), Status: store.StatusDone,
			AppBundle: "com.apple.Notes", AppName: "Notes",
			RawText:     "um so let's move the design review to thursday afternoon and I'll send the new agenda tonight",
			Result:      "Let's move the design review to Thursday afternoon.\nI'll send the new agenda tonight.",
			Delivery:    "insert",
			ASRProvider: "asr.accurate", ASRModel: "asr.accurate", LLMProvider: "llm.balanced", LLMModel: "llm.balanced",
			AudioMS: 7400, ASRMS: 620, LLMMS: 1180, TotalMS: 9350, Audio: histInserted + ".wav",
			DSP: "highpass+level", GainDB: 7.5,
		},
		{
			ID: histCard, Kind: store.KindTranscribe, CreatedAt: now.Add(-42 * time.Minute), Status: store.StatusDone,
			AppBundle: "com.tinyspeck.slackmacgap", AppName: "Slack",
			RawText:     "明天上午十点的站会改到线上 链接我稍后发群里",
			Result:      "明天上午十点的站会改到线上，链接我稍后发到群里。",
			Delivery:    "card",
			ASRProvider: "asr.fast", ASRModel: "asr.fast",
			AudioMS: 4100, ASRMS: 380, TotalMS: 4620,
		},
		{
			ID: histFailed, Kind: store.KindTranscribe, CreatedAt: now.Add(-3 * time.Hour), Status: store.StatusFailed,
			AppBundle: "com.apple.Safari", AppName: "Safari", URLHost: "github.com",
			Error:       "asr.accurate: the speech server stopped answering (15s)",
			ASRProvider: "asr.accurate", ASRModel: "asr.accurate",
			AudioMS: 12800, ASRMS: 30000, TotalMS: 42900,
		},
		{
			ID: histCopied, Kind: store.KindTranscribe, CreatedAt: now.Add(-50 * time.Hour), Status: store.StatusDone,
			AppBundle: "com.apple.mail", AppName: "Mail",
			RawText:     "hi Anna thanks for the quick turnaround on the contract the changes look good to me",
			Result:      "Hi Anna, thanks for the quick turnaround on the contract. The changes look good to me.",
			Delivery:    "copy",
			ASRProvider: "asr.fast", ASRModel: "asr.fast", LLMProvider: "llm.balanced", LLMModel: "llm.balanced",
			AudioMS: 5200, ASRMS: 1450, LLMMS: 940, TotalMS: 7720,
		},
	} {
		if err := a.lib.store.History().Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
}

// Translate and Rewrite entries (seedWriteHistory).
const (
	histTranslated = "01HIST0000000000000000TRL1" // speech into English, inserted
	histTransSel   = "01HIST0000000000000000TRL2" // a selection on a web page into Chinese, in the card
	histRewritten  = "01HIST0000000000000000RWR1" // the selection rewritten
	histWritten    = "01HIST0000000000000000RWR2" // written from scratch, long: in the card
)

// seedWriteHistory writes Translate and Rewrite entries.
func seedWriteHistory(t *testing.T, a *app) {
	t.Helper()
	now := time.Now()
	for _, e := range []store.Entry{
		{
			ID: histTranslated, Kind: store.KindTranslate, CreatedAt: now.Add(-8 * time.Minute), Status: store.StatusDone,
			AppBundle: "com.tinyspeck.slackmacgap", AppName: "Slack",
			RawText: "嗯 帮我跟他们说 周四下午的评审改到周五上午", Result: "Let them know the Thursday afternoon review moves to Friday morning.",
			TargetLang: "en", Delivery: "insert",
			ASRProvider: "asr.accurate", ASRModel: "asr.accurate", LLMProvider: "llm.balanced", LLMModel: "llm.balanced",
			AudioMS: 4300, ASRMS: 520, LLMMS: 1340, TotalMS: 6400,
		},
		{
			ID: histTransSel, Kind: store.KindTranslate, CreatedAt: now.Add(-2 * time.Hour), Status: store.StatusDone,
			AppBundle: "com.google.Chrome", AppName: "Google Chrome", URLHost: "go.dev",
			SourceText: "A goroutine is a lightweight thread managed by the Go runtime.",
			Result:     "goroutine 是由 Go 运行时管理的轻量级线程。", TargetLang: "zh-Hans", Delivery: "card",
			LLMProvider: "llm.balanced", LLMModel: "llm.balanced", LLMMS: 980, TotalMS: 1100,
		},
		{
			ID: histRewritten, Kind: store.KindRewrite, CreatedAt: now.Add(-20 * time.Minute), Status: store.StatusDone,
			AppBundle: "com.apple.mail", AppName: "Mail",
			PromptText: "make it more formal and shorter",
			SourceText: "hey, the release slips to thursday because the tests are still running, sorry!",
			Result:     "Apologies: the release moves to Thursday while the tests finish.", Delivery: "insert",
			ASRProvider: "asr.accurate", ASRModel: "asr.accurate", LLMProvider: "llm.balanced", LLMModel: "llm.balanced",
			AudioMS: 2600, ASRMS: 410, LLMMS: 1650, TotalMS: 4900,
		},
		{
			ID: histWritten, Kind: store.KindRewrite, CreatedAt: now.Add(-26 * time.Hour), Status: store.StatusDone,
			AppBundle: "com.apple.Notes", AppName: "Notes",
			PromptText:  "写一段两百字的自我介绍，用于技术分享会的开场",
			Result:      "大家好，我是关伟，一名专注于桌面应用和语音交互的工程师。过去几年，我主要在做跨平台的原生界面框架……",
			Delivery:    "card",
			ASRProvider: "asr.fast", ASRModel: "asr.fast", LLMProvider: "llm.balanced", LLMModel: "llm.balanced",
			AudioMS: 5100, ASRMS: 300, LLMMS: 6200, TotalMS: 11900,
		},
	} {
		if err := a.lib.store.History().Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
}

// snapshotHistoryDetail renders the History page with an entry's details
// (TestSnapshots), on each tab.
func snapshotHistoryDetail(a *app, save func(string)) {
	a.navigate("/history")
	a.hist.sel = histInserted
	save("history-detail")
	for _, tab := range []struct {
		i    int
		name string
		sel  string
	}{{1, "history-translate", histTransSel}, {2, "history-rewrite", histRewritten}} {
		a.hist.tab, a.hist.sel = tab.i, ""
		a.loadHistory()
		save(tab.name)
		a.hist.sel = tab.sel
		save(tab.name + "-detail")
	}
	a.hist.tab, a.hist.sel = 0, ""
	a.loadHistory()
	snapshotHistoryRetry(a, save) // history_retry_test.go
}

func TestHistoryTranslateAndRewrite(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	seedWriteHistory(t, a)
	a.navigate("/history")
	a.hist.tab = 1
	a.loadHistory()
	frames(tt)
	if len(a.hist.list) != 2 || a.hist.list[0].ID != histTranslated {
		t.Fatalf("translate tab %v", historyIDs(a))
	}
	if !has(tt, "Let them know the Thursday afternoon review moves to Friday morning.") || !hasPrefix(tt, "8 minutes ago · Slack · → English") {
		t.Errorf("rows %q", firstTexts(tt))
	}
	a.hist.sel = histTransSel
	frames(tt)
	for _, s := range []string{"Selected text", "A goroutine is a lightweight thread managed by the Go runtime.", "Target language", "Simplified Chinese", "Shown in the answer card"} {
		if !has(tt, s) {
			t.Errorf("translate detail: no %q in %q", s, firstTexts(tt))
		}
	}
	if has(tt, "Recognised text") {
		t.Error("a selection shows a recognised text")
	}
	a.hist.tab, a.hist.sel = 2, histRewritten
	a.loadHistory()
	frames(tt)
	for _, s := range []string{"Instruction", "make it more formal and shorter", "Selected text", "Apologies: the release moves to Thursday while the tests finish."} {
		if !has(tt, s) {
			t.Errorf("rewrite detail: no %q in %q", s, firstTexts(tt))
		}
	}
	if has(tt, "Target language") {
		t.Error("a rewrite has a target language")
	}
	// The search covers the instruction and the selection.
	a.hist.query = "formal"
	a.loadHistory()
	frames(tt)
	if len(a.hist.list) != 1 || a.hist.list[0].ID != histRewritten {
		t.Errorf("search %v", historyIDs(a))
	}
}

func TestHistoryList(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	a.navigate("/history")
	frames(tt)
	if len(a.hist.list) != 4 || a.hist.list[0].ID != histInserted || a.hist.list[3].ID != histCopied {
		t.Fatalf("list = %v", historyIDs(a))
	}
	for _, s := range []string{
		"Let's move the design review to Thursday afternoon.", // the result's first line
		"明天上午十点的站会改到线上，链接我稍后发到群里。",
		"asr.accurate: the speech server stopped answering (15s)", // a failure's error
		"5 minutes ago · Notes · 0:07",
		"Failed", "Not inserted", "Copied", "4 entries",
	} {
		if !has(tt, s) {
			t.Errorf("the list does not show %q (texts: %q)", s, firstTexts(tt))
		}
	}
}

func TestHistoryTabs(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	a.navigate("/history")
	clickSegment(t, tt, "Feature", 1, 3)
	if a.hist.kind() != store.KindTranslate || len(a.hist.list) != 0 {
		t.Fatalf("translate tab: kind %s, %v", a.hist.kind(), historyIDs(a))
	}
	if !has(tt, "No translations yet") {
		t.Errorf("no empty state on the Translate tab: %q", firstTexts(tt))
	}
	clickSegment(t, tt, "Feature", 2, 3)
	if !has(tt, "No rewrites yet") {
		t.Errorf("no empty state on the Rewrite tab: %q", firstTexts(tt))
	}
	clickSegment(t, tt, "Feature", 0, 3)
	if len(a.hist.list) != 4 {
		t.Fatalf("transcribe tab: %v", historyIDs(a))
	}
}

func TestHistoryEmpty(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/history")
	if !has(tt, "No transcriptions yet") {
		t.Fatalf("no empty state: %q", firstTexts(tt))
	}
	click(t, tt, "Set up Transcribe")
	if a.currentPage() != "transcribe" {
		t.Errorf("page = %s", a.currentPage())
	}
}

func TestHistoryOff(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepHistory = false })
	a.navigate("/history")
	if !has(tt, "History is off: new results are not kept.") {
		t.Fatalf("no notice: %q", firstTexts(tt))
	}
}

func TestHistorySearch(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	a.navigate("/history")
	frames(tt)
	if err := tt.Click("Search the history"); err != nil {
		t.Fatal(err)
	}
	tt.Type("agenda") // full text (3 characters and more)
	frames(tt)
	if ids := historyIDs(a); len(ids) != 1 || ids[0] != histInserted {
		t.Fatalf("search agenda: %v", ids)
	}
	if has(tt, "Hi Anna, thanks for the quick turnaround on the contract. The changes look good to me.") {
		t.Error("an entry that does not match shows")
	}
	a.hist.query = "站会" // under 3 characters: a substring
	a.loadHistory()
	if ids := historyIDs(a); len(ids) != 1 || ids[0] != histCard {
		t.Fatalf("search 站会: %v", ids)
	}
	a.hist.query = "nothing like this"
	a.loadHistory()
	if !has(tt, "Nothing matches “nothing like this”") {
		t.Fatalf("no empty search state: %q", firstTexts(tt))
	}
	click(t, tt, "Clear the search")
	if a.hist.query != "" || len(a.hist.list) != 4 {
		t.Errorf("after Clear the search: %q, %v", a.hist.query, historyIDs(a))
	}
}

func TestHistoryDetail(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	var revealed []string
	was := revealInFinder
	revealInFinder = func(path string) { revealed = append(revealed, path) }
	t.Cleanup(func() { revealInFinder = was })
	a.navigate("/history")
	click(t, tt, "Let's move the design review to Thursday afternoon.")
	if a.hist.sel != histInserted {
		t.Fatalf("selected %q", a.hist.sel)
	}
	for _, s := range []string{
		"Entry details", "Result", "Recognised text", "Inserted", "Notes",
		"Accurate speech recognition", "Balanced language model", "7.4 s", "620 ms", "1.2 s", "9.3 s",
		"Sound optimization", "Hum filtered, volume +7.5 dB",
	} {
		if !has(tt, s) {
			t.Errorf("the details do not show %q (texts: %q)", s, tt.Texts())
		}
	}
	click(t, tt, "Copy the result")
	if got := tt.Clipboard(); got != "Let's move the design review to Thursday afternoon.\nI'll send the new agenda tonight." {
		t.Errorf("clipboard = %q", got)
	}
	click(t, tt, "Copy the recognised text")
	if got := tt.Clipboard(); got != "um so let's move the design review to thursday afternoon and I'll send the new agenda tonight" {
		t.Errorf("clipboard = %q", got)
	}
	click(t, tt, "Show the recording in Finder")
	if want := filepath.Join(a.lib.store.History().AudioDir(), histInserted+".wav"); len(revealed) != 1 || revealed[0] != want {
		t.Errorf("revealed %v, want %s", revealed, want)
	}

	// A failed entry: why, no language model, no recording.
	click(t, tt, "asr.accurate: the speech server stopped answering (15s)")
	if a.hist.sel != histFailed {
		t.Fatalf("selected %q", a.hist.sel)
	}
	if !has(tt, "Failed: asr.accurate: the speech server stopped answering (15s)") {
		t.Errorf("no failure notice: %q", tt.Texts())
	}
	if tt.HasText("Show the recording in Finder") {
		t.Error("Show in Finder without a recording")
	}
	click(t, tt, "Close")
	if a.hist.sel != "" {
		t.Errorf("still selected: %q", a.hist.sel)
	}
}

func TestHistoryDelete(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	a.navigate("/history")
	click(t, tt, "明天上午十点的站会改到线上，链接我稍后发到群里。")
	click(t, tt, "Delete the entry…")
	if !a.hist.delAsk || !has(tt, "Delete this entry?") {
		t.Fatalf("no confirmation: %q", firstTexts(tt))
	}
	click(t, tt, "Cancel")
	if _, err := a.lib.store.History().Get(context.Background(), histCard); err != nil {
		t.Fatalf("Cancel deleted it: %v", err)
	}
	click(t, tt, "Delete the entry…")
	if err := tt.Click("Delete"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if _, err := a.lib.store.History().Get(context.Background(), histCard); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("still there: %v", err)
	}
	if a.hist.sel != "" || len(a.hist.list) != 3 || has(tt, "明天上午十点的站会改到线上，链接我稍后发到群里。") {
		t.Errorf("after Delete: selected %q, %v", a.hist.sel, historyIDs(a))
	}
}

func TestHistoryChanged(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/history")
	frames(tt)
	put := func(id, result string) {
		e := store.Entry{ID: id, Kind: store.KindTranscribe, CreatedAt: time.Now(), Result: result, Delivery: "insert"}
		if err := a.lib.store.History().Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
	// Shown: it reads again at once.
	put("01HISTNEW1", "A new result")
	a.historyChanged()
	if !has(tt, "A new result") {
		t.Fatalf("not refreshed while shown: %q", firstTexts(tt))
	}
	// Not shown: it reads again when it shows.
	a.navigate("/logs")
	frames(tt)
	put("01HISTNEW2", "Another result")
	a.historyChanged()
	if !a.hist.stale {
		t.Error("not marked stale")
	}
	a.navigate("/history")
	if !has(tt, "Another result") || a.hist.stale {
		t.Fatalf("not refreshed when shown: %q", firstTexts(tt))
	}
}

func historyIDs(a *app) []string {
	var ids []string
	for _, e := range a.hist.list {
		ids = append(ids, e.ID)
	}
	return ids
}

// clickSegment clicks segment i of n of the segmented control labelled
// label (its segments' texts may show elsewhere, as in the sidebar).
func clickSegment(t *testing.T, tt *ui.Tester, label string, i, n int) {
	t.Helper()
	frames(tt)
	r, ok := tt.Find(label)
	if !ok {
		t.Fatalf("no %q", label)
	}
	tt.ClickAt(r.X+r.W*(float32(i)+0.5)/float32(n), r.Y+r.H/2)
	frames(tt)
}
