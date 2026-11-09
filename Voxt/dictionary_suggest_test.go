package main

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
)

// suggestTester starts the app over a library that already holds the
// suggestions, in Chinese, the dictionary page not opened.
func suggestTester(t *testing.T, suggestions ...learn.Suggestion) (*app, *ui.Tester) {
	t.Helper()
	dir := testDir(t)
	lib, err := openLibrary(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := learn.NewBook(lib.store.Documents(), lib.store.Dictionary()).Propose(context.Background(), suggestions); err != nil {
		t.Fatal(err)
	}
	lib.close()
	a, _, tt := newTestAppIn(t, dir)
	i18n.SetLocale("zh-Hans")
	tt.SetSize(windowWidth, 1200)
	frames(tt)
	return a, tt
}

func clickControl(t *testing.T, tt *ui.Tester, label string) {
	t.Helper()
	frames(tt)
	if err := tt.ClickControl(label); err != nil {
		t.Fatalf("%v (labels %q)", err, tt.AccessLabels())
	}
	frames(tt)
}

var twoSuggestions = []learn.Suggestion{
	{Term: "Ghostty", Source: learn.FromHistory},
	{Term: "Claude Code", Heard: []string{"Cloud Code"}, Source: learn.FromEdit},
}

func TestSuggestionsCountShowsBeforeThePageOpens(t *testing.T) {
	a, tt := suggestTester(t, twoSuggestions...)
	if a.dv.read {
		t.Fatal("the dictionary page was read")
	}
	if !hasExact(tt, "2") {
		t.Fatalf("no count in the sidebar: %q", firstTexts(tt))
	}
}

func TestSuggestionsAddThenRowGoes(t *testing.T) {
	a, tt := suggestTester(t, twoSuggestions...)
	a.navigate("/dictionary")
	frames(tt)
	labels := tt.AccessLabels()
	for _, want := range []string{"把 Claude Code 加入词典", "先编辑 Claude Code 再加入", "忽略 Claude Code"} {
		if !slices.Contains(labels, want) {
			t.Errorf("no control %q: %q", want, labels)
		}
	}
	if !has(tt, "听成：Cloud Code") || !has(tt, "来自你的改正") || !has(tt, "来自历史记录") {
		t.Errorf("row %q", tt.Texts())
	}
	clickControl(t, tt, "把 Claude Code 加入词典")
	ts := terms(t, a)
	if len(ts) != 1 || ts[0].Term != "Claude Code" || len(ts[0].Aliases) != 0 {
		t.Fatalf("terms %+v", ts)
	}
	labels = tt.AccessLabels()
	if slices.Contains(labels, "把 Claude Code 加入词典") {
		t.Errorf("the row stayed: %q", labels)
	}
	if !hasExact(tt, "1") {
		t.Errorf("no count of 1 in the sidebar: %q", firstTexts(tt))
	}
}

func TestSuggestionIgnoredStaysAway(t *testing.T) {
	a, tt := suggestTester(t, twoSuggestions...)
	a.navigate("/dictionary")
	clickControl(t, tt, "忽略 Claude Code")
	if slices.Contains(tt.AccessLabels(), "忽略 Claude Code") {
		t.Fatal("the row stayed")
	}
	if n, err := a.book.Propose(context.Background(), []learn.Suggestion{{Term: "Claude Code", Source: learn.FromEdit}}); n != 0 || err != nil {
		t.Fatalf("proposed again: %d, %v", n, err)
	}
	a.loadSuggestions()
	if slices.Contains(tt.AccessLabels(), "忽略 Claude Code") {
		t.Error("the ignored term is back")
	}
}

func TestSuggestionEditOpensDialogAndSaveAccepts(t *testing.T) {
	a, tt := suggestTester(t, twoSuggestions...)
	a.navigate("/dictionary")
	clickControl(t, tt, "先编辑 Claude Code 再加入")
	if !a.dv.editOpen || a.dv.edit.term != "Claude Code" || a.dv.edit.aliases != "" || a.dv.edit.id != "" {
		t.Fatalf("dialog %+v", a.dv.edit)
	}
	click(t, tt, "词条的别名")
	tt.Type("Cloud Code")
	click(t, tt, "保存")
	ts := terms(t, a)
	if len(ts) != 1 || ts[0].Term != "Claude Code" || !slices.Equal(ts[0].Aliases, []string{"Cloud Code"}) || a.dv.editOpen {
		t.Fatalf("terms %+v open %v", ts, a.dv.editOpen)
	}
	if !hasExact(tt, "1") || slices.Contains(tt.AccessLabels(), "忽略 Claude Code") {
		t.Errorf("the suggestion stayed: %q", firstTexts(tt))
	}
}

func TestSuggestionsShowFiveThenAll(t *testing.T) {
	var six []learn.Suggestion
	for i := range 6 {
		six = append(six, learn.Suggestion{Term: fmt.Sprintf("Term%d", i), Source: learn.FromHistory})
	}
	a, tt := suggestTester(t, six...)
	a.navigate("/dictionary")
	frames(tt)
	if has(tt, "Term0") || !has(tt, "Term5") {
		t.Fatalf("five, newest first: %q", tt.Texts())
	}
	clickControl(t, tt, "全部（6）")
	if !has(tt, "Term0") || has(tt, "全部（6）") {
		t.Errorf("all: %q", tt.Texts())
	}
}
