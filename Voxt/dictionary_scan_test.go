package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// A scan runs on a real goroutine, as in the app: scanChat stops it at
// each request, posts wait in a queue, and the test goes on when they
// arrive.
type scanAnswer struct {
	text string
	err  error
}

type scanChat struct {
	requests chan llm.Request
	answers  chan scanAnswer
}

func newScanChat() *scanChat {
	return &scanChat{requests: make(chan llm.Request, 16), answers: make(chan scanAnswer, 16)}
}

func (f *scanChat) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	f.requests <- req
	select {
	case a := <-f.answers:
		return llm.Response{Text: a.text}, a.err
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

// scanTester is the app in Chinese on the Dictionary page, chat as the
// language model Transcribe uses, over count unread transcriptions.
func scanTester(t *testing.T, count int) (*app, *ui.Tester, *scanChat, chan func()) {
	t.Helper()
	a, _, tt := newTestApp(t)
	chat := newScanChat()
	a.local = &fakeLocal{Chat: chat}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.LLM = fakeLLMModel.ID })
	i18n.SetLocale("zh-Hans")
	tt.SetSize(windowWidth, 1200)
	for i := range count {
		putTranscription(t, a, i)
	}
	a.navigate("/dictionary")
	frames(tt)
	posts := make(chan func(), 64)
	a.postQueue = func(fn func()) { posts <- fn }
	clickControl(t, tt, "扫描历史…")
	drainPosts(posts)
	frames(tt)
	return a, tt, chat, posts
}

func putTranscription(t *testing.T, a *app, i int) {
	t.Helper()
	e := store.Entry{ID: fmt.Sprintf("01J%04d", i), Kind: store.KindTranscribe, Status: store.StatusDone,
		CreatedAt: time.Date(2026, 11, 7, 8, i, 0, 0, time.UTC), Result: fmt.Sprintf("Name%02d said so", i)}
	if err := a.lib.store.History().Put(context.Background(), e); err != nil {
		t.Fatal(err)
	}
}

// drainPosts runs the posts already waiting.
func drainPosts(posts chan func()) {
	for {
		select {
		case fn := <-posts:
			fn()
		default:
			return
		}
	}
}

// nextPost waits for a post and runs it.
func nextPost(tt *ui.Tester, posts chan func()) {
	(<-posts)()
	frames(tt)
}

func TestScanButtonNeedsTheTranscribeLanguageModel(t *testing.T) {
	for _, c := range []struct {
		name   string
		id     string
		models *fakeLocal
	}{
		{"nothing chosen", "", readyLocal()},
		{"nothing chosen, nothing installed", "", &fakeLocal{}},
		{"chosen one gone, another installed", "models-gone", readyLocal()},
		{"chosen, nothing installed", fakeLLMModel.ID, &fakeLocal{}},
	} {
		t.Run(c.name, func(t *testing.T) {
			a, tt := dictTester(t)
			a.local = c.models
			a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.LLM = c.id })
			i18n.SetLocale("zh-Hans")
			frames(tt)
			if !slices.Contains(tt.AccessLabels(), "扫描历史…") {
				t.Fatalf("no button: %q", tt.AccessLabels())
			}
			_ = tt.ClickControl("扫描历史…")
			frames(tt)
			if a.dv.scan.open || has(tt, "开始") {
				t.Fatalf("the button opened the dialog: %q", tt.Texts())
			}
			if !has(tt, "先给转录选一个语言模型") {
				t.Errorf("no explanation: %q", tt.Texts())
			}
			clickControl(t, tt, "打开转录页")
			if a.nav != "/transcribe" {
				t.Errorf("the link goes to %s", a.nav)
			}
		})
	}
}

func TestScanButtonOpensAndShowsWhatItWillSend(t *testing.T) {
	a, tt, _, _ := scanTester(t, 5)
	want := fmt.Sprintf("Voxt 会把 5 条转录结果发给 %s，找出人名和术语。你确认后才会加入词典。", fakeLLMModel.Name)
	if !a.dv.scan.open || !has(tt, want) {
		t.Fatalf("dialog: %q", tt.Texts())
	}
	if has(tt, "先给转录选一个语言模型") {
		t.Error("the note is on with a model chosen")
	}
}

func TestScanWithNothingNewCannotStart(t *testing.T) {
	a, tt, chat, _ := scanTester(t, 0)
	if !has(tt, "没有新的转录结果可以扫描。") {
		t.Fatalf("dialog: %q", tt.Texts())
	}
	_ = tt.ClickControl("开始")
	frames(tt)
	if a.dv.scan.running || len(chat.requests) != 0 {
		t.Error("it started")
	}
}

func TestScanShowsProgressThenTheNewSuggestions(t *testing.T) {
	a, tt, chat, posts := scanTester(t, 13)
	a.lib.store.Dictionary().Put(context.Background(), store.Term{ID: "t", Term: "OnlyInTheDictionary", Enabled: true})
	clickControl(t, tt, "开始")
	if !has(tt, "已读 0/13 · 0 条建议") {
		t.Fatalf("no progress at the start: %q", tt.Texts())
	}
	first := <-chat.requests
	chat.answers <- scanAnswer{text: `{"terms":["Name00"]}`}
	nextPost(tt, posts)
	if !has(tt, "已读 12/13 · 1 条建议") {
		t.Fatalf("no progress: %q", tt.Texts())
	}
	<-chat.requests
	chat.answers <- scanAnswer{text: `{"terms":["Name12"]}`}
	nextPost(tt, posts)
	if !has(tt, "已读 13/13 · 2 条建议") {
		t.Fatalf("no progress: %q", tt.Texts())
	}
	nextPost(tt, posts) // the end
	drainPosts(posts)   // the suggestions
	frames(tt)
	if !has(tt, "找到 2 条建议") {
		t.Fatalf("no result: %q", tt.Texts())
	}
	if has(tt, "停止") {
		t.Error("Stop stayed")
	}
	click(t, tt, "完成")
	if a.dv.scan.open {
		t.Error("the dialog stayed")
	}
	if !has(tt, "Name00") || !has(tt, "Name12") || !has(tt, "来自历史记录") {
		t.Errorf("no new suggestions: %q", tt.Texts())
	}
	if !hasExact(tt, "2") {
		t.Errorf("the sidebar has no 2: %q", firstTexts(tt))
	}
	prompt := first.Messages[0].Content + first.Messages[1].Content
	if strings.Contains(prompt, "OnlyInTheDictionary") || !strings.Contains(prompt, "Name00 said so") {
		t.Errorf("the request: %s", prompt)
	}
	if !strings.Contains(first.Messages[0].Content, "用户主要口语语言：中文") {
		t.Errorf("not in the user's language: %s", first.Messages[0].Content)
	}
}

func TestScanStopKeepsWhatItFound(t *testing.T) {
	a, tt, chat, posts := scanTester(t, 13)
	clickControl(t, tt, "开始")
	<-chat.requests
	chat.answers <- scanAnswer{text: `{"terms":["Name00"]}`}
	nextPost(tt, posts)
	<-chat.requests // the second request is out
	clickControl(t, tt, "停止")
	nextPost(tt, posts) // the end
	drainPosts(posts)
	frames(tt)
	if !has(tt, "已停止，找到 1 条建议") {
		t.Fatalf("no result: %q", tt.Texts())
	}
	if a.dv.scan.running {
		t.Error("still running")
	}
	click(t, tt, "完成")
	if !has(tt, "Name00") {
		t.Errorf("the suggestion is gone: %q", tt.Texts())
	}
	clickControl(t, tt, "扫描历史…")
	drainPosts(posts)
	frames(tt)
	if !has(tt, "Voxt 会把 1 条转录结果") {
		t.Errorf("the next scan does not start after the first batch: %q", tt.Texts())
	}
	if len(chat.requests) != 0 {
		t.Error("a request after Stop")
	}
}

func TestScanFailureShowsInTheDialog(t *testing.T) {
	_, tt, chat, posts := scanTester(t, 2)
	clickControl(t, tt, "开始")
	<-chat.requests
	chat.answers <- scanAnswer{err: &llm.Error{Code: llm.CodeServer, Err: errors.New("503")}}
	nextPost(tt, posts) // the end
	drainPosts(posts)
	frames(tt)
	if !has(tt, "语言模型已停止或没有响应。请稍后再试。") {
		t.Fatalf("no error: %q", tt.Texts())
	}
	if !has(tt, "找到 0 条建议") {
		t.Errorf("no count: %q", tt.Texts())
	}
}

func TestScanRunsOnWithTheDialogClosed(t *testing.T) {
	a, tt, chat, posts := scanTester(t, 2)
	clickControl(t, tt, "开始")
	<-chat.requests
	click(t, tt, "关闭")
	if a.dv.scan.open || !a.dv.scan.running {
		t.Fatalf("open %v running %v", a.dv.scan.open, a.dv.scan.running)
	}
	clickControl(t, tt, "扫描历史…")
	if !has(tt, "已读 0/2 · 0 条建议") {
		t.Fatalf("reopening lost the progress: %q", tt.Texts())
	}
	click(t, tt, "关闭")
	chat.answers <- scanAnswer{text: `{"terms":["Name00"]}`}
	nextPost(tt, posts) // the progress
	nextPost(tt, posts) // the end
	drainPosts(posts)
	if !has(tt, "找到 1 条建议") {
		t.Errorf("no word of the result: %q", tt.Texts())
	}
	if !has(tt, "Name00") {
		t.Errorf("no suggestion: %q", tt.Texts())
	}
}
