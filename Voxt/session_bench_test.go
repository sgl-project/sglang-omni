package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type benchmarkSessionApp struct {
	*app
	tb       testing.TB
	queue    chan func()
	pasteLog string
}

type benchmarkChat struct {
	fakeChat
	onDelta    func(string)
	afterDelta func()
}

func (c *benchmarkChat) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	next := req.OnDelta
	if next != nil {
		req.OnDelta = func(text string) {
			if c.onDelta != nil {
				c.onDelta(text)
			}
			next(text)
			if c.afterDelta != nil {
				c.afterDelta()
			}
		}
	}
	return c.fakeChat.Chat(ctx, req)
}

// newBenchmarkSessionApp keeps benchmark setup independent of session_e2e_test.go.
func newBenchmarkSessionApp(tb testing.TB, opts string, set func(*settings.Settings), svc local.Services) *benchmarkSessionApp {
	tb.Helper()
	self, err := os.Executable()
	if err != nil {
		tb.Fatal(err)
	}
	lib, err := openLibrary(tb.TempDir())
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(lib.close)
	if err := lib.settings.Update(context.Background(), set); err != nil {
		tb.Fatal(err)
	}
	if svc == nil {
		svc = local.NotInstalled{}
	}
	if real, ok := svc.(realLocal); ok {
		if real.hub != "" {
			testHubs.Store(lib.store.Dir(), real.hub)
			tb.Cleanup(func() { testHubs.Delete(lib.store.Dir()) })
		} else if err := modelstest.Seed(filepath.Join(lib.store.Dir(), "hub"), real.installed...); err != nil {
			tb.Fatal(err)
		}
	} else {
		old := newLocalServices
		newLocalServices = func(*app) local.Services { return svc }
		tb.Cleanup(func() { newLocalServices = old })
	}
	pasteLog := filepath.Join(tb.TempDir(), "paste.jsonl")
	a := newApp(lib, &helperHost{bin: self, env: helpertest.Env("frame-ms=20," + helpertest.Opt("paste-log", pasteLog) + "," + opts)})
	a.measureHook = guessText
	s := &benchmarkSessionApp{app: a, tb: tb, queue: make(chan func(), 4096), pasteLog: pasteLog}
	a.copyText = func(string) error { return nil }
	a.postQueue = func(fn func()) { s.queue <- fn }
	a.muteJournal = outputJournal(tb.TempDir(), "")
	a.connectHelper()
	a.startHelper()
	tb.Cleanup(func() {
		a.quit()
		for {
			select {
			case fn := <-s.queue:
				fn()
			case <-time.After(50 * time.Millisecond):
				return
			}
		}
	})
	s.pump("the helper", func() bool { return a.helperStatus().State == helper.Ready })
	return s
}

func (s *benchmarkSessionApp) pump(what string, cond func() bool) {
	s.tb.Helper()
	deadline := time.After(10 * time.Second)
	for !cond() {
		select {
		case fn := <-s.queue:
			fn()
		case <-deadline:
			s.tb.Fatalf("timed out waiting for %s (session %s %+v)", what, s.sess.State, s.sess.Problem)
		}
	}
}

func benchmarkScript(tb testing.TB, lines ...string) string {
	tb.Helper()
	path := filepath.Join(tb.TempDir(), "events.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		tb.Fatal(err)
	}
	return helpertest.Opt("events", path)
}

func benchmarkFileSetup(tb testing.TB, asrText string, chat *fakeChat) (func(*settings.Settings), *fakeLocal, *fakeTranscriptions) {
	tb.Helper()
	tr := &fakeTranscriptions{text: asrText}
	return func(*settings.Settings) {}, withChat(&fakeLocal{ASRP: tr}, chat), tr
}
