// Package llmrttest is a scripted voxtnew-llm for tests: a test binary
// runs it instead of its tests when VOXTNEW_FAKE_LLM is set, so the
// runtime spawns a real process and talks to it over real pipes, without
// macOS or a model. In the test package:
//
//	func TestMain(m *testing.M) {
//		if llmrttest.Enabled() {
//			llmrttest.Main()
//		}
//		os.Exit(m.Run())
//	}
//
// and point llmrt.Config at os.Executable() with Env: llmrttest.Env(opts).
//
// VOXTNEW_FAKE_LLM is a comma-separated list of options; values are
// URL-query escaped (a comma is %2C):
//
//	(empty)          speak the protocol: a chat answers "Fake answer." in pieces
//	reply=TEXT       a chat answers TEXT (pieces of up to 4 runes)
//	piece-ms=N       wait N ms before each piece (default 1)
//	first-ms=N       wait N ms before the first piece of the process's first chat (a slow prefill)
//	stall-at=K       wait stall-ms before piece K+1 (a model that stalls after K pieces)
//	stall-ms=N       that wait, in ms (default 10000)
//	load-ms=N        a load takes N ms (default 5)
//	mismatch         answer llm.hello with protocol 9
//	silent           never answer llm.hello
//	crash-start      exit 3 at once
//	crash-chat       exit 3 at the first chat's first piece
//	crash-once=PATH  crash-chat when PATH does not exist (creating it): the next process works
//	hang-cancel      ignore job.cancel
//	load-fail        every load fails with load_failed
//	chat-log=PATH    append each chat's params to PATH (JSON lines)
package llmrttest

import (
	"bufio"
	"encoding/json"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
)

const fakeEnv = "VOXTNEW_FAKE_LLM"

// Enabled reports whether this process should be the fake.
func Enabled() bool {
	_, ok := os.LookupEnv(fakeEnv)
	return ok
}

// Env is the environment that makes a re-executed test binary the fake
// with opts.
func Env(opts string) []string { return []string{fakeEnv + "=" + opts} }

// Opt escapes a value for an option (texts with commas).
func Opt(key, value string) string { return key + "=" + url.QueryEscape(value) }

// Main runs the fake with the options of the environment, and exits.
func Main() {
	run(parse(os.Getenv(fakeEnv)))
	os.Exit(0)
}

type options struct {
	reply                 string
	pieceMs, loadMs       int
	firstMs               int
	stallAt, stallMs      int
	mismatch, silent      bool
	crashChat, hangCancel bool
	loadFail              bool
	chatLog               string
}

func parse(s string) options {
	o := options{reply: "Fake answer.", pieceMs: 1, loadMs: 5, stallMs: 10000}
	for _, kv := range strings.Split(s, ",") {
		k, v, _ := strings.Cut(kv, "=")
		v, _ = url.QueryUnescape(v)
		n, _ := strconv.Atoi(v)
		switch k {
		case "reply":
			o.reply = v
		case "piece-ms":
			o.pieceMs = n
		case "load-ms":
			o.loadMs = n
		case "first-ms":
			o.firstMs = n
		case "stall-at":
			o.stallAt = n
		case "stall-ms":
			o.stallMs = n
		case "mismatch":
			o.mismatch = true
		case "silent":
			o.silent = true
		case "crash-start":
			os.Exit(3)
		case "crash-chat":
			o.crashChat = true
		case "crash-once":
			if _, err := os.Stat(v); err != nil {
				os.WriteFile(v, nil, 0o644)
				o.crashChat = true
			}
		case "hang-cancel":
			o.hangCancel = true
		case "load-fail":
			o.loadFail = true
		case "chat-log":
			o.chatLog = v
		}
	}
	return o
}

// fake is the process's state.
type fake struct {
	o       options
	outMu   sync.Mutex
	out     *bufio.Writer
	mu      sync.Mutex // guards loaded and jobs
	loaded  *wire.Model
	jobs    map[string]chan struct{} // a chat's cancel
	queue   sync.Mutex               // loads and chats one at a time
	started time.Time
	chats   int // chats started (guarded by queue)
	// prefixes are the MLX prompt prefixes "kept" (guarded by queue).
	prefixes map[string]bool
}

func run(o options) {
	f := &fake{o: o, out: bufio.NewWriter(os.Stdout), jobs: map[string]chan struct{}{}, started: time.Now()}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 64*1024), wire.MaxLine)
	for sc.Scan() {
		m, err := wire.Decode(sc.Bytes())
		if err != nil || m.Kind() != wire.KindRequest {
			continue
		}
		f.handle(m)
	}
}

func (f *fake) send(m wire.Message) {
	line, _ := wire.Encode(m)
	f.outMu.Lock()
	f.out.Write(line)
	f.out.Flush()
	f.outMu.Unlock()
}

func (f *fake) result(id string, v any) {
	m, _ := wire.NewResult(id, v)
	f.send(m)
}

func (f *fake) fail(id, code, msg string) {
	f.send(wire.NewError(id, &wire.Error{Code: code, Message: msg}))
}

func (f *fake) notify(method string, v any) {
	m, _ := wire.NewNotification(method, v)
	f.send(m)
}

func (f *fake) memory() wire.Memory {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.loaded == nil {
		return wire.Memory{FootprintBytes: 30 << 20, PeakFootprintBytes: 2 << 30, ResidentBytes: 40 << 20}
	}
	return wire.Memory{FootprintBytes: 2 << 30, PeakFootprintBytes: 2 << 30, ResidentBytes: 2 << 30, MLXActiveBytes: 1 << 30}
}

func (f *fake) handle(m wire.Message) {
	switch m.Method {
	case wire.MethodHello:
		switch {
		case f.o.silent:
		case f.o.mismatch:
			f.result(m.ID, wire.HelloResult{Protocol: 9, Version: "fake", PID: os.Getpid(), Methods: wire.Requests})
		default:
			f.notify(wire.MethodLog, wire.LogParams{Level: "info", Message: "fake voxtnew-llm ready"})
			f.result(m.ID, wire.HelloResult{Protocol: wire.Version, Version: "fake", PID: os.Getpid(), Methods: wire.Requests,
				Device: wire.Device{Name: "Fake", OS: "Linux", MemoryBytes: 16 << 30}})
		}
	case wire.MethodShutdown:
		f.result(m.ID, wire.Empty{})
		os.Exit(0)
	case wire.MethodStatus:
		f.mu.Lock()
		model := f.loaded
		f.mu.Unlock()
		f.result(m.ID, wire.StatusResult{Model: model, Memory: f.memory()})
	case wire.MethodJobCancel:
		var p wire.CancelParams
		m.DecodeParams(&p)
		f.mu.Lock()
		c, ok := f.jobs[p.Job]
		if ok && !f.o.hangCancel {
			delete(f.jobs, p.Job)
			close(c)
		}
		f.mu.Unlock()
		if f.o.hangCancel {
			return
		}
		if ok {
			// Answered after the chat, which sees the close first.
			go func() {
				time.Sleep(20 * time.Millisecond)
				f.result(m.ID, wire.CancelResult{Cancelled: true})
			}()
			return
		}
		f.result(m.ID, wire.CancelResult{Cancelled: false})
	case wire.MethodLoad:
		var p wire.LoadParams
		m.DecodeParams(&p)
		go func() {
			f.queue.Lock()
			defer f.queue.Unlock()
			ms, err := f.load(p.Model)
			if err != nil {
				f.fail(m.ID, wire.CodeLoadFailed, err.Error())
				return
			}
			f.result(m.ID, wire.LoadResult{Model: p.Model, Cached: ms == 0, LoadMs: ms, Memory: f.memory()})
		}()
	case wire.MethodUnload:
		f.mu.Lock()
		was := f.loaded != nil
		f.loaded = nil
		f.prefixes = nil
		f.mu.Unlock()
		f.result(m.ID, wire.UnloadResult{Unloaded: was, Memory: f.memory()})
	case wire.MethodChat:
		var p wire.ChatParams
		m.DecodeParams(&p)
		cancel := make(chan struct{})
		f.mu.Lock()
		f.jobs[m.ID] = cancel
		f.mu.Unlock()
		go f.chat(m.ID, p, cancel)
	default:
		f.fail(m.ID, wire.CodeMethodNotFound, "Unknown method "+m.Method)
	}
}

// load "loads" model: 0 ms when it is loaded already.
func (f *fake) load(model wire.Model) (float64, error) {
	if f.o.loadFail {
		return 0, &wire.Error{Code: wire.CodeLoadFailed, Message: "No config.json in " + model.Path}
	}
	f.mu.Lock()
	same := f.loaded != nil && *f.loaded == model
	f.mu.Unlock()
	if same {
		return 0, nil
	}
	time.Sleep(time.Duration(f.o.loadMs) * time.Millisecond)
	f.mu.Lock()
	f.loaded = &model
	f.mu.Unlock()
	f.prefixes = nil
	return float64(f.o.loadMs), nil
}

func (f *fake) chat(id string, p wire.ChatParams, cancel chan struct{}) {
	f.queue.Lock()
	defer f.queue.Unlock()
	defer func() {
		f.mu.Lock()
		delete(f.jobs, id)
		f.mu.Unlock()
	}()
	if f.o.chatLog != "" {
		if fl, err := os.OpenFile(f.o.chatLog, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			json.NewEncoder(fl).Encode(p)
			fl.Close()
		}
	}
	start := time.Now()
	loadMs, err := f.load(p.Model)
	if err != nil {
		f.fail(id, wire.CodeLoadFailed, err.Error())
		return
	}
	cache := f.promptCache(p)
	f.chats++
	rs := []rune(f.o.reply)
	var first time.Duration
	var text strings.Builder
	tokens := 0
	finish := wire.FinishStop
	for i := 0; i < len(rs); i += 4 {
		if tokens >= p.MaxTokens {
			finish = wire.FinishLength
			break
		}
		wait := f.o.pieceMs
		switch {
		case tokens == 0 && f.chats == 1 && f.o.firstMs > 0:
			wait = f.o.firstMs
		case f.o.stallAt > 0 && tokens == f.o.stallAt:
			wait = f.o.stallMs
		}
		select {
		case <-cancel:
			f.fail(id, wire.CodeCancelled, "The chat was cancelled")
			return
		case <-time.After(time.Duration(wait) * time.Millisecond):
		}
		if f.o.crashChat {
			os.Exit(3)
		}
		piece := string(rs[i:min(i+4, len(rs))])
		text.WriteString(piece)
		tokens++
		if tokens == 1 {
			first = time.Since(start)
		}
		f.notify(wire.MethodChatDelta, wire.DeltaParams{Job: id, Text: piece})
	}
	elapsed := time.Since(start)
	rate := 0.0
	if tokens > 1 {
		rate = float64(tokens-1) / (elapsed - first).Seconds()
	}
	f.result(id, wire.ChatResult{
		Text: text.String(), Finish: finish,
		Usage: wire.Usage{PromptTokens: 40, CompletionTokens: tokens},
		Timings: wire.Timings{LoadMs: loadMs, PromptMs: 1, FirstTokenMs: float64(first.Microseconds()) / 1000,
			TotalMs: float64(elapsed.Microseconds()) / 1000, TokensPerSecond: rate, Cache: cache},
		Memory: f.memory(),
	})
}

// promptCache is the prompt cache's use as an MLX model reports it: the
// messages before the last are the shared prefix, 30 "tokens" each, a hit
// when a chat had them before.
func (f *fake) promptCache(p wire.ChatParams) *wire.CacheUse {
	if p.Model.Format != wire.FormatMLX {
		return nil
	}
	if len(p.Messages) < 2 {
		return &wire.CacheUse{}
	}
	key, _ := json.Marshal(p.Messages[:len(p.Messages)-1])
	hit := f.prefixes[string(key)]
	if f.prefixes == nil {
		f.prefixes = map[string]bool{}
	}
	f.prefixes[string(key)] = true
	return &wire.CacheUse{Hit: hit, PrefixTokens: 30 * (len(p.Messages) - 1)}
}
