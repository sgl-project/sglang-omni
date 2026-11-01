package main

// Test helpers shared by the view tests: a test app over a temporary
// library and a fake helper supervisor (in process), and the test binary
// as the helper's fake process for the tests that start a real one.

import (
	"context"
	"encoding/json"
	"github.com/hehehai/actnowkit/sounds"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/registry"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
)

// The test binary is also the fake helper (helpertest) and the fake
// Omni server (omnitest).
func TestMain(m *testing.M) {
	if helpertest.Enabled() {
		helpertest.Main()
	}
	if omnitest.Enabled() {
		omnitest.Main()
	}
	if llmrttest.Enabled() {
		llmrttest.Main()
	}
	os.Exit(runIsolated(m))
}

// runIsolated runs the tests away from the user's ActNow family and
// Hugging Face cache: each app's family directory and model library are
// in its data folder (familyConfig), and the environment points
// elsewhere too, for what resolves the defaults.
func runIsolated(m *testing.M) int {
	tmp, err := os.MkdirTemp("", "voxtnew-family-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmp)
	realHub = models.Hub()
	os.Setenv(family.EnvDir, filepath.Join(tmp, "ActNow"))
	os.Setenv("HF_HUB_CACHE", filepath.Join(tmp, "hub"))
	os.Unsetenv("HF_HOME")
	os.Unsetenv("HF_ENDPOINT")
	familyConfig = func(a *app) actnow.Config {
		d := a.dataDir()
		hub := filepath.Join(d, "hub")
		if h, ok := testHubs.Load(d); ok {
			hub = h.(string)
		}
		return actnow.Config{Dir: family.At(filepath.Join(d, "ActNow")), Hub: hub,
			BundleID: "dev.hehehai.voxtnew", Version: "0.1.0-test", System: noApps{}}
	}
	return m.Run()
}

// testHubs are the model libraries of test apps that use another than
// their data folder's (a real one), by data folder.
var testHubs sync.Map

// realHub is the user's model library, before runIsolated moved the
// environment's: the Mac's tests of the real models read it (eval_test.go).
var realHub string

// noApps is a registry.System that finds no installed or running app.
type noApps struct{}

func (noApps) AppPaths(string) []string                { return nil }
func (noApps) Running(string) []registry.Process       { return nil }
func (noApps) BundleID(appPath string) (string, error) { return registry.BundleIDAt(appPath) }

// fakeSupervisor is a helper.Supervisor in process: Start makes it ready
// (publishing StatusChanged), and Call answers from handlers, by default
// permissions.* (the microphone not asked yet, Accessibility not allowed)
// and capture.devices (the fake helper's inputs).
type fakeSupervisor struct {
	bus *core.Bus

	mu       sync.Mutex
	status   helper.Status
	calls    []proto.Message
	handlers map[string]func(json.RawMessage) (any, error)
	// out answers output.*: pretend speakers (the default) and AirPods,
	// with the Swift helper's rules (helpertest.OutputHelper).
	out *helpertest.OutputHelper
}

func newFakeSupervisor(bus *core.Bus) *fakeSupervisor {
	f := &fakeSupervisor{bus: bus, status: helper.Status{State: helper.Stopped}, handlers: map[string]func(json.RawMessage) (any, error){}}
	perms := proto.PermissionsResult{Microphone: proto.PermNotDetermined}
	f.handlers[proto.MethodPermissionsStatus] = func(json.RawMessage) (any, error) { return perms, nil }
	f.handlers[proto.MethodCaptureDevices] = func(json.RawMessage) (any, error) {
		return proto.DevicesResult{Devices: helpertest.Devices()}, nil
	}
	f.out = &helpertest.OutputHelper{Sys: helpertest.NewOutputSystem(), PID: 4242}
	output := func(method string, fn func(raw json.RawMessage) (any, error)) { f.handlers[method] = fn }
	output(proto.MethodOutputWatch, func(raw json.RawMessage) (any, error) {
		var p proto.OutputWatchParams
		json.Unmarshal(raw, &p)
		return f.out.Watch(context.Background(), p.On)
	})
	output(proto.MethodOutputMute, func(raw json.RawMessage) (any, error) {
		var p proto.OutputMuteParams
		json.Unmarshal(raw, &p)
		return f.out.Mute(context.Background(), p)
	})
	output(proto.MethodOutputRelease, func(raw json.RawMessage) (any, error) {
		var p proto.OutputReleaseParams
		json.Unmarshal(raw, &p)
		return f.out.Release(context.Background(), p)
	})
	output(proto.MethodOutputRecover, func(raw json.RawMessage) (any, error) {
		var p proto.OutputRecoverParams
		json.Unmarshal(raw, &p)
		return f.out.Recover(context.Background(), p)
	})
	f.handlers[proto.MethodPermissionsRequest] = func(raw json.RawMessage) (any, error) {
		var p proto.PermissionsRequestParams
		json.Unmarshal(raw, &p)
		if p.Permission == proto.PermissionMicrophone {
			perms.Microphone = proto.PermGranted
		}
		return perms, nil
	}
	return f
}

// set changes the status and publishes it.
func (f *fakeSupervisor) set(st helper.Status) {
	f.mu.Lock()
	f.status = st
	f.mu.Unlock()
	f.bus.Publish(helper.StatusChanged{Status: st})
}

func (f *fakeSupervisor) Start(context.Context) error {
	f.set(helper.Status{State: helper.Ready, PID: 4242, Since: time.Now(),
		Hello: &proto.HelloResult{Protocol: proto.Version, Version: "0.1.0", PID: 4242, Methods: helpertest.Methods()}})
	return nil
}

func (f *fakeSupervisor) Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error) {
	req, _ := proto.NewRequest(id, method, params)
	f.mu.Lock()
	f.calls = append(f.calls, req)
	h, ready := f.handlers[method], f.status.State == helper.Ready
	f.mu.Unlock()
	if !ready {
		return nil, helper.ErrExited
	}
	if h == nil {
		return nil, proto.Errorf(proto.CodeMethodNotFound, "Unknown method %s", method)
	}
	v, err := h(req.Params)
	if err != nil {
		return nil, err
	}
	return json.Marshal(v)
}

func (f *fakeSupervisor) Status() helper.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.status
}

func (f *fakeSupervisor) Restart() error { return f.Start(context.Background()) }

func (f *fakeSupervisor) Stop() { f.set(helper.Status{State: helper.Stopped}) }

// methods are the methods called so far.
func (f *fakeSupervisor) methods() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, m := range f.calls {
		out = append(out, m.Method)
	}
	return out
}

// newTestApp is the app over a new library in a temporary directory, with
// the fake supervisor started and fake permissions (microphone allowed,
// Accessibility not).
func newTestApp(t *testing.T) (*app, *fakeSupervisor, *ui.Tester) {
	t.Helper()
	return newTestAppIn(t, testDir(t))
}

// newTestAppIn is newTestApp over the library in dir, which may hold one
// from an earlier "run".
func newTestAppIn(t *testing.T, dir string) (*app, *fakeSupervisor, *ui.Tester) {
	t.Helper()
	i18n.SetLocale("en")
	t.Cleanup(func() { i18n.SetLocale("en") })
	lib, err := openLibrary(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.close)
	fs := newFakeSupervisor(lib.bus)
	a := newApp(lib, &helperHost{sup: fs})
	a.permSys = &permission.Fake{Mic: permission.Granted, Grant: true}
	useTestSounds(t, a)                  // sounds_fixture_test.go
	a.playSound = func(sounds.Target) {} // tests that listen replace it
	a.measureHook = guessText            // the overlay's texts, without its view
	a.connectHelper()
	a.upd = newFakeUpdates() // updates_test.go
	a.connectUpdates()
	a.startHelper()
	t.Cleanup(a.quit)
	return a, fs, ui.NewTester(a.view, windowWidth, windowHeight)
}

// requireCJKFont fails a test that measures Chinese text's pixels when
// Linux has no font for it: Pango then draws nothing ("no ink"), which
// reads as a layout bug. CI installs fonts-noto-cjk (the root .github/workflows/voxtnew-ci.yaml);
// a Mac always has one.
func requireCJKFont(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		return
	}
	if _, err := exec.LookPath("fc-list"); err != nil {
		return // no fontconfig tools to ask: the ink check tells
	}
	out, err := exec.Command("fc-list", ":lang=zh", "family").Output()
	if err == nil && len(strings.TrimSpace(string(out))) == 0 {
		t.Fatal("no font for Chinese (fc-list :lang=zh finds none): install one, e.g. fonts-noto-cjk, as CI does")
	}
}

// frames runs the tester a few frames, so state set in one shows.
func frames(tt *ui.Tester) {
	for range 3 {
		tt.Frame()
	}
}

// click clicks a label and runs the frames after it.
func click(t *testing.T, tt *ui.Tester, label string) {
	t.Helper()
	frames(tt)
	if err := tt.Click(label); err != nil {
		t.Fatalf("click %q: %v (texts: %q)", label, err, firstTexts(tt))
	}
	frames(tt)
}

func firstTexts(tt *ui.Tester) []string {
	texts := tt.Texts()
	return texts[:min(len(texts), 40)]
}

// has reports whether the view shows s.
func has(tt *ui.Tester, s string) bool {
	frames(tt)
	return tt.HasText(s) || slices.Contains(tt.Texts(), s)
}

// hasExact reports whether the view shows a text that is s.
func hasExact(tt *ui.Tester, s string) bool {
	frames(tt)
	return slices.Contains(tt.Texts(), s)
}

// hasPrefix reports whether a text on screen starts with s.
func hasPrefix(tt *ui.Tester, s string) bool {
	for _, x := range tt.Texts() {
		if strings.HasPrefix(x, s) {
			return true
		}
	}
	return false
}

// queuePosts makes a's posts wait in a queue that the returned drain runs
// on the test's goroutine: for view tests that start real sessions, whose
// goroutines would otherwise call into the app while the test reads it.
func queuePosts(a *app) (drain func()) {
	q := make(chan func(), 4096)
	a.postQueue = func(fn func()) { q <- fn }
	return func() {
		for {
			select {
			case fn := <-q:
				fn()
			case <-time.After(20 * time.Millisecond):
				return
			}
		}
	}
}

// fakeTimers stands for the clock of app.later: tests move it on by hand
// (advance) and the timers due run, in order, on the test's goroutine.
type fakeTimers struct {
	now    time.Duration
	seq    int
	timers []fakeTimer
	added  int // timers set so far
}

type fakeTimer struct {
	at  time.Duration
	seq int
	fn  func()
}

// install makes a's timers run on f.
func (f *fakeTimers) install(a *app) *fakeTimers {
	a.laterHook = func(d time.Duration, fn func()) {
		f.seq++
		f.added++
		f.timers = append(f.timers, fakeTimer{at: f.now + d, seq: f.seq, fn: fn})
	}
	return f
}

// advance moves the clock on by d, running the timers due (and those
// they set that are due too).
func (f *fakeTimers) advance(d time.Duration) {
	end := f.now + d
	for {
		next := -1
		for i, t := range f.timers {
			if t.at <= end && (next < 0 || t.at < f.timers[next].at || t.at == f.timers[next].at && t.seq < f.timers[next].seq) {
				next = i
			}
		}
		if next < 0 {
			break
		}
		t := f.timers[next]
		f.timers = append(f.timers[:next], f.timers[next+1:]...)
		f.now = t.at
		t.fn()
	}
	f.now = end
}

// guessWidth is how wide the overlay's texts are in tests that build no
// overlay view (measureHook is guessText).
func guessWidth(k ovSized) float32 { w, _ := guessText(k); return w }

// hudTester tests a view of a floating window (floating.go), whose text
// is drawn as hudContent draws it.
func hudTester(view func(*ui.Context), w, h int) *ui.Tester {
	tt := ui.NewTester(view, w, h)
	tt.SetFontSmoothing(hudSmoothing)
	return tt
}

// cardTester is a tester of the overlay's window as large as the
// highest answer card.
func cardTester(a *app) *ui.Tester {
	return hudTester(a.overlayView, layout.AnswerW+2*layout.Pad, layout.CardMaxH+2*layout.Pad)
}
