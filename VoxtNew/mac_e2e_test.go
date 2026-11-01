package main

// Headless end-to-end tests on a real Mac (VOXTNEW_MAC_E2E=1), for what
// Linux cannot reach: the real supervisor and the real Swift helper. They
// open no window, never change the app in front, post no key, install no
// event tap and leave the clipboard alone:
//
//   - the helper records from a file (VOXTNEW_HELPER_CAPTURE_FILE, the
//     test-only input), never the microphone, so no prompt shows;
//   - text I/O is macText: the real, read-only textio.focus, and a paste
//     that only keeps the text;
//   - hotkey.status only: no hotkey.configure;
//   - browser.url only for browsers already running, by pid, three times;
//     once app.configure lists the app's Chromium browsers, one without a
//     page tree gets AXManualAccessibility, as in the app (an attribute,
//     no change of the app in front).
//   - app.icons draws icons: no window, no focus.
//
// TestMacHelperReadOnly, TestMacHelperIcons, TestMacHelperCaptureFile and
// TestMacE2ESessionLocal (a session over the fake local models) need no
// model and reach no network. docs/manual-test.md, "Automatic Mac tests
// (headless)", says how to run them.

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"image/png"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// The helper's test-only file input (helper/README.md).
const (
	envCaptureFile  = "VOXTNEW_HELPER_CAPTURE_FILE"
	envCaptureSpeed = "VOXTNEW_HELPER_CAPTURE_SPEED"
)

// macE2E skips t unless VOXTNEW_MAC_E2E=1 on macOS, and returns the
// helper to run: $VOXTNEW_HELPER, else the staged one
// (scripts/build-helper.sh).
func macE2E(t *testing.T) string {
	t.Helper()
	if os.Getenv("VOXTNEW_MAC_E2E") != "1" {
		t.Skip("set VOXTNEW_MAC_E2E=1 on a Mac (docs/manual-test.md, automatic Mac tests)")
	}
	if runtime.GOOS != "darwin" {
		t.Skip("macOS only")
	}
	bin := os.Getenv("VOXTNEW_HELPER")
	if bin == "" {
		bin = filepath.Join("resources", "darwin-arm64", "helper", "voxtnew-helper")
	}
	if _, err := os.Stat(bin); err != nil {
		t.Fatalf("no helper at %s: scripts/build-helper.sh [--out DIR], or VOXTNEW_HELPER (%v)", bin, err)
	}
	abs, _ := filepath.Abs(bin)
	return abs
}

// macHelper is the real helper under the real supervisor.
type macHelper struct {
	helper.Supervisor
	t *testing.T

	mu      sync.Mutex
	notices map[string]int // notifications without a job, by method
}

// startMacHelper starts the helper with env added and waits for its
// handshake; it is shut down, and checked gone, when t ends.
func startMacHelper(t *testing.T, bin string, env ...string) *macHelper {
	t.Helper()
	h := &macHelper{t: t, notices: map[string]int{}}
	h.Supervisor = helper.New(helper.Config{
		Bin: bin, AppVersion: "mac-e2e", Env: env, Bus: core.NewBus(),
		Log: func(e core.LogEntry) { t.Logf("helper %s: %s", e.Level, e.Message) },
		OnNotify: func(m proto.Message) {
			h.mu.Lock()
			h.notices[m.Method]++
			h.mu.Unlock()
		},
		Logf: t.Logf,
	})
	if err := h.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(15 * time.Second)
	for h.Status().State != helper.Ready {
		if st := h.Status(); st.State == helper.Failed || time.Now().After(deadline) {
			t.Fatalf("helper %s: %s %s", st.State, st.Err, st.StderrTail)
		}
		time.Sleep(20 * time.Millisecond)
	}
	pid := h.Status().PID
	t.Cleanup(func() {
		h.Stop()
		for i := 0; i < 50 && syscall.Kill(pid, 0) == nil; i++ {
			time.Sleep(100 * time.Millisecond)
		}
		if syscall.Kill(pid, 0) == nil {
			t.Errorf("the helper (pid %d) is still running", pid)
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.notices[proto.MethodHotkeyEvent] > 0 || h.notices[proto.MethodHotkeyState] > 0 {
			t.Errorf("hotkey notifications without a tap: %v", h.notices)
		}
	})
	return h
}

func macCall[T any](t *testing.T, h helper.Supervisor, method string, params any) T {
	t.Helper()
	var out T
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	raw, err := h.Call(ctx, "", method, params, nil)
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	return out
}

// TestMacHelperIcons asks the real helper for apps' icons (app.icons):
// the Finder's and Safari's by bundle id, WeChat's and Arc's when they
// are installed, the first installed apps by path, and a removed app's
// (none). With VOXTNEW_E2E_ICONS=<dir> it
// saves them there to look at. Read-only: no window, no focus.
func TestMacHelperIcons(t *testing.T) {
	h := startMacHelper(t, macE2E(t))
	installed := macCall[proto.AppListResult](t, h, proto.MethodAppList, proto.AppListParams{Installed: true})
	optional := map[string]bool{"com.tencent.xinWeChat": true, "company.thebrowser.Browser": true}
	refs := []proto.AppRef{{BundleID: "com.apple.finder"}, {BundleID: "com.apple.Safari"}, {BundleID: "com.tencent.xinWeChat"},
		{BundleID: "company.thebrowser.Browser"}, {BundleID: "com.example.not-installed"}}
	for _, a := range installed.Apps {
		if len(refs) >= 10 {
			break
		}
		if a.Path != "" && !slices.ContainsFunc(refs, func(r proto.AppRef) bool { return strings.EqualFold(r.BundleID, a.BundleID) }) {
			refs = append(refs, proto.AppRef{BundleID: a.BundleID, Path: a.Path})
		}
	}
	t0 := time.Now()
	icons := macCall[proto.AppIconsResult](t, h, proto.MethodAppIcons, proto.AppIconsParams{Apps: refs, Size: proto.IconSize})
	t.Logf("app.icons: %d icons in %v", len(icons.Icons), time.Since(t0).Round(time.Millisecond))
	if len(icons.Icons) != len(refs) {
		t.Fatalf("app.icons: %d icons for %d apps", len(icons.Icons), len(refs))
	}
	iconDir := os.Getenv("VOXTNEW_E2E_ICONS")
	for i, ic := range icons.Icons {
		if ic.BundleID != refs[i].BundleID {
			t.Errorf("app.icons %d: %s, asked for %s", i, ic.BundleID, refs[i].BundleID)
		}
		if refs[i].BundleID == "com.example.not-installed" {
			if ic.PNG != nil {
				t.Errorf("app.icons: an icon for a removed app")
			}
			continue
		}
		if ic.PNG == nil && optional[ic.BundleID] {
			t.Logf("  %s: not installed", ic.BundleID)
			continue
		}
		img, err := png.Decode(bytes.NewReader(ic.PNG))
		if err != nil || img.Bounds().Dx() != proto.IconSize || img.Bounds().Dy() != proto.IconSize {
			t.Errorf("app.icons %s: %d bytes, %v", ic.BundleID, len(ic.PNG), err)
			continue
		}
		t.Logf("  %s: %d bytes", ic.BundleID, len(ic.PNG))
		if iconDir != "" {
			os.MkdirAll(iconDir, 0o755)
			os.WriteFile(filepath.Join(iconDir, ic.BundleID+".png"), ic.PNG, 0o644)
		}
	}
}

// TestMacHelperReadOnly runs the helper's read-only methods from wherever
// the test runs (an ssh session: Accessibility as sshd has it, the
// microphone not asked), and checks nothing changed the app in front.
func TestMacHelperReadOnly(t *testing.T) {
	h := startMacHelper(t, macE2E(t))
	hello := h.Status().Hello
	t.Logf("hello: helper %s, protocol %d, %s %s, %d methods", hello.Version, hello.Protocol, hello.Device.Name, hello.Device.OS, len(hello.Methods))

	perms := macCall[proto.PermissionsResult](t, h, proto.MethodPermissionsStatus, nil)
	t.Logf("permissions.status: microphone %s, accessibility %v", perms.Microphone, perms.Accessibility)

	hs := macCall[proto.HotkeyStatus](t, h, proto.MethodHotkeyStatus, nil)
	t.Logf("hotkey.status: %+v", hs)
	if hs.Installed || hs.Bindings != 0 || hs.Events != 0 {
		t.Errorf("a tap without bindings: %+v", hs)
	}
	if hs.Accessibility != perms.Accessibility {
		t.Errorf("hotkey.status says accessibility %v, permissions.status %v", hs.Accessibility, perms.Accessibility)
	}

	front := macCall[proto.App](t, h, proto.MethodAppFrontmost, nil)
	t.Logf("app.frontmost: pid %d %s %q", front.PID, front.BundleID, front.Name)
	if front.PID <= 0 {
		t.Errorf("no app in front: %+v", front)
	}

	running := macCall[proto.AppListResult](t, h, proto.MethodAppList, proto.AppListParams{})
	var names []string
	for _, a := range running.Apps {
		names = append(names, a.Name)
		if !a.Running || a.PID <= 0 {
			t.Errorf("app.list: %+v is not running", a)
		}
	}
	t.Logf("app.list: %d running: %s", len(running.Apps), strings.Join(names, ", "))
	if len(running.Apps) == 0 {
		t.Error("app.list: nothing running")
	}
	installed := macCall[proto.AppListResult](t, h, proto.MethodAppList, proto.AppListParams{Installed: true})
	t.Logf("app.list installed: %d apps", len(installed.Apps))
	if len(installed.Apps) < len(running.Apps) {
		t.Errorf("app.list installed: %d apps, fewer than the %d running", len(installed.Apps), len(running.Apps))
	}

	devs := macCall[proto.DevicesResult](t, h, proto.MethodCaptureDevices, nil)
	t.Logf("capture.devices: %d inputs", len(devs.Devices))
	for _, d := range devs.Devices {
		t.Logf("  %s (%d ch, %d Hz, %s, default %v)", d.Name, d.Channels, d.SampleRate, d.Transport, d.Default)
	}

	// Browsers already running, by pid; never one that is not. As the
	// helper starts (no list yet: a Chromium browser without a page tree
	// reads none), after app.configure (it is retried after
	// AXManualAccessibility), and once more (the tree is there).
	browsers := []string{"company.thebrowser.Browser", "com.google.Chrome", "com.apple.Safari", "com.microsoft.edgemac", "com.brave.Browser"}
	for round := 1; round <= 3; round++ {
		if round == 2 {
			macCall[proto.Empty](t, h, proto.MethodAppConfigure, proto.AppConfigureParams{ManualAccessibility: appctx.TreeOnDemand(nil)})
		}
		for _, id := range browsers {
			var app *proto.AppInfo
			for i := range running.Apps {
				if running.Apps[i].BundleID == id {
					app = &running.Apps[i]
					break
				}
			}
			if app == nil {
				if round == 1 {
					t.Logf("browser.url %s: not running, not asked", id)
				}
				continue
			}
			r := macCall[proto.BrowserURLResult](t, h, proto.MethodBrowserURL, proto.BrowserURLParams{PID: app.PID})
			host := ""
			if u, err := url.Parse(r.URL); err == nil && r.URL != "" {
				host = u.Scheme + "://" + u.Host // the address's path stays private
			}
			t.Logf("browser.url #%d %s (pid %d, in front %v): via %s, %s in %.1f ms, retried %v %s", round, id, app.PID, app.PID == front.PID, r.Via, host, r.MS, r.Retried, r.Error)
			if perms.Accessibility && r.URL == "" && app.PID == front.PID {
				// Not a failure of the test: the window may have no focus.
				t.Logf("FINDING browser.url %s is in front but gave no URL (%s)", id, r.Error)
			}
		}
	}

	focus := macCall[proto.FocusResult](t, h, proto.MethodTextFocus, nil)
	t.Logf("textio.focus: pid %d (%+v) role %s/%s editable %v, value %d chars %s", focus.PID, focus.App, focus.Role, focus.Subrole, focus.Editable, focus.ValueLength, focus.Error)
	if focus.App == nil || focus.App.PID != focus.PID && focus.Error == "" {
		t.Errorf("textio.focus names no app, or another one than its pid's: %+v", focus.App)
	}
	sel := macCall[proto.SelectionResult](t, h, proto.MethodTextSelection, proto.SelectionParams{AllowCopy: false})
	t.Logf("textio.selection (no ⌘C): via %s, %d chars, %.1f ms", sel.Via, len([]rune(sel.Text)), sel.MS)
	if sel.Via == proto.ViaCopy {
		t.Error("textio.selection copied without allowCopy")
	}

	after := macCall[proto.App](t, h, proto.MethodAppFrontmost, nil)
	if after.PID != front.PID {
		t.Logf("the app in front changed meanwhile: %s → %s (the user, or another test)", front.Name, after.Name)
	}
	again := macCall[proto.PermissionsResult](t, h, proto.MethodPermissionsStatus, nil)
	if again != perms {
		t.Errorf("permissions changed: %+v → %+v", perms, again)
	}
}

// capture is one capture.start job's notifications and answer.
type capture struct {
	started *proto.CaptureStartedParams
	pcm     []byte
	frames  []proto.PCMParams
	levels  []proto.LevelParams
	result  proto.CaptureResult
	err     error
	took    time.Duration
}

// record runs a capture.start job; stopAfter > 0 stops it then, cancelAfter
// > 0 cancels it (its context) then.
func record(t *testing.T, h helper.Supervisor, id string, p proto.CaptureParams, stopAfter, cancelAfter time.Duration) *capture {
	t.Helper()
	c := &capture{}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if cancelAfter > 0 {
		time.AfterFunc(cancelAfter, cancel)
	}
	if stopAfter > 0 {
		time.AfterFunc(stopAfter, func() {
			if _, err := h.Call(context.Background(), "", proto.MethodCaptureStop, proto.CaptureStopParams{Job: id}, nil); err != nil {
				t.Errorf("capture.stop: %v", err)
			}
		})
	}
	t0 := time.Now()
	raw, err := h.Call(ctx, id, proto.MethodCaptureStart, p, func(m proto.Message) {
		switch m.Method {
		case proto.MethodCaptureStarted:
			c.started = new(proto.CaptureStartedParams)
			m.DecodeParams(c.started)
		case proto.MethodCapturePCM:
			var f proto.PCMParams
			m.DecodeParams(&f)
			c.frames = append(c.frames, f)
			c.pcm = append(c.pcm, f.Data...)
		case proto.MethodCaptureLevel:
			var l proto.LevelParams
			m.DecodeParams(&l)
			c.levels = append(c.levels, l)
		}
	})
	c.took = time.Since(t0)
	if c.err = err; err == nil {
		c.err = json.Unmarshal(raw, &c.result)
	}
	return c
}

// checkFrames: frames from 0 without gaps, counted by the result.
func (c *capture) checkFrames(t *testing.T) {
	t.Helper()
	off := 0
	for i, f := range c.frames {
		if f.Seq != i || f.Offset != off {
			t.Fatalf("frame %d: seq %d offset %d, want offset %d", i, f.Seq, f.Offset, off)
		}
		off += len(f.Data) / 2
	}
	if c.result.Frames != len(c.frames) || c.result.Samples != off {
		t.Errorf("result %d frames %d samples, got %d frames %d samples", c.result.Frames, c.result.Samples, len(c.frames), off)
	}
}

func samples16(b []byte) []float64 {
	out := make([]float64, len(b)/2)
	for i := range out {
		out[i] = float64(int16(binary.LittleEndian.Uint16(b[2*i:]))) / 32768
	}
	return out
}

// bestMatch is the normalised correlation of got with want at the lag
// (0–maxLag samples of delay in got) where they match best.
func bestMatch(want, got []float64, maxLag int) (corr float64, lag int) {
	n := len(want) - maxLag
	for l := 0; l <= maxLag; l++ {
		if l+n > len(got) {
			break
		}
		var xy, xx, yy float64
		for i := 0; i < n; i++ {
			x, y := want[i], got[i+l]
			xy += x * y
			xx += x * x
			yy += y * y
		}
		if c := xy / math.Sqrt(xx*yy+1e-12); c > corr {
			corr, lag = c, l
		}
	}
	return corr, lag
}

// TestMacHelperCaptureFile: the helper records the speech sample through
// its capture path (the test-only file input), at 4× speed: the frames
// match the file, stop, cancel and maxSeconds end the job as they end a
// microphone's, and the microphone is never asked for.
func TestMacHelperCaptureFile(t *testing.T) {
	bin := macE2E(t)
	sample := speechSample()
	data, err := os.ReadFile(sample)
	if err != nil {
		t.Fatal(err)
	}
	want, rate, _, err := asr.ReadWAV(data)
	if err != nil {
		t.Fatal(err)
	}
	seconds := float64(len(want)/2) / float64(rate)
	h := startMacHelper(t, bin, envCaptureFile+"="+sample, envCaptureSpeed+"=4")
	before := macCall[proto.PermissionsResult](t, h, proto.MethodPermissionsStatus, nil)
	params := proto.CaptureParams{SampleRate: 16000, FrameMS: 100, LevelHz: 20}

	t.Run("stop", func(t *testing.T) {
		// The file at 4×, then silence until the stop.
		stopAt := time.Duration(seconds/4*float64(time.Second)) + 600*time.Millisecond
		c := record(t, h, "mt-stop", params, stopAt, 0)
		if c.err != nil {
			t.Fatal(c.err)
		}
		t.Logf("started %+v; result %+v in %v; %d frames, %d levels", *c.started, c.result, c.took.Round(time.Millisecond), len(c.frames), len(c.levels))
		if c.started == nil || c.started.DeviceName != "sample-zh-en.wav" || c.started.Device != "file:sample-zh-en.wav" ||
			c.started.DeviceSampleRate != 16000 || c.started.DeviceChannels != 1 || c.started.SampleRate != 16000 || c.started.FellBack {
			t.Errorf("started %+v", c.started)
		}
		if c.result.Reason != proto.CaptureStopped {
			t.Errorf("reason %s", c.result.Reason)
		}
		c.checkFrames(t)
		got := samples16(c.pcm)
		ref := samples16(want)
		if len(got) < len(ref) {
			t.Fatalf("%d samples, the file has %d", len(got), len(ref))
		}
		corr, lag := bestMatch(ref, got, 256)
		t.Logf("matches the file: correlation %.5f at a lag of %d samples", corr, lag)
		if corr < 0.99 {
			t.Errorf("the recording does not match the file: correlation %.4f", corr)
		}
		// Levels at ~20 Hz of what was played (4× the wall clock).
		if n := len(c.levels); n < int(seconds*20)-2 {
			t.Errorf("%d levels for %.1f s", n, seconds)
		}
		var loud int
		for _, l := range c.levels {
			if l.Peak > -20 {
				loud++
			}
		}
		if loud < int(seconds*10) {
			t.Errorf("only %d of %d levels show speech", loud, len(c.levels))
		}
	})

	t.Run("cancel", func(t *testing.T) {
		c := record(t, h, "mt-cancel", params, 0, 400*time.Millisecond)
		var pe *proto.Error
		if !errors.As(c.err, &pe) || pe.Code != proto.CodeCancelled {
			t.Errorf("cancel: %v", c.err)
		}
		t.Logf("cancelled after %d frames: %v", len(c.frames), c.err)
	})

	t.Run("maxSeconds", func(t *testing.T) {
		p := params
		p.MaxSeconds = 1
		c := record(t, h, "mt-max", p, 0, 0)
		if c.err != nil || c.result.Reason != proto.CaptureMaxSeconds || math.Abs(c.result.Seconds-1) > 0.15 {
			t.Errorf("maxSeconds: %+v %v", c.result, c.err)
		}
		c.checkFrames(t)
		t.Logf("maxSeconds: %+v in %v", c.result, c.took.Round(time.Millisecond))
	})

	after := macCall[proto.PermissionsResult](t, h, proto.MethodPermissionsStatus, nil)
	if after.Microphone != before.Microphone {
		t.Errorf("the microphone's permission changed: %s → %s", before.Microphone, after.Microphone)
	}

	// A file at another rate and with two channels: resampled and mixed.
	t.Run("stereo 22.05 kHz", func(t *testing.T) {
		tone, _ := filepath.Abs(filepath.Join("helper", "Fixtures", "audio", "tone-22050-stereo.wav"))
		h2 := startMacHelper(t, bin, envCaptureFile+"="+tone)
		c := record(t, h2, "mt-tone", params, 700*time.Millisecond, 0)
		if c.err != nil {
			t.Fatal(c.err)
		}
		if c.started.DeviceSampleRate != 22050 || c.started.DeviceChannels != 2 {
			t.Errorf("started %+v", c.started)
		}
		c.checkFrames(t)
		var peak float64
		for _, s := range samples16(c.pcm) {
			peak = max(peak, math.Abs(s))
		}
		t.Logf("tone: %+v, peak %.3f", c.result, peak)
		if math.Abs(peak-0.375) > 0.03 {
			t.Errorf("peak %.3f, want the mix of 0.5 and 0.25: 0.375", peak)
		}
	})
}

// macText is text I/O that changes nothing: the helper's real textio.focus
// is read (and logged), then the session sees an editable field in the
// target, and a paste only keeps the text (no clipboard, no ⌘V).
type macText struct {
	t   *testing.T
	h   helper.Supervisor
	pid int

	mu     sync.Mutex
	pasted []string
	focus  int
}

func (m *macText) Focus(ctx context.Context) (proto.FocusResult, error) {
	f, err := session.HelperText{H: m.h}.Focus(ctx)
	if err != nil {
		return f, err
	}
	m.mu.Lock()
	m.focus++
	m.mu.Unlock()
	m.t.Logf("textio.focus (real): pid %d role %s editable %v", f.PID, f.Role, f.Editable)
	return proto.FocusResult{PID: m.pid, Role: "AXTextArea", Editable: true}, nil
}

func (m *macText) Paste(_ context.Context, p proto.PasteParams) (proto.PasteResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pasted = append(m.pasted, p.Text)
	return proto.PasteResult{Posted: true}, nil
}

// macSession is a session.Manager as the app wires it (session_link.go):
// the helper's audio, the app's resolver over local models, the history
// and dictionary of a library in a temporary directory; macText instead
// of the helper's text I/O.
type macSession struct {
	*session.Manager
	lib   *library
	text  *macText
	ended chan session.Snapshot
	seen  chan session.Snapshot
	mu    sync.Mutex
	logs  []string // the session's log lines
}

// logged reports whether a session's log line contains sub.
func (ms *macSession) logged(sub string) bool {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	return slices.ContainsFunc(ms.logs, func(l string) bool { return strings.Contains(l, sub) })
}

func newMacSession(t *testing.T, h helper.Supervisor, target proto.App, svc local.Services, set func(*settings.Settings)) *macSession {
	t.Helper()
	lib, err := openLibrary(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.close)
	if err := lib.store.Dictionary().Put(context.Background(), store.Term{ID: "t1", Term: "VoxtNew", Aliases: []string{"Aura Link", "Oralink"}, Enabled: true}); err != nil {
		t.Fatal(err)
	}
	s := settings.Defaults()
	s.General.MainLanguage = "zh"
	s.Privacy.KeepAudio = true
	set(&s)
	ms := &macSession{lib: lib, text: &macText{t: t, h: h, pid: target.PID}, ended: make(chan session.Snapshot, 4), seen: make(chan session.Snapshot, 1024)}
	ms.Manager = session.New(session.Config{
		Audio: session.HelperAudio{H: h}, Text: ms.text, Resolve: resolver(svc),
		History: lib.store.History(),
		Glossary: func(ctx context.Context) ([]store.Term, error) {
			return lib.store.Dictionary().List(ctx, "")
		},
		Settings: func() settings.Settings { return s },
		OnChange: func(sn session.Snapshot) {
			select {
			case ms.seen <- sn:
			default:
			}
			if sn.State.Ended() {
				ms.ended <- sn
			}
		},
		Log: func(level, msg string) {
			t.Log("session", level, msg)
			ms.mu.Lock()
			ms.logs = append(ms.logs, level+" "+msg)
			ms.mu.Unlock()
		},
		LLMTimeout: 20 * time.Second,
		NewID:      func() string { return core.NewID(time.Now()) },
	})
	return ms
}

func (ms *macSession) wait(t *testing.T) session.Snapshot {
	t.Helper()
	select {
	case end := <-ms.ended:
		return end
	case <-time.After(90 * time.Second):
		t.Fatal("no end")
	}
	return session.Snapshot{}
}

func (ms *macSession) history(t *testing.T) []store.Entry {
	t.Helper()
	es, err := ms.lib.store.History().List(context.Background(), store.HistoryQuery{})
	if err != nil {
		t.Fatal(err)
	}
	return es
}

// TestMacE2ESessionLocal: a Transcribe session as the app runs it, the
// speech sample played by the real helper in real time, then recognised
// and cleaned up by the fake local models (local_fakes_test.go: no model
// runs; TestMacE2EOmni runs the real ones): the helper's audio reaches the speech
// model whole, the result is delivered and kept with its recording.
func TestMacE2ESessionLocal(t *testing.T) {
	bin := macE2E(t)
	sample := speechSample()
	h := startMacHelper(t, bin, envCaptureFile+"="+sample)
	front := macCall[proto.App](t, h, proto.MethodAppFrontmost, nil)
	tr := &fakeTranscriptions{text: "讨论一下 voxtnew 的发布计划"}
	chat := &fakeChat{text: "讨论一下 VoxtNew 的发布计划。"}
	ms := newMacSession(t, h, front, withChat(&fakeLocal{ASRP: tr}, chat), func(*settings.Settings) {})
	t0 := time.Now()
	id, err := ms.Start(session.StartRequest{Kind: store.KindTranscribe, Target: front})
	if err != nil {
		t.Fatal(err)
	}
	var heard session.Snapshot
	for heard.HeardAt.IsZero() {
		select {
		case sn := <-ms.seen:
			if sn.State.Ended() {
				t.Fatalf("ended before audio: %s %+v", sn.State, sn.Problem)
			}
			heard = sn
		case <-time.After(10 * time.Second):
			t.Fatal("no audio")
		}
	}
	t.Logf("heard after %v from %s", heard.HeardAt.Sub(t0).Round(time.Millisecond), heard.Device)
	time.Sleep(10*time.Second + 800*time.Millisecond) // the sample is 10.5 s
	ms.Stop(id)
	end := ms.wait(t)
	calls, size := tr.uploads()
	t.Logf("%q → %q (audio %v, sent %v in %d bytes, deliver %v, total %v)", end.Raw, end.Result,
		end.Timings.Audio.Round(time.Millisecond), end.Timings.Sent.Round(time.Millisecond), size,
		end.Timings.Deliver.Round(time.Millisecond), end.Timings.Total.Round(time.Millisecond))
	if end.State != session.Done || end.Raw != tr.text || end.Result != chat.text || !end.Inserted || end.Problem != nil {
		t.Fatalf("end %s %+v", end.State, end.Problem)
	}
	if end.ASR.ID != fakeASRModel.ID || end.LLM.ID != fakeLLMModel.ID || end.Device != "sample-zh-en.wav" {
		t.Errorf("models %+v %+v, device %q", end.ASR, end.LLM, end.Device)
	}
	// The speech went to the model (less the silence around it), once.
	if tm := end.Timings; calls != 1 || size < 200_000 || tm.Audio < 10*time.Second || tm.Audio > 13*time.Second || tm.Deliver > 2*time.Second {
		t.Errorf("%d uploads of %d bytes, timings %+v", calls, size, tm)
	}
	if reqs := chat.requests(); len(reqs) != 1 || !strings.Contains(chatMessages(reqs[0])[0][1], "VoxtNew") {
		t.Errorf("the chat lacks the dictionary's VoxtNew: %v", reqs)
	}
	if p := ms.text.pasted; len(p) != 1 || p[0] != end.Result {
		t.Errorf("pasted %q", p)
	}
	es := ms.history(t)
	if len(es) != 1 {
		t.Fatalf("%d history entries", len(es))
	}
	e := es[0]
	if e.ID != id || e.Kind != store.KindTranscribe || e.Status != store.StatusDone || e.Result != end.Result || e.RawText != end.Raw ||
		e.ASRProvider != fakeASRModel.ID || e.LLMProvider != fakeLLMModel.ID || e.Delivery != "insert" ||
		e.AppName != front.Name || e.AudioMS < 10000 || e.Audio == "" {
		t.Errorf("history %+v", e)
	}
	if fi, err := os.Stat(filepath.Join(ms.lib.store.History().AudioDir(), e.Audio)); err != nil || fi.Size() < 300_000 {
		t.Errorf("kept recording: %v", err)
	}
}

// speechSample is the speech sample's path (testdata/speech), from the
// module's root.
func speechSample() string {
	dir, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Join(dir, "testdata", "speech", "sample-zh-en.wav")
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return filepath.Join("testdata", "speech", "sample-zh-en.wav")
		}
		dir = parent
	}
}
