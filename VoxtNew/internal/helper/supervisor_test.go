package helper

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// The test binary is also the fake helper (helpertest).
func TestMain(m *testing.M) {
	if helpertest.Enabled() {
		helpertest.Main()
	}
	os.Exit(m.Run())
}

// harness is a supervisor over the fake helper, recording what it
// publishes and the notifications that belong to no request.
type harness struct {
	sup      Supervisor
	mu       sync.Mutex
	statuses []Status
	logs     []core.LogEntry
	notes    []proto.Message
}

func start(t *testing.T, opts string, tweak ...func(*Config)) *harness {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	h := &harness{}
	bus := core.NewBus()
	core.On(bus, func(ev StatusChanged) { h.mu.Lock(); h.statuses = append(h.statuses, ev.Status); h.mu.Unlock() })
	cfg := Config{
		Bin: self, AppVersion: "test", LogDir: t.TempDir(), Bus: bus, Env: helpertest.Env(opts),
		Log:      func(e core.LogEntry) { h.mu.Lock(); h.logs = append(h.logs, e); h.mu.Unlock() },
		OnNotify: func(m proto.Message) { h.mu.Lock(); h.notes = append(h.notes, m); h.mu.Unlock() },
		Logf:     t.Logf, HelloTimeout: 2 * time.Second, CancelGrace: 500 * time.Millisecond,
	}
	for _, fn := range tweak {
		fn(&cfg)
	}
	h.sup = New(cfg)
	t.Cleanup(h.sup.Stop)
	h.sup.Start(context.Background())
	return h
}

// wait polls cond for up to 5 s.
func wait(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

func (h *harness) ready(t *testing.T) Status {
	t.Helper()
	wait(t, "the helper to be ready", func() bool { return h.sup.Status().State == Ready })
	return h.sup.Status()
}

func request[T any](t *testing.T, h *harness, method string, params any) (T, error) {
	t.Helper()
	var out T
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	raw, err := h.sup.Call(ctx, "", method, params, nil)
	if err == nil {
		err = json.Unmarshal(raw, &out)
	}
	return out, err
}

func TestHandshakeAndControl(t *testing.T) {
	h := start(t, "")
	st := h.ready(t)
	if st.Hello == nil || st.Hello.Version != "fake" || st.PID == 0 || !st.Has(proto.MethodBrowserURL) || st.Has("tts.synthesize") {
		t.Fatalf("status %+v", st)
	}
	devs, err := request[proto.DevicesResult](t, h, proto.MethodCaptureDevices, nil)
	if err != nil || len(devs.Devices) != 2 || !devs.Devices[0].Default {
		t.Fatalf("devices %+v %v", devs, err)
	}
	u, err := request[proto.BrowserURLResult](t, h, proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: "com.google.Chrome"})
	if err != nil || !strings.HasPrefix(u.URL, "https://mail.google.com/") || u.Via != proto.URLViaAXDocument {
		t.Errorf("url %+v %v", u, err)
	}
	// By pid (the shortcut's front), and an app with no address.
	if u, err := request[proto.BrowserURLResult](t, h, proto.MethodBrowserURL, proto.BrowserURLParams{PID: 903}); err != nil || u.BundleID != "company.thebrowser.Browser" || !strings.HasPrefix(u.URL, "https://linear.app/") {
		t.Errorf("arc by pid: %+v %v", u, err)
	}
	if u, err := request[proto.BrowserURLResult](t, h, proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: "org.mozilla.firefox"}); err != nil || u.URL != "" || u.Via != proto.URLViaNone || u.Error == "" {
		t.Errorf("firefox: %+v %v", u, err)
	}
	apps, err := request[proto.AppListResult](t, h, proto.MethodAppList, nil)
	if err != nil || len(apps.Apps) != 5 || apps.Apps[0].Name != "Arc" || !apps.Apps[0].Running {
		t.Errorf("app.list %+v %v", apps, err)
	}
	all, err := request[proto.AppListResult](t, h, proto.MethodAppList, proto.AppListParams{Installed: true})
	if err != nil || len(all.Apps) != 8 {
		t.Errorf("app.list installed %+v %v", all, err)
	}
	icons, err := request[proto.AppIconsResult](t, h, proto.MethodAppIcons, proto.AppIconsParams{
		Apps: []proto.AppRef{{BundleID: "com.apple.Notes", Path: "/System/Applications/Notes.app"}, {BundleID: "com.example.gone"}}, Size: 32})
	if err != nil || len(icons.Icons) != 2 || !bytes.HasPrefix(icons.Icons[0].PNG, []byte("\x89PNG")) || icons.Icons[1].PNG != nil {
		t.Errorf("app.icons %+v %v", icons, err)
	}
	front, err := request[proto.App](t, h, proto.MethodAppFrontmost, nil)
	if err != nil || front.BundleID != "com.tinyspeck.slackmacgap" || front.PID != 812 {
		t.Errorf("front %+v %v", front, err)
	}
	// The fake's log line and stderr reach the log, as source "helper".
	wait(t, "log lines", func() bool {
		h.mu.Lock()
		defer h.mu.Unlock()
		return slices.ContainsFunc(h.logs, func(e core.LogEntry) bool { return e.Message == "fake helper ready" && e.Source == LogSource }) &&
			slices.ContainsFunc(h.logs, func(e core.LogEntry) bool { return e.Message == "fake helper starting" })
	})
}

func TestPermissions(t *testing.T) {
	h := start(t, "mic=notDetermined,untrusted")
	h.ready(t)
	st, err := request[proto.PermissionsResult](t, h, proto.MethodPermissionsStatus, nil)
	if err != nil || st.Microphone != proto.PermNotDetermined || st.Accessibility {
		t.Fatalf("status %+v %v", st, err)
	}
	st, err = request[proto.PermissionsResult](t, h, proto.MethodPermissionsRequest, proto.PermissionsRequestParams{Permission: proto.PermissionMicrophone})
	if err != nil || st.Microphone != proto.PermGranted {
		t.Fatalf("request %+v %v", st, err)
	}
	if _, err := request[proto.PermissionsResult](t, h, proto.MethodPermissionsRequest, proto.PermissionsRequestParams{Permission: "camera"}); !errors.Is(err, &proto.Error{Code: proto.CodeBadRequest}) {
		t.Errorf("camera: %v", err)
	}
	// Without Accessibility, text I/O is refused.
	_, err = request[proto.PasteResult](t, h, proto.MethodTextPaste, proto.PasteParams{Text: "x"})
	var perr *proto.Error
	if !errors.As(err, &perr) || perr.Code != proto.CodePermissionDenied || !strings.Contains(string(perr.Data), "accessibility") {
		t.Errorf("paste untrusted: %v", err)
	}
	d := start(t, "deny-mic")
	d.ready(t)
	_, err = request[proto.CaptureResult](t, d, proto.MethodCaptureStart, proto.CaptureParams{})
	if !errors.As(err, &perr) || perr.Code != proto.CodePermissionDenied || !strings.Contains(string(perr.Data), "microphone") {
		t.Errorf("capture denied: %v", err)
	}
}

// capture runs a capture.start job on a goroutine, collecting its
// notifications.
type captureRun struct {
	mu      sync.Mutex
	started *proto.CaptureStartedParams
	levels  int
	seqs    []int
	pcm     []byte
	done    chan struct{}
	res     proto.CaptureResult
	err     error
}

func (c *captureRun) frames() int { c.mu.Lock(); defer c.mu.Unlock(); return len(c.seqs) }

func startCapture(ctx context.Context, h *harness, id string, p proto.CaptureParams) *captureRun {
	c := &captureRun{done: make(chan struct{})}
	go func() {
		defer close(c.done)
		raw, err := h.sup.Call(ctx, id, proto.MethodCaptureStart, p, func(m proto.Message) {
			c.mu.Lock()
			defer c.mu.Unlock()
			switch m.Method {
			case proto.MethodCaptureStarted:
				c.started = new(proto.CaptureStartedParams)
				m.DecodeParams(c.started)
			case proto.MethodCaptureLevel:
				c.levels++
			case proto.MethodCapturePCM:
				var f proto.PCMParams
				m.DecodeParams(&f)
				if f.Offset != len(c.pcm)/2 {
					c.err = errors.New("pcm offset gap")
				}
				c.seqs = append(c.seqs, f.Seq)
				c.pcm = append(c.pcm, f.Data...)
			}
		})
		if err == nil {
			err = json.Unmarshal(raw, &c.res)
		}
		c.mu.Lock()
		if c.err == nil {
			c.err = err
		}
		c.mu.Unlock()
	}()
	return c
}

func TestCaptureStreamsPCM(t *testing.T) {
	h := start(t, "frame-ms=10")
	h.ready(t)
	c := startCapture(context.Background(), h, "S1", proto.CaptureParams{Devices: []string{"Unplugged", "AirPodsPro:input"}})
	wait(t, "five frames", func() bool { return c.frames() >= 5 })
	if st := h.sup.Status(); !slices.Equal(st.Running, []string{"S1"}) {
		t.Errorf("running %v", st.Running)
	}
	// A second recording is refused while one runs.
	if _, err := request[proto.CaptureResult](t, h, proto.MethodCaptureStart, proto.CaptureParams{}); !errors.Is(err, &proto.Error{Code: proto.CodeBusy}) {
		t.Errorf("second capture: %v", err)
	}
	stop, err := request[proto.CaptureStopResult](t, h, proto.MethodCaptureStop, proto.CaptureStopParams{Job: "S1"})
	if err != nil || !stop.Stopped {
		t.Fatalf("stop %+v %v", stop, err)
	}
	<-c.done
	if c.err != nil || c.res.Reason != proto.CaptureStopped || c.res.Frames < 5 || c.res.Device != "AirPodsPro:input" {
		t.Fatalf("capture %+v %v", c.res, c.err)
	}
	if c.started == nil || !c.started.FellBack || c.started.Device != "AirPodsPro:input" || c.started.SampleRate != 16000 {
		t.Errorf("started %+v", c.started)
	}
	for i, s := range c.seqs {
		if s != i {
			t.Fatalf("seq %v", c.seqs)
		}
	}
	if c.levels != len(c.seqs) || len(c.pcm) != c.res.Frames*1600*2 || c.res.Samples*2 != len(c.pcm) {
		t.Errorf("%d levels, %d bytes for %d frames (%d samples)", c.levels, len(c.pcm), c.res.Frames, c.res.Samples)
	}
	// Stopped already: nothing to stop.
	if stop, _ := request[proto.CaptureStopResult](t, h, proto.MethodCaptureStop, proto.CaptureStopParams{Job: "S1"}); stop.Stopped {
		t.Error("stopped twice")
	}
	// The chosen input goes away: the recording ends with what it has.
	c = startCapture(context.Background(), h, "S1b", proto.CaptureParams{Devices: []string{"lost"}})
	<-c.done
	if c.err != nil || c.res.Reason != proto.CaptureDeviceLost || c.res.Frames != 3 {
		t.Errorf("lost: %+v %v", c.res, c.err)
	}
}

// TestCapturePlaysRecording: pcm= replays a WAV, resampled to the asked
// rate, then silence.
func TestCapturePlaysRecording(t *testing.T) {
	pcm := make([]byte, 2*8000) // 0.25 s at 32 kHz: a ramp
	for i := range 8000 {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(i)))
	}
	path := filepath.Join(t.TempDir(), "say.wav")
	os.WriteFile(path, helpertest.WAV(pcm, 32000), 0o644)
	h := start(t, "frame-ms=5,"+helpertest.Opt("pcm", path))
	h.ready(t)
	c := startCapture(context.Background(), h, "S9", proto.CaptureParams{FrameMS: 50})
	wait(t, "seven frames", func() bool { return c.frames() >= 7 })
	request[proto.CaptureStopResult](t, h, proto.MethodCaptureStop, proto.CaptureStopParams{Job: "S9"})
	<-c.done
	if c.err != nil || len(c.pcm) < 4000*2 {
		t.Fatalf("%v, %d bytes", c.err, len(c.pcm))
	}
	// 16 kHz: every other sample of the ramp, then zeros.
	for i := range 4000 {
		if v := int16(binary.LittleEndian.Uint16(c.pcm[2*i:])); v != int16(2*i) {
			t.Fatalf("sample %d = %d", i, v)
		}
	}
	if v := binary.LittleEndian.Uint16(c.pcm[2*4000+10:]); v != 0 {
		t.Errorf("after the recording: %d", v)
	}
}

func TestCancelCapture(t *testing.T) {
	h := start(t, "frame-ms=10")
	h.ready(t)
	ctx, cancel := context.WithCancel(context.Background())
	c := startCapture(ctx, h, "S2", proto.CaptureParams{})
	wait(t, "a frame", func() bool { return c.frames() >= 1 })
	cancel()
	<-c.done
	if !errors.Is(c.err, &proto.Error{Code: proto.CodeCancelled}) {
		t.Fatalf("cancelled capture: %v", c.err)
	}
	if h.sup.Status().State != Ready {
		t.Error("the helper did not stay")
	}
}

func TestStuckCancelRestarts(t *testing.T) {
	h := start(t, "hang-cancel,frame-ms=10")
	first := h.ready(t).PID
	ctx, cancel := context.WithCancel(context.Background())
	c := startCapture(ctx, h, "S3", proto.CaptureParams{})
	wait(t, "the capture", func() bool { return len(h.sup.Status().Running) == 1 })
	cancel()
	<-c.done
	if !errors.Is(c.err, ErrExited) {
		t.Fatalf("stuck capture: %v", c.err)
	}
	wait(t, "a new helper", func() bool { st := h.sup.Status(); return st.State == Ready && st.PID != first })
	if st := h.sup.Status(); st.Restarts != 1 {
		t.Errorf("restarts %d", st.Restarts)
	}
}

var fnTap = []proto.HotkeyBinding{
	{ID: "transcribe/0", Modifiers: map[string]string{proto.ModFn: proto.SideAny}, Behavior: proto.BehaviorTap},
	{ID: "translate/0", Modifiers: map[string]string{proto.ModFn: proto.SideAny, proto.ModShift: proto.SideAny}, Behavior: proto.BehaviorTap},
}

func (h *harness) notes1() []proto.Message {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.notes)
}

func TestHotkeyEventsGoToOnNotify(t *testing.T) {
	h := start(t, "press=transcribe/0,front=com.google.Chrome")
	h.ready(t)
	res, err := request[proto.HotkeyStatus](t, h, proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{Bindings: fnTap})
	if err != nil || !res.Installed || !res.Accessibility || res.Bindings != 2 {
		t.Fatalf("configure %+v %v", res, err)
	}
	wait(t, "the event", func() bool { return len(h.notes1()) == 1 })
	var e proto.HotkeyEventParams
	n := h.notes1()[0]
	n.DecodeParams(&e)
	if n.Method != proto.MethodHotkeyEvent || e.Binding != "transcribe/0" || e.Phase != proto.PhaseTrigger || e.Front == nil || e.Front.BundleID != "com.google.Chrome" {
		t.Errorf("event %s %+v", n.Method, e)
	}

	// No Accessibility: not installed, until it is granted.
	u := start(t, "untrusted,trust-after=300")
	u.ready(t)
	res, err = request[proto.HotkeyStatus](t, u, proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{Bindings: fnTap[:1]})
	if err != nil || res.Installed || res.Accessibility || res.Error == "" {
		t.Errorf("untrusted %+v %v", res, err)
	}
	wait(t, "hotkey.state", func() bool { return len(u.notes1()) == 1 })
	var st proto.HotkeyStatus
	u.notes1()[0].DecodeParams(&st)
	if u.notes1()[0].Method != proto.MethodHotkeyState || !st.Installed || !st.Accessibility {
		t.Errorf("state %+v", st)
	}
	if res, _ := request[proto.HotkeyStatus](t, u, proto.MethodHotkeyStatus, nil); !res.Installed {
		t.Errorf("status %+v", res)
	}
}

// TestEventScript: events= plays notifications of no request in order,
// with their delays.
func TestEventScript(t *testing.T) {
	script := filepath.Join(t.TempDir(), "events.jsonl")
	os.WriteFile(script, []byte(`{"method":"app.activated","params":{"pid":901,"bundleId":"com.google.Chrome","name":"Google Chrome"}}
{"afterMs":50,"method":"hotkey.event","params":{"binding":"transcribe/1","phase":"down","behavior":"longPress","heldMs":300,"t":10}}
{"afterMs":100,"method":"hotkey.event","params":{"binding":"transcribe/1","phase":"up","behavior":"longPress","heldMs":400,"t":110}}
`), 0o644)
	h := start(t, helpertest.Opt("events", script))
	h.ready(t)
	t0 := time.Now()
	request[proto.HotkeyStatus](t, h, proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{Bindings: fnTap})
	wait(t, "three notifications", func() bool { return len(h.notes1()) == 3 })
	if d := time.Since(t0); d < 150*time.Millisecond {
		t.Errorf("played in %s", d)
	}
	var got []string
	for _, m := range h.notes1() {
		var e proto.HotkeyEventParams
		m.DecodeParams(&e)
		got = append(got, m.Method+":"+e.Phase)
	}
	if !slices.Equal(got, []string{"app.activated:", "hotkey.event:down", "hotkey.event:up"}) {
		t.Errorf("notes %v", got)
	}
}

func TestTextIO(t *testing.T) {
	log := filepath.Join(t.TempDir(), "paste.jsonl")
	h := start(t, helpertest.Opt("selection", "Hello, world")+","+helpertest.Opt("paste-log", log))
	h.ready(t)
	sel, err := request[proto.SelectionResult](t, h, proto.MethodTextSelection, proto.SelectionParams{})
	if err != nil || sel.Text != "Hello, world" || sel.Via != proto.ViaAX {
		t.Errorf("selection %+v %v", sel, err)
	}
	focus, err := request[proto.FocusResult](t, h, proto.MethodTextFocus, nil)
	if err != nil || !focus.Editable || focus.PID != 812 || focus.Selection == nil || focus.Selection.Length != 12 {
		t.Errorf("focus %+v %v", focus, err)
	}
	r, err := request[proto.PasteResult](t, h, proto.MethodTextPaste, proto.PasteParams{Text: "你好", TargetPID: 812, Press: []proto.KeyPress{{Key: 36}}})
	if err != nil || !r.Posted || !r.Restored {
		t.Errorf("paste %+v %v", r, err)
	}
	// The target is not in front: nothing is pasted.
	r, err = request[proto.PasteResult](t, h, proto.MethodTextPaste, proto.PasteParams{Text: "lost", TargetPID: 1})
	if err != nil || r.Posted || r.Front == nil || r.Front.PID != 812 {
		t.Errorf("paste elsewhere %+v %v", r, err)
	}
	data, _ := os.ReadFile(log)
	if lines := strings.Split(strings.TrimSpace(string(data)), "\n"); len(lines) != 1 || !strings.Contains(lines[0], `"text":"你好"`) || !strings.Contains(lines[0], `"press":[{"key":36}]`) {
		t.Errorf("paste log %q", data)
	}
	// ⌘C only when allowed.
	c := start(t, helpertest.Opt("copy-selection", "line"))
	c.ready(t)
	if sel, _ := request[proto.SelectionResult](t, c, proto.MethodTextSelection, proto.SelectionParams{}); sel.Text != "" || sel.Via != proto.ViaNone {
		t.Errorf("no copy %+v", sel)
	}
	if sel, _ := request[proto.SelectionResult](t, c, proto.MethodTextSelection, proto.SelectionParams{AllowCopy: true}); sel.Text != "line" || sel.Via != proto.ViaCopy {
		t.Errorf("copy %+v", sel)
	}
}

// TestTrace: every call reports its phases; a slow control request is
// logged with them.
func TestTrace(t *testing.T) {
	var mu sync.Mutex
	var traces []Trace
	h := start(t, "delay=app.frontmost:250", func(c *Config) {
		c.SlowCall = 200 * time.Millisecond
		c.OnTrace = func(tr Trace) { mu.Lock(); traces = append(traces, tr); mu.Unlock() }
	})
	h.ready(t)
	request[proto.DevicesResult](t, h, proto.MethodCaptureDevices, nil)
	request[proto.App](t, h, proto.MethodAppFrontmost, nil)
	mu.Lock()
	defer mu.Unlock()
	if len(traces) != 2 {
		t.Fatalf("traces %+v", traces)
	}
	for _, tr := range traces {
		if tr.ID == "" || tr.Written < tr.Ready || tr.Answered < tr.Written || tr.Returned < tr.Answered || tr.Err != "" {
			t.Errorf("phases out of order: %+v", tr)
		}
	}
	slow := traces[1]
	if slow.Method != proto.MethodAppFrontmost || slow.Answered-slow.Written < 250*time.Millisecond {
		t.Errorf("slow %+v", slow)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !slices.ContainsFunc(h.logs, func(e core.LogEntry) bool {
		return e.Level == core.SeverityWarning && strings.HasPrefix(e.Message, "slow call: app.frontmost") && strings.Contains(e.Message, "helper 2")
	}) {
		t.Errorf("no slow call in the log: %+v", h.logs)
	}
	if slices.ContainsFunc(h.logs, func(e core.LogEntry) bool { return strings.Contains(e.Message, "capture.devices") }) {
		t.Error("a fast call was logged")
	}
}

// TestTraceAnsweredBeforeWritten: the reader can read a response before
// the caller notes that its write returned (CI saw "helper -30µs"); the
// helper's phase is then 0, not negative.
func TestTraceAnsweredBeforeWritten(t *testing.T) {
	start := time.Now()
	tr := Trace{Start: start, Written: 120 * time.Microsecond}
	tr.answered(&call{answered: start.Add(90 * time.Microsecond)})
	if tr.Answered != tr.Written {
		t.Errorf("answered %v, written %v", tr.Answered, tr.Written)
	}
	tr.answered(&call{answered: start.Add(time.Millisecond)})
	if tr.Answered != time.Millisecond {
		t.Errorf("answered %v", tr.Answered)
	}
}

func TestCrashRestartsAndFailsCalls(t *testing.T) {
	h := start(t, "frame-ms=10")
	first := h.ready(t).PID
	_, err := h.sup.Call(context.Background(), "S4", proto.MethodCaptureStart, proto.CaptureParams{Devices: []string{"crash"}}, nil)
	if !errors.Is(err, ErrExited) || !errors.Is(err, core.ErrHelper) {
		t.Fatalf("crash: %v", err)
	}
	wait(t, "the failure", func() bool {
		h.mu.Lock()
		defer h.mu.Unlock()
		return slices.ContainsFunc(h.statuses, func(s Status) bool { return s.State == Failed })
	})
	h.mu.Lock()
	var failed *Status
	for i := range h.statuses {
		if h.statuses[i].State == Failed {
			failed = &h.statuses[i]
		}
	}
	h.mu.Unlock()
	if failed == nil || failed.Reason != FailCrashed || !strings.Contains(failed.StderrTail, "the audio engine stopped") || failed.RestartAt.IsZero() {
		t.Fatalf("failed status %+v", failed)
	}
	// Calls wait for the restart.
	if _, err := request[proto.DevicesResult](t, h, proto.MethodCaptureDevices, nil); err != nil {
		t.Fatalf("after the restart: %v", err)
	}
	if st := h.sup.Status(); st.PID == first || st.Restarts != 1 {
		t.Errorf("status %+v", st)
	}
	logs, _ := os.ReadFile(filepath.Join(filepathDir(t, h), "helper.log"))
	if !strings.Contains(string(logs), "the audio engine stopped") {
		t.Errorf("helper.log: %q", logs)
	}
}

// filepathDir is the LogDir the harness gave the supervisor.
func filepathDir(t *testing.T, h *harness) string {
	return h.sup.(*supervisor).cfg.LogDir
}

func TestFailures(t *testing.T) {
	for _, tc := range []struct {
		opts   string
		reason FailReason
		retry  bool
	}{
		{"mismatch", FailProtocol, false},
		{"refuse", FailProtocol, false},
		{"silent", FailHandshake, true},
		{"crash-start", FailCrashed, true},
	} {
		t.Run(tc.opts, func(t *testing.T) {
			h := start(t, tc.opts, func(c *Config) { c.HelloTimeout = 300 * time.Millisecond; c.MaxRestarts = 1 })
			wait(t, "the failure", func() bool { return h.sup.Status().State == Failed })
			st := h.sup.Status()
			if st.Reason != tc.reason || st.RestartAt.IsZero() != !tc.retry {
				t.Fatalf("status %+v", st)
			}
			if !tc.retry {
				if _, err := h.sup.Call(context.Background(), "", proto.MethodCaptureDevices, nil, nil); !errors.Is(err, core.ErrHelper) {
					t.Errorf("call to a failed helper: %v", err)
				}
			}
		})
	}
	// No executable.
	h := start(t, "", func(c *Config) { c.Bin = "" })
	if st := h.sup.Status(); st.State != Failed || st.Reason != FailMissing {
		t.Fatalf("missing: %+v", st)
	}
}

func TestRestartAndStop(t *testing.T) {
	h := start(t, "")
	first := h.ready(t).PID
	if err := h.sup.Restart(); err != nil {
		t.Fatal(err)
	}
	st := h.ready(t)
	if st.PID == first || st.Restarts != 0 {
		t.Errorf("restart %+v", st)
	}
	h.sup.Stop()
	if st := h.sup.Status(); st.State != Stopped || st.PID != 0 {
		t.Errorf("stop %+v", st)
	}
	if _, err := h.sup.Call(context.Background(), "", proto.MethodCaptureDevices, nil, nil); !errors.Is(err, core.ErrHelper) {
		t.Errorf("call after stop: %v", err)
	}
}

// TestReplayFixtures runs every fixture's requests through the
// supervisor against the fake replaying it, and checks the answers match.
func TestReplayFixtures(t *testing.T) {
	paths, _ := filepath.Glob("../../helper/Fixtures/*.jsonl")
	if len(paths) == 0 {
		t.Fatal("no fixtures")
	}
	for _, path := range []string{"../../helper/Fixtures/browser.jsonl", "../../helper/Fixtures/hotkey.jsonl"} {
		t.Run(filepath.Base(path), func(t *testing.T) {
			abs, _ := filepath.Abs(path)
			h := start(t, "replay="+abs)
			h.ready(t)
			u, err := request[proto.BrowserURLResult](t, h, proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: "com.google.Chrome"})
			if strings.HasSuffix(path, "browser.jsonl") {
				if err != nil || u.URL != "https://mail.google.com/mail/u/0/#inbox" || u.Via != proto.URLViaAXDocument {
					t.Errorf("replayed url %+v %v", u, err)
				}
				return
			}
			if !errors.Is(err, &proto.Error{Code: proto.CodeMethodNotFound}) {
				t.Errorf("not in the fixture: %v", err)
			}
			res, err := request[proto.HotkeyStatus](t, h, proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{Bindings: fnTap})
			if err != nil || res.Bindings != 6 || !res.Installed {
				t.Errorf("replayed configure %+v %v", res, err)
			}
			wait(t, "the replayed event", func() bool { return len(h.notes1()) == 1 })
		})
	}
}

func TestFind(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("VOXTNEW_HELPER", "")
	os.Unsetenv("VOXTNEW_HELPER")
	if Find(dir) != "" {
		t.Error("found nothing as something")
	}
	bin := filepath.Join(dir, "helper", BinName)
	os.MkdirAll(filepath.Dir(bin), 0o755)
	os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755)
	if Find(dir) != bin {
		t.Errorf("Find = %q", Find(dir))
	}
	t.Setenv("VOXTNEW_HELPER", "/opt/x")
	if Find(dir) != "/opt/x" {
		t.Error("the environment does not win")
	}
}

// TestUpdatedApp: once an update replaced the app on disk (Config.Stale),
// the supervisor starts no helper, the new version's, neither after a
// crash nor on Restart, and a helper speaking another protocol is put
// down to the update (FailUpdated) rather than a build to redo.
func TestUpdatedApp(t *testing.T) {
	var stale atomicBool
	t.Run("crash", func(t *testing.T) {
		stale.set(false)
		h := start(t, "", func(c *Config) { c.Stale = stale.get })
		first := h.ready(t).PID
		stale.set(true) // the update is installed while it runs
		if p, err := os.FindProcess(first); err == nil {
			p.Kill()
		}
		wait(t, "the update to be reported", func() bool { return h.sup.Status().Reason == FailUpdated })
		st := h.sup.Status()
		if st.State != Failed || st.PID != 0 || !st.RestartAt.IsZero() || !strings.Contains(st.Err, "restart it") {
			t.Fatalf("after the crash: %+v", st)
		}
		// Restart starts nothing either.
		h.sup.Restart()
		if st := h.sup.Status(); st.State != Failed || st.Reason != FailUpdated || st.PID != 0 {
			t.Fatalf("after Restart: %+v", st)
		}
		if _, err := h.sup.Call(context.Background(), "", proto.MethodCaptureDevices, nil, nil); !errors.Is(err, core.ErrHelper) {
			t.Errorf("call: %v", err)
		}
		// Each restart asked Stale and gave up: no helper ran since.
		h.mu.Lock()
		defer h.mu.Unlock()
		for _, s := range h.statuses[1:] {
			if s.State == Ready && s.PID != first {
				t.Errorf("a helper started after the update: %+v", s)
			}
		}
	})
	for _, opts := range []string{"mismatch", "refuse"} {
		t.Run(opts, func(t *testing.T) {
			calls := 0
			var mu sync.Mutex
			h := start(t, opts, func(c *Config) {
				// Not stale when it starts, stale by its hello: an
				// update came in between.
				c.Stale = func() bool { mu.Lock(); defer mu.Unlock(); calls++; return calls > 1 }
				c.HelloTimeout = 300 * time.Millisecond
			})
			wait(t, "the failure", func() bool { return h.sup.Status().State == Failed })
			if st := h.sup.Status(); st.Reason != FailUpdated || !st.RestartAt.IsZero() {
				t.Fatalf("status %+v", st)
			}
		})
	}
}

// atomicBool is a bool for Config.Stale, set from the test.
type atomicBool struct {
	mu sync.Mutex
	v  bool
}

func (b *atomicBool) set(v bool) { b.mu.Lock(); b.v = v; b.mu.Unlock() }
func (b *atomicBool) get() bool  { b.mu.Lock(); defer b.mu.Unlock(); return b.v }
