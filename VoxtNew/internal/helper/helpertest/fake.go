// Package helpertest is a scripted voxtnew-helper for tests: a test
// binary runs it instead of its tests when VOXTNEW_FAKE_HELPER is set,
// so the supervisor spawns a real process and talks to it over real
// pipes, without macOS. In the test package:
//
//	func TestMain(m *testing.M) {
//		if helpertest.Enabled() {
//			helpertest.Main()
//		}
//		os.Exit(m.Run())
//	}
//
// and point helper.Config at os.Executable() with Env: helpertest.Env(opts).
// cmd/fakehelper builds the same fake as a program of its own, to try the
// app without the Swift helper (VOXTNEW_HELPER=path/to/fakehelper).
//
// The fake serves every method of package proto, so whole sessions run on
// Linux: capture.start plays a 440 Hz tone, or a recording (pcm=), in real
// time as capture.pcm frames and capture.level; hotkey.configure installs
// a pretend tap and can play a script of notifications (events=, press=);
// textio.* read a scripted selection and log what they paste (paste-log=);
// browser.url answers from Browsers, app.list from Apps and Installed,
// app.icons with Icon for those and Browsers;
// app.configure keeps its list (no-tree=); output.* mute pretend speakers
// and AirPods (output.go).
// (From AudioSmith's engine/enginetest, with the helper's methods.)
package helpertest

//
// VOXTNEW_FAKE_HELPER is a comma-separated list of options; values are
// URL-query escaped (a comma is %2C):
//
//	(empty)          speak the protocol (the default)
//	mismatch         answer helper.hello with protocol 9
//	refuse           answer helper.hello with protocol_mismatch
//	silent           never answer helper.hello
//	crash-start      exit 3 at once
//	hang-cancel      ignore job.cancel (the capture keeps running)
//	mic=STATE        the Microphone: granted (default), denied, notDetermined, restricted;
//	                 notDetermined becomes granted on permissions.request or capture.start
//	deny-mic         mic=denied
//	untrusted        no Accessibility: the tap is not installed, textio fails
//	trust-after=MS   with untrusted: Accessibility comes MS ms after the start
//	                 (then the tap installs and hotkey.state is sent)
//	frame-ms=N       a capture.pcm frame (and level) every N ms of wall time (default 100)
//	pcm=PATH         capture.pcm plays this WAV (16-bit mono, any rate) or raw
//	                 PCM16 at the requested rate, then silence
//	events=PATH      once the tap is installed, play a script: JSON lines
//	                 {"afterMs":N,"method":"hotkey.event","params":{…}}, any notification;
//	                 an app.activated line also brings its app to the front (the user moved on)
//	press=ID         once the tap is installed, binding ID fires (a trigger, with Front)
//	front=BUNDLE     the app in front (default com.tinyspeck.slackmacgap, Slack, pid 812)
//	selection=TEXT   textio.selection reads TEXT through Accessibility
//	copy-selection=TEXT  textio.selection reads TEXT only with allowCopy (via copy)
//	readonly[=BUNDLE]  textio.focus: the focused element is not editable (of BUNDLE only)
//	focus=none       textio.focus: nothing is focused (Finder's desktop)
//	focus=secure     textio.focus: a password field (AXSecureTextField)
//	switch=BUNDLE    textio.* see BUNDLE in front: the user went there after the shortcut
//	paste-check=C    textio.paste with verifyMs: its check is C (unchanged, unknown; default inserted)
//	secure-input=APP textio.paste: APP has secure event input on
//	paste-log=PATH   append each textio.paste's params to PATH (JSON lines)
//	delay=METHOD:MS  answer METHOD MS ms late (slow calls)
//	no-url[=BUNDLE]  browser.url reads no URL (of BUNDLE, or of every browser)
//	no-tree=BUNDLE   browser.url reads BUNDLE's URL only once app.configure listed
//	                 it (Chromium builds no tree without AXManualAccessibility)
//	replay=PATH      replay a fixture transcript (helper/Fixtures/*.jsonl)
//	out=PATH         the output devices (output.*, output.go) kept in PATH, outliving
//	                 the process: speakers (the default) and AirPods at first
//	out-default=X    the default output starts muted (the user's), fixed (a mute that
//	                 cannot be set), none (no mute) or empty (no output device)
//	out-switch=MS    MS ms after a mute, AirPods become the default output
//	out-unmute=MS    MS ms after a mute, the user unmutes the device
//	crash-mute       exit 3 right after a mute (its claim stays in the journal)
//	no-output        an older helper: no output.* in hello, method_not_found
//
// A capture.start whose first device is "crash" exits 3 after its first
// frame; one whose first device is "gone" fails with device_unavailable;
// "lost" ends the recording with deviceLost after its third frame.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

const fakeEnv = "VOXTNEW_FAKE_HELPER"

// Enabled reports whether this process should be the fake helper.
func Enabled() bool {
	_, ok := os.LookupEnv(fakeEnv)
	return ok
}

// Env returns the environment that makes a re-executed test binary the
// fake helper with opts (see the comment at the top of the file).
func Env(opts string) []string { return []string{fakeEnv + "=" + opts} }

// Opt escapes a value for an option (texts with commas).
func Opt(key, value string) string { return key + "=" + url.QueryEscape(value) }

// Main runs the fake helper with the options of the environment, and
// exits.
func Main() {
	Run(os.Getenv(fakeEnv))
	os.Exit(0)
}

// Methods is what the fake implements (hello.methods): every request.
func Methods() []string { return slices.Clone(proto.Requests) }

// oldMethods are an older helper's (no-output): without output.*.
func oldMethods() []string {
	return slices.DeleteFunc(Methods(), func(m string) bool { return strings.HasPrefix(m, "output.") })
}

// Devices are the fake's audio inputs.
func Devices() []proto.CaptureDevice {
	return []proto.CaptureDevice{
		{ID: "BuiltInMicrophoneDevice", Name: "MacBook Air Microphone", Channels: 1, SampleRate: 48000, Default: true, Transport: "builtin"},
		{ID: "AirPodsPro:input", Name: "AirPods Pro", Channels: 1, SampleRate: 24000, Transport: "bluetooth"},
	}
}

// Apps are the apps the fake knows, by bundle id (front=).
var Apps = map[string]proto.App{
	"com.tinyspeck.slackmacgap":  {PID: 812, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"},
	"com.google.Chrome":          {PID: 901, BundleID: "com.google.Chrome", Name: "Google Chrome"},
	"com.apple.Safari":           {PID: 902, BundleID: "com.apple.Safari", Name: "Safari"},
	"company.thebrowser.Browser": {PID: 903, BundleID: "company.thebrowser.Browser", Name: "Arc"},
	"com.apple.TextEdit":         {PID: 530, BundleID: "com.apple.TextEdit", Name: "TextEdit"},
}

// Browsers are the URLs browser.url answers, by bundle id; other apps
// have none (via "none").
var Browsers = map[string]string{
	"com.apple.Safari":           "https://developer.apple.com/documentation/",
	"com.google.Chrome":          "https://mail.google.com/mail/u/0/#inbox",
	"com.microsoft.edgemac":      "https://github.com/sgl-project/sglang-omni/VoxtNew/pulls",
	"com.brave.Browser":          "https://news.ycombinator.com/",
	"company.thebrowser.Browser": "https://linear.app/team/issue/AUR-12",
}

// Installed are apps the fake lists with app.list's installed, besides
// Apps (which run).
var Installed = []proto.AppInfo{
	{BundleID: "com.apple.Notes", Name: "Notes", Path: "/System/Applications/Notes.app"},
	{BundleID: "com.microsoft.edgemac", Name: "Microsoft Edge", Path: "/Applications/Microsoft Edge.app"},
	{BundleID: "com.apple.mail", Name: "Mail", Path: "/System/Applications/Mail.app"},
}

// Script is one line of an events= script.
type Script struct {
	AfterMS int             `json:"afterMs,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type fake struct {
	opts  map[string]string
	out   *bufio.Writer
	wmu   sync.Mutex
	start time.Time

	mu        sync.Mutex
	hello     bool
	mic       string
	trusted   bool
	bindings  int
	installed bool
	scripted  bool // the events= script and press= ran
	hkActive  bool
	manual    []string                 // app.configure's manualAccessibility
	running   map[string]bool          // capture jobs
	cancel    map[string]bool          // asked to stop by job.cancel
	stop      map[string]chan struct{} // closed by capture.stop
	finished  map[string]chan struct{}
	outSim    *OutputHelper // output.* (output.go)
	moved     *proto.App    // the app in front since a script's app.activated
}

// Run serves the protocol on stdin and stdout with opts (see the top of
// the file) until stdin closes or helper.shutdown.
func Run(optString string) {
	opts := map[string]string{}
	for _, o := range strings.Split(optString, ",") {
		k, v, _ := strings.Cut(strings.TrimSpace(o), "=")
		if k != "" {
			if u, err := url.QueryUnescape(v); err == nil {
				v = u
			}
			opts[k] = v
		}
	}
	f := &fake{opts: opts, out: bufio.NewWriter(os.Stdout), start: time.Now(), mic: proto.PermGranted, trusted: true,
		running: map[string]bool{}, cancel: map[string]bool{},
		stop: map[string]chan struct{}{}, finished: map[string]chan struct{}{}}
	if v, ok := opts["mic"]; ok {
		f.mic = v
	}
	if f.has("deny-mic") {
		f.mic = proto.PermDenied
	}
	if f.has("untrusted") {
		f.trusted = false
		if ms, err := strconv.Atoi(opts["trust-after"]); err == nil {
			time.AfterFunc(time.Duration(ms)*time.Millisecond, f.trust)
		}
	}
	fmt.Fprintln(os.Stderr, "fake helper starting")
	if f.has("crash-start") {
		fmt.Fprintln(os.Stderr, "fatal: could not start")
		os.Exit(3)
	}
	if path, ok := opts["replay"]; ok {
		f.replay(path)
		return
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 64*1024), proto.MaxLine)
	for sc.Scan() {
		m, err := proto.Decode(sc.Bytes())
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad line:", err)
			continue
		}
		f.handle(m)
	}
	f.mu.Lock()
	for id := range f.running {
		f.cancel[id] = true
	}
	f.mu.Unlock()
	f.outputExit() // the app went away: its mutes are undone
}

func (f *fake) has(opt string) bool { _, ok := f.opts[opt]; return ok }

func (f *fake) send(m proto.Message) {
	line, _ := proto.Encode(m)
	f.wmu.Lock()
	defer f.wmu.Unlock()
	f.out.Write(line)
	f.out.Flush()
}

func (f *fake) writeRaw(line string) {
	f.wmu.Lock()
	defer f.wmu.Unlock()
	f.out.WriteString(line + "\n")
	f.out.Flush()
}

func (f *fake) result(id string, v any) {
	m, _ := proto.NewResult(id, v)
	f.send(m)
}

func (f *fake) fail(id, code, format string, args ...any) {
	f.send(proto.NewError(id, proto.Errorf(code, format, args...)))
}

func (f *fake) failData(id, code string, data any, format string, args ...any) {
	e := proto.Errorf(code, format, args...)
	e.Data, _ = json.Marshal(data)
	f.send(proto.NewError(id, e))
}

func (f *fake) notify(method string, params any) {
	m, _ := proto.NewNotification(method, params)
	f.send(m)
}

// now is the fake's monotonic clock, in ms (hotkey.event's t).
func (f *fake) now() float64 { return float64(time.Since(f.start).Microseconds())/1000 + 1 }

func (f *fake) front() proto.App {
	f.mu.Lock()
	moved := f.moved
	f.mu.Unlock()
	if moved != nil {
		return *moved
	}
	if a, ok := Apps[f.opts["front"]]; ok {
		return a
	}
	if id := f.opts["front"]; id != "" {
		return proto.App{PID: 700, BundleID: id, Name: id}
	}
	return Apps["com.tinyspeck.slackmacgap"]
}

func (f *fake) isTrusted() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.trusted
}

func (f *fake) handle(m proto.Message) {
	if method, ms, ok := strings.Cut(f.opts["delay"], ":"); ok && method == m.Method {
		d, _ := strconv.Atoi(ms)
		go func() {
			time.Sleep(time.Duration(d) * time.Millisecond)
			f.serve(m)
		}()
		return
	}
	f.serve(m)
}

func (f *fake) serve(m proto.Message) {
	if m.Method == proto.MethodHello {
		f.handleHello(m)
		return
	}
	f.mu.Lock()
	ready := f.hello
	f.mu.Unlock()
	if !ready {
		f.fail(m.ID, proto.CodeNotInitialized, "helper.hello must come first")
		return
	}
	switch m.Method {
	case proto.MethodShutdown:
		f.outputExit()
		f.result(m.ID, nil)
		os.Exit(0)
	case proto.MethodJobCancel:
		var p proto.CancelParams
		m.DecodeParams(&p)
		f.mu.Lock()
		running := f.running[p.Job]
		if running && !f.has("hang-cancel") {
			f.cancel[p.Job] = true
		}
		done := f.finished[p.Job]
		f.mu.Unlock()
		if !running {
			f.result(m.ID, proto.CancelResult{Cancelled: false})
			return
		}
		go func() {
			<-done
			f.result(m.ID, proto.CancelResult{Cancelled: true})
		}()
	case proto.MethodPermissionsStatus:
		f.result(m.ID, f.permissions())
	case proto.MethodPermissionsRequest:
		var p proto.PermissionsRequestParams
		m.DecodeParams(&p)
		switch p.Permission {
		case proto.PermissionMicrophone:
			f.mu.Lock()
			if f.mic == proto.PermNotDetermined {
				f.mic = proto.PermGranted
			}
			f.mu.Unlock()
		case proto.PermissionAccessibility:
		default:
			f.fail(m.ID, proto.CodeBadRequest, "Unknown permission %s", p.Permission)
			return
		}
		f.result(m.ID, f.permissions())
	case proto.MethodCaptureDevices:
		f.result(m.ID, proto.DevicesResult{Devices: Devices()})
	case proto.MethodCaptureStart:
		f.startCapture(m)
	case proto.MethodCaptureStop:
		f.captureStop(m)
	case proto.MethodHotkeyConfigure:
		f.hotkeys(m)
	case proto.MethodHotkeyStatus:
		f.result(m.ID, f.hotkeyStatus())
	case proto.MethodAppFrontmost:
		f.result(m.ID, f.front())
	case proto.MethodTextFocus, proto.MethodTextSelection, proto.MethodTextPaste:
		f.textio(m)
	case proto.MethodBrowserURL:
		var p proto.BrowserURLParams
		m.DecodeParams(&p)
		if p.PID != 0 {
			for _, app := range Apps {
				if app.PID == p.PID {
					p.BundleID = app.BundleID
				}
			}
		}
		if p.BundleID == "" {
			p.BundleID = f.front().BundleID
		}
		res := proto.BrowserURLResult{BundleID: p.BundleID, Via: proto.URLViaNone, MS: 1}
		u, ok := Browsers[p.BundleID]
		switch {
		case !ok:
			res.Error = "no AXDocument or web area URL (visited 12)"
		case f.has("no-url") && (f.opts["no-url"] == "" || f.opts["no-url"] == p.BundleID):
			res.Error = "no AXDocument or web area URL (visited 27)"
		case f.opts["no-tree"] == p.BundleID && !f.listed(p.BundleID):
			res.Error = "no AXDocument or web area URL (visited 400)"
		default:
			res.URL, res.Via = u, proto.URLViaAXDocument
		}
		f.result(m.ID, res)
	case proto.MethodAppConfigure:
		var p proto.AppConfigureParams
		m.DecodeParams(&p)
		f.mu.Lock()
		f.manual = p.ManualAccessibility
		f.mu.Unlock()
		f.result(m.ID, proto.Empty{})
	case proto.MethodAppList:
		var p proto.AppListParams
		m.DecodeParams(&p)
		var apps []proto.AppInfo
		for _, app := range Apps {
			apps = append(apps, proto.AppInfo{BundleID: app.BundleID, Name: app.Name, Running: true, PID: app.PID, Path: "/Applications/" + app.Name + ".app"})
		}
		if p.Installed {
			apps = append(apps, Installed...)
		}
		slices.SortFunc(apps, func(x, y proto.AppInfo) int { return strings.Compare(strings.ToLower(x.Name), strings.ToLower(y.Name)) })
		f.result(m.ID, proto.AppListResult{Apps: apps})
	case proto.MethodAppIcons:
		var p proto.AppIconsParams
		m.DecodeParams(&p)
		if len(p.Apps) > proto.MaxIcons {
			f.fail(m.ID, proto.CodeBadRequest, "At most %d apps a request", proto.MaxIcons)
			return
		}
		res := proto.AppIconsResult{Icons: []proto.AppIcon{}}
		for _, ref := range p.Apps {
			icon := proto.AppIcon{BundleID: ref.BundleID}
			if known(ref.BundleID) {
				icon.PNG = Icon(ref.BundleID, min(p.Size, proto.MaxIconSize))
			}
			res.Icons = append(res.Icons, icon)
		}
		f.result(m.ID, res)
	case proto.MethodOutputWatch, proto.MethodOutputMute, proto.MethodOutputRelease, proto.MethodOutputRecover:
		if f.has("no-output") {
			f.fail(m.ID, proto.CodeMethodNotFound, "Unknown method %s", m.Method)
			return
		}
		f.output(m)
	default:
		f.fail(m.ID, proto.CodeMethodNotFound, "Unknown method %s", m.Method)
	}
}

func (f *fake) permissions() proto.PermissionsResult {
	f.mu.Lock()
	defer f.mu.Unlock()
	return proto.PermissionsResult{Microphone: f.mic, Accessibility: f.trusted}
}

func (f *fake) handleHello(m proto.Message) {
	var p proto.HelloParams
	m.DecodeParams(&p)
	if f.has("silent") {
		return
	}
	if f.has("refuse") || p.Protocol != proto.Version {
		f.send(proto.NewError(m.ID, &proto.Error{Code: proto.CodeProtocolMismatch, Message: "This helper speaks protocol 1", Data: json.RawMessage(`{"supported":[1]}`)}))
		return
	}
	res := proto.HelloResult{Protocol: proto.Version, Version: "fake", PID: os.Getpid(), Methods: Methods(),
		Device: proto.Device{Name: "Fake Mac", OS: "test"}}
	if f.has("no-output") {
		res.Methods = oldMethods()
	}
	if f.has("mismatch") {
		res.Protocol = 9
	}
	f.mu.Lock()
	f.hello = true
	f.mu.Unlock()
	f.notify(proto.MethodLog, proto.LogParams{Level: "info", Message: "fake helper ready"})
	f.result(m.ID, res)
}
