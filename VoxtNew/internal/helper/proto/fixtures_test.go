package proto

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// fixturesDir holds the golden transcripts shared with the Swift helper.
const fixturesDir = "../../../helper/Fixtures"

// fixtureLine is one line of a fixture: a wire message and its direction.
type fixtureLine struct {
	To  string          `json:"to"` // "helper" (app → helper) or "app"
	Msg json.RawMessage `json:"msg"`
}

func readFixture(t *testing.T, path string) []fixtureLine {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var lines []fixtureLine
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(nil, MaxLine)
	for n := 1; sc.Scan(); n++ {
		if len(bytes.TrimSpace(sc.Bytes())) == 0 {
			continue
		}
		var l fixtureLine
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&l); err != nil {
			t.Fatalf("%s:%d: %v", filepath.Base(path), n, err)
		}
		if l.To != "helper" && l.To != "app" {
			t.Fatalf("%s:%d: to %q", filepath.Base(path), n, l.To)
		}
		lines = append(lines, l)
	}
	return lines
}

// strict decodes raw into v rejecting unknown fields, then checks that v
// encodes back to the same JSON: the Go types carry every field the
// fixture has, with the same omitempty behaviour.
func strict(t *testing.T, where string, raw json.RawMessage, v any) {
	t.Helper()
	if len(raw) == 0 {
		raw = json.RawMessage(`{}`)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		t.Fatalf("%s: %v\n%s", where, err, raw)
	}
	again, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if !jsonEqual(raw, again) {
		t.Fatalf("%s: does not round-trip\nwire: %s\ngo:   %s", where, raw, again)
	}
}

func jsonEqual(a, b []byte) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}

// TestFixtures decodes every golden transcript with the Go types and
// checks the protocol's sequencing rules on it (docs/helper-protocol.md).
func TestFixtures(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join(fixturesDir, "*.jsonl"))
	if err != nil || len(paths) < 9 {
		t.Fatalf("fixtures: %v %v", paths, err)
	}
	for _, path := range paths {
		name := filepath.Base(path)
		t.Run(name, func(t *testing.T) {
			pending := map[string]string{} // request id → method
			nextSeq, nextOffset := map[string]int{}, map[string]int{}
			started := map[string]bool{}
			for i, l := range readFixture(t, path) {
				where := fmt.Sprintf("%s:%d", name, i+1)
				m, err := Decode(l.Msg)
				if err != nil {
					t.Fatalf("%s: %v", where, err)
				}
				enc, err := Encode(m)
				if err != nil || !jsonEqual(l.Msg, bytes.TrimSuffix(enc, []byte("\n"))) {
					t.Fatalf("%s: envelope does not round-trip: %s", where, enc)
				}
				switch k := m.Kind(); {
				case l.To == "helper" && k == KindRequest:
					if _, dup := pending[m.ID]; dup {
						t.Fatalf("%s: id %s reused while pending", where, m.ID)
					}
					pending[m.ID] = m.Method
					if p := ParamsFor(m.Method); p != nil && slices.Contains(Requests, m.Method) {
						strict(t, where+" params", m.Params, p)
						if c, ok := p.(*HotkeyConfigureParams); ok {
							for _, b := range c.Bindings {
								if !validBinding(b) {
									t.Fatalf("%s: binding %+v", where, b)
								}
							}
						}
					}
				case l.To == "app" && k == KindNotification:
					p := ParamsFor(m.Method)
					if p == nil || !slices.Contains(Notifications, m.Method) {
						t.Fatalf("%s: unknown notification %s", where, m.Method)
					}
					strict(t, where+" params", m.Params, p)
					job := ""
					switch v := p.(type) {
					case *CaptureStartedParams:
						job = v.Job
						if v.Device == "" || v.SampleRate <= 0 || v.DeviceSampleRate <= 0 || v.DeviceChannels <= 0 || (v.FallbackReason != "") != v.FellBack {
							t.Fatalf("%s: started %+v", where, v)
						}
						started[job] = true
					case *LevelParams:
						job = v.Job
						if v.RMS < -120 || v.Peak < v.RMS || v.Peak > 0 {
							t.Fatalf("%s: level %+v", where, v)
						}
					case *PCMParams:
						job = v.Job
						if v.Seq != nextSeq[job] || v.Offset != nextOffset[job] || len(v.Data) == 0 || len(v.Data)%2 != 0 {
							t.Fatalf("%s: pcm %+v (want seq %d offset %d)", where, v, nextSeq[job], nextOffset[job])
						}
						nextSeq[job]++
						nextOffset[job] += len(v.Data) / 2
					case *RestartedParams:
						job = v.Job
						if !slices.Contains([]string{CaptureNoCallback, CaptureConfigurationChange, CaptureDeviceLost}, v.Reason) || v.Attempt < 1 {
							t.Fatalf("%s: restarted %+v", where, v)
						}
					case *HotkeyEventParams:
						if !validEvent(*v) {
							t.Fatalf("%s: hotkey event %+v", where, v)
						}
					case *App:
						if v.PID <= 0 {
							t.Fatalf("%s: app %+v", where, v)
						}
					case *OutputState:
						if !validOutput(*v) {
							t.Fatalf("%s: output %+v", where, v)
						}
					case *OutputReleasedParams:
						if v.Reason != OutputReasonSleep {
							t.Fatalf("%s: released %+v", where, v)
						}
					case *LogParams:
						job = v.Job
						if !slices.Contains([]string{"debug", "info", "warning", "error"}, v.Level) {
							t.Fatalf("%s: log level %q", where, v.Level)
						}
					}
					if job != "" {
						if pending[job] != MethodCaptureStart {
							t.Fatalf("%s: notification for %s, which is not a running capture", where, job)
						}
						if m.Method != MethodCaptureStarted && m.Method != MethodLog && !started[job] {
							t.Fatalf("%s: %s before capture.started", where, m.Method)
						}
					}
				case l.To == "app" && k == KindResponse:
					method, ok := pending[m.ID]
					if !ok {
						t.Fatalf("%s: response to unknown id %s", where, m.ID)
					}
					delete(pending, m.ID)
					if m.Error != nil {
						if !slices.Contains(Codes, m.Error.Code) || m.Error.Message == "" {
							t.Fatalf("%s: error %+v", where, m.Error)
						}
						if ParamsFor(method) == nil && m.Error.Code != CodeMethodNotFound {
							t.Fatalf("%s: %s must answer method_not_found", where, method)
						}
						if m.Error.Code == CodeCancelled && !IsJob(method) {
							t.Fatalf("%s: cancelled error for control request %s", where, method)
						}
						break
					}
					r := ResultFor(method)
					if r == nil {
						t.Fatalf("%s: result for %s, which has none", where, method)
					}
					strict(t, where+" result", m.Result, r)
					switch o := r.(type) {
					case *OutputState:
						if !validOutput(*o) {
							t.Fatalf("%s: output %+v", where, o)
						}
					case *OutputMuteResult:
						if !validOutput(o.State) || o.State.Device == nil || (o.Took && !o.State.IsMuted()) {
							t.Fatalf("%s: mute %+v", where, o)
						}
					}
					if c, ok := r.(*CaptureResult); ok {
						if c.Frames != nextSeq[m.ID] || c.Samples != nextOffset[m.ID] || !slices.Contains([]string{CaptureStopped, CaptureMaxSeconds, CaptureDeviceLost, CaptureNoCallback}, c.Reason) {
							t.Fatalf("%s: capture result %+v after %d frames, %d samples", where, c, nextSeq[m.ID], nextOffset[m.ID])
						}
					}
				default:
					t.Fatalf("%s: a %s cannot go to the %s", where, k, l.To)
				}
			}
			if len(pending) > 0 {
				t.Fatalf("unanswered requests: %v", pending)
			}
		})
	}
}

// validBinding: known families and sides, a behaviour, and a modifier for
// a chord.
func validBinding(b HotkeyBinding) bool {
	if b.ID == "" || !slices.Contains([]string{BehaviorTap, BehaviorDoubleTap, BehaviorLongPress}, b.Behavior) {
		return false
	}
	if b.Key == nil && len(b.Modifiers) == 0 {
		return false
	}
	for fam, side := range b.Modifiers {
		if !slices.Contains([]string{ModFn, ModShift, ModControl, ModOption, ModCommand}, fam) ||
			!slices.Contains([]string{SideAny, SideLeft, SideRight}, side) || (fam == ModFn && side != SideAny) {
			return false
		}
	}
	return true
}

// validOutput: a device with a UID; a settable mute that can be read; no
// mute without a device.
func validOutput(s OutputState) bool {
	if s.Device == nil {
		return s.Muted == nil && !s.Settable
	}
	return s.Device.UID != "" && (!s.Settable || s.Muted != nil)
}

// validEvent: a phase that fits the behaviour.
func validEvent(e HotkeyEventParams) bool {
	if e.Binding == "" || e.T <= 0 {
		return false
	}
	switch e.Behavior {
	case BehaviorTap, BehaviorDoubleTap:
		return e.Phase == PhaseTrigger
	case BehaviorLongPress:
		return e.Phase == PhaseDown || e.Phase == PhaseUp
	}
	return false
}

// TestStopAnswersAfterTheJob: a capture.stop or job.cancel that found its
// job is answered after that job's own response.
func TestStopAnswersAfterTheJob(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(fixturesDir, "*.jsonl"))
	for _, path := range paths {
		lines := readFixture(t, path)
		answered := map[string]int{} // id → line of its response
		requests := map[string]Message{}
		for i, l := range lines {
			m, _ := Decode(l.Msg)
			switch m.Kind() {
			case KindRequest:
				requests[m.ID] = m
			case KindResponse:
				answered[m.ID] = i
			}
		}
		for id, req := range requests {
			if req.Method != MethodCaptureStop && req.Method != MethodJobCancel {
				continue
			}
			var p CaptureStopParams
			req.DecodeParams(&p)
			resp := lines[answered[id]]
			m, _ := Decode(resp.Msg)
			var found struct {
				Stopped   bool `json:"stopped"`
				Cancelled bool `json:"cancelled"`
			}
			m.DecodeResult(&found)
			if !found.Stopped && !found.Cancelled {
				continue
			}
			if job, ok := answered[p.Job]; !ok || job > answered[id] {
				t.Errorf("%s: %s %s answered before job %s", filepath.Base(path), req.Method, id, p.Job)
			}
		}
	}
}

// TestFixturesCoverMethods: every method has a fixture showing it.
func TestFixturesCoverMethods(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(fixturesDir, "*.jsonl"))
	seen := map[string]bool{}
	for _, path := range paths {
		for _, l := range readFixture(t, path) {
			var m Message
			json.Unmarshal(l.Msg, &m)
			seen[m.Method] = true
		}
	}
	for _, m := range slices.Concat(Requests, Notifications) {
		if !seen[m] {
			t.Errorf("no fixture shows %s", m)
		}
	}
}

func TestEncodeDecode(t *testing.T) {
	req, err := NewRequest("j1", MethodBrowserURL, BrowserURLParams{BundleID: "com.apple.Safari \"x\"\nnew line"})
	if err != nil {
		t.Fatal(err)
	}
	line, err := Encode(req)
	if err != nil || bytes.Count(line, []byte("\n")) != 1 || !bytes.HasSuffix(line, []byte("\n")) {
		t.Fatalf("one message must be one line: %q %v", line, err)
	}
	m, err := Decode(line)
	if err != nil || m.Kind() != KindRequest {
		t.Fatalf("%+v %v", m, err)
	}
	var p BrowserURLParams
	if err := m.DecodeParams(&p); err != nil || p.BundleID != "com.apple.Safari \"x\"\nnew line" {
		t.Fatalf("%+v %v", p, err)
	}
	res, _ := NewResult("c1", nil)
	if res.Kind() != KindResponse || string(res.Result) != "{}" {
		t.Errorf("empty result %+v", res)
	}
	n, _ := NewNotification(MethodLog, LogParams{Level: "info", Message: "x"})
	if n.Kind() != KindNotification {
		t.Errorf("notification kind %v", n.Kind())
	}
	for _, bad := range []string{`{}`, `{"id":"1"}`, `{"id":"1","result":{},"error":{"code":"x","message":"y"}}`, `{"method":"x","result":{}}`, `not json`} {
		if _, err := Decode([]byte(bad)); err == nil {
			t.Errorf("Decode(%s) succeeded", bad)
		}
	}
}

func TestErrors(t *testing.T) {
	var err error = NewError("j1", Errorf(CodeBusy, "busy with %s", "j0")).Error
	if !errors.Is(err, &Error{Code: CodeBusy}) || errors.Is(err, &Error{Code: CodeTimeout}) {
		t.Error("errors.Is by code")
	}
	if !strings.Contains(err.Error(), "busy with j0") {
		t.Error(err)
	}
	for _, m := range Requests {
		if ParamsFor(m) == nil || ResultFor(m) == nil {
			t.Errorf("request %s lacks types", m)
		}
	}
	for _, m := range Notifications {
		if ParamsFor(m) == nil || ResultFor(m) != nil {
			t.Errorf("notification %s", m)
		}
	}
	if !IsJob(MethodCaptureStart) || IsJob(MethodBrowserURL) {
		t.Error("IsJob")
	}
}

// TestHelloListsEveryRequest: the handshake fixture's helper implements
// the whole protocol, as the Swift helper must (its tests check its own
// list against the same fixture).
func TestHelloListsEveryRequest(t *testing.T) {
	for _, l := range readFixture(t, filepath.Join(fixturesDir, "handshake.jsonl")) {
		m, _ := Decode(l.Msg)
		var h HelloResult
		if m.Kind() != KindResponse || m.DecodeResult(&h) != nil || h.Protocol == 0 {
			continue
		}
		got, want := slices.Sorted(slices.Values(h.Methods)), slices.Sorted(slices.Values(Requests))
		if !slices.Equal(got, want) {
			t.Fatalf("hello lists %v, the protocol has %v", got, want)
		}
		return
	}
	t.Fatal("no hello result in handshake.jsonl")
}
