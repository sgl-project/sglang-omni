package wire

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// fixturesDir holds the golden transcripts shared with the Swift package.
const fixturesDir = "../../../llm/Fixtures"

// fixtureLine is one line of a fixture: a wire message and its direction.
type fixtureLine struct {
	To  string          `json:"to"` // "llm" (app → process) or "app"
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
		if l.To != "llm" && l.To != "app" {
			t.Fatalf("%s:%d: to %q", filepath.Base(path), n, l.To)
		}
		lines = append(lines, l)
	}
	return lines
}

func fixturePaths(t *testing.T) []string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(fixturesDir, "*.jsonl"))
	if err != nil || len(paths) < 4 {
		t.Fatalf("fixtures: %v %v", paths, err)
	}
	return paths
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
// checks the protocol's sequencing rules on it (docs/llm-protocol.md).
func TestFixtures(t *testing.T) {
	for _, path := range fixturePaths(t) {
		name := filepath.Base(path)
		t.Run(name, func(t *testing.T) {
			pending := map[string]string{} // request id → method
			deltas := map[string]string{}  // job → its text so far
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
				case l.To == "llm" && k == KindRequest:
					if _, dup := pending[m.ID]; dup {
						t.Fatalf("%s: id %s reused while pending", where, m.ID)
					}
					pending[m.ID] = m.Method
					if p := ParamsFor(m.Method); p != nil && slices.Contains(Requests, m.Method) {
						strict(t, where+" params", m.Params, p)
						checkParams(t, where, p)
					}
				case l.To == "app" && k == KindNotification:
					p := ParamsFor(m.Method)
					if p == nil || !slices.Contains(Notifications, m.Method) {
						t.Fatalf("%s: unknown notification %s", where, m.Method)
					}
					strict(t, where+" params", m.Params, p)
					job := ""
					switch v := p.(type) {
					case *DeltaParams:
						job = v.Job
						if v.Text == "" {
							t.Fatalf("%s: an empty delta", where)
						}
						deltas[job] += v.Text
					case *LogParams:
						job = v.Job
						if !slices.Contains([]string{"debug", "info", "warning", "error"}, v.Level) {
							t.Fatalf("%s: log level %q", where, v.Level)
						}
					}
					if job != "" && !IsJob(pending[job]) {
						t.Fatalf("%s: notification for %s, which is not a running job", where, job)
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
					if c, ok := r.(*ChatResult); ok {
						if !strings.HasPrefix(deltas[m.ID], c.Text) || !slices.Contains([]string{FinishStop, FinishLength}, c.Finish) {
							t.Fatalf("%s: chat result %q (%s) after deltas %q", where, c.Text, c.Finish, deltas[m.ID])
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

// checkParams holds requests to the spec's values.
func checkParams(t *testing.T, where string, p any) {
	t.Helper()
	switch v := p.(type) {
	case *LoadParams:
		checkModel(t, where, v.Model)
	case *ChatParams:
		if v.Model.Format == FormatMLX || v.Model.Format == FormatGGUF {
			checkModel(t, where, v.Model)
		}
		if len(v.Messages) == 0 || v.MaxTokens < 0 || v.Sampling.Temperature < 0 {
			t.Fatalf("%s: chat %+v", where, v)
		}
		for _, m := range v.Messages {
			if !slices.Contains([]string{"system", "user", "assistant"}, m.Role) {
				t.Fatalf("%s: role %q", where, m.Role)
			}
		}
	}
}

func checkModel(t *testing.T, where string, m Model) {
	t.Helper()
	if m.ID == "" || m.Path == "" {
		t.Fatalf("%s: model %+v", where, m)
	}
	if m.Format == FormatGGUF && !strings.HasSuffix(m.Path, ".gguf") {
		t.Fatalf("%s: a gguf model is a .gguf file: %s", where, m.Path)
	}
}

// TestCancelAnswersAfterTheJob: a job.cancel that found its job is
// answered after that job's own response.
func TestCancelAnswersAfterTheJob(t *testing.T) {
	for _, path := range fixturePaths(t) {
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
			if req.Method != MethodJobCancel {
				continue
			}
			var p CancelParams
			req.DecodeParams(&p)
			m, _ := Decode(lines[answered[id]].Msg)
			var r CancelResult
			m.DecodeResult(&r)
			if !r.Cancelled {
				continue
			}
			if job, ok := answered[p.Job]; !ok || job > answered[id] {
				t.Errorf("%s: job.cancel %s answered before job %s", filepath.Base(path), id, p.Job)
			}
		}
	}
}

// TestFixturesCoverMethods: every method has a fixture showing it.
func TestFixturesCoverMethods(t *testing.T) {
	seen := map[string]bool{}
	for _, path := range fixturePaths(t) {
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

// TestHelloListsEveryRequest: the handshake's methods are the protocol's.
func TestHelloListsEveryRequest(t *testing.T) {
	for _, l := range readFixture(t, filepath.Join(fixturesDir, "handshake.jsonl")) {
		m, _ := Decode(l.Msg)
		var r HelloResult
		if m.Kind() != KindResponse || m.DecodeResult(&r) != nil || r.Protocol == 0 {
			continue
		}
		if !slices.Equal(r.Methods, Requests) {
			t.Fatalf("hello lists %v, want %v", r.Methods, Requests)
		}
		return
	}
	t.Fatal("no hello result in handshake.jsonl")
}

func TestEncodeDecode(t *testing.T) {
	req, err := NewRequest("c1", MethodChat, ChatParams{Messages: []ChatMessage{{Role: "user", Content: "a \"b\"\nc"}}, MaxTokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	line, err := Encode(req)
	if err != nil || bytes.Count(line, []byte("\n")) != 1 {
		t.Fatalf("one message must be one line: %q %v", line, err)
	}
	m, err := Decode(line)
	if err != nil || m.Kind() != KindRequest {
		t.Fatalf("%+v %v", m, err)
	}
	var p ChatParams
	if err := m.DecodeParams(&p); err != nil || p.Messages[0].Content != "a \"b\"\nc" {
		t.Fatalf("%+v %v", p, err)
	}
	res, _ := NewResult("c1", nil)
	if res.Kind() != KindResponse || string(res.Result) != "{}" {
		t.Errorf("empty result %+v", res)
	}
	for _, bad := range []string{`{}`, `{"id":"1"}`, `{"id":"1","result":{},"error":{"code":"x","message":"y"}}`, `{"method":"x","result":{}}`, `not json`} {
		if _, err := Decode([]byte(bad)); err == nil {
			t.Errorf("Decode(%s) did not fail", bad)
		}
	}
	e := &Error{Code: CodeCancelled, Message: "x"}
	if !e.Is(&Error{Code: CodeCancelled}) || e.Is(&Error{Code: CodeInternal}) {
		t.Error("Error.Is matches by code")
	}
}
