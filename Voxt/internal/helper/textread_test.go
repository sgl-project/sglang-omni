package helper

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// TestTextRead: the fake helper answers textio.read from its script (the
// field's states, one a call, the last one staying), clamps the range to
// the value, refuses an app not in front or a password field, and logs
// every call.
func TestTextRead(t *testing.T) {
	dir := t.TempDir()
	script, log := filepath.Join(dir, "read.jsonl"), filepath.Join(dir, "read-log.jsonl")
	var lines []byte
	for _, f := range []helpertest.Field{{Value: "Cloud Code 👍", Element: 7}, {Value: "Claude Code 👍", Element: 7}} {
		line, _ := json.Marshal(f)
		lines = append(append(lines, line...), '\n')
	}
	if err := os.WriteFile(script, lines, 0o644); err != nil {
		t.Fatal(err)
	}
	h := start(t, helpertest.Opt("read", script)+","+helpertest.Opt("read-log", log))
	h.ready(t)
	for i, tc := range []struct {
		p    proto.ReadParams
		want proto.ReadResult
	}{
		{proto.ReadParams{PID: 812, Range: proto.TextRange{Location: -300, Length: 610}},
			proto.ReadResult{Text: "Cloud Code 👍", Range: proto.TextRange{Location: 0, Length: 13}, Length: 13, Element: 7}},
		{proto.ReadParams{PID: 812, Range: proto.TextRange{Location: 0, Length: 6}},
			proto.ReadResult{Text: "Claude", Range: proto.TextRange{Location: 0, Length: 6}, Length: 14, Element: 7}},
		{proto.ReadParams{PID: 812, Range: proto.TextRange{Location: 7, Length: 100}},
			proto.ReadResult{Text: "Code 👍", Range: proto.TextRange{Location: 7, Length: 7}, Length: 14, Element: 7}},
		{proto.ReadParams{PID: 901, Range: proto.TextRange{Location: 0, Length: 6}}, proto.ReadResult{Error: proto.ReadNotFront}},
	} {
		got, err := request[proto.ReadResult](t, h, proto.MethodTextRead, tc.p)
		if err != nil || got != tc.want {
			t.Errorf("read %d %+v: %+v %v, want %+v", i, tc.p, got, err, tc.want)
		}
	}
	data, _ := os.ReadFile(log)
	calls := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(calls) != 4 || calls[0] != `{"pid":812,"range":{"location":-300,"length":610}}` || !strings.Contains(calls[3], `"pid":901`) {
		t.Errorf("read log %q", data)
	}

	s := start(t, "focus=secure")
	s.ready(t)
	if got, err := request[proto.ReadResult](t, s, proto.MethodTextRead, proto.ReadParams{PID: 812}); err != nil || got != (proto.ReadResult{Error: proto.ReadSecure}) {
		t.Errorf("password field: %+v %v", got, err)
	}
}

// TestUndecodableLine: a line from the helper that does not decode (a
// textio.read answer cut short as it crashed) leaves its size in the
// failed status's stderr tail, never its text.
func TestUndecodableLine(t *testing.T) {
	h := start(t, helpertest.Opt("garble", "SECRET-TEXT"))
	h.ready(t)
	if _, err := request[proto.ReadResult](t, h, proto.MethodTextRead, proto.ReadParams{PID: 812}); !errors.Is(err, ErrExited) {
		t.Fatalf("read: %v", err)
	}
	var failed Status
	wait(t, "the failure", func() bool {
		h.mu.Lock()
		defer h.mu.Unlock()
		i := slices.IndexFunc(h.statuses, func(s Status) bool { return s.State == Failed })
		if i >= 0 {
			failed = h.statuses[i]
		}
		return i >= 0
	})
	if failed.Reason != FailCrashed || strings.Contains(failed.StderrTail, "SECRET-TEXT") || !strings.Contains(failed.StderrTail, "stdout: undecodable line (") || !strings.Contains(failed.StderrTail, " bytes)") {
		t.Errorf("failed status %+v", failed)
	}
}
