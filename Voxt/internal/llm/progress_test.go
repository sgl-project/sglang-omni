package llm

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// pacedBackend streams deltas, each after its pause, then answers their
// text; a pause of -1 blocks until the attempt ends. It counts its calls.
type pacedBackend struct {
	deltas []string
	pauses []time.Duration
	calls  atomic.Int32
}

func (b *pacedBackend) Info() BackendInfo { return BackendInfo{ID: "paced"} }

func (b *pacedBackend) Chat(ctx context.Context, req Request) (Response, error) {
	b.calls.Add(1)
	var text strings.Builder
	for i, d := range b.deltas {
		pause := b.pauses[min(i, len(b.pauses)-1)]
		if pause < 0 {
			<-ctx.Done()
			return Response{}, ctx.Err()
		}
		select {
		case <-ctx.Done():
			return Response{}, ctx.Err()
		case <-time.After(pause):
		}
		text.WriteString(d)
		if req.OnDelta != nil {
			req.OnDelta(d)
		}
	}
	return Response{Text: text.String(), FinishReason: FinishStop}, nil
}

// TestProgressDeadlines: a streamed attempt ends when its first token is
// late, when it stalls, or at its cap; a slow but steady stream passes
// however long it takes below the cap. A progress timeout is not retried.
func TestProgressDeadlines(t *testing.T) {
	ms := time.Millisecond
	words := strings.Split(strings.Repeat("word ", 20), " ")[:20]
	for _, c := range []struct {
		name    string
		backend *pacedBackend
		req     Request
		wait    string // the CodeTimeout's wait; "-" for a timeout of the cap, "" for success
		deltas  string
		most    time.Duration // the call returns within this
	}{
		{"first token late", &pacedBackend{deltas: []string{"Hi"}, pauses: []time.Duration{2 * time.Second}},
			Request{FirstTokenTimeout: 80 * ms, StallTimeout: 50 * ms, Timeout: 5 * time.Second}, WaitFirstToken, "0", time.Second},
		{"stall", &pacedBackend{deltas: []string{"a", "b", "c", "d"}, pauses: []time.Duration{5 * ms, 5 * ms, 5 * ms, -1}},
			Request{FirstTokenTimeout: time.Second, StallTimeout: 80 * ms, Timeout: 5 * time.Second}, WaitStall, "3", time.Second},
		{"cap", &pacedBackend{deltas: words, pauses: []time.Duration{20 * ms}},
			Request{FirstTokenTimeout: 100 * ms, StallTimeout: 100 * ms, Timeout: 150 * ms}, "-", "", time.Second},
		{"slow but steady", &pacedBackend{deltas: words, pauses: []time.Duration{60 * ms, 30 * ms}},
			Request{FirstTokenTimeout: 100 * ms, StallTimeout: 50 * ms, Timeout: 5 * time.Second}, "", "", 3 * time.Second},
		{"without OnDelta", &pacedBackend{deltas: words, pauses: []time.Duration{10 * ms}},
			Request{FirstTokenTimeout: 100 * ms, StallTimeout: 50 * ms}, "", "", 3 * time.Second},
		{"a reasoning block counts", &pacedBackend{deltas: []string{"<think>", "hm", "</think>", "Hi"}, pauses: []time.Duration{30 * ms}},
			Request{FirstTokenTimeout: 100 * ms, StallTimeout: 50 * ms}, "", "", 3 * time.Second},
	} {
		t.Run(c.name, func(t *testing.T) {
			client := &Client{Backend: c.backend, Sleep: noSleep}
			var streamed strings.Builder
			req := c.req
			req.Messages = msgs("x")
			if c.name != "without OnDelta" {
				req.OnDelta = func(d string) { streamed.WriteString(d) }
			}
			start := time.Now()
			resp, err := client.Chat(context.Background(), req)
			took := time.Since(start)
			if took > c.most {
				t.Errorf("took %v", took)
			}
			if c.backend.calls.Load() != 1 {
				t.Errorf("%d attempts", c.backend.calls.Load())
			}
			var e *Error
			switch c.wait {
			case "":
				if err != nil || resp.Text == "" || strings.Contains(resp.Text, "think") {
					t.Fatalf("resp %+v, err %v", resp, err)
				}
				if req.OnDelta != nil && streamed.String() != resp.Text {
					t.Errorf("streamed %q, answered %q", streamed.String(), resp.Text)
				}
			case "-":
				if !errors.As(err, &e) || e.Code != CodeTimeout || e.Params["wait"] != "" {
					t.Fatalf("err %v", err)
				}
			default:
				if !errors.As(err, &e) || e.Code != CodeTimeout || e.Params["wait"] != c.wait || e.Params["deltas"] != c.deltas || e.Retryable {
					t.Fatalf("err %v", err)
				}
			}
		})
	}
}

// TestProgressCancel: the caller's cancel stays a cancel, not a timeout.
func TestProgressCancel(t *testing.T) {
	b := &pacedBackend{deltas: []string{"a"}, pauses: []time.Duration{-1}}
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(30*time.Millisecond, cancel)
	_, err := (&Client{Backend: b}).Chat(ctx, Request{Messages: msgs("x"), FirstTokenTimeout: time.Second, StallTimeout: time.Second})
	if Code(err) != CodeCancelled {
		t.Fatalf("err %v", err)
	}
}
