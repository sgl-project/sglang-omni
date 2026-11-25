package omni

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func eventSocket(t *testing.T) (*websocket.Conn, chan<- []byte, func()) {
	t.Helper()
	messages := make(chan []byte, 8)
	accepted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		close(accepted)
		defer conn.Close(websocket.StatusNormalClosure, "")
		for {
			select {
			case message := <-messages:
				if err := conn.Write(r.Context(), websocket.MessageText, message); err != nil {
					return
				}
			case <-r.Context().Done():
				return
			}
		}
	}))
	conn, _, err := websocket.Dial(context.Background(), "ws"+strings.TrimPrefix(server.URL, "http"), nil)
	if err != nil {
		server.Close()
		t.Fatal(err)
	}
	select {
	case <-accepted:
	case <-time.After(5 * time.Second):
		conn.CloseNow()
		server.Close()
		t.Fatal("the event socket did not open")
	}
	return conn, messages, func() {
		conn.CloseNow()
		server.Close()
	}
}

func TestLateCandidatesAreOptIn(t *testing.T) {
	for _, tc := range []struct {
		name      string
		kind      string
		request   asr.StreamRequest
		candidate bool
	}{
		{"Qwen3-ASR opted in", kindQwen3ASR, asr.StreamRequest{LatePartials: true}, true},
		{"Qwen3-ASR without opt-in", kindQwen3ASR, asr.StreamRequest{}, false},
		{"other realtime model", kindNemotron, asr.StreamRequest{LatePartials: true}, false},
		{"LiveResult", kindQwen3ASR, asr.StreamRequest{LatePartials: true, LiveResult: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runtime := NewRuntime(Config{})
			t.Cleanup(func() { runtime.Close() })
			spec, ok := Lookup(ASRFast)
			if !ok {
				t.Fatal("Qwen3-ASR spec is missing")
			}
			a := &ASR{rt: runtime, spec: spec}
			l := &lease{s: &server{rt: runtime, spec: spec}, p: &process{}}
			a.spec.Kind = tc.kind
			ctx, cancel := context.WithCancel(context.Background())
			conn, messages, closeSocket := eventSocket(t)
			t.Cleanup(func() {
				cancel()
				closeSocket()
			})
			stream := &live{
				a: a, l: l, req: tc.request, ctx: ctx, cancel: cancel, conn: conn, preview: true,
				out:       make(chan []byte, 1),
				completed: make(chan string, 1), events: make(chan asr.Event, eventBuffer), readDone: make(chan struct{}),
			}
			var releasePieces func()
			if !tc.request.LiveResult {
				piecesExited := make(chan struct{})
				var piecesReleaseOnce sync.Once
				releasePieces = func() { piecesReleaseOnce.Do(func() { close(piecesExited) }) }
				stream.pc = &pieces{wake: make(chan struct{}, 1), exited: piecesExited}
				t.Cleanup(releasePieces)
			}
			go stream.read()
			messages <- []byte(`{"type":"transcription.segment","event_index":1,"segment_id":0,"text":"before finish","is_final":false}`)
			select {
			case event := <-stream.Events():
				if event.Kind != asr.EventPartial || event.Text != "before finish" {
					t.Fatalf("ordinary partial %+v", event)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("the ordinary partial was not emitted")
			}
			if err := stream.Finish(); err != nil {
				t.Fatal(err)
			}
			messages <- []byte(`{"type":"transcription.segment","event_index":2,"segment_id":0,"text":"late transcript","is_final":false}`)
			if tc.candidate {
				select {
				case event := <-stream.Events():
					if event.Kind != asr.EventCandidate || event.Text != "late transcript" {
						t.Fatalf("late event %+v", event)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("the opted-in candidate was not emitted")
				}
			} else {
				messages <- []byte(`{"type":"transcription.completed","text":"late transcript"}`)
				select {
				case <-stream.readDone:
				case <-time.After(5 * time.Second):
					t.Fatal("the reader did not consume the late preview")
				}
			}
			if releasePieces != nil {
				releasePieces()
			}
			events := drain(t, stream)
			for _, event := range events {
				if event.Kind == asr.EventCandidate && !tc.candidate {
					t.Fatalf("unrequested candidate %+v", event)
				}
			}
			if last := events[len(events)-1]; last.Kind != asr.EventFinal {
				t.Fatalf("terminal event %+v", last)
			}
		})
	}
}
