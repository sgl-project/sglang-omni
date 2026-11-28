package speakerrt

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	if len(os.Args) == 3 && os.Args[1] == "--model" && strings.HasPrefix(os.Args[2], "test-peer:") {
		mode := strings.TrimPrefix(os.Args[2], "test-peer:")
		if mode == "loading" {
			io.Copy(io.Discard, os.Stdin)
			return
		}
		fmt.Println(`{"ready":true}`)
		scan := bufio.NewScanner(os.Stdin)
		for scan.Scan() {
			if mode == "stalled" {
				io.Copy(io.Discard, os.Stdin)
				return
			}
			var req struct {
				Op    string
				Audio []byte
			}
			if err := json.Unmarshal(scan.Bytes(), &req); err != nil {
				os.Exit(2)
			}
			if len(req.Audio) != 6400 {
				os.Exit(3)
			}
			fmt.Println(`{"start_frame":7,"frames":[[0.1,0.9,0,0],[0,0,0.95,0]]}`)
		}
		return
	}
	os.Exit(m.Run())
}

func TestClientFramesAndCancellation(t *testing.T) {
	for _, mode := range []string{"normal", "loading", "stalled", "budget"} {
		t.Run(mode, func(t *testing.T) {
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			var closed atomic.Int32
			model := Model{Binary: exe, Dir: "test-peer:" + mode, StartTimeout: 30 * time.Millisecond, RequestTimeout: 30 * time.Millisecond, OnClose: func() { closed.Add(1) }}
			if mode == "budget" {
				var reads atomic.Int32
				model.Memory = func(int) (int64, error) {
					if reads.Add(1) > 1 {
						return 20, nil
					}
					return 5, nil
				}
				model.MaxMemory = 10
			}
			if mode != "loading" {
				model.StartTimeout = 5 * time.Second
			}
			stream, err := model.Open(context.Background())
			if mode == "loading" {
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("loading deadline: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { stream.Close() })
			frames, err := stream.Push(context.Background(), make([]byte, 6400))
			if mode == "stalled" {
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("request deadline: %v", err)
				}
			} else if mode == "budget" {
				if err == nil || !strings.Contains(err.Error(), "exceeds budget") {
					t.Fatalf("memory budget: %v", err)
				}
			} else if err != nil || len(frames) != 2 || frames[0].Start != 70*time.Millisecond || frames[1].Scores[2] != 0.95 {
				t.Fatalf("frames %+v: %v", frames, err)
			}
			stream.Close()
			stream.Close()
			if closed.Load() != 1 {
				t.Fatalf("teardown callbacks %d", closed.Load())
			}
		})
	}
}
