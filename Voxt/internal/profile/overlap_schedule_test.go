package profile

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestMeetingOverlapFakeStartsLLMBeforeRecordingStops(t *testing.T) {
	config := fakeOverlapConfig("dictation")
	config.SimulatedLLM.LatencyMS = 0
	options := fakeOverlapOptions(config)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	started := make(chan struct{})
	var startedOnce sync.Once
	var mu sync.Mutex
	var events []string
	record := func(event string) {
		mu.Lock()
		events = append(events, event)
		mu.Unlock()
	}
	delegate := newSimulatedLLM(0)
	options.newLLM = func() overlapLLM {
		return overlapLLMFunc(func(ctx context.Context, request simulatedLLMRequest) (string, error) {
			record("llm:" + request.Operation)
			startedOnce.Do(func() { close(started) })
			return delegate.Call(ctx, request)
		})
	}
	options.afterFeed = func(ctx context.Context, index int) error {
		record(fmt.Sprintf("feed:%d", index))
		if index == 0 {
			select {
			case <-started:
				record("llm-started-before-next-feed")
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	}
	speech := overlapASRFunc(func(_ context.Context, request asr.FileRequest) (asr.Result, error) {
		return asr.Result{Text: fakeOverlapText(request.Audio)}, nil
	})
	report := runOverlapMethod(ctx, config, fakeOverlapPCM(), speech, nil, "overlap", options)
	if report.Status != "success" {
		t.Fatalf("overlap did not start LLM independently of recording: %+v", report.Errors)
	}
	mu.Lock()
	defer mu.Unlock()
	feedZero, llmStart, resumed, feedOne := -1, -1, -1, -1
	for index, event := range events {
		switch event {
		case "feed:0":
			feedZero = index
		case "llm:cleanup":
			if llmStart == -1 {
				llmStart = index
			}
		case "llm-started-before-next-feed":
			resumed = index
		case "feed:1":
			if feedOne == -1 {
				feedOne = index
			}
		}
	}
	if !(feedZero >= 0 && feedZero < llmStart && llmStart < resumed && resumed < feedOne) {
		t.Errorf("wrong scheduling order: %v", events)
	}
	if report.BacklogAtStop.ASRSegments.Value == nil || report.BacklogAtStop.LLMSegments.Value == nil {
		t.Error("stop backlog was not measured")
	}
}
