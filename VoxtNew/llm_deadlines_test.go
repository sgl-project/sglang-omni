package main

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
)

// TestLLMDeadlines: the progress deadlines over voxtnew-llm (the fake, or
// the real model with VOXTNEW_LLM_REAL=1): a first token that cannot come
// in time stops the chat, with a log line of how far it got and of the
// Mac's memory, and the process answers the next chat (it cancelled the
// stopped one, it was not killed).
func TestLLMDeadlines(t *testing.T) {
	svc, l := llmServices(t, &fakeTranscriptions{text: "unused"}, "Let's meet on Wednesday at three.", llmrt.Balanced)
	chat, _, err := svc.LLM(context.Background(), llmrt.Balanced)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []llm.Message{{Role: llm.User, Content: "Um, let's meet on Tuesday, no wait, on Wednesday at three. Clean this up."}}
	steady := llm.Request{Messages: msgs, Skill: llm.SkillTranscribeEnhance, FirstTokenTimeout: 30 * time.Second, StallTimeout: 5 * time.Second, Timeout: 2 * time.Minute}
	if _, err := chat.Chat(context.Background(), steady); err != nil {
		t.Fatal(err)
	}
	late := steady
	late.FirstTokenTimeout = time.Millisecond
	_, err = chat.Chat(context.Background(), late)
	var e *llm.Error
	if !errors.As(err, &e) || e.Code != llm.CodeTimeout || e.Params["wait"] != llm.WaitFirstToken {
		t.Fatalf("err %v", err)
	}
	want := "the Mac's memory unknown"
	if runtime.GOOS == "darwin" {
		want = "memory pressure "
	}
	var stopped []string
	for _, line := range l.lines("stopped after") {
		if strings.Contains(line, want) {
			stopped = append(stopped, line)
		}
	}
	if len(stopped) != 1 {
		t.Errorf("stopped lines %q", l.lines(""))
	}
	t0 := time.Now()
	if _, err := chat.Chat(context.Background(), steady); err != nil {
		t.Fatal(err)
	}
	if st := l.rt.Status(); st.Starts != 1 {
		t.Errorf("the process started %d times", st.Starts)
	}
	t.Logf("%s\nthe next chat took %v", strings.Join(stopped, "\n"), time.Since(t0).Round(time.Millisecond))
}
