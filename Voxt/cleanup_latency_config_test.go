package main

import (
	"os"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestCleanupLatencyEnvironment(t *testing.T) {
	t.Run("unset", func(t *testing.T) {
		previous, present := os.LookupEnv(cleanupFallbackEnvironment)
		if err := os.Unsetenv(cleanupFallbackEnvironment); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if present {
				_ = os.Setenv(cleanupFallbackEnvironment, previous)
			} else {
				_ = os.Unsetenv(cleanupFallbackEnvironment)
			}
		})
		policy, diagnostic := cleanupLatencyFromEnv()
		if policy.AutoFallbackAfter != 0 || diagnostic != "" {
			t.Fatalf("unset environment: policy %+v, diagnostic %q", policy, diagnostic)
		}
	})

	for _, tc := range []struct {
		name       string
		value      string
		want       time.Duration
		wantNotice bool
	}{
		{name: "positive", value: "1500", want: 1500 * time.Millisecond},
		{name: "zero", value: "0"},
		{name: "invalid", value: "fast", wantNotice: true},
		{name: "negative", value: "-1", wantNotice: true},
		{name: "overflow", value: "9223372036854775808", wantNotice: true},
		{name: "unreasonable", value: "60001", wantNotice: true},
		{name: "empty", value: "", wantNotice: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(cleanupFallbackEnvironment, tc.value)
			policy, diagnostic := cleanupLatencyFromEnv()
			if policy.AutoFallbackAfter != tc.want || (diagnostic != "") != tc.wantNotice {
				t.Fatalf("value %q: policy %+v, diagnostic %q", tc.value, policy, diagnostic)
			}
		})
	}
}

func TestSessionAppReadsCleanupLatencyEnvironment(t *testing.T) {
	t.Setenv(cleanupFallbackEnvironment, "1000")
	chat := &fakeChat{text: "model answer should not win", delay: 3 * time.Second}
	set, keys, _ := fileSetup(t, "spoken words", chat)
	events := script(t,
		hotkeyLine(0, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(400, "transcribe/0", proto.PhaseTrigger),
	)
	a := newSessionApp(t, events, func(s *settings.Settings) {
		set(s)
		s.Features.Transcribe.Enhance = true
	}, keys)
	a.pump("the app recording", func() bool { return a.sess.State == session.Recording })
	a.pump("the app's experimental fallback", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Done || a.sess.Result != "spoken words" || a.sess.Problem != nil {
		t.Fatalf("app session %+v", a.sess)
	}
	if requests := chat.requests(); len(requests) != 1 {
		t.Fatalf("the app admitted %d cleanup requests, want one before the configured deadline", len(requests))
	}
	if pasted := a.pasted(); len(pasted) != 1 || pasted[0].Text != "spoken words" {
		t.Fatalf("app pasted %+v", pasted)
	}
}
