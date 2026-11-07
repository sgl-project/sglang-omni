package providers

// TestRealServices sends one real request to each remote service whose key
// is in the login Keychain (VOXTNEW_REAL_SERVICES=1, on a Mac). Save the
// keys on Voxt's Language models and Speech models pages first: the test
// reads the same items (service VoxtNew, accounts llm.<id> and asr.<id>).
// Each service gets its connection test at its default address and model:
// a tiny chat for a language model, a second of silence for a speech
// service. A language model then gets a streamed translation, as
// Translate sends it: the log has its first-token time, its total time
// and whether the stream carried a <think> block. The log names the
// service and the outcome, never a key (ErrorLog blanks it out of an
// error). The Custom setup has no default address: name one, and its
// model, in VOXTNEW_REAL_CUSTOM ("<address>|<model>").
//
//	VOXTNEW_REAL_SERVICES=1 go test ./internal/providers -run TestRealServices -v

import (
	"context"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/secrets"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestRealServices(t *testing.T) {
	if os.Getenv("VOXTNEW_REAL_SERVICES") != "1" || runtime.GOOS != "darwin" {
		t.Skip("set VOXTNEW_REAL_SERVICES=1 on a Mac, with the keys saved in Voxt")
	}
	ctx := context.Background()
	custom := strings.SplitN(os.Getenv("VOXTNEW_REAL_CUSTOM"), "|", 2)
	tested := 0
	check := func(kind local.Kind, id string, test func(settings.Provider, string) TestResult) {
		key, err := secrets.Key(Account(kind, id), "").Get()
		if err != nil || key == "" {
			return
		}
		tested++
		p := settings.Provider{ID: id}
		if id == llm.Custom && len(custom) == 2 {
			p.Endpoint, p.Model = custom[0], custom[1]
		}
		res := test(p, key)
		if res.Err != nil {
			t.Errorf("%s %s: %s", kind, id, ErrorLog(res.Err, key))
			return
		}
		t.Logf("%s %s: ok in %v", kind, id, res.Latency.Round(time.Millisecond))
	}
	for _, pr := range llm.Presets() {
		check(local.LLM, pr.ID, func(p settings.Provider, key string) TestResult {
			res := Test(ctx, p, key, nil)
			if res.Err == nil {
				res.Err = streamTranslate(t, ctx, p, key)
			}
			return res
		})
	}
	for _, pr := range speechPresets {
		check(local.ASR, pr.ID, func(p settings.Provider, key string) TestResult { return TestSpeech(ctx, p, key, nil) })
	}
	if tested == 0 {
		t.Skip("no key in the Keychain: save them in Voxt first")
	}
}

// streamTranslate streams a translation from p's model, as Translate
// does, and logs how long it took and whether a <think> block streamed.
func streamTranslate(t *testing.T, ctx context.Context, p settings.Provider, key string) error {
	b, err := NewLLM(p, key, nil)
	if err != nil {
		return err
	}
	var first time.Duration
	var streamed strings.Builder
	start := time.Now()
	resp, err := (&llm.Client{Backend: b}).Chat(ctx, llm.Request{
		Messages: []llm.Message{
			{Role: llm.System, Content: "Translate the user's text into Simplified Chinese. Reply with the translation only."},
			{Role: llm.User, Content: "Good morning, team. The release notes are ready for review."},
		},
		OnDelta: func(d string) {
			if first == 0 {
				first = time.Since(start)
			}
			streamed.WriteString(d)
		},
	})
	if err != nil {
		return err
	}
	if strings.TrimSpace(resp.Text) == "" {
		t.Errorf("%s: an empty translation", p.ID)
	}
	t.Logf("%s: streamed translation %q: first token %v, total %v, <think> streamed: %v", p.ID, resp.Text,
		first.Round(time.Millisecond), time.Since(start).Round(time.Millisecond), strings.Contains(streamed.String(), "<think>"))
	return nil
}
