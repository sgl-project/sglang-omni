package providers

// TestRealServices sends one real request to each remote service whose key
// is in the login Keychain (VOXTNEW_REAL_SERVICES=1, on a Mac). Save the
// keys on Voxt's Language models and Speech models pages first: the test
// reads the same items (service VoxtNew, accounts llm.<id> and asr.<id>).
// Each service gets its connection test at its default address and model:
// a tiny chat for a language model, a second of silence for a speech
// service. The log names the service and the outcome, never a key
// (ErrorLog blanks it out of an error).
//
//	VOXTNEW_REAL_SERVICES=1 go test ./internal/providers -run TestRealServices -v

import (
	"context"
	"os"
	"runtime"
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
	tested := 0
	check := func(kind local.Kind, id string, test func(settings.Provider, string) TestResult) {
		key, err := secrets.Key(Account(kind, id), "").Get()
		if err != nil || key == "" {
			return
		}
		tested++
		res := test(settings.Provider{ID: id}, key)
		if res.Err != nil {
			t.Errorf("%s %s: %s", kind, id, ErrorLog(res.Err, key))
			return
		}
		t.Logf("%s %s: ok in %v", kind, id, res.Latency.Round(time.Millisecond))
	}
	for _, pr := range llm.Presets() {
		check(local.LLM, pr.ID, func(p settings.Provider, key string) TestResult { return Test(ctx, p, key, nil) })
	}
	for _, pr := range speechPresets {
		check(local.ASR, pr.ID, func(p settings.Provider, key string) TestResult { return TestSpeech(ctx, p, key, nil) })
	}
	if tested == 0 {
		t.Skip("no key in the Keychain: save them in Voxt first")
	}
}
