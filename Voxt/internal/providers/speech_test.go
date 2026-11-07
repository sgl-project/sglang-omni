package providers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestSpeechPresets: every speech service builds a client with its
// defaults, which streams exactly when its preset says Realtime, and
// refuses an address of the wrong scheme.
func TestSpeechPresets(t *testing.T) {
	for _, pr := range SpeechPresets() {
		c, err := NewASR(settings.Provider{ID: pr.ID}, "k", nil)
		if err != nil {
			t.Errorf("%s: %v", pr.ID, err)
			continue
		}
		if caps := c.Caps(); caps.Stream != pr.Stream || caps.File == pr.Stream || pr.Model == "" || len(pr.Models) == 0 || pr.KeyURL == "" {
			t.Errorf("%s: caps %+v, preset %+v", pr.ID, caps, pr)
		}
		bad := "http://example.com/x"
		if pr.Stream {
			bad = "ws://example.com/x"
		}
		if _, err := NewASR(settings.Provider{ID: pr.ID, Endpoint: bad}, "k", nil); asr.KindOf(err) != asr.KindConfig {
			t.Errorf("%s accepts %s: %v", pr.ID, bad, err)
		}
		if _, err := NewASR(settings.Provider{ID: pr.ID}, "", nil); asr.KindOf(err) != asr.KindConfig {
			t.Errorf("%s works without a key: %v", pr.ID, err)
		}
	}
}

// TestSpeechReady: a speech setup is offered once its key is saved (an
// OpenAI-compatible server on this Mac needs none), in the presets' order;
// removing one puts the features that used it back on automatic, and
// leaves the language models alone.
func TestSpeechReady(t *testing.T) {
	var s settings.Settings
	PutSpeech(&s, settings.Provider{ID: "gemini"})
	PutSpeech(&s, settings.Provider{ID: "doubao", AppID: "a"})
	PutSpeech(&s, settings.Provider{ID: "openai"}) // no key
	PutSpeech(&s, settings.Provider{ID: "doubao", AppID: "b", Model: "volc.bigasr.sauc.duration"})
	PutSpeech(&s, settings.Provider{ID: StepFunASR, Endpoint: "http://127.0.0.1:8000/v1"}) // a Whisper server on this Mac: no key needed
	Put(&s, settings.Provider{ID: "openai"})
	hasKey := func(id string) bool { return id != "openai" && id != StepFunASR }
	var got []string
	for _, p := range SpeechReady(s, hasKey) {
		got = append(got, p.ID+"="+SpeechModel(p)+"/"+p.AppID)
	}
	if strings.Join(got, " ") != "doubao=volc.bigasr.sauc.duration/b stepfun=step-asr/ gemini=gemini-3.5-transcribe-live/" || len(s.Providers.ASR) != 4 {
		t.Fatalf("ready %q, setups %+v", got, s.Providers.ASR)
	}
	s.Features.Transcribe.ASR, s.Features.Rewrite.ASR, s.Features.Translate.LLM = "openai", "asr.local", "openai"
	RemoveSpeech(&s, "openai")
	if _, ok := SpeechSetup(s, "openai"); ok || s.Features.Transcribe.ASR != "" || s.Features.Rewrite.ASR != "asr.local" ||
		s.Features.Translate.LLM != "openai" || len(s.Providers.LLM) != 1 {
		t.Fatalf("after RemoveSpeech: %+v %+v", s.Providers, s.Features)
	}
}

// localSpeech is a local.Services with one speech model.
type localSpeech struct{ local.NotInstalled }

var localASR = local.Model{Kind: local.ASR, ID: "asr.qwen3", Name: "Qwen3-ASR 1.7B (8bit)"}

func (localSpeech) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{localASR}
	}
	return nil
}

func (localSpeech) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	return nil, localASR, nil
}

// TestServicesSpeech: the ready speech services come after the local
// speech model in every speech menu, never in the language models'; a
// feature's choice opens the remote one, automatic keeps the local one.
func TestServicesSpeech(t *testing.T) {
	opened := ""
	svc := NewServices(localSpeech{}, func(kind local.Kind, p settings.Provider) string { return string(kind) + ":" + p.ID },
		nil, func(_ context.Context, p settings.Provider) (asr.Provider, error) {
			opened = p.ID
			return NewASR(p, "k", nil)
		})
	svc.SetReady(local.ASR, []settings.Provider{{ID: "doubao"}})
	list := svc.Installed(local.ASR)
	if len(list) != 2 || list[0].ID != localASR.ID || list[1].ID != "doubao" || list[1].Name != "asr:doubao" || list[1].Kind != local.ASR || len(svc.Installed(local.LLM)) != 0 {
		t.Fatalf("installed %+v", list)
	}
	if _, m, err := svc.ASR(context.Background(), ""); err != nil || m.ID != localASR.ID || opened != "" {
		t.Fatalf("automatic: %+v %v %q", m, err, opened)
	}
	p, m, err := svc.ASR(context.Background(), "doubao")
	if err != nil || m.ID != "doubao" || opened != "doubao" || !p.Caps().Stream {
		t.Fatalf("remote: %+v %v %q", m, err, opened)
	}
	svc.SetReady(local.ASR, nil)
	if _, m, _ := svc.ASR(context.Background(), "doubao"); m.ID != localASR.ID {
		t.Fatalf("not ready any more: %+v", m)
	}
}

// TestSpeechTest: the connection test recognises a second of silence:
// Zhipu's upload gets its form (file, model, no language) and the answer
// "" works; a refused key fails, and the log line never shows it.
func TestSpeechTest(t *testing.T) {
	status, form := 200, ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(1 << 20)
		form = r.URL.Path + " " + r.Header.Get("Authorization") + " model=" + r.FormValue("model") + " language=" + r.FormValue("language")
		w.WriteHeader(status)
		if status != 200 {
			io.WriteString(w, `{"error":{"code":"1000","message":"身份验证失败 key zk-secret-9"}}`)
			return
		}
		io.WriteString(w, `{"id":"x","model":"glm-asr-2512","text":""}`)
	}))
	defer srv.Close()
	p := settings.Provider{ID: ZhipuASR, Endpoint: srv.URL + "/api/paas/v4/audio/transcriptions"}
	if res := TestSpeech(context.Background(), p, "zk-secret-9", nil); res.Err != nil || res.Latency <= 0 ||
		form != "/api/paas/v4/audio/transcriptions Bearer zk-secret-9 model=glm-asr-2512 language=" {
		t.Fatalf("test %+v, request %q", res, form)
	}
	status = 401
	res := TestSpeech(context.Background(), p, "zk-secret-9", nil)
	if asr.KindOf(res.Err) != asr.KindAuth {
		t.Fatalf("refused: %v", res.Err)
	}
	if level, msg := SpeechTestLog(p, res, "zk-secret-9"); level != "warning" || strings.Contains(msg, "zk-secret-9") || !strings.Contains(msg, "speech service zhipu (model glm-asr-2512)") {
		t.Errorf("log %s %q", level, msg)
	}
}

// TestSpeechErrorsHideKey: every speech service whose server quotes the
// key in a refusal (an upload's 400, a WebSocket handshake's) returns the
// error with the key blanked out, also when the message is long enough
// for the detail's 300-byte clip to cut the key in two.
func TestSpeechErrorsHideKey(t *testing.T) {
	const key = "sk-speech-secret-31-0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz"
	for _, pad := range []string{"", strings.Repeat("x", 252)} {
		for _, pr := range speechPresets {
			url := asrtest.RefuseServer(t, 400, `{"error":{"message":"`+pad+`bad key `+key+`"}}`)
			if !pr.Stream {
				url = "http" + strings.TrimPrefix(url, "ws")
			}
			c, err := NewASR(settings.Provider{ID: pr.ID, Endpoint: url}, key, nil)
			if err != nil {
				t.Fatalf("%s: %v", pr.ID, err)
			}
			_, err = asr.Recognize(context.Background(), c, asr.Silence(time.Second), "", nil, "", asr.Feed{})
			if err == nil || strings.Contains(err.Error(), key[:12]) || !strings.Contains(err.Error(), "[key]") {
				t.Errorf("%s (%d bytes before the key): %v", pr.ID, len(pad), err)
			}
		}
	}
}
