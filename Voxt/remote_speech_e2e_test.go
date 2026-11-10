package main

// The remote speech services in the app: whole Transcribe sessions over
// the fake helper through an upload service (OpenAI Transcribe) and a
// streaming one (Google Gemini), each against a local mock of its API,
// and the Speech models page's remote services: the dialog saving a
// setup and its key.

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/doubao"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestRemoteSpeechUpload: Transcribe chose OpenAI Transcribe over the
// local speech model: the recording goes to /v1/audio/transcriptions as a
// WAV with the key and the model, and its text is pasted and kept, the
// local model untouched.
func TestRemoteSpeechUpload(t *testing.T) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, params, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		form := multipart.NewReader(r.Body, params["boundary"])
		line := r.URL.Path + " " + r.Header.Get("Authorization")
		for {
			part, err := form.NextPart()
			if err != nil {
				break
			}
			b, _ := io.ReadAll(part)
			if part.FormName() == "file" {
				pcm, rate, _, err := asr.ReadWAV(b)
				line += " file:" + map[bool]string{true: "wav16k", false: "bad"}[err == nil && rate == asr.SampleRate && len(pcm) > 0]
			} else {
				line += " " + part.FormName() + "=" + string(b)
			}
		}
		mu.Lock()
		got = append(got, line)
		mu.Unlock()
		io.WriteString(w, `{"text":"Hello from the mock."}`)
	}))
	defer srv.Close()
	useKeys(t, &memKeys{m: map[string]string{"asr.openai": "sk-asr-123"}})
	set, lm, tr := fileSetup(t, "the local model's text", nil)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Transcribe.Enhance = false
		s.Features.Transcribe.ASR = "openai"
		s.Providers.ASR = []settings.Provider{{ID: "openai", Endpoint: srv.URL + "/v1"}}
	}, lm)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if end := a.sess; end.State != session.Done || end.Result != "Hello from the mock." || !end.Inserted || end.ASR.ID != "openai" {
		t.Fatalf("end %+v", end)
	}
	if p := a.pasted(); len(p) != 1 || p[0].Text != "Hello from the mock." {
		t.Errorf("pasted %+v", p)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || !strings.HasPrefix(got[0], "/v1/audio/transcriptions Bearer sk-asr-123 file:wav16k model=gpt-4o-mini-transcribe response_format=json") {
		t.Fatalf("requests %q", got)
	}
	if tr.calls != 0 {
		t.Errorf("the local model transcribed too")
	}
	if es := a.history(); len(es) != 1 || es[0].ASRProvider != "openai" {
		t.Errorf("history %+v", es)
	}
	// The upload is answered once recognised: the remote services' client
	// does not give up after the downloads' 30 s wait for the headers.
	if rt, ok := a.remoteHTTP.Load().Transport.(*http.Transport); !ok || rt.ResponseHeaderTimeout != 0 {
		t.Errorf("the remote services' client bounds the wait for an answer")
	}
}

// TestRemoteSpeechDoubaoFlash: Transcribe on Doubao with its flash file
// recognition as the model: once the recording stops it goes in one POST,
// as a WAV, with the setup's App ID and token and the flash resource, and
// the text of the answer is pasted.
func TestRemoteSpeechDoubaoFlash(t *testing.T) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Audio struct {
				Data string `json:"data"`
			} `json:"audio"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		wav, _ := base64.StdEncoding.DecodeString(body.Audio.Data)
		pcm, rate, _, err := asr.ReadWAV(wav)
		mu.Lock()
		got = append(got, fmt.Sprintf("%s %s %s %s wav16k=%v", r.URL.Path, r.Header.Get("X-Api-App-Key"), r.Header.Get("X-Api-Access-Key"),
			r.Header.Get("X-Api-Resource-Id"), err == nil && rate == asr.SampleRate && len(pcm) > 0))
		mu.Unlock()
		w.Header().Set("X-Api-Status-Code", "20000000")
		io.WriteString(w, `{"audio_info":{"duration":800},"result":{"text":"豆包极速版。"}}`)
	}))
	defer srv.Close()
	useKeys(t, &memKeys{m: map[string]string{"asr.doubao": "token-0123456789"}})
	set, lm, tr := fileSetup(t, "the local model's text", nil)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Transcribe.Enhance = false
		s.Features.Transcribe.ASR = "doubao"
		s.Providers.ASR = []settings.Provider{{ID: "doubao", AppID: "app-42", Model: doubao.ModelFlash, Endpoint: srv.URL + "/api/v3/auc/bigmodel/recognize/flash"}}
	}, lm)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if end := a.sess; end.State != session.Done || end.Result != "豆包极速版。" || end.ASR.ID != "doubao" {
		t.Fatalf("end %+v", end)
	}
	mu.Lock()
	defer mu.Unlock()
	if want := "/api/v3/auc/bigmodel/recognize/flash app-42 token-0123456789 volc.bigasr.auc_turbo wav16k=true"; len(got) != 1 || got[0] != want {
		t.Fatalf("requests %q", got)
	}
	if tr.calls != 0 {
		t.Errorf("the local model transcribed too")
	}
}

// TestRemoteSpeechFlashDialog: Doubao's dialog says the recording goes
// once it stops while the flash recognition is its model, and realtime
// for its streaming models.
func TestRemoteSpeechFlashDialog(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{"asr.doubao": "token-0123456789"}})
	a, _, tt := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) {
		providers.PutSpeech(s, settings.Provider{ID: "doubao", AppID: "app-42", Model: doubao.ModelFlash})
	})
	a.navigate(modelsPath)
	a.openProviderDialog(local.ASR, "doubao")
	if !has(tt, "Remote speech recognition · the recording, once you stop") || has(tt, "Remote speech recognition · realtime, while you speak") {
		t.Fatalf("the flash recognition's dialog: %q", firstTexts(tt))
	}
	a.provDlg.model = doubao.ModelV2
	if !has(tt, "Remote speech recognition · realtime, while you speak") {
		t.Errorf("the streaming model's dialog: %q", firstTexts(tt))
	}
}

// TestRemoteSpeechErrorKeepsNoKey: an upload service that quotes the key
// in its refusal fails the session, and the error kept in the history has
// the key blanked out.
func TestRemoteSpeechErrorKeepsNoKey(t *testing.T) {
	const key = "sk-asr-secret-55"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"invalid header `+r.Header.Get("Authorization")+`"}}`)
	}))
	defer srv.Close()
	useKeys(t, &memKeys{m: map[string]string{"asr.openai": key}})
	set, lm, _ := fileSetup(t, "the local model's text", nil)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Transcribe.Enhance = false
		s.Features.Transcribe.ASR = "openai"
		s.Providers.ASR = []settings.Provider{{ID: "openai", Endpoint: srv.URL + "/v1"}}
	}, lm)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Failed {
		t.Fatalf("end %+v", a.sess)
	}
	es := a.history()
	if len(es) != 1 || !strings.Contains(es[0].Error, "invalid header Bearer [key]") || strings.Contains(es[0].Error, key) {
		t.Fatalf("history %+v", es)
	}
}

// TestRemoteSpeechStreaming: with no local speech model, Transcribe
// streams the recording to Gemini while it records (the key in the
// handshake's header) and pastes the transcription once the audio ends.
func TestRemoteSpeechStreaming(t *testing.T) {
	audio := make(chan int, 1)
	url := asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		if c.Req.Header.Get("x-goog-api-key") != "AIza-mock" {
			t.Errorf("handshake key %q", c.Req.Header.Get("x-goog-api-key"))
		}
		c.Read() // the setup
		c.WriteText(`{"setupComplete":{}}`)
		n := 0
		for {
			_, data, ok := c.TryRead()
			if !ok || strings.Contains(string(data), "audioStreamEnd") {
				break
			}
			if n++; n == 1 {
				c.WriteText(`{"serverContent":{"interimInputTranscription":{"text":"你好"}}}`)
			}
		}
		audio <- n
		c.WriteText(`{"serverContent":{"inputTranscription":{"text":"你好，Voxt。"}}}`)
		c.WriteText(`{"serverContent":{"turnComplete":true}}`)
		c.TryRead()
	})
	useKeys(t, &memKeys{m: map[string]string{"asr.gemini": "AIza-mock"}})
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(600, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.Providers.ASR = []settings.Provider{{ID: "gemini", Endpoint: url}}
	}, local.NotInstalled{})
	sawLive := false
	a.pump("the end", func() bool {
		sawLive = sawLive || strings.Contains(a.sess.Partial, "你好")
		return a.sess.State.Ended()
	})
	if end := a.sess; end.State != session.Done || end.Result != "你好，Voxt。" || !end.Inserted || end.ASR.ID != "gemini" {
		t.Fatalf("end %+v", end)
	}
	if n := <-audio; n < 2 || !sawLive {
		t.Errorf("%d audio messages, live text seen %v", n, sawLive)
	}
	if code := a.problemCode[problemModels]; code != "llm" { // the key read at start made Gemini a speech model
		t.Errorf("the sidebar's card says %q is missing", code)
	}
}

// TestRemoteSpeechPage: setting Doubao up on the Speech models page saves
// its Access Token in the keyring as asr.doubao (never the settings) and
// its App ID in the settings; the sidebar no longer says a speech model is
// missing, Transcribe's speech menu offers it, and Remove takes it away
// and puts the feature back on automatic.
func TestRemoteSpeechPage(t *testing.T) {
	keys := &memKeys{m: map[string]string{}}
	useKeys(t, keys)
	a, _, tt := newTestApp(t)
	a.navigate(modelsPath)
	frames(tt)
	if !has(tt, "Online services") || hasExact(tt, "Set up") || !has(tt, "OpenAI Transcribe") {
		t.Fatal("no speech services on the Speech models page, or a Set up group with none set up")
	}
	// A service that takes no key from a server on this Mac says so.
	keyless := "Optional for a server on this Mac. Saved in this Mac's Keychain."
	scrollToText(tt, "OpenAI Transcribe")
	click(t, tt, "OpenAI Transcribe")
	frames(tt)
	if !has(tt, keyless) {
		t.Errorf("OpenAI Transcribe's key hint: %q", tt.Texts())
	}
	a.provDlg = providerDialog{}
	frames(tt)
	before := a.problemCode[problemModels]
	scrollToText(tt, "Doubao (Volcengine)")
	click(t, tt, "Doubao (Volcengine)")
	frames(tt)
	if has(tt, keyless) || !has(tt, "Saved in this Mac's Keychain.") {
		t.Errorf("Doubao's key hint: %q", tt.Texts())
	}
	a.provDlg.key, a.provDlg.appID = "token-0123456789", "app-42"
	click(t, tt, "Save")
	frames(tt)
	if keys.m["asr.doubao"] != "token-0123456789" || keys.m["llm.doubao"] != "" {
		t.Fatalf("keys %v", keys.m)
	}
	p := a.lib.settings.Get().Providers
	if len(p.ASR) != 1 || p.ASR[0].ID != "doubao" || p.ASR[0].AppID != "app-42" || len(p.LLM) != 0 {
		t.Fatalf("setups %+v", p)
	}
	raw, _ := json.Marshal(a.lib.settings.Get())
	if strings.Contains(string(raw), "token-0123") {
		t.Fatal("the key is in the settings")
	}
	if !hasExact(tt, "Set up") || !has(tt, "Used by Transcribe · Translate · Rewrite") {
		t.Errorf("the card does not say where Doubao is used: %q", tt.Texts())
	}
	if before != "both" || a.problemCode[problemModels] != "llm" {
		t.Errorf("the sidebar's card: %q, then %q", before, a.problemCode[problemModels])
	}
	m, ok := a.featureModel(local.ASR, hotkey.Transcribe)
	if !ok || m.ID != "doubao" || m.Name != "Doubao (Volcengine) · volc.seedasr.sauc.duration" {
		t.Fatalf("Transcribe's speech model %+v %v", m, ok)
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = "doubao" })
	a.navigate("/transcribe")
	frames(tt)
	if !has(tt, "Doubao (Volcengine) · volc.seedasr.sauc.duration") {
		t.Error("Transcribe's speech menu does not show the remote service")
	}
	a.navigate(modelsPath)
	frames(tt)
	scrollToText(tt, "Doubao (Volcengine)")
	click(t, tt, "Doubao (Volcengine)")
	frames(tt)
	click(t, tt, "Remove Doubao (Volcengine)")
	frames(tt)
	s := a.lib.settings.Get()
	if len(keys.m) != 0 || len(s.Providers.ASR) != 0 || s.Features.Transcribe.ASR != "" || len(a.installed(local.ASR)) != 0 {
		t.Fatalf("after Remove: keys %v, setups %+v, Transcribe's ASR %q", keys.m, s.Providers.ASR, s.Features.Transcribe.ASR)
	}
}

// speechShots: the Speech models page's remote services with Doubao set
// up and tested, and its dialog (TestSnapshots).
func speechShots(a *app, tt *ui.Tester, save func(string)) {
	a.keyMasks["asr.doubao"] = maskKey("token-0123456789")
	a.provTests["asr.doubao"] = providerTest{res: providers.TestResult{Latency: 412 * time.Millisecond, At: time.Now()}}
	a.updateSettings(func(s *settings.Settings) { providers.PutSpeech(s, settings.Provider{ID: "doubao", AppID: "app-42"}) })
	a.navigate(modelsPath)
	frames(tt)
	scrollToText(tt, i18n.T("Set up"))
	save("page-models-remote-setup")
	a.openProviderDialog(local.ASR, "doubao")
	save("remote-speech-dialog")
	a.provDlg = providerDialog{}
	a.updateSettings(func(s *settings.Settings) { providers.RemoveSpeech(s, "doubao") })
	delete(a.keyMasks, "asr.doubao")
	delete(a.provTests, "asr.doubao")
}
