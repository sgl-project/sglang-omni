package doubao

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
)

func TestModeOf(t *testing.T) {
	for raw, want := range map[string]mode{
		DefaultEndpoint:                        modeAsync,
		DuplexEndpoint:                         modeDuplex,
		NostreamEndpoint:                       modeNostream,
		"wss://relay.example/bigmodel/":        modeDuplex,
		"ws://127.0.0.1:9/x/BIGMODEL_NOSTREAM": modeNostream,
		"wss://relay.example/asr":              modeAsync,
	} {
		u, _ := url.Parse(raw)
		if got := modeOf(u); got != want {
			t.Errorf("%s: mode %d, want %d", raw, got, want)
		}
	}
}

func TestLanguage(t *testing.T) {
	for hint, want := range map[string][2]string{
		"":           {"", ""},
		"zh":         {"", ""},
		"zh-Hans":    {"", ""},
		"zh-Hant":    {"", "tw"},
		"zh_TW":      {"", "tw"},
		"zh-HK":      {"", "hk"},
		"zh-Hant-CN": {"", "traditional"},
		"en":         {"en-US", ""},
		"en-GB":      {"en-US", ""},
		"ja":         {"ja-JP", ""},
		"yue":        {"yue-CN", ""},
		"sw":         {"", ""}, // not one of the service's languages: detect
	} {
		l, v := language(hint)
		if l != want[0] || v != want[1] {
			t.Errorf("%q: %q %q, want %q %q", hint, l, v, want[0], want[1])
		}
	}
}

func TestHotwords(t *testing.T) {
	got := hotwords([]string{" Voxt ", "voxt", "", "\n", "火山\n引擎", "MyGo"}, modeAsync)
	if strings.Join(got, "|") != "Voxt|火山 引擎|MyGo" {
		t.Errorf("cleaned: %q", got)
	}
	// The duplex endpoints take about 100 tokens: 60 two-character words
	// stop at 50.
	var many []string
	for i := range 60 {
		many = append(many, fmt.Sprintf("词%c", '一'+rune(i)))
	}
	if n := len(hotwords(many, modeAsync)); n != 50 {
		t.Errorf("async kept %d words, want 50", n)
	}
	if n := len(hotwords(many, modeNostream)); n != 60 {
		t.Errorf("nostream kept %d words, want 60", n)
	}
	for s, want := range map[string]int{"Voxt": 1, "Visual Studio Code": 3, "火山引擎": 4, "GPT-4o": 2, "": 1} {
		if got := estimateTokens(s); got != want {
			t.Errorf("estimateTokens(%q) = %d, want %d", s, got, want)
		}
	}
}

// The full client request for each endpoint and option, pretty-printed.
func TestRequests(t *testing.T) {
	cases := []struct {
		name   string
		m      mode
		req    asr.StreamRequest
		prompt string
	}{
		{"async, plain", modeAsync, asr.StreamRequest{}, ""},
		{"async, lang en (not sent), hints, prompt", modeAsync, asr.StreamRequest{Lang: "en", Hints: []string{"Voxt", "火山引擎"}}, "A note about \"Voxt\"."},
		{"async, traditional Chinese", modeAsync, asr.StreamRequest{Lang: "zh-Hant"}, ""},
		{"duplex, hints", modeDuplex, asr.StreamRequest{Hints: []string{"MyGo"}}, ""},
		{"nostream, lang ja", modeNostream, asr.StreamRequest{Lang: "ja"}, ""},
	}
	var b strings.Builder
	for _, c := range cases {
		raw, err := buildRequest(c.m, c.req, c.prompt)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&b, "== %s\n%s\n", c.name, prettyJSON(t, raw))
	}
	asrtest.Golden(t, "requests", b.String())
}

func TestParseResponse(t *testing.T) {
	cases := []struct {
		payload string
		text    string
		dur     int64
	}{
		{``, "", 0},
		{`{"audio_info":{"duration":0},"result":{"additions":{"log_id":"x"},"text":""}}`, "", 0},
		{`{"audio_info":{"duration":3696},"result":{"text":" 这是字节跳动，今日头条母公司。 ","utterances":[{"text":"这是字节跳动，"}]}}`, "这是字节跳动，今日头条母公司。", 3696},
		{`{"result":{"text":"","utterances":[{"text":"Hello, "},{"text":"world."}]}}`, "Hello, world.", 0},
		{`{"result":[{"text":"一"},{"text":"二"}]}`, "一二", 0},
		{`{"result":null}`, "", 0},
	}
	for _, c := range cases {
		p, err := parseResponse([]byte(c.payload))
		if err != nil || p.Text != c.text || p.DurationMS != c.dur {
			t.Errorf("%s: %+v %v, want %q %d", c.payload, p, err, c.text, c.dur)
		}
	}
	for _, bad := range []string{`{`, `{"result":"x"}`, `{"result":[1]}`} {
		if _, err := parseResponse([]byte(bad)); asr.KindOf(err) != asr.KindProtocol {
			t.Errorf("%s: %v, want a protocol error", bad, err)
		}
	}
}

func TestCodeError(t *testing.T) {
	cases := []struct {
		code      uint32
		payload   string
		kind      asr.Kind
		retryable bool
	}{
		{45000001, `{"error":"invalid request: field audio.format"}`, asr.KindProtocol, false},
		{45000002, ``, asr.KindAudio, false},
		{45000151, `audio format error`, asr.KindAudio, false},
		{45000081, `{"error":"wait package timeout"}`, asr.KindNetwork, true},
		{55000031, `{"error":"server busy"}`, asr.KindServer, true},
		{55000099, `internal`, asr.KindServer, true},
		{45000030, `{"error":"quota exceeded for resource volc.seedasr.sauc.duration"}`, asr.KindQuota, false},
		{45000031, `{"error":"concurrency exceeded"}`, asr.KindQuota, true},
		{45000040, `{"error":"invalid access token"}`, asr.KindAuth, false},
		{45000099, `{"code":45000099,"message":"something else"}`, asr.KindProtocol, false},
	}
	for _, c := range cases {
		e := codeError(c.code, []byte(c.payload))
		if e.Kind != c.kind || e.Retryable != c.retryable || e.Provider != ID || e.Code != fmt.Sprint(c.code) || e.Detail == "" {
			t.Errorf("%d %s: %+v", c.code, c.payload, e)
		}
	}
}

func TestHandshakeError(t *testing.T) {
	for _, c := range []struct {
		status int
		body   string
		kind   asr.Kind
	}{
		{401, `{"error":"load grant: requested grant not found"}`, asr.KindAuth},
		{403, `{"error":"[resource_id=volc.seedasr.sauc.duration] requested resource not granted"}`, asr.KindQuota},
		{429, `{"error":"too many requests"}`, asr.KindQuota},
	} {
		code, msg := asr.ErrorBody([]byte(c.body))
		err := handshakeError(asr.HTTPError(ID, c.status, code, msg))
		if asr.KindOf(err) != c.kind {
			t.Errorf("%d %s: %v, want %s", c.status, c.body, err, c.kind)
		}
	}
}

func prettyJSON(t testing.TB, raw []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("not JSON: %v: %s", err, raw)
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	enc.Encode(v) // map keys sorted
	return strings.TrimRight(b.String(), "\n")
}
