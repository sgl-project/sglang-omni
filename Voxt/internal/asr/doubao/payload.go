package doubao

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// mode is which of the three SAUC endpoints a URL is; they take
// different request fields.
type mode uint8

const (
	// modeAsync: /bigmodel_async, the optimised duplex stream. Results come
	// only when they change; it alone takes enable_nonstream (the second
	// pass that re-recognises each finished sentence with the non-streaming
	// model).
	modeAsync mode = iota
	// modeDuplex: /bigmodel, the original duplex stream (one result per
	// audio packet).
	modeDuplex
	// modeNostream: /bigmodel_nostream, streaming input with results only
	// every 15 s or at the end; it alone takes audio.language.
	modeNostream
)

// modeOf tells the endpoint's mode by its path; anything else (a relay, a
// test server) is taken as the async endpoint, the default.
func modeOf(u *url.URL) mode {
	p := strings.ToLower(strings.TrimRight(u.Path, "/"))
	switch {
	case strings.HasSuffix(p, "/bigmodel_nostream"):
		return modeNostream
	case strings.HasSuffix(p, "/bigmodel"):
		return modeDuplex
	}
	return modeAsync
}

// Hot word budgets: the duplex endpoints take "100 tokens" of hot words,
// the nostream one 5000 words (the docs, corpus.context).
const (
	streamHintTokens = 100
	nostreamHints    = 5000
	// maxPrompt bounds the context text sent (the service keeps 800
	// tokens of it and truncates the rest itself).
	maxPrompt = 4000
)

// request is the full client request's JSON.
type request struct {
	User    reqUser    `json:"user"`
	Audio   reqAudio   `json:"audio"`
	Request reqRequest `json:"request"`
}

type reqUser struct {
	UID string `json:"uid"`
}

type reqAudio struct {
	Format   string `json:"format"`
	Codec    string `json:"codec"`
	Rate     int    `json:"rate"`
	Bits     int    `json:"bits"`
	Channel  int    `json:"channel"`
	Language string `json:"language,omitempty"`
}

type reqRequest struct {
	ModelName       string     `json:"model_name"`
	EnableITN       bool       `json:"enable_itn"`
	EnablePunc      bool       `json:"enable_punc"`
	EnableDDC       bool       `json:"enable_ddc"`
	ShowUtterances  bool       `json:"show_utterances"`
	EnableNonstream bool       `json:"enable_nonstream,omitempty"`
	OutputZhVariant string     `json:"output_zh_variant,omitempty"`
	Corpus          *reqCorpus `json:"corpus,omitempty"`
}

type reqCorpus struct {
	// Context is a JSON document in a string: hot words and/or dialog
	// context.
	Context string `json:"context"`
}

// corpusContext is what corpus.context holds.
type corpusContext struct {
	Hotwords    []hotword     `json:"hotwords,omitempty"`
	ContextType string        `json:"context_type,omitempty"`
	ContextData []contextItem `json:"context_data,omitempty"`
}

type hotword struct {
	Word string `json:"word"`
}

type contextItem struct {
	Text string `json:"text"`
}

// buildRequest is the full client request for req on an endpoint of mode
// m, with prompt the request's prompt or the provider's.
func buildRequest(m mode, req asr.StreamRequest, prompt string) ([]byte, error) {
	r := request{
		User: reqUser{UID: "voxt"},
		Audio: reqAudio{
			Format: "pcm", Codec: "raw", Rate: asr.SampleRate, Bits: 16, Channel: 1,
		},
		Request: reqRequest{
			ModelName:      "bigmodel",
			EnableITN:      true,
			EnablePunc:     true,
			EnableDDC:      true,
			ShowUtterances: true,
		},
	}
	if m == modeAsync {
		r.Request.EnableNonstream = true
	}
	lang, variant := language(req.Lang)
	if m == modeNostream {
		r.Audio.Language = lang
	}
	r.Request.OutputZhVariant = variant

	var cc corpusContext
	for _, w := range hotwords(req.Hints, m) {
		cc.Hotwords = append(cc.Hotwords, hotword{Word: w})
	}
	if p := strings.TrimSpace(prompt); p != "" {
		cc.ContextType = "dialog_ctx"
		cc.ContextData = []contextItem{{Text: clipRunes(p, maxPrompt)}}
	}
	if len(cc.Hotwords) > 0 || len(cc.ContextData) > 0 {
		ctx, err := json.Marshal(cc)
		if err != nil {
			return nil, err
		}
		r.Request.Corpus = &reqCorpus{Context: string(ctx)}
	}
	return json.Marshal(r)
}

// hotwords cleans the hints (one line each, trimmed, no duplicates by
// case) and keeps as many as the endpoint takes, in order.
func hotwords(hints []string, m mode) []string {
	var out []string
	seen := map[string]bool{}
	tokens := 0
	for _, h := range hints {
		w := strings.Join(strings.Fields(h), " ")
		if w == "" {
			continue
		}
		k := strings.ToLower(w)
		if seen[k] {
			continue
		}
		if m == modeNostream {
			if len(out) == nostreamHints {
				break
			}
		} else {
			n := estimateTokens(w)
			if tokens+n > streamHintTokens {
				break
			}
			tokens += n
		}
		seen[k] = true
		out = append(out, w)
	}
	return out
}

// estimateTokens guesses a word's token count: one per CJK (or other
// non-Latin letter) rune, one per run of Latin letters or digits. It is
// rough, but errs on the side of sending fewer words.
func estimateTokens(s string) int {
	n := 0
	inWord := false
	for _, r := range s {
		switch {
		case r < utf8.RuneSelf && (unicode.IsLetter(r) || unicode.IsDigit(r)):
			if !inWord {
				n++
			}
			inWord = true
		case unicode.IsSpace(r) || unicode.IsPunct(r):
			inWord = false
		default:
			n++
			inWord = false
		}
	}
	return max(n, 1)
}

// language maps a language hint to the nostream endpoint's audio.language
// and to request.output_zh_variant (traditional Chinese). Chinese itself
// goes unset: the default model takes Chinese mixed with English.
func language(hint string) (lang, variant string) {
	h := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(hint), "_", "-"))
	switch h {
	case "zh-hant", "zh-hant-tw", "zh-tw":
		return "", "tw"
	case "zh-hant-hk", "zh-hk", "zh-mo":
		return "", "hk"
	case "zh-hant-cn":
		return "", "traditional"
	}
	base, _, _ := strings.Cut(h, "-")
	return languages[base], ""
}

// languages are the nostream endpoint's languages (audio.language) by
// base code; "zh" is left to the default model.
var languages = map[string]string{
	"en": "en-US", "ja": "ja-JP", "id": "id-ID", "es": "es-MX", "pt": "pt-BR",
	"de": "de-DE", "fr": "fr-FR", "ko": "ko-KR", "fil": "fil-PH", "tl": "fil-PH",
	"ms": "ms-MY", "th": "th-TH", "ar": "ar-SA", "it": "it-IT", "bn": "bn-BD",
	"el": "el-GR", "nl": "nl-NL", "ru": "ru-RU", "tr": "tr-TR", "vi": "vi-VN",
	"pl": "pl-PL", "ro": "ro-RO", "ne": "ne-NP", "uk": "uk-UA", "yue": "yue-CN",
}

func clipRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n])
}

// ---- Responses ----

// response is a full server response's JSON.
type response struct {
	AudioInfo struct {
		Duration int64 `json:"duration"` // ms
	} `json:"audio_info"`
	Result json.RawMessage `json:"result"`
}

type result struct {
	Text       string `json:"text"`
	Utterances []struct {
		Text     string `json:"text"`
		Definite bool   `json:"definite"`
	} `json:"utterances"`
}

// parsed is what a full server response says.
type parsed struct {
	Text       string
	DurationMS int64
}

// parseResponse reads a full server response's payload: the whole text so
// far (result.text, which with the default result_type "full" holds every
// sentence; else the utterances joined) and the audio's duration.
func parseResponse(payload []byte) (parsed, error) {
	var p parsed
	if len(strings.TrimSpace(string(payload))) == 0 {
		return p, nil
	}
	var r response
	if err := json.Unmarshal(payload, &r); err != nil {
		return p, protoErr("unreadable result: %v", err)
	}
	p.DurationMS = r.AudioInfo.Duration
	// Some documents show result as a list of results.
	var results []result
	var one result
	if len(r.Result) > 0 && r.Result[0] == '[' {
		if err := json.Unmarshal(r.Result, &results); err != nil {
			return p, protoErr("unreadable result: %v", err)
		}
	} else if len(r.Result) > 0 && string(r.Result) != "null" {
		if err := json.Unmarshal(r.Result, &one); err != nil {
			return p, protoErr("unreadable result: %v", err)
		}
		results = []result{one}
	}
	var b strings.Builder
	for _, res := range results {
		t := strings.TrimSpace(res.Text)
		if t == "" {
			for _, u := range res.Utterances {
				t += u.Text
			}
			t = strings.TrimSpace(t)
		}
		b.WriteString(t)
	}
	p.Text = b.String()
	return p, nil
}

// ---- Errors ----

// codeSuccess is the service's "OK" code.
const codeSuccess = 20000000

// codeError classifies an error frame's code and message (the docs' error
// code table, plus the words of the messages for what it leaves out).
func codeError(code uint32, payload []byte) *asr.Error {
	c, msg := asr.ErrorBody(payload)
	if msg == "" {
		msg = strings.TrimSpace(string(payload))
	}
	if c == "" || c == "0" {
		c = strconv.FormatUint(uint64(code), 10)
	}
	e := &asr.Error{Provider: ID, Code: c, Detail: asr.Clip(msg, 300)}
	low := strings.ToLower(msg)
	switch {
	case code == 45000002: // empty audio
		e.Kind = asr.KindAudio
		if e.Detail == "" {
			e.Detail = "no audio"
		}
	case code == 45000151: // bad audio format
		e.Kind = asr.KindAudio
	case code == 45000081: // waited too long for a packet
		e.Kind, e.Retryable = asr.KindNetwork, true
		if e.Detail == "" {
			e.Detail = "the service waited too long for audio"
		}
	case code == 55000031: // busy
		e.Kind, e.Retryable = asr.KindServer, true
	case code/100000 == 550: // internal errors
		e.Kind, e.Retryable = asr.KindServer, true
	case quotaMessage(low):
		e.Kind = asr.KindQuota
		e.Retryable = strings.Contains(low, "concurren") || strings.Contains(low, "rate") || strings.Contains(low, "qps")
	case authMessage(low):
		e.Kind = asr.KindAuth
	default: // 45000001 (bad parameters) and the rest
		e.Kind = asr.KindProtocol
	}
	if e.Detail == "" {
		e.Detail = fmt.Sprintf("error %s", c)
	}
	return e
}

func quotaMessage(low string) bool {
	for _, w := range []string{"quota", "not granted", "concurren", "rate limit", "too many", "qps", "arrear", "balance", "insufficient", "not activated", "not open", "expired"} {
		if strings.Contains(low, w) {
			return true
		}
	}
	return false
}

func authMessage(low string) bool {
	for _, w := range []string{"auth", "token", "access key", "app key", "appid", "app id", "grant not found", "permission", "forbidden", "unauthorized", "invalid key"} {
		if strings.Contains(low, w) {
			return true
		}
	}
	return false
}

// handshakeError refines Dial's classification of a refused handshake:
// a 403 for a resource the account has not opened ("requested resource
// not granted") is KindQuota, not KindAuth.
func handshakeError(err error) error {
	e, ok := err.(*asr.Error)
	if !ok || e.Status == 0 {
		return err
	}
	low := strings.ToLower(e.Code + " " + e.Detail)
	if (e.Status == 401 || e.Status == 403) && (strings.Contains(low, "not granted") || strings.Contains(low, "quota") || strings.Contains(low, "concurren")) {
		e.Kind = asr.KindQuota
	}
	return e
}
