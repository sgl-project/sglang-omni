package redact

import (
	"errors"
	"net/url"
	"strings"
	"testing"
)

func TestURL(t *testing.T) {
	for in, want := range map[string]string{
		"https://translation.googleapis.com/language/translate/v2?key=AIzaSyA1234567890abcdefghijklmnopqrs&q=hi":                  "https://translation.googleapis.com/language/translate/v2?key=REDACTED&q=hi",
		"https://bucket.r2.cloudflarestorage.com/a.dmg?X-Amz-Algorithm=AWS4&X-Amz-Credential=abc%2F2026&X-Amz-Signature=deadbeef": "https://bucket.r2.cloudflarestorage.com/a.dmg?X-Amz-Algorithm=AWS4&X-Amz-Credential=REDACTED&X-Amz-Signature=REDACTED",
		"socks5://user:hunter2@127.0.0.1:7897": "socks5://user:REDACTED@127.0.0.1:7897",
		"http://tok@proxy:8080":                "http://REDACTED@proxy:8080",
		"https://huggingface.co/api/models/x":  "https://huggingface.co/api/models/x",
	} {
		if got := URL(in); got != want {
			t.Errorf("URL(%q)\n got %q\nwant %q", in, got, want)
		}
	}
}

func TestText(t *testing.T) {
	e := &url.Error{Op: "Post", URL: "https://translation.googleapis.com/v2?key=AIzaSyA1234567890abcdefghijklmnopqrs", Err: errors.New("proxyconnect tcp: dial tcp 127.0.0.1:7897: connection refused")}
	got := Text(e.Error())
	if strings.Contains(got, "AIza") || !strings.Contains(got, "key=REDACTED") || !strings.Contains(got, "connection refused") {
		t.Errorf("url.Error: %q", got)
	}
	for in, bad := range map[string]string{
		"--proxy http://bob:pa55@10.0.0.1:3128 -- https://youtu.be/x": "pa55",
		"Authorization: Bearer sk-abcdefghijklmnopqrstuvwx":           "sk-abc",
		`{"api_key":"x", "token=abcdef123456"}`:                       "abcdef123456",
		"using hf_abcdefghijklmnopqrstuvwxyz0123 for gated":           "hf_abc",
		"x-goog-api-key: AIzaSyA1234567890abcdefghijklmnopqrs":        "AIza",
		"see https://example.com/x?access_token=zzz.":                 "zzz",
	} {
		if got := Text(in); strings.Contains(got, bad) {
			t.Errorf("Text(%q) = %q", in, got)
		}
	}
	plain := "loaded mlx-community/Qwen3-ASR-0.6B-4bit in 2.1 s; key frames 12"
	if got := Text(plain); got != plain {
		t.Errorf("plain text changed: %q", got)
	}
}

func TestError(t *testing.T) {
	base := errors.New("boom")
	err := Error(&url.Error{Op: "Get", URL: "https://x/?token=abc", Err: base})
	if strings.Contains(err.Error(), "abc") || !errors.Is(err, base) {
		t.Errorf("%v", err)
	}
	if Error(nil) != nil || Error(base) != base {
		t.Error("passthrough")
	}
}
