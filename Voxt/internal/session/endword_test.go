package session

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestWithoutEndWord: the end word counts at the text's end, after a
// space or a punctuation mark (Voxt 1.x's rules), ignoring case and the
// punctuation after it; it goes with the punctuation before it.
func TestWithoutEndWord(t *testing.T) {
	for _, c := range []struct {
		text, word, without string
		ends                bool
	}{
		{"hello world over", "over", "hello world", true},
		{"Hello world, Over.", "over", "Hello world", true},
		{"send it now  over ", " over ", "send it now", true},
		{"ok, that's all.", "that's  all", "ok", true},
		{"我准备好了。好了", "好了", "我准备好了", true},
		{"over", "over", "", true},
		{"pushover", "over", "pushover", false},
		{"over and out", "over", "over and out", false},
		{"我准备好了", "好了", "我准备好了", false},
		{"hello over", "", "hello over", false},
	} {
		got, ends := withoutEndWord(c.text, c.word)
		if got != c.without || ends != c.ends {
			t.Errorf("withoutEndWord(%q, %q) = %q, %v; want %q, %v", c.text, c.word, got, ends, c.without, c.ends)
		}
	}
}

// TestEndWordStops: with an end word, a session whose speech model
// streams stops by itself once the live text ends with the word and
// stays so for Config.EndWordPause, and the word is not in the result.
// Without an end word, or with the word inside another one, it records
// until Stop; a model that does not stream never hears the word.
func TestEndWordStops(t *testing.T) {
	for _, c := range []struct {
		name  string
		word  string
		prov  asr.Provider
		stops bool
		raw   string
	}{
		{"the word", "over", &fakeStreamer{final: "hello world, over.", partials: []string{"hello", "hello world", "hello world over"}}, true, "hello world"},
		{"no end word", "", &fakeStreamer{final: "hello world, over.", partials: []string{"hello", "hello world", "hello world over"}}, false, "hello world, over."},
		{"inside a word", "over", &fakeStreamer{final: "a pushover", partials: []string{"a", "a pushover"}}, false, "a pushover"},
		{"no stream", "over", &fakeFile{text: "hello world over"}, false, "hello world over"},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, c.prov, nil, func(cfg *Config, s *settings.Settings) {
				s.Features.EndWord = c.word
				cfg.EndWordPause = 30 * time.Millisecond
			})
			id := h.start()
			if !c.stops {
				// note (luojiaxuan): 200 frames are 200 ms, several pauses after the last live text.
				h.speak(id, 200)
			}
			end := h.rec.wait(t)
			if end.State != Done || end.Raw != c.raw || !slices.Equal(h.text.pastes(), []string{c.raw}) {
				t.Errorf("end %s %q, pasted %q; want %q", end.State, end.Raw, h.text.pastes(), c.raw)
			}
		})
	}
}

// streamAndFile streams and transcribes whole recordings, as the local
// speech models do.
type streamAndFile struct {
	*fakeStreamer
	*fakeFile
}

func (streamAndFile) Caps() asr.Caps { return asr.Caps{Stream: true, File: true} }

// TestEndWordRetry: Retry leaves the end word out too when the speech
// model streams, as the session did; it keeps it for one that does not.
func TestEndWordRetry(t *testing.T) {
	for _, c := range []struct {
		name string
		prov asr.Provider
		raw  string
	}{
		{"streams", streamAndFile{&fakeStreamer{}, &fakeFile{text: "hello world, over."}}, "hello world"},
		{"no stream", &fakeFile{text: "hello world, over."}, "hello world, over."},
	} {
		t.Run(c.name, func(t *testing.T) {
			h, e := retryHarness(t, c.prov, nil, false)
			base := h.m.cfg.Settings
			h.m.cfg.Settings = func() settings.Settings {
				s := base()
				s.Features.EndWord = "over"
				return s
			}
			out, err := h.m.Retry(context.Background(), RetryRequest{Entry: e})
			if err != nil || out.RawText != c.raw || out.Result != c.raw {
				t.Fatalf("entry raw %q, result %q, %v; want %q", out.RawText, out.Result, err, c.raw)
			}
		})
	}
}
