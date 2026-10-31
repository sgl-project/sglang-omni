package asr_test

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
)

func TestWAVAndFrames(t *testing.T) {
	pcm := make([]byte, 3200*3+100)
	for i := range pcm {
		pcm[i] = byte(i)
	}
	got, rate, ch, err := asr.ReadWAV(asr.WAV(pcm, 0))
	if err != nil || rate != 16000 || ch != 1 || !bytes.Equal(got, pcm) {
		t.Fatalf("round trip: %v %d %d %d", err, rate, ch, len(got))
	}
	if _, _, _, err := asr.ReadWAV([]byte("RIFF0000WAVE")); err == nil {
		t.Error("no data chunk accepted")
	}
	if d := asr.Duration(asr.Silence(2*time.Second), 0); d != 2*time.Second {
		t.Errorf("duration %v", d)
	}
	fr := asr.Frames(pcm, 100*time.Millisecond, 0)
	if len(fr) != 4 || len(fr[0]) != 3200 || len(fr[3]) != 100 {
		t.Errorf("frames %d", len(fr))
	}
	r := asr.Rechunk{Size: 6400}
	var n int
	for _, f := range fr {
		for _, full := range r.Add(f) {
			n++
			if len(full) != 6400 {
				t.Errorf("frame of %d", len(full))
			}
		}
	}
	if rest := r.Rest(); n != 1 || len(rest) != 3200+100 {
		t.Errorf("%d full, rest %d", n, len(rest))
	}
}

func TestHTTPErrorKinds(t *testing.T) {
	for _, tc := range []struct {
		status int
		msg    string
		kind   asr.Kind
		retry  bool
	}{
		{401, "invalid key", asr.KindAuth, false},
		{403, "Arrearage: the account is overdue", asr.KindQuota, false},
		{403, "forbidden", asr.KindAuth, false},
		{429, "rate limit", asr.KindQuota, true},
		{429, "insufficient_quota", asr.KindQuota, false},
		{413, "too big", asr.KindAudio, false},
		{400, "audio duration too long", asr.KindAudio, false},
		{400, "model foo does not exist", asr.KindConfig, false},
		{400, "bad field", asr.KindProtocol, false},
		{404, "", asr.KindConfig, false},
		{502, "", asr.KindServer, true},
	} {
		e := asr.HTTPError("p", tc.status, "", tc.msg)
		if e.Kind != tc.kind || e.Retryable != tc.retry {
			t.Errorf("%d %q: %s retry %v", tc.status, tc.msg, e.Kind, e.Retryable)
		}
	}
	e := asr.HTTPError("p", 401, "", "x")
	if !errors.Is(e, &asr.Error{Kind: asr.KindAuth}) || errors.Is(e, &asr.Error{Kind: asr.KindQuota}) {
		t.Error("Is by kind")
	}
	if c, m := asr.ErrorBody([]byte(`{"header":{"event":"task-failed","error_code":"InvalidApiKey","error_message":"Invalid API-key provided."}}`)); c != "InvalidApiKey" || m != "Invalid API-key provided." {
		t.Errorf("dashscope body %q %q", c, m)
	}
	if c, m := asr.ErrorBody([]byte(`{"code":45000001,"message":"bad"}`)); c != "45000001" || m != "bad" {
		t.Errorf("code body %q %q", c, m)
	}
}

// fakeStreamer echoes how many bytes it got as partials.
type fakeStreamer struct{ fail error }

func (fakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (f fakeStreamer) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if f.fail != nil {
		return nil, f.fail
	}
	s := &fakeStream{ev: make(chan asr.Event, 100)}
	return s, nil
}

type fakeStream struct {
	ev chan asr.Event
	n  int
}

func (s *fakeStream) Send(p []byte) error {
	s.n += len(p)
	s.ev <- asr.Event{Kind: asr.EventPartial, Text: strings.Repeat("a", s.n/3200)}
	return nil
}
func (s *fakeStream) Finish() error {
	s.ev <- asr.Event{Kind: asr.EventFinal, Text: "done", Result: asr.Result{Model: "m"}}
	close(s.ev)
	return nil
}
func (s *fakeStream) Close()                   {}
func (s *fakeStream) Events() <-chan asr.Event { return s.ev }

func TestStreamAll(t *testing.T) {
	var partials []string
	res, err := asr.StreamAll(context.Background(), fakeStreamer{}, asr.StreamRequest{}, asr.Silence(500*time.Millisecond),
		asr.Feed{OnPartial: func(s string) { partials = append(partials, s) }})
	if err != nil || res.Text != "done" || res.Model != "m" || len(partials) != 5 || partials[4] != "aaaaa" {
		t.Fatalf("%v %+v %q", err, res, partials)
	}
	start := time.Now()
	if _, err := asr.StreamAll(context.Background(), fakeStreamer{}, asr.StreamRequest{}, asr.Silence(300*time.Millisecond), asr.Feed{Pace: 2}); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d < 100*time.Millisecond {
		t.Errorf("paced feed took %v", d)
	}
	want := &asr.Error{Kind: asr.KindAuth}
	if _, err := asr.Recognize(context.Background(), fakeStreamer{fail: want}, nil, "", nil, "", asr.Feed{}); err != want {
		t.Errorf("open error %v", err)
	}
}

func TestNoSpeech(t *testing.T) {
	for _, tc := range []struct {
		code, detail string
		want         bool
	}{
		{"", "no speech found", true},
		{"request_params_invalid", "No speech found", true},
		{"NO_VALID_AUDIO_ERROR", "", true},
		{"", "no valid audio", true},
		{"", "Speech not detected in the audio", true},
		{"", "invalid model", false},
		{"", "audio too long", false},
		{"InternalError", "", false},
	} {
		if got := asr.NoSpeech(tc.code, tc.detail); got != tc.want {
			t.Errorf("NoSpeech(%q, %q) = %v", tc.code, tc.detail, got)
		}
	}
}
