package asr

import (
	"bytes"
	"context"
	"errors"
	"testing"
)

func loud(n int) []byte { return bytes.Repeat([]byte{0x00, 0x40}, n/2) }

func TestSplitPCM(t *testing.T) {
	// 10 s of sound with a 300 ms pause at 8.5 s; pieces of at most 9 s.
	pcm := loud(10 * BytesPerSecond)
	pause := 85 * BytesPerSecond / 10
	clear(pcm[pause : pause+3*BytesPerSecond/10])
	got := SplitPCM(pcm, 0, 9*BytesPerSecond)
	if len(got) != 2 || len(got[0]) <= pause || len(got[0]) >= pause+3*BytesPerSecond/10 || !bytes.Equal(append(got[0], got[1]...), pcm) {
		t.Errorf("pieces %d, first %d bytes; want the cut in the pause at %d", len(got), len(got[0]), pause)
	}
	for _, c := range []struct {
		name     string
		pcm      []byte
		max, out int
	}{
		{"empty", nil, 100, 1},
		{"fits", loud(100), 100, 1},
		{"no pause: cut at the limit", loud(10 * BytesPerSecond), 4 * BytesPerSecond, 3},
		{"limit below the window, odd", loud(1000), 101, 10},
		{"no limit", loud(100), 0, 1},
	} {
		got := SplitPCM(c.pcm, SampleRate, c.max)
		var joined []byte
		for _, p := range got {
			if c.max > 0 && len(p) > c.max || len(p)%2 != 0 {
				t.Errorf("%s: piece of %d bytes (max %d)", c.name, len(p), c.max)
			}
			joined = append(joined, p...)
		}
		if len(got) != c.out || !bytes.Equal(joined, c.pcm) {
			t.Errorf("%s: %d pieces, want %d; lossless %v", c.name, len(got), c.out, bytes.Equal(joined, c.pcm))
		}
	}
}

func TestTranscribeInPieces(t *testing.T) {
	var sent []int
	answers := []string{"Hello there.", " ", "你好。", "and bye"}
	one := func(_ context.Context, r FileRequest) (Result, error) {
		sent = append(sent, len(r.Audio))
		return Result{Text: answers[len(sent)-1], Lang: "en", Model: "m"}, nil
	}
	res, err := TranscribeInPieces(context.Background(), FileRequest{Audio: loud(4000), Format: "pcm", Prompt: "p"}, 44+1000, one)
	if err != nil || res.Text != "Hello there.你好。and bye" || res.Lang != "en" || res.Model != "m" || len(sent) != 4 {
		t.Errorf("%+v %v, sent %v", res, err, sent)
	}
	sent = nil
	if _, err := TranscribeInPieces(context.Background(), FileRequest{Audio: loud(4000), Format: "wav"}, 1000, one); err != nil || len(sent) != 1 {
		t.Errorf("wav: %v, sent %v", err, sent)
	}
	boom := errors.New("boom")
	calls := 0
	if _, err := TranscribeInPieces(context.Background(), FileRequest{Audio: loud(4000), Format: "pcm"}, 1044, func(context.Context, FileRequest) (Result, error) {
		calls++
		return Result{}, boom
	}); !errors.Is(err, boom) || calls != 1 {
		t.Errorf("error: %v after %d calls", err, calls)
	}
}
