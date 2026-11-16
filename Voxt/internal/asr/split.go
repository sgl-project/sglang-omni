package asr

import (
	"context"
	"encoding/binary"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// SplitPCM cuts PCM16 mono at rate Hz (0: SampleRate) into pieces of at
// most max bytes. Each cut falls in the quietest 100 ms of the piece's
// last quarter, so it lands in a pause rather than inside a word.
func SplitPCM(pcm []byte, rate, max int) [][]byte {
	if rate <= 0 {
		rate = SampleRate
	}
	max &^= 1
	if max <= 0 {
		return [][]byte{pcm}
	}
	win := rate / 10 * 2
	var out [][]byte
	for len(pcm) > max {
		cut, best := max, int64(-1)
		for end := max; end-win >= max*3/4; end -= win / 2 {
			if e := energy(pcm[end-win : end]); best < 0 || e < best {
				cut, best = (end-win/2)&^1, e
			}
		}
		out = append(out, pcm[:cut])
		pcm = pcm[cut:]
	}
	return append(out, pcm)
}

func energy(pcm []byte) int64 {
	var e int64
	for i := 0; i+1 < len(pcm); i += 2 {
		v := int64(int16(binary.LittleEndian.Uint16(pcm[i:])))
		e += v * v
	}
	return e
}

// TranscribeInPieces transcribes a PCM recording whose WAV upload would
// pass maxWAV bytes (or req.MaxBytes, when smaller): it splits it with SplitPCM, sends the pieces one by
// one through one, and joins their text. Smaller recordings and other
// formats go to one as they are.
func TranscribeInPieces(ctx context.Context, req FileRequest, maxWAV int, one func(context.Context, FileRequest) (Result, error)) (Result, error) {
	const header = 44
	if req.MaxBytes > header && req.MaxBytes < maxWAV {
		maxWAV = req.MaxBytes
	}
	if req.Format != "pcm" || header+len(req.Audio) <= maxWAV {
		return one(ctx, req)
	}
	start := time.Now()
	var all Result
	var texts []string
	for _, piece := range SplitPCM(req.Audio, req.SampleRate, maxWAV-header) {
		r := req
		r.Audio = piece
		res, err := one(ctx, r)
		if err != nil {
			return Result{}, err
		}
		if t := strings.TrimSpace(res.Text); t != "" {
			texts = append(texts, t)
		}
		if all.Lang == "" {
			all.Lang = res.Lang
		}
		all.Duration += res.Duration
		all.Model = res.Model
	}
	all.Text = JoinText(texts)
	all.Latency = time.Since(start)
	return all, nil
}

// JoinText joins sentences: with a space between two that meet in a
// spaced script (Latin, Cyrillic…), with nothing where either side is
// Chinese, Japanese or Korean, or full-width punctuation. (Voxt joins
// with a space everywhere, which puts spaces between Chinese sentences.)
func JoinText(parts []string) string {
	var b strings.Builder
	for i, p := range parts {
		if i > 0 {
			last, _ := utf8.DecodeLastRuneInString(b.String())
			first, _ := utf8.DecodeRuneInString(p)
			if !wide(last) && !wide(first) && !unicode.IsSpace(last) {
				b.WriteByte(' ')
			}
		}
		b.WriteString(p)
	}
	return b.String()
}

// wide reports a rune of a script written without spaces, or full-width
// punctuation.
func wide(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar) ||
		(r >= 0x3000 && r <= 0x303F) || (r >= 0xFF00 && r <= 0xFFEF)
}
