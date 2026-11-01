package omni

import (
	"encoding/binary"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
)

// sampleRate is the runtime's (and VoxtNew's) audio rate.
const sampleRate = 16000

// tokenBudget is a Final's budget of new tokens for audio of d (Voxt's
// postStopFinalMaxTokens).
func (t *ASRTuning) tokenBudget(d time.Duration) int {
	n := int(math.Ceil(max(d.Seconds(), 0)*t.TokensPerSecond)) + t.ExtraTokens
	return min(max(n, t.MinTokens), t.MaxTokens)
}

// languageHint is what the model gets for the language code code ("zh",
// "zh-Hans", "en-US"): the model's own code when it knows the language,
// else "" (it detects). Voxt's Qwen3-ASR policy: the user's main
// language is always the hint when the model has it.
func (s Spec) languageHint(code string) string {
	if strings.TrimSpace(code) == "" {
		return ""
	}
	c, ok := lang.Match(s.Languages, code)
	if !ok {
		return ""
	}
	return c
}

// bias is a Final request's context and the dictionary's terms in it.
type bias struct {
	context string
	terms   []string
}

// contextBias is the Final's context: prompt (free text), then the
// dictionary's terms one per line in the order given (the most recently
// changed first), at most t.ContextTerms terms and t.ContextChars
// characters (Voxt's asrPromptTermsText; a first term over the limit is
// kept alone).
func (t *ASRTuning) contextBias(hints []string, prompt string) bias {
	var terms []string
	seen := map[string]bool{}
	chars := 0
	for _, h := range hints {
		h = strings.Join(strings.Fields(h), " ")
		key := strings.ToLower(h)
		if h == "" || seen[key] {
			continue
		}
		next := chars + utf8.RuneCountInString(h)
		if len(terms) > 0 {
			next++
		}
		if len(terms) > 0 && next > t.ContextChars {
			break
		}
		seen[key] = true
		terms = append(terms, h)
		chars = next
		if len(terms) >= t.ContextTerms || chars >= t.ContextChars {
			break
		}
	}
	parts := []string{}
	if p := strings.TrimSpace(prompt); p != "" {
		parts = append(parts, p)
	}
	if len(terms) > 0 {
		parts = append(parts, strings.Join(terms, "\n"))
	}
	return bias{context: strings.Join(parts, "\n"), terms: terms}
}

// chunk is a piece of a recording for one Final request.
type chunk struct {
	pcm    []byte
	offset time.Duration
}

// splitChunks cuts PCM16 mono at 16 kHz into requests of at most
// t.ChunkSeconds, each cut at the quietest t.CutWindow within t.CutSearch
// of the chunk's nominal end (MLXAudio's splitter, which Voxt's Qwen3-ASR
// used: past the end too, unless that could pass the server's limit), at
// least a second after the chunk's start; chunks shorter than
// t.MinChunkSeconds are padded with silence.
func (t *ASRTuning) splitChunks(pcm []byte) []chunk {
	total := len(pcm) / 2
	minSamples := int(t.MinChunkSeconds * sampleRate)
	padded := func(from, to int) []byte {
		b := pcm[from*2 : to*2]
		if to-from < minSamples {
			b = append(append([]byte(nil), b...), make([]byte, (minSamples-(to-from))*2)...)
		}
		return b
	}
	at := func(s int) time.Duration { return time.Duration(s) * time.Second / sampleRate }
	maxSamples := int(t.ChunkSeconds * sampleRate)
	if total <= maxSamples {
		return []chunk{{pcm: padded(0, total)}}
	}
	search := int(t.CutSearch.Seconds() * sampleRate)
	window := max(int(t.CutWindow.Seconds()*sampleRate), 1)
	pastEnd := t.ChunkSeconds+t.CutSearch.Seconds() <= 1200
	var out []chunk
	for start := 0; start < total; {
		end := min(start+maxSamples, total)
		if end >= total {
			out = append(out, chunk{pcm: padded(start, total), offset: at(start)})
			break
		}
		from := max(start, end-search)
		to := end
		if pastEnd {
			to = min(total, end+search)
		}
		cut := end
		if to-from > window {
			cut = quietest(pcm, from, to, window)
		}
		cut = max(cut, start+sampleRate)
		out = append(out, chunk{pcm: padded(start, min(cut, total)), offset: at(start)})
		start = cut
	}
	return out
}

// quietest is the middle of the quietest window samples of PCM16 pcm
// between samples from and to (a range longer than the window).
func quietest(pcm []byte, from, to, window int) int {
	sample := func(i int) float64 {
		return float64(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768
	}
	var sum float64
	for i := from; i < from+window; i++ {
		sum += sample(i) * sample(i)
	}
	best, bestAt := sum, 0
	for off := 1; off <= to-from-window; off++ {
		leaving, entering := sample(from+off-1), sample(from+off+window-1)
		sum += entering*entering - leaving*leaving
		if sum < best {
			best, bestAt = sum, off
		}
	}
	return from + bestAt + window/2
}

// unspaced are the scripts written without spaces between words (the
// server's list): a join puts a space only between two other characters.
var unspaced = [][2]rune{
	{0x0E00, 0x0EFF}, {0x1000, 0x109F}, {0x1780, 0x17FF}, {0x2E80, 0x303F},
	{0x3040, 0x30FF}, {0x3400, 0x9FFF}, {0xF900, 0xFAFF}, {0xFF00, 0xFFEF},
	{0x20000, 0x2FA1F},
}

func spaced(r rune) bool {
	if r == utf8.RuneError || strings.ContainsRune(" \t\n\r\v\f 　", r) {
		return false
	}
	for _, u := range unspaced {
		if r >= u[0] && r <= u[1] {
			return false
		}
	}
	return true
}

// joinParts joins transcript pieces as the server joins segments: each
// trimmed, a space only between two characters of spaced scripts.
func joinParts(parts []string) string {
	var b strings.Builder
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if b.Len() > 0 {
			last, _ := utf8.DecodeLastRuneInString(b.String())
			first, _ := utf8.DecodeRuneInString(p)
			if spaced(last) && spaced(first) {
				b.WriteByte(' ')
			}
		}
		b.WriteString(p)
	}
	return b.String()
}

func itoa(v int) string { return strconv.Itoa(v) }

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
