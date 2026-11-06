package omni

// The speech models beside Qwen3-ASR, asked as Voxt 1.x asked their
// servers (MLXDetachedInference.swift): one Final request per recording,
// which the server cuts itself (Whisper's 30 s windows, MOSS's 1200 s
// chunks, Parakeet's chunks, SenseVoice's speech segments).

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// Voxt's requests of the other models.
const (
	// mossPlainPrompt is MOSS's instruction for dictation (Voxt's
	// moss-plain-text.txt); the dictionary's terms follow as hot words.
	mossPlainPrompt = "Transcribe the audio as plain text without timestamps or speaker labels."
	// parakeetChunkSeconds: 1199 or more is the whole recording at once,
	// cut by the server.
	parakeetChunkSeconds = "1200"
	// SenseVoice decodes a recording up to senseVoiceDirect in one pass;
	// a longer one is cut by the server's Silero VAD (the speech model's
	// VAD folder) into speech chunks.
	senseVoiceDirect = 30 * time.Second
)

// senseVoiceSegments are Voxt's VAD settings of SenseVoice's long
// recordings: chunks of at most 24 s overlapping by 0.35 s.
var senseVoiceSegments = map[string]string{
	"vad_threshold": "0.5", "vad_min_speech_ms": "220", "vad_min_silence_ms": "420", "vad_speech_pad_ms": "180",
	"vad_max_chunk_seconds": "24", "vad_chunk_overlap_seconds": "0.35",
}

// finalWhole is the Final of a model whose server cuts the recording
// itself: one request with the model's fields; MOSS's text without its
// tags.
func (a *ASR) finalWhole(ctx context.Context, l *lease, pcm []byte, language string, b bias) (string, string, error) {
	d := asr.Duration(pcm, sampleRate)
	req := finalRequest{pcm: pcm, language: language, maxTokens: a.spec.ASR.tokenBudget(d)}
	switch a.spec.Kind {
	case kindMoss:
		req.context = mossPrompt(b.terms)
	case kindParakeet:
		req.fields = map[string]string{"chunk_duration": parakeetChunkSeconds}
	case kindSenseVoice:
		req.fields = map[string]string{"use_itn": "false"}
		if a.vad != nil && d > senseVoiceDirect {
			for k, v := range senseVoiceSegments {
				req.fields[k] = v
			}
			req.fields["vad_model_directory"] = a.vad.dir
		}
	}
	r, err := transcribe(ctx, l, req, a.rt.cfg.StallTimeout)
	if err != nil {
		return "", "", err
	}
	if a.spec.Kind == kindMoss {
		r.text = mossText(r.text)
	}
	return strings.TrimSpace(r.text), r.language, nil
}

// mossPrompt is MOSS's instruction with the dictionary's terms as hot
// words (Voxt's resolvedPrompt).
func mossPrompt(terms []string) string {
	if len(terms) == 0 {
		return mossPlainPrompt
	}
	return mossPlainPrompt + " Hotwords: " + strings.Join(terms, ", ")
}

// MOSS's tags (Voxt's removingStructuredTags): timestamps, speaker
// labels, acoustic events ("[sniff]"), and the unfinished tag a live
// decode may end with.
var (
	mossTimestamp = regexp.MustCompile(`\[\d+(?:[.,]\d+)?\]`)
	mossTags      = regexp.MustCompile(`\[S\d+\]\s*|\[[a-z][a-z0-9 _-]{0,31}\]\s*|\[S[0-9O]{0,3}\s+`)
	mossCutTag    = regexp.MustCompile(`\[(?:[\d.,]*|S[0-9O]*|[a-z][a-z0-9 _-]*)$`)
)

// mossText is MOSS's output as plain text (Voxt's plainText rendering):
// tags removed, the segments joined.
func mossText(raw string) string {
	s := mossCutTag.ReplaceAllString(mossTags.ReplaceAllString(mossTimestamp.ReplaceAllString(raw, "\n"), ""), "")
	return joinParts(strings.Split(s, "\n"))
}

// joinDirs is a folder of links to the files of dirs, the first's winning
// a name they share: a server that reads its tokenizer beside its weights
// (whisper_server) gets Whisper's two repos as one. The folder is named by
// dirs, so that opening the model again reuses it.
func joinDirs(dirs []string) (string, error) {
	sum := sha256.Sum256([]byte(strings.Join(dirs, "\x00")))
	out := filepath.Join(os.TempDir(), "voxtnew-omni-"+hex.EncodeToString(sum[:8]))
	if err := os.MkdirAll(out, 0o700); err != nil {
		return "", err
	}
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			return "", err
		}
		for _, e := range entries {
			err := os.Symlink(filepath.Join(d, e.Name()), filepath.Join(out, e.Name()))
			if err != nil && !errors.Is(err, os.ErrExist) {
				return "", err
			}
		}
	}
	return out, nil
}
