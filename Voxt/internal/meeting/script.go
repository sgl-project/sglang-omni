package meeting

import (
	"context"
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// inScript writes recognised Chinese in script, "zh-Hans" or "zh-Hant"
// (TrackASR.Script; "" keeps the speech model's), as dictation does
// (#4534): with the language left to it, Whisper mixes the scripts in a
// sentence (11-25: 嵐 and 為 in a Simplified meeting). Text in another
// language stays: said is the language the model reported ("" when it
// did not), and kana marks Japanese either way.
func inScript(text, said, script string) string {
	if script == "" || text == "" {
		return text
	}
	if c := lang.Canonical(said); c != "" && c != "zh" && c != "yue" {
		return text
	}
	if strings.ContainsFunc(text, func(c rune) bool { return unicode.In(c, unicode.Hiragana, unicode.Katakana) }) {
		return text
	}
	return lang.RenderChineseScript(text, script)
}

// scriptedFile and scriptedDiarizer write their results in script.
type scriptedFile struct {
	asr.FileTranscriber
	script string
}

func (s scriptedFile) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	res, err := s.FileTranscriber.Transcribe(ctx, req)
	return resultInScript(res, s.script), err
}

type scriptedDiarizer struct {
	asr.Diarizer
	script string
}

func (s scriptedDiarizer) Diarize(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	res, err := s.Diarizer.Diarize(ctx, req)
	return resultInScript(res, s.script), err
}

func resultInScript(res asr.Result, script string) asr.Result {
	res.Text = inScript(res.Text, res.Lang, script)
	if len(res.Segments) > 0 {
		segs := make([]asr.SpeakerSegment, len(res.Segments))
		for i, s := range res.Segments {
			s.Text = inScript(s.Text, res.Lang, script)
			segs[i] = s
		}
		res.Segments = segs
	}
	return res
}
