package meeting

import (
	"context"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// A meeting writes Chinese in the user's script, as dictation does; other
// languages, and Japanese kanji, stay as the model wrote them (11-25:
// 嵐, 為 and 覺 in a Simplified meeting).
func TestInScript(t *testing.T) {
	for _, c := range []struct{ text, said, script, want string }{
		{"你陪我為何覺得", "", "zh-Hans", "你陪我为何觉得"},
		{"你陪我为何觉得", "", "zh-Hant", "你陪我為何覺得"},
		{"你陪我為何覺得", "zh", "zh-Hans", "你陪我为何觉得"},
		{"為何覺得", "", "", "為何覺得"},
		{"ご視聴ありがとうございました", "", "zh-Hans", "ご視聴ありがとうございました"},
		{"會議", "ja", "zh-Hans", "會議"},
		{"Thank you", "en", "zh-Hans", "Thank you"},
	} {
		got := inScript(c.text, c.said, c.script)
		if c.script == "zh-Hant" {
			// Core Foundation writes 為; the non-Mac dictionary writes 爲.
			// Both are Traditional, so accept either without allowing 为.
			got = strings.ReplaceAll(got, "爲", "為")
		}
		if got != c.want {
			t.Errorf("inScript(%q, said %q, %q) = %q, want %q", c.text, c.said, c.script, got, c.want)
		}
	}
}

type traditionalFile struct{}

func (traditionalFile) Caps() asr.Caps { return asr.Caps{File: true} }
func (traditionalFile) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	return asr.Result{Text: "會議結束了", Segments: []asr.SpeakerSegment{{Text: "為什麼"}}}, nil
}

func TestScriptedFile(t *testing.T) {
	res, err := scriptedFile{traditionalFile{}, "zh-Hans"}.Transcribe(context.Background(), asr.FileRequest{})
	if err != nil || res.Text != "会议结束了" || res.Segments[0].Text != "为什么" {
		t.Fatalf("got %+v, %v", res, err)
	}
}
