package meeting

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type chineseFileASR struct{ text string }

func (f *chineseFileASR) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *chineseFileASR) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	return asr.Result{Text: f.text}, nil
}

func TestTranscribeChunksChineseScript(t *testing.T) {
	for _, test := range []struct {
		name, languageTag, source, want string
	}{
		{name: "simplified", languageTag: "zh-Hans", source: "我們今天討論 Qwen 的會議轉寫效果。", want: "我们今天讨论 Qwen 的会议转写效果。"},
		{name: "traditional", languageTag: "zh-Hant", source: "我们今天讨论 Qwen 的会议转写效果。", want: "我們今天討論 Qwen 的會議轉寫效果。"},
		{name: "unspecified script", languageTag: "zh", source: "我們今天討論會議轉寫。", want: "我們今天討論會議轉寫。"},
	} {
		t.Run(test.name, func(t *testing.T) {
			rate := asr.SampleRate
			segs, err := transcribeChunks(context.Background(), SourceMic, &chineseFileASR{text: test.source}, asr.FileRequest{Lang: test.languageTag}, rate, make([]byte, rate*2), []Chunk{{Start: 0, End: rate}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(segs) != 1 || segs[0].Text != test.want {
				t.Fatalf("segments %+v, want text %q", segs, test.want)
			}
		})
	}
}

func TestTranscribeChunksSuppressesCrossScriptGuidance(t *testing.T) {
	prompt := "请准确记录会议中的完整发言内容，保留技术名词和产品名称，并按照用户实际表达输出，不要自行补充任何解释或摘要。"
	for _, test := range []struct {
		name, source string
		request      asr.FileRequest
	}{
		{name: "prompt", source: "請準確記錄會議中的完整發言內容，保留技術名詞和產品名稱，並按照用戶實際表達輸出，不要自行補充任何解釋或摘要。", request: asr.FileRequest{Lang: "zh-Hans", Prompt: prompt}},
		{name: "hints", source: "語音識別，會議轉寫", request: asr.FileRequest{Lang: "zh-Hans", Hints: []string{"语音识别", "会议转写"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			rate := asr.SampleRate
			segs, err := transcribeChunks(context.Background(), SourceMic, &chineseFileASR{text: test.source}, test.request, rate, make([]byte, rate*2), []Chunk{{Start: 0, End: rate}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(segs) != 0 {
				t.Fatalf("guidance became transcript: %+v", segs)
			}
		})
	}

	rate := asr.SampleRate
	segs, err := transcribeChunks(context.Background(), SourceMic, &chineseFileASR{text: "我們討論語音識別的準確率。"}, asr.FileRequest{Lang: "zh-Hans", Hints: []string{"语音识别"}}, rate, make([]byte, rate*2), []Chunk{{Start: 0, End: rate}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 || segs[0].Text != "我们讨论语音识别的准确率。" {
		t.Fatalf("ordinary speech was suppressed or not rendered: %+v", segs)
	}
}

func TestDiarizeTrackChineseScript(t *testing.T) {
	diarizer := &seqDiarizer{results: []asr.Result{{Segments: []asr.SpeakerSegment{{End: time.Second, Speaker: "S01", Text: "我們今天討論會議轉寫。"}}}}}
	rate := asr.SampleRate
	segs, err := diarizeTrack(context.Background(), SourceSystem, diarizer, asr.FileRequest{Lang: "zh-Hans"}, rate, make([]byte, rate*2), []vad.Segment{{Start: 0, End: rate}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 || segs[0].Text != "我们今天讨论会议转写。" {
		t.Fatalf("segments %+v", segs)
	}
}
