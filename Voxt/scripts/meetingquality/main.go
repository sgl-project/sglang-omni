package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/sgl-project/sglang-omni/Voxt/internal/eval"
)

type clipsFile struct {
	Clips []struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	} `json:"clips"`
	Long []struct {
		ID  string   `json:"id"`
		Gap float64  `json:"gap"`
		Of  []string `json:"of"`
	} `json:"long"`
}

type clipInfo struct {
	ID         string
	Text       string
	Start, End float64
}

type token struct {
	Text string
	Clip int
}

type operation struct {
	Kind     string
	Ref, Hyp string
	Clip     int
}

type resultFile struct {
	AudioDurationMs int `json:"audio_duration_ms"`
	Stages          []struct {
		Name    string `json:"name"`
		Index   int    `json:"index"`
		AudioMs int    `json:"audio_ms"`
	} `json:"stages"`
	Transcript struct {
		Text string `json:"text"`
	} `json:"transcript"`
}

type requestWindow struct {
	Request       int     `json:"request_index"`
	TimelineStart float64 `json:"timeline_start_s"`
	TimelineEnd   float64 `json:"timeline_end_s"`
	NativeStart   float64 `json:"native_audio_start_s"`
	NativeEnd     float64 `json:"native_audio_end_s"`
	Substitutions int     `json:"substitutions"`
	Deletions     int     `json:"deletions"`
	Insertions    int     `json:"insertions"`
}

type output struct {
	Method         string          `json:"method"`
	CapSeconds     int             `json:"cap_seconds"`
	AudioDuration  int             `json:"audio_duration_ms"`
	Grouping       string          `json:"grouping_rule"`
	RequestWindows []requestWindow `json:"requests"`
}

func tokensWithClips(text string, clips []clipInfo) []token {
	values := eval.Tokens(text)
	out := make([]token, len(values))
	for i, value := range values {
		out[i] = token{Text: value, Clip: -1}
	}
	index := 0
	for ci, clip := range clips {
		for range eval.Tokens(clip.Text) {
			if index < len(out) {
				out[index].Clip = ci
			}
			index++
		}
	}
	return out
}

func operations(ref, hyp []token) []operation {
	n, m := len(ref), len(hyp)
	d := make([][]int, n+1)
	for i := range d {
		d[i] = make([]int, m+1)
		d[i][0] = i
	}
	for j := 0; j <= m; j++ {
		d[0][j] = j
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			cost := 1
			if ref[i-1].Text == hyp[j-1].Text {
				cost = 0
			}
			d[i][j] = min(d[i-1][j-1]+cost, d[i-1][j]+1, d[i][j-1]+1)
		}
	}
	var out []operation
	for i, j := n, m; i > 0 || j > 0; {
		op := operation{}
		switch {
		case i > 0 && j > 0 && d[i][j] == d[i-1][j-1] && ref[i-1].Text == hyp[j-1].Text:
			i, j = i-1, j-1
			continue
		case i > 0 && j > 0 && d[i][j] == d[i-1][j-1]+1:
			op.Kind = "sub"
			op.Ref, op.Hyp, op.Clip = ref[i-1].Text, hyp[j-1].Text, ref[i-1].Clip
			i, j = i-1, j-1
		case i > 0 && d[i][j] == d[i-1][j]+1:
			op.Kind = "del"
			op.Ref, op.Clip = ref[i-1].Text, ref[i-1].Clip
			i--
		default:
			op.Kind = "ins"
			op.Hyp, op.Clip = hyp[j-1].Text, hyp[j-1].Clip
			j--
		}
		out = append(out, op)
	}
	return out
}

func requestWindows(result resultFile, capSeconds int) ([]requestWindow, error) {
	if capSeconds <= 0 {
		return nil, fmt.Errorf("cap seconds must be positive")
	}
	var windows []requestWindow
	for _, stage := range result.Stages {
		if stage.Name != "diarization" {
			continue
		}
		request := len(windows) + 1
		timelineStart := float64(request-1) * float64(capSeconds-1)
		timelineEnd := min(timelineStart+float64(capSeconds), float64(result.AudioDurationMs)/1000)
		nativeStart := max(0, timelineStart-0.12)
		windows = append(windows, requestWindow{
			Request: request, TimelineStart: timelineStart, TimelineEnd: timelineEnd,
			NativeStart: nativeStart, NativeEnd: nativeStart + float64(stage.AudioMs)/1000,
		})
	}
	if len(windows) == 0 {
		return nil, fmt.Errorf("result has no diarization stages")
	}
	return windows, nil
}

func requestAt(windows []requestWindow, seconds float64) (int, error) {
	request := 0
	for _, window := range windows {
		if seconds >= window.TimelineStart {
			request = window.Request
		}
	}
	if request == 0 || seconds > windows[len(windows)-1].NativeEnd {
		return 0, fmt.Errorf("clip at %.3f s is outside the %d-request timeline", seconds, len(windows))
	}
	return request, nil
}

func main() {
	if len(os.Args) != 3 {
		panic("usage: go run ./scripts/meetingquality CAP_SECONDS RESULT_JSON")
	}
	var capSeconds int
	if _, err := fmt.Sscanf(os.Args[1], "%d", &capSeconds); err != nil {
		panic(err)
	}
	var input clipsFile
	readJSON("testdata/eval/clips.json", &input)
	textByID := map[string]string{}
	for _, clip := range input.Clips {
		textByID[clip.ID] = clip.Text
	}
	of := input.Long[0].Of
	for len(of) < 81 {
		of = append(of, input.Long[0].Of...)
	}
	of = of[:81]
	clips := make([]clipInfo, len(of))
	start := 0.0
	for i, id := range of {
		data, err := os.ReadFile("testdata/eval/" + id + ".wav")
		if err != nil {
			panic(err)
		}
		duration := float64(len(data)/2) / 16000
		clips[i] = clipInfo{ID: id, Text: textByID[id], Start: start, End: start + duration}
		start += duration + input.Long[0].Gap
	}
	var result resultFile
	readJSON(os.Args[2], &result)
	windows, err := requestWindows(result, capSeconds)
	if err != nil {
		panic(err)
	}
	refText := ""
	for i, clip := range clips {
		if i > 0 {
			refText += "\n"
		}
		refText += clip.Text
	}
	ref := tokensWithClips(refText, clips)
	hyp := tokensWithClips(result.Transcript.Text, clips)
	for _, op := range operations(ref, hyp) {
		if op.Clip < 0 {
			op.Clip = len(clips) - 1
		}
		request, err := requestAt(windows, clips[op.Clip].Start)
		if err != nil {
			panic(err)
		}
		switch op.Kind {
		case "sub":
			windows[request-1].Substitutions++
		case "del":
			windows[request-1].Deletions++
		case "ins":
			windows[request-1].Insertions++
		}
	}
	encoded, err := json.MarshalIndent(output{
		Method:     "Levenshtein traceback with eval.Tokens and clip timings from the public eval WAVs",
		CapSeconds: capSeconds, AudioDuration: result.AudioDurationMs,
		Grouping:       "Each edit belongs to the request containing its source clip's start on the unpadded timeline; native boundaries include the 1 s hard overlap and 120 ms padding",
		RequestWindows: windows,
	}, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(encoded))
}

func readJSON(path string, target any) {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		panic(err)
	}
}
