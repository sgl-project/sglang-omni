package aliyun

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// The inference protocol (Model Studio: "Fun-ASR-Realtime WebSocket API",
// client and server events):
//
//	→ {"header":{"action":"run-task","task_id":…,"streaming":"duplex"},
//	   "payload":{"task_group":"audio","task":"asr","function":"recognition",
//	              "model":…,"parameters":{"format":"pcm","sample_rate":16000,…},"input":{…}}}
//	← {"header":{"event":"task-started",…}}
//	→ binary PCM frames (100 ms each, as the official samples send)
//	← {"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"sentence_id":1,"text":…,"sentence_end":false}}}}
//	→ {"header":{"action":"finish-task","task_id":…,"streaming":"duplex"},"payload":{"input":{}}}
//	← result-generated… then {"header":{"event":"task-finished"}}
//	← {"header":{"event":"task-failed","error_code":…,"error_message":…}} at any point
//
// Each result-generated carries the whole current sentence; sentence_end
// marks it final. A heartbeat result (heartbeat: true) carries nothing.

// maxContext is the service's bound on one turn of input.context, in
// characters (longer text is cut at its end by the service; we cut it
// first so the hot words, which come first, survive).
const maxContext = 400

// maxVocabulary bounds the inline hot words (the service takes 500 to
// 2000 depending on the model).
const maxVocabulary = 500

// vocabularyWeight is the weight of an inline hot word: the documented
// recommendation, which Voxt uses too.
const vocabularyWeight = 4

// frameBytes is 100 ms of PCM16 mono at 16 kHz, the frame both protocols'
// official samples send.
const frameBytes = 3200

type inferenceProto struct {
	model  string
	info   model
	req    asr.StreamRequest
	taskID string
	text   transcript
}

type infHeader struct {
	Action    string `json:"action"`
	TaskID    string `json:"task_id"`
	Streaming string `json:"streaming"`
}

type infContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type infMessage struct {
	Role    string       `json:"role"`
	Content []infContent `json:"content"`
}

type infInput struct {
	Context []infMessage `json:"context,omitempty"`
}

type infRunPayload struct {
	TaskGroup  string         `json:"task_group"`
	Task       string         `json:"task"`
	Function   string         `json:"function"`
	Model      string         `json:"model"`
	Parameters map[string]any `json:"parameters"`
	Input      infInput       `json:"input"`
}

type infControl struct {
	Header  infHeader `json:"header"`
	Payload any       `json:"payload"`
}

// runTask is the run-task message.
func (p *inferenceProto) runTask() []byte {
	params := map[string]any{"format": "pcm", "sample_rate": p.req.SampleRate}
	if l := p.info.lang(p.req.Lang); l != "" && p.info.hints > 0 {
		params["language_hints"] = []string{l}
	}
	if p.info.vocabulary {
		if v := vocabulary(p.req.Hints); len(v) > 0 {
			params["vocabulary"] = v
		}
	}
	var in infInput
	if p.info.context {
		if t := contextText(p.req.Hints, p.req.Prompt, !p.info.vocabulary); t != "" {
			in.Context = []infMessage{{Role: "user", Content: []infContent{{Type: "input_text", Text: t}}}}
		}
	}
	return mustJSON(infControl{
		Header:  infHeader{Action: "run-task", TaskID: p.taskID, Streaming: "duplex"},
		Payload: infRunPayload{TaskGroup: "audio", Task: "asr", Function: "recognition", Model: p.model, Parameters: params, Input: in},
	})
}

// finishTask is the finish-task message.
func (p *inferenceProto) finishTask() []byte {
	return mustJSON(infControl{
		Header:  infHeader{Action: "finish-task", TaskID: p.taskID, Streaming: "duplex"},
		Payload: map[string]any{"input": map[string]any{}},
	})
}

// vocabulary is the inline hot word map: each term once, at most
// maxVocabulary of them.
func vocabulary(hints []string) map[string]int {
	v := map[string]int{}
	for _, h := range hints {
		h = strings.TrimSpace(h)
		if h == "" || v[h] != 0 {
			continue
		}
		if len(v) == maxVocabulary {
			break
		}
		v[h] = vocabularyWeight
	}
	return v
}

// contextText is the one user message of input.context: the hot words
// (when withHints; the context works by matching the words it contains),
// one per line, then the prompt, cut to maxContext characters.
func contextText(hints []string, prompt string, withHints bool) string {
	var lines []string
	if withHints {
		seen := map[string]bool{}
		for _, h := range hints {
			h = strings.TrimSpace(h)
			if h != "" && !seen[h] {
				seen[h] = true
				lines = append(lines, h)
			}
		}
	}
	if p := strings.TrimSpace(prompt); p != "" {
		lines = append(lines, p)
	}
	return clipRunes(strings.Join(lines, "\n"), maxContext)
}

func (p *inferenceProto) start() [][]byte { return [][]byte{p.runTask()} }

func (p *inferenceProto) audio(frame []byte) (websocket.MessageType, []byte) {
	return websocket.MessageBinary, frame
}

func (p *inferenceProto) finish() [][]byte { return [][]byte{p.finishTask()} }

func (p *inferenceProto) frameSize() int { return frameBytes }

func (p *inferenceProto) current() string { return p.text.String() }

func (p *inferenceProto) language() string { return "" }

// infEvent is what this package reads of a server event.
type infEvent struct {
	Header struct {
		Event        string `json:"event"`
		ErrorCode    string `json:"error_code"`
		ErrorMessage string `json:"error_message"`
	} `json:"header"`
	Payload struct {
		Output struct {
			Sentence *struct {
				SentenceID  *json.Number `json:"sentence_id"`
				Text        string       `json:"text"`
				SentenceEnd bool         `json:"sentence_end"`
				Heartbeat   bool         `json:"heartbeat"`
			} `json:"sentence"`
		} `json:"output"`
	} `json:"payload"`
}

func (p *inferenceProto) handle(data []byte, finishing bool) (step, error) {
	var ev infEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return stepIgnore, nil // not ours to understand; the next event decides
	}
	switch strings.ToLower(strings.TrimSpace(ev.Header.Event)) {
	case "task-started":
		return stepStarted, nil
	case "result-generated":
		s := ev.Payload.Output.Sentence
		if s == nil || s.Heartbeat {
			return stepIgnore, nil
		}
		id := ""
		if s.SentenceID != nil {
			if n, err := s.SentenceID.Int64(); err == nil && n > 0 {
				id = strconv.FormatInt(n, 10)
			}
		}
		p.text.set(id, s.Text, s.SentenceEnd)
		return stepText, nil
	case "task-finished":
		return stepDone, nil
	case "task-failed", "error":
		code, msg := asr.ErrorBody(data)
		if noSpeech(code, msg) {
			return stepDone, nil
		}
		return stepIgnore, serviceError(code, msg)
	}
	return stepIgnore, nil // unknown events are ignored
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err) // only our own types are marshalled
	}
	return b
}
