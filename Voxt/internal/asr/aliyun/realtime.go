package aliyun

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// The realtime protocol (Model Studio: "Qwen-ASR-Realtime" client and
// server events), on …/api-ws/v1/realtime?model=… with the header
// OpenAI-Beta: realtime=v1:
//
//	← {"type":"session.created",…}                      (ignored)
//	→ {"event_id":…,"type":"session.update","session":{"modalities":["text"],
//	   "input_audio_format":"pcm","sample_rate":16000,
//	   "input_audio_transcription":{"language":"zh"},
//	   "turn_detection":{"type":"server_vad","threshold":0.35,"silence_duration_ms":800}}}
//	← {"type":"session.updated",…}
//	→ {"event_id":…,"type":"input_audio_buffer.append","audio":"<base64 PCM>"}…
//	← {"type":"conversation.item.input_audio_transcription.text","item_id":…,"text":…,"stash":…}
//	← {"type":"conversation.item.input_audio_transcription.completed","item_id":…,"transcript":…,"language":…}
//	→ {"event_id":…,"type":"session.finish"}
//	← …completed for the last item, then {"type":"session.finished"}
//	← {"type":"error","error":{"type":…,"code":…,"message":…}} at any point
//
// Server VAD stays on, as in Voxt (its default settings, threshold 0.35,
// 800 ms of silence): the service cuts the speech into items and streams
// each one's text, so partial text shows while the user speaks.
// (input_audio_buffer.commit is for manual mode only and is not sent.)
// A text event's preview is text + stash: text is the settled prefix of
// the item, stash the draft after it (Voxt shows text alone).
//
// The reference has no context or hot word field for this model, so it
// takes neither (Caps.Hints, Caps.Prompt false).

// Voxt's default server VAD settings.
const (
	vadThreshold = 0.35
	vadSilenceMS = 800
)

type realtimeProto struct {
	model string
	info  model
	req   asr.StreamRequest
	text  transcript
	lang  string // as the service reported it
	// newEventID makes event ids (tests make them fixed).
	newEventID func() string
}

func (p *realtimeProto) eventID() string {
	if p.newEventID != nil {
		return p.newEventID()
	}
	return "event_" + newID()
}

type rtTurnDetection struct {
	Type              string  `json:"type"`
	Threshold         float64 `json:"threshold"`
	SilenceDurationMS int     `json:"silence_duration_ms"`
}

type rtTranscription struct {
	Language string `json:"language,omitempty"`
}

type rtSession struct {
	Modalities         []string         `json:"modalities"`
	InputAudioFormat   string           `json:"input_audio_format"`
	SampleRate         int              `json:"sample_rate"`
	InputTranscription *rtTranscription `json:"input_audio_transcription,omitempty"`
	TurnDetection      rtTurnDetection  `json:"turn_detection"`
}

type rtEvent struct {
	EventID string     `json:"event_id"`
	Type    string     `json:"type"`
	Session *rtSession `json:"session,omitempty"`
	Audio   string     `json:"audio,omitempty"`
}

// sessionUpdate is the session.update message.
func (p *realtimeProto) sessionUpdate() []byte {
	s := &rtSession{
		Modalities:       []string{"text"},
		InputAudioFormat: "pcm",
		SampleRate:       p.req.SampleRate,
		TurnDetection:    rtTurnDetection{Type: "server_vad", Threshold: vadThreshold, SilenceDurationMS: vadSilenceMS},
	}
	if l := p.info.lang(p.req.Lang); l != "" {
		s.InputTranscription = &rtTranscription{Language: l}
	}
	return mustJSON(rtEvent{EventID: p.eventID(), Type: "session.update", Session: s})
}

func (p *realtimeProto) start() [][]byte { return [][]byte{p.sessionUpdate()} }

func (p *realtimeProto) audio(frame []byte) (websocket.MessageType, []byte) {
	return websocket.MessageText, mustJSON(rtEvent{EventID: p.eventID(), Type: "input_audio_buffer.append", Audio: base64.StdEncoding.EncodeToString(frame)})
}

func (p *realtimeProto) finish() [][]byte {
	return [][]byte{mustJSON(rtEvent{EventID: p.eventID(), Type: "session.finish"})}
}

func (p *realtimeProto) frameSize() int { return frameBytes }

func (p *realtimeProto) current() string { return p.text.String() }

func (p *realtimeProto) language() string { return p.lang }

type rtServerEvent struct {
	Type       string `json:"type"`
	ItemID     string `json:"item_id"`
	Text       string `json:"text"`
	Stash      string `json:"stash"`
	Transcript string `json:"transcript"`
	Language   string `json:"language"`
}

func (p *realtimeProto) handle(data []byte, finishing bool) (step, error) {
	var ev rtServerEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return stepIgnore, nil
	}
	switch strings.ToLower(strings.TrimSpace(ev.Type)) {
	case "session.updated":
		return stepStarted, nil
	case "conversation.item.input_audio_transcription.text":
		p.text.set(ev.ItemID, ev.Text+ev.Stash, false)
		if ev.Language != "" {
			p.lang = ev.Language
		}
		return stepText, nil
	case "conversation.item.input_audio_transcription.completed":
		p.text.set(ev.ItemID, ev.Transcript, true)
		if ev.Language != "" {
			p.lang = ev.Language
		}
		return stepText, nil
	case "session.finished":
		return stepDone, nil
	case "error", "conversation.item.input_audio_transcription.failed":
		code, msg := asr.ErrorBody(data)
		if noSpeech(code, msg) {
			return stepDone, nil
		}
		// Voxt saw the service answer session.finish with a bare "task
		// failed" once the text was in; keep the text then.
		if finishing && genericFailure(msg) && p.text.String() != "" {
			return stepDone, nil
		}
		e := serviceError(code, msg)
		if ev.Type != "error" && e.Kind == asr.KindProtocol {
			e.Kind, e.Retryable = asr.KindServer, true // recognition of an item failed
		}
		return stepIgnore, e
	}
	return stepIgnore, nil // session.created, speech_started, committed, item.created…
}

func genericFailure(msg string) bool {
	m := strings.ToLower(strings.TrimSpace(msg))
	return m == "" || m == "task failed" || m == "task failed."
}
