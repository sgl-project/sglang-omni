package session

import (
	"context"
	"encoding/json"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// Caller is how sessions reach the helper: helper.Supervisor's Call.
type Caller interface {
	Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error)
}

// HelperAudio is the microphone through the helper (capture.start as a
// job named by the session's id, capture.stop).
type HelperAudio struct{ H Caller }

func (h HelperAudio) Record(ctx context.Context, id string, p proto.CaptureParams, fn func(AudioEvent)) (proto.CaptureResult, error) {
	var out proto.CaptureResult
	raw, err := h.H.Call(ctx, id, proto.MethodCaptureStart, p, func(m proto.Message) {
		var ev AudioEvent
		switch m.Method {
		case proto.MethodCapturePCM:
			ev.PCM = new(proto.PCMParams)
			if m.DecodeParams(ev.PCM) != nil {
				return
			}
		case proto.MethodCaptureLevel:
			ev.Level = new(proto.LevelParams)
			if m.DecodeParams(ev.Level) != nil {
				return
			}
		case proto.MethodCaptureStarted:
			ev.Started = new(proto.CaptureStartedParams)
			if m.DecodeParams(ev.Started) != nil {
				return
			}
		case proto.MethodCaptureRestarted:
			ev.Restarted = new(proto.RestartedParams)
			if m.DecodeParams(ev.Restarted) != nil {
				return
			}
		default:
			return
		}
		fn(ev)
	})
	if err != nil {
		return out, err
	}
	err = json.Unmarshal(raw, &out)
	return out, err
}

func (h HelperAudio) Stop(ctx context.Context, id string) error {
	_, err := h.H.Call(ctx, "", proto.MethodCaptureStop, proto.CaptureStopParams{Job: id}, nil)
	return err
}

// HelperText is the text I/O of the app in front through the helper
// (textio.*, app.frontmost); it is a deliver.TextIO.
type HelperText struct{ H Caller }

func call[T any](ctx context.Context, h Caller, method string, params any) (T, error) {
	var out T
	raw, err := h.Call(ctx, "", method, params, nil)
	if err != nil {
		return out, err
	}
	err = json.Unmarshal(raw, &out)
	return out, err
}

func (h HelperText) Focus(ctx context.Context) (proto.FocusResult, error) {
	return call[proto.FocusResult](ctx, h.H, proto.MethodTextFocus, nil)
}

func (h HelperText) Paste(ctx context.Context, p proto.PasteParams) (proto.PasteResult, error) {
	return call[proto.PasteResult](ctx, h.H, proto.MethodTextPaste, p)
}

// Selection reads the selected text (Translate and Rewrite).
func (h HelperText) Selection(ctx context.Context, p proto.SelectionParams) (proto.SelectionResult, error) {
	return call[proto.SelectionResult](ctx, h.H, proto.MethodTextSelection, p)
}

// Frontmost is the app in front.
func (h HelperText) Frontmost(ctx context.Context) (proto.App, error) {
	return call[proto.App](ctx, h.H, proto.MethodAppFrontmost, nil)
}
