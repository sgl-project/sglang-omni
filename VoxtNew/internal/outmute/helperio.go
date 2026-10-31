package outmute

import (
	"context"
	"encoding/json"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// Caller is how the Controller reaches the helper: helper.Supervisor's
// Call.
type Caller interface {
	Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error)
}

// HelperOutput is the output through the helper (output.*): an Output.
type HelperOutput struct{ H Caller }

func call[T any](ctx context.Context, h Caller, method string, params any) (T, error) {
	var out T
	raw, err := h.Call(ctx, "", method, params, nil)
	if err != nil {
		return out, err
	}
	err = json.Unmarshal(raw, &out)
	return out, err
}

func (h HelperOutput) Watch(ctx context.Context, on bool) (proto.OutputState, error) {
	return call[proto.OutputState](ctx, h.H, proto.MethodOutputWatch, proto.OutputWatchParams{On: on})
}

func (h HelperOutput) Mute(ctx context.Context, p proto.OutputMuteParams) (proto.OutputMuteResult, error) {
	return call[proto.OutputMuteResult](ctx, h.H, proto.MethodOutputMute, p)
}

func (h HelperOutput) Release(ctx context.Context, p proto.OutputReleaseParams) (proto.OutputReleaseResult, error) {
	return call[proto.OutputReleaseResult](ctx, h.H, proto.MethodOutputRelease, p)
}

func (h HelperOutput) Recover(ctx context.Context, p proto.OutputRecoverParams) (proto.OutputRecoverResult, error) {
	return call[proto.OutputRecoverResult](ctx, h.H, proto.MethodOutputRecover, p)
}
