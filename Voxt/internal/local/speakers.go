package local

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
)

// SpeakerModels is the optional audio-only speaker service. It does not
// change which model ASR opens or occupy that model's inference queue.
type SpeakerModels interface {
	Speakers(context.Context) (speaker.Stream, error)
}

func (l *Library) Speakers(ctx context.Context) (speaker.Stream, error) {
	if !l.models.Installed(models.SpeakerDiarization) {
		return nil, NotInstalledError("speaker", models.SpeakerDiarization)
	}
	if l.open.Speakers == nil {
		return nil, ErrNoRuntime
	}
	l.claim(ctx, models.SpeakerDiarization)
	return l.open.Speakers(ctx, l.models.Dir(models.SpeakerDiarization))
}

// OpenSpeakers forwards the optional service through provider/dictation
// wrappers, and gives fakes without this capability a normal runtime error.
func OpenSpeakers(ctx context.Context, s Services) (speaker.Stream, error) {
	if backend, ok := s.(SpeakerModels); ok {
		return backend.Speakers(ctx)
	}
	return nil, ErrNoRuntime
}
