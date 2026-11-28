package main

import (
	"context"
	"os"
	"path/filepath"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/resident"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speakerrt"
)

type meetingSpeakerModel struct{ services local.Services }

func (m meetingSpeakerModel) Open(ctx context.Context) (speaker.Stream, error) {
	return local.OpenSpeakers(ctx, m.services)
}

func (a *app) openSpeaker(ctx context.Context, dir string) (speaker.Stream, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, err
	}
	binary := filepath.Clean(filepath.Join(filepath.Dir(exe), "..", "Resources", "speaker", "voxtnew-speaker"))
	model := speakerrt.Model{Binary: binary, Dir: dir, Memory: procMemory, MaxMemory: 768 << 20}
	if rt := a.famRuntime(); rt != nil {
		// Reserve measured CPU inference memory with headroom, not the
		// much smaller download size. The process has a separate hard budget.
		rt.Reserve(ctx, models.SpeakerDiarization, 512<<20)
	}
	if events := a.modelEvents("speaker"); events != nil {
		model.OnReady = func(pid int) { events(resident.Event{Kind: resident.Loaded, ID: models.SpeakerDiarization, PID: pid}) }
		model.OnClose = func() { events(resident.Event{Kind: resident.Unloaded, ID: models.SpeakerDiarization}) }
	}
	return model.Open(ctx)
}
