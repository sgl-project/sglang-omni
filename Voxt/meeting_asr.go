package main

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

func (a *app) meetingVAD() vad.Model {
	a.omniMu.Lock()
	defer a.omniMu.Unlock()
	rt, err := a.omniRuntime()
	if err != nil {
		return nil
	}
	if v := a.omniSilero(rt); v != nil {
		return v
	}
	return nil
}

func (a *app) meetingTrackASR(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
	id := a.prefs.Features.Meeting.ASR
	p, _, err := a.local.ASR(ctx, id)
	if err != nil {
		return meeting.TrackASR{}, err
	}
	caps := p.Caps()
	l := session.SpeechLang(a.prefs, i18n.Locale())
	ta := meeting.TrackASR{Caps: caps, Lang: l, StreamReq: asr.StreamRequest{Lang: l}}
	if ft, ok := p.(asr.FileTranscriber); ok && caps.File {
		ta.File = ft
	}
	if st, ok := p.(asr.Streamer); ok && caps.Stream {
		ta.Streamer = st
	}
	if va, ok := p.(local.VoiceActivity); ok {
		if ms, ok := va.VAD().(vad.ModelStream); ok {
			ta.VAD = ms
		}
	}
	return ta, nil
}
