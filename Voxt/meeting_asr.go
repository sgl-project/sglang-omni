package main

import (
	"context"
	"strings"

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
	// The dictionary terms and speech context dictation sends (#3988); a
	// dictionary that will not read leaves the meeting without terms.
	var hints []string
	if a.prefs.Dictionary.Enabled {
		if terms, err := a.lib.store.Dictionary().List(ctx, ""); err == nil {
			hints = session.SpeechHints(a.prefs, terms)
		}
	}
	prompt := asr.BuildSpeechPrompt(strings.TrimSpace(a.prefs.Dictionary.SpeechContext), "", session.MainLanguage(a.prefs, i18n.Locale()), a.prefs.General.OtherLanguages, hints)
	ta := meeting.TrackASR{Caps: caps, Lang: l, Prompt: prompt, Hints: hints, StreamReq: asr.StreamRequest{Lang: l, Prompt: prompt, Hints: hints}}
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
