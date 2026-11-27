package main

import (
	"cmp"
	"context"
	"fmt"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
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

func (a *app) meetingMTDInstalled() bool {
	if a.local == nil {
		return false
	}
	for _, m := range a.local.Installed(local.ASR) {
		if m.ID == models.ASRMoss {
			return true
		}
	}
	return false
}

// meetingDiarizerLanguageOK says the meeting's language is one MTD
// knows: MTD knows only zh and en, so a meeting in another language
// keeps the selected model's transcription instead of a wrong-language
// one. An empty hint means the language is unknown (SpeechLang stays
// empty on purpose, #3398), so the same call.
func (a *app) meetingDiarizerLanguageOK() bool {
	l := a.meetingLang()
	if l == "" {
		return false
	}
	if spec, ok := omni.Lookup(models.ASRMoss); ok {
		_, fit := lang.Match(spec.Languages, l)
		return fit
	}
	return true
}

// meetingDiarizerGates says the automatic MTD diarizer would run for a
// meeting with the current settings, without opening a model: the final
// pass is on, the mode diarizes, MTD is installed and the meeting's
// language is one MTD knows (#4067).
func (a *app) meetingDiarizerGates() bool {
	if a.prefs.Features.Meeting.NoFinalPass {
		return false
	}
	if !meetingModeFrom(a.prefs.Features.Meeting).Diarize() {
		return false
	}
	return a.meetingMTDInstalled() && a.meetingDiarizerLanguageOK()
}

func (a *app) meetingDiarizer(ctx context.Context) (asr.Diarizer, error) {
	if !a.meetingMTDInstalled() || !a.meetingDiarizerLanguageOK() {
		return nil, nil
	}
	p, opened, err := a.local.ASR(ctx, models.ASRMoss)
	if err != nil {
		return nil, err
	}
	if opened.ID != models.ASRMoss {
		return nil, nil
	}
	d, ok := p.(asr.Diarizer)
	if !ok {
		return nil, fmt.Errorf("meeting: %s has no speaker diarization support", models.ASRMoss)
	}
	return d, nil
}

// meetingLang is the language a meeting is spoken in: its own setting,
// else the main language's hint (#3387).
func (a *app) meetingLang() string {
	return cmp.Or(a.prefs.Features.Meeting.Lang, session.SpeechLang(a.prefs, i18n.Locale()))
}

// meetingASRModel is the speech model a meeting uses: its own choice,
// resolved as meetingTrackASR opens it (a deleted selection falls back
// to an installed model), else Automatic's language-gated pick for the
// meeting's language (meetingTrackASR's rule), without opening it.
func (a *app) meetingASRModel() (local.Model, bool) {
	if id := strings.TrimSpace(a.prefs.Features.Meeting.ASR); id != "" {
		return local.Choose(a.installed(local.ASR), id)
	}
	return local.Choose(automaticSpeechFor(a.meetingLang(), a.installed(local.ASR)), "")
}

func (a *app) meetingTrackASR(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
	id := strings.TrimSpace(a.prefs.Features.Meeting.ASR)
	if id == "" {
		// Automatic follows the features' rule (session_link.go's
		// resolver): a model the language gate keeps for the meeting's
		// language, so a meeting in a language MTD does not know never
		// lands on it (#4067).
		m, ok := local.Choose(automaticSpeechFor(a.meetingLang(), a.local.Installed(local.ASR)), "")
		if !ok {
			return meeting.TrackASR{}, local.NotInstalledError(local.ASR, id)
		}
		id = m.ID
	}
	p, _, err := a.local.ASR(ctx, id)
	if err != nil {
		return meeting.TrackASR{}, err
	}
	caps := p.Caps()
	l := a.meetingLang()
	// The dictionary terms and speech context dictation sends (#3988); a
	// dictionary that will not read leaves the meeting without terms.
	var hints []string
	if a.prefs.Dictionary.Enabled {
		if terms, err := a.lib.store.Dictionary().List(ctx, ""); err == nil {
			hints = session.SpeechHints(a.prefs, terms)
		}
	}
	// What dictation's speechContext gives an on-device model: no service template.
	prompt := strings.TrimSpace(a.prefs.Dictionary.SpeechContext)
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
