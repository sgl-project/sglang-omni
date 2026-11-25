package main

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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

func (a *app) meetingDiarizer(ctx context.Context) (asr.Diarizer, error) {
	if !a.meetingMTDInstalled() {
		return nil, nil
	}
	// MTD knows only zh and en: a meeting in another language keeps the
	// selected model's transcription instead of a wrong-language one. An
	// empty hint means the language is unknown (SpeechLang stays empty on
	// purpose, #3398), so the same call.
	if l := cmp.Or(a.prefs.Features.Meeting.Lang, session.SpeechLang(a.prefs, i18n.Locale())); l != "" {
		if spec, ok := omni.Lookup(models.ASRMoss); ok {
			if _, fit := lang.Match(spec.Languages, l); !fit {
				return nil, nil
			}
		}
	} else {
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

func (a *app) meetingTrackASR(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
	id := a.prefs.Features.Meeting.ASR
	p, _, err := a.local.ASR(ctx, id)
	if err != nil {
		return meeting.TrackASR{}, err
	}
	caps := p.Caps()
	l := cmp.Or(a.prefs.Features.Meeting.Lang, session.SpeechLang(a.prefs, i18n.Locale()))
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
	ta.Script = meetingScript(a.prefs, i18n.Locale(), readingLanguages())
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

// meetingScript is the script a meeting writes Chinese in, "zh-Hans" or
// "zh-Hant" ("": the speech model's): the one its spoken language names,
// else dictation's (#4534). With the language left to the model, Chinese
// it hears follows the Chinese the user reads even when the main language
// is another (11-25: an English Voxt on a Mac reading zh-Hans got 嵐 and
// 為 in a Chinese meeting).
func meetingScript(s settings.Settings, locale string, reads []string) string {
	main := cmp.Or(s.Features.Meeting.Lang, session.MainLanguage(s, locale))
	if c := lang.Canonical(main); s.Features.Meeting.Lang == "" && c != "zh" && c != "yue" {
		main = "zh"
	}
	return lang.ChineseScript(main, append(slices.Clip(reads), locale))
}
