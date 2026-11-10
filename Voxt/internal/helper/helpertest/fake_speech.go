package helpertest

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// ---- speech.* (the speech option) ----

// SpeechLangs are the languages the fake recognises on device; a
// speech.recognize of another one is unsupported.
var SpeechLangs = []string{"en", "zh"}

// speechText is what speech.recognize hears without a speech=TEXT.
const speechText = "hello from system dictation"

func isSpeech(method string) bool { return strings.HasPrefix(method, "speech.") }

// speechJob is a running speech.recognize: it hears a word of its text
// for each speech.audio that brings audio.
type speechJob struct {
	words []string
	heard int
}

func (f *fake) speechRecognize(m proto.Message) {
	var p proto.SpeechParams
	if err := m.DecodeParams(&p); err != nil {
		f.fail(m.ID, proto.CodeBadRequest, "Bad params: %v", err)
		return
	}
	f.mu.Lock()
	if f.speechPerm == proto.PermNotDetermined {
		f.speechPerm = proto.PermGranted
	}
	perm := f.speechPerm
	f.mu.Unlock()
	lang, _, _ := strings.Cut(p.Lang, "-")
	switch {
	case perm != proto.PermGranted:
		f.failData(m.ID, proto.CodePermissionDenied, map[string]string{"permission": proto.PermissionSpeechRecognition}, "Speech Recognition is not allowed")
		return
	case lang != "" && !slices.Contains(SpeechLangs, lang):
		f.failData(m.ID, proto.CodeUnsupported, map[string]string{"capability": "onDeviceSpeech"}, "macOS has no on-device speech recognition for %s", p.Lang)
		return
	}
	text := f.opts["speech"]
	if text == "" {
		text = speechText
	}
	f.mu.Lock()
	f.speech[m.ID] = &speechJob{words: strings.Fields(text)}
	f.mu.Unlock()
	locale := map[string]string{"": "en-US", "en": "en-US", "zh": "zh-CN"}[lang]
	f.notify(proto.MethodSpeechStarted, proto.SpeechStartedParams{Job: m.ID, Locale: locale})
}

func (f *fake) speechAudio(m proto.Message) {
	var p proto.SpeechAudioParams
	m.DecodeParams(&p)
	f.mu.Lock()
	j := f.speech[p.Job]
	var partial, final string
	if j != nil {
		if len(p.Data) > 0 && j.heard < len(j.words) {
			j.heard++
			partial = strings.Join(j.words[:j.heard], " ")
		}
		if p.End {
			delete(f.speech, p.Job)
			if j.heard > 0 {
				final = strings.Join(j.words, " ")
			}
		}
	}
	f.mu.Unlock()
	if j == nil {
		f.fail(m.ID, proto.CodeBadRequest, "No speech.recognize job %s", p.Job)
		return
	}
	if partial != "" {
		f.notify(proto.MethodSpeechPartial, proto.SpeechPartialParams{Job: p.Job, Text: partial})
	}
	f.result(m.ID, proto.Empty{})
	if p.End {
		f.result(p.Job, proto.SpeechResult{Text: final})
	}
}

// cancelSpeech ends speech.recognize job (it answers cancelled); false
// when it is not one that runs.
func (f *fake) cancelSpeech(job string) bool {
	f.mu.Lock()
	_, ok := f.speech[job]
	delete(f.speech, job)
	f.mu.Unlock()
	if ok {
		f.fail(job, proto.CodeCancelled, "Cancelled")
	}
	return ok
}
