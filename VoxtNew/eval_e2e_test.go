package main

// App enhancement and the dictionary end to end on the real models
// (VOXTNEW_EVAL=1, eval_test.go): the app's own Library and runtimes over
// the model library ($HUB) with asr.accurate, Silero and llm.balanced, the
// fake helper playing an eval clip as the recording (never the
// microphone or the speaker) with an app in front. An app group's prompt
// changes the answer; a dictionary term reaches the speech model's
// context (the transcript) and the language model's glossary (the
// answer, when the speech model did not get it).

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// evalSession runs one Transcribe session on the real models in an app of
// its own, quit before the next (a subtest: its runtimes stop at its
// end): clip id played by the fake helper (five times real time) with
// front in front, stopped 1.5 s after the clip's end; it returns the end.
func evalSession(t *testing.T, e *evalRun, id, front string, terms []store.Term, set func(*settings.Settings), rules *appctx.Rules) session.Snapshot {
	t.Helper()
	var end session.Snapshot
	if !t.Run(id+" in "+front, func(t *testing.T) { end = runEvalSession(t, e, id, front, terms, set, rules) }) {
		t.FailNow()
	}
	return end
}

func runEvalSession(t *testing.T, e *evalRun, id, front string, terms []store.Term, set func(*settings.Settings), rules *appctx.Rules) session.Snapshot {
	t.Helper()
	c, ok := e.set.Clip(id)
	if !ok {
		t.Fatalf("no clip %s", id)
	}
	for _, m := range []string{models.ASRAccurate, models.VADSilero, models.LLMBalanced} {
		e.dir(t, m) // installed
	}
	svc := realLocal{hub: e.hub}
	wav, _ := filepath.Abs(filepath.Join("testdata", "eval", id+".wav")) // the helper runs elsewhere
	opts := script(t, hotkeyIn(300, front), hotkeyIn(300+int(c.Duration()/time.Millisecond/5)+1500, front)) +
		",front=" + front + "," + helpertest.Opt("pcm", wav)
	a := newSessionApp(t, opts, func(s *settings.Settings) {
		s.General.MainLanguage = c.Lang
		s.Features.Transcribe.ASR, s.Features.Transcribe.LLM = models.ASRAccurate, models.LLMBalanced
		if set != nil {
			set(s)
		}
	}, svc)
	for _, term := range terms {
		if err := a.lib.store.Dictionary().Put(context.Background(), term); err != nil {
			t.Fatal(err)
		}
	}
	if rules != nil {
		a.setRules(*rules)
	}
	t0 := time.Now()
	a.pumpLong("the end", 3*time.Minute, func() bool { return a.sess.ID != "" && a.sess.State.Ended() })
	end := a.sess
	if end.State != session.Done || end.Result == "" {
		t.Fatalf("%s: end %s %+v %q", id, end.State, end.Problem, end.Result)
	}
	t.Logf("%s in %s: %v (audio %v of %v, sent %v; ASR %v, LLM %v); profile %s\n    heard  %q\n    result %q", id, front, time.Since(t0).Round(time.Millisecond),
		end.Timings.Audio.Round(time.Millisecond), c.Duration().Round(time.Millisecond), end.Timings.Sent.Round(time.Millisecond),
		end.Timings.ASR.Round(time.Millisecond), end.Timings.LLM.Round(time.Millisecond), end.Profile, end.Raw, end.Result)
	return end
}

// TestEvalAppEnhancement: Slack's group asks for lowercase without a final
// period; the same dictation in TextEdit (no group) keeps its capitals.
func TestEvalAppEnhancement(t *testing.T) {
	e := evalSetup(t)
	rules := appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{{ID: "g-chat", Name: "Chat", Enabled: true,
		Prompt: "Casual chat: write everything in lowercase letters, with no period at the end.",
		Apps:   []appctx.App{{BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}}, Features: appctx.DefaultFeatures}}}
	chat := evalSession(t, e, "en-3", "com.tinyspeck.slackmacgap", nil, nil, &rules)
	plain := evalSession(t, e, "en-3", "com.apple.TextEdit", nil, nil, &rules)
	if chat.Scene == nil || chat.Scene.GroupID() != "g-chat" || !strings.Contains(chat.Profile, "scene") {
		t.Errorf("Slack: scene %+v, profile %s", chat.Scene, chat.Profile)
	}
	if plain.Scene != nil && plain.Scene.GroupID() != "" || strings.Contains(plain.Profile, "scene") {
		t.Errorf("TextEdit: scene %+v, profile %s", plain.Scene, plain.Profile)
	}
	upper := func(s string) bool { return strings.IndexFunc(s, unicode.IsUpper) >= 0 }
	if upper(chat.Result) || strings.HasSuffix(chat.Result, ".") {
		t.Errorf("Slack's group: %q, want lowercase without a final period", chat.Result)
	}
	if !upper(plain.Result) {
		t.Errorf("no group: %q, want its capitals", plain.Result)
	}
}

// TestEvalDictionary: "VoxtNew" in the dictionary makes the speech model
// write it (its context bias; without the dictionary it hears "Vox
// Tunew"), and, kept from the speech model (Settings: no hints), the
// language model writes it from what was heard (the glossary). An alias
// needs no model: "Or Link", as the speech model hears Auralink, becomes
// the term with the clean-up off, while the history keeps what was heard.
func TestEvalDictionary(t *testing.T) {
	e := evalSetup(t)
	terms := []store.Term{{ID: "t1", Term: "VoxtNew", Enabled: true}, {ID: "t2", Term: "MLX", Enabled: true}}
	both := evalSession(t, e, "dict-1", "com.apple.TextEdit", terms, nil, nil)
	glossary := evalSession(t, e, "dict-1", "com.apple.TextEdit", terms, func(s *settings.Settings) { s.Dictionary.ASRHints = false }, nil)
	none := evalSession(t, e, "dict-1", "com.apple.TextEdit", terms, func(s *settings.Settings) { s.Dictionary.Enabled = false }, nil)
	if !strings.Contains(both.Raw, "VoxtNew") || !strings.Contains(both.Result, "VoxtNew") {
		t.Errorf("the speech model's context: heard %q, %q", both.Raw, both.Result)
	}
	if strings.Contains(glossary.Raw, "VoxtNew") || !strings.Contains(glossary.Profile, "glossary") || !strings.Contains(glossary.Result, "VoxtNew") {
		t.Errorf("the glossary alone: profile %s, %q (heard %q)", glossary.Profile, glossary.Result, glossary.Raw)
	}
	if strings.Contains(none.Raw, "VoxtNew") || strings.Contains(none.Result, "VoxtNew") || strings.Contains(none.Profile, "glossary") {
		t.Errorf("without the dictionary: profile %s, %q (heard %q)", none.Profile, none.Result, none.Raw)
	}
	aliases := []store.Term{{ID: "t3", Term: "Auralink", Aliases: []string{"Or Link"}, Enabled: true}}
	plain := evalSession(t, e, "dict-5", "com.apple.TextEdit", aliases, func(s *settings.Settings) {
		s.Dictionary.ASRHints, s.Features.Transcribe.Enhance = false, false
	}, nil)
	if !strings.Contains(plain.Raw, "Or Link") || !strings.Contains(plain.Result, "Auralink") || strings.Contains(plain.Result, "Or Link") || plain.Timings.LLM != 0 {
		t.Errorf("the alias without a language model: %q (heard %q, LLM %v)", plain.Result, plain.Raw, plain.Timings.LLM)
	}
}
