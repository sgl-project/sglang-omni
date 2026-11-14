package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

// TestRemoteGenerationSettings: a language model's dialog has Voxt 1.x's
// generation settings for its service, the sampling ones and the extra
// body JSON only in the folded Advanced (no Expert section); saved, they are in the setup and in the dialog again; a
// field that is not a number, or JSON that is not an object, is refused
// with what is wrong.
func TestRemoteGenerationSettings(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400) // the whole dialog in view, Advanced open too
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, llm.Custom) // its card is below the fold
	frames(tt)
	for _, s := range []string{"Generation settings", "Thinking", "Max output tokens (optional)", "Verbosity", "Advanced"} {
		if !has(tt, s) {
			t.Errorf("no %q in the dialog", s)
		}
	}
	if has(tt, "Temperature") || has(tt, "Top P") || has(tt, "Request parameters") || has(tt, "Expert") {
		t.Errorf("sampling or JSON fields outside the folded Advanced, or an Expert section: %q", tt.Texts())
	}
	click(t, tt, "Advanced")
	if !has(tt, "Temperature") || !has(tt, "Top P") || !has(tt, "Request parameters") {
		t.Errorf("Advanced open: %q", tt.Texts())
	}
	d := &a.provDlg
	d.endpoint, d.customModel = "https://example.com/v1", "m"
	d.gen.think, d.gen.effort = thinkTitle(llm.ThinkEffort), "none"
	d.gen.maxTokens, d.gen.temperature, d.gen.topP = "131072", "0", "0.9"
	d.gen.extraBody = `{"user": "me"}`

	d.gen.temperature = "warm"
	click(t, tt, "Save")
	frames(tt)
	if d.err != "Temperature is not a number." || len(a.lib.settings.Get().Providers.LLM) != 0 {
		t.Fatalf("a bad number: %q, setups %+v", d.err, a.lib.settings.Get().Providers.LLM)
	}
	d.gen.temperature, d.gen.extraBody = "0", "[1]"
	click(t, tt, "Save")
	frames(tt)
	if d.err != "The request parameters are not a JSON object." {
		t.Fatalf("bad JSON: %q", d.err)
	}
	d.gen.extraBody = `{"user": "me"}`
	click(t, tt, "Save")
	frames(tt)
	setups := a.lib.settings.Get().Providers.LLM
	if len(setups) != 1 {
		t.Fatalf("setups %+v (error %q)", setups, d.err)
	}
	g := setups[0].Generation
	if g.Thinking != llm.ThinkEffort || g.ThinkingEffort != "none" || *g.MaxOutputTokens != 131072 || *g.Temperature != 0 ||
		*g.TopP != 0.9 || g.ExtraBodyJSON != `{"user": "me"}` || g.TopK != nil {
		t.Fatalf("saved %+v", g)
	}

	// Again: the dialog shows what was saved, and nothing is unsaved.
	a.openProviderDialog(local.LLM, llm.Custom)
	frames(tt)
	if f := a.provDlg.gen; f.maxTokens != "131072" || f.temperature != "0" || f.topP != "0.9" || f.think != thinkTitle(llm.ThinkEffort) || f.effort != "none" {
		t.Fatalf("reopened %+v", f)
	}
	if a.provDlg.unsaved() {
		t.Error("an untouched dialog counts as unsaved")
	}
}

// TestGenerationNumbers: 0 output tokens (or fewer) asks for no limit, as
// an empty field does, instead of answers cut to one token; NaN and Inf
// are refused, since the settings cannot be saved with them.
func TestGenerationNumbers(t *testing.T) {
	for _, s := range []string{"0", "-5"} {
		if g, bad := (&genForm{maxTokens: s}).settings(); bad != "" || g.MaxOutputTokens != nil {
			t.Errorf("max output tokens %q: %v, bad %q", s, g.MaxOutputTokens, bad)
		}
	}
	for _, s := range []string{"NaN", "inf", "-Inf"} {
		if _, bad := (&genForm{temperature: s}).settings(); bad != "Temperature" {
			t.Errorf("temperature %q: bad %q", s, bad)
		}
	}
}

// TestGenerationFieldAloneKeepsItsRoom: a number field alone on its row
// (OpenAI's Max output tokens, Ollama's Keep the model loaded for) is as
// high as its label and input, so the next row starts below its input.
// It was 0 high: the next row's label sat on its label and covered its
// input.
func TestGenerationFieldAloneKeepsItsRoom(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400) // the whole dialog in view: a clipped label's box is cut
	a.navigate(remotePath)
	frames(tt)
	for _, c := range []struct{ service, field, next string }{
		{"openai", "Max output tokens (optional)", "Verbosity"},
		{"ollama", "Keep the model loaded for", "Advanced"},
	} {
		a.openProviderDialog(local.LLM, c.service)
		frames(tt)
		field, okField := tt.Find(c.field)
		next, okNext := tt.Find(c.next)
		if !okField || !okNext {
			t.Fatalf("%s: %q found %v, %q found %v", c.service, c.field, okField, c.next, okNext)
		}
		if next.Y < field.Y+2*field.H { // its label, then an input taller than the label
			t.Errorf("%s: %q at y %.0f covers the input of %q (its label at y %.0f, %.0f high)", c.service, c.next, next.Y, c.field, field.Y, field.H)
		}
		a.provDlg = providerDialog{}
		frames(tt)
	}
}

// TestGenerationDialogUntouched: StepFun's Think has no Default (off
// only); its dialog opens with off chosen, and untouched it does not count
// as unsaved, so Escape and a click beside it close it.
func TestGenerationDialogUntouched(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, "stepfun")
	frames(tt)
	if a.provDlg.unsaved() {
		t.Errorf("an untouched dialog counts as unsaved: Think %q", a.provDlg.gen.think)
	}
}

// TestRemoteAPIFormat: OpenAI and custom services choose the API format in
// Advanced, Chat Completions unless changed; Responses hides the fields it
// has no use for, is saved, and comes back in the dialog.
func TestRemoteAPIFormat(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate(remotePath)
	frames(tt)
	for service, want := range map[string]bool{"openai": true, llm.Custom: true, "deepseek": false} {
		a.openProviderDialog(local.LLM, service)
		frames(tt)
		if has(tt, "API format") {
			t.Errorf("%s: API format outside the folded Advanced", service)
		}
		click(t, tt, "Advanced")
		if has(tt, "API format") != want {
			t.Errorf("%s: API format shown: %v", service, !want)
		}
		if a.provDlg.unsaved() {
			t.Errorf("%s: an untouched dialog counts as unsaved", service)
		}
		a.provDlg.open = false
		frames(tt)
	}

	a.openProviderDialog(local.LLM, llm.Custom)
	frames(tt)
	click(t, tt, "Advanced")
	d := &a.provDlg
	if d.gen.api != "Chat Completions" || d.setup().APIFormat != "" {
		t.Fatalf("a new setup: %q, %q", d.gen.api, d.setup().APIFormat)
	}
	// Exact matches: the API format's help names these fields.
	if !slices.Contains(tt.Texts(), "Stop sequences") || !slices.Contains(tt.Texts(), "Logprobs") {
		t.Errorf("Chat Completions lost its fields: %q", tt.Texts())
	}
	d.endpoint, d.customModel = "https://example.com/v1", "m"
	d.gen.api = "Responses"
	frames(tt)
	if slices.Contains(tt.Texts(), "Stop sequences") || slices.Contains(tt.Texts(), "Logprobs") {
		t.Errorf("Responses shows fields it does not send: %q", tt.Texts())
	}
	click(t, tt, "Save")
	frames(tt)
	setups := a.lib.settings.Get().Providers.LLM
	if len(setups) != 1 || setups[0].APIFormat != llm.APIResponses {
		t.Fatalf("setups %+v (error %q)", setups, d.err)
	}
	a.openProviderDialog(local.LLM, llm.Custom)
	frames(tt)
	if a.provDlg.gen.api != "Responses" || a.provDlg.unsaved() {
		t.Errorf("reopened: %q, unsaved %v", a.provDlg.gen.api, a.provDlg.unsaved())
	}
}

// TestThinkingOffForShortInputs (#3951): a service set to think offers
// Off for short inputs; on, it saves the limit (200 when empty), it shows
// again on reopen, and Voxt's default or Off thinking saves none.
func TestThinkingOffForShortInputs(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, llm.Custom)
	frames(tt)
	if has(tt, "Off for short inputs") {
		t.Fatal("Voxt's default thinking offers Off for short inputs")
	}
	d := &a.provDlg
	d.endpoint, d.customModel = "https://example.com/v1", "m"
	d.gen.think, d.gen.effort = thinkTitle(llm.ThinkEffort), "high"
	frames(tt)
	if !has(tt, "Off for short inputs") || has(tt, "Fewer characters than") {
		t.Fatalf("thinking on: %q", tt.Texts())
	}
	click(t, tt, "Off for short inputs")
	frames(tt)
	if !d.gen.shortOff || !has(tt, "Fewer characters than") {
		t.Fatalf("switched on: %v %q", d.gen.shortOff, tt.Texts())
	}
	click(t, tt, "Save")
	frames(tt)
	if s := a.lib.settings.Get().Providers.LLM; len(s) != 1 || s[0].Generation.ThinkingOffUnder != 200 {
		t.Fatalf("saved %+v (error %q)", s, d.err)
	}
	a.openProviderDialog(local.LLM, llm.Custom)
	frames(tt)
	if f := a.provDlg.gen; !f.shortOff || f.shortUnder != "200" || a.provDlg.unsaved() {
		t.Fatalf("reopened %+v, unsaved %v", f, a.provDlg.unsaved())
	}
	for _, c := range []struct {
		f    genForm
		want int
	}{
		{genForm{think: thinkTitle(llm.ThinkOn), shortOff: true, shortUnder: "80"}, 80},
		{genForm{think: thinkTitle(llm.ThinkOn), shortOff: true, shortUnder: "0"}, 200},
		{genForm{think: thinkTitle(llm.ThinkOff), shortOff: true, shortUnder: "80"}, 0},
		{genForm{think: thinkTitle(llm.ThinkDefault), shortOff: true}, 0},
		{genForm{think: thinkTitle(llm.ThinkOn), shortUnder: "80"}, 0},
	} {
		if g, bad := c.f.settings(); bad != "" || g.ThinkingOffUnder != c.want {
			t.Errorf("%+v: ThinkingOffUnder %d (bad %q), want %d", c.f, g.ThinkingOffUnder, bad, c.want)
		}
	}
	if _, bad := (&genForm{think: thinkTitle(llm.ThinkOn), shortOff: true, shortUnder: "many"}).settings(); bad != "Fewer characters than" {
		t.Errorf("a limit that is not a number: bad %q", bad)
	}
}
