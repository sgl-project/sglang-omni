package main

// The local models in Settings → About: their lines in the diagnostics
// (installed, loaded and their memory, the language models' process) and
// their self-checks (each kind's runtime is in this copy of VoxtNew, and
// a model of it is installed).

import (
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// llmLocate finds voxtnew-llm, "" when this copy has none (a test seam,
// beside omniLocate).
var llmLocate = func() string {
	resources, _ := mygo.App.Path(mygo.PathResources)
	return llmrt.Find(resources)
}

// modelDiagnostics adds the local models' lines to the diagnostics
// (memory only: the main thread).
func (a *app) modelDiagnostics(line func(k, v string)) {
	var installed []string
	for _, m := range models.Catalog() {
		if a.modelStatus[m.ID].Installed {
			installed = append(installed, m.ID)
		}
	}
	line("Models installed", strings.Join(installed, ", "))
	var loaded []string
	for id, b := range a.modelMem {
		loaded = append(loaded, id+" "+uitext.MemoryBytes(b))
	}
	slices.Sort(loaded)
	line("Models loaded", strings.Join(loaded, ", "))
	if rt := a.llmMade(); rt != nil {
		st := rt.Status()
		v := string(st.State)
		if st.Hello != nil {
			v = st.Hello.Version + " (" + v + ")"
		}
		if st.Err != "" {
			first, _, _ := strings.Cut(st.Err, "\n")
			v += ": " + first
		}
		line("Language model process", v)
	}
}

// modelCheckInput is what the models' self-checks need from the main
// thread: the installed models' names by kind.
type modelCheckInput struct {
	asr, llm []string
}

// modelCheckInput reads the installed models (memory only).
func (a *app) modelCheckInput() modelCheckInput {
	var in modelCheckInput
	for _, m := range a.installed(local.ASR) {
		in.asr = append(in.asr, m.Name)
	}
	for _, m := range a.installed(local.LLM) {
		in.llm = append(in.llm, m.Name)
	}
	return in
}

// modelChecks checks each kind of model: its runtime is there (a look
// on the disk: off the main thread), and a model of it is installed.
func modelChecks(in modelCheckInput, add func(id, title, status, detail string)) {
	missing := i18n.T("Its runtime is missing from this copy of VoxtNew.")
	title := i18n.T("Speech recognition")
	switch _, err := omniLocate(); {
	case err != nil:
		add("speech", title, "fail", missing)
	case len(in.asr) == 0:
		add("speech", title, "warn", i18n.T("No model installed: download one on the Local models page."))
	default:
		add("speech", title, "pass", strings.Join(in.asr, ", "))
	}
	title = i18n.T("Language models")
	switch {
	case llmLocate() == "":
		add("language", title, "fail", missing)
	case len(in.llm) == 0:
		add("language", title, "warn", i18n.T("No model installed: Transcribe types the text as recognised."))
	default:
		add("language", title, "pass", strings.Join(in.llm, ", "))
	}
}
