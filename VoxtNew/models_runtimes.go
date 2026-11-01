package main

// The runtimes behind the local models (internal/local's Openers): the
// one place that says which runtime opens which model. A runtime that is
// not wired here is "not available" (local.ErrNoRuntime). ASR is
// internal/omni's (the speech model's snapshot, and the VAD's from
// a.models.Dir(models.VADSilero)), LLM the voxtnew-llm process's
// (internal/llmrt). Both load from the family's model library: the
// snapshot folder $HUB/models--<owner>--<name>/snapshots/<commit> of the
// commit installed (models.Manager.Dir).
//
// Developers' options, from the environment: HF_ENDPOINT=<url> downloads
// from that Hub only (huggingface_hub's variable); HF_HUB_CACHE or
// HF_HOME move the model library, unless the family's settings name one
// (models.root); ACTNOW_FAMILY_DIR moves the family directory.

import (
	"log"

	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// newLocalServices is the local models a new app uses (tests swap in
// fakes).
var newLocalServices = func(a *app) local.Services {
	return local.NewLibrary(a.models, a.localOpeners(), uitext.ModelName)
}

// localOpeners are the runtimes this build has.
func (a *app) localOpeners() local.Openers {
	return local.Openers{
		ASR:    a.openOmniASR, // models_omni.go
		LLM:    a.openLLM,     // llm_link.go
		Unload: a.unloadModel,
	}
}

// unloadModel is Openers.Unload: the runtime of model id's kind lets it
// go.
func (a *app) unloadModel(id string) {
	if m, ok := models.Lookup(id); ok && m.Kind == models.LLM {
		a.unloadLLM(id) // llm_link.go
		return
	}
	a.unloadOmni(id) // models_omni.go
}

// systemProxies are the Mac's web proxies, for the downloads.
func systemProxies() models.SystemProxies { return models.SystemProxies(macx.SystemProxies()) }

// systemProxy is the downloads' automatic proxy: the environment's, else
// the Mac's (read again every 5 s).
var systemProxy = models.SystemProxy(systemProxies, log.Printf)

// familyEndpoints are the Hub endpoints a network.hfEndpoints value races.
func familyEndpoints(v string) []string { return famsettings.Endpoints(v) }

// managerOptions are the models' options: the family's library, the
// network, onChange.
func (a *app) managerOptions(onChange func()) models.Options {
	return models.Options{Store: a.fam.Store(), Network: a.networkFor(a.prefs), ReserveBytes: modelsReserve, OnChange: onChange}
}

// modelsReserve is the room a download leaves free (0: hfhub's 2 GB;
// tests turn the check off).
var modelsReserve int64
