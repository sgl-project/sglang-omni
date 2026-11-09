package main

// The runtimes behind the local models (internal/local's Openers): the
// one place that says which runtime opens which model. A runtime that is
// not wired here is "not available" (local.ErrNoRuntime). ASR is
// internal/omni's (the speech model's snapshot, and the VAD's from
// a.models.Dir(models.VADSilero)); no runtime opens a local language
// model (they are retired: Voxt's language models are remote). The speech
// models load from the family's model library: the snapshot folder $HUB/models--<owner>--<name>/snapshots/<commit> of the
// commit installed (models.Manager.Dir).
//
// Developers' options, from the environment: HF_ENDPOINT=<url> downloads
// from that Hub only (huggingface_hub's variable); ACTNOW_FAMILY_DIR
// moves the family directory and with it the default model library
// (SGLang Omni/models beside it, models.HubIn), unless the family's
// settings name one (models.root); HF_HUB_CACHE and HF_HOME only name Hugging Face caches
// that models are imported from.

import (
	"log"

	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// newLocalServices is the local models a new app uses (tests swap in
// fakes).
var newLocalServices = func(a *app) local.Services {
	return local.NewLibrary(a.models, a.localOpeners(), uitext.ModelName)
}

// localOpeners are the runtimes this build has.
func (a *app) localOpeners() local.Openers {
	return local.Openers{
		ASR:      a.openOmniASR, // models_omni.go
		Unload:   a.unloadOmni,  // models_omni.go
		ASRFirst: func() string { return recommendedSpeech(a.ob.memoryGB) },
		Hold:     a.omniMeetingHold,
	}
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
