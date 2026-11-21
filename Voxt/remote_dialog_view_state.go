package main

// Serializable display state for the remote service dialog, including its
// generation and ChatGPT controls. Editable values are duplicated here only
// so JSON fixtures can seed the same local widget state as production.

import (
	"slices"
	"strconv"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/secrets"
)

type remoteDialogViewState struct {
	Open              bool                       `json:"open"`
	Kind              string                     `json:"kind"`
	ID                string                     `json:"id"`
	Account           string                     `json:"account"`
	Name              string                     `json:"name"`
	Subtitle          string                     `json:"subtitle"`
	What              string                     `json:"what"`
	HasKeyURL         bool                       `json:"has_key_url,omitempty"`
	ShowAppID         bool                       `json:"show_app_id,omitempty"`
	ChatGPT           *remoteChatGPTViewState    `json:"chatgpt,omitempty"`
	Key               *remoteKeyViewState        `json:"key,omitempty"`
	Endpoint          *remoteEndpointViewState   `json:"endpoint,omitempty"`
	ModelOptions      []string                   `json:"model_options"`
	OtherModel        string                     `json:"other_model"`
	RecognitionPrompt bool                       `json:"recognition_prompt,omitempty"`
	Generation        *remoteGenerationViewState `json:"generation,omitempty"`
	Error             string                     `json:"error,omitempty"`
	ErrorMore         string                     `json:"error_more,omitempty"`
	CanTest           bool                       `json:"can_test"`
	Test              remoteTestViewState        `json:"test"`
	ShowRemove        bool                       `json:"show_remove"`
	Saving            bool                       `json:"saving"`
	Unsaved           bool                       `json:"unsaved"`
	Form              remoteDialogFormViewState  `json:"form"`
}

type remoteDialogFormViewState struct {
	AppID             string                        `json:"app_id,omitempty"`
	Endpoint          string                        `json:"endpoint,omitempty"`
	Model             string                        `json:"model"`
	CustomModel       string                        `json:"custom_model,omitempty"`
	RecognitionPrompt string                        `json:"recognition_prompt,omitempty"`
	HasTypedKey       bool                          `json:"has_typed_key,omitempty"`
	TypedKeySpaces    int                           `json:"typed_key_spaces,omitempty"`
	RemoveKey         bool                          `json:"remove_key,omitempty"`
	ShowKey           bool                          `json:"show_key,omitempty"`
	Generation        remoteGenerationFormViewState `json:"generation"`
}

type remoteKeyViewState struct {
	Info        string `json:"info"`
	Placeholder string `json:"placeholder"`
	Warning     string `json:"warning,omitempty"`
	HasSaved    bool   `json:"has_saved"`
}

type remoteEndpointViewState struct {
	Info        string `json:"info"`
	Placeholder string `json:"placeholder"`
}

type remoteChatGPTViewState struct {
	Running   bool   `json:"running"`
	SignedIn  bool   `json:"signed_in"`
	Email     string `json:"email,omitempty"`
	Error     string `json:"error,omitempty"`
	ErrorMore string `json:"error_more,omitempty"`
	Code      string `json:"code,omitempty"`
}

type remoteGenerationViewState struct {
	ThinkOptions          []remoteOptionViewState       `json:"think_options"`
	Thinking              string                        `json:"thinking"`
	EffortOptions         []remoteOptionViewState       `json:"effort_options,omitempty"`
	EffortSelection       string                        `json:"effort_selection,omitempty"`
	ShortUnderPlaceholder string                        `json:"short_under_placeholder,omitempty"`
	ShowMax               bool                          `json:"show_max,omitempty"`
	VerbosityOptions      []remoteOptionViewState       `json:"verbosity_options,omitempty"`
	ShowKeepAlive         bool                          `json:"show_keep_alive,omitempty"`
	ShowUsage             bool                          `json:"show_usage,omitempty"`
	APIOptions            []remoteOptionViewState       `json:"api_options,omitempty"`
	ServiceDefault        string                        `json:"service_default"`
	ShowTemperature       bool                          `json:"show_temperature,omitempty"`
	ShowTopP              bool                          `json:"show_top_p,omitempty"`
	ShowTopK              bool                          `json:"show_top_k,omitempty"`
	ShowMinP              bool                          `json:"show_min_p,omitempty"`
	FrequencyOnly         bool                          `json:"frequency_only,omitempty"`
	ShowPenalties         bool                          `json:"show_penalties,omitempty"`
	ShowSeed              bool                          `json:"show_seed,omitempty"`
	ResponseFormatOptions []remoteOptionViewState       `json:"response_format_options,omitempty"`
	LogprobsCap           bool                          `json:"logprobs_cap,omitempty"`
	StopCap               bool                          `json:"stop_cap,omitempty"`
	ShowExtraBody         bool                          `json:"show_extra_body,omitempty"`
	ExtraBodyExample      string                        `json:"extra_body_example,omitempty"`
	Form                  remoteGenerationFormViewState `json:"form"`
}

type remoteGenerationFormViewState struct {
	Think       string `json:"think,omitempty"`
	Effort      string `json:"effort,omitempty"`
	Budget      string `json:"budget,omitempty"`
	ShortOff    bool   `json:"short_off,omitempty"`
	ShortUnder  string `json:"short_under,omitempty"`
	MaxTokens   string `json:"max_tokens,omitempty"`
	Temperature string `json:"temperature,omitempty"`
	TopP        string `json:"top_p,omitempty"`
	TopK        string `json:"top_k,omitempty"`
	MinP        string `json:"min_p,omitempty"`
	Seed        string `json:"seed,omitempty"`
	Presence    string `json:"presence,omitempty"`
	Frequency   string `json:"frequency,omitempty"`
	Repetition  string `json:"repetition,omitempty"`
	Logprobs    bool   `json:"logprobs,omitempty"`
	TopLogprobs string `json:"top_logprobs,omitempty"`
	Format      string `json:"format,omitempty"`
	Schema      string `json:"schema,omitempty"`
	Stop        string `json:"stop,omitempty"`
	Verbosity   string `json:"verbosity,omitempty"`
	KeepAlive   string `json:"keep_alive,omitempty"`
	Usage       bool   `json:"usage"`
	ExtraBody   string `json:"extra_body,omitempty"`
	Advanced    bool   `json:"advanced,omitempty"`
	API         string `json:"api,omitempty"`
}

type remoteDialogStateInput struct {
	Dialog       *providerDialog
	Service      service
	Model        string
	Capabilities providers.GenCaps
	Mask         string
	KeySpaced    bool
	ChatEmail    string
	ChatLogin    chatgptLogin
	Unsaved      bool
}

func buildRemoteDialogViewState(translator *i18n.Translator, input remoteDialogStateInput) remoteDialogViewState {
	d, sv := input.Dialog, input.Service
	model := input.Model
	streams := d.kind == local.ASR && sv.speech.Streams(model)
	state := remoteDialogViewState{
		Open: d.open, Kind: string(d.kind), ID: d.id, Account: providers.Account(d.kind, d.id), Name: sv.name,
		Subtitle:  translator.T("Remote language model · OpenAI-compatible API"),
		What:      translator.T("Recognised and selected text is sent to %s.", sv.name),
		HasKeyURL: sv.keyURL != "", ShowAppID: sv.appID,
		ModelOptions: remoteModelOptions(sv.models, d.served, d.model, otherModelFor(translator)), OtherModel: otherModelFor(translator),
		RecognitionPrompt: d.kind == local.ASR,
		Error:             d.err, ErrorMore: d.errMore, CanTest: !d.test.running && !d.saving && (!sv.signIn || input.ChatEmail != ""),
		Test: buildRemoteTestViewState(translator, d.test), ShowRemove: d.exists, Saving: d.saving, Unsaved: input.Unsaved,
		Form: remoteDialogFormStateOf(d),
	}
	switch {
	case sv.signIn:
		state.Subtitle = translator.T("Remote language model · your ChatGPT plan")
	case streams:
		state.Subtitle = translator.T("Remote speech recognition · realtime, while you speak")
	case d.kind == local.ASR:
		state.Subtitle = translator.T("Remote speech recognition · the recording, once you stop")
	}
	if d.kind == local.ASR {
		state.What = translator.T("Your recordings are sent to %s to be recognised.", sv.name)
	} else if sv.endpoint == "" {
		state.What = translator.T("Recognised and selected text is sent to the server at the address below.")
	}
	if sv.signIn {
		state.ChatGPT = &remoteChatGPTViewState{Running: input.ChatLogin.running, SignedIn: input.ChatEmail != "", Email: input.ChatEmail,
			Error: input.ChatLogin.err, ErrorMore: input.ChatLogin.more, Code: input.ChatLogin.code}
	} else {
		keyInfo := i18n.PlatformFor(translator, "Saved in this Mac's Keychain.", "Saved in this PC's Credential Manager.")
		if sv.keyOptional {
			keyInfo = i18n.PlatformFor(translator, "Optional for a server on this Mac. Saved in this Mac's Keychain.", "Optional for a server on this PC. Saved in this PC's Credential Manager.")
		}
		placeholder := translator.T("Paste the key")
		if input.Mask != "" && !d.removeKey {
			placeholder = translator.T("Saved: %s · paste a new one to replace it", input.Mask)
		}
		state.Key = &remoteKeyViewState{Info: keyInfo, Placeholder: placeholder, HasSaved: input.Mask != "" && !d.removeKey}
		_, inner := secrets.Clean(d.key)
		switch {
		case d.removeKey:
			state.Key.Warning = translator.T("The saved key is removed when you save.")
		case inner > 0:
			state.Key.Warning = translator.N(inner, "The key has %d space or line break in it, removed when you test or save.", "The key has %d spaces or line breaks in it, removed when you test or save.", inner)
		case d.key == "" && input.KeySpaced:
			state.Key.Warning = translator.T("The saved key has a space or line break in it, so the service refuses it. Paste the key again.")
		}
		endpointInfo := translator.T("The API root, ending in /v1. Leave it empty for the service's own. HTTP sends the API key and text without encryption.")
		switch {
		case streams:
			endpointInfo = i18n.PlatformFor(translator, "Leave it empty for the service's own. Only wss://, or ws:// to this Mac.", "Leave it empty for the service's own. Only wss://, or ws:// to this PC.")
		case d.kind == local.ASR:
			endpointInfo = i18n.PlatformFor(translator, "Leave it empty for the service's own. Only https://, or http:// to this Mac.", "Leave it empty for the service's own. Only https://, or http:// to this PC.")
		case sv.endpoint == "":
			endpointInfo = translator.T("The API root of any OpenAI-compatible server, ending in /v1. HTTP sends the API key and text without encryption.")
		}
		placeholder = sv.endpoint
		if d.kind == local.ASR {
			placeholder = sv.speech.EndpointOf(model)
		}
		if placeholder == "" {
			placeholder = "https://…/v1"
		}
		state.Endpoint = &remoteEndpointViewState{Info: endpointInfo, Placeholder: placeholder}
	}
	if d.kind == local.LLM && !sv.signIn {
		generation := buildRemoteGenerationViewState(translator, d.gen, input.Capabilities)
		state.Generation = &generation
	}
	return state
}

func remoteDialogFormStateOf(d *providerDialog) remoteDialogFormViewState {
	_, spaces := secrets.Clean(d.key)
	return remoteDialogFormViewState{
		AppID: d.appID, Endpoint: d.endpoint, Model: d.model, CustomModel: d.customModel, RecognitionPrompt: d.recognitionPrompt,
		HasTypedKey: d.key != "", TypedKeySpaces: spaces, RemoveKey: d.removeKey, ShowKey: d.showKey,
		Generation: remoteGenerationFormStateOf(d.gen),
	}
}

func remoteDialogLocalState(state remoteDialogViewState) providerDialogLocalUI {
	local := providerDialogLocalUI{
		open: state.Open, appID: state.Form.AppID, endpoint: state.Form.Endpoint, model: state.Form.Model,
		customModel: state.Form.CustomModel, recognitionPrompt: state.Form.RecognitionPrompt,
		err: state.Error, errMore: state.ErrorMore, removeKey: state.Form.RemoveKey, showKey: state.Form.ShowKey,
		gen: remoteGenerationLocalState(state.Form.Generation),
	}
	if state.Form.HasTypedKey {
		local.key = "fixture"
		for range state.Form.TypedKeySpaces {
			local.key += " "
		}
	}
	return local
}

func buildRemoteGenerationViewState(translator *i18n.Translator, form genForm, caps providers.GenCaps) remoteGenerationViewState {
	state := remoteGenerationViewState{
		ThinkOptions:          remoteOptions(caps.ThinkModes, func(value string) string { return thinkTitleFor(translator, value) }),
		EffortOptions:         remoteOptions(caps.Efforts, func(value string) string { return effortTitleFor(translator, value) }),
		ShortUnderPlaceholder: strconv.Itoa(shortUnderDefault), ShowMax: caps.MaxOutputTokens,
		VerbosityOptions: remoteOptions(verbosities, func(value string) string { return verbosityTitleFor(translator, value) }),
		ShowKeepAlive:    caps.KeepAlive, ShowUsage: caps.IncludeUsage,
		APIOptions: remoteOptions(apiFormats, apiFormatTitle), ServiceDefault: translator.T("Service default"),
		ShowTemperature: caps.Temperature, ShowTopP: caps.TopP, ShowTopK: caps.TopK, ShowMinP: caps.MinP,
		FrequencyOnly: caps.FrequencyOnly, ShowPenalties: caps.Penalties, ShowSeed: caps.Seed,
		ShowExtraBody: caps.ExtraBody, Form: remoteGenerationFormStateOf(form),
	}
	state.Thinking = valueOf(form.think, caps.ThinkModes, func(value string) string { return thinkTitleFor(translator, value) })
	state.EffortSelection = effortTitleFor(translator, form.effort)
	if len(caps.Efforts) > 0 {
		effort := form.effort
		if !slices.Contains(caps.Efforts, effort) {
			effort = caps.Efforts[0]
		}
		state.EffortSelection = effortTitleFor(translator, effort)
	}
	if !caps.Verbosity {
		state.VerbosityOptions = nil
	}
	if !caps.APIFormat {
		state.APIOptions = nil
	}
	formats := responseFormats[:2]
	if caps.JSONSchema {
		formats = responseFormats
	}
	if caps.ResponseFormat {
		state.ResponseFormatOptions = remoteOptions(formats, func(value string) string { return formatTitleFor(translator, value) })
	}
	state.LogprobsCap = caps.Logprobs
	state.StopCap = caps.Stop
	if caps.ExtraBody {
		state.ExtraBodyExample = `{"user": "me"}`
		if caps.KeepAlive {
			state.ExtraBodyExample = `{"options": {"num_ctx": 8192}}`
		}
	}
	return state
}

func remoteGenerationFormStateOf(form genForm) remoteGenerationFormViewState {
	return remoteGenerationFormViewState{
		Think: form.think, Effort: form.effort, Budget: form.budget, ShortOff: form.shortOff, ShortUnder: form.shortUnder,
		MaxTokens: form.maxTokens, Temperature: form.temperature, TopP: form.topP, TopK: form.topK, MinP: form.minP, Seed: form.seed,
		Presence: form.presence, Frequency: form.frequency, Repetition: form.repetition, Logprobs: form.logprobs,
		TopLogprobs: form.topLogprobs, Format: form.format, Schema: form.schema, Stop: form.stop, Verbosity: form.verbosity,
		KeepAlive: form.keepAlive, Usage: form.usage, ExtraBody: form.extraBody, Advanced: form.advanced, API: form.api,
	}
}

func remoteGenerationLocalState(state remoteGenerationFormViewState) genForm {
	return genForm{
		think: state.Think, effort: state.Effort, budget: state.Budget, shortOff: state.ShortOff, shortUnder: state.ShortUnder,
		maxTokens: state.MaxTokens, temperature: state.Temperature, topP: state.TopP, topK: state.TopK, minP: state.MinP, seed: state.Seed,
		presence: state.Presence, frequency: state.Frequency, repetition: state.Repetition, logprobs: state.Logprobs,
		topLogprobs: state.TopLogprobs, format: state.Format, schema: state.Schema, stop: state.Stop, verbosity: state.Verbosity,
		keepAlive: state.KeepAlive, usage: state.Usage, extraBody: state.ExtraBody, advanced: state.Advanced, api: state.API,
	}
}

func remoteOptions(values []string, titleOf func(string) string) []remoteOptionViewState {
	options := make([]remoteOptionViewState, len(values))
	for index, value := range values {
		options[index] = remoteOptionViewState{Title: titleOf(value), Value: value}
	}
	return options
}
